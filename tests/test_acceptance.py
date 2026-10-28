import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_acceptance", ROOT / "tools/check_acceptance.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        def make(name, data):
            (self.base / name).write_bytes(data)
            return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
        commit = "a" * 40
        self.model = make("model", b"fake model for validation")
        self.binary = make("engine", b"binary")
        self.benchmark = make("benchmark", b"separate benchmark executable")
        self.result = make("locked", b"locked results")
        self.result2 = make("unlocked", b"unlocked results")
        self.manifest = {"schema_version": 1, "gates": {f"G{i}": {"status": "not_run"} for i in range(1, 8)},
            "inventory": {"os": "Windows 11", "native": True, "gpu": "TU102", "compute_capability": "7.5", "vram_bytes": 22 * 1024**3, "raw": make("inventory", b"inventory")},
            "source": {"commit": "b" * 40, "strata_commit": commit, "patches": [make("patch", b"integration patch")], "lock": make("lock", json.dumps({"strata": {"commit": commit}}).encode())},
            "binary": self.binary, "model": self.model, "build_record": make("build", b"build record")}
        self.model_lock = {"file": {"lfs": {"sha256": self.model["sha256"]}}}

    def check(self, value):
        # Tests use a tiny fixture model while exercising identity checking.
        real_loads = json.loads
        def loads(data, *args, **kwargs):
            parsed = real_loads(data, *args, **kwargs)
            if isinstance(parsed, dict) and parsed.get("repository") == "InternScience/Sage-A1.5-Q4_K_M-GGUF":
                return self.model_lock
            return parsed
        with mock.patch.object(checker.json, "loads", side_effect=loads):
            return checker.check(value, self.base)

    def passed(self, gate="G5"):
        m = copy.deepcopy(self.manifest)
        m["gates"][gate] = {"status": "passed", "environment": "native_windows_hardware", "binary_sha256": self.binary["sha256"], "model_sha256": self.model["sha256"], "artifacts": [self.result]}
        run = {"environment": "native_windows_hardware", "binary_sha256": self.benchmark["sha256"], "device_id": "device", "driver_version": "driver", "test_id": "mma", "shape": "1024x1024x1024", "dtype": "fp16-fp32", "raw_result": self.result}
        m["post_driver_mma"] = {"benchmark": self.benchmark, "locked": run, "unlocked": dict(run, raw_result=self.result2)}
        return m

    def current_scope(self):
        m = copy.deepcopy(self.manifest)
        m.update(schema_version=2, profile="strata_ninfer")
        for gate in ("G5", "G6"):
            m["gates"][gate] = {"status": "deferred"}
        return m

    def test_current_scope_defers_unlock_without_accepting_it(self):
        m = self.current_scope()
        r = self.check(m)
        self.assertTrue(r["valid"], r["errors"])
        self.assertEqual(r["required_gates"], ["G1", "G2", "G3", "G4", "G7"])
        self.assertEqual(r["deferred_gates"], ["G5", "G6"])
        self.assertFalse(r["hardware_unlocked"])
        m["gates"]["G1"] = self.passed("G1")["gates"]["G1"]
        r = self.check(m)
        self.assertTrue(r["valid"], r["errors"])
        self.assertEqual(r["assessment"], "candidate_evidence_requires_review")
        self.assertNotIn("post_driver_mma", m)

    def test_deferred_gates_cannot_claim_pass(self):
        for gate in ("G5", "G6"):
            m = self.current_scope()
            m["gates"][gate] = self.passed(gate)["gates"][gate]
            self.assertFalse(self.check(m)["valid"])

    def test_scope_is_explicit_and_legacy_contract_is_preserved(self):
        m = self.current_scope()
        del m["profile"]
        self.assertFalse(self.check(m)["valid"])
        m = self.current_scope()
        m["profile"] = "anything"
        self.assertFalse(self.check(m)["valid"])
        m = self.passed("G6")
        del m["post_driver_mma"]
        self.assertFalse(self.check(m)["valid"])
        m = self.current_scope()
        m["schema_version"] = 1
        self.assertFalse(self.check(m)["valid"])

    def test_active_gate_still_needs_real_hardware_evidence(self):
        m = self.current_scope()
        m["gates"]["G2"] = {"status": "passed"}
        self.assertFalse(self.check(m)["valid"])
        m = self.current_scope()
        m["gates"]["G3"] = {"status": "deferred"}
        self.assertFalse(self.check(m)["valid"])

    def test_not_run_has_no_claim(self):
        r = self.check(self.manifest)
        self.assertTrue(r["valid"])
        self.assertFalse(r["hardware_unlocked"])

    def test_candidate_never_grants_hardware_claim(self):
        r = self.check(self.passed())
        self.assertTrue(r["valid"], r["errors"])
        self.assertEqual(r["assessment"], "candidate_evidence_requires_review")
        self.assertFalse(r["hardware_unlocked"])

    def test_simulation_cannot_be_claimed_as_unlock(self):
        m = self.passed()
        m["post_driver_mma"]["unlocked"]["environment"] = "simulated_host_test"
        self.assertFalse(self.check(m)["valid"])
        m = self.passed()
        m["gates"]["G5"]["environment"] = "simulated_host_test"
        self.assertFalse(self.check(m)["valid"])

    def test_mismatched_artifacts_and_identity(self):
        m = self.passed()
        m["post_driver_mma"]["unlocked"]["binary_sha256"] = "b" * 64
        self.assertFalse(self.check(m)["valid"])
        m = self.passed()
        m["inventory"]["vram_bytes"] = 11 * 1024**3
        self.assertFalse(self.check(m)["valid"])
        m = self.passed()
        (self.base / "unlocked").write_text("tampered")
        self.assertFalse(self.check(m)["valid"])

    def test_source_provenance_requires_separate_revisions_and_patches(self):
        m = self.passed()
        self.assertTrue(self.check(m)["valid"])
        m["source"]["strata_commit"] = "c" * 40
        self.assertFalse(self.check(m)["valid"])
        m = self.passed()
        m["source"]["patches"] = []
        self.assertFalse(self.check(m)["valid"])
        m = self.passed()
        (self.base / "patch").write_bytes(b"modified")
        self.assertFalse(self.check(m)["valid"])

    def test_mma_benchmark_is_distinct_and_pair_matched(self):
        m = self.passed()
        self.assertNotEqual(self.benchmark["sha256"], self.binary["sha256"])
        self.assertTrue(self.check(m)["valid"])
        m["post_driver_mma"]["unlocked"]["shape"] = "2048x2048x2048"
        self.assertFalse(self.check(m)["valid"])
        m = self.passed()
        del m["post_driver_mma"]["locked"]["dtype"]
        self.assertFalse(self.check(m)["valid"])
        m = self.passed("G6")
        m["gates"]["G6"]["model_sha256"] = "b" * 64
        self.assertFalse(self.check(m)["valid"])

    def test_booleans_are_not_evidence(self):
        m = self.passed()
        m["post_driver_mma"] = {"correct": True, "unlocked": True}
        self.assertFalse(self.check(m)["valid"])


if __name__ == "__main__":
    unittest.main()
