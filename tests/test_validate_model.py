"""CPU-only tests of the evidence harness; no model inference or hardware gate."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import struct
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_model", ROOT / "tools/validate_model.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ModelValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = self.root / "engine with spaces.exe"
        self.engine.write_bytes(b"test engine (not executable)")
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"GGUFtest payload")
        self.lock = self.root / "model.lock.json"
        self.lock.write_text(json.dumps({"file": {"size": self.model.stat().st_size,
                                                "lfs": {"sha256": hashlib.sha256(self.model.read_bytes()).hexdigest()}}}),
                             encoding="utf-8")
        self.tokens = self.root / "tokens.json"
        self.write_tokens([0, 21, 37], [42, 43])
        self.output = self.root / "evidence"
        self.args = ["--engine", str(self.engine), "--model", str(self.model),
                     "--model-lock", str(self.lock), "--tokens-file", str(self.tokens),
                     "--output", str(self.output)]

    def write_tokens(self, prompt, follow):
        self.tokens.write_text(json.dumps({"prompt_ids": prompt, "follow_ids": follow}), encoding="utf-8")

    def call(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()):
            return validator.main([*self.args, *extra])

    def report(self):
        return json.loads((self.output / "report.json").read_text(encoding="utf-8"))

    @staticmethod
    def fake_run(command, **kwargs):
        kwargs["stdout"].write(b"42,43\n")
        kwargs["stderr"].write(
            b"strata-poly: loaded in 0.1 s\n"
            b"dump pos=0 blk=0 attn n=3 sum=1.000000 abs=1.000000 first=0.0 0.0 1.0\n"
            b"top 42:0.6 43:0.3\n"
            b"top 43:0.7 44:0.2\n"
            b"prompt 3 tokens 1.0 tok/s, generated 2 tokens 1.0 tok/s (length); experts from VRAM 100.0%\n")
        return subprocess.CompletedProcess(command, 0)

    def test_dry_run_records_exact_tokens_context_and_provenance_without_subprocess(self):
        with patch.object(validator.subprocess, "run") as run:
            self.assertEqual(self.call(["--dry-run"]), 0)
        run.assert_not_called()
        report = self.report()
        self.assertEqual(report["status"], "inventory_only")
        self.assertEqual(report["prompt_tokens"], 3)
        self.assertEqual(report["generation_budget"], 2)
        self.assertEqual(report["max_context"], 32770)
        self.assertEqual((self.output / "prompt.ids").read_text(), "0,21,37\n")
        self.assertEqual((self.output / "follow.ids").read_text(), "42,43\n")
        self.assertEqual(report["token_input"]["sha256"], hashlib.sha256(self.tokens.read_bytes()).hexdigest())
        self.assertEqual(report["engine"]["sha256"], hashlib.sha256(self.engine.read_bytes()).hexdigest())
        self.assertIsNone(report["model"]["sha256"])
        self.assertEqual(report["model"]["identity"], "size_only_identity_unverified")
        self.assertEqual(report["gates"], {"G1": "not_established", "G2": "not_established"})
        self.assertFalse(report["runs"])
        baseline, candidate = report["launches"]["baseline"], report["launches"]["candidate"]
        self.assertEqual(baseline["controlled_environment"]["STRATA_SAGE_RMSNORM"], "0")
        self.assertEqual(candidate["controlled_environment"]["STRATA_SAGE_RMSNORM"], "1")
        self.assertEqual(candidate["controlled_environment"]["STRATA_SAGE_EXPERT_Q4K"], "1")
        self.assertNotEqual(baseline["controlled_environment_sha256"], candidate["controlled_environment_sha256"])
        self.assertEqual(baseline["argv"], candidate["argv"])
        self.assertIn("--tokens-file", baseline["argv"])
        self.assertIn("--follow-file", baseline["argv"])
        self.assertNotIn("--tokens", baseline["argv"])
        self.assertNotIn("--follow", baseline["argv"])

    def test_32768_prompt_keeps_generation_capacity_and_short_windows_command(self):
        self.write_tokens([123456] * 32768, [42, 43])
        self.assertEqual(self.call(["--dry-run"]), 0)
        report = self.report()
        self.assertEqual(report["prompt_tokens"], 32768)
        self.assertEqual(report["max_context"], 32770)
        self.assertLess(len(subprocess.list2cmdline(report["launches"]["baseline"]["argv"])), 32767)

    def test_each_candidate_selects_only_requested_backend(self):
        for candidate, rms, q4k in [("rmsnorm", "1", "0"), ("expert-q4k", "0", "1")]:
            with self.subTest(candidate=candidate):
                self.output = self.root / candidate
                args = self.args[:-1] + [str(self.output), "--candidate", candidate, "--dry-run"]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(validator.main(args), 0)
                env = self.report()["launches"]["candidate"]["controlled_environment"]
                self.assertEqual(env["STRATA_SAGE_RMSNORM"], rms)
                self.assertEqual(env["STRATA_SAGE_EXPERT_Q4K"], q4k)

    def test_cpu_mock_complete_output_is_unqualified_and_never_numerical_parity(self):
        with patch.object(validator.platform, "system", return_value="Windows"), \
                patch.object(validator.subprocess, "run", side_effect=self.fake_run) as run:
            self.assertEqual(self.call(), 0)
        self.assertEqual(run.call_count, 2)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["env"]["STRATA_FREEZE_EXPERTS"], "1")
        report = self.report()
        self.assertEqual(report["status"], "completed_unqualified")
        self.assertEqual(report["comparison"]["numerical_logits_parity"], "not_established")
        self.assertTrue(report["comparison"]["generated_ids_equal"])
        self.assertTrue(report["runs"]["baseline"]["diagnostics"]["forced_continuation_matches"])
        self.assertEqual(report["runs"]["baseline"]["exit_code"], 0)
        self.assertGreaterEqual(report["runs"]["baseline"]["wall_seconds"], 0)
        self.assertEqual((self.output / "baseline.stdout.txt").read_bytes(), b"42,43\n")
        self.assertIn(b"dump pos=0", (self.output / "candidate.stderr.txt").read_bytes())

    def run_with_changed_candidate(self, change):
        count = 0

        def fake(command, **kwargs):
            nonlocal count
            count += 1
            result = self.fake_run(command, **kwargs)
            if count == 2:
                change(result, kwargs)
            return result

        with patch.object(validator.platform, "system", return_value="Windows"), \
                patch.object(validator.subprocess, "run", side_effect=fake):
            code = self.call()
        return code, self.report()

    @staticmethod
    def replace_log(kwargs, name, old, new):
        stream = kwargs[name]
        stream.flush()
        path = Path(stream.name)
        path.write_bytes(path.read_bytes().replace(old, new))

    def test_generated_token_mismatch_fails_even_with_zero_exit(self):
        code, report = self.run_with_changed_candidate(
            lambda result, kwargs: self.replace_log(kwargs, "stdout", b"42,43", b"42,44"))
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["comparison"]["generated_ids_equal"])
        self.assertFalse(report["runs"]["candidate"]["diagnostics"]["forced_continuation_matches"])

    def test_short_eos_output_is_failure_for_fixed_continuation(self):
        def shorten(result, kwargs):
            self.replace_log(kwargs, "stdout", b"42,43", b"42")
            self.replace_log(kwargs, "stderr", b"generated 2", b"generated 1")
            self.replace_log(kwargs, "stderr", b"(length)", b"(stop)")
        code, report = self.run_with_changed_candidate(shorten)
        self.assertEqual(code, 1)
        self.assertIn("engine reported a short or excess continuation",
                      report["runs"]["candidate"]["diagnostics"]["errors"])

    def test_incomplete_prompt_report_fails_even_if_forced_ids_equal(self):
        code, report = self.run_with_changed_candidate(
            lambda result, kwargs: self.replace_log(kwargs, "stderr", b"prompt 3 tokens", b"prompt 2 tokens"))
        self.assertEqual(code, 1)
        self.assertIn("engine did not report reading the complete prompt",
                      report["runs"]["candidate"]["diagnostics"]["errors"])

    def test_missing_completion_or_diagnostics_fails(self):
        code, report = self.run_with_changed_candidate(
            lambda result, kwargs: self.replace_log(kwargs, "stderr", b"prompt 3 tokens", b"truncated 3 tokens"))
        self.assertEqual(code, 1)
        self.assertIn("expected one completion stats line, got 0",
                      report["runs"]["candidate"]["diagnostics"]["errors"])

    def test_top_probability_difference_diagnostic_keeps_unqualified_status(self):
        code, report = self.run_with_changed_candidate(
            lambda result, kwargs: self.replace_log(kwargs, "stderr", b"42:0.6", b"42:0.5"))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "completed_unqualified")
        self.assertFalse(report["comparison"]["rounded_top_probabilities_equal"])
        self.assertEqual(report["gates"]["G2"], "not_established")

    def test_malformed_or_missing_diagnostics_cannot_qualify_a_run(self):
        changes = [
            ("stdout", b"42,43\n", b"42,43\nERR failed after output\n"),
            ("stderr", b"top 42:0.6", b"top 42:nan"),
            ("stderr", b"sum=1.000000", b"sum=inf"),
            ("stderr", b"dump pos=0", b"missing pos=0"),
            ("stderr", b"top 43:0.7 44:0.2\n", b""),
        ]
        for index, (name, old, new) in enumerate(changes):
            with self.subTest(name=name, old=old):
                self.output = self.root / f"malformed-{index}"
                self.args[-1] = str(self.output)
                code, report = self.run_with_changed_candidate(
                    lambda result, kwargs: self.replace_log(kwargs, name, old, new))
                self.assertEqual(code, 1)
                self.assertTrue(report["runs"]["candidate"]["diagnostics"]["errors"])

    def test_nonzero_exit_fails_even_with_valid_protocol(self):
        code, report = self.run_with_changed_candidate(
            lambda result, kwargs: setattr(result, "returncode", 9))
        self.assertEqual(code, 1)
        self.assertEqual(report["runs"]["candidate"]["exit_code"], 9)
        self.assertEqual(report["status"], "failed")

    def test_spawn_failure_keeps_raw_logs_and_failed_report(self):
        with patch.object(validator.platform, "system", return_value="Windows"), \
                patch.object(validator.subprocess, "run", side_effect=OSError("no executable")):
            self.assertEqual(self.call(), 1)
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["runs"]["baseline"]["error"], "no executable")
        self.assertIsNone(report["runs"]["baseline"]["exit_code"])
        self.assertTrue((self.output / "baseline.stderr.txt").exists())

    def test_timeout_keeps_partial_raw_output(self):
        def timeout(command, **kwargs):
            kwargs["stdout"].write(b"42")
            raise subprocess.TimeoutExpired(command, 1)
        with patch.object(validator.platform, "system", return_value="Windows"), \
                patch.object(validator.subprocess, "run", side_effect=timeout):
            self.assertEqual(self.call(["--timeout", "1"]), 1)
        self.assertEqual((self.output / "baseline.stdout.txt").read_bytes(), b"42")
        self.assertEqual(self.report()["status"], "failed")

    def test_native_windows_required_for_real_run(self):
        with patch.object(validator.platform, "system", return_value="Linux"), \
                patch.object(validator.subprocess, "run") as run, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.call()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_token_arrays_fail_before_launch(self):
        for prompt, follow in [([True], [1]), ([1], [False]), ([-1], [1]),
                               ([1.0], [1]), (["1"], [1]), ([], [1]), ([1], []),
                               ([2 ** 31], [1]), ([1], None)]:
            with self.subTest(prompt=prompt, follow=follow):
                self.write_tokens(prompt, follow)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.call(["--dry-run"])
                self.assertFalse(self.output.exists())

    def test_context_budget_and_forcing_length_are_exact(self):
        self.write_tokens(list(range(16)), [42, 43])
        for extra in [["--max-context", "17"], ["--max-new", "1"], ["--seed", "0"], ["--timeout", "nan"]]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.call(["--dry-run", *extra])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.call(["--dry-run", "--max-context", "18"]), 0)
        self.assertEqual(self.report()["max_context"], 18)

    def test_full_model_hash_is_opt_in_and_verified(self):
        self.assertEqual(self.call(["--dry-run", "--hash-model"]), 0)
        self.assertEqual(self.report()["model"]["identity"], "full_hash_matches_lock")
        self.assertEqual(self.report()["model"]["sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())

    def test_model_lock_hash_or_size_mismatch_rejected(self):
        for key, value in [("size", self.model.stat().st_size + 1), ("sha256", "0" * 64)]:
            with self.subTest(key=key):
                lock = json.loads(self.lock.read_text())
                if key == "size":
                    lock["file"]["size"] = value
                else:
                    lock["file"]["size"] = self.model.stat().st_size
                    lock["file"]["lfs"]["sha256"] = value
                self.lock.write_text(json.dumps(lock))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.call(["--dry-run", "--hash-model"])

    def test_existing_output_directory_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing evidence")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.call(["--dry-run"])
        self.assertEqual(marker.read_text(), "existing evidence")


    def logits_run(self, command, delta=0.0, **kwargs):
        result = self.fake_run(command, **kwargs)
        path = Path(command[command.index("--logits-file") + 1])
        values = [float(i) / 64 for i in range(128)]
        if path.name == "candidate.logits":
            values[0] += delta
        path.write_bytes(struct.pack("<8sIIII", b"SAGELOG1", 1, 64, 3, 2)
                         + struct.pack("<5i", 0, 21, 37, 42, 43) + struct.pack("<128f", *values))
        return result

    def test_full_logits_compare_all_values(self):
        with patch.object(validator.platform, "system", return_value="Windows"), \
             patch.object(validator.subprocess, "run", side_effect=self.logits_run):
            self.assertEqual(self.call(["--full-logits"]), 0)
        compare = self.report()["comparison"]["full_logits"]
        self.assertEqual(compare["status"], "passed")
        self.assertEqual(compare["values"], 128)
        self.assertEqual(self.report()["gates"]["G2"], "not_established")

    def test_full_logits_numeric_failure_is_nonzero(self):
        def run(command, **kwargs):
            return self.logits_run(command, delta=1.0, **kwargs)
        with patch.object(validator.platform, "system", return_value="Windows"), \
             patch.object(validator.subprocess, "run", side_effect=run):
            self.assertEqual(self.call(["--full-logits"]), 1)
        self.assertEqual(self.report()["comparison"]["full_logits"]["failed"], 1)

    def test_missing_full_logits_is_failure(self):
        with patch.object(validator.platform, "system", return_value="Windows"), \
             patch.object(validator.subprocess, "run", side_effect=self.fake_run):
            self.assertEqual(self.call(["--full-logits"]), 1)
        self.assertIn("error", self.report()["comparison"]["full_logits"])

    def test_require_dispatch_rejects_unused_candidates(self):
        with patch.object(validator.platform, "system", return_value="Windows"), \
             patch.object(validator.subprocess, "run", side_effect=self.fake_run):
            self.assertEqual(self.call(["--require-dispatch"]), 1)
        self.assertEqual(self.report()["comparison"]["dispatch_missing"], ["expert-q4k", "rmsnorm"])

    def test_require_dispatch_records_successful_host_enqueues(self):
        def run(command, **kwargs):
            result = self.fake_run(command, **kwargs)
            if kwargs["env"]["STRATA_SAGE_RMSNORM"] == "1":
                kwargs["stderr"].write(b"sage-dispatch: rmsnorm\nsage-dispatch: expert-q4k\n")
            return result
        with patch.object(validator.platform, "system", return_value="Windows"), \
             patch.object(validator.subprocess, "run", side_effect=run):
            self.assertEqual(self.call(["--require-dispatch"]), 0)
        self.assertEqual(self.report()["comparison"]["dispatch_missing"], [])

    def test_all_candidate_plan_includes_q6k_and_baseline_remains_original(self):
        self.assertEqual(self.call(["--dry-run", "--candidate", "all"]), 0)
        launches = self.report()["launches"]
        for key in ("STRATA_SAGE_RMSNORM", "STRATA_SAGE_EXPERT_Q4K", "STRATA_SAGE_EXPERT_Q6K"):
            self.assertEqual(launches["baseline"]["controlled_environment"][key], "0")
            self.assertEqual(launches["candidate"]["controlled_environment"][key], "1")

    def test_q6k_dispatch_evidence_is_recognized(self):
        def run(command, **kwargs):
            result = self.fake_run(command, **kwargs)
            if kwargs["env"]["STRATA_SAGE_EXPERT_Q6K"] == "1":
                kwargs["stderr"].write(b"sage-dispatch: expert-q6k\n")
            return result
        with patch.object(validator.platform, "system", return_value="Windows"), \
             patch.object(validator.subprocess, "run", side_effect=run):
            self.assertEqual(self.call(["--candidate", "expert-q6k", "--require-dispatch"]), 0)
        self.assertEqual(self.report()["comparison"]["dispatch_missing"], [])


if __name__ == "__main__":
    unittest.main()
