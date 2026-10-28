"""Check acceptance evidence structure and local artifact hashes, never grant hardware acceptance.

See docs/RESEARCH_FINDINGS.md for the manifest format and limits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
GATES = tuple(f"G{i}" for i in range(1, 8))


def check(manifest, base):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    def artifact(value, label):
        if not isinstance(value, dict):
            errors.append(f"{label}: artifact object required")
            return None
        path, expected = value.get("path"), value.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(expected, str) or not SHA.fullmatch(expected):
            errors.append(f"{label}: path and lowercase SHA-256 required")
            return None
        try:
            p = Path(path)
            if not p.is_absolute():
                p = base / p
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                errors.append(f"{label}: SHA-256 mismatch")
                return None
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: cannot read artifact: {exc}")
            return None
        return expected
    if not isinstance(manifest, dict):
        return {"valid": False, "errors": ["manifest must be an object"], "hardware_unlocked": False}
    require(manifest.get("schema_version") == 1, "schema_version must be 1")
    gates = manifest.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(GATES):
        errors.append("gates must contain exactly G1 through G7")
        gates = {}
    for gate, value in gates.items():
        require(isinstance(value, dict) and value.get("status") in {"not_run", "failed", "passed"}, f"{gate}: invalid status")
    passed = [g for g in GATES if isinstance(gates.get(g), dict) and gates[g].get("status") == "passed"]
    if passed:
        inv = manifest.get("inventory")
        if not isinstance(inv, dict):
            errors.append("inventory required for passed gates")
            inv = {}
        require(inv.get("os") == "Windows 11" and inv.get("native") is True, "native Windows 11 inventory required")
        require(inv.get("gpu") == "TU102" and inv.get("compute_capability") == "7.5", "TU102 SM75 inventory required")
        vram = inv.get("vram_bytes")
        require(type(vram) is int and 22548578304 <= vram <= 24696061952, "target must report approximately 22 GiB VRAM")
        artifact(inv.get("raw"), "inventory.raw")
        source = manifest.get("source")
        require(isinstance(source, dict) and isinstance(source.get("commit"), str) and COMMIT.fullmatch(source["commit"]) is not None, "project source commit must be full 40-character SHA")
        require(isinstance(source, dict) and isinstance(source.get("strata_commit"), str) and COMMIT.fullmatch(source["strata_commit"]) is not None, "strata_commit must be full 40-character SHA")
        patches = source.get("patches") if isinstance(source, dict) else None
        if not isinstance(patches, list) or not patches:
            errors.append("source.patches must contain at least one hashed patch artifact")
        else:
            for i, item in enumerate(patches):
                artifact(item, f"source.patches[{i}]")
        artifact(source.get("lock") if isinstance(source, dict) else None, "source.lock")
        if isinstance(source, dict) and isinstance(source.get("lock"), dict):
            try:
                lock_path = Path(source["lock"]["path"])
                lock = json.loads((lock_path if lock_path.is_absolute() else base / lock_path).read_text(encoding="utf-8"))
                require(lock["strata"]["commit"] == source.get("strata_commit"), "strata_commit does not match source lock")
            except (OSError, KeyError, TypeError, ValueError):
                errors.append("source lock must contain strata.commit")
        binary = artifact(manifest.get("binary"), "binary")
        artifact(manifest.get("build_record"), "build_record")
        model = artifact(manifest.get("model"), "model")
        expected_model = json.loads((ROOT / "config/model.lock.json").read_text(encoding="utf-8"))["file"]["lfs"]["sha256"]
        require(model == expected_model, "model SHA must match repository model lock")
        for gate in passed:
            record = gates[gate]
            require(record.get("environment") == "native_windows_hardware", f"{gate}: passed requires native Windows hardware environment")
            require(record.get("binary_sha256") == binary and binary is not None, f"{gate}: binary SHA mismatch")
            require(record.get("model_sha256") == model and model is not None, f"{gate}: model SHA mismatch")
            evidence = record.get("artifacts")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{gate}: at least one evidence artifact required")
            else:
                for i, item in enumerate(evidence):
                    artifact(item, f"{gate}.artifacts[{i}]")
        if "G5" in passed or "G6" in passed:
            pair = manifest.get("post_driver_mma")
            if not isinstance(pair, dict):
                errors.append("post_driver_mma pair required for G5/G6")
            else:
                benchmark = artifact(pair.get("benchmark"), "post_driver_mma.benchmark")
                for state in ("locked", "unlocked"):
                    run = pair.get(state)
                    if not isinstance(run, dict):
                        errors.append(f"post_driver_mma.{state} required")
                        continue
                    require(run.get("environment") == "native_windows_hardware", f"{state}: simulated/host runs do not qualify")
                    require(run.get("binary_sha256") == benchmark and benchmark is not None, f"{state}: benchmark SHA mismatch")
                    for field in ("device_id", "driver_version", "test_id", "shape", "dtype"):
                        require(isinstance(run.get(field), str) and bool(run[field].strip()), f"{state}: {field} required")
                    artifact(run.get("raw_result"), f"{state}.raw_result")
                a, b = pair.get("locked"), pair.get("unlocked")
                if isinstance(a, dict) and isinstance(b, dict):
                    for field in ("device_id", "driver_version", "test_id", "shape", "dtype"):
                        require(a.get(field) == b.get(field) and bool(a.get(field)), f"MMA pair {field} must match")
                    require(a.get("raw_result") != b.get("raw_result"), "MMA pair must have separate raw results")
    return {"valid": not errors, "errors": errors, "submitted_passed_gates": passed,
            "assessment": "candidate_evidence_requires_review" if passed and not errors else "no_accepted_hardware_claim",
            "hardware_unlocked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        report = check(json.loads(args.manifest.read_text(encoding="utf-8")), args.manifest.parent)
    except (OSError, ValueError) as exc:
        report = {"valid": False, "errors": [str(exc)], "hardware_unlocked": False}
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
