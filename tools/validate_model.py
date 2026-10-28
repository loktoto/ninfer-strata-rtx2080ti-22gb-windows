"""Capture strict baseline/candidate teacher-forced diagnostics on native Windows.

Input is one JSON object with exact pretokenized ``prompt_ids`` and ``follow_ids``
arrays. No tokenizer or character-count estimate is used. A matching forced output
is a protocol check, not numerical parity: --follow replaces the model's picks.
The engine needs the local --tokens-file / --follow-file patch for long prompts.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import struct
import time


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validation_strict_launcher", ROOT / "integration/strata/launch_strict.py")
_launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_launcher)
_compare_spec = importlib.util.spec_from_file_location("full_logits_comparator", ROOT / "tools/compare_logits.py")
_comparator = importlib.util.module_from_spec(_compare_spec)
_compare_spec.loader.exec_module(_comparator)
ENV_KEYS = ("STRATA_ROUTE_TAIL_SKIP", "STRATA_FREEZE_EXPERTS",
            "STRATA_SAGE_RMSNORM", "STRATA_SAGE_EXPERT_Q4K", "STRATA_SAGE_EXPERT_Q6K", "STRATA_SAGE_TRACE")
CANDIDATES = {
    "rmsnorm": {"rmsnorm"}, "expert-q4k": {"expert-q4k"}, "expert-q6k": {"expert-q6k"},
    "both": {"rmsnorm", "expert-q4k"}, "all": {"rmsnorm", "expert-q4k", "expert-q6k"},
}
INT32_MAX = 2 ** 31 - 1
STATS = re.compile(
    r"^prompt (\d+) tokens (\S+) tok/s, generated (\d+) tokens "
    r"(\S+) tok/s \(([^)]+)\);")
DUMP = re.compile(
    r"^dump pos=(\d+) blk=(-?\d+) \S+\s+n=(\d+) sum=(\S+) abs=(\S+) "
    r"first=(\S+) (\S+) (\S+)$")


def fingerprint(path, hash_contents=True):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    result = {"path": str(path), "bytes": before.st_size,
              "mtime_ns": before.st_mtime_ns, "sha256": None}
    if hash_contents:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"File changed while hashing: {path}")
        result["sha256"] = digest.hexdigest()
    return result


def token_ids(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty JSON array of token IDs")
    if any(type(item) is not int or not 0 <= item <= INT32_MAX for item in value):
        raise ValueError(f"{name} IDs must be integers in [0, {INT32_MAX}] (not booleans)")
    return value


def load_tokens(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"prompt_ids", "follow_ids"}:
        raise ValueError("Token JSON must contain exactly prompt_ids and follow_ids arrays")
    return token_ids(data["prompt_ids"], "prompt_ids"), token_ids(data["follow_ids"], "follow_ids")


def model_inventory(model, lock_path, hash_model):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    try:
        expected_size = lock["file"]["size"]
        expected_hash = lock["file"]["lfs"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Model lock needs file.size and file.lfs.sha256") from exc
    if (type(expected_size) is not int or expected_size < 4
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None):
        raise ValueError("Model lock has invalid size or SHA-256")
    with model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {model}")
    info = fingerprint(model, hash_contents=hash_model)
    if info["bytes"] != expected_size:
        raise ValueError(f"Model size {info['bytes']} does not match lock size {expected_size}")
    if hash_model and info["sha256"] != expected_hash:
        raise ValueError("Model SHA-256 does not match the supplied model lock")
    info.update(expected_sha256=expected_hash,
                identity="full_hash_matches_lock" if hash_model else "size_only_identity_unverified")
    return info, {"file": fingerprint(lock_path), "contents": lock}


def parse_logs(stdout_path, stderr_path, prompt_count, follow, top_count):
    """Parse the one-shot CSV/std-error protocol in PolyStrata main.cpp::once."""
    errors, ids, output_lines, stats, top_rows = [], [], 0, [], []
    dispatched = set()
    dump_count = 0
    dump_hash = hashlib.sha256()
    with stdout_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            output_lines += 1
            if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", line) is None:
                errors.append("stdout is not the one-shot CSV token-ID protocol")
            else:
                ids.extend(int(value) for value in line.split(","))
    if output_lines != 1:
        errors.append(f"expected one CSV output line, got {output_lines}")
    with stderr_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if line in {"sage-dispatch: rmsnorm", "sage-dispatch: expert-q4k", "sage-dispatch: expert-q6k"}:
                dispatched.add(line.split(": ", 1)[1])
            match = STATS.match(line)
            if match:
                stats.append({"prompt_tokens": int(match[1]), "generated_tokens": int(match[3]),
                              "finish": match[5]})
            if line.startswith("top"):
                row = []
                try:
                    if not line.startswith("top "):
                        raise ValueError()
                    for field in line.split()[1:]:
                        token, probability = field.split(":")
                        token, probability = int(token), float(probability)
                        if not 0 <= token <= INT32_MAX or not math.isfinite(probability) or not 0 <= probability <= 1:
                            raise ValueError()
                        row.append([token, probability])
                    if not 1 <= len(row) <= top_count or len({item[0] for item in row}) != len(row):
                        raise ValueError()
                except ValueError:
                    errors.append("malformed top-probabilities diagnostic")
                top_rows.append(row)
            if line.startswith("dump "):
                dump_count += 1
                dump_hash.update((line + "\n").encode("utf-8"))
                match = DUMP.fullmatch(line)
                try:
                    if match is None or not all(math.isfinite(float(value)) for value in match.groups()[3:]):
                        raise ValueError()
                except ValueError:
                    errors.append("malformed or nonfinite layer-checksum diagnostic")
    if len(stats) != 1:
        errors.append(f"expected one completion stats line, got {len(stats)}")
    else:
        if stats[0]["prompt_tokens"] != prompt_count:
            errors.append("engine did not report reading the complete prompt")
        if stats[0]["generated_tokens"] != len(follow):
            errors.append("engine reported a short or excess continuation")
        if stats[0]["finish"] not in {"length", "stop"}:
            errors.append(f"engine finish reason: {stats[0]['finish']}")
    if ids != follow:
        errors.append("generated IDs do not exactly match the complete forced continuation")
    if len(top_rows) != len(follow):
        errors.append("top-probabilities step count does not match requested continuation")
    if not dump_count:
        errors.append("no layer-checksum diagnostics were emitted")
    return {"generated_ids": ids, "forced_continuation_matches": ids == follow,
            "stats": stats, "top_probabilities": top_rows, "layer_checksum_lines": dump_count,
            "layer_checksums_sha256": dump_hash.hexdigest() if dump_count else None,
            "errors": sorted(set(errors)), "dispatch_enqueues": sorted(dispatched)}


def run_one(command, env, output, name, timeout, prompt_count, follow, top_count):
    stdout_path, stderr_path = output / f"{name}.stdout.txt", output / f"{name}.stderr.txt"
    started = time.perf_counter()
    result = {"status": "failed", "exit_code": None,
              "stdout": stdout_path.name, "stderr": stderr_path.name}
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            completed = subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                                       timeout=timeout, check=False)
            result["exit_code"] = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["error"] = str(exc)
    result["wall_seconds"] = time.perf_counter() - started
    result["diagnostics"] = parse_logs(stdout_path, stderr_path, prompt_count, follow, top_count)
    if result["exit_code"] == 0 and not result["diagnostics"]["errors"]:
        result["status"] = "completed_unqualified"
    return result


def compare_runs(baseline, candidate):
    left, right = baseline["diagnostics"], candidate["diagnostics"]
    comparison = {
        "generated_ids_equal": left["generated_ids"] == right["generated_ids"],
        "rounded_top_probabilities_equal": left["top_probabilities"] == right["top_probabilities"],
        "layer_checksum_text_equal": left["layer_checksums_sha256"] == right["layer_checksums_sha256"],
        "numerical_logits_parity": "not_established",
        "interpretation": "Output IDs are teacher-forced; equality does not test the model's token picks. "
                          "Top probabilities are rounded and truncated, and layer sums are lossy diagnostics.",
    }
    return comparison


def write_report(output, report):
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens-file", type=Path, required=True,
                        help="JSON object containing exact prompt_ids and follow_ids arrays")
    parser.add_argument("--model-lock", type=Path, default=ROOT / "config/model.lock.json")
    parser.add_argument("--output", type=Path, required=True, help="A new directory for provenance and raw logs")
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), default="both")
    parser.add_argument("--max-context", type=int,
                        help="Total prompt + generation capacity; default 32768 + len(follow_ids)")
    parser.add_argument("--max-new", type=int,
                        help="Must equal len(follow_ids); defaults to that exact length")
    parser.add_argument("--top-probs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds per engine run")
    parser.add_argument("--hash-model", action="store_true", help="Read and verify the entire GGUF once")
    parser.add_argument("--dry-run", action="store_true", help="Write launch inventory without running; allowed on Linux")
    parser.add_argument("--require-dispatch", action="store_true", help="Fail if selected candidate paths never report a host enqueue")
    parser.add_argument("--full-logits", action="store_true", help="Capture and compare every teacher-forced FP32 logit")
    parser.add_argument("--logits-atol", type=float, default=1e-4)
    parser.add_argument("--logits-rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    try:
        if not args.dry_run and platform.system() != "Windows":
            raise ValueError("Real runs require native Windows; use --dry-run for Linux/WSL inventory")
        if args.output.exists():
            raise ValueError("Output already exists; choose a new directory")
        if any(not math.isfinite(t) or t < 0 for t in (args.logits_atol, args.logits_rtol)):
            raise ValueError("logit tolerances must be finite and nonnegative")
        prompt, follow = load_tokens(args.tokens_file)
        max_new = len(follow) if args.max_new is None else args.max_new
        if max_new != len(follow):
            raise ValueError("--max-new must equal the exact follow_ids length")
        max_context = 32768 + max_new if args.max_context is None else args.max_context
        if not 16 <= max_context <= INT32_MAX or len(prompt) + max_new > max_context:
            raise ValueError("prompt_ids count + generation budget exceeds --max-context (minimum 16)")
        if not 1 <= args.top_probs <= INT32_MAX or not 1 <= args.seed < 2 ** 64:
            raise ValueError("--top-probs must be positive and --seed must be in [1, 2**64-1]")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be finite and positive")
        model, lock = model_inventory(args.model, args.model_lock, args.hash_model)
        engine, tokens = fingerprint(args.engine), fingerprint(args.tokens_file)
        output = args.output.resolve()
        common = ["--tokens-file", str(output / "prompt.ids"), "--follow-file", str(output / "follow.ids"),
                  "--max-context", str(max_context), "--max-new", str(max_new),
                  "--seed", str(args.seed), "--temperature", "0", "--top-probs", str(args.top_probs), "--dump-layers"]
        launches, environments = {}, {}
        for name in ("baseline", "candidate"):
            selected = CANDIDATES[args.candidate] if name == "candidate" else set()
            command, env = _launcher.launch_spec(args.engine, args.model, common + (["--logits-file", str(output / f"{name}.logits")] if args.full_logits else []),
                                                ninfer_rmsnorm="rmsnorm" in selected,
                                                ninfer_expert_q4k="expert-q4k" in selected,
                                                ninfer_expert_q6k="expert-q6k" in selected)
            env["STRATA_SAGE_TRACE"] = "1" if args.require_dispatch else "0"
            controlled = {key: env[key] for key in ENV_KEYS}
            launches[name] = {"argv": command, "controlled_environment": controlled,
                              "controlled_environment_sha256": hashlib.sha256(
                                  json.dumps(controlled, sort_keys=True).encode("utf-8")).hexdigest()}
            environments[name] = env
        output.mkdir(parents=True, exist_ok=False)
        for name, ids in [("prompt", prompt), ("follow", follow)]:
            (output / f"{name}.ids").write_text(",".join(map(str, ids)) + "\n", encoding="ascii")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = {
        "schema_version": 1, "status": "inventory_only",
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(), "candidate": args.candidate,
        "engine": engine, "model": model, "model_lock": lock, "token_input": tokens,
        "strict_launcher": fingerprint(ROOT / "integration/strata/launch_strict.py"),
        "prompt_tokens": len(prompt), "generation_budget": max_new, "max_context": max_context,
        "seed": args.seed, "timeout_seconds": args.timeout,
        "full_logits": {"requested": args.full_logits, "atol": args.logits_atol, "rtol": args.logits_rtol},
        "token_files": {name: fingerprint(output / f"{name}.ids") for name in ("prompt", "follow")},
        "launches": launches, "runs": {}, "gates": {"G1": "not_established", "G2": "not_established"},
        "limitations": ["Teacher-forced output equality is a protocol check, not a comparison of model picks.",
                        "Rounded top probabilities and layer checksums cannot establish full-logit numerical parity.",
                        "Diagnostic dumping changes execution and timings; this is not a throughput benchmark.",
                        "No hardware acceptance gate is granted by this harness; native Windows evidence needs review."]}
    write_report(output, report)
    if not args.dry_run:
        for name in ("baseline", "candidate"):
            report["runs"][name] = run_one(launches[name]["argv"], environments[name], output, name,
                                           args.timeout, len(prompt), follow, args.top_probs)
            report["status"] = "failed"
            write_report(output, report)
        report["comparison"] = compare_runs(report["runs"]["baseline"], report["runs"]["candidate"])
        if all(run["status"] == "completed_unqualified" for run in report["runs"].values()):
            report["status"] = "completed_unqualified"
        if args.full_logits:
            comparison = {"status": "failed", "atol": args.logits_atol, "rtol": args.logits_rtol}
            try:
                result = _comparator.compare(output / "baseline.logits", output / "candidate.logits",
                                             args.logits_atol, args.logits_rtol)
                comparison.update(result, status="passed" if result["failed"] == 0 else "failed")
                comparison["artifacts"] = {name: fingerprint(output / f"{name}.logits")
                                            for name in ("baseline", "candidate")}
            except (OSError, ValueError, struct.error) as exc:
                comparison["error"] = str(exc)
            report["comparison"]["full_logits"] = comparison
            if comparison["status"] == "failed":
                report["status"] = "failed"
        if args.require_dispatch:
            required = CANDIDATES[args.candidate]
            observed = set(report["runs"]["candidate"]["diagnostics"]["dispatch_enqueues"])
            report["comparison"]["dispatch_missing"] = sorted(required - observed)
            report["comparison"]["dispatch_scope"] = "Host enqueue only, not GPU execution or completion"
            if required - observed:
                report["status"] = "failed"
        write_report(output, report)
    print(f"{report['status']}: {output / 'report.json'}; G1/G2 not established")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
