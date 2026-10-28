"""Run ordered, exact-token native Windows inference experiments; no gate is auto-approved."""
import argparse
import datetime
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validation_model", ROOT / "tools/validate_model.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
launcher = validation._launcher
ENV_KEYS = validation.ENV_KEYS


def read_manifest(path, plan):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "stages"} or manifest["schema_version"] != 1:
        raise ValueError("fixture manifest must contain schema_version 1 and stages")
    entries = manifest["stages"]
    if not isinstance(entries, list) or len(entries) != len(plan["stages"]):
        raise ValueError("fixture manifest must contain every plan stage in order")
    result = []
    for item, target in zip(entries, plan["stages"]):
        if not isinstance(item, dict) or set(item) != {"name", "tokens_file", "task_expectation"} or item["name"] != target["name"]:
            raise ValueError("each stage must contain name, tokens_file and task_expectation in plan order")
        if not isinstance(item["task_expectation"], str) or not item["task_expectation"].strip():
            raise ValueError("task_expectation must describe an externally reviewed task expectation")
        token_path = (path.parent / item["tokens_file"]).resolve(strict=True)
        prompt, follow = validation.load_tokens(token_path)
        total = len(prompt) + len(follow)
        if target["total_tokens"] is None:
            if not 16 <= total < 32768:
                raise ValueError("short fixture total must be between 16 and 32767")
        elif total != target["total_tokens"]:
            raise ValueError(f"{item['name']} requires exactly {target['total_tokens']} prompt + output IDs, got {total}")
        result.append({**item, "tokens_file": str(token_path), "prompt_tokens": len(prompt),
                       "output_tokens": len(follow), "total_tokens": total,
                       "token_fingerprint": validation.fingerprint(token_path)})
    return result


def api_check(config, timeout):
    # Explicit caller-supplied request; this exercises only the requested local endpoint.
    req = urllib.request.Request(config["url"], data=json.dumps(config["body"]).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
        ok = status == config["expected_status"] and ("expected_substring" not in config or config["expected_substring"] in body)
        return {"status": status, "response": body, "elapsed_seconds": time.perf_counter() - started, "passed": ok}
    except (OSError, urllib.error.URLError) as exc:
        return {"error": str(exc), "elapsed_seconds": time.perf_counter() - started, "passed": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-lock", type=Path, default=ROOT / "config/model.lock.json")
    p.add_argument("--source-lock", type=Path, default=ROOT / "config/sources.lock.json")
    p.add_argument("--candidate", choices=("rmsnorm", "expert-q4k", "both"), default="both")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--api-config", type=Path, help="Optional JSON endpoint, request and assertions")
    p.add_argument("--logits-atol", type=float, default=0.0001)
    p.add_argument("--logits-rtol", type=float, default=0.001)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    try:
        if not args.dry_run and platform.system() != "Windows":
            raise ValueError("real suite runs require native Windows")
        if args.output.exists():
            raise ValueError("output already exists")
        if args.warmups < 0 or args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or any(not math.isfinite(v) or v < 0 for v in (args.logits_atol, args.logits_rtol)):
            raise ValueError("invalid warmups, repeats or timeout")
        plan = json.loads((ROOT / "config/validation-plan.json").read_text(encoding="utf-8"))
        stages = read_manifest(args.manifest.resolve(strict=True), plan)
        for path in (args.engine, args.model, args.model_lock, args.source_lock):
            if not path.is_file():
                raise ValueError(f"file not found: {path}")
        api = None
        if args.api_config:
            api = json.loads(args.api_config.read_text(encoding="utf-8"))
            from urllib.parse import urlsplit
            parsed = urlsplit(api["url"])
            if parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError("API URL must be a local HTTP(S) endpoint")
            if not isinstance(api["body"], dict) or type(api["expected_status"]) is not int or type(api.get("requests", 1)) is not int or api.get("requests", 1) < 1:
                raise ValueError("invalid API configuration")
            if "expected_substring" in api and not isinstance(api["expected_substring"], str):
                raise ValueError("expected_substring must be a string")
        output = args.output.resolve()
        planned = []
        capture = [sys.executable, str(ROOT / "tools/capture_baseline.py"), "--engine", str(args.engine.resolve()), "--model", str(args.model.resolve()), "--source-lock", str(args.source_lock.resolve()), "--output", str(output / "baseline.json")]
        for stage in stages:
            stage_dir = output / stage["name"]
            diagnostic = [sys.executable, str(ROOT / "tools/validate_model.py"), "--engine", str(args.engine.resolve()), "--model", str(args.model.resolve()), "--model-lock", str(args.model_lock.resolve()), "--tokens-file", stage["tokens_file"], "--output", str(stage_dir / "diagnostic"), "--max-context", str(stage["total_tokens"]), "--candidate", args.candidate, "--timeout", str(args.timeout), "--full-logits", "--logits-atol", str(args.logits_atol), "--logits-rtol", str(args.logits_rtol)]
            if not planned:
                diagnostic.append("--hash-model")
            common = ["--tokens-file", str(stage_dir / "diagnostic/prompt.ids"), "--follow-file", str(stage_dir / "diagnostic/follow.ids"), "--max-context", str(stage["total_tokens"]), "--max-new", str(stage["output_tokens"]), "--seed", "1", "--temperature", "0"]
            benchmarks = {}
            for name in ("baseline", "candidate"):
                candidate = name == "candidate"
                cmd, env = launcher.launch_spec(args.engine, args.model, common, ninfer_rmsnorm=candidate and args.candidate in ("rmsnorm", "both"), ninfer_expert_q4k=candidate and args.candidate in ("expert-q4k", "both"))
                benchmarks[name] = {"argv": cmd, "controlled_environment": {k: env[k] for k in ENV_KEYS}}
            planned.append({**stage, "diagnostic_argv": diagnostic, "benchmarks": benchmarks, "status": "not_run", "semantic_quality": "not_assessed"})
        output.mkdir(parents=True)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    report = {"schema_version": 1, "status": "dry_run" if args.dry_run else "running", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "fixture_manifest": validation.fingerprint(args.manifest), "capture_argv": capture, "stages": planned, "warmups": args.warmups, "repeats": args.repeats, "gates": {f"G{i}": "not_established" for i in range(1, 8)}, "mtp": "not_tested", "dynamic_graphs": "not_tested", "api": {"status": "not_run", "config": api}}
    def save():
        (output / "suite.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    if args.dry_run:
        print(f"dry_run: {output / 'suite.json'}")
        return 0
    active_stage = None
    try:
        captured = subprocess.run(capture, timeout=args.timeout, check=False)
        if captured.returncode:
            raise RuntimeError("baseline capture failed")
        identity = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in (args.engine, args.model)}
        for stage in planned:
            active_stage = stage
            if any((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != state for path, state in identity.items()):
                raise RuntimeError("engine or model changed after baseline capture")
            if validation.fingerprint(stage["tokens_file"]) != stage["token_fingerprint"]:
                raise RuntimeError("fixture changed after planning")
            stage_dir = output / stage["name"]
            stage_dir.mkdir()
            run = subprocess.run(stage["diagnostic_argv"], timeout=args.timeout, check=False)
            if run.returncode:
                stage["status"] = "failed_diagnostic"
                raise RuntimeError(f"{stage['name']} diagnostic failed")
            stage["benchmark_runs"] = {}
            for name, launch in stage["benchmarks"].items():
                stage["benchmark_runs"][name] = []
                env = dict(__import__("os").environ, **launch["controlled_environment"])
                for i in range(args.warmups + args.repeats):
                    start = time.perf_counter()
                    proc = subprocess.run(launch["argv"], env=env, capture_output=True, timeout=args.timeout)
                    elapsed = time.perf_counter() - start
                    prefix = stage_dir / f"{name}-{i}"
                    prefix.with_suffix(".stdout.txt").write_bytes(proc.stdout)
                    prefix.with_suffix(".stderr.txt").write_bytes(proc.stderr)
                    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
                    lines = proc.stderr.decode("utf-8", errors="replace").splitlines()
                    stats = [validation.STATS.match(line) for line in lines]
                    stats = [s for s in stats if s]
                    _, follow = validation.load_tokens(stage["tokens_file"])
                    expected = ",".join(map(str, follow))
                    ok = proc.returncode == 0 and stdout == expected and len(stats) == 1 and int(stats[0][1]) == stage["prompt_tokens"] and int(stats[0][3]) == stage["output_tokens"] and stats[0][5] in ("length", "stop")
                    if ok:
                        try:
                            ok = all(math.isfinite(float(stats[0][index])) and float(stats[0][index]) > 0 for index in (2, 4))
                        except ValueError:
                            ok = False
                    entry = {"warmup": i < args.warmups, "wall_seconds": elapsed, "exit_code": proc.returncode, "valid": ok, "stdout": prefix.name + ".stdout.txt", "stderr": prefix.name + ".stderr.txt"}
                    if stats:
                        entry.update(prompt_tok_per_sec=stats[0][2], generation_tok_per_sec=stats[0][4])
                    stage["benchmark_runs"][name].append(entry)
                    save()
                    if not ok:
                        stage["status"] = "failed_benchmark"
                        raise RuntimeError(f"{stage['name']} benchmark failed")
            if any((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != state for path, state in identity.items()):
                raise RuntimeError("engine or model changed during stage")
            if validation.fingerprint(stage["tokens_file"]) != stage["token_fingerprint"]:
                raise RuntimeError("fixture changed during stage")
            stage["status"] = "completed_unqualified"
            save()
        active_stage = None
        if api:
            report["api"]["results"] = [api_check(api, args.timeout) for _ in range(api.get("requests", 1))]
            report["api"]["status"] = "completed" if all(r["passed"] for r in report["api"]["results"]) else "failed"
            if report["api"]["status"] == "failed":
                raise RuntimeError("API request failed")
        report["status"] = "completed_unqualified"
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        if active_stage is not None and active_stage["status"] == "not_run":
            active_stage["status"] = "failed_execution"
        report["status"] = "failed"
        report["error"] = str(exc)
    save()
    print(f"{report['status']}: {output / 'suite.json'}; G1-G7 not established")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
