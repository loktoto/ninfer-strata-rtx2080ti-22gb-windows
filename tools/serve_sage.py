"""Start the pinned PolyStrata API with one native Windows Sage runtime."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration" / "strata"))
from launch_strict import launch_spec

ENV_KEYS = ("STRATA_ROUTE_TAIL_SKIP", "STRATA_FREEZE_EXPERTS",
            "STRATA_SAGE_TRACE", "STRATA_SAGE_RMSNORM", "STRATA_SAGE_EXPERT_Q4K",
            "STRATA_SAGE_EXPERT_Q6K", "STRATA_SAGE_EXPERT_IQ4XS", "STRATA_SAGE_FUSED_NORM_Q8",
            "STRATA_Q6_PACKED", "STRATA_Q6P_SELFTEST")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def validate_upstream(upstream):
    """Require the prepared source pin; never clone, fetch, or run upstream setup."""
    for relative in ("serve/server.py", "tools/strata_tokenizer.py", "requirements.txt"):
        if not (upstream / relative).is_file():
            raise ValueError(f"Missing pinned upstream file: {upstream / relative}; run tools/bootstrap.py first")
    expected = json.loads((ROOT / "config" / "sources.lock.json").read_text(encoding="utf-8"))["strata"]["commit"]
    result = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False)
    if result.returncode or result.stdout.strip() != expected:
        raise ValueError(f"Upstream must be pinned to {expected}; run tools/bootstrap.py first")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="built strata-poly.exe")
    parser.add_argument("--model", type=Path, required=True, help="complete, locally supplied Sage GGUF")
    parser.add_argument("--upstream", type=Path, default=ROOT / ".deps" / "PolyStrata",
                        help="prepared PolyStrata tree matching config/sources.lock.json")
    parser.add_argument("--output-dir", type=Path,
                        help="new directory for the saved config and extracted tokenizer (default: unique directory in current directory)")
    parser.add_argument("--model-name", default="sage-a1.5", help="model id exposed by /v1/models")
    parser.add_argument("--max-context", type=positive_int, default=8192,
                        help="allocated context tokens (default 8192; 256K is not qualified)")
    parser.add_argument("--threads", type=positive_int, help="CPU threads for uncached experts")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="allow CUDA graphs (diagnostic default disables them)")
    parser.add_argument("--draft-window", type=int, choices=(2, 3, 4),
                        help="experimental verification window: one known token plus up to 1–3 MTP guesses; requires --cuda-graphs")
    parser.add_argument("--ninfer-rmsnorm", action="store_true", help="opt into the experimental RMSnorm candidate")
    parser.add_argument("--ninfer-expert-q4k", action="store_true", help="opt into the experimental Q4_K candidate")
    parser.add_argument("--ninfer-expert-iq4xs", action="store_true", help="opt into the experimental IQ4_XS expert candidate")
    parser.add_argument("--ninfer-expert-q6k", action="store_true", help="opt into the Q6_K candidate")
    parser.add_argument("--fused-norm-q8", action="store_true", help="opt into experimental native Strata residual/norm/Q8 fusion; requires patch 0010")
    parser.add_argument("--dry-run", action="store_true", help="validate paths and print the plan without writing files or starting inference; also works on Linux")
    args = parser.parse_args(argv)
    if not args.dry_run and platform.system() != "Windows":
        parser.error("native startup requires Windows; use --dry-run for a Linux plan")
    if not args.model_name.strip():
        parser.error("--model-name must not be empty")
    if args.draft_window is not None and not args.cuda_graphs:
        parser.error("--draft-window requires --cuda-graphs to select the measured experimental profile")
    try:
        upstream = args.upstream.resolve()
        extra = ["--max-context", str(args.max_context)]
        if args.threads is not None:
            extra += ["--threads", str(args.threads)]
        engine_command, env = launch_spec(args.engine, args.model, extra,
                                         args.ninfer_rmsnorm, ninfer_expert_q4k=args.ninfer_expert_q4k, ninfer_expert_q6k=args.ninfer_expert_q6k, ninfer_expert_iq4xs=args.ninfer_expert_iq4xs,
                                         fused_norm_q8=args.fused_norm_q8)
        if args.cuda_graphs:
            engine_command.remove("--no-graphs")
        if args.draft_window is not None:
            engine_command.remove("--no-draft")
            engine_command[engine_command.index("--window") + 1] = str(args.draft_window)
        if Path(engine_command[0]).stem.lower() not in {"strata-poly", "strata-glm"}:
            raise ValueError("--engine must be named strata-poly.exe (or strata-poly on Linux dry runs); the server selects its GGUF protocol by executable name")
        validate_upstream(upstream)
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        output = (args.output_dir or Path.cwd() / ("sage-api-" + suffix)).resolve()
        if output.exists():
            raise ValueError(f"Output directory already exists; choose a new --output-dir: {output}")
        if not output.parent.is_dir():
            raise ValueError(f"Output parent directory does not exist: {output.parent}")
        config_path = output / "strata-sage.json"
        tokenizer = output / "pack" / "tokenizer"
        config = {
            "exe": engine_command[0], "args": engine_command[1:], "cwd": str(upstream),
            "tokenizer": str(tokenizer), "model_name": args.model_name,
            "arch": "qwen35moe", "backend": "cuda", "host": "127.0.0.1", "port": 8080,
            "fit_max_tokens": False, "drop_oldest": False,
            "env": {key: env[key] for key in ENV_KEYS},
        }
        extract_command = [sys.executable, str(upstream / "tools" / "strata_tokenizer.py"),
                           "--gguf", str(args.model.resolve()), "--out", str(output / "pack")]
        server_command = [sys.executable, "-m", "serve.server", "--engine", "strata",
                          "--config", str(config_path), "--host", "127.0.0.1", "--port", "8080"]
        if args.dry_run:
            print(json.dumps({"config_path": str(config_path), "config": config,
                              "tokenizer_argv": extract_command, "server_argv": server_command,
                              "cwd": str(upstream), "environment": config["env"],
                              "note": "Plan only; tokenizer extraction, GGUF chat-template presence and inference are unverified."}, indent=2))
            return 0
        output.mkdir()  # exclusive creation: never overwrite an earlier run or tokenizer
        print(f"Extracting the supplied GGUF tokenizer into {tokenizer}", flush=True)
        result = subprocess.run(extract_command, cwd=str(upstream), env=env, check=False)
        if result.returncode:
            return result.returncode
        for required in ("vocab.json", "tokenizer.json", "merges.txt", "token_type.json", "chat_template.jinja"):
            path = tokenizer / required
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty GGUF tokenizer asset: {path}; refusing an invented template or tokenizer")
        with config_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        print(f"Saved API configuration: {config_path}", flush=True)
        return subprocess.run(server_command, cwd=str(upstream), env=env, check=False).returncode
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

