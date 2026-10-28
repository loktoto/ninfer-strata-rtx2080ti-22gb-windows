"""Launch the single Strata runtime with complete routing and diagnostic defaults."""
import argparse
import json
import os
from pathlib import Path
import subprocess

# These options belong to this wrapper; disallow duplicates and '=' spellings.
CONTROLLED = frozenset({"--model", "--tail-skip", "--skip-weight", "--kv", "--window",
                        "--lookup", "--no-draft", "--draft-model", "--draft-vocab",
                        "--no-graphs"})
DEFAULTS = ["--tail-skip", "0", "--skip-weight", "0", "--no-draft", "--lookup", "0",
            "--window", "1", "--no-graphs", "--kv", "fp16"]


def launch_spec(engine, model, extra=(), ninfer_rmsnorm=False, environ=None,
                ninfer_expert_q4k=False, ninfer_expert_q6k=False):
    """Return shell-free argv/environment, rejecting conflicting passthrough flags."""
    extra = list(extra)
    if extra and extra[0] == "--":
        extra.pop(0)
    for arg in extra:
        if arg.split("=", 1)[0] in CONTROLLED:
            raise ValueError(f"{arg}: controlled by the strict launcher")
    engine, model = Path(engine).resolve(), Path(model).resolve()
    if not engine.is_file():
        raise ValueError(f"Engine file does not exist: {engine}")
    if not model.is_file():
        raise ValueError(f"Model file does not exist: {model}")
    env = dict(os.environ if environ is None else environ)
    env.update(STRATA_ROUTE_TAIL_SKIP="0", STRATA_FREEZE_EXPERTS="1", STRATA_SAGE_TRACE="0",
               STRATA_SAGE_RMSNORM="1" if ninfer_rmsnorm else "0",
               STRATA_SAGE_EXPERT_Q4K="1" if ninfer_expert_q4k else "0",
               STRATA_SAGE_EXPERT_Q6K="1" if ninfer_expert_q6k else "0")
    return [str(engine), "--model", str(model), *DEFAULTS, *extra], env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ninfer-rmsnorm", action="store_true",
                        help="Opt into the experimental NInfer-derived FP32 RMSnorm")
    parser.add_argument("--ninfer-expert-q4k", action="store_true",
                        help="Opt into experimental direct Q4_K cached expert-slot products")
    parser.add_argument("--ninfer-expert-q6k", action="store_true", help="Opt into NInfer-all-derived Q6_K slot products")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact launch without running")
    parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        command, env = launch_spec(args.engine, args.model, args.engine_args, args.ninfer_rmsnorm,
                                  ninfer_expert_q4k=args.ninfer_expert_q4k, ninfer_expert_q6k=args.ninfer_expert_q6k)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"argv": command, "environment": {key: env[key] for key in
              ("STRATA_ROUTE_TAIL_SKIP", "STRATA_FREEZE_EXPERTS", "STRATA_SAGE_RMSNORM",
               "STRATA_SAGE_EXPERT_Q4K", "STRATA_SAGE_EXPERT_Q6K")}}, indent=2))
        return 0
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
