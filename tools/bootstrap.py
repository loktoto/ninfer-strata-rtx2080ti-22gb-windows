"""Fetch the pinned Strata tree and apply the unified integration patch."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(directory, *args, check=True):
    return subprocess.run(["git", "-C", str(directory), *args], check=check,
                          text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / ".deps" / "PolyStrata")
    args = parser.parse_args()
    target = args.directory.resolve()
    source = json.loads((ROOT / "config/sources.lock.json").read_text())["strata"]
    if not target.exists():
        target.mkdir(parents=True)
        git(target, "init")
        git(target, "remote", "add", "origin", source["url"])
        git(target, "fetch", "--depth", "1", "origin", source["commit"])
        git(target, "checkout", "--detach", "FETCH_HEAD")
    head = git(target, "rev-parse", "HEAD").stdout.strip()
    if head != source["commit"]:
        raise SystemExit(f"Refusing to patch {head}; expected {source['commit']}")
    patches = sorted((ROOT / "patches").glob("*.patch"))
    if not patches:
        raise SystemExit("No integration patches found")
    # Reverse-check makes repeated setup safe without overwriting local edits.
    for patch in patches:
        if git(target, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
            print(f"Already applied: {patch.name}")
            continue
        git(target, "apply", "--check", str(patch))
        git(target, "apply", str(patch))
        print(f"Applied: {patch.name}")
    print(f"Strata prepared at {target}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr or str(exc)) from exc
