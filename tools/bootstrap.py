"""Fetch the pinned Strata tree and apply the unified integration patch."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def git(directory, *args, check=True):
    return subprocess.run(["git", "-C", str(directory), *args], check=check,
                          text=True, capture_output=True)


def apply_series(target, patches):
    """Recognize complete or partial ordered patch series, even overlapping hunks.

    Reconstruct expected snapshots from HEAD in an isolated temporary Git tree.
    Refuse divergent user edits rather than overwriting them.
    """
    paths = sorted({line[6:] for patch in patches for line in patch.read_text().splitlines()
                    if line.startswith("+++ b/")})
    if not paths or any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
        raise ValueError("Invalid integration patch paths")
    def snapshot(directory):
        return {p: (directory / p).read_bytes().replace(b"\r\n", b"\n") for p in paths}
    current = snapshot(target)
    with tempfile.TemporaryDirectory(prefix="sage-patch-check-") as directory:
        temp = Path(directory)
        git(temp, "init", "--quiet")
        for p in paths:
            data = subprocess.check_output(["git", "-C", str(target), "show", "HEAD:" + p])
            (temp / p).parent.mkdir(parents=True, exist_ok=True)
            (temp / p).write_bytes(data)
        matched = 0 if current == snapshot(temp) else None
        for index, patch in enumerate(patches, 1):
            git(temp, "apply", str(patch.resolve()))
            if current == snapshot(temp):
                matched = index
        if matched is None:
            raise ValueError("Patched source differs from every expected patch-series state; preserve/review local edits before setup")
    for patch in patches[:matched]:
        print(f"Already applied: {patch.name}")
    for patch in patches[matched:]:
        git(target, "apply", "--check", str(patch.resolve()))
        git(target, "apply", str(patch.resolve()))
        print(f"Applied: {patch.name}")


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
    apply_series(target, patches)
    print(f"Strata prepared at {target}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr or str(exc)) from exc
