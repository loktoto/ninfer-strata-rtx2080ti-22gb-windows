#!/usr/bin/env python3
"""Install the pinned compiler-only CUDA toolkit; no GPU or driver is required.

Python 3.12+, standard library only. Archives are cached in the system temporary
directory by default. The explicit destination must be absent or empty.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile


LOCK = Path(__file__).resolve().parents[1] / "config/cuda-compile.lock.json"


def archive_path(name):
    """Reject traversal, Windows drive paths, and ambiguous archive names."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(component, cache):
    url, expected = component["url"], component["sha256"]
    archive = cache / (expected + "-" + url.rsplit("/", 1)[1])
    if archive.exists() and checksum(archive) == expected:
        return archive
    # A temporary file prevents an interrupted transfer from becoming a cache hit.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, stream)
        if checksum(temporary) != expected:
            raise ValueError(f"SHA256 mismatch for {url}")
        temporary.replace(archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return archive


def extract(archive, destination):
    if archive.name.endswith(".tar.xz"):
        with tarfile.open(archive, "r:xz") as bundle:
            for member in bundle.getmembers():
                archive_path(member.name)
            # Python's data filter rejects external links, special files, and
            # writes through symlinks while retaining safe relative symlinks.
            bundle.extractall(destination, filter="data")
    else:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                relative = archive_path(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("Unexpected symlink in Windows CUDA archive")
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir() or roots[0].is_symlink():
        raise ValueError(f"Expected one top-level CUDA directory in {archive.name}")
    return roots[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path,
                        default=Path(tempfile.gettempdir()) / "sage-cuda-archives")
    args = parser.parse_args()
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64") or platform.system() not in ("Linux", "Windows"):
        parser.error("Only Linux x86_64 and Windows x86_64 are supported")
    system = "linux-x86_64" if platform.system() == "Linux" else "windows-x86_64"
    directory = args.directory.absolute()
    if directory.is_symlink() or (directory.exists() and
                                (not directory.is_dir() or any(directory.iterdir()))):
        parser.error("--directory must be absent or an empty directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    args.cache_directory.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix=".sage-cuda-", dir=directory.parent) as work:
        work = Path(work)
        merged = work / "toolkit"
        merged.mkdir()
        for name, component in lock["components"].items():
            print(f"Installing {name} {component['version']} ({system})", flush=True)
            archive = download(component[system], args.cache_directory)
            unpacked = work / name
            unpacked.mkdir()
            root = extract(archive, unpacked)
            shutil.copytree(root, merged, symlinks=True, dirs_exist_ok=True)
        if system == "linux-x86_64":
            # NVIDIA's redistributables use lib; nvcc's toolkit layout uses lib64.
            (merged / "lib64").symlink_to("lib", target_is_directory=True)
        if directory.exists():
            directory.rmdir()
        merged.replace(directory)
    print(f"CUDA {lock['cuda_release']} compiler toolkit installed at {directory}")


if __name__ == "__main__":
    main()
