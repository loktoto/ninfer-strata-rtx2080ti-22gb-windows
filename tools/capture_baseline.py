"""Collect P0 evidence on Windows; never marks a runtime gate as passed."""
import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess


def probe(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=60)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "error": str(exc)}


def fingerprint(path):
    path = pathlib.Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, action="append", required=True,
                        help="Repeat for every GGUF shard; hashes the entire file")
    parser.add_argument("--engine", type=pathlib.Path, required=True)
    parser.add_argument("--source-lock", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Windows":
        parser.error("P0 target is native Windows. Linux/WSL captures are not accepted.")
    if args.output.exists():
        parser.error("Output already exists; choose a new run filename.")
    lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    for path in args.model:
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                parser.error(f"Not a GGUF file: {path}")
    probes = {
        "gpu": probe(["nvidia-smi", "-q", "-x"]),
        "cuda_compiler": probe(["nvcc", "--version"]),
        "cmake": probe(["cmake", "--version"]),
        "host_compiler": probe(["cl"]),
        "windows_cpu_ram": probe(["powershell.exe", "-NoProfile", "-NonInteractive",
            "-Command", "$ErrorActionPreference='Stop'; [Console]::OutputEncoding="
            "[System.Text.Encoding]::UTF8; @{os=(Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption,Version,BuildNumber); cpu=@(Get-CimInstance Win32_Processor | "
            "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors); "
            "ram=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory} | ConvertTo-Json -Depth 5"]),
    }
    report = {
        "schema_version": 1,
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "inventory_only_requires_review",
        "gates": {f"G{i}": "not_run" for i in range(1, 8)},
        "platform": platform.platform(),
        "models": [fingerprint(p) for p in args.model],
        "engine": fingerprint(args.engine),
        "source_lock": lock,
        "source_lock_file": fingerprint(args.source_lock),
        "probes": probes,
        "limitations": ["Source lock is supplied provenance, not proof of binary provenance.",
                         "GGUF architecture, tensor types and shard completeness need inspection.",
                         "No inference, SM75 capability, unlock or performance test was run."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Inventory saved: {args.output}. No acceptance gate has passed.")


if __name__ == "__main__":
    main()
