#!/usr/bin/env python3
"""Stage a development Windows bundle after resolving its PE import closure."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ('strata-poly.exe', 'sage_rmsnorm_parity.exe', 'sage_q4k_slots_parity.exe', 'sage_q6k_slots_parity.exe', 'sage_mma_probe.exe')
# Conservative names supplied by Windows or the NVIDIA display driver. Do not treat
# arbitrary unresolved imports as system DLLs. API-set contracts are OS forwarders.
SYSTEM_DLLS = frozenset('kernel32.dll kernelbase.dll ntdll.dll user32.dll gdi32.dll gdi32full.dll advapi32.dll shell32.dll ole32.dll oleaut32.dll comdlg32.dll comctl32.dll ws2_32.dll winmm.dll version.dll bcrypt.dll crypt32.dll secur32.dll rpcrt4.dll shlwapi.dll setupapi.dll cfgmgr32.dll imm32.dll ucrtbase.dll msvcrt.dll nvcuda.dll'.split())
DLL_RE = re.compile(r'^\s+([^\s\\/:]+\.dll)\s*$', re.I)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def is_system(name: str) -> bool:
    name = name.lower()
    return name in SYSTEM_DLLS or bool(re.fullmatch(r'(?:api|ext)-ms-win-[a-z0-9-]+\.dll', name))


def dependencies(path: Path, dumpbin: str) -> list[str]:
    proc = subprocess.run([dumpbin, '/DEPENDENTS', str(path)], capture_output=True, text=True, errors='replace')
    if proc.returncode:
        raise ValueError(f'dumpbin failed for {path}: {proc.stderr or proc.stdout}')
    # Require the expected heading; a successful but malformed tool result is not proof.
    if 'Image has the following dependencies:' not in proc.stdout:
        raise ValueError(f'dumpbin did not report dependencies for {path}')
    section = proc.stdout.split('Image has the following dependencies:', 1)[1].split('Summary', 1)[0]
    return [m.group(1) for line in section.splitlines() if (m := DLL_RE.match(line))]


def resolve(binaries: list[Path], directories: list[Path], dumpbin: str) -> dict[str, Path]:
    index: dict[str, list[Path]] = {}
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f'DLL directory does not exist: {directory}')
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() == '.dll':
                index.setdefault(path.name.lower(), []).append(path.resolve())
    result: dict[str, Path] = {}
    seen: set[Path] = set()
    queue = list(binaries)
    while queue:
        path = queue.pop(0).resolve()
        if path in seen:
            continue
        seen.add(path)
        for name in dependencies(path, dumpbin):
            key = name.lower()
            if is_system(key):
                continue
            candidates = index.get(key, [])
            if not candidates:
                raise ValueError(f'missing DLL {name}, required by {path}')
            hashes = {digest(candidate) for candidate in candidates}
            if len(hashes) != 1:
                raise ValueError(f'conflicting DLLs named {name}: ' + ', '.join(map(str, candidates)))
            if key not in result:
                result[key] = candidates[0]
                queue.append(candidates[0])
    return result


def provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    return {'source_commit': git('rev-parse', 'HEAD'),
            'working_tree_dirty': bool(git('status', '--porcelain')),
            'patch_sha256': {p.name: digest(p) for p in sorted((ROOT / 'patches').glob('*.patch'))},
            'sources_lock_sha256': digest(ROOT / 'config' / 'sources.lock.json')}


def verify_directory(stage: Path) -> None:
    manifest = json.loads((stage / 'manifest.json').read_text(encoding='utf-8'))
    expected = manifest['files']
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()} - {'manifest.json'}
    if actual != set(expected):
        raise ValueError('manifest file list does not match staged files')
    for name, value in expected.items():
        if digest(stage / name) != value:
            raise ValueError(f'checksum mismatch: {name}')


def package(args: argparse.Namespace) -> dict:
    build = args.build_dir.resolve()
    if not build.is_dir():
        raise ValueError(f'build directory does not exist: {build}')
    if args.binary:
        binaries = [p.resolve() for p in args.binary]
    else:
        binaries = []
        for name in DEFAULTS:
            matches = list(build.rglob(name))
            if len(matches) > 1:
                raise ValueError(f'multiple {name} files; select explicitly with --binary')
            if matches:
                binaries.append(matches[0].resolve())
        if not any(p.name.lower() == 'strata-poly.exe' for p in binaries):
            raise ValueError('strata-poly.exe not found in build directory')
    if not any(p.name.lower() == 'strata-poly.exe' for p in binaries):
        raise ValueError('selection must include strata-poly.exe')
    if len({p.name.lower() for p in binaries}) != len(binaries):
        raise ValueError('select distinct executable basenames')
    for path in binaries:
        if not path.is_file() or path.suffix.lower() != '.exe' or not path.is_relative_to(build):
            raise ValueError(f'binary must be an existing .exe inside build directory: {path}')
    dirs = list(dict.fromkeys([p.parent for p in binaries] + [p.resolve() for p in args.dll_dir]))
    dlls = resolve(binaries, dirs, args.dumpbin)
    licenses = [p.resolve() for p in args.license_file]
    if any(not p.is_file() for p in licenses):
        raise ValueError('all --license-file paths must be existing files')
    cuda = [n for n in dlls if n.startswith(('cudart', 'cublas', 'cufft', 'curand', 'cusolver', 'cusparse', 'nvrtc', 'nvjitlink', 'nvfatbin', 'nvptxcompiler', 'nvblas', 'cudnn'))]
    if cuda and not args.cuda_license:
        raise ValueError('CUDA DLLs require an explicit --cuda-license file from the toolkit or NVIDIA')
    if args.cuda_license:
        p = args.cuda_license.resolve()
        if not p.is_file():
            raise ValueError(f'CUDA license file does not exist: {p}')
        licenses.append(p)
    if len({p.name.lower() for p in licenses}) != len(licenses):
        raise ValueError('license files must have distinct basenames')
    info = {'status': 'development_unqualified', 'provenance': provenance(),
            'binaries': [p.name for p in binaries], 'dlls': sorted(dlls), 'cuda_dlls': sorted(cuda)}
    if args.dry_run:
        return info
    out = args.output.resolve()
    if out.exists():
        raise ValueError(f'output already exists: {out}')
    if not out.parent.is_dir():
        raise ValueError(f'output parent does not exist: {out.parent}')
    with tempfile.TemporaryDirectory(prefix='strata-package-') as temp:
        stage = Path(temp)
        for path in binaries + list(dlls.values()):
            shutil.copyfile(path, stage / path.name)
        licdir = stage / 'licenses'
        licdir.mkdir()
        shutil.copyfile(ROOT / 'LICENSE', licdir / 'LICENSE')
        shutil.copyfile(ROOT / 'NOTICE', licdir / 'NOTICE')
        for path in licenses:
            if path.name.lower() in {'license', 'notice'}:
                raise ValueError(f'license basename collides with project license: {path.name}')
            shutil.copyfile(path, licdir / path.name)
        (stage / 'README.txt').write_text('Development package, unqualified for release. Native engine and GPU parity executables require a compatible NVIDIA driver and GPU; execution has not been verified by this packaging tool. Models are not included.\n\nThe Python API is not included. From the prepared repository use: python tools/serve_sage.py --engine <extracted-directory>\\strata-poly.exe --model <local-model> --output-dir <new-directory>\nSee docs/API.md and docs/WINDOWS_PACKAGE.md in the repository.\n', encoding='utf-8')
        info['files'] = {p.relative_to(stage).as_posix(): digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        verify_directory(stage)
        try:
            with out.open('xb') as f:
                try:
                    with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED) as z:
                        for path in sorted(stage.rglob('*')):
                            if path.is_file():
                                z.write(path, path.relative_to(stage).as_posix())
                except BaseException:
                    f.close()
                    out.unlink()
                    raise
        except FileExistsError:
            raise ValueError(f'output already exists: {out}') from None
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new ZIP path; never overwritten')
    parser.add_argument('--binary', type=Path, action='append', help='explicit built .exe, repeatable')
    parser.add_argument('--dll-dir', type=Path, action='append', default=[], help='additional runtime DLL directory, repeatable')
    parser.add_argument('--license-file', type=Path, action='append', default=[])
    parser.add_argument('--cuda-license', type=Path, help='official NVIDIA license covering bundled CUDA files')
    parser.add_argument('--dumpbin', default='dumpbin')
    parser.add_argument('--dry-run', action='store_true', help='resolve and report without creating output')
    args = parser.parse_args()
    if os.name != 'nt' and not args.dry_run:
        parser.error('packaging requires native Windows; --dry-run permits inspection elsewhere with a compatible dumpbin')
    try:
        print(json.dumps(package(args), indent=2))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
