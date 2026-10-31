"""Keep a local snapshot of evidence and source; publish only its digest index."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT / 'evidence-archives'
    destination.mkdir(exist_ok=True)
    archive = destination / f'session-{stamp}.zip'
    sources = []
    for directory in ('evidence', 'docs', 'tools', 'tests', 'patches', 'integration', 'kernels', 'config', 'research'):
        sources.extend(p for p in (ROOT / directory).rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts
                       and 'published-benchmarks' not in p.parts)
    sources.extend(ROOT / p for p in ('README.md', 'CMakeLists.txt', 'NOTICE', 'LICENSE'))
    records = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        for path in sorted(set(sources)):
            relative = path.relative_to(ROOT).as_posix()
            digest = hashlib.sha256()
            size = 0
            with path.open('rb') as source, output.open(relative, 'w', force_zip64=True) as saved:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    saved.write(block)
                    digest.update(block)
                    size += len(block)
            records.append({'path': relative, 'bytes': size, 'sha256': digest.hexdigest()})
    with zipfile.ZipFile(archive) as saved:
        bad_file = saved.testzip()
        if bad_file:
            raise RuntimeError(f'snapshot CRC verification failed: {bad_file}')
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    archive_digest = digest.hexdigest()
    manifest = {'schema_version': 1, 'captured_at_utc': stamp,
                'local_archive': archive.relative_to(ROOT).as_posix(),
                'archive_sha256': archive_digest, 'archive_bytes': archive.stat().st_size,
                'file_count': len(records), 'files': records,
                'scope': 'Raw local evidence and project source/docs at capture time; no model weights, build trees, or dependency checkouts. Raw archive is local only.'}
    manifest_path = ROOT / 'evidence/published-benchmarks/2026-10-31-local-evidence-index.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
