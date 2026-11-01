"""Run balanced MTP off/on/on/off tests for an explicitly selected fusion flag."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / 'tools/benchmark_api.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--engine', required=True, type=Path, help='same built engine used for all four starts')
    parser.add_argument('--reference-engine', type=Path, help='optional previous binary for two-start build sanity check')
    parser.add_argument('--fusion-env', required=True, help='explicit environment key controlling fusion')
    parser.add_argument('--candidate-label', default='fusion', choices=('fusion', 'packed-head'),
                        help='label for the enabled condition; does not change the selected environment key')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--port-base', type=int, default=18400)
    args = parser.parse_args()
    if not re.fullmatch(r'STRATA_[A-Z0-9_]+', args.fusion_env):
        parser.error('fusion environment key must be a STRATA_ identifier')
    if not args.engine.is_file() or not COMPARE.is_file() or (args.reference_engine and not args.reference_engine.is_file()):
        parser.error('missing built engine or local benchmark harness')
    original = json.loads(args.config.read_text(encoding='utf8'))
    if '--no-draft' in original['args'] or original['args'][original['args'].index('--window') + 1] != '4':
        parser.error('input config must use MTP window 4')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop('GGML_CUDA_DISABLE_GRAPHS', None)
    env['PATH'] = str(ROOT / '.cuda/bin/x64') + os.pathsep + env['PATH']
    results = []
    runs = [('control', False, args.engine), (args.candidate_label, True, args.engine),
            (args.candidate_label, True, args.engine), ('control', False, args.engine)]
    if args.reference_engine:
        runs = [('reference', False, args.reference_engine), ('build_sanity', False, args.engine)] + runs
    for index, (profile, enabled, engine) in enumerate(runs, 1):
        port = args.port_base + index
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'port {port} is in use')
        config = json.loads(json.dumps(original))
        config['exe'] = str(engine.resolve())
        config['host'] = '127.0.0.1'
        config['port'] = port
        config['env'][args.fusion_env] = '1' if enabled else '0'
        label = f'{index}-{profile}'
        cfg_file = out / (label + '-config.json')
        cfg_file.write_text(json.dumps(config, indent=2), encoding='utf8')
        compare_file = out / (label + '-comparison.json')
        command = [sys.executable, '-m', 'serve.server', '--engine', 'strata', '--config', str(cfg_file),
                   '--host', '127.0.0.1', '--port', str(port)]
        with (out / (label + '-server.log')).open('w', encoding='utf8') as log:
            server = subprocess.Popen(command, cwd=ROOT / '.deps/PolyStrata', env=env, stdout=log,
                                      stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    if server.poll() is not None: raise RuntimeError(f'{label} exited {server.returncode}')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2) as response:
                            if response.status == 200: break
                    except (OSError, TimeoutError): time.sleep(2)
                else: raise TimeoutError(f'{label} startup timeout')
                print(label, 'ready', flush=True)
                subprocess.run([sys.executable, str(COMPARE), '--name', label, '--port', str(port),
                                '--process', 'strata-poly.exe', '--output', str(compare_file)], env=env,
                               cwd=ROOT, check=True, timeout=300)
            finally:
                subprocess.run(['taskkill', '/PID', str(server.pid), '/T', '/F'], capture_output=True, timeout=20)
        results.append((profile, json.loads(compare_file.read_text(encoding='utf8'))['results']))
    summary = {'schema_version': 1, 'cases': {}}
    for case in sorted({r['case'] for _, rows in results for r in rows}):
        ref = next(r for _, rows in results for r in rows if r['case'] == case)
        groups = {}
        for label in ('control', args.candidate_label):
            rows = [r for profile, entries in results if profile == label for r in entries if r['case'] == case and r['iteration'] > 1]
            if len(rows) != 4 or any(not r['done'] or r.get('error') for r in rows):
                raise RuntimeError(f'{case}/{label}: incomplete comparison')
            rates = [r['timings']['predicted_per_second'] for r in rows]
            groups[label] = {'decode_samples': rates, 'median': statistics.median(rates),
                'completion_tokens': [r['usage']['completion_tokens'] for r in rows],
                'draft_offered': sum(r['timings'].get('draft_n', 0) for r in rows),
                'draft_accepted': sum(r['timings'].get('draft_n_accepted', 0) for r in rows),
                'content_and_reasoning_match_control': all((r['content'], r['reasoning']) == (ref['content'], ref['reasoning']) for r in rows)}
        groups[args.candidate_label + '_vs_control_percent'] = 100 * (groups[args.candidate_label]['median'] / groups['control']['median'] - 1)
        if args.reference_engine:
            groups['build_sanity'] = {}
            for label in ('reference', 'build_sanity'):
                rows = [r for profile, entries in results if profile == label for r in entries if r['case'] == case and r['iteration'] > 1]
                if len(rows) != 2 or any(not r['done'] or r.get('error') for r in rows):
                    raise RuntimeError(f'{case}/{label}: incomplete build sanity')
                groups['build_sanity'][label] = {'decode_samples': [r['timings']['predicted_per_second'] for r in rows],
                    'content_and_reasoning_match_reference': all((r['content'], r['reasoning']) == (ref['content'], ref['reasoning']) for r in rows)}
        summary['cases'][case] = groups
    (out / 'numeric-summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
