"""Opt-in local activity capture of one bounded API request; never installs anything."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROMPT = 'A warehouse has 17 crates with 12 bottles each. It sells 47 bottles and receives 5 more crates of 12. How many bottles remain? Show the arithmetic.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='existing local Strata config')
    parser.add_argument('--output-dir', type=Path, required=True, help='new local directory')
    parser.add_argument('--port', type=int, default=18360)
    parser.add_argument('--cupti-root', type=Path, required=True, help='extracted CUDA CUPTI archive root')
    parser.add_argument('--native', action='store_true', help='use native Strata expert kernels')
    parser.add_argument('--no-draft', action='store_true', help='disable speculative drafting')
    args = parser.parse_args()
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('Selected local port is already in use')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    trace = out / 'activity.tsv'
    dll = ROOT / '.profiling/sage_cupti_activity.dll'
    if not dll.is_file() or not (args.cupti_root / 'lib/cupti64_2025.3.1.dll').is_file():
        raise FileNotFoundError('Build the local tracing DLL and provide the extracted CUPTI root')
    config = json.loads(args.config.read_text(encoding='utf8'))
    config['host'] = '127.0.0.1'
    config['port'] = args.port
    config['env']['STRATA_SAGE_EXPERT_IQ4XS'] = '0' if args.native else '1'
    for key in ('STRATA_SAGE_RMSNORM', 'STRATA_SAGE_EXPERT_Q4K', 'STRATA_SAGE_EXPERT_Q6K', 'STRATA_SAGE_TRACE', 'STRATA_SAGE_FUSED_NORM_Q8'):
        config['env'][key] = '0'
    config['env'].pop('GGML_CUDA_DISABLE_GRAPHS', None)
    config['args'] = [arg for arg in config['args'] if arg != '--no-graphs']
    if args.no_draft:
        config['args'][config['args'].index('--window') + 1] = '1'
        if '--no-draft' not in config['args']:
            config['args'].append('--no-draft')
    for key in ('CUDA_INJECTION32_PATH', 'CUDA_INJECTION64_PATH', 'NVTX_INJECTION32_PATH', 'NVTX_INJECTION64_PATH', 'SAGE_CUPTI_TRACE_PATH'):
        config['env'].pop(key, None)
    config['env']['CUDA_INJECTION64_PATH'] = str(dll)
    config['env']['SAGE_CUPTI_TRACE_PATH'] = str(trace)
    config_file = out / 'local-config.json'
    config_file.write_text(json.dumps(config, indent=2), encoding='utf8')
    env = os.environ.copy()
    env.pop('GGML_CUDA_DISABLE_GRAPHS', None)
    for key in ('CUDA_INJECTION32_PATH', 'CUDA_INJECTION64_PATH', 'NVTX_INJECTION32_PATH', 'NVTX_INJECTION64_PATH', 'SAGE_CUPTI_TRACE_PATH'):
        env.pop(key, None)
    env['PATH'] = os.pathsep.join([str(args.cupti_root / 'lib'), str(ROOT / '.cuda/bin/x64'), env['PATH']])
    command = [sys.executable, '-m', 'serve.server', '--engine', 'strata', '--config', str(config_file),
               '--host', '127.0.0.1', '--port', str(args.port)]
    summary = {'schema_version': 1, 'max_tokens': 256, 'request_completed': False}
    log = (out / 'local-server.log').open('w', encoding='utf8')
    server = subprocess.Popen(command, cwd=ROOT / '.deps/PolyStrata', env=env, stdout=log, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'server exited: {server.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError('server startup timed out')
        body = {'model': config['model_name'], 'messages': [{'role': 'user', 'content': PROMPT}],
                'temperature': 0, 'top_p': 1, 'max_tokens': 256, 'seed': 1, 'cache_prompt': False,
                'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
        request = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        summary['request_start_unix_ns'] = time.time_ns()
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.load(response)
            summary['http_status'] = response.status
        summary['request_end_unix_ns'] = time.time_ns()
        summary['usage'] = data.get('usage')
        summary['timings'] = data.get('timings')
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '') or ''
        summary['output_sha256'] = hashlib.sha256(content.encode()).hexdigest()
        summary['request_completed'] = True
        time.sleep(4)  # allow periodic flush while the engine remains alive
    finally:
        (out / 'request-metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
        subprocess.run(['taskkill', '/PID', str(server.pid), '/T', '/F'], capture_output=True, timeout=20)
        log.close()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
