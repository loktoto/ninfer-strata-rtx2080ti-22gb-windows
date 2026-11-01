import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
import psutil

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--process', required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
records = '\n'.join(f'Record {i:02d}: shelf {chr(65 + (i % 8))}; code K{i:03d}-{(i * 37) % 997:03d}; status checked.' for i in range(1, 51))
prompts = [
    ('arithmetic', 'A warehouse has 17 crates with 12 bottles each. It sells 47 bottles and receives 5 more crates of 12. How many bottles remain? Show the arithmetic.'),
    ('python', 'Write a Python function is_prime(n) that returns True exactly when an integer n is prime. Do not import libraries. Give two short example calls.'),
    ('retrieval', 'Read the records below. What exact code belongs to Record 27? Reply with the code only.\n\n' + records),
]
all_results = {'engine': args.name, 'port': args.port, 'process': args.process,
               'settings': {'temperature': 0, 'top_p': 1, 'max_tokens': 256,
                            'enable_thinking': False, 'repetitions': 3},
               'results': [], 'memory_samples': []}
stop = threading.Event()
def monitor():
    while not stop.is_set():
        try:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'], text=True, timeout=5).strip().splitlines()[0]
            used, free, sm, mem, temp, watts, gpu_util, mem_util = [float(x.strip()) for x in gpu.split(',')]
            rss = sum(p.memory_info().rss for p in psutil.process_iter(['name']) if (p.info.get('name') or '').lower() == args.process.lower())
            all_results['memory_samples'].append({'gpu_used_mib': used, 'gpu_free_mib': free, 'process_working_set_bytes': rss, 'sm_clock_mhz': sm, 'memory_clock_mhz': mem, 'temperature_c': temp, 'power_watts': watts, 'gpu_utilization_pct': gpu_util, 'memory_utilization_pct': mem_util})
        except (subprocess.SubprocessError, OSError, ValueError, psutil.Error):
            pass
        stop.wait(1.0)
def save():
    args.output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def request(case, content, iteration):
    body = {'model': 'sage-a1.5', 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'top_p': 1, 'max_tokens': 256, 'seed': 1, 'cache_prompt': False, 'stream': True,
            'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
        data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    start = time.perf_counter()
    result = {'case': case, 'iteration': iteration, 'request': body, 'content': '', 'reasoning': '',
              'ttft_seconds': None, 'finish_reason': None, 'usage': None, 'timings': None, 'done': False}
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            result['http_status'] = response.status
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'): continue
                data = line[5:].strip()
                if data == '[DONE]':
                    result['done'] = True
                    break
                obj = json.loads(data)
                if obj.get('error'): result['error'] = obj['error']
                for choice in obj.get('choices', []):
                    delta = choice.get('delta', {})
                    for field, key in [('content', 'content'), ('reasoning_content', 'reasoning')]:
                        part = delta.get(field)
                        if isinstance(part, str) and part:
                            if result['ttft_seconds'] is None: result['ttft_seconds'] = time.perf_counter() - start
                            result[key] += part
                    if choice.get('finish_reason'): result['finish_reason'] = choice['finish_reason']
                if obj.get('usage') is not None: result['usage'] = obj['usage']
                if obj.get('timings') is not None: result['timings'] = obj['timings']
    except Exception as e:
        result['error'] = repr(e)
    result['wall_seconds'] = time.perf_counter() - start
    return result

thread = threading.Thread(target=monitor, daemon=True)
thread.start()
try:
    for iteration in range(1, 4):
        for case, prompt in prompts:
            item = request(case, prompt, iteration)
            all_results['results'].append(item)
            save()
            print(args.name, case, iteration, 'finish=', item['finish_reason'], 'usage=', item['usage'], 'ttft=', item['ttft_seconds'], 'error=', item.get('error'), flush=True)
finally:
    stop.set()
    thread.join(timeout=10)
    save()
