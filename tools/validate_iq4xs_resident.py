"""Compare graph-eligible resident IQ4_XS execution against the native path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from compare_logits import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True, type=Path)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--prompt-file', required=True, type=Path)
    p.add_argument('--follow-file', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='new directory')
    p.add_argument('--layers', type=int, default=40)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    follow = [int(x.strip()) for x in a.follow_file.read_text(encoding='utf-8').strip().split(',')]
    if not follow or a.layers < 1:
        p.error('follow-file and layers must be nonempty')
    a.output.mkdir(parents=True, exist_ok=False)
    result = {'binary_sha256': hashlib.sha256(a.engine.read_bytes()).hexdigest(), 'runs': {}}
    for candidate in (False, True):
        name = 'candidate' if candidate else 'native'
        command = [str(a.engine.resolve()), '--model', str(a.model.resolve()), '--tail-skip', '0',
                   '--skip-weight', '0', '--no-draft', '--lookup', '0', '--window', '1', '--kv', 'fp16',
                   '--tokens-file', str(a.prompt_file.resolve()), '--follow-file', str(a.follow_file.resolve()),
                   '--max-context', '32768', '--max-new', str(len(follow)), '--seed', '1', '--temperature', '0',
                   '--threads', '16', '--logits-file', str((a.output / (name + '.logits')).resolve())]
        flags = {'STRATA_ROUTE_TAIL_SKIP': '0', 'STRATA_FREEZE_EXPERTS': '1', 'STRATA_SAGE_RMSNORM': '0',
                 'STRATA_SAGE_EXPERT_Q4K': '0', 'STRATA_SAGE_EXPERT_Q6K': '0',
                 'STRATA_SAGE_EXPERT_IQ4XS': '1' if candidate else '0', 'STRATA_SAGE_TRACE': '1'}
        env = dict(os.environ, **flags)
        start = time.monotonic()
        with (a.output / (name + '.stdout.txt')).open('wb') as out, (a.output / (name + '.stderr.txt')).open('wb') as err:
            proc = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=a.timeout)
        result['runs'][name] = {'argv': command, 'flags': flags, 'seconds': time.monotonic() - start, 'exit': proc.returncode}
        if proc.returncode:
            raise RuntimeError(f'{name} exited {proc.returncode}; see stderr')
    result['comparison'] = compare(a.output / 'native.logits', a.output / 'candidate.logits', 0.0, 0.0)
    stderr = (a.output / 'candidate.stderr.txt').read_text(encoding='utf-8', errors='replace')
    stdout = (a.output / 'candidate.stdout.txt').read_text(encoding='utf-8', errors='replace')
    counts = {int(layer): [int(gate), int(down)] for layer, gate, down in
              re.findall(r'sage-device-count: layer=(\d+) in2048=(\d+) other=(\d+)', stderr)}
    result['counts'] = counts
    result['device_ids_marker'] = 'sage-device: expert-iq4xs-ids executed' in stdout
    result['passed'] = (result['comparison']['failed'] == 0 and result['device_ids_marker'] and
                        all(i in counts and min(counts[i]) >= len(follow) for i in range(a.layers)))
    (a.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'comparison': result['comparison'],
                      'counted_layers': len(counts)}, indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
