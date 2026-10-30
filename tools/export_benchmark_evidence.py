"""Export numeric benchmark evidence without prompts, responses or local paths."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sample(row):
    usage = row.get('usage') or {}
    timing = row.get('timings') or {}
    return {
        'case': row['case'], 'iteration': row['iteration'],
        'http_status': row.get('http_status'), 'done': row.get('done'),
        'finish_reason': row.get('finish_reason'), 'had_error': bool(row.get('error')),
        'prompt_tokens': usage.get('prompt_tokens'),
        'completion_tokens': usage.get('completion_tokens'),
        'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
        'decode_tok_s': timing.get('predicted_per_second'),
        'prefill_tok_s': timing.get('prompt_per_second'),
        'prompt_ms': timing.get('prompt_ms'), 'decode_ms': timing.get('predicted_ms'),
        'ttft_seconds': row.get('ttft_seconds'), 'wall_seconds': row.get('wall_seconds'),
        'draft_tokens': timing.get('draft_n'), 'accepted_draft_tokens': timing.get('draft_n_accepted'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    batches = [('2026-10-29', ROOT/'evidence/iq4xs-20261029'),
               ('2026-10-30-optimization', ROOT/'evidence/iq4xs-optimization-20261030'),
               ('2026-10-30-abcd', ROOT/'evidence/abcd-20261030')]
    for label, folder in batches:
        runs = []
        for file in sorted(folder.glob('*comparison.json')):
            data = read(file)
            runs.append({'source_file': file.relative_to(ROOT).as_posix(),
                         'samples': [sample(row) for row in data['results']]})
        (a.output/(label+'-samples.json')).write_text(json.dumps({'schema_version': 1, 'runs': runs}, indent=2)+'\n')
    old = ROOT/'evidence/iq4xs-20261029'
    extra = {'schema_version': 1, 'near_32k': [], 'quality_checks': [], 'rmsnorm': {}}
    for file in sorted(old.glob('*-long.json')):
        data = read(file); response = data['response']; usage = response['usage']
        extra['near_32k'].append({'source_file': file.relative_to(ROOT).as_posix(),
            'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': usage['completion_tokens'],
            'cached_tokens': usage['prompt_tokens_details']['cached_tokens'],
            'prefill_tok_s': response['timings']['prompt_per_second'],
            'prompt_ms': response['timings']['prompt_ms'], 'wall_seconds': data['wall_seconds'],
            'answer_correct': response['choices'][0]['message']['content'].strip()==data['fixture']['expected']})
    for file in sorted(old.glob('*-quality.json')):
        data = read(file)
        extra['quality_checks'].append({'source_file': file.relative_to(ROOT).as_posix(),
            'cases': [{'case': row['case'], 'passed': row['pass']} for row in data['results']]})
    parity = read(old/'rmsnorm-integration/report.json')['comparison']['full_logits']
    extra['rmsnorm'] = {k:v for k,v in parity.items() if k!='artifacts'}
    (a.output/'2026-10-29-additional-tests.json').write_text(json.dumps(extra,indent=2)+'\n')
    optimized = ROOT/'evidence/iq4xs-optimization-20261030'
    result = read(optimized/'logits-clean/results.json')
    clean = {k: result[k] for k in ['comparison','counts','device_ids_marker','passed']}
    (a.output/'2026-10-30-correctness.json').write_text(json.dumps(clean,indent=2)+'\n')
    summary = read(optimized/'abba-summary.json')
    # Keep workload metrics; exclude machine sampling records from this export.
    (a.output/'2026-10-30-optimization-summary.json').write_text(
        json.dumps({k: summary[k] for k in ['arithmetic','python','retrieval']},indent=2)+'\n')
    print('exported', len(list(a.output.glob('*.json'))), 'numeric JSON files')


if __name__ == '__main__':
    main()
