"""Export the bounded optimization runs as numeric data and exact-match booleans."""
import argparse
import json
import statistics
from pathlib import Path
from export_benchmark_evidence import sample

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--correctness', type=Path, required=True)
    parser.add_argument('--stem', required=True, help='filename stem in published-benchmarks')
    args = parser.parse_args()
    if Path(args.stem).name != args.stem or '/' in args.stem or '\\' in args.stem:
        parser.error('stem must be a filename')
    runs = []
    reference = {}
    for file in sorted(args.folder.glob('*-comparison.json')):
        data = json.loads(file.read_text(encoding='utf8'))
        samples = []
        for row in data['results']:
            if not row['done'] or row.get('error'):
                raise ValueError('incomplete request in ' + file.name)
            values = sample(row)
            text = (row['content'], row['reasoning'])
            reference.setdefault(row['case'], text)
            values['text_and_reasoning_match_reference'] = text == reference[row['case']]
            samples.append(values)
        telemetry = {}
        for key in ('gpu_used_mib', 'gpu_free_mib', 'sm_clock_mhz', 'memory_clock_mhz',
                    'temperature_c', 'power_watts', 'gpu_utilization_pct', 'memory_utilization_pct'):
            values = [r[key] for r in data.get('memory_samples', []) if isinstance(r.get(key), (int, float))]
            if values:
                telemetry[key] = dict(min=min(values), max=max(values), median=statistics.median(values), samples=len(values))
        runs.append(dict(run=file.name.removesuffix('-comparison.json'), samples=samples, telemetry=telemetry))
    if len(runs) not in (4, 6):
        raise ValueError('expected four ABBA starts and optionally two build controls')
    result = {'schema_version': 1, 'runs': runs,
              'paired_summary': json.loads((args.folder/'numeric-summary.json').read_text(encoding='utf8'))}
    raw = json.loads(args.correctness.read_text(encoding='utf8'))
    result['correctness'] = {key: raw[key] for key in ('binary_sha256', 'candidate', 'comparison', 'counts',
                            'device_ids_marker', 'passed', 'fused_host_dispatch', 'packed_head_selftest', 'q6_grid') if key in raw}
    target = ROOT/'evidence/published-benchmarks'/(args.stem+'.json')
    target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('exported', len(runs), 'runs and', sum(len(r['samples']) for r in runs), 'numeric requests')


if __name__ == '__main__':
    main()
