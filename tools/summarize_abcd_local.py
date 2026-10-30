"""Summarize the local A/B/D trial; C is explicitly unmeasured."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'evidence/abcd-20261030'
RUNS = {'A': ['6-A', '7-A'], 'B': ['2-B', '5-B'], 'D': ['3-D', '4-D']}


def main():
    result = {'schema_version': 1, 'C': {'status': 'not_measured',
        'reason': 'Exact Sage GGUF-to-NInfer artifact import and standalone runtime not qualified.'},
        'excluded_runs': {'1-A': 'Prompt-prefix caching enabled; superseded by cache-disabled controls.'},
        'order': ['1-A-excluded','2-B','3-D','4-D','5-B','6-A','7-A'], 'variants': {}}
    texts = {case: {} for case in ['arithmetic','python','retrieval']}
    for variant, names in RUNS.items():
        rows = [row for name in names for row in json.loads((FOLDER/(name+'-comparison.json')).read_text())['results']]
        assert len(rows) == 18
        assert all(row.get('done') and not row.get('error') and row['http_status']==200 for row in rows)
        assert all(row['usage']['prompt_tokens_details']['cached_tokens']==0 for row in rows)
        cases = {}
        for case in texts:
            all_rows = [r for r in rows if r['case']==case]
            warmed = [r for r in all_rows if r['iteration']>1]
            assert len(warmed)==4
            texts[case][variant] = {(r['content'],r['reasoning']) for r in all_rows}
            assert len(texts[case][variant])==1
            cases[case] = {
                'decode_samples_tok_s': [r['timings']['predicted_per_second'] for r in warmed],
                'prefill_samples_tok_s': [r['timings']['prompt_per_second'] for r in warmed],
                'ttft_samples_s': [r['ttft_seconds'] for r in warmed],
                'prompt_tokens': [r['usage']['prompt_tokens'] for r in warmed],
                'completion_tokens': [r['usage']['completion_tokens'] for r in warmed],
                'cached_tokens': [0]*4,
            }
            for metric in ['decode','prefill']:
                cases[case][metric+'_median_tok_s'] = statistics.median(cases[case][metric+'_samples_tok_s'])
            cases[case]['ttft_median_s'] = statistics.median(cases[case]['ttft_samples_s'])
        result['variants'][variant] = {'runs': names, 'cases': cases}
    result['exact_text_agreement'] = {case: {pair: texts[case][pair[0]]==texts[case][pair[1]]
        for pair in ['AB','AD','BD']} for case in texts}
    result['decode_percent_change'] = {case: {pair: 100*(
        result['variants'][pair[1]]['cases'][case]['decode_median_tok_s']/
        result['variants'][pair[0]]['cases'][case]['decode_median_tok_s']-1)
        for pair in ['AB','AD','BD']} for case in texts}
    out = ROOT/'evidence/published-benchmarks/abcd-summary.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
