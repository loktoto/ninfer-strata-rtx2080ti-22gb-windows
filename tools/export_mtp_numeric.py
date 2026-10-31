"""Export local draft-window experiments as aggregate numeric records, excluding request text and local paths."""
import json
import statistics
from pathlib import Path
from export_benchmark_evidence import sample

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'evidence' / 'iq4xs-mtp-20261030'
OUT = ROOT / 'evidence' / 'published-benchmarks'

def main():
    files=sorted(DIR.glob('*-comparison.json'))
    rows=[]
    for f in files:
        d=json.loads(f.read_text(encoding='utf8'))
        samples=[sample(r) for r in d['results']]
        if any(r['had_error'] or not r['done'] for r in samples): raise ValueError(f'incomplete run: {f.name}')
        metrics=d.get('memory_samples',[])
        keys=('gpu_used_mib','sm_clock_mhz','memory_clock_mhz','temperature_c','power_watts','gpu_utilization_pct','memory_utilization_pct')
        telemetry={}
        for k in keys:
            values=[m[k] for m in metrics if isinstance(m.get(k),(int,float))]
            if values:telemetry[k]={'min':min(values),'max':max(values),'median':statistics.median(values),'samples':len(values)}
        rows.append({'run':f.name.removesuffix('-comparison.json'),'samples':samples,'telemetry':telemetry})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'2026-10-30-mtp-samples.json').write_text(json.dumps({'schema_version':1,'runs':rows},indent=2)+'\n',encoding='utf8')
    summary=json.loads((DIR/'summary.json').read_text(encoding='utf8'))
    (OUT/'2026-10-30-mtp-abba-summary.json').write_text(json.dumps({'schema_version':1,'cases':summary},indent=2)+'\n',encoding='utf8')
    best = {name: json.loads((DIR/f'best-{name}-comparison.json').read_text(encoding='utf8'))
            for name in ('1-A','2-D','3-D','4-A','5-A-MTP')}
    cases = {}
    for case in sorted({r['case'] for r in best['1-A']['results']}):
        ref = next(r for r in best['1-A']['results'] if r['case'] == case)
        groups = {}
        for group, runs in {'A':('1-A','4-A'), 'D':('2-D','3-D'), 'A-MTP':('5-A-MTP',)}.items():
            selected = [r for run in runs for r in best[run]['results'] if r['case']==case and r['iteration']>1]
            rates = [r['timings']['predicted_per_second'] for r in selected]
            offered = sum(r['timings'].get('draft_n',0) for r in selected)
            accepted = sum(r['timings'].get('draft_n_accepted',0) for r in selected)
            groups[group] = {'decode_samples':rates, 'median':statistics.median(rates),
                'completion_tokens':[r['usage']['completion_tokens'] for r in selected],
                'draft_offered':offered, 'draft_accepted':accepted,
                'text_and_reasoning_match_A':all((r['content'],r['reasoning']) == (ref['content'],ref['reasoning']) for r in selected)}
        groups['D_vs_A_percent'] = 100*(groups['D']['median']/groups['A']['median']-1)
        cases[case] = groups
    (OUT/'2026-10-30-mtp-best-abba-summary.json').write_text(json.dumps({'schema_version':1,'cases':cases},indent=2)+'\n',encoding='utf8')
    print('exported',len(rows),'numeric runs')
if __name__=='__main__':main()
