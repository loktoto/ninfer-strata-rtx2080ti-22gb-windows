"""Preserve head-only timings and correctness without local launch paths."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
paired = re.compile(r'ncols (\d+): GGUF ([\d.]+) us \(([\d.]+)-([\d.]+)\), packed ([\d.]+) us \(([\d.]+)-([\d.]+)\), 11 paired samples')
sweep = re.compile(r'grid (\d+) ncols (\d+): ([\d.]+) us \(([\d.]+)-([\d.]+)\)')


def main():
    runs = []
    for folder, grid in [('q6-head-logits-20261101', 640), ('q6-head-grid-20261101', 640),
                         ('q6-head-tuned-logits-20261101', 20480)]:
        directory = ROOT/'evidence'/folder
        filename = 'stderr.txt' if 'grid' in folder else 'candidate.stderr.txt'
        text = (directory/filename).read_text(encoding='utf8', errors='replace')
        timings = [dict(columns=int(c), native_median_us=float(n), native_min_us=float(nmin),
                        native_max_us=float(nmax), packed_median_us=float(p),
                        packed_min_us=float(pmin), packed_max_us=float(pmax), paired_samples=11)
                   for c,n,nmin,nmax,p,pmin,pmax in paired.findall(text)]
        grids = [dict(grid=int(g), columns=int(c), median_us=float(m), min_us=float(lo), max_us=float(hi), samples=7)
                 for g,c,m,lo,hi in sweep.findall(text)]
        if len(timings) != 3 or 'outputs BITWISE EQUAL (1-4 columns)' not in text:
            raise ValueError('incomplete head self-test: '+folder)
        run = dict(run=folder, selected_grid=grid, paired=timings, exploratory_grid_sweep=grids,
                   selftest_byte_equal=True)
        if (directory/'results.json').is_file():
            raw = json.loads((directory/'results.json').read_text(encoding='utf8'))
            run['model_check'] = {k: raw[k] for k in ('binary_sha256','comparison','passed','packed_head_selftest')}
        runs.append(run)
    data = dict(schema_version=1, rows=248320, columns=2048, additional_device_bytes=417177600,
                runs=runs, note='Head-only CUDA-event timing; no end-to-end speed claim. The grid sweep is exploratory, not paired with a new native measurement at each grid.')
    (ROOT/'evidence/published-benchmarks/2026-11-01-q6-head.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    print('exported',len(runs),'head experiments')


if __name__ == '__main__':
    main()
