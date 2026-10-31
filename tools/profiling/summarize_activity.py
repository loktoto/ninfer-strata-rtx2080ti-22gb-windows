"""Summarize an opt-in CUPTI trace within its recorded local request interval."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads((args.run / 'request-metrics.json').read_text(encoding='utf8'))
    if not request['request_completed']:
        raise ValueError('request did not complete')
    clocks, statuses, problems = [], [], []
    with (args.run / 'activity.tsv').open(encoding='utf8') as stream:
        for line in stream:
            row = line.rstrip('\n').split('\t')
            if row[0] == 'clock':
                if len(row) != 4 or row[3] != '0': raise ValueError('invalid or failed clock anchor')
                clocks.append((int(row[1]), (int(row[2]) - 116444736000000000) * 100))
            elif row[0] in ('kernel', 'api'):
                expected = 7 if row[0] == 'kernel' else 6
                if len(row) != expected: raise ValueError('malformed activity record')
                start, end = int(row[1]), int(row[2])
                if start <= 0 or end <= 0 or end < start: raise ValueError('invalid activity interval')
            elif row[0] == 'status': statuses.append(row[1:])
            elif row[0] in ('dropped', 'record_error'): problems.append(row)
    if (len(clocks) < 2 or problems or any(s[1] != '0' for s in statuses)
            or not any(s[0] == 'initialized' for s in statuses)):
        raise ValueError('invalid clock anchors, CUPTI status, or dropped/error records')
    if any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(clocks, clocks[1:])):
        raise ValueError('nonmonotonic clock anchors')
    if request['request_end_unix_ns'] <= request['request_start_unix_ns']:
        raise ValueError('invalid request interval')
    offsets = [wall - gpu for gpu, wall in clocks]
    offset = statistics.median(offsets)
    first = request['request_start_unix_ns'] - offset
    last = request['request_end_unix_ns'] - offset
    kernels = collections.defaultdict(lambda: {'count': 0, 'duration_ns': 0})
    apis = collections.defaultdict(lambda: {'count': 0, 'duration_ns': 0})
    graph_ids = set()
    kernel_count = graph_kernel_count = 0
    with (args.run / 'activity.tsv').open(encoding='utf8') as stream:
        for line in stream:
            row = line.rstrip('\n').split('\t')
            if row[0] not in ('kernel', 'api'): continue
            start, end = int(row[1]), int(row[2])
            if not (first <= start <= end <= last): continue
            if row[0] == 'kernel':
                name = row[6]; graph = int(row[4]); kernel_count += 1
                if graph: graph_ids.add(graph); graph_kernel_count += 1
                dest = kernels[name]
            else:
                name = row[5]; dest = apis[name]
            dest['count'] += 1
            dest['duration_ns'] += end - start
    if not kernel_count: raise ValueError('no GPU kernels in request window')
    result = {'schema_version': 1, 'request_completed': True, 'completion_tokens': request['usage']['completion_tokens'],
              'clock_anchor_count': len(clocks), 'clock_offset_spread_ns': max(offsets)-min(offsets),
              'drop_or_record_errors': len(problems), 'kernel_records': kernel_count,
              'graph_kernel_records': graph_kernel_count, 'distinct_nonzero_graph_ids': len(graph_ids),
              'kernels': dict(sorted(kernels.items(), key=lambda item: -item[1]['duration_ns'])),
              'launch_apis': dict(sorted(apis.items(), key=lambda item: -item[1]['duration_ns'])),
              'note': 'Request window includes prefill and decode; sums can overlap and instrumentation changes timing.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('kernel records', kernel_count, 'graph records', graph_kernel_count,
          'graph IDs', len(graph_ids), 'clock anchors', len(clocks), 'drops/errors', len(problems))


if __name__ == '__main__':
    main()
