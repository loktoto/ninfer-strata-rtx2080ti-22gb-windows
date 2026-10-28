#!/usr/bin/env python3
"""Compare complete SAGELOG1 teacher-forced logit dumps (reference first)."""
import argparse
import array
import math
import os
from pathlib import Path
import struct
import sys

MAX_VOCAB = 10_000_000
MAX_TOKENS = 1_000_000


class Dump:
    def __init__(self, path):
        self.file = open(path, 'rb')
        try:
            header = self.file.read(24)
            if len(header) != 24:
                raise ValueError('truncated header')
            magic, version, self.vocab, prompts, self.steps = struct.unpack('<8sIIII', header)
            if magic != b'SAGELOG1' or version != 1:
                raise ValueError('unsupported magic or version')
            if not 0 < self.vocab <= MAX_VOCAB or not 0 < prompts <= MAX_TOKENS or not 0 < self.steps <= MAX_TOKENS:
                raise ValueError('invalid dimensions')
            expected = 24 + 4 * (prompts + self.steps + self.vocab * self.steps)
            if os.fstat(self.file.fileno()).st_size != expected:
                raise ValueError('file size does not match dimensions')
            ids = self.file.read(4 * (prompts + self.steps))
            self.ids = struct.unpack(f'<{prompts + self.steps}i', ids)
            if any(i < 0 or i >= self.vocab for i in self.ids):
                raise ValueError('token outside vocabulary')
            self.prompts = prompts
        except Exception:
            self.file.close()
            raise

    def row(self):
        values = array.array('f')
        values.frombytes(self.file.read(self.vocab * 4))
        if sys.byteorder != 'little':
            values.byteswap()
        if len(values) != self.vocab or not all(map(math.isfinite, values)):
            raise ValueError('truncated or non-finite logit row')
        return values

    def close(self):
        self.file.close()


def compare(reference, candidate, atol, rtol):
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError('tolerances must be finite and nonnegative')
    a = Dump(reference)
    try:
        b = Dump(candidate)
        try:
            if (a.vocab, a.prompts, a.steps, a.ids) != (b.vocab, b.prompts, b.steps, b.ids):
                raise ValueError('dimensions or teacher-forcing token IDs differ')
            count = failed = top1 = 0
            max_abs = sum_sq = kl = 0.0
            for _ in range(a.steps):
                x, y = a.row(), b.row()
                mx, my = max(x), max(y)
                top1 += x.index(mx) == y.index(my)
                sx = sum(math.exp(v - mx) for v in x)
                sy = sum(math.exp(v - my) for v in y)
                log_sx, log_sy = math.log(sx), math.log(sy)
                for xv, yv in zip(x, y):
                    diff = abs(xv - yv)
                    max_abs = max(max_abs, diff)
                    sum_sq += diff * diff
                    failed += diff > atol + rtol * abs(xv)
                    logp = xv - mx - log_sx
                    logq = yv - my - log_sy
                    kl += math.exp(logp) * (logp - logq)
                count += a.vocab
            return {'steps': a.steps, 'vocab': a.vocab, 'values': count,
                    'failed': failed, 'max_abs': max_abs, 'rms': math.sqrt(sum_sq / count),
                    'top1_agree': top1, 'mean_kl_ref_to_candidate': kl / a.steps}
        finally:
            b.close()
    finally:
        a.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--atol', type=float, required=True)
    p.add_argument('--rtol', type=float, required=True)
    args = p.parse_args()
    try:
        result = compare(args.reference, args.candidate, args.atol, args.rtol)
    except (OSError, ValueError, struct.error) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(' '.join(f'{k}={v}' for k, v in result.items()))
    return 1 if result['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
