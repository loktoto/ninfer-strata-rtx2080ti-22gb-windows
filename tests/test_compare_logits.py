import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from compare_logits import compare


def dump(path, rows, prompt=(0,), follow=(1,)):
    vocab = len(rows[0])
    path.write_bytes(struct.pack('<8sIIII', b'SAGELOG1', 1, vocab, len(prompt), len(follow))
                     + struct.pack(f'<{len(prompt) + len(follow)}i', *(prompt + follow))
                     + struct.pack(f'<{sum(map(len, rows))}f', *(v for row in rows for v in row)))


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = Path(self.tmp.name) / 'a'
        self.b = Path(self.tmp.name) / 'b'

    def test_numeric_threshold_and_metrics(self):
        dump(self.a, [[1, 2]])
        dump(self.b, [[1.125, 2]])
        ok = compare(self.a, self.b, 0.125, 0)
        self.assertEqual(ok['failed'], 0)
        self.assertEqual(ok['max_abs'], 0.125)
        self.assertAlmostEqual(ok['rms'], 0.125 / math.sqrt(2))
        self.assertEqual(ok['top1_agree'], 1)
        self.assertGreater(ok['mean_kl_ref_to_candidate'], 0)
        self.assertEqual(compare(self.a, self.b, 0.124, 0)['failed'], 1)
        self.assertEqual(compare(self.a, self.b, 0, 0.125)['failed'], 0)

    def test_multiple_steps_and_top1(self):
        dump(self.a, [[1, 2], [4, 0]], prompt=(0, 1), follow=(1, 0))
        dump(self.b, [[2, 1], [4, 0]], prompt=(0, 1), follow=(1, 0))
        result = compare(self.a, self.b, 0, 0)
        self.assertEqual(result['steps'], 2)
        self.assertEqual(result['failed'], 2)
        self.assertEqual(result['top1_agree'], 1)

    def test_corrupt_or_mismatched(self):
        dump(self.a, [[1, 2]])
        dump(self.b, [[1, 2]], follow=(0,))
        with self.assertRaisesRegex(ValueError, 'teacher-forcing'):
            compare(self.a, self.b, 0, 0)
        dump(self.b, [[1, 2]], follow=(2,))
        with self.assertRaisesRegex(ValueError, 'outside vocabulary'):
            compare(self.a, self.b, 0, 0)
        dump(self.b, [[1, float('nan')]])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            compare(self.a, self.b, 0, 0)
        self.b.write_bytes(self.a.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'size'):
            compare(self.a, self.b, 0, 0)
        self.b.write_bytes(self.a.read_bytes() + b'x')
        with self.assertRaisesRegex(ValueError, 'size'):
            compare(self.a, self.b, 0, 0)
        with self.assertRaisesRegex(ValueError, 'tolerances'):
            compare(self.a, self.a, float('nan'), 0)


if __name__ == '__main__':
    unittest.main()
