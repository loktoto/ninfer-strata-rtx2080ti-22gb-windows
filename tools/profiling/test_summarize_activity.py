"""Host-only checks for request window selection and trace validity gates."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SUMMARIZER = Path(__file__).with_name('summarize_activity.py')


class SummaryTests(unittest.TestCase):
    def test_window_selection_and_rejection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.profiling') as tmp:
            root = Path(tmp)
            # FILETIME 116444736000000000 corresponds to Unix time zero.
            (root / 'request-metrics.json').write_text(json.dumps({
                'request_completed': True, 'request_start_unix_ns': 1000,
                'request_end_unix_ns': 3000, 'usage': {'completion_tokens': 1}}))
            trace = root / 'activity.tsv'
            trace.write_text('schema\t1\nstatus\tinitialized\t0\tCUPTI_SUCCESS\n'
                'clock\t1000\t116444736000000010\t0\nclock\t3000\t116444736000000030\t0\n'
                'kernel\t1200\t2000\t1\t7\t1\tin_window\n'
                'kernel\t500\t999\t1\t0\t1\tout_of_window\n'
                'api\t1300\t1500\t1\t1\tcudaGraphLaunch\n')
            output = root / 'summary.json'
            cmd = [sys.executable, str(SUMMARIZER), str(root), '--output', str(output)]
            subprocess.run(cmd, check=True, capture_output=True)
            result = json.loads(output.read_text())
            self.assertEqual(result['kernel_records'], 1)
            self.assertEqual(result['graph_kernel_records'], 1)
            self.assertEqual(result['kernels']['in_window']['duration_ns'], 800)
            self.assertNotIn('out_of_window', result['kernels'])
            for error in ('dropped\t1\t0\n', 'record_error\t123\n', 'status\tflush\t123\terror\n',
                          'clock\t4000\t116444736000000040\t1\n',
                          'clock\t2000\t116444736000000020\t0\n',
                          'kernel\t2000\t1000\t1\t7\t1\tbad\n',
                          'api\t0\t2000\t1\t1\tbad\n', 'kernel\t12\t13\n'):
                trace.write_text(trace.read_text() + error)
                self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
                trace.write_text(trace.read_text().removesuffix(error))


if __name__ == '__main__':
    unittest.main()
