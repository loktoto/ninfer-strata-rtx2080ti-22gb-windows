import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('benchmark_export', Path(__file__).resolve().parents[1]/'tools/export_benchmark_evidence.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class BenchmarkExportTests(unittest.TestCase):
    def test_metrics_survive_without_request_or_response_text(self):
        row = {'case': 'arithmetic', 'iteration': 2, 'http_status': 200, 'done': True,
               'finish_reason': 'stop', 'request': {'messages': ['PRIVATE_PROMPT']},
               'content': 'PRIVATE_RESPONSE', 'reasoning': 'PRIVATE_REASONING',
               'usage': {'prompt_tokens': 52, 'completion_tokens': 240,
                         'prompt_tokens_details': {'cached_tokens': 0}},
               'timings': {'predicted_per_second': 115.1, 'prompt_per_second': 361.4,
                           'draft_n': 0}, 'ttft_seconds': 0.16, 'wall_seconds': 2.3}
        result = exporter.sample(row)
        self.assertEqual(result['decode_tok_s'], 115.1)
        self.assertEqual(result['completion_tokens'], 240)
        self.assertEqual(result['cached_tokens'], 0)
        self.assertFalse(result['had_error'])
        self.assertNotIn('PRIVATE_', json.dumps(result))
        self.assertNotIn('request', result)

    def test_missing_measurements_remain_missing_and_errors_are_retained(self):
        result = exporter.sample({'case': 'python', 'iteration': 1, 'error': 'PRIVATE_ERROR'})
        self.assertTrue(result['had_error'])
        self.assertIsNone(result['decode_tok_s'])
        self.assertIsNone(result['cached_tokens'])
        self.assertNotIn('PRIVATE_', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
