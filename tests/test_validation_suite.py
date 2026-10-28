import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('suite', ROOT / 'tools/run_validation_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class SuiteTests(unittest.TestCase):
    def test_exact_total_and_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            plan = {'stages': [{'name': 'short', 'total_tokens': None}, {'name': '32k', 'total_tokens': 32768}]}
            (base / 's.json').write_text(json.dumps({'prompt_ids': [1] * 15, 'follow_ids': [2]}))
            (base / 'l.json').write_text(json.dumps({'prompt_ids': [1] * 32760, 'follow_ids': [2] * 8}))
            manifest = {'schema_version': 1, 'stages': [{'name': 'short', 'tokens_file': 's.json', 'task_expectation': 'reviewed'}, {'name': '32k', 'tokens_file': 'l.json', 'task_expectation': 'reviewed'}]}
            path = base / 'manifest.json'
            path.write_text(json.dumps(manifest))
            stages = suite.read_manifest(path, plan)
            self.assertEqual(stages[1]['total_tokens'], 32768)
            manifest['stages'][1]['name'] = '64k'
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                suite.read_manifest(path, plan)
            manifest['stages'][1]['name'] = '32k'
            (base / 'l.json').write_text(json.dumps({'prompt_ids': [1] * 32768, 'follow_ids': [2]}))
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                suite.read_manifest(path, plan)

    def test_dry_run_and_stops_on_failed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            entries = []
            for name, total in [('short', 16), ('32k', 32768), ('64k', 65536), ('128k', 131072), ('256k', 262144)]:
                token = base / (name + '.json')
                token.write_text(json.dumps({'prompt_ids': [1] * (total - 1), 'follow_ids': [2]}))
                entries.append({'name': name, 'tokens_file': token.name, 'task_expectation': 'independently reviewed'})
            manifest = base / 'manifest.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'stages': entries}))
            for name in ('engine', 'model', 'lock', 'source'):
                (base / name).write_text('fixture')
            args = ['--engine', str(base / 'engine'), '--model', str(base / 'model'), '--manifest', str(manifest), '--model-lock', str(base / 'lock'), '--source-lock', str(base / 'source')]
            self.assertEqual(suite.main(args + ['--output', str(base / 'dry'), '--dry-run']), 0)
            dry = json.loads((base / 'dry/suite.json').read_text())
            self.assertIn('--full-logits', dry['stages'][0]['diagnostic_argv'])
            self.assertNotIn('--dump-layers', dry['stages'][0]['benchmarks']['baseline']['argv'])
            self.assertEqual(dry['profile'], 'strata_ninfer')
            self.assertEqual({g for g, status in dry['gates'].items() if status == 'deferred'}, {'G5', 'G6'})
            self.assertTrue(all(dry['gates'][g] == 'not_established' for g in ('G1', 'G2', 'G3', 'G4', 'G7')))
            completed = mock.Mock(returncode=0)
            failed = mock.Mock(returncode=1)
            with mock.patch.object(suite.platform, 'system', return_value='Windows'), mock.patch.object(suite.subprocess, 'run', side_effect=[completed, failed]) as run:
                self.assertEqual(suite.main(args + ['--output', str(base / 'real')]), 1)
            report = json.loads((base / 'real/suite.json').read_text())
            self.assertEqual(run.call_count, 2)
            self.assertEqual(report['stages'][0]['status'], 'failed_diagnostic')
            self.assertEqual(report['stages'][1]['status'], 'not_run')

    def test_task_expectation_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps({'schema_version': 1, 'stages': [{'name': 'short', 'tokens_file': 'x', 'task_expectation': ''}]}))
            with self.assertRaises(ValueError):
                suite.read_manifest(path, {'stages': [{'name': 'short', 'total_tokens': None}]})


if __name__ == '__main__':
    unittest.main()
