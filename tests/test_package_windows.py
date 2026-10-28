import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

from tools import package_windows as pkg


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / 'build'
        self.build.mkdir()
        self.engine = self.build / 'strata-poly.exe'
        # Small PE-like binary fixtures; dumpbin behavior is mocked, not a Windows build.
        self.engine.write_bytes(b'MZ\x00\x00engine')
        self.args = argparse.Namespace(build_dir=self.build, binary=None, dll_dir=[],
            dumpbin='dumpbin', license_file=[], cuda_license=None,
            dry_run=False, output=self.root / 'bundle.zip')

    def dll(self, name, contents=b'MZ\x00\x00dll'):
        p = self.build / name
        p.write_bytes(contents)
        return p

    @staticmethod
    def dumpbin(graph):
        def run(args, **kwargs):
            name = Path(args[-1]).name.lower()
            text = 'Image has the following dependencies:\n\n' + ''.join('    ' + x + '\n' for x in graph.get(name, [])) + '\n  Summary\n'
            return subprocess.CompletedProcess(args, 0, text, '')
        return mock.patch.object(pkg.subprocess, 'run', side_effect=run)

    def test_transitive_closure_and_archive(self):
        self.dll('first.dll')
        self.dll('second.dll')
        with self.dumpbin({'strata-poly.exe': ['first.dll', 'KERNEL32.dll', 'api-ms-win-core-file-l1-1-0.dll'], 'first.dll': ['second.dll']}):
            info = pkg.package(self.args)
        self.assertEqual(info['dlls'], ['first.dll', 'second.dll'])
        with zipfile.ZipFile(self.args.output) as archive:
            self.assertIn('second.dll', archive.namelist())
            self.assertEqual(json.loads(archive.read('manifest.json'))['status'], 'development_unqualified')
            archive.extractall(self.root / 'extract')
        pkg.verify_directory(self.root / 'extract')
        (self.root / 'extract' / 'first.dll').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            pkg.verify_directory(self.root / 'extract')

    def test_missing_dependency_and_system_contracts(self):
        self.assertTrue(pkg.is_system('api-ms-win-core-file-l1-1-0.dll'))
        self.assertTrue(pkg.is_system('ext-ms-win-ntuser-window-l1-1-0.dll'))
        self.assertFalse(pkg.is_system('made-up.dll'))
        with self.dumpbin({'strata-poly.exe': ['made-up.dll']}):
            with self.assertRaisesRegex(ValueError, 'missing DLL made-up.dll'):
                pkg.package(self.args)
        self.assertFalse(self.args.output.exists())

    def test_conflicting_dll_names(self):
        self.dll('foo.dll', b'a')
        other = self.root / 'other'
        other.mkdir()
        (other / 'FOO.dll').write_bytes(b'b')
        self.args.dll_dir = [other]
        with self.dumpbin({'strata-poly.exe': ['foo.dll']}):
            with self.assertRaisesRegex(ValueError, 'conflicting DLLs'):
                pkg.package(self.args)

    def test_preserves_existing_output(self):
        self.args.output.write_bytes(b'keep')
        with self.dumpbin({}):
            with self.assertRaisesRegex(ValueError, 'already exists'):
                pkg.package(self.args)
        self.assertEqual(self.args.output.read_bytes(), b'keep')

    def test_cuda_requires_license(self):
        self.dll('cudart64_12.dll')
        with self.dumpbin({'strata-poly.exe': ['cudart64_12.dll']}):
            with self.assertRaisesRegex(ValueError, 'CUDA DLLs require'):
                pkg.package(self.args)
            license_file = self.root / 'nvidia-eula.txt'
            license_file.write_text('license')
            self.args.cuda_license = license_file
            pkg.package(self.args)
        with zipfile.ZipFile(self.args.output) as archive:
            self.assertIn('licenses/nvidia-eula.txt', archive.namelist())

    def test_explicit_binary_and_dry_run(self):
        q6 = self.build / 'sage_q6k_slots_parity.exe'
        q6.write_bytes(b'MZq6')
        self.args.binary = [self.engine, q6]
        self.args.dry_run = True
        with self.dumpbin({}):
            info = pkg.package(self.args)
        self.assertIn(q6.name, info['binaries'])
        self.assertFalse(self.args.output.exists())

    def test_dumpbin_failure_or_malformed_output_fails_closed(self):
        for code, output in [(1, 'failure'), (0, 'unexpected')]:
            with mock.patch.object(pkg.subprocess, 'run', return_value=subprocess.CompletedProcess([], code, output, '')):
                with self.assertRaises(ValueError):
                    pkg.dependencies(self.engine, 'dumpbin')


if __name__ == '__main__':
    unittest.main()
