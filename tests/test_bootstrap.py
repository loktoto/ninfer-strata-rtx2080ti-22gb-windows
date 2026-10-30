import contextlib
import difflib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "tools/bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class PatchSeriesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.run_git("init", "--quiet")
        self.run_git("config", "user.name", "Fixture")
        self.run_git("config", "user.email", "fixture@example.invalid")
        self.run_git("config", "core.autocrlf", "false")
        self.source = self.repo / "main.cpp"
        self.source.write_bytes(b"first\nsecond\nthird\n")
        self.run_git("add", ".")
        self.run_git("commit", "--quiet", "-m", "fixture")
        self.patches = []
        old = "first\nsecond\nthird\n"
        for index, new in enumerate(("first\nchanged\nthird\n", "new-first\nchanged\nthird\n")):
            patch = self.root / f"{index}.patch"
            patch.write_text("".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                        fromfile="a/main.cpp", tofile="b/main.cpp")), newline="\n")
            self.patches.append(patch)
            old = new

    def run_git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def apply(self, patches):
        with contextlib.redirect_stdout(io.StringIO()):
            bootstrap.apply_series(self.repo, patches)

    def test_overlapping_patches_repeat_and_extend(self):
        self.apply(self.patches[:1])
        self.apply(self.patches)
        self.apply(self.patches)
        self.assertEqual(self.source.read_text(), "new-first\nchanged\nthird\n")

    def test_divergent_edit_preserved(self):
        self.apply(self.patches[:1])
        self.source.write_text("user edits\n")
        with self.assertRaises(ValueError):
            self.apply(self.patches)
        self.assertEqual(self.source.read_text(), "user edits\n")

    def test_crlf_checkout(self):
        self.run_git("config", "core.autocrlf", "true")
        self.source.unlink()
        self.run_git("checkout", "--", "main.cpp")
        self.apply(self.patches)
        self.apply(self.patches)
        self.assertEqual(self.source.read_text(), "new-first\nchanged\nthird\n")


if __name__ == "__main__":
    unittest.main()

