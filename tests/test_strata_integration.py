import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launch_strict", ROOT / "integration/strata/launch_strict.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class StrictLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = Path(self.tmp.name) / "engine with spaces.exe"
        self.model = Path(self.tmp.name) / "model.gguf"
        self.engine.touch()
        self.model.touch()

    def test_default_overrides_inherited_skipping_and_backend(self):
        args, env = launcher.launch_spec(self.engine, self.model, ["--", "--tokens", "1,2"],
            environ={"STRATA_ROUTE_TAIL_SKIP": "8", "STRATA_SAGE_RMSNORM": "1",
                     "STRATA_SAGE_EXPERT_Q4K": "1", "PATH": "keep"})
        self.assertEqual(args[:3], [str(self.engine.resolve()), "--model", str(self.model.resolve())])
        self.assertEqual(args[-2:], ["--tokens", "1,2"])
        self.assertEqual(env["STRATA_ROUTE_TAIL_SKIP"], "0")
        self.assertEqual(env["STRATA_SAGE_RMSNORM"], "0")
        self.assertEqual(env["STRATA_SAGE_EXPERT_Q4K"], "0")
        self.assertEqual(env["PATH"], "keep")
        self.assertEqual(args[args.index("--skip-weight") + 1], "0")
        self.assertIn("--no-draft", args)
        self.assertIn("--no-graphs", args)

    def test_backend_requires_explicit_flag(self):
        _, env = launcher.launch_spec(self.engine, self.model, ninfer_rmsnorm=True, environ={})
        self.assertEqual(env["STRATA_SAGE_RMSNORM"], "1")
        self.assertEqual(env["STRATA_SAGE_EXPERT_Q4K"], "0")

    def test_expert_backend_requires_explicit_flag(self):
        _, env = launcher.launch_spec(self.engine, self.model, ninfer_expert_q4k=True, environ={})
        self.assertEqual(env["STRATA_SAGE_EXPERT_Q4K"], "1")
        self.assertEqual(env["STRATA_SAGE_RMSNORM"], "0")

    def test_dry_run_reports_expert_backend_without_launching(self):
        output = io.StringIO()
        with patch.object(launcher.subprocess, "run") as run, contextlib.redirect_stdout(output):
            result = launcher.main(["--engine", str(self.engine), "--model", str(self.model),
                                    "--ninfer-expert-q4k", "--dry-run"])
        self.assertEqual(result, 0)
        run.assert_not_called()
        launch = json.loads(output.getvalue())
        self.assertEqual(launch["environment"]["STRATA_SAGE_EXPERT_Q4K"], "1")
        self.assertNotIn("--ninfer-expert-q4k", launch["argv"])

    def test_reject_controlled_forwarded_options(self):
        for option in launcher.CONTROLLED:
            for arg in [option, option + "=1"]:
                with self.subTest(arg=arg), self.assertRaises(ValueError):
                    launcher.launch_spec(self.engine, self.model, [arg, "1"])

    def test_missing_artifact(self):
        with self.assertRaises(ValueError):
            launcher.launch_spec(self.engine, self.model.with_name("absent"))

    def test_cli_uses_argument_vector_and_propagates_exit(self):
        with patch.object(launcher.subprocess, "run") as run:
            run.return_value.returncode = 7
            result = launcher.main(["--engine", str(self.engine), "--model", str(self.model),
                                    "--", "--tokens", "1,2"])
        self.assertEqual(result, 7)
        self.assertIsInstance(run.call_args.args[0], list)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["env"]["STRATA_ROUTE_TAIL_SKIP"], "0")
        self.assertEqual(run.call_args.kwargs["env"]["STRATA_SAGE_EXPERT_Q4K"], "0")

    def test_cli_passes_expert_backend_environment(self):
        with patch.object(launcher.subprocess, "run") as run:
            run.return_value.returncode = 0
            result = launcher.main(["--engine", str(self.engine), "--model", str(self.model),
                                    "--ninfer-expert-q4k", "--", "--tokens", "1,2"])
        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.kwargs["env"]["STRATA_SAGE_EXPERT_Q4K"], "1")
        self.assertEqual(run.call_args.args[0][-2:], ["--tokens", "1,2"])

    def test_q6k_is_independent_and_default_off(self):
        _, env = launcher.launch_spec(self.engine, self.model, environ={"STRATA_SAGE_EXPERT_Q6K": "1"})
        self.assertEqual(env["STRATA_SAGE_EXPERT_Q6K"], "0")
        _, env = launcher.launch_spec(self.engine, self.model, ninfer_expert_q6k=True, environ={})
        self.assertEqual(env["STRATA_SAGE_EXPERT_Q6K"], "1")
        self.assertEqual(env["STRATA_SAGE_EXPERT_Q4K"], "0")


if __name__ == "__main__":
    unittest.main()
