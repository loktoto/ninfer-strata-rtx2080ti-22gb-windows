"""Host-only checks for the pinned, single-runtime native API launcher."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "serve_sage.py"
SPEC = importlib.util.spec_from_file_location("serve_sage", TOOL)
serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve)
PIN = json.loads((ROOT / "config" / "sources.lock.json").read_text())["strata"]["commit"]


class ServeSageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sage api ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = self.root / "strata-poly.exe"
        self.engine.write_bytes(b"placeholder engine: never executed")
        self.model = self.root / "Sage actual.gguf"
        self.model.write_bytes(b"GGUF placeholder: subprocess extraction is mocked")
        self.upstream = self.root / "PolyStrata"
        for relative in ("serve/server.py", "tools/strata_tokenizer.py", "requirements.txt"):
            path = self.upstream / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n")
        self.output = self.root / "saved run"
        self.argv = ["--engine", str(self.engine), "--model", str(self.model),
                     "--upstream", str(self.upstream), "--output-dir", str(self.output)]
        self.calls = []
        self.extract_status = 0
        self.server_status = 0
        self.template = "{{ messages | tojson }} actual GGUF template"

    def subprocess(self, command, **kwargs):
        self.calls.append((command, kwargs))
        self.assertIsInstance(command, list)
        self.assertNotIn("shell", kwargs)
        self.assertEqual(kwargs["check"], False)
        if command[0] == "git":
            self.assertEqual(command, ["git", "-C", str(self.upstream), "rev-parse", "HEAD"])
            return subprocess.CompletedProcess(command, 0, stdout=PIN + "\n")
        self.assertEqual(kwargs["cwd"], str(self.upstream))
        if command[1] == str(self.upstream / "tools/strata_tokenizer.py"):
            self.assertEqual(command[2:4], ["--gguf", str(self.model)])
            self.assertEqual(command[4:6], ["--out", str(self.output / "pack")])
            if not self.extract_status:
                tokenizer = self.output / "pack" / "tokenizer"
                tokenizer.mkdir(parents=True)
                for name in ("vocab.json", "tokenizer.json", "merges.txt", "token_type.json"):
                    (tokenizer / name).write_text("{}")
                if self.template is not None:
                    (tokenizer / "chat_template.jinja").write_text(self.template)
            return subprocess.CompletedProcess(command, self.extract_status)
        self.assertEqual(command[1:5], ["-m", "serve.server", "--engine", "strata"])
        self.assertEqual(command[5:], ["--config", str(self.output / "strata-sage.json"),
                                      "--host", "127.0.0.1", "--port", "8080"])
        self.assertTrue((self.output / "strata-sage.json").is_file())
        return subprocess.CompletedProcess(command, self.server_status)

    def invoke(self, *extra, system="Windows"):
        with mock.patch.object(serve.platform, "system", return_value=system), \
             mock.patch.object(serve.subprocess, "run", side_effect=self.subprocess), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = serve.main(self.argv + list(extra))
        return result, stdout.getvalue()

    def config(self):
        return json.loads((self.output / "strata-sage.json").read_text())

    def test_native_pipeline_and_strict_config_use_explicit_paths(self):
        with mock.patch.dict(os.environ, {"STRATA_SAGE_RMSNORM": "1", "STRATA_SAGE_EXPERT_Q4K": "1",
                                          "STRATA_ROUTE_TAIL_SKIP": "1", "STRATA_FREEZE_EXPERTS": "0"}):
            code, _ = self.invoke("--max-context", "16384", "--threads", "8")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 3)  # read pin, extract tokenizer, run real server
        cfg = self.config()
        self.assertEqual(cfg["exe"], str(self.engine))
        self.assertEqual(cfg["cwd"], str(self.upstream))
        self.assertEqual(cfg["tokenizer"], str(self.output / "pack" / "tokenizer"))
        self.assertEqual(cfg["args"], ["--model", str(self.model), "--tail-skip", "0", "--skip-weight", "0",
                                      "--no-draft", "--lookup", "0", "--window", "1", "--no-graphs",
                                      "--kv", "fp16", "--max-context", "16384", "--threads", "8"])
        self.assertNotIn("--serve", cfg["args"])  # StrataEngine adds --serve exactly once
        self.assertFalse(cfg["fit_max_tokens"])
        self.assertFalse(cfg["drop_oldest"])
        self.assertEqual(cfg["host"], "127.0.0.1")
        self.assertEqual(cfg["port"], 8080)
        self.assertEqual(cfg["model_name"], "sage-a1.5")
        self.assertEqual(cfg["env"], {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_FREEZE_EXPERTS": "1",
                                      "STRATA_SAGE_RMSNORM": "0", "STRATA_SAGE_EXPERT_Q4K": "0", "STRATA_SAGE_EXPERT_Q6K": "0"})
        for _, kwargs in self.calls[1:]:
            for key, value in cfg["env"].items():
                self.assertEqual(kwargs["env"][key], value)
        self.assertEqual((self.output / "pack/tokenizer/chat_template.jinja").read_text(), self.template)

    def test_candidates_are_independent_explicit_opt_ins(self):
        self.invoke("--ninfer-rmsnorm", "--model-name", "sage-candidate")
        self.assertEqual(self.config()["env"]["STRATA_SAGE_RMSNORM"], "1")
        self.assertEqual(self.config()["env"]["STRATA_SAGE_EXPERT_Q4K"], "0")
        self.assertEqual(self.config()["model_name"], "sage-candidate")

    def test_q6k_opt_in(self):
        code, _ = self.invoke("--ninfer-expert-q6k")
        self.assertEqual(code, 0)
        self.assertEqual(self.config()["env"]["STRATA_SAGE_EXPERT_Q6K"], "1")
        self.assertEqual(self.config()["env"]["STRATA_SAGE_EXPERT_Q4K"], "0")

    def test_q4k_opt_in(self):
        self.invoke("--ninfer-expert-q4k")
        self.assertEqual(self.config()["env"]["STRATA_SAGE_EXPERT_Q4K"], "1")
        self.assertEqual(self.config()["env"]["STRATA_SAGE_RMSNORM"], "0")

    def test_dry_run_reads_pin_but_does_not_write_extract_or_launch(self):
        code, output = self.invoke("--dry-run", system="Linux")
        plan = json.loads(output)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(plan["config"]["args"][-2:], ["--max-context", "8192"])
        self.assertEqual(plan["tokenizer_argv"][3], str(self.model))
        self.assertIn("unverified", plan["note"])

    def test_non_windows_run_requires_dry_run(self):
        with mock.patch.object(serve.platform, "system", return_value="Linux"), \
             mock.patch.object(serve.subprocess, "run") as run, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            serve.main(self.argv)
        self.assertEqual(exc.exception.code, 2)
        run.assert_not_called()

    def test_extraction_failure_propagates_and_never_starts_api(self):
        self.extract_status = 23
        self.assertEqual(self.invoke()[0], 23)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.output / "strata-sage.json").exists())

    def test_server_failure_propagates(self):
        self.server_status = 19
        self.assertEqual(self.invoke()[0], 19)
        self.assertTrue((self.output / "strata-sage.json").exists())

    def test_missing_template_refuses_upstream_fallback(self):
        self.template = None
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            self.invoke()
        self.assertEqual(exc.exception.code, 2)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.output / "strata-sage.json").exists())

    def test_existing_output_is_never_clobbered(self):
        self.output.mkdir()
        sentinel = self.output / "strata-sage.json"
        sentinel.write_text("keep this config")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke("--dry-run")
        self.assertEqual(sentinel.read_text(), "keep this config")
        self.assertEqual(len(self.calls), 1)

    def test_wrong_pin_is_rejected_without_download(self):
        with mock.patch.object(serve.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="wrong\n")) as run, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            serve.main(self.argv + ["--dry-run"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][-2:], ["rev-parse", "HEAD"])
        self.assertFalse(self.output.exists())

    def test_missing_paths_fail_even_during_dry_run(self):
        self.model.unlink()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), \
             mock.patch.object(serve.subprocess, "run") as run:
            serve.main(self.argv + ["--dry-run"])
        run.assert_not_called()

    def test_unknown_executable_name_refuses_wrong_server_protocol(self):
        renamed = self.root / "other-engine.exe"
        self.engine.rename(renamed)
        argv = [str(renamed) if value == str(self.engine) else value for value in self.argv]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), \
             mock.patch.object(serve.subprocess, "run") as run:
            serve.main(argv + ["--dry-run"])
        run.assert_not_called()

    def test_cli_rejects_conflicting_options_and_invalid_context(self):
        for extra in (["--max-context", "0"], ["--threads", "-1"], ["--fit-max-tokens"],
                      ["--host", "0.0.0.0"], ["--", "--kv", "q8_0"]):
            with self.subTest(extra=extra):
                run = subprocess.run([sys.executable, str(TOOL), *self.argv, "--dry-run", *extra],
                                     text=True, capture_output=True, check=False)
                self.assertEqual(run.returncode, 2)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
