import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from artsy_harness.cli import main
from artsy_harness.models import ModelResponse


class CLITests(unittest.TestCase):
    def test_all_sources_default_to_no_execution(self):
        for source in ("fixture", "program", "prompt"):
            for flag in ([], ["--no-execute"]):
                with self.subTest(source=source, flag=flag), tempfile.TemporaryDirectory() as root:
                    program = Path(root) / "input.py"
                    program.write_text("raise RuntimeError('not executed')\n")
                    source_args = [] if source == "fixture" else (
                        ["--program", str(program)] if source == "program" else ["--prompt", "circle"])
                    output, errors = io.StringIO(), io.StringIO()
                    response = ModelResponse(b'{"response":"pass","done":true}', 200, text="pass")
                    with patch("sys.argv", ["artsy-harness", *source_args, *flag, "--episodes", root]), \
                         patch("artsy_harness.models.Ollama.generate", return_value=response), \
                         patch("artsy_harness.harness._render") as renderer, \
                         patch("subprocess.Popen") as process, \
                         redirect_stdout(output), redirect_stderr(errors):
                        self.assertEqual(main(), 0)
                    renderer.assert_not_called()
                    process.assert_not_called()
                    record = json.loads(output.getvalue())
                    self.assertFalse(record["execution"]["enabled"])
                    self.assertFalse(record["execution"]["executed"])
                    self.assertEqual(errors.getvalue(), "")

    def test_execute_opts_in_for_all_sources_and_warns(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        for source in ("fixture", "program", "prompt"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as root:
                program = Path(root) / "input.py"
                program.write_text("pass\n")
                source_args = [] if source == "fixture" else (
                    ["--program", str(program)] if source == "program" else ["--prompt", "circle"])
                output, errors = io.StringIO(), io.StringIO()
                response = ModelResponse(b'{"response":"pass","done":true}', 200, text="pass")
                with patch("sys.argv", ["artsy-harness", *source_args, "--execute", "--episodes", root]), \
                     patch("artsy_harness.models.Ollama.generate", return_value=response), \
                     patch("artsy_harness.harness._render", return_value=(
                         {"returncode": 0}, buffer.getvalue())) as renderer, \
                     redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(main(), 0)
                renderer.assert_called_once()
                record = json.loads(output.getvalue())
                self.assertTrue(record["execution"]["enabled"])
                self.assertTrue(record["execution"]["executed"])
                self.assertFalse(record["execution"]["sandboxed"])
                self.assertIn("UNSANDBOXED", errors.getvalue())
                self.assertIn("personally reviewed", errors.getvalue())

    def test_no_execute_overrides_execute(self):
        with tempfile.TemporaryDirectory() as root, \
             patch("sys.argv", ["artsy-harness", "--execute", "--no-execute", "--episodes", root]), \
             patch("artsy_harness.harness._render") as renderer, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(), 0)
            renderer.assert_not_called()

    def test_invalid_model_output_is_retained_without_execution(self):
        for execute in ([], ["--execute"]):
            with tempfile.TemporaryDirectory() as root:
                response = ModelResponse(b'{"response":"def broken(","done":true}',
                                         200, text="def broken(")
                output = io.StringIO()
                with patch("sys.argv", ["artsy-harness", "--prompt", "circle", *execute, "--episodes", root]), \
                     patch("artsy_harness.models.Ollama.generate", return_value=response), \
                     patch("subprocess.Popen") as process, \
                     redirect_stdout(output), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(), 1)
                process.assert_not_called()
                record = json.loads(output.getvalue())
                self.assertEqual(record["status"], "syntax_failed")
                directory = Path(record["episode"])
                self.assertEqual((directory / "program.py").read_text(), response.text)
                self.assertEqual((directory / "response.body").read_bytes(), response.raw)


if __name__ == "__main__":
    unittest.main()
