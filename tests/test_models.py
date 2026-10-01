import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.resources import files
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from PIL import Image

from artsy_harness.cli import main
from artsy_harness.harness import run_episode
from artsy_harness.models import Ollama


class FakeResponse(io.BytesIO):
    status = 200


class ModelTests(unittest.TestCase):
    def run_response(self, body, *, render=None):
        model = Ollama()
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.models.urlopen", return_value=FakeResponse(body)) as call, \
                 patch("artsy_harness.harness.subprocess.run") as inspect, \
                 patch("artsy_harness.harness._render", return_value=render) as renderer:
                inspect.return_value.stdout = "sha256:test\n"
                directory, record = run_episode(None, root, model=model, prompt="A blue circle")
            self.assertEqual((directory / "response.body").read_bytes(), body)
            self.assertEqual(json.loads((directory / "record.json").read_bytes()), record)
            if record["program"]:
                expected = json.loads(body)["response"].encode("utf-8")
                self.assertEqual((directory / "program.py").read_bytes(), expected)
                self.assertEqual((directory / "generated.txt").read_bytes(), expected)
            payload = json.loads(call.call_args.args[0].data)
            self.assertEqual(payload, record["generation"]["request"])
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["model"], "qwen3.5:9b-q4_K_M")
            self.assertIs(payload["think"], True)
            self.assertEqual(payload["options"]["seed"], 0)
            self.assertEqual(payload["options"]["num_predict"], 8192)
            self.assertEqual(payload["options"]["num_ctx"], 16384)
            self.assertEqual(call.call_args.kwargs["timeout"], 600)
            self.assertEqual(record["generation"]["timeout_seconds"], 600)
            return record, renderer.called

    def test_success_preserves_exact_evidence(self):
        program = files("artsy_harness").joinpath("fixtures/drawing.py").read_text()
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256)).save(buffer, format="PNG")
        body = json.dumps({"response": program, "done": True, "model": "returned-model",
                           "digest": "sha256:model"}, indent=2).encode() + b"\n"
        record, called = self.run_response(body, render=({"returncode": 0}, buffer.getvalue()))
        self.assertTrue(called)
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["generation"]["returned_digest"], "sha256:model")

    def test_syntax_failures_do_not_render(self):
        for text in ("", "```python\nprint('hi')\n```", "Here is your drawing:", "def broken("):
            with self.subTest(text=text):
                record, called = self.run_response(json.dumps({"response": text, "done": True}).encode())
                self.assertFalse(called)
                self.assertEqual(record["status"], "syntax_failed")

    def test_separate_thinking_is_not_program_text(self):
        body = json.dumps({"response": "pass\n", "thinking": "This is not Python.",
                           "done": True}).encode()
        record, called = self.run_response(body, render=({"returncode": 1}, None))
        self.assertTrue(called)
        self.assertEqual(record["generation"]["syntax"], "valid")

    def test_protocol_failures_do_not_render(self):
        for body in (b"not json", b"[]", b'{"response": 5, "done": true}',
                     b'{"response": "pass", "done": false}', b'{"error": "missing model"}'):
            with self.subTest(body=body):
                record, called = self.run_response(body)
                self.assertFalse(called)
                self.assertEqual(record["status"], "protocol_failed")

    def test_http_failure_preserves_body(self):
        with tempfile.TemporaryDirectory() as root:
            error = HTTPError("http://localhost:11434/api/generate", 404, "missing model", {},
                              io.BytesIO(b'{"error":"missing model"}'))
            with patch("artsy_harness.models.urlopen", side_effect=error):
                directory, record = run_episode(None, root, model=Ollama(), prompt="circle")
            self.assertEqual(record["status"], "protocol_failed")
            self.assertEqual(record["generation"]["http_status"], 404)
            self.assertEqual((directory / "response.body").read_bytes(), b'{"error":"missing model"}')

    def test_connection_failure_is_append_only(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.models.urlopen", side_effect=URLError("offline")):
                first, record = run_episode(None, root, model=Ollama(), prompt="circle")
                original = (first / "record.json").read_bytes()
                second, _ = run_episode(None, root, model=Ollama(), prompt="circle")
            self.assertEqual(record["status"], "model_call_failed")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "record.json").read_bytes(), original)

    def test_generated_execution_and_validation_failures(self):
        body = b'{"response": "raise RuntimeError()", "done": true}'
        record, _ = self.run_response(body, render=({"returncode": 1}, None))
        self.assertEqual(record["status"], "execution_failed")
        record, _ = self.run_response(body, render=({"returncode": 0}, b"invalid PNG"))
        self.assertEqual(record["status"], "validation_failed")
        record, _ = self.run_response(body, render=({"returncode": 0, "output_error": "missing PNG"}, None))
        self.assertEqual(record["status"], "validation_failed")
        self.assertEqual(record["renderer"]["returncode"], 0)

    def test_cli_passes_explicit_settings(self):
        args = ["artsy-harness", "--prompt", "circle", "--ollama-url", "http://localhost:1234",
                "--seed", "42", "--temperature", "0.4", "--top-p", "0.8", "--num-predict", "512",
                "--num-ctx", "4096", "--model-timeout", "240"]
        with patch("sys.argv", args), redirect_stdout(io.StringIO()), \
             patch("artsy_harness.cli.run_episode", return_value=("episode", {"status": "success"})) as run:
            self.assertEqual(main(), 0)
        model = run.call_args.kwargs["model"]
        self.assertEqual(model.endpoint, "http://localhost:1234/api/generate")
        self.assertEqual(model.model, "qwen3.5:9b-q4_K_M")
        self.assertIs(model.think, True)
        self.assertEqual(model.timeout, 240)
        self.assertEqual(model.options, {"seed": 42, "temperature": 0.4, "top_p": 0.8,
                                         "num_predict": 512, "num_ctx": 4096})

    def test_cli_can_override_model_and_thinking(self):
        args = ["artsy-harness", "--prompt", "circle", "--model", "other-model", "--no-think"]
        with patch("sys.argv", args), redirect_stdout(io.StringIO()), \
             patch("artsy_harness.cli.run_episode", return_value=("episode", {"status": "success"})) as run:
            self.assertEqual(main(), 0)
        model = run.call_args.kwargs["model"]
        self.assertEqual(model.request("circle")["model"], "other-model")
        self.assertIs(model.request("circle")["think"], False)
        self.assertEqual(model.options["num_predict"], 8192)
        self.assertEqual(model.options["num_ctx"], 16384)
        self.assertEqual(model.timeout, 600)


@unittest.skipUnless(os.environ.get("ARTSY_OLLAMA_TESTS") == "1", "Opt-in Ollama + Docker smoke test")
class LiveModelTests(unittest.TestCase):
    def test_live_generated_episode(self):
        directory, record = run_episode(
            None, "episodes", model=Ollama(base_url=os.environ.get("ARTSY_OLLAMA_URL", "http://localhost:11434")),
            prompt="Draw a filled blue circle centered on a white background.",
        )
        self.assertEqual(record["status"], "success", (str(directory), record))
        self.assertTrue((directory / "image.png").exists())


if __name__ == "__main__":
    unittest.main()
