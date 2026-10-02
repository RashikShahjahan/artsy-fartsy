import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from artsy_harness.cli import main
from artsy_harness.harness import generate_only_episode
from artsy_harness.models import Ollama, PROMPT_TEMPLATE


class FakeResponse(io.BytesIO):
    status = 200


class ModelTests(unittest.TestCase):
    def run_response(self, body):
        model = Ollama()
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.models.urlopen", return_value=FakeResponse(body)) as call, \
                 patch("artsy_harness.harness._render", side_effect=AssertionError("execution")) as renderer, \
                 patch("subprocess.Popen", side_effect=AssertionError("subprocess")) as process:
                directory, record = generate_only_episode(root, model=model, prompt="A blue circle")
            renderer.assert_not_called()
            process.assert_not_called()
            self.assertEqual(record["execution"], {"backend": "none", "executed": False, "enabled": False})
            self.assertIsNone(record["renderer"])
            self.assertIsNone(record["image"])
            self.assertFalse((directory / "image.png").exists())
            self.assertEqual((directory / "prompt.txt").read_text(), PROMPT_TEMPLATE + "A blue circle")
            self.assertEqual((directory / "response.body").read_bytes(), body)
            self.assertEqual(json.loads((directory / "record.json").read_bytes()), record)
            if record["program"]:
                expected = json.loads(body)["response"].encode("utf-8")
                self.assertEqual((directory / "program.py").read_bytes(), expected)
                self.assertEqual((directory / "generated.txt").read_bytes(), expected)
            if "thinking" in record["generation"]:
                self.assertEqual((directory / "thinking.txt").read_text(), json.loads(body)["thinking"])
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
        program = "# Verbatim Unicode: café\r\npass\r\n"
        body = json.dumps({"response": program, "done": True, "model": "returned-model",
                           "digest": "sha256:model"}, indent=2).encode() + b"\n"
        record, called = self.run_response(body)
        self.assertFalse(called)
        self.assertEqual(record["status"], "generated")
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
        record, called = self.run_response(body)
        self.assertFalse(called)
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
                directory, record = generate_only_episode(root, model=Ollama(), prompt="circle")
            self.assertEqual(record["status"], "protocol_failed")
            self.assertEqual(record["generation"]["http_status"], 404)
            self.assertEqual((directory / "response.body").read_bytes(), b'{"error":"missing model"}')

    def test_connection_failure_is_append_only(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.models.urlopen", side_effect=URLError("offline")):
                first, record = generate_only_episode(root, model=Ollama(), prompt="circle")
                original = (first / "record.json").read_bytes()
                second, _ = generate_only_episode(root, model=Ollama(), prompt="circle")
            self.assertEqual(record["status"], "model_call_failed")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "record.json").read_bytes(), original)

    def test_valid_program_that_would_raise_is_never_executed(self):
        body = b'{"response": "raise RuntimeError()", "done": true}'
        record, called = self.run_response(body)
        self.assertFalse(called)
        self.assertEqual(record["status"], "generated")

    def test_incomplete_response_retains_proposed_code_and_thinking(self):
        body = b'{"response": "def unfinished(", "thinking": "partial reasoning", "done": false}'
        record, called = self.run_response(body)
        self.assertFalse(called)
        self.assertEqual(record["status"], "protocol_failed")
        self.assertEqual(record["program"], "program.py")
        self.assertEqual(record["generation"]["thinking"], "thinking.txt")

    def test_cli_passes_explicit_settings(self):
        args = ["artsy-harness", "--prompt", "circle", "--ollama-url", "http://localhost:1234",
                "--seed", "42", "--temperature", "0.4", "--top-p", "0.8", "--num-predict", "512",
                "--num-ctx", "4096", "--model-timeout", "240"]
        with patch("sys.argv", args), redirect_stdout(io.StringIO()), \
             patch("artsy_harness.cli.run_episode", return_value=("episode", {"status": "generated"})) as run:
            self.assertEqual(main(), 0)
        model = run.call_args.kwargs["model"]
        self.assertIs(run.call_args.kwargs["execute"], False)
        self.assertEqual(model.endpoint, "http://localhost:1234/api/generate")
        self.assertEqual(model.model, "qwen3.5:9b-q4_K_M")
        self.assertIs(model.think, True)
        self.assertEqual(model.timeout, 240)
        self.assertEqual(model.options, {"seed": 42, "temperature": 0.4, "top_p": 0.8,
                                         "num_predict": 512, "num_ctx": 4096})

    def test_cli_can_override_model_and_thinking(self):
        args = ["artsy-harness", "--prompt", "circle", "--model", "other-model", "--no-think"]
        with patch("sys.argv", args), redirect_stdout(io.StringIO()), \
             patch("artsy_harness.cli.run_episode", return_value=("episode", {"status": "generated"})) as run:
            self.assertEqual(main(), 0)
        model = run.call_args.kwargs["model"]
        self.assertEqual(model.request("circle")["model"], "other-model")
        self.assertIs(model.request("circle")["think"], False)
        self.assertEqual(model.options["num_predict"], 8192)
        self.assertEqual(model.options["num_ctx"], 16384)
        self.assertEqual(model.timeout, 600)


@unittest.skipUnless(os.environ.get("ARTSY_OLLAMA_TESTS") == "1", "Opt-in Ollama smoke test")
class LiveModelTests(unittest.TestCase):
    def test_live_generated_episode(self):
        directory, record = generate_only_episode(
            "episodes", model=Ollama(base_url=os.environ.get("ARTSY_OLLAMA_URL", "http://localhost:11434")),
            prompt="Draw a filled blue circle centered on a white background.",
        )
        self.assertEqual(record["status"], "generated", (str(directory), record))
        self.assertTrue((directory / "program.py").exists())
        self.assertFalse((directory / "image.png").exists())


if __name__ == "__main__":
    unittest.main()
