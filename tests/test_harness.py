import io
import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from artsy_harness.harness import MAX_LOG_BYTES, _collect, run_episode, validate_png


class EpisodeTests(unittest.TestCase):
    def test_success_and_append_only(self):
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256), "blue").save(buffer, format="PNG")
        fixture = files("artsy_harness").joinpath("fixtures/drawing.py").read_text()
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.harness._render", return_value=(
                     {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False},
                     buffer.getvalue())):
                first, record = run_episode(fixture, root)
                original = (first / "record.json").read_bytes()
                second, _ = run_episode(fixture, root)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "record.json").read_bytes(), original)
            self.assertEqual(record["status"], "success")
            self.assertEqual(record["image"]["width"], 256)
            self.assertEqual((first / "program.py").read_text(), fixture)
            self.assertEqual(json.loads(original)["episode_id"], first.name)

    def test_execution_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.harness._render", return_value=(
                     {"returncode": 1, "stderr": "RuntimeError: fixture failure"}, None)):
                directory, record = run_episode("raise RuntimeError('fixture failure')", root)
            self.assertEqual(record["status"], "execution_failed")
            self.assertTrue((directory / "record.json").exists())
            self.assertFalse((directory / "image.png").exists())

    def test_missing_python_is_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.harness.subprocess.Popen", side_effect=FileNotFoundError("python")):
                directory, record = run_episode("print('not executed')", root)
            self.assertEqual(record["status"], "execution_failed")
            self.assertIn("python", record["error"])
            self.assertTrue((directory / "record.json").exists())

    def test_invalid_png_is_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.harness._render", return_value=({"returncode": 0}, b"not PNG")):
                _, record = run_episode("invalid output", root)
            self.assertEqual(record["status"], "validation_failed")

    def test_png_limits(self):
        with self.assertRaises(ValueError):
            validate_png(b"x" * (8 * 1024 * 1024 + 1))
        buffer = io.BytesIO()
        Image.new("RGB", (4097, 1)).save(buffer, format="PNG")
        with self.assertRaises(ValueError):
            validate_png(buffer.getvalue())

    def test_logs_are_drained_but_bounded(self):
        stream = io.BytesIO(b"x" * (MAX_LOG_BYTES * 2))
        output = bytearray()
        _collect(stream, output)
        self.assertEqual(output, b"x" * MAX_LOG_BYTES)
        self.assertTrue(stream.closed)

    def test_invalid_timeout_creates_no_episode(self):
        with tempfile.TemporaryDirectory() as root:
            for timeout in (0, -1, 31):
                with self.assertRaises(ValueError):
                    run_episode("", root, timeout=timeout)
            self.assertEqual(list(Path(root).iterdir()), [])


class LocalRenderingTests(unittest.TestCase):
    def test_real_png_output(self):
        program = "from PIL import Image\nImage.new('RGB', (32, 32), 'blue').save('output.png')\n"
        with tempfile.TemporaryDirectory() as root:
            directory, record = run_episode(program, root)
            self.assertEqual(record["status"], "success", record)
            self.assertEqual(record["image"]["width"], 32)
            self.assertTrue((directory / "image.png").exists())

    def test_real_fixture(self):
        fixture = files("artsy_harness").joinpath("fixtures/drawing.py").read_text()
        with tempfile.TemporaryDirectory() as root:
            directory, record = run_episode(fixture, root)
            self.assertEqual(record["status"], "success", record)
            self.assertEqual(record["image"]["width"], 256)
            self.assertTrue((directory / "image.png").exists())
            self.assertEqual(record["execution"]["backend"], "local_python")
            self.assertFalse(record["execution"]["sandboxed"])
            self.assertEqual(record["limits"], {"wall_seconds": 30})

    def test_real_failure(self):
        with tempfile.TemporaryDirectory() as root:
            _, record = run_episode("raise RuntimeError('fixture failure')", root)
            self.assertEqual(record["status"], "execution_failed", record)
            self.assertEqual(record["renderer"]["returncode"], 1, record)
            self.assertIn("fixture failure", record["renderer"]["stderr"])

    def test_symlink_output_is_rejected(self):
        program = "from pathlib import Path\nPath('output.png').symlink_to('missing.png')\n"
        with tempfile.TemporaryDirectory() as root:
            _, record = run_episode(program, root)
            self.assertEqual(record["status"], "validation_failed", record)

    def test_missing_output_is_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            _, record = run_episode("pass", root)
            self.assertEqual(record["status"], "validation_failed", record)

    def test_large_logs_are_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            _, record = run_episode(f"print('x' * {MAX_LOG_BYTES * 2})", root)
            self.assertEqual(len(record["renderer"]["stdout"]), MAX_LOG_BYTES)

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as root:
            _, record = run_episode("while True: pass", root, timeout=2)
            self.assertEqual(record["status"], "execution_failed", record)
            self.assertTrue(record["renderer"]["timed_out"], record)


if __name__ == "__main__":
    unittest.main()
