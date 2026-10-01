import io
import json
import os
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
            with patch("artsy_harness.harness.subprocess.run") as inspect, \
                 patch("artsy_harness.harness._render", return_value=(
                     {"returncode": 0, "stdout": "", "stderr": "", "timed_out": False},
                     buffer.getvalue())):
                inspect.return_value.stdout = "sha256:test\n"
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
            with patch("artsy_harness.harness.subprocess.run") as inspect, \
                 patch("artsy_harness.harness._render", return_value=(
                     {"returncode": 1, "stderr": "RuntimeError: fixture failure"}, None)):
                inspect.return_value.stdout = "sha256:test\n"
                directory, record = run_episode("raise RuntimeError('fixture failure')", root)
            self.assertEqual(record["status"], "execution_failed")
            self.assertTrue((directory / "record.json").exists())
            self.assertFalse((directory / "image.png").exists())

    def test_missing_docker_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.harness.subprocess.run", side_effect=FileNotFoundError("docker")):
                directory, record = run_episode("print('never run on host')", root)
            self.assertEqual(record["status"], "execution_failed")
            self.assertIn("docker", record["error"])
            self.assertTrue((directory / "record.json").exists())

    def test_invalid_png_is_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("artsy_harness.harness.subprocess.run") as inspect, \
                 patch("artsy_harness.harness._render", return_value=({"returncode": 0}, b"not PNG")):
                inspect.return_value.stdout = "sha256:test\n"
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


@unittest.skipUnless(os.environ.get("ARTSY_DOCKER_TESTS") == "1", "Opt-in real Docker tests")
class DockerTests(unittest.TestCase):
    def test_real_fixture(self):
        fixture = files("artsy_harness").joinpath("fixtures/drawing.py").read_text()
        with tempfile.TemporaryDirectory() as root:
            directory, record = run_episode(fixture, root)
            self.assertEqual(record["status"], "success", record)
            self.assertEqual(record["image"]["width"], 256)
            self.assertEqual(
                record["image"]["sha256"],
                "61279ba2d13404232b678b999ad6a3bb2ac136c6ae4b197b97b31be742a7ac73",
            )
            self.assertTrue((directory / "image.png").exists())
            command = record["renderer"]["command"]
            limits = record["limits"]
            for option, key in (("memory", "memory_bytes"), ("memory-swap", "memory_bytes"),
                                ("cpus", "cpus"), ("pids-limit", "pids")):
                self.assertIn(f"--{option}={limits[key]}", command)
            for option, key in (("cpu", "cpu_seconds"), ("fsize", "file_bytes")):
                self.assertIn(f"{option}={limits[key]}:{limits[key]}", command)

    def test_real_failure(self):
        with tempfile.TemporaryDirectory() as root:
            _, record = run_episode("raise RuntimeError('fixture failure')", root)
            self.assertEqual(record["status"], "execution_failed", record)
            self.assertEqual(record["renderer"]["returncode"], 1, record)
            self.assertIn("fixture failure", record["renderer"]["stderr"])

    def test_readonly_and_network_restrictions(self):
        program = """import socket
try:
    open('/renderer/escape', 'w').write('bad')
except OSError:
    pass
else:
    raise RuntimeError('root filesystem was writable')
try:
    socket.create_connection(('1.1.1.1', 443), timeout=1)
except OSError:
    pass
else:
    raise RuntimeError('external networking available')
""" + files("artsy_harness").joinpath("fixtures/drawing.py").read_text()
        with tempfile.TemporaryDirectory() as root:
            _, record = run_episode(program, root)
            self.assertEqual(record["status"], "success", record)

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as root:
            _, record = run_episode("while True: pass", root, timeout=2)
            self.assertEqual(record["status"], "execution_failed", record)
            self.assertTrue(record["renderer"]["timed_out"], record)


if __name__ == "__main__":
    unittest.main()
