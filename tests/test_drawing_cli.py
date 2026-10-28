from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from drawing.renderer import validate_png

ROOT = Path(__file__).resolve().parents[1]
CODE = (
    "from artcanvas import ArtCanvas\n"
    "with ArtCanvas(1920, 1200) as canvas:\n"
    "    canvas.set_color(0.1, 0.3, 0.8)\n"
    "    canvas.draw_circle(960, 600, 240, True)\n"
)


class FakeTeacher:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers.get("Content-Length", "0"))
                owner.calls.append(json.loads(self.rfile.read(size)))
                payload = json.dumps({
                    "model": "fake",
                    "choices": [{"message": {"content": json.dumps({"code": CODE})}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                return

        return Handler

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class DrawingCliTests(unittest.TestCase):
    def clean_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in list(env):
            if key == "TEACHER_API_KEY" or key.startswith("ARTSY_"):
                env.pop(key)
        return env

    def run_cli(self, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        env = self.clean_env()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
        # Keep local .env judge endpoints/credentials out of fake-teacher tests.
        with tempfile.TemporaryDirectory(prefix="artsy-cli-cwd-") as cwd:
            return subprocess.run(
                [sys.executable, "-m", "drawing.generate_dataset", *args],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )

    def test_generation_and_resume_use_local_teacher_and_render_submission(self):
        with tempfile.TemporaryDirectory(prefix="artsy-drawing-cli-") as directory, FakeTeacher() as teacher:
            output = Path(directory) / "dataset"
            prompts = Path(directory) / "prompts.jsonl"
            prompts.write_text(json.dumps({"id": "circle", "family": "circles", "prompt": "Draw a blue circle", "seed": 42}) + "\n")
            args = (
                "--output", str(output), "--input", str(prompts), "--candidates", "1",
                "--teacher-model", "fake", "--teacher-url", teacher.url,
                "--repairs", "0",
            )
            first = self.run_cli(*args)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            summary = json.loads(first.stdout)
            self.assertEqual(summary["review"], 1)
            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["requests"], 1)
            self.assertEqual(len(teacher.calls), 1)

            rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            record = rows[0]
            self.assertEqual(record["status"], "review")
            source_path = output / record["artifacts"]["source"]
            image_path = output / record["artifacts"]["image"]
            self.assertEqual(source_path.read_text(), CODE)
            png = image_path.read_bytes()
            self.assertEqual(validate_png(png, {"width": 1920, "height": 1200}),
                             {"width": 1920, "height": 1200})
            self.assertNotIn("draftSources", json.loads((output / "manifest.json").read_text()))

            second = self.run_cli(*args)
            self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
            resumed = json.loads(second.stdout)
            self.assertEqual(resumed["review"], 1)
            self.assertEqual(resumed["requests"], 1)
            self.assertEqual(len(teacher.calls), 1, "completed candidate should resume without another teacher call")

    def test_help_and_input_dry_run(self):
        help_result = subprocess.run(
            [sys.executable, "-m", "drawing.generate_dataset", "--help"],
            cwd=ROOT, text=True, capture_output=True, timeout=10, check=False,
            env=self.clean_env(),
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("Python dataset generation", help_result.stdout)

        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.jsonl"
            prompts.write_text(json.dumps({"family": "circles", "prompt": "Draw a circle", "seed": 42}) + "\n")
            result = self.run_cli("--dry-run", "--input", str(prompts))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 1)
            self.assertNotIn("split", json.loads(result.stdout))

        unknown = self.run_cli("--not-a-real-option")
        self.assertNotEqual(unknown.returncode, 0)
        invalid_curriculum = self.run_cli("--dry-run", "--curriculum", "unknown")
        self.assertNotEqual(invalid_curriculum.returncode, 0)
        self.assertIn("unrecognized arguments", invalid_curriculum.stderr)


if __name__ == "__main__":
    unittest.main()
