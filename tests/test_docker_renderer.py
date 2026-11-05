"""Optional local Docker boundary checks; no hosted model calls.

Build evaluation/Dockerfile as artsy-rl-runtime:latest, then set
ARTSY_RENDERER_DOCKER_TEST=1 to run this suite.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from drawing.renderer import DockerRenderer


@unittest.skipUnless(os.getenv("ARTSY_RENDERER_DOCKER_TEST") == "1", "opt-in local Docker test")
class DockerBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_render_isolated_from_host_files_environment_and_network(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "private.txt"
            sentinel.write_text("host-only")
            source = (
                "import json, os, socket, cairo\n"
                "s = socket.socket(); s.settimeout(1)\n"
                "try:\n s.connect(('203.0.113.1', 80)); network = True\n"
                "except OSError:\n network = False\n"
                "finally:\n s.close()\n"
                f"print(json.dumps(dict(host_file=os.path.exists({str(sentinel)!r}), "
                "host_secret=os.getenv('ARTSY_TEST_HOST_SECRET'), network=network)))\n"
                "surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1920, 1200)\n"
                "c = cairo.Context(surface)\nc.arc(960,600,240,0,6.283185307179586)\nc.fill()\n"
                "surface.write_to_png('output.png')\n"
            )
            with patch.dict(os.environ, {"ARTSY_TEST_HOST_SECRET": "host-only"}):
                result = await DockerRenderer(seconds=5).render(source, 42)
            self.assertTrue(result["ok"], result)
            self.assertEqual(json.loads(result["stdout"]),
                             {"host_file": False, "host_secret": None, "network": False})
            self.assertEqual((result["width"], result["height"]), (1920, 1200))
            self.assertEqual(sentinel.read_text(), "host-only")

    async def test_bad_program_and_timeout_are_bounded(self):
        renderer = DockerRenderer(seconds=1)
        invalid = await renderer.render("raise RuntimeError('broken')", 42)
        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["failureKind"], "invalid_code")
        timed_out = await renderer.render("while True: pass", 42)
        self.assertFalse(timed_out["ok"])
        self.assertIn("timed out", timed_out["error"])
