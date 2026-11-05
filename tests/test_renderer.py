import base64
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from drawing.renderer import LocalRenderer, RUNTIME_VERSION, validate_png


def make_png(width=32, height=32):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    pixels = b"".join(b"\0" + b"\xff\x00\x00\xff" * width for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


class RendererTests(unittest.IsolatedAsyncioTestCase):
    async def test_validate_png_checks_dimensions_and_checksums(self):
        png = make_png()
        self.assertEqual(validate_png(png), {"width": 32, "height": 32})
        self.assertEqual(validate_png(png, {"width": 32, "height": 32})["height"], 32)
        with self.assertRaisesRegex(ValueError, "Expected 31"):
            validate_png(png, {"width": 31, "height": 32})
        damaged = bytearray(png)
        damaged[png.index(b"IDAT") + 4] ^= 1
        with self.assertRaisesRegex(ValueError, "Invalid PNG"):
            validate_png(bytes(damaged))
        with self.assertRaisesRegex(ValueError, "Invalid PNG"):
            validate_png(png + b"extra")
        with self.assertRaisesRegex(ValueError, "Invalid PNG"):
            validate_png(png[:-1])
        with self.assertRaisesRegex(ValueError, "Invalid or oversized"):
            validate_png(b"not png")

    async def test_local_renderer_runs_code_bounds_logs_and_cleans_workspace(self):
        png64 = base64.b64encode(make_png()).decode("ascii")
        code = (
            "import base64, os\n"
            "print(os.getcwd())\n"
            "print('x' * 20000)\n"
            f"open('output.png', 'wb').write(base64.b64decode('{png64}'))\n"
        )
        with tempfile.TemporaryDirectory() as root:
            result = await LocalRenderer().render(code, 5)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["runtimeVersion"], RUNTIME_VERSION)
            self.assertEqual((result["width"], result["height"]), (32, 32))
            self.assertLessEqual(len(result["stdout"].encode()), 8192)
            self.assertFalse(Path(result["stdout"].splitlines()[0]).exists())
            self.assertEqual(list(Path(root).iterdir()), [])

    async def test_local_renderer_rejects_invalid_output_and_times_out(self):
        with tempfile.TemporaryDirectory() as root:
            invalid = await LocalRenderer().render("open('output.png','wb').write(b'x')", 0)
            self.assertFalse(invalid["ok"])
            self.assertEqual(invalid["failureKind"], "invalid_code")
            timeout = await LocalRenderer(seconds=1).render("while True: pass", 0)
            self.assertFalse(timeout["ok"])
            self.assertEqual(timeout["error"], "Rendering timed out")
            invalid_seed = await LocalRenderer().render("pass", -1)
            self.assertEqual(invalid_seed["failureKind"], "invalid_code")

    async def test_pycairo_runtime_is_seeded_and_recovers_after_bad_output(self):
        source = (Path(__file__).resolve().parents[1] / "examples" / "transforms.py").read_text()
        source = "import cairo\nassert hasattr(cairo, 'CAPI')\n" + source
        renderer = LocalRenderer(seconds=1)
        first = await renderer.render(source, 42)
        second = await renderer.render(source, 42)
        other_seed = await renderer.render(source, 43)
        self.assertTrue(first["ok"], first)
        self.assertEqual(first["png"], second["png"])
        self.assertNotEqual(first["png"], other_seed["png"])
        self.assertFalse((await renderer.render("while True: pass", 0))["ok"])
        self.assertFalse((await renderer.render(
            "open('output.png','wb').write(b'x' * (8*1024*1024 + 1))", 0))["ok"])
        self.assertFalse((await renderer.render('import cairo\ncairo.ImageSurface(cairo.FORMAT_RGB24,5000,5000).write_to_png("output.png")', 0))["ok"])
        good = 'import cairo\nsurface = cairo.ImageSurface(cairo.FORMAT_RGB24, 32, 32)\nc = cairo.Context(surface)\nc.arc(16,16,8,0,6.283185307179586)\nc.fill()\nsurface.write_to_png("output.png")'
        self.assertTrue((await renderer.render(good, 0))["ok"])

    async def test_renderer_rejects_symlinks_and_handles_repeated_saves(self):
        code = (
            "import os\n"
            "os.symlink('/etc/passwd', 'output.png')\n"
        )
        symlink = await LocalRenderer().render(code, 0)
        self.assertFalse(symlink["ok"])
        repeated_saves = (
            'import cairo\nsurface = cairo.ImageSurface(cairo.FORMAT_RGB24, 32, 32)\nc = cairo.Context(surface)\nsurface.write_to_png("output.png")\nsurface.write_to_png("output.png")\nsurface.write_to_png("output.png")\n'
        )
        result = await LocalRenderer().render(repeated_saves, 0)
        self.assertTrue(result["ok"], result)

    async def test_missing_interpreter_is_an_infrastructure_error(self):
        with tempfile.TemporaryDirectory() as root:
            result = await LocalRenderer("/nonexistent-artsy-python").render("pass", 0)
            self.assertFalse(result["ok"])
            self.assertEqual(result["failureKind"], "infrastructure")


if __name__ == "__main__":
    unittest.main()

class DockerRendererTests(unittest.IsolatedAsyncioTestCase):
    async def test_environment_defaults_to_isolated_backend(self):
        from drawing.environment import DrawingEnvironment
        from drawing.renderer import DockerRenderer
        with tempfile.TemporaryDirectory() as root:
            self.assertIsInstance(DrawingEnvironment(root).renderer, DockerRenderer)

    async def test_docker_command_is_isolated_and_validates_result(self):
        import json
        import os
        from unittest.mock import patch
        from drawing.renderer import DockerRenderer

        png = make_png(1920, 1200)
        with tempfile.TemporaryDirectory() as root:
            executable = Path(root) / "docker"
            args_file = Path(root) / "args"
            encoded = base64.b64encode(png).decode("ascii")
            executable.write_text(
                "#!/usr/bin/env python3\nimport json, os, sys\n"
                "if sys.argv[1] == 'run':\n"
                " open(os.environ['TEST_DOCKER_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n"
                " sys.stdin.read()\n"
                " print(json.dumps({'ok': True, 'stdout': '', 'stderr': ''}))\n"
                f" print('<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>data:image/png;base64,{encoded}</MSWEA_MULTIMODAL_CONTENT>')\n"
            )
            executable.chmod(0o755)
            with patch.dict(os.environ, {"TEST_DOCKER_ARGS": str(args_file)}):
                result = await DockerRenderer(docker=str(executable)).render("pass", 42)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["png"], png)
            args = json.loads(args_file.read_text())
            for argument in ("--network", "none", "--read-only", "--cap-drop", "ALL", "--user", "65534:65534"):
                self.assertIn(argument, args)
            self.assertNotIn("--privileged", args)
            self.assertNotIn("--volume", args)
            self.assertNotIn("--env", args)

    async def test_missing_docker_is_infrastructure_failure(self):
        from drawing.renderer import DockerRenderer
        result = await DockerRenderer(docker="/nonexistent-artsy-docker").render("pass", 0)
        self.assertEqual(result["failureKind"], "infrastructure")
