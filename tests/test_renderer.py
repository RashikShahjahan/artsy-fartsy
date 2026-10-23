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

    async def test_artcanvas_runtime_is_seeded_and_recovers_after_bad_output(self):
        source = (Path(__file__).resolve().parents[1] / "examples" / "transforms.py").read_text()
        renderer = LocalRenderer()
        first = await renderer.render(source, 42)
        second = await renderer.render(source, 42)
        other_seed = await renderer.render(source, 43)
        self.assertTrue(first["ok"], first)
        self.assertEqual(first["png"], second["png"])
        self.assertNotEqual(first["png"], other_seed["png"])
        self.assertFalse((await renderer.render("while True: pass", 0))["ok"])
        self.assertFalse((await renderer.render(
            "open('output.png','wb').write(b'x' * (8*1024*1024 + 1))", 0))["ok"])
        self.assertFalse((await renderer.render("from artcanvas import ArtCanvas\nArtCanvas(5000,5000)", 0))["ok"])
        good = "from artcanvas import ArtCanvas\nwith ArtCanvas(32,32) as c:\n c.draw_circle(16,16,8,True)"
        self.assertTrue((await renderer.render(good, 0))["ok"])

    async def test_renderer_rejects_symlinks_and_handles_repeated_saves(self):
        code = (
            "import os\n"
            "os.symlink('/etc/passwd', 'output.png')\n"
        )
        symlink = await LocalRenderer().render(code, 0)
        self.assertFalse(symlink["ok"])
        repeated_saves = (
            "from artcanvas import ArtCanvas\n"
            "with ArtCanvas(32,32) as c:\n c.save()\n c.save()\n"
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
