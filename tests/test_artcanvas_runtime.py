"""Smoke-test the restored ArtCanvas API through the actual local renderer."""
import unittest

from drawing.renderer import LocalRenderer
from drawing.render_contract import RUNTIME_VERSION


class ArtCanvasRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_artcanvas_program_renders_valid_png(self):
        code = """from artcanvas import ArtCanvas

with ArtCanvas(1920, 1200, "output.png") as canvas:
    canvas.fill_background(1, 1, 1)
    canvas.set_gradient(0, 0, 1920, 1200,
                        [(0, 0.1, 0.2, 0.7, 1), (1, 0.8, 0.9, 1, 1)])
    canvas.draw_rectangle(0, 0, 1920, 1200, fill=True)
    canvas.set_color(1, 0, 0)
    canvas.draw_circle(960, 600, 150, fill=True)
"""
        result = await LocalRenderer().render(code, 42)
        self.assertTrue(result["ok"], result.get("stderr", result))
        self.assertEqual(result["runtimeVersion"], RUNTIME_VERSION)
        self.assertEqual((result["width"], result["height"]), (1920, 1200))
        self.assertTrue(result["png"].startswith(b"\\x89PNG\\r\\n\\x1a\\n"))

    def test_reference_matches_implementation(self):
        from drawing.documentation import DOCUMENTATION_LOCATION, api_documentation
        import json
        from pathlib import Path
        reference = Path(__file__).resolve().parents[1] / "docs" / "artcanvas.md"
        self.assertIn("set_radial_gradient", reference.read_text())
        self.assertIn("draw_bezier_curve", DOCUMENTATION_LOCATION)
        self.assertEqual(json.loads(api_documentation())["api"], "artcanvas-v2")
