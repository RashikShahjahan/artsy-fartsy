"""Pixel-level regression tests for ArtCanvas detailed drawing operations."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from artcanvas import ArtCanvas


class DetailAPITests(unittest.TestCase):
    def test_ellipse_clip_fill_preserve_and_stroke_styles(self):
        with TemporaryDirectory() as tmp:
            dest = str(Path(tmp) / "drawing.png")
            with ArtCanvas(180, 120, dest) as c:
                c.set_color(1, 0, 0)
                c.draw_ellipse(45, 55, 30, 14, fill=True)
                c.new_path()
                c.draw_ellipse(125, 50, 40, 30, fill=False)

                c.new_path()
                c.ellipse_path(125, 50, 25, 20)
                c.clip()
                c.set_color(0, 0, 1)
                c.draw_rectangle(100, 0, 75, 110, fill=True)
                c.reset_clip()

                c.new_path()
                c.move_to(10, 100)
                c.line_to(55, 100)
                c.set_line_cap("round")
                c.set_line_join("round")
                c.set_line_width(8)
                c.stroke_preserve()
                self.assertTrue(len(c.create_path()) > 0)
                c.new_path()
                c.draw_rectangle(70, 90, 15, 15, fill=False)
                c.fill_preserve()
                c.stroke()
            with Image.open(dest) as image:
                self.assertEqual(image.getpixel((45, 55))[:3], (255, 0, 0))
                self.assertEqual(image.getpixel((85, 55))[:3], (255, 255, 255))
                self.assertEqual(image.getpixel((125, 50))[:3], (0, 0, 255))
                self.assertEqual(image.getpixel((170, 50))[:3], (255, 255, 255))

    def test_clipped_patterns_restore_color_and_are_deterministic(self):
        def render(path):
            with ArtCanvas(128, 96, path) as c:
                c.set_color(1, 0, 0)
                c.new_path()
                c.move_to(20, 15)
                c.line_to(112, 15)
                c.line_to(112, 75)
                c.line_to(20, 75)
                c.close_path()
                c.clip()
                c.fill_pattern("crosshatch", spacing=7, angle=30,
                               line_width=1, opacity=0.7, color=(0, 0, 0))
                c.fill_pattern("stipple", spacing=7, line_width=2, seed=42)
                c.fill_pattern("noise", spacing=8, line_width=1.3, seed=1)
                c.set_pattern("dots", spacing=6, color=(0, 0, 1))
                c.draw_rectangle(70, 25, 40, 40, fill=True)
                c.reset_clip()
                c.draw_circle(8, 8, 5, fill=True)
        with TemporaryDirectory() as tmp:
            a, b = str(Path(tmp) / "a.png"), str(Path(tmp) / "b.png")
            render(a)
            render(b)
            self.assertEqual(hashlib.sha256(Path(a).read_bytes()).digest(),
                             hashlib.sha256(Path(b).read_bytes()).digest())
            with Image.open(a) as image:
                self.assertEqual(image.getpixel((127, 95))[:3], (255, 255, 255))
                self.assertNotEqual(image.getpixel((70, 45))[:3], (255, 255, 255))

    def test_reusable_path_and_combined_fill_stroke(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "repeat.png")
            with ArtCanvas(150, 80, path) as c:
                c.new_path()
                c.move_to(0, 0)
                c.line_to(20, 0)
                c.line_to(10, 20)
                c.close_path()
                shape = c.create_path()
                c.set_color(1, 0, 0)
                c.set_line_width(2)
                for dx in (20, 80):
                    c.save_state()
                    c.translate(dx, 20)
                    c.draw_path(shape, fill=True, stroke=True)
                    c.restore_state()
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((30, 25))[:3], (255, 0, 0))
                self.assertEqual(image.getpixel((90, 25))[:3], (255, 0, 0))
                self.assertEqual(image.getpixel((55, 25))[:3], (255, 255, 255))

    def test_nested_layers_blend_modes_and_blur(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "layers.png")
            with ArtCanvas(120, 100, path) as c:
                c.set_color(0, 0, 1)
                c.draw_rectangle(0, 0, 120, 100, fill=True)
                c.begin_layer(0.5, "screen")
                c.set_color(1, 0, 0)
                c.draw_rectangle(15, 15, 70, 50, fill=True)
                c.begin_layer(1, "over")
                c.set_color(1, 1, 1)
                c.draw_circle(45, 40, 6, fill=True)
                c.end_layer()
                c.end_layer()
                with c.layer(opacity=0.8, blend_mode="multiply", blur_radius=4):
                    c.set_color(0, 0, 0)
                    c.draw_rectangle(93, 40, 10, 10, fill=True)
            with Image.open(path) as image:
                self.assertEqual(image.size, (120, 100))
                self.assertEqual(image.getpixel((0, 0))[:3], (0, 0, 255))
                self.assertNotEqual(image.getpixel((12, 15))[:3], (0, 0, 255))
                self.assertNotEqual(image.getpixel((99, 45))[:3], (0, 0, 255))

    def test_error_validation(self):
        with TemporaryDirectory() as tmp:
            with ArtCanvas(50, 50, str(Path(tmp) / "error.png")) as c:
                with self.assertRaises(ValueError):
                    c.draw_ellipse(10, 10, -2, 5)
                with self.assertRaises(ValueError):
                    c.set_line_cap("fuzzy")
                with self.assertRaises(ValueError):
                    c.set_line_join("sharp")
                with self.assertRaises(ValueError):
                    c.set_pattern("plaid")
                with self.assertRaises(ValueError):
                    c.set_pattern("dots", spacing=0)
                with self.assertRaises(ValueError):
                    c.begin_layer(2)
                with self.assertRaises(ValueError):
                    c.begin_layer(blend_mode="unknown")
                with self.assertRaises(RuntimeError):
                    c.end_layer()
                c.begin_layer()
                with self.assertRaises(ValueError):
                    c.end_layer(-1)
                c.end_layer()


if __name__ == "__main__":
    unittest.main()
