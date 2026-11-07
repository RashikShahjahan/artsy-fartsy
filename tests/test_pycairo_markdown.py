"""Tests for pycairo docs generated from the installed stub."""
from importlib.metadata import distribution
from pathlib import Path
import tempfile
import unittest

from scripts.generate_pycairo_markdown import generate


class MarkdownReferenceTests(unittest.TestCase):
    def test_complete_signatures_and_overloads(self):
        source = '''from typing import overload

class Context:
    """A drawing context."""
    MODE: int = ...
    """Active mode."""

    @overload
    def arc(self, x: float, y: float) -> None:
        """:param x: horizontal
        :param y: vertical
        """

    @overload
    def arc(self, x: float, y: float, radius: float) -> None:
        """:param radius: radius
        """

def cairo_version() -> int:
    """:returns: library version"""
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "stub.pyi"
            stub.write_text(source)
            output = root / "docs"
            counts = generate(stub, output)
            self.assertEqual(counts, {"classes": 1, "methods": 2})
            arc = (output / "Context/arc.md").read_text()
            self.assertEqual(arc.count("def arc("), 2)
            self.assertIn("@overload", arc)
            self.assertIn("radius: float", arc)
            self.assertIn("**x**: horizontal", arc)
            self.assertIn("**radius**: radius", arc)
            self.assertIn("Active mode", (output / "Context/README.md").read_text())
            self.assertIn("**Returns:** library version",
                          (output / "module/cairo_version.md").read_text())
            (output / "Context/stale.md").write_text("old")
            generate(stub, output)
            self.assertFalse((output / "Context/stale.md").exists())

    def test_installed_reference_is_complete(self):
        stub = distribution("pycairo").locate_file("cairo/__init__.pyi")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "docs"
            counts = generate(stub, output)
            self.assertGreaterEqual(counts["classes"], 40)
            self.assertGreaterEqual(counts["methods"], 250)
            arc = (output / "Context/arc.md").read_text()
            self.assertIn("radius: float", arc)
            self.assertIn("**angle1**:", arc)
            self.assertIn("radians", arc)
            self.assertLess(len(arc.encode()), 16384)
            self.assertIn("Context/README.md", (output / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
