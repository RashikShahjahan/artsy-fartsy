import json
import struct
import tempfile
import unittest
from pathlib import Path
import zlib

from drawing.environment import DrawingEnvironment, ENVIRONMENT_CONFIG


def make_png(width=32, height=32):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    pixels = b"".join(b"\0" + b"\xff\x00\x00\xff" * width for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


PNG = make_png()
CONFIG = {**ENVIRONMENT_CONFIG, "width": 32, "height": 32}
GOOD = "from artcanvas import ArtCanvas\nwith ArtCanvas(32, 32) as c:\n c.draw_circle(16,16,8,True)\n"


class FakeRenderer:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    async def render(self, code, seed):
        self.calls.append((code, seed))
        if self.fail_first and len(self.calls) == 1:
            return {"ok": False, "stdout": "", "stderr": "SyntaxError", "failureKind": "invalid_code"}
        return {"ok": True, "stdout": "", "stderr": "", "png": PNG}


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_repairs_fresh_final_render_records_and_reset(self):
        with tempfile.TemporaryDirectory() as root:
            renderer = FakeRenderer(fail_first=True)
            env = DrawingEnvironment(root, config=CONFIG, renderer=renderer)
            first = await env.reset("circle", 42)
            await env.write_code("broken")
            self.assertFalse((await env.run_code())["ok"])
            await env.write_code(GOOD)
            self.assertTrue((await env.run_code())["ok"])
            final = await env.submit()
            self.assertEqual(final["outcome"], "submitted")
            self.assertEqual([call[0] for call in renderer.calls], ["broken", GOOD, GOOD])
            self.assertEqual([call[1] for call in renderer.calls], [42, 42, 42])
            episode = Path(first["directory"])
            self.assertEqual((episode / "program.py").read_text(), GOOD)
            submission = json.loads((episode / "submission.json").read_text())
            self.assertEqual(submission["seed"], 42)
            trajectory = [json.loads(line) for line in (episode / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual([row["operation"] for row in trajectory],
                             ["reset", "write_code", "run_code", "write_code", "run_code", "submit"])
            self.assertIn("SyntaxError", (episode / "trajectory.jsonl").read_text())
            self.assertFalse((await env.write_code("late edit"))["ok"])
            second = await env.reset("new prompt", 7)
            self.assertNotEqual(first["directory"], second["directory"])
            self.assertIn("Write code first", (await env.submit())["error"])
            self.assertEqual((await env.refuse("Unsupported request"))["outcome"], "refused")
            self.assertEqual((await env.run_code())["outcome"], "refused")

    async def test_dimensions_attempt_call_token_and_deadline_limits(self):
        with tempfile.TemporaryDirectory() as root:
            renderer = FakeRenderer()
            wrong_size = DrawingEnvironment(root, renderer=renderer)
            await wrong_size.reset("circle")
            await wrong_size.write_code(GOOD)
            self.assertIn("Expected 1920", (await wrong_size.submit())["error"])

            config = {**CONFIG, "maxAttempts": 1, "maxCalls": 5}
            env = DrawingEnvironment(root, config=config, renderer=renderer)
            await env.reset("circle")
            await env.write_code(GOOD)
            await env.run_code()
            self.assertIn("Preview budget", (await env.run_code())["error"])
            self.assertEqual((await env.submit())["outcome"], "submitted")

            await env.reset("circle")
            for _ in range(5):
                await env.run_code()
            self.assertEqual((await env.submit())["outcome"], "budget_exhausted")
            await env.reset("circle")
            await env.charge_tokens(config["maxPolicyTokens"])
            self.assertEqual((await env.submit())["outcome"], "budget_exhausted")

            expired = DrawingEnvironment(root, config={**CONFIG, "episodeMs": 0}, renderer=renderer)
            await expired.reset("circle")
            self.assertEqual((await expired.submit())["outcome"], "timed_out")

    async def test_renderer_exception_is_recorded_and_can_recover(self):
        class RecoveringRenderer(FakeRenderer):
            async def render(self, code, seed):
                if not self.calls:
                    self.calls.append((code, seed))
                    raise RuntimeError("worker crashed")
                return await super().render(code, seed)

        with tempfile.TemporaryDirectory() as root:
            env = DrawingEnvironment(root, config=CONFIG, renderer=RecoveringRenderer())
            await env.reset("circle")
            await env.write_code(GOOD)
            self.assertEqual((await env.run_code())["failureKind"], "infrastructure")
            self.assertEqual((await env.submit())["outcome"], "submitted")

    async def test_sequential_operations_and_validation(self):
        with tempfile.TemporaryDirectory() as root:
            env = DrawingEnvironment(root, renderer=FakeRenderer(), config=CONFIG)
            with self.assertRaisesRegex(ValueError, "prompt"):
                await env.reset("  ")
            await env.reset("circle")
            self.assertFalse((await env.write_code("  "))["ok"])
            self.assertFalse((await env.refuse(""))["ok"])
            self.assertEqual(env.state["runtimeVersion"], ENVIRONMENT_CONFIG["runtimeVersion"])


if __name__ == "__main__":
    unittest.main()
