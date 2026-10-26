"""The pilot must preserve within-prompt grouping and stochastic generation."""
import asyncio
import base64
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from inspect_ai.model import GenerateConfig
from evaluation.sample_grpo import score_outputs, select_prompts, summarize_rewards
from evaluation.harness import drawing_task, sha256


class PilotTests(unittest.TestCase):
    def test_stratification_and_group_advantages(self):
        specs = [{"id": str(i), "difficulty": level} for i, level in enumerate(
            ["hard", "easy", "very-hard", "medium", "easy"])]
        self.assertEqual([s["id"] for s in select_prompts(specs)], ["1", "3", "0", "2"])
        row = summarize_rewards([0.2, 0.6, 0.8, 0.4])
        self.assertAlmostEqual(sum(row["advantages"]), 0)
        self.assertTrue(row["has_learning_signal"])
        self.assertFalse(summarize_rewards([1.0] * 4)["has_learning_signal"])

    def test_mini_swe_temperature_override_at_bridge_boundary(self):
        with patch("evaluation.harness.mini_swe_agent") as mini:
            drawing_task([{"id": "x", "prompt": "Draw a disk", "seed": 1}],
                         "/tmp/artsy-pilot-test", temperature=0.8)
        callback = mini.call_args.kwargs["filter"]
        self.assertEqual(mini.call_args.kwargs["env"]["MSWEA_MODEL_NAME"], "openai/inspect")
        original = GenerateConfig(temperature=0)
        result = asyncio.run(callback(None, [], [], None, original))
        self.assertEqual(result.config.temperature, 0.8)
        self.assertEqual(original.temperature, 0)


class ConcurrentJudgingTests(unittest.IsolatedAsyncioTestCase):
    def artifacts(self, output, group_size=3):
        specs = [{"id": name, "prompt": name, "difficulty": "easy", "seed": 1}
                 for name in ("a", "b")]
        for index, spec in enumerate(specs):
            for epoch in range(1, group_size + 1):
                directory = output / "artifacts" / f"{spec['id']}-{epoch}"
                directory.mkdir(parents=True)
                png = str(index * group_size + epoch).encode()
                (directory / "image.png").write_bytes(png)
                (directory / "result.json").write_text(json.dumps({
                    "scores": {"render_valid": True}, "imageHash": sha256(png),
                    "outcome": "submitted"}))
        return specs

    async def test_bounded_concurrency_preserves_groups_despite_out_of_order_completion(self):
        active, peak = 0, 0
        completed = []
        second_done = asyncio.Event()

        async def complete(messages, *args, **kwargs):
            nonlocal active, peak
            value = int(base64.b64decode(messages[0]["content"][1]["image_url"]["url"].split(",")[1]))
            active += 1
            peak = max(peak, active)
            try:
                if value == 1:
                    await second_done.wait()
                else:
                    await asyncio.sleep(0)
                completed.append(value)
                if value == 2:
                    second_done.set()
                return {"text": json.dumps({"score": value / 10, "reason": "scored"})}
            finally:
                active -= 1

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp)
            specs = self.artifacts(output)
            result = await asyncio.wait_for(score_outputs(
                specs, output, 3, SimpleNamespace(model="judge", complete=complete), 2), timeout=5)
            self.assertEqual(peak, 2)
            self.assertEqual(completed[0], 2)
            self.assertEqual([g["id"] for g in result], ["a", "b"])
            self.assertEqual(result[0]["rewards"], [0.1, 0.2, 0.3])
            self.assertEqual(result[1]["rewards"], [0.4, 0.5, 0.6])
            self.assertEqual([a["epoch"] for a in result[1]["attempts"]], [1, 2, 3])
            self.assertEqual(len(list(output.glob("artifacts/*/correctness.json"))), 6)

    async def test_failure_propagates_and_cancels_pending_judgments(self):
        started = asyncio.Event()
        active = 0

        async def complete(messages, *args, **kwargs):
            nonlocal active
            value = int(base64.b64decode(messages[0]["content"][1]["image_url"]["url"].split(",")[1]))
            active += 1
            try:
                if value == 1:
                    await started.wait()
                    raise RuntimeError("Judge unavailable")
                started.set()
                await asyncio.Event().wait()
            finally:
                active -= 1

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            specs = self.artifacts(output)
            with self.assertRaisesRegex(RuntimeError, "Judge unavailable"):
                await asyncio.wait_for(score_outputs(
                    specs, output, 3, SimpleNamespace(model="judge", complete=complete), 2), timeout=5)
            self.assertEqual(active, 0)
            self.assertEqual(list(output.glob("artifacts/*/correctness.json")), [])

    async def test_cached_and_invalid_renders_skip_judge(self):
        async def complete(*args, **kwargs):
            self.fail("Cached or invalid drawings must not call the judge")

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp)
            specs = self.artifacts(output, group_size=1)
            (output / "artifacts/a-1/correctness.json").write_text(json.dumps({"score": 0.8}))
            (output / "artifacts/b-1/result.json").write_text(json.dumps({
                "scores": {"render_valid": False}, "outcome": "refused"}))
            result = await score_outputs(specs, output, 1, SimpleNamespace(complete=complete), 2)
            self.assertEqual([g["rewards"] for g in result], [[0.8], [0.0]])
