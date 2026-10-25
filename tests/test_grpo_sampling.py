"""The pilot must preserve within-prompt grouping and stochastic generation."""
import asyncio
import unittest
from unittest.mock import patch

from inspect_ai.model import GenerateConfig
from evaluation.sample_grpo import select_prompts, summarize_rewards
from evaluation.harness import drawing_task


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
