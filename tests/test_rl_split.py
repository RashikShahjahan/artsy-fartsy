"""Pure regression tests for the disjoint 227/100 Parti RL stage split."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.rl_split import (RL_TOTAL, RLVR_COUNT, RLHF_COUNT, SPLIT_VERSION,
                               partition_rl_prompts, stage_specs)


def sample_specs(total=RL_TOTAL):
    return [
        {"id": f"parti-{index:04d}", "prompt": f"Draw scene {index}",
         "seed": index, "source": {
             "category": f"category-{index % 5}",
             "challenge": f"challenge-{index % 3}"}}
        for index in range(total)
    ]


class RLStageSplitTests(unittest.TestCase):
    def test_exact_counts_zero_overlap_and_original_order(self):
        specs = sample_specs()
        result = partition_rl_prompts(specs)
        self.assertEqual(len(result["rlvr"]), RLVR_COUNT)
        self.assertEqual(len(result["rlhf"]), RLHF_COUNT)
        selected = {row["id"] for row in result["rlvr"]}
        held = {row["id"] for row in result["rlhf"]}
        self.assertFalse(selected & held)
        self.assertEqual(selected | held, {row["id"] for row in specs})
        self.assertEqual(result["rlvr"], [row for row in specs if row["id"] in selected])
        self.assertEqual(result["rlhf"], [row for row in specs if row["id"] in held])

    def test_membership_does_not_depend_on_source_order(self):
        specs = sample_specs()
        forward = partition_rl_prompts(specs)
        backward = partition_rl_prompts(list(reversed(specs)))
        for stage in ("rlvr", "rlhf"):
            self.assertEqual(
                {row["id"] for row in forward[stage]},
                {row["id"] for row in backward[stage]})

    def test_stage_selector_only_applies_to_canonical_rl_pool(self):
        specs = sample_specs()
        with tempfile.TemporaryDirectory() as folder:
            canonical = Path(folder) / "rl" / "prompts.jsonl"
            custom = Path(folder) / "other.jsonl"
            with patch("training.rl_split.DEFAULT_RL_DATASET", canonical):
                first, version1 = stage_specs(canonical, specs, "rlvr")
                second, version2 = stage_specs(canonical, specs, "rlhf")
                self.assertEqual((len(first), len(second)), (227, 100))
                self.assertEqual((version1, version2), (SPLIT_VERSION, SPLIT_VERSION))
                self.assertFalse(
                    {row["id"] for row in first} & {row["id"] for row in second})
                untouched, version3 = stage_specs(custom, specs[:3], "rlvr")
                self.assertEqual(untouched, specs[:3])
                self.assertIsNone(version3)
                with self.assertRaisesRegex(ValueError, "327"):
                    stage_specs(canonical, specs[:-1], "rlvr")

    def test_invalid_or_duplicate_inputs_are_rejected(self):
        for rows in ([], sample_specs(1)):
            with self.assertRaises(ValueError):
                partition_rl_prompts(rows)
        rows = sample_specs()
        rows[-1]["id"] = rows[0]["id"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            partition_rl_prompts(rows)

    def test_stage_balance_preserves_source_strata(self):
        specs = sample_specs()
        results = partition_rl_prompts(specs)
        for category in {row["source"]["category"] for row in specs}:
            count = sum(row["source"]["category"] == category for row in specs)
            selected = sum(row["source"]["category"] == category for row in results["rlvr"])
            self.assertLessEqual(abs(selected - count * RLVR_COUNT / RL_TOTAL), 2)


if __name__ == "__main__":
    unittest.main()
