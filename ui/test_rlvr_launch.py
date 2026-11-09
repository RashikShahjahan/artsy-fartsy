"""RLVR launch and model selection regressions."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import launch
from inference.model_catalog import model_catalog
from drawing.config import MODEL_ID, MODEL_REVISION
from training.sft import RENDERER


class RLVRLaunchTests(unittest.TestCase):
    def test_rl_stage_is_allowlisted_and_passed_to_trainer(self):
        argv, output = launch.command("rl", {
            "reward_mode": "rlvr", "harness": "mini-swe",
            "max_steps": 1, "dry_run": True}, "a" * 32)
        self.assertEqual(argv[argv.index("--reward-mode") + 1], "rlvr")
        self.assertIn("training.grpo", argv)
        self.assertIn("--dry-run", argv)
        self.assertEqual(output.name, "ui-rl-" + "a" * 32)
        with self.assertRaisesRegex(ValueError, "Unknown RL reward stage"):
            launch.command("rl", {"reward_mode": "made-up"}, "a" * 32)

    def test_completed_rlvr_is_selectable_but_rlhf_is_not_an_initializer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, algorithm in (("sft", None),
                                    ("rlvr", "tinker-cookbook-rlvr-v1"),
                                    ("rlhf", "tinker-cookbook-rlhf-v8")):
                path = root / "checkpoints" / name / "final.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "config": {"model": MODEL_ID, "renderer": RENDERER,
                               "tokenizer_revision": MODEL_REVISION,
                               "lora_rank": 32, "algorithm": algorithm},
                    "state_path": f"tinker://{name}/state",
                    "sampler_path": f"tinker://{name}/sampler",
                    "step": 1}))
            with patch.object(launch, "ROOT", root):
                options = launch.model_choices()["rl"]
            self.assertEqual({option["value"] for option in options}, {
                "checkpoints/sft/final.json", "checkpoints/rlvr/final.json"})
            self.assertIn("RLVR", next(option["label"] for option in options
                                      if option["value"].startswith("checkpoints/rlvr/")))
            entries = {entry["id"]: entry for entry in model_catalog(root)}
            self.assertIn("RLVR", entries["artsy/rlvr"]["label"])


if __name__ == "__main__":
    unittest.main()
