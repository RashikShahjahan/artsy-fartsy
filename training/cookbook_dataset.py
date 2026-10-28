"""Prompt-group dataset for the Tinker Cookbook RL trainer."""

from __future__ import annotations

import os
from pathlib import Path

import chz
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder

from drawing.dataset_client import chat_client
from training.cookbook_env import ArtsyGroupBuilder
from training.sft import batches


class ArtsyRLDataset(RLDataset):
    def __init__(self, specs, root, batch_size, group_size, epochs, seed,
                 max_tokens, max_length, normalize_rewards, judge):
        self.batches = list(batches(list(specs), batch_size, epochs, seed))
        self.root, self.group_size, self.judge = Path(root), group_size, judge
        self.max_tokens, self.max_length = max_tokens, max_length
        self.normalize_rewards = normalize_rewards

    def get_batch(self, index):
        return [
            ArtsyGroupBuilder(spec, self.group_size, index + 1, self.root, self.judge,
                              self.max_tokens, self.max_length, self.normalize_rewards)
            for spec in self.batches[index]
        ]

    def __len__(self):
        return len(self.batches)


@chz.chz
class ArtsyDatasetBuilder(RLDatasetBuilder):
    specs: tuple[dict, ...]
    output_dir: str
    batch_size: int
    group_size: int
    epochs: int
    seed: int
    max_tokens: int
    max_length: int
    normalize_rewards: bool
    judge_model: str
    judge_url: str
    judge_renderer: str = "qwen3_8_medium_reasoning"
    judge_revision: str | None = None
    judge_max_tokens: int = 16384

    safety_config: dict | None = None
    code_safety_config: dict | None = None

    async def __call__(self):
        judge = chat_client({"model": self.judge_model, "baseUrl": self.judge_url,
                             "apiKey": os.getenv("ARTSY_CORRECTNESS_JUDGE_API_KEY"),
                             "timeoutMs": 600000 if self.judge_url == "tinker://native" else 120000,
                             "renderer": self.judge_renderer,
                             "revision": self.judge_revision,
                             "maxTokens": self.judge_max_tokens})
        judges = {"adherence": judge}
        for role, config, env_prefix in (
                ("safety", self.safety_config, "ARTSY_SAFETY_JUDGE"),
                ("code_safety", self.code_safety_config, "ARTSY_CODE_SAFETY_JUDGE")):
            if not config or not config.get("model") or not config.get("baseUrl"):
                raise ValueError(role + " judge must be independently configured")
            judges[role] = chat_client({**config, "apiKey": os.getenv(env_prefix + "_API_KEY")})
        return ArtsyRLDataset(self.specs, self.output_dir, self.batch_size,
                              self.group_size, self.epochs, self.seed,
                              self.max_tokens, self.max_length,
                              self.normalize_rewards, judges), None
