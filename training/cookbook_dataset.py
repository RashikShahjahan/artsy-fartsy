"""Prompt-group dataset for the Tinker Cookbook RL trainer."""

from __future__ import annotations

import os
from pathlib import Path

import chz
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder

from training.hpsv3 import HPSv3Client
from training.cookbook_env import ArtsyGroupBuilder
from training.sft import batches


class ArtsyRLDataset(RLDataset):
    def __init__(self, specs, root, batch_size, group_size, epochs, seed,
                 max_tokens, max_length, normalize_rewards, judge, model, renderer,
                 revision):
        self.batches = list(batches(list(specs), batch_size, epochs, seed))
        self.root, self.group_size, self.judge = Path(root), group_size, judge
        self.max_tokens, self.max_length = max_tokens, max_length
        self.normalize_rewards = normalize_rewards
        self.model, self.renderer, self.revision = model, renderer, revision

    def get_batch(self, index):
        return [
            ArtsyGroupBuilder(spec, self.group_size, index + 1, self.root, self.judge,
                              self.max_tokens, self.max_length, self.normalize_rewards,
                              self.model, self.renderer, self.revision)
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
    max_tokens: int | None
    max_length: int
    normalize_rewards: bool
    model: str
    renderer: str
    revision: str | None
    hpsv3_model: str
    hpsv3_url: str
    hpsv3_center: float = 0.0
    hpsv3_scale: float = 5.0

    async def __call__(self):
        judges = {"hpsv3": HPSv3Client(
            self.hpsv3_url, self.hpsv3_model, center=self.hpsv3_center,
            scale=self.hpsv3_scale, api_key=os.getenv("ARTSY_HPSV3_API_KEY"))}
        return ArtsyRLDataset(self.specs, self.output_dir, self.batch_size,
                              self.group_size, self.epochs, self.seed,
                              self.max_tokens, self.max_length,
                              self.normalize_rewards, judges, self.model, self.renderer,
                              self.revision), None
