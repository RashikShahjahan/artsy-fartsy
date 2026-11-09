"""Verifiable-reward RL after SFT and before preference RLHF.

Example:
    python -m training.rlvr --init-checkpoint checkpoints/sft/final.json \
        --dataset datasets/parti-prompts/rl/prompts.jsonl \
        --output-dir checkpoints/rlvr
"""

import sys

from training.grpo import main


if __name__ == "__main__":
    raise SystemExit(main(["--reward-mode", "rlvr", *sys.argv[1:]]))
