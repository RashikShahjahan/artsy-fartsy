"""Run judge-free verifiable-reward RL between SFT and RLHF.

Example:
    python -m training.rlvr --init-checkpoint checkpoints/sft/final.json \
        --dataset datasets/parti-prompts/rl/prompts.jsonl \
        --output-dir checkpoints/rlvr
"""

import sys

from training.grpo import main


if __name__ == "__main__":
    # Give this stage its own output namespace, without overriding a chosen path.
    arguments = sys.argv[1:]
    if not any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in arguments):
        arguments = ["--output-dir", "checkpoints/rlvr-v1", *arguments]
    # The RLVR entrypoint cannot be switched to preference scoring accidentally.
    raise SystemExit(main([*arguments, "--reward-mode", "rlvr"]))
