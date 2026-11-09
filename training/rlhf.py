"""Run HPSv3-preference RLHF initialized from the RLVR stage.

The legacy `training.grpo` CLI still accepts SFT checkpoints for older runs;
this stage entrypoint defaults to `checkpoints/rlvr-v1/final.json`.
"""

import sys

from training.grpo import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    continuing_or_finalizing = any(
        arg == "--continue-from" or arg.startswith("--continue-from=")
        or arg == "--finalize-only" for arg in arguments)
    if not continuing_or_finalizing and not any(
            arg == "--init-checkpoint" or arg.startswith("--init-checkpoint=")
            for arg in arguments):
        arguments = ["--init-checkpoint", "checkpoints/rlvr-v1/final.json", *arguments]
    if not any(arg == "--output-dir" or arg.startswith("--output-dir=")
               for arg in arguments):
        arguments = ["--output-dir", "checkpoints/rlhf-v1", *arguments]
    raise SystemExit(main([*arguments, "--reward-mode", "rlhf"]))
