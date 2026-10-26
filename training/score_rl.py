"""Trusted, one-shot RL scorer executed only in a fresh Docker container.

The training host passes untrusted Python source through stdin; this wrapper
returns render feedback plus the verified PNG without any filesystem mount.
"""

import argparse
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, "/opt/artsy")
from render import SOURCE_BYTES, render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**53 - 1:
        parser.error("Invalid seed")
    source = sys.stdin.buffer.read(SOURCE_BYTES + 1).decode("utf-8")
    output = Path("/tmp/artsy-rl-score")
    feedback = render(source, args.seed, output, Path("/opt/artsy/runtime"), seconds=60)
    payload = {"feedback": feedback}
    if feedback["ok"]:
        payload["image"] = base64.b64encode((output / "image.png").read_bytes()).decode()
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
