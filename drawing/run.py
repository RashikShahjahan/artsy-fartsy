"""Render a source file as a fresh drawing episode.

Usage: python -m drawing.run examples/circle.py --prompt "Draw a blue circle"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from .environment import DrawingEnvironment


async def _run(args: argparse.Namespace) -> int:
    try:
        source = Path(args.source).resolve().read_text(encoding="utf-8")
        episode_root = Path(args.episode) if args.episode else Path("episodes") / str(uuid.uuid4())
        env = DrawingEnvironment(episode_root.resolve())
        episode = await env.reset(args.prompt, args.seed)
        written = await env.write_code(source)
        result = await env.submit() if written["ok"] else written
        print(json.dumps({"directory": episode["directory"], **result}, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Python drawing source file")
    parser.add_argument("--prompt", required=True, help="Prompt describing the drawing")
    parser.add_argument("--episode", help="Artifact root for this episode")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2**53 - 1:
        parser.error("seed must be a nonnegative safe integer")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
