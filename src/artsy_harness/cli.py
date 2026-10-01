import argparse
import json
from importlib.resources import files
from pathlib import Path

from .harness import run_episode


def main():
    parser = argparse.ArgumentParser(description="Render and record one restricted episode")
    parser.add_argument("--program", type=Path, help="Defaults to the fixed drawing fixture")
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    parser.add_argument("--image", default="artsy-renderer:0.1.0")
    args = parser.parse_args()
    if args.program:
        program = args.program.read_text()
    else:
        program = files("artsy_harness").joinpath("fixtures/drawing.py").read_text()
    directory, record = run_episode(program, args.episodes, image=args.image)
    print(json.dumps({"episode": str(directory), **record}, indent=2))
    return 0 if record["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
