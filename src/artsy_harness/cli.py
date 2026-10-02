import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path

from .harness import run_episode
from .models import DEFAULT_MODEL, Ollama


def main():
    parser = argparse.ArgumentParser(description="Save and inspect one episode; execution is disabled by default")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--program", type=Path, help="Save/inspect a program; defaults to the drawing fixture")
    source.add_argument("--prompt", help="Generate and save a drawing program with Ollama (generate-only by default)")
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable UNSANDBOXED host execution, not containment. Use only on code you personally reviewed; --no-execute is the default")
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable model thinking (default: on)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--model-timeout", type=float, default=600)
    args = parser.parse_args()
    if args.execute:
        print("WARNING: UNSANDBOXED host execution enabled, not containment. "
              "Code has your user's file, environment, and network access. "
              "Use only on code you personally reviewed. --prompt --execute runs fresh, "
              "unreviewed model output; prefer saving first and reviewing --program.", file=sys.stderr)
    program = None
    model = None
    if args.prompt is not None:
        model = Ollama(base_url=args.ollama_url, model=args.model, timeout=args.model_timeout,
                       think=args.think,
                       options={"seed": args.seed, "temperature": args.temperature,
                                "top_p": args.top_p, "num_predict": args.num_predict,
                                "num_ctx": args.num_ctx})
    elif args.program:
        program = args.program.read_text()
    else:
        program = files("artsy_harness").joinpath("fixtures/drawing.py").read_text()
    directory, record = run_episode(program, args.episodes, model=model, prompt=args.prompt,
                                    execute=args.execute)
    print(json.dumps({"episode": str(directory), **record}, indent=2))
    return 0 if record["status"] in ("success", "generated", "inspected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
