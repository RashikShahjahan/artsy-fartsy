import argparse
import json
from importlib.resources import files
from pathlib import Path

from .harness import run_episode
from .models import DEFAULT_MODEL, Ollama


def main():
    parser = argparse.ArgumentParser(description="Render and record one restricted episode")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--program", type=Path, help="Defaults to the fixed drawing fixture")
    source.add_argument("--prompt", help="Generate a drawing program with Ollama")
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    parser.add_argument("--image", default="artsy-renderer:0.1.0")
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
    directory, record = run_episode(program, args.episodes, image=args.image,
                                    model=model, prompt=args.prompt)
    print(json.dumps({"episode": str(directory), **record}, indent=2))
    return 0 if record["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
