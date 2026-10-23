"""Draw with the same Mini SWE Agent used by the baseline evaluation."""

import argparse
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from inspect_ai import eval

from .harness import BASE_MODEL, drawing_task, model_from_env, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="+")
    parser.add_argument("--episode", type=Path, default=Path("episodes") / str(uuid4()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=os.getenv("ARTSY_MODEL", BASE_MODEL))
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**53 - 1:
        parser.error("seed must be a nonnegative safe integer")
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        parser.error("prompt must not be empty")
    request = {"prompt": prompt, "seed": args.seed, "model": args.model,
               "base_url": os.getenv("ARTSY_MODEL_URL", "http://127.0.0.1:8080/v1")}
    directory = args.episode.resolve()
    if directory.exists() and any(directory.iterdir()):
        manifest = directory / "episode.json"
        if not manifest.exists() or json.loads(manifest.read_text()) != request:
            parser.error("This episode belongs to another request; choose a new --episode directory")
        result_file = directory / "result.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            if result["outcome"] == "refused":
                print(json.dumps(result, indent=2))
                return 0
            if result["outcome"] == "submitted" and all(
                (directory / name).is_file() and sha256((directory / name).read_bytes()) == result[key]
                for name, key in (("program.py", "sourceHash"), ("image.png", "imageHash"))
            ):
                print(json.dumps(result, indent=2))
                return 0
        parser.error("Episode is incomplete or its artifacts changed; choose a new --episode directory")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "episode.json").write_text(json.dumps(request, indent=2) + "\n")
    spec = {"id": "drawing", "prompt": prompt, "requirements": [prompt], "seed": args.seed}
    logs = eval(drawing_task([spec], str(directory / "artifacts")),
                model=model_from_env(args.model), log_dir=str(directory / "logs"), max_samples=1)
    artifact = directory / "artifacts/drawing-1"
    for name in ("program.py", "image.png", "result.json"):
        if (artifact / name).is_file():
            shutil.copyfile(artifact / name, directory / name)
    success = bool(logs) and all(log.status == "success" for log in logs) and all(
        sample.error is None for log in logs for sample in (log.samples or []))
    result_file = directory / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else {"outcome": "incomplete"}
    print(json.dumps(result, indent=2))
    return 0 if success and result["outcome"] in ("submitted", "refused") else 1


if __name__ == "__main__":
    raise SystemExit(main())
