"""Sample a stratified correctness-only pilot without updating model weights.

python -m evaluation.sample_grpo --output eval-results/grpo-pilot
Requires Docker and the native Tinker bridge (make serve).
"""

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics

from dotenv import load_dotenv
from inspect_ai import eval

from drawing.dataset_client import chat_client
from drawing.dataset_specs import DIFFICULTIES
from drawing.judges import JUDGE_MAX_TOKENS, correctness_judge_prompt, parse_correctness_judgment
from evaluation.harness import (ROOT, check_native_bridge, drawing_task,
                                load_specs, model_from_env, sha256)
from inference.model_catalog import model_catalog


def select_prompts(specs, per_difficulty=1):
    selected = []
    for level in DIFFICULTIES:
        candidates = [spec for spec in specs if spec.get("difficulty") == level]
        if len(candidates) < per_difficulty:
            raise ValueError(f"Not enough {level} prompts")
        selected.extend(candidates[:per_difficulty])
    return selected


def summarize_rewards(rewards):
    mean = statistics.mean(rewards)
    return {"rewards": rewards, "mean": mean, "std": statistics.pstdev(rewards),
            "advantages": [reward - mean for reward in rewards],
            "has_learning_signal": len(set(rewards)) > 1}


async def score_outputs(specs, output, group_size, judge, concurrency=4):
    """Judge final renders only. Judge failures stop the pilot, never score zero."""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Judge concurrency must be a positive integer")
    semaphore = asyncio.Semaphore(concurrency)

    async def score_attempt(spec, epoch):
        directory = output / "artifacts" / f"{spec['id']}-{epoch}"
        result = json.loads((directory / "result.json").read_text())
        record_path = directory / "correctness.json"
        if record_path.exists():
            judgment = json.loads(record_path.read_text())
        elif result["scores"]["render_valid"]:
            png = (directory / "image.png").read_bytes()
            if sha256(png) != result["imageHash"]:
                raise ValueError("Final image hash mismatch")
            async with semaphore:
                response = await judge.complete([{"role": "user", "content": [
                    {"type": "text", "text": correctness_judge_prompt(spec["prompt"])},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                ]}], JUDGE_MAX_TOKENS, temperature=0)
            if response.get("finishReason") == "length":
                raise ValueError("Correctness judgment was truncated")
            judgment = {**parse_correctness_judgment(response["text"]),
                        "model": judge.model, "response": response,
                        "image_sha256": result["imageHash"]}
            record_path.write_text(json.dumps(judgment, indent=2) + "\n")
        else:
            judgment = {"score": 0.0, "reason": "No valid final rendered drawing",
                        "outcome": result["outcome"]}
            record_path.write_text(json.dumps(judgment, indent=2) + "\n")
        reward = judgment["score"]
        print(json.dumps({"id": spec["id"], "difficulty": spec["difficulty"],
                          "epoch": epoch, "reward": reward}), flush=True)
        return {"epoch": epoch, "reward": reward,
                "render_valid": bool(result["scores"]["render_valid"]),
                "source_sha256": result.get("sourceHash"),
                "image_sha256": result.get("imageHash"),
                "trajectory": result.get("trajectory"),
                "artifacts": str(directory.resolve())}

    tasks = [asyncio.create_task(score_attempt(spec, epoch))
             for spec in specs for epoch in range(1, group_size + 1)]
    try:
        attempts = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    groups = []
    for index, spec in enumerate(specs):
        members = attempts[index * group_size:(index + 1) * group_size]
        groups.append({"id": spec["id"], "prompt": spec["prompt"],
                       "difficulty": spec["difficulty"], "seed": spec["seed"],
                       **summarize_rewards([attempt["reward"] for attempt in members]), "attempts": members})
    return groups


def main(argv=None):
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/grpo-v2/prompts.jsonl")
    parser.add_argument("--model", default="artsy/sft-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--per-difficulty", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--score-only", action="store_true", help="Retry judging already completed rollouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.group_size < 2 or args.per_difficulty < 1 or args.max_samples < 1
            or not math.isfinite(args.temperature) or not 0 < args.temperature <= 2):
        parser.error("group size must be >=2; counts positive; temperature in (0, 2]")
    specs = select_prompts(load_specs(args.dataset), args.per_difficulty)
    entry = next((item for item in model_catalog() if item["id"] == args.model), None)
    if not entry or not entry["model_path"]:
        parser.error("Select a completed SFT checkpoint from the model catalog")
    judge_model = os.getenv("ARTSY_CORRECTNESS_JUDGE_MODEL")
    judge_url = os.getenv("ARTSY_CORRECTNESS_JUDGE_URL")
    if not judge_model or not judge_url:
        parser.error("Set ARTSY_CORRECTNESS_JUDGE_MODEL and ARTSY_CORRECTNESS_JUDGE_URL")
    judge = chat_client({"model": judge_model, "baseUrl": judge_url,
                         "apiKey": os.getenv("ARTSY_CORRECTNESS_JUDGE_API_KEY"),
                         "timeoutMs": 120000})
    plan = {"model": args.model, "sampler_path": entry["model_path"],
            "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset.read_bytes()),
            "samples": len(specs), "group_size": args.group_size,
            "episodes": len(specs) * args.group_size, "temperature": args.temperature,
            "max_samples": args.max_samples, "correctness_judge_model": judge.model,
            "correctness_judge_url": judge.baseUrl, "judge_temperature": 0,
            "reward": "final-image correctness only", "training_updates": 0,
            "specs": specs}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.score_only:
        if json.loads((args.output / "plan.json").read_text()) != plan:
            parser.error("Scoring configuration differs from the recorded plan")
    else:
        check_native_bridge(args.model)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        logs = eval(drawing_task(specs, str(args.output / "artifacts"),
                                 temperature=args.temperature, max_connections=args.max_samples),
                    model=model_from_env(args.model), epochs=args.group_size,
                    max_samples=args.max_samples, log_dir=str(args.output / "logs"))
        if not logs or any(log.status != "success" for log in logs):
            raise RuntimeError("Rollout collection did not complete successfully; inspect logs before retrying")
    # Incomplete/errored collection must not enter reward statistics.
    for spec in specs:
        for epoch in range(1, args.group_size + 1):
            if not (args.output / "artifacts" / f"{spec['id']}-{epoch}" / "result.json").is_file():
                raise RuntimeError("Missing rollout result; cannot compute reward variation")
    groups = asyncio.run(score_outputs(specs, args.output, args.group_size, judge, args.max_samples))
    summary = {**plan, "completed_at": datetime.now(timezone.utc).isoformat(), "groups": groups,
               "groups_with_learning_signal": sum(g["has_learning_signal"] for g in groups),
               "valid_renders": sum(a["render_valid"] for g in groups for a in g["attempts"])}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str((args.output / "summary.json").resolve()),
                      "groups_with_learning_signal": summary["groups_with_learning_signal"],
                      "valid_renders": summary["valid_renders"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
