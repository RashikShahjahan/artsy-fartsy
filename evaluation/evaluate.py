"""Evaluate Qwen's drawing baseline with Inspect SWE's Mini SWE Agent.

Run `python -m evaluation.evaluate --help` from the repository root.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import get_model

from .harness import BASE_MODEL, ROOT, drawing_task, load_specs, model_from_env


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/prompts.jsonl")
    parser.add_argument("--model", default=os.getenv("ARTSY_MODEL", BASE_MODEL), help="Model ID served by ARTSY_MODEL_URL")
    parser.add_argument("--output", type=Path, default=Path("eval-results") / datetime.now(timezone.utc).strftime("base-%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--limit", type=positive)
    parser.add_argument("--epochs", type=positive, default=1)
    parser.add_argument("--judge-model", default=os.getenv("ARTSY_JUDGE_MODEL"), help="Independent vision judge; optionally set ARTSY_JUDGE_URL/API_KEY")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--execution-only", action="store_true", help="Smoke test rendering without the correctness and harm judge")
    parser.add_argument("--dry-run", action="store_true", help="Validate specs and show the plan without Docker or model requests")
    args = parser.parse_args()
    if args.execution_only:
        args.judge_model = None
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if args.judge_model and args.judge_model.removeprefix("openai/") == args.model:
        parser.error("select an independent judge rather than the model under evaluation")
    specs = load_specs(args.dataset)
    selected = specs[:args.limit] if args.limit else specs
    plan = {"model": args.model, "base_url": os.getenv("ARTSY_MODEL_URL", "http://127.0.0.1:8080/v1"),
            "dataset": str(args.dataset.resolve()), "samples": len(selected), "epochs": args.epochs,
            "judge_model": args.judge_model, "threshold": args.threshold,
            "mode": "execution-only" if args.execution_only else "full",
            "ids": [s["id"] for s in selected]}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.execution_only and not args.judge_model:
        parser.error("Full evaluation requires --judge-model or ARTSY_JUDGE_MODEL for image correctness and harmfulness; use --execution-only for a rendering smoke test")
    args.output.mkdir(parents=True, exist_ok=False)
    roles = {}
    if args.judge_model:
        if url := os.getenv("ARTSY_JUDGE_URL"):
            roles["grader"] = get_model("openai/" + args.judge_model.removeprefix("openai/"),
                base_url=url, api_key=os.getenv("ARTSY_JUDGE_API_KEY", "local-no-auth"),
                responses_api=False)
        else:
            roles["grader"] = get_model(args.judge_model)
    logs = eval(
        drawing_task(selected, str(args.output / "artifacts"), args.judge_model, args.threshold),
        model=model_from_env(args.model), model_roles=roles,
        log_dir=str(args.output / "logs"), epochs=args.epochs, max_samples=1,
    )
    samples = [sample for log in logs for sample in (log.samples or [])]
    errors = sum(sample.error is not None for sample in samples)
    summary = {**plan, "completed_samples": len(samples), "sample_errors": errors,
               "logs": [log.location for log in logs],
               "results": [log.results.model_dump(mode="json") if log.results else None for log in logs],
               "usage": [log.stats.model_dump(mode="json") for log in logs],
               "status": "success" if logs and all(log.status == "success" for log in logs)
                   and not errors and len(samples) == len(selected) * args.epochs else "incomplete"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Results: {args.output.resolve() / 'summary.json'}")
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
