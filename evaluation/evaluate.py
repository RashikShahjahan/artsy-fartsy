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
    parser.add_argument("--correctness-judge-model", default=os.getenv("ARTSY_CORRECTNESS_JUDGE_MODEL"),
                        help="Vision model for prompt correctness; optionally set ARTSY_CORRECTNESS_JUDGE_URL/API_KEY")
    parser.add_argument("--safety-judge-model", default=os.getenv("ARTSY_SAFETY_JUDGE_MODEL"),
                        help="Vision model for harmful content; optionally set ARTSY_SAFETY_JUDGE_URL/API_KEY")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--execution-only", action="store_true", help="Smoke test rendering without either vision judge")
    parser.add_argument("--dry-run", action="store_true", help="Validate specs and show the plan without Docker or model requests")
    args = parser.parse_args()
    if args.execution_only:
        args.correctness_judge_model = None
        args.safety_judge_model = None
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    for role, model in (("correctness", args.correctness_judge_model), ("safety", args.safety_judge_model)):
        if model and model.removeprefix("openai/") == args.model:
            parser.error(f"select an independent {role} judge rather than the model under evaluation")
    specs = load_specs(args.dataset)
    selected = specs[:args.limit] if args.limit else specs
    plan = {"model": args.model, "base_url": os.getenv("ARTSY_MODEL_URL", "http://127.0.0.1:8080/v1"),
            "dataset": str(args.dataset.resolve()), "samples": len(selected), "epochs": args.epochs,
            "correctness_judge_model": args.correctness_judge_model,
            "safety_judge_model": args.safety_judge_model, "threshold": args.threshold,
            "mode": "execution-only" if args.execution_only else "full",
            "ids": [s["id"] for s in selected]}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.execution_only and not (args.correctness_judge_model and args.safety_judge_model):
        parser.error("Full evaluation requires both --correctness-judge-model and --safety-judge-model "
                     "(or the corresponding ARTSY_*_JUDGE_MODEL variables); use --execution-only for render tests")
    args.output.mkdir(parents=True, exist_ok=False)

    # Inspect records each role and its model calls in the evaluation log.
    roles = {}
    for role, model in (("correctness", args.correctness_judge_model), ("safety", args.safety_judge_model)):
        if not model:
            continue
        prefix = f"ARTSY_{role.upper()}_JUDGE"
        if url := os.getenv(f"{prefix}_URL"):
            roles[f"{role}_grader"] = get_model("openai/" + model.removeprefix("openai/"),
                base_url=url, api_key=os.getenv(f"{prefix}_API_KEY", "local-no-auth"),
                responses_api=False)
        else:
            roles[f"{role}_grader"] = get_model(model)
    logs = eval(
        drawing_task(selected, str(args.output / "artifacts"), args.correctness_judge_model,
                     args.safety_judge_model, args.threshold),
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
