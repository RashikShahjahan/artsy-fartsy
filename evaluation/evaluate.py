"""Evaluate Qwen's drawing baseline with the host-side Inspect agent.

Run `python -m evaluation.evaluate --help` from the repository root.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model

from .harness import (BASE_MODEL, ROOT, check_native_bridge, drawing_task,
                      load_specs, model_from_env, model_url)


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _result(path: Path, cache: dict[Path, dict | None]) -> dict | None:
    if path not in cache:
        cache[path] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return cache[path]


def _recover_log(log, expected_samples: int):
    """Read persisted samples only when the returned log may be incomplete."""
    samples = getattr(log, "samples", None)
    results = getattr(log, "results", None)
    incomplete = (getattr(log, "status", None) != "success" or samples is None
                  or len(samples) != expected_samples)
    if results is not None and samples is not None:
        for field in ("total_samples", "logged_samples"):
            count = getattr(results, field, None)
            if count is not None and count != len(samples):
                incomplete = True
    return read_eval_log(log.location) if incomplete and Path(log.location).is_file() else log


def difficulty_summary(specs: list[dict], artifacts: Path, epochs: int,
                       result_cache: dict[Path, dict | None] | None = None) -> dict:
    """Aggregate observed outcomes by authored difficulty, not repair-prompt labels."""
    cache = result_cache if result_cache is not None else {}
    groups: dict[str, dict] = {}
    for spec in specs:
        level = spec.get("difficulty", "unspecified")
        group = groups.setdefault(level, {"completed": 0, "render_valid": 0,
            "expected_refusals": 0, "refusal_correct": 0,
            "previews_observed": 0, "initial_execution_failures": 0,
            "code_checked": 0, "code_malicious": 0,
            "revisions_attempted": 0, "execution_recoveries": 0})
        for epoch in range(1, epochs + 1):
            path = artifacts / f"{spec['id']}-{epoch}" / "result.json"
            result = _result(path, cache)
            if result is None:
                continue
            scores = result.get("scores") or {}
            trajectory = result.get("trajectory") or {}
            group["completed"] += 1
            group["render_valid"] += int(bool(scores.get("render_valid")))
            group["code_checked"] += int(bool(scores.get("code_checked")))
            group["code_malicious"] += int(bool(scores.get("code_malicious")))
            if spec.get("kind") == "refusal":
                group["expected_refusals"] += 1
                group["refusal_correct"] += int(bool(result.get("refusalCorrect")))
            group["previews_observed"] += int(trajectory.get("initialRenderValid") is not None)
            group["initial_execution_failures"] += int(trajectory.get("initialRenderValid") is False)
            group["revisions_attempted"] += int(bool(trajectory.get("revisionAttempted")))
            group["execution_recoveries"] += int(bool(trajectory.get("executionRecovered")))
    for group in groups.values():
        group["refusal_success_rate"] = (
            group["refusal_correct"] / group["expected_refusals"]
            if group["expected_refusals"] else None)
        group["render_rate"] = (group["render_valid"] / group["completed"]
                                if group["completed"] else None)
        group["execution_recovery_rate"] = (
            group["execution_recoveries"] / group["initial_execution_failures"]
            if group["initial_execution_failures"] else None)
    return groups


def main() -> int:
    # Read endpoint/judge settings before constructing defaults and model clients.
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/prompts.jsonl")
    parser.add_argument("--model", default=os.getenv("ARTSY_MODEL", BASE_MODEL), help="Model ID served by ARTSY_MODEL_URL")
    parser.add_argument("--output", type=Path, default=Path("eval-results") / datetime.now(timezone.utc).strftime("base-%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--limit", type=positive)
    parser.add_argument("--epochs", type=positive, default=1)
    parser.add_argument("--max-connections", type=positive, default=10, help="Maximum concurrent model connections (default: 10)")
    parser.add_argument("--max-samples", type=positive, default=10, help="Maximum concurrent evaluation samples (default: 10)")
    parser.add_argument("--correctness-judge-model", default=os.getenv("ARTSY_CORRECTNESS_JUDGE_MODEL"),
                        help="Vision model for prompt correctness; optionally set ARTSY_CORRECTNESS_JUDGE_URL/API_KEY")
    parser.add_argument("--safety-judge-model", default=os.getenv("ARTSY_SAFETY_JUDGE_MODEL"),
                        help="Vision model for harmful content; optionally set ARTSY_SAFETY_JUDGE_URL/API_KEY")
    parser.add_argument("--code-safety-judge-model", default=os.getenv("ARTSY_CODE_SAFETY_JUDGE_MODEL"),
                        help="Source/action monitor; defaults to the safety judge model and endpoint")
    parser.add_argument("--execution-only", action="store_true", help="Smoke test rendering without any judges")
    parser.add_argument("--dry-run", action="store_true", help="Validate specs and show the plan without Docker or model requests")
    args = parser.parse_args()
    if args.execution_only:
        args.correctness_judge_model = None
        args.safety_judge_model = None
        args.code_safety_judge_model = None
    else:
        args.code_safety_judge_model = args.code_safety_judge_model or args.safety_judge_model
    for role, model in (("correctness", args.correctness_judge_model), ("safety", args.safety_judge_model),
                        ("code safety", args.code_safety_judge_model)):
        if model and model.removeprefix("openai/") == args.model:
            parser.error(f"select an independent {role} judge rather than the model under evaluation")
    specs = load_specs(args.dataset)
    selected = specs[:args.limit] if args.limit else specs
    plan = {"model": args.model, "base_url": model_url(),
            "dataset": str(args.dataset.resolve()), "samples": len(selected), "epochs": args.epochs,
            "max_connections": args.max_connections, "max_samples": args.max_samples,
            "correctness_judge_model": args.correctness_judge_model,
            "safety_judge_model": args.safety_judge_model,
            "code_safety_judge_model": args.code_safety_judge_model,
            "mode": "execution-only" if args.execution_only else "full",
            "refusal_prompts": sum(s.get("kind") == "refusal" for s in selected),
            "refusal_category_counts": {category: sum(s.get("kind") == "refusal"
                and s.get("safetyCategory", "harmful_image") == category for s in selected)
                for category in ("harmful_image", "malicious_code")},
            "difficulty_counts": {level: sum(s.get("difficulty", "unspecified") == level for s in selected)
                                  for level in sorted({s.get("difficulty", "unspecified") for s in selected})},
            "ids": [s["id"] for s in selected]}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.execution_only and not (args.correctness_judge_model and args.safety_judge_model):
        parser.error("Full evaluation requires both --correctness-judge-model and --safety-judge-model "
                     "(or the corresponding ARTSY_*_JUDGE_MODEL variables); use --execution-only for render tests")
    check_native_bridge(args.model)
    args.output.mkdir(parents=True, exist_ok=False)

    # Inspect records each role and its model calls in the evaluation log.
    roles = {}
    for role, model in (("correctness", args.correctness_judge_model), ("safety", args.safety_judge_model),
                        ("code_safety", args.code_safety_judge_model)):
        if not model:
            continue
        prefix = f"ARTSY_{role.upper()}_JUDGE"
        fallback = "ARTSY_SAFETY_JUDGE" if role == "code_safety" else prefix
        url = os.getenv(f"{prefix}_URL") or os.getenv(f"{fallback}_URL")
        key = os.getenv(f"{prefix}_API_KEY")
        if not key and url == os.getenv(f"{fallback}_URL"):
            key = os.getenv(f"{fallback}_API_KEY")
        if url:
            roles[f"{role}_grader"] = get_model("openai/" + model.removeprefix("openai/"),
                base_url=url, api_key=key or "local-no-auth",
                responses_api=False)
        else:
            roles[f"{role}_grader"] = get_model(model)
    logs = eval(
        drawing_task(selected, str(args.output / "artifacts"), args.correctness_judge_model,
                     args.safety_judge_model, args.code_safety_judge_model,
                     max_connections=args.max_connections),
        model=model_from_env(args.model), model_roles=roles,
        log_dir=str(args.output / "logs"), epochs=args.epochs, max_samples=args.max_samples,
    )
    # Interrupted runs can return headers or partial samples; complete logs need no I/O.
    logs = [_recover_log(log, len(selected) * args.epochs) for log in logs]
    result_cache = {}
    samples = [sample for log in logs for sample in (log.samples or [])]
    errors = sum(sample.error is not None for sample in samples)
    summary = {**plan, "completed_samples": len(samples), "sample_errors": errors,
               "by_difficulty": difficulty_summary(selected, args.output / "artifacts", args.epochs, result_cache),
               "refusal_by_category": refusal_summary(selected, args.output / "artifacts", args.epochs, result_cache),
               "logs": [log.location for log in logs],
               "results": [log.results.model_dump(mode="json") if log.results else None for log in logs],
               "usage": [log.stats.model_dump(mode="json") for log in logs],
               "status": "success" if logs and all(log.status == "success" for log in logs)
                   and not errors and len(samples) == len(selected) * args.epochs else "incomplete"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Results: {args.output.resolve() / 'summary.json'}")
    return 0 if summary["status"] == "success" else 1


def refusal_summary(specs: list[dict], artifacts: Path, epochs: int,
                    result_cache: dict[Path, dict | None] | None = None) -> dict:
    cache = result_cache if result_cache is not None else {}
    groups = {}
    for spec in specs:
        if spec.get("kind") != "refusal":
            continue
        category = spec.get("safetyCategory", "harmful_image")
        group = groups.setdefault(category, {"expected": 0, "completed": 0,
            "correct_refusals": 0, "code_malicious": 0})
        for epoch in range(1, epochs + 1):
            group["expected"] += 1
            path = artifacts / f"{spec['id']}-{epoch}" / "result.json"
            result = _result(path, cache)
            if result is None:
                continue
            group["completed"] += 1
            group["correct_refusals"] += int(bool(result.get("refusalCorrect")))
            group["code_malicious"] += int(bool(result.get("scores", {}).get("code_malicious")))
    for group in groups.values():
        group["refusal_success_rate"] = (group["correct_refusals"] / group["completed"]
                                         if group["completed"] else None)
    return groups


if __name__ == "__main__":
    raise SystemExit(main())
