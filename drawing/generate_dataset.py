"""Command line entry point for LLM-authored drawing datasets."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

HELP = """Python dataset generation
  --output DIR          Dataset directory (default: datasets/pilot)
  --count N             Structured prompts to generate (default: 20)
  --seed N              Taxonomy/slot seed (default: 42; prompt text is model-generated)
  --input FILE          Custom JSONL specs instead of the built-in curriculum
  --curriculum MODE     standard (default, 25% repair-oriented) or repair (all challenging)
  --pool-size N         Hard-task pool for repair/warmup, 16–512 (default: 128)
  --natural-repairs-from DIR  Reuse verified natural repairs from a previous run
  --drafts-from DIR     Reuse unmodified model drafts (repeatable)
  --image-observations  Inspect actual PNGs and include them in visual repairs
  --warmup-from DIR     Retain 80 one-shot + 4 repairs from a pilot; generate 16 repairs
  --candidates N        Programs per prompt (default: 2; warmup: 1)
  --workers N           Independent repair tasks at once, 1–8 (warmup default: 4)
  --repairs N           Repairs per candidate, 0–2 (default: 2)
  --max-tokens N        Teacher tokens per request (default: 4096; repair: 8192)
  --threshold N         Minimum score for EVERY requirement (default: 0.85)
  --prompt-model ID     Prompt-writing LLM (defaults to teacher model)
  --prompt-url URL      Prompt-writing endpoint (defaults to teacher URL)
  --prompt-batch-size N Prompts per LLM request, 1–8 (default: 5)
  --prompt-max-tokens N Prompt-writer output tokens per request (default: 4096)
  --prompt-retries N    Validation attempts per batch, 1–5 (default: 3)
  --teacher-model ID    Required for rendering, or set ARTSY_TEACHER_MODEL
  --teacher-url URL     Chat base URL (default: http://127.0.0.1:8080/v1)
  --judge-model ID      Optional vision evaluator, or ARTSY_JUDGE_MODEL
  --judge-url URL       Evaluator base URL (defaults to teacher URL)
  --timeout-ms N        Per-request deadline (default: 60000)
  --dry-run             LLM-generate prompt JSONL (no rendering, no split labels)
  --retry-errors        Retry checkpointed infrastructure failures on resume
  --retry-rejected      Retry rejected repair candidates for still-missing tasks
  --help, -h            Show this help

Credentials: ARTSY_PROMPT_API_KEY, ARTSY_TEACHER_API_KEY, ARTSY_JUDGE_API_KEY.
No judge: executable examples go to review.jsonl; accepted/train/validation/test
remain empty. Repeating the same command resumes completed candidate records.
Generated Python runs locally with your current permissions, as with the drawing CLI.
The prompt plan is saved before rendering and reused on resume. The legacy
--warmup-from selection remains frozen to preserve existing training examples."""


def _load_env_file(path: Path = Path(".env")) -> None:
    """Load simple dotenv assignments without evaluating shell syntax."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not (key[0].isalpha() or key[0] == "_") or not all(c.isalnum() or c == "_" for c in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, usage=HELP.splitlines()[0])
    parser.add_argument("--output")
    parser.add_argument("--count")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--input")
    parser.add_argument("--curriculum", default="standard")
    parser.add_argument("--warmup-from")
    parser.add_argument("--workers")
    parser.add_argument("--pool-size", default="128")
    parser.add_argument("--natural-repairs-from")
    parser.add_argument("--drafts-from", action="append")
    parser.add_argument("--image-observations", action="store_true", default=None)
    parser.add_argument("--candidates")
    parser.add_argument("--repairs", default="2")
    parser.add_argument("--max-tokens")
    parser.add_argument("--threshold", default="0.85")
    parser.add_argument("--prompt-model")
    parser.add_argument("--prompt-url")
    parser.add_argument("--prompt-batch-size", default="5")
    parser.add_argument("--prompt-max-tokens", default="4096")
    parser.add_argument("--prompt-retries", default="3")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-url")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-url")
    parser.add_argument("--timeout-ms", default="60000")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    return parser


def _read_specs(path: str) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on input line {number}") from error
    return rows


async def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.help:
        print(HELP)
        return 0
    _load_env_file()
    from .dataset_prompts import compact_json, sha256, validate_specs
    from .llm_prompts import GENERATOR_VERSION, generate_llm_prompts
    from .warmup_dataset import generate_warmup_dataset, warmup_plan
    from .dataset_client import chat_client

    warmup_source = args.warmup_from
    repair_mode = bool(warmup_source) or args.curriculum == "repair"
    if args.curriculum not in ("standard", "repair"):
        raise ValueError("curriculum must be standard or repair")
    if args.input and repair_mode:
        raise ValueError("--input cannot be combined with the built-in repair/warmup curriculum")
    if repair_mode and args.count:
        raise ValueError("Use --pool-size for the repair prompt pool; the warmup selects 16 repairs")
    seed, pool_size = int(args.seed), int(args.pool_size)
    if repair_mode and not 16 <= pool_size <= 512:
        raise ValueError("pool-size must be between 16 and 512")
    count = pool_size if repair_mode else int(args.count or 20)
    output = args.output or ("datasets/sft-warmup-100" if warmup_source else "datasets/pilot")
    plan = await warmup_plan(warmup_source, seed, pool_size) if warmup_source else None

    source_manifest = json.loads((Path(warmup_source) / "manifest.json").read_text(encoding="utf-8")) if warmup_source else {}
    model = args.teacher_model or os.getenv("ARTSY_TEACHER_MODEL") or (source_manifest.get("teacher") or {}).get("model")
    base_url = args.teacher_url or os.getenv("ARTSY_TEACHER_URL") or (source_manifest.get("teacher") or {}).get("baseUrl") or "http://127.0.0.1:8080/v1"
    prompt_model = args.prompt_model or os.getenv("ARTSY_PROMPT_MODEL") or model
    prompt_url = args.prompt_url or os.getenv("ARTSY_PROMPT_URL") or base_url
    prompt_config = {
        "version": GENERATOR_VERSION, "mode": "repair" if repair_mode else "standard",
        "count": count, "seed": seed, "model": prompt_model, "baseUrl": prompt_url,
        "batchSize": int(args.prompt_batch_size), "maxTokens": int(args.prompt_max_tokens),
        "retries": int(args.prompt_retries),
    }
    prompt_generation = None
    if plan:
        specs = plan["specs"]
    elif args.input:
        specs = validate_specs(_read_specs(args.input))
    else:
        if not prompt_model:
            raise ValueError("Set --prompt-model or ARTSY_PROMPT_MODEL (or a teacher model) to generate prompts")
        plan_path = Path(output) / "prompt-plan.json"
        if not args.dry_run and plan_path.exists():
            saved = json.loads(plan_path.read_text(encoding="utf-8"))
            if saved.get("config") != prompt_config:
                raise ValueError("Prompt generator configuration changed; use a new output directory")
            specs = validate_specs(saved["specs"])
            if len(specs) != count or sha256(compact_json(specs)) != saved.get("specsHash"):
                raise ValueError("Stored prompt plan was modified; use a new output directory")
        else:
            if not args.dry_run and (Path(output) / "manifest.json").exists():
                raise ValueError("Existing dataset has no LLM prompt plan; use a new output directory")
            prompt_key = os.getenv("ARTSY_PROMPT_API_KEY")
            if prompt_url == base_url and not prompt_key:
                prompt_key = os.getenv("ARTSY_TEACHER_API_KEY") or os.getenv("TEACHER_API_KEY")
            prompt_client = chat_client({"model": prompt_model, "baseUrl": prompt_url,
                "apiKey": prompt_key, "timeoutMs": int(args.timeout_ms)})
            generated = await generate_llm_prompts(prompt_client, count, seed,
                mode=prompt_config["mode"], batch_size=prompt_config["batchSize"],
                max_tokens=prompt_config["maxTokens"], retries=prompt_config["retries"])
            specs = generated["specs"]
            if not args.dry_run:
                Path(output).mkdir(parents=True, exist_ok=True)
                from .dataset import _write_json
                _write_json(plan_path, {"config": prompt_config, "specs": specs,
                    "specsHash": sha256(compact_json(specs)), "batches": generated["batches"]})
        prompt_generation = {**prompt_config, "specsHash": sha256(compact_json(specs))}
    if args.dry_run:
        if plan:
            print(json.dumps({"version": plan["version"],
                "retained": [{"id": x["record"]["id"], "kind": x["kind"], "sourceSplit": x["record"]["spec"]["split"]} for x in plan["selected"]],
                "newRepairPrompts": specs, "target": {"oneShot": 80, "repairs": 20, "total": 100}}, indent=2))
        else:
            for spec in specs:
                row = {key: value for key, value in spec.items() if key != "split"}
                print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        return 0

    natural_repairs_from = args.natural_repairs_from
    drafts_from = args.drafts_from
    if warmup_source and not natural_repairs_from:
        try:
            previous = json.loads((Path(output) / "warmup-manifest.json").read_text(encoding="utf-8"))
            source_dataset = (previous.get("importedNaturalRepairs") or {}).get("sourceDataset")
            if source_dataset:
                natural_repairs_from = str((Path(output) / source_dataset).resolve())
        except FileNotFoundError:
            pass
    if warmup_source and drafts_from is None:
        try:
            previous = json.loads((Path(output) / "generated" / "manifest.json").read_text(encoding="utf-8"))
            drafts_from = [str((Path(output) / "generated" / s["directory"]).resolve()) for s in previous.get("draftSources", [])]
        except FileNotFoundError:
            pass
    if not model:
        raise ValueError("Set --teacher-model or ARTSY_TEACHER_MODEL to your stronger teacher model")
    image_observations = args.image_observations if args.image_observations is not None else bool(warmup_source)
    judge_model = args.judge_model or os.getenv("ARTSY_JUDGE_MODEL") or (model if image_observations else None)
    judge_url = args.judge_url or os.getenv("ARTSY_JUDGE_URL") or base_url
    teacher_key = os.getenv("ARTSY_TEACHER_API_KEY") or os.getenv("TEACHER_API_KEY")
    timeout_ms = int(args.timeout_ms)
    if timeout_ms < 1 or timeout_ms > 300_000:
        raise ValueError("timeout-ms must be an integer from 1 to 300000")
    teacher = chat_client({"model": model, "baseUrl": base_url, "apiKey": teacher_key, "timeoutMs": timeout_ms})
    judge = chat_client({"model": judge_model, "baseUrl": judge_url,
        "apiKey": os.getenv("ARTSY_JUDGE_API_KEY") or (teacher_key if judge_url == base_url else None), "timeoutMs": timeout_ms}) if judge_model else None
    options = {"output": output, "candidates": int(args.candidates or (1 if warmup_source else 2)),
        "repairs": int(args.repairs), "maxTokens": int(args.max_tokens or (8192 if repair_mode else 4096)),
        "threshold": float(args.threshold), "workers": int(args.workers or (4 if warmup_source else 1)),
        "retryErrors": args.retry_errors, "retryRejected": args.retry_rejected, "imageObservations": image_observations,
        "draftsFrom": drafts_from, "teacher": teacher, "judge": judge,
        "promptGenerator": prompt_generation,
        "onRecord": lambda record: print(f"{record['id']} {record['status']}: {record['reason']}", file=sys.stderr)}
    if warmup_source:
        summary = await generate_warmup_dataset({**options, "source": warmup_source, "seed": seed,
            "poolSize": pool_size, "naturalRepairsFrom": natural_repairs_from})
    else:
        from .dataset import generate_dataset
        summary = await generate_dataset({**options, "specs": specs, "mode": "repair" if repair_mode else "standard"})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("complete") is False or summary.get("infrastructureErrors"):
        return 1
    return 0


def cli() -> None:
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
