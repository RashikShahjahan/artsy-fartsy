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
  --count N             Total number of prompts (default: 20)
  --harmful-count N     Refusal-worthy prompt/target pairs within the total (default: 0)
  --difficulty LEVEL    mixed (default), easy, medium, hard, or very-hard
  --seed N              Taxonomy/slot seed (default: 42; prompt text is model-generated)
  --input FILE          Custom JSONL specs instead of the built-in curriculum
  --drafts-from DIR     Reuse unmodified model drafts (repeatable)
  --image-observations  Inspect actual PNGs and include them in visual repairs
  --candidates N        Programs per prompt (default: 2)
  --workers N           Independent prompts processed concurrently, 1–8
  --repairs N           Repairs per candidate, 0–2 (default: 2)
  --max-tokens N        Teacher tokens per request (default: 4096; repair: 8192)
  --threshold N         Minimum score for EVERY requirement (default: 0.85)
  --prompt-model ID     Prompt-writing LLM (defaults to teacher model)
  --prompt-url URL      Prompt-writing endpoint (defaults to teacher URL)
  --prompt-batch-size N Concurrent single-prompt LLM requests, 1–8 (default: 8)
  --prompt-max-tokens N Prompt-writer output tokens per request (default: 4096)
  --prompt-retries N    Attempts per individual prompt, 1–5 (default: 5)
  --teacher-model ID    Required for rendering, or set ARTSY_TEACHER_MODEL
  --teacher-url URL     Chat base URL (default: http://127.0.0.1:8080/v1)
  --correctness-judge-model ID  Vision correctness judge, or ARTSY_CORRECTNESS_JUDGE_MODEL
  --correctness-judge-url URL   Correctness endpoint (defaults to teacher URL)
  --safety-judge-model ID       Vision safety judge, or ARTSY_SAFETY_JUDGE_MODEL
  --safety-judge-url URL        Safety endpoint (defaults to teacher URL)
  --timeout-ms N        Per-request deadline (default: 60000)
  --dry-run             LLM-generate prompt JSONL (no rendering, no split labels)
  --retry-errors        Retry checkpointed infrastructure failures on resume
  --help, -h            Show this help

Credentials: ARTSY_PROMPT_API_KEY, ARTSY_TEACHER_API_KEY, ARTSY_CORRECTNESS_JUDGE_API_KEY, ARTSY_SAFETY_JUDGE_API_KEY.
Both vision judges are required for automatic acceptance. No judges: executable examples go to review.jsonl; accepted/train/validation/test
remain empty. Repeating the same command resumes checkpointed prompts and candidate records.
Generated Python runs locally with your current permissions, as with the drawing CLI.
The prompt plan is saved before rendering and reused on resume."""


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
    parser.add_argument("--harmful-count", default="0")
    parser.add_argument("--difficulty", default="mixed")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--input")
    parser.add_argument("--workers")
    parser.add_argument("--drafts-from", action="append")
    parser.add_argument("--image-observations", action="store_true", default=None)
    parser.add_argument("--candidates")
    parser.add_argument("--repairs", default="2")
    parser.add_argument("--max-tokens")
    parser.add_argument("--threshold", default="0.85")
    parser.add_argument("--prompt-model")
    parser.add_argument("--prompt-url")
    parser.add_argument("--prompt-batch-size", default="8")
    parser.add_argument("--prompt-max-tokens", default="4096")
    parser.add_argument("--prompt-retries", default="5")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-url")
    parser.add_argument("--correctness-judge-model")
    parser.add_argument("--correctness-judge-url")
    parser.add_argument("--safety-judge-model")
    parser.add_argument("--safety-judge-url")
    parser.add_argument("--timeout-ms", default="60000")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
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
    from .dataset_specs import compact_json, sha256, validate_specs
    from .llm_prompts import GENERATOR_VERSION, generate_llm_prompts
    from .dataset_client import chat_client
    if args.difficulty not in ("mixed", "easy", "medium", "hard", "very-hard"):
        raise ValueError("difficulty must be mixed, easy, medium, hard, or very-hard")
    seed = int(args.seed)
    count = int(args.count) if args.count is not None else 20
    harmful_count = int(args.harmful_count)
    if count < 1:
        raise ValueError("count must be positive")
    if not 0 <= harmful_count <= count:
        raise ValueError("--harmful-count must be between zero and --count")
    output = args.output or "datasets/pilot"
    model = args.teacher_model or os.getenv("ARTSY_TEACHER_MODEL")
    base_url = args.teacher_url or os.getenv("ARTSY_TEACHER_URL") or "http://127.0.0.1:8080/v1"
    prompt_model = args.prompt_model or os.getenv("ARTSY_PROMPT_MODEL") or model
    prompt_url = args.prompt_url or os.getenv("ARTSY_PROMPT_URL") or base_url
    prompt_config = {
        "version": GENERATOR_VERSION,
        "count": count, "harmfulCount": harmful_count, "difficulty": args.difficulty,
        "seed": seed, "model": prompt_model, "baseUrl": prompt_url,
        "batchSize": int(args.prompt_batch_size), "maxTokens": int(args.prompt_max_tokens),
        "retries": int(args.prompt_retries),
    }
    prompt_generation = None
    if args.input:
        if harmful_count:
            raise ValueError("--harmful-count cannot be used with --input; include refusal pairs in the input file")
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
            completed = {}
            on_record = None
            lock_path = None
            lock_fd = None
            try:
                if not args.dry_run:
                    from .dataset import _write_json
                    dataset_path = Path(output)
                    dataset_path.mkdir(parents=True, exist_ok=True)
                    lock_path = dataset_path / ".prompt-generation.lock"
                    try:
                        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    except FileExistsError:
                        raise RuntimeError(f"Prompt generation is already locked: {lock_path}") from None
                    progress_path = dataset_path / "prompt-progress.json"
                    records_dir = dataset_path / "prompt-records"
                    if progress_path.exists():
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                        if progress.get("config") != prompt_config:
                            raise ValueError("Prompt generator configuration changed; use a new output directory")
                    else:
                        if records_dir.exists() and any(records_dir.iterdir()):
                            raise ValueError("Prompt records exist without a progress manifest")
                        _write_json(progress_path, {"config": prompt_config})
                    records_dir.mkdir(parents=True, exist_ok=True)
                    for path in sorted(records_dir.iterdir()):
                        if path.suffix != ".json" or not path.stem.isdecimal() or len(path.stem) != 6:
                            raise ValueError(f"Unexpected prompt checkpoint file: {path.name}")
                        record = json.loads(path.read_text(encoding="utf-8"))
                        index = int(path.stem)
                        if record.get("index") != index or index in completed:
                            raise ValueError(f"Invalid prompt checkpoint: {path.name}")
                        completed[index] = record

                    def save_prompt_record(record: dict[str, Any]) -> None:
                        _write_json(records_dir / f"{record['index']:06d}.json", record)
                    on_record = save_prompt_record

                generated = await generate_llm_prompts(prompt_client, count, seed,
                    difficulty=args.difficulty, harmful_count=harmful_count,
                    batch_size=prompt_config["batchSize"],
                    max_tokens=prompt_config["maxTokens"], retries=prompt_config["retries"],
                    completed=completed, on_record=on_record)
                specs = generated["specs"]
                if not args.dry_run:
                    _write_json(plan_path, {"config": prompt_config, "specs": specs,
                        "specsHash": sha256(compact_json(specs)), "batches": generated["batches"]})
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)
                    lock_path.unlink(missing_ok=True)
        prompt_generation = {**prompt_config, "specsHash": sha256(compact_json(specs))}
    if args.dry_run:
        for spec in specs:
            row = {key: value for key, value in spec.items() if key != "split"}
            print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        return 0
    if not model and any(spec.get("kind") != "refusal" for spec in specs):
        raise ValueError("Set --teacher-model or ARTSY_TEACHER_MODEL for drawing solutions")
    image_observations = bool(args.image_observations)
    correctness_model = args.correctness_judge_model or os.getenv("ARTSY_CORRECTNESS_JUDGE_MODEL")
    safety_model = args.safety_judge_model or os.getenv("ARTSY_SAFETY_JUDGE_MODEL")
    if bool(correctness_model) != bool(safety_model):
        raise ValueError("Set both correctness and safety judge models, or neither")
    if image_observations and not correctness_model:
        raise ValueError("--image-observations requires both vision judges")
    correctness_url = args.correctness_judge_url or os.getenv("ARTSY_CORRECTNESS_JUDGE_URL") or base_url
    safety_url = args.safety_judge_url or os.getenv("ARTSY_SAFETY_JUDGE_URL") or base_url
    for role, judge_model, judge_url in (("correctness", correctness_model, correctness_url),
                                          ("safety", safety_model, safety_url)):
        if judge_model and judge_model.removeprefix("openai/") == model and judge_url == base_url:
            raise ValueError(f"Select an independent {role} judge instead of the teacher model")
    teacher_key = os.getenv("ARTSY_TEACHER_API_KEY") or os.getenv("TEACHER_API_KEY")
    timeout_ms = int(args.timeout_ms)
    if timeout_ms < 1 or timeout_ms > 300_000:
        raise ValueError("timeout-ms must be an integer from 1 to 300000")
    teacher = chat_client({"model": model, "baseUrl": base_url, "apiKey": teacher_key, "timeoutMs": timeout_ms}) if model else None
    correctness_judge = chat_client({
        "model": correctness_model, "baseUrl": correctness_url,
        "apiKey": os.getenv("ARTSY_CORRECTNESS_JUDGE_API_KEY")
                  or (teacher_key if correctness_url == base_url else None),
        "timeoutMs": timeout_ms,
    }) if correctness_model else None
    safety_judge = chat_client({
        "model": safety_model, "baseUrl": safety_url,
        "apiKey": os.getenv("ARTSY_SAFETY_JUDGE_API_KEY")
                  or (teacher_key if safety_url == base_url else None),
        "timeoutMs": timeout_ms,
    }) if safety_model else None
    options = {"output": output, "candidates": int(args.candidates or 2),
        "repairs": int(args.repairs), "maxTokens": int(args.max_tokens or 4096),
        "threshold": float(args.threshold), "workers": int(args.workers or 1),
        "retryErrors": args.retry_errors, "imageObservations": image_observations,
        "draftsFrom": args.drafts_from, "teacher": teacher,
        "correctnessJudge": correctness_judge, "safetyJudge": safety_judge,
        "promptGenerator": prompt_generation,
        "onRecord": lambda record: print(f"{record['id']} {record['status']}: {record['reason']}", file=sys.stderr)}
    from .dataset import generate_dataset
    summary = await generate_dataset({**options, "specs": specs})
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
