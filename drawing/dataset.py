"""Checkpointed generation of unfiltered pycairo drawing candidates."""

from __future__ import annotations

from drawing.documentation import api_documentation

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .dataset_client import supports_image_input
from .dataset_specs import compact_json, sha256, validate_specs
from .agent_contract import SFT_MAX_TURNS
from .config import MAX_RESPONSE_TOKENS
from .tool_dataset import candidate as _candidate, FORMAT

DATASET_VERSION = "artsy-dataset-v8-drawing-tools"


def _default_config() -> dict[str, Any]:
    from .environment import ENVIRONMENT_CONFIG
    return dict(ENVIRONMENT_CONFIG)


def _write_json(path: Path, value: Any) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = Path(str(path) + ".tmp")
    text = "".join(compact_json(row) + "\n" for row in rows)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _provenance(client: Any) -> dict[str, Any]:
    result = {"model": client.model, "baseUrl": client.baseUrl}
    if hasattr(client, "reasoning_enabled"):
        result["reasoningEnabled"] = client.reasoning_enabled
    return result


def _pixel_hash(png: bytes) -> str:
    """Hash decoded RGBA pixels, matching PNG.sync.read + width/height in TS."""
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    return sha256(f"{image.width}x{image.height}:".encode() + image.tobytes())


def _refusal_candidate(spec: dict[str, Any], api_doc: str, config: dict[str, Any]) -> dict[str, Any]:
    """A prompt-writer-authored refusal pair; no unsafe program is executed."""
    return {
        "version": DATASET_VERSION,
        "id": f"{spec['id']}-c0",
        "spec": spec,
        "candidate": 0,
        "status": "review",
        "reason": "Authored refusal target; human review pending",
        "refusal": spec["refusal"],
        "generation": {"initialDraft": "authored_refusal"},
        "refusalSource": "prompt_plan",
        "trajectoryFormat": FORMAT,
        "trace": [],
        "evaluations": [],
        "artifacts": {},
        "runtimeVersion": config["runtimeVersion"],
        "apiDocHash": sha256(api_doc),
        "humanReviewed": False,
    }


async def generate_dataset(options: dict[str, Any]) -> dict[str, Any]:
    specs = validate_specs(options["specs"])
    harness = options.get('harness', 'drawing-tools')
    candidate, trajectory_format = _candidate, FORMAT
    if harness == 'mini-swe':
        if any(s.get('kind') == 'refusal' for s in specs):
            raise ValueError('mini-swe has no refusal tool; use drawing-only specs for experiment 5')
        if options.get('executionBackend') == 'local':
            raise ValueError('mini-swe requires a Docker editing workspace')
        from .mini_dataset import candidate, inspect_image
        editing_image = await asyncio.to_thread(inspect_image)
        options = {**options, 'agentImage': editing_image['id']}
        from .mini_agent import FORMAT as trajectory_format
    elif harness != 'drawing-tools':
        raise ValueError('Unknown coding harness')
    teacher_client = options.get("teacher")
    if (teacher_client and any(s.get("kind") != "refusal" for s in specs)
            and not supports_image_input(teacher_client.model, teacher_client.baseUrl)):
        raise ValueError("SFT generation requires a teacher model with image input support")
    if teacher_client and teacher_client.baseUrl == "tinker://native":
        raise ValueError("SFT drawing tools require an HTTP chat endpoint; use the native inference bridge")
    if options.get("draftsFrom"):
        raise ValueError("Legacy JSON drafts cannot be reused in drawing-tool trajectories")
    output = Path(options["output"]).resolve()
    candidates = options.get("candidates", 2)
    max_tokens, workers = options.get("maxTokens", MAX_RESPONSE_TOKENS), options.get("workers", 1)
    if type(workers) is not int or not 1 <= workers <= 128: raise ValueError("workers must be an integer from 1 to 128")
    if type(candidates) is not int or candidates < 1: raise ValueError("candidates must be a positive integer")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1): raise ValueError("maxTokens must be positive when supplied")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".generation.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Dataset is locked: {lock_path}. If the previous process died, verify it is stopped before removing the lock.") from None
    try:
        os.write(fd, compact_json({"pid": os.getpid(), "started": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z")}).encode())
        root = Path(__file__).resolve().parent.parent
        api_doc = api_documentation()
        runtime_hashes = {name: sha256((root / "runtime" / name).read_bytes()) for name in ("requirements.txt",)}
        config = options.get("config") or _default_config()
        teacher = _provenance(options["teacher"]) if options.get("teacher") else None
        manifest: dict[str, Any] = {"version": DATASET_VERSION, "promptVersion": "artsy-specs-v1", "specs": specs,
            "candidates": candidates, "maxTokens": max_tokens, "maxTurns": SFT_MAX_TURNS,
            "trajectoryFormat": trajectory_format,
            "teacher": teacher,
            "config": config, "apiDocHash": sha256(api_doc), "runtimeHashes": runtime_hashes}
        if harness == 'mini-swe':
            from .mini_agent import TOOLS, VERSION, system_prompt, task_prompt
            manifest['harness'] = {'name': 'mini-swe-agent', 'version': VERSION, 'tools': TOOLS,
                                   'editingImage': editing_image,
                                   'prompts': {s['id']: system_prompt(s['seed']) for s in specs},
                                   'taskPrompts': {s['id']: task_prompt(s['prompt']) for s in specs},
                                   'sourceHashes': {name: sha256((root / name).read_bytes()) for name in
                                       ('drawing/mini_agent.py', 'drawing/mini_dataset.py', 'drawing/mini_session.py', 'drawing/mini_protocol.py', 'evaluation/agent_commands.py')}}
        if options.get("executionBackend"):
            manifest["executionBackend"] = options["executionBackend"]
        if options.get("promptGenerator"):
            manifest["promptVersion"] = options["promptGenerator"]["version"]
            manifest["promptGenerator"] = options["promptGenerator"]
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous != manifest:
                uncapped_previous = {**previous, "maxTokens": None} if isinstance(previous, dict) else None
                if manifest["maxTokens"] is None and uncapped_previous == manifest:
                    _write_json(manifest_path, manifest)
                else:
                    raise ValueError("Dataset configuration changed; use a new output directory")
        else: _write_json(manifest_path, manifest)
        (output / "prompts.jsonl").write_text("".join(compact_json(s) + "\n" for s in specs), encoding="utf-8")
        records_path = output / "records"; records_path.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []

        async def process_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
            saved_records = []
            candidate_count = 1 if spec.get("kind") == "refusal" else candidates
            for index in range(candidate_count):
                path = records_path / f"{spec['id']}-c{index}.json"
                saved_records.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
            out: list[dict[str, Any]] = []
            for index, saved in enumerate(saved_records):
                filename = f"{spec['id']}-c{index}.json"
                retry = saved and saved.get("status") == "infrastructure_error" and options.get("retryErrors")
                candidate_options = {**options, "output": str(output), "maxTokens": max_tokens}
                if retry and harness == 'mini-swe':
                    from .mini_dataset import resume_options
                    recovery = resume_options(saved, output)
                    if recovery:
                        candidate_options['resume'] = recovery
                record = saved if saved and not retry else (
                    _refusal_candidate(spec, api_doc, config) if spec.get("kind") == "refusal"
                    else await candidate(candidate_options, spec, index, api_doc)
                )
                if retry:
                    record["previousAttempts"] = [*(saved.get("previousAttempts", [])), {k: v for k, v in saved.items() if k != "previousAttempts"}]
                if record.get("id") != f"{spec['id']}-c{index}" or record.get("spec") != spec:
                    raise ValueError(f"Invalid checkpoint: {filename}")
                _write_json(records_path / filename, record)
                out.append(record)
                callback = options.get("onRecord")
                if callback: callback(record)
            return out

        pending_specs = iter(specs)

        async def worker() -> None:
            # Each worker claims its next prompt immediately, without waiting
            # for slower trajectories assigned to other workers.
            for spec in pending_specs:
                records.extend(await process_spec(spec))

        tasks = [asyncio.create_task(worker()) for _ in range(min(workers, len(specs)))]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # Drain workers before releasing the dataset lock on failure/cancel.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        order = {s["id"]: i for i, s in enumerate(specs)}
        records.sort(key=lambda r: (order[r["spec"]["id"]], r["candidate"]))
        # Keep every candidate, including failed runs and teacher refusals.
        # Human review determines which complete trajectories to export for SFT.
        _write_jsonl_atomic(output / "candidates.jsonl", records)
        completions = []
        for record in records:
            for row in [record, *record.get("previousAttempts", [])]:
                if (row is not record and record.get('generation', {}).get('continuation')
                        and row.get('trace') == record.get('trace', [])[:len(row.get('trace', []))]):
                    continue  # The resumed trace already contains these requests.
                completions.extend(t["completion"] for t in row.get("trace", []) if not t.get("reusedDraft"))
                if row.get("submission", {}).get("completion"):
                    completions.append(row["submission"]["completion"])
        count = len(records)
        summary = {"version": DATASET_VERSION, "prompts": len(specs), "candidates": count,
            "accepted": sum(r["status"] == "accepted" for r in records), "review": sum(r["status"] == "review" for r in records),
            "refusalPairs": sum(r["spec"].get("kind") == "refusal" and r["status"] in {"accepted", "review"} for r in records),
            "maliciousCodeRefusalPairs": sum(r["spec"].get("safetyCategory") == "malicious_code"
                                             and r["status"] in {"accepted", "review"} for r in records),
            "rejected": sum(r["status"] == "rejected" for r in records), "infrastructureErrors": sum(r["status"] == "infrastructure_error" for r in records),
            "requests": len(completions), "promptTokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in completions),
            "completionTokens": sum(c.get("usage", {}).get("completion_tokens", 0) for c in completions),
            "requestsWithoutUsage": sum("usage" not in c for c in completions),
            "reusedDrafts": sum(any(t.get("reusedDraft") for t in r.get("trace", [])) for r in records),
            "acceptanceRate": sum(r["status"] == "accepted" for r in records) / count if count else float("nan")}
        _write_json(output / "summary.json", summary)
        return summary
    finally:
        os.close(fd)
        try: lock_path.unlink()
        except FileNotFoundError: pass
