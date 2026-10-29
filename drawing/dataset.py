"""Checkpointed generation of unfiltered ArtCanvas drawing candidates."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from .dataset_client import Message, parse_json_reply, supports_image_input
from .dataset_specs import compact_json, sha256, validate_specs

DATASET_VERSION = "artsy-dataset-v7"


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


def _provenance(client: Any) -> dict[str, str]:
    return {"model": client.model, "baseUrl": client.baseUrl}


def _parse_target(text: str) -> dict[str, Any]:
    target = parse_json_reply(text)
    if isinstance(target, dict) and set(target) == {"submit"} and target["submit"] is True:
        return target
    if not isinstance(target, dict):
        raise ValueError("Return one JSON object with code or refusal")
    if len(target) != 1:
        raise ValueError("Return exactly one nonempty string field: code or refusal")
    field, value = next(iter(target.items()))
    if field not in ("code", "refusal") or not isinstance(value, str) or not value.strip():
        raise ValueError("Return exactly one nonempty string field: code or refusal")
    return target


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
        "trace": [],
        "evaluations": [],
        "artifacts": {},
        "runtimeVersion": config["runtimeVersion"],
        "apiDocHash": sha256(api_doc),
        "humanReviewed": False,
    }


async def _candidate(options: dict[str, Any], spec: dict[str, Any], index: int, api_doc: str) -> dict[str, Any]:
    from .environment import DrawingEnvironment
    output = Path(options["output"]).resolve()
    config = options.get("config") or _default_config()
    env = DrawingEnvironment(str(output / "artifacts"), renderer=options.get("renderer"), config=config)
    episode = await env.reset(spec["prompt"], spec["seed"])
    episode_dir = Path(episode["directory"])
    rel = lambda p: os.path.relpath(str(p), str(output))
    record: dict[str, Any] = {
        "version": DATASET_VERSION, "id": f"{spec['id']}-c{index}", "spec": spec, "candidate": index,
        "status": "rejected", "reason": "Model turn budget exhausted",
        "artifacts": {"directory": rel(episode_dir), "trajectory": rel(episode_dir / "trajectory.jsonl")},
        "trace": [], "evaluations": [], "teacher": _provenance(options["teacher"]),
        "runtimeVersion": config["runtimeVersion"], "apiDocHash": sha256(api_doc), "humanReviewed": False,
    }
    record["generation"] = {"initialDraft": "natural", "executionFailures": 0, "imagePreviews": True}

    messages: list[Message] = [
        {"role": "system", "content": ("Write an editable Python/ArtCanvas program for the drawing request. Return ONLY JSON with exactly one string field: "
          "{\"code\":\"complete Python source\"}, or {\"refusal\":\"concise reason\"} for a request you cannot fulfill. "
          "After each successful render you will receive the actual PNG preview. Inspect it against the original request. "
          "Return complete revised code to fix visible mistakes, or {\"submit\":true} only after inspecting a satisfactory preview. "
          "Import ArtCanvas from artcanvas, use only its documented drawing API plus math/random, and save output.png in the working directory. "
          "Do not load files, use external images, access the network, or execute commands. Canvas size must be "
          f"{config['width']}x{config['height']}. The renderer seeds random with {spec['seed']}. "
          f"Budget: {config['maxCalls']} environment operations and {config['maxPolicyTokens']} generated tokens. "
          f"Source limit: {config['sourceBytes']} bytes; "
          f"render timeout: {config['renderSeconds']} seconds.\n\n{api_doc}")},
        {"role": "user", "content": compact_json({"prompt": spec["prompt"], "candidate": index})},
    ]
    reused = None
    for directory in options.get("draftsFrom") or []:
        path = Path(directory) / "records" / f"{spec['id']}-c{index}.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        saved_spec = {k: v for k, v in (saved.get("spec") or {}).items() if k not in ("requirements", "expectedOutcome")}
        if (saved.get("generation", {}).get("initialDraft") != "natural" or saved_spec != spec
                or saved.get("teacher") != _provenance(options["teacher"])):
            raise ValueError("Draft source is not a matching natural teacher trajectory")
        first = next((t for t in saved.get("trace", []) if "executedCode" in t), None)
        if first:
            parsed = parse_json_reply(first["completion"]["text"])
            if parsed.get("code") != first["executedCode"]:
                raise ValueError(f"Draft was modified: {saved['id']}")
            reused = {"completion": first["completion"], "dataset": os.path.relpath(Path(directory).resolve(), output), "attempt": first["attempt"]}
            break

    last_preview = None
    try:
        # Bound malformed replies too; code repairs have no separate attempt cap.
        for attempt in range(config["maxCalls"]):
            completion = reused["completion"] if attempt == 0 and reused else await options["teacher"].complete(messages, options.get("maxTokens", 4096))
            trace: dict[str, Any] = {"attempt": attempt, "completion": completion}
            if attempt == 0 and reused:
                trace["reusedDraft"] = {"dataset": reused["dataset"], "recordId": f"{spec['id']}-c{index}", "attempt": reused["attempt"]}
            record["trace"].append(trace)
            messages.append({"role": "assistant", "content": completion["text"]})
            if not completion.get("usage"):
                trace["error"] = "Token usage unavailable; completion tokens could not be charged"
            charged = await env.charge_tokens(completion.get("usage", {}).get("completion_tokens", 0))
            if not charged.get("ok"):
                record["reason"] = "Episode token budget exhausted"; break
            try:
                if completion.get("finishReason") == "length": raise ValueError("Teacher completion was truncated")
                target = _parse_target(completion["text"])
            except Exception as exc:
                trace["error"] = str(exc); record["reason"] = str(exc)
                messages.append({"role": "user", "content": f"{exc}. Return the complete corrected JSON object."})
                continue
            if "submit" in target:
                if last_preview is None:
                    trace["error"] = "Render and inspect a successful preview before submitting"
                    messages.append({"role": "user", "content": trace["error"] + ". Return complete code as JSON."})
                    continue
                record["trace"].pop()
                feedback = await env.submit()
                record["submission"] = {"completion": completion, "feedback": feedback,
                                        "preview": last_preview}
                if not feedback.get("ok"):
                    record["reason"] = feedback.get("error") or "Fresh submission failed"
                    if feedback.get("failureKind") == "infrastructure":
                        record["status"] = "infrastructure_error"
                    break
                record["artifacts"].update({"image": rel(feedback["artifact"]), "source": rel(episode_dir / "program.py")})
                image = Path(feedback["artifact"]).read_bytes()
                record["sourceHash"] = sha256(re.sub(r"\r\n", "\n", record["code"]).strip())
                record["imageHash"] = sha256(image)
                record["pixelHash"] = _pixel_hash(image)
                record.update(status="review", reason="Preview inspected by teacher; human review pending")
                break
            if "refusal" in target:
                record["refusal"] = target["refusal"]
                feedback = await env.refuse(target["refusal"]); trace["feedback"] = feedback
                if not feedback.get("ok"):
                    record["reason"] = feedback.get("error") or feedback.get("outcome"); break
                record.update(status="review", reason="Teacher refusal; human review pending")
                break
            last_preview = None
            record["code"] = target["code"]
            trace["executedCode"] = target["code"]
            feedback = await env.write_code(target["code"])
            if feedback.get("ok"): feedback = await env.run_code()
            trace["feedback"] = feedback
            if feedback.get("failureKind") == "infrastructure":
                record.update(status="infrastructure_error", reason=feedback.get("error") or "Renderer infrastructure failure"); break
            if not feedback.get("ok"):
                if feedback.get("failureKind") == "invalid_code":
                    record["generation"]["executionFailures"] += 1
                record["reason"] = feedback.get("error") or feedback.get("stderr") or feedback.get("outcome")
                messages.append({"role": "user", "content": f"Execution feedback: {compact_json(feedback)}. Return a complete corrected program as JSON."})
                if feedback.get("outcome") != "active": break
                continue
            image = Path(feedback["artifact"]).read_bytes()
            last_preview = {"image": rel(feedback["artifact"]), "imageHash": sha256(image)}
            trace["previewObservation"] = last_preview
            record["reason"] = "Preview review budget exhausted without teacher submission"
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Inspect your rendered preview against the original request. "
                 "Return {\"submit\":true} if it is ready, or complete corrected code as JSON. "
                 f"Execution feedback: {compact_json(feedback)}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}},
            ]})
    except Exception as exc:
        record.update(status="infrastructure_error", reason=str(exc))
    return record


async def generate_dataset(options: dict[str, Any]) -> dict[str, Any]:
    specs = validate_specs(options["specs"])
    teacher_client = options.get("teacher")
    if (teacher_client and any(s.get("kind") != "refusal" for s in specs)
            and not supports_image_input(teacher_client.model, teacher_client.baseUrl)):
        raise ValueError("SFT generation requires a teacher model with image input support")
    output = Path(options["output"]).resolve()
    candidates = options.get("candidates", 2)
    max_tokens, workers = options.get("maxTokens", 4096), options.get("workers", 1)
    if type(workers) is not int or not 1 <= workers <= 8: raise ValueError("workers must be an integer from 1 to 8")
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
        api_doc = (root / "docs" / "artcanvas.md").read_text(encoding="utf-8")
        runtime_hashes = {name: sha256((root / "runtime" / name).read_bytes()) for name in ("artcanvas.py", "cairo.py", "requirements.txt")}
        draft_sources = []
        for directory in options.get("draftsFrom") or []:
            names = sorted(p.name for p in (Path(directory) / "records").iterdir() if p.name.endswith(".json"))
            hashes = [[name, sha256((Path(directory) / "records" / name).read_bytes())] for name in names]
            draft_sources.append({"directory": os.path.relpath(Path(directory).resolve(), output), "recordsHash": sha256(compact_json(hashes))})
        config = options.get("config") or _default_config()
        teacher = _provenance(options["teacher"]) if options.get("teacher") else None
        manifest: dict[str, Any] = {"version": DATASET_VERSION, "promptVersion": "artsy-specs-v1", "specs": specs,
            "candidates": candidates, "maxTokens": max_tokens,
            "teacher": teacher,
            "config": config, "apiDocHash": sha256(api_doc), "runtimeHashes": runtime_hashes}
        if options.get("promptGenerator"):
            manifest["promptVersion"] = options["promptGenerator"]["version"]
            manifest["promptGenerator"] = options["promptGenerator"]
        if draft_sources: manifest["draftSources"] = draft_sources
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
                record = saved if saved and not retry else (
                    _refusal_candidate(spec, api_doc, config) if spec.get("kind") == "refusal"
                    else await _candidate({**options, "output": str(output)}, spec, index, api_doc)
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

        for offset in range(0, len(specs), workers):
            # Parallelism is bounded by independent specifications, matching TS batching.
            batch = await asyncio.gather(*(process_spec(s) for s in specs[offset:offset + workers]), return_exceptions=True)
            error = next((result for result in batch if isinstance(result, BaseException)), None)
            if error: raise error
            for group in batch: records.extend(group)
        order = {s["id"]: i for i, s in enumerate(specs)}
        records.sort(key=lambda r: (order[r["spec"]["id"]], r["candidate"]))
        # Keep every candidate, including failed runs and teacher refusals.
        # Human review determines which complete trajectories to export for SFT.
        _write_jsonl_atomic(output / "candidates.jsonl", records)
        completions = []
        for record in records:
            for row in [record, *record.get("previousAttempts", [])]:
                completions.extend(t["completion"] for t in row.get("trace", []) if not t.get("reusedDraft"))
                if row.get("submission"):
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
