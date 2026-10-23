"""Select, generate, and export portable 80/20 warmup datasets."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .dataset import EVALUATOR_VERSION, evaluate, generate_dataset, parse_evaluation
from .dataset_prompts import sha256
from .dataset_client import parse_json_reply
from .repair_prompts import REPAIR_CURRICULUM_VERSION, generate_repair_prompts

WARMUP_VERSION = "artsy-warmup-100-vlm-natural-v4-prompt-only"


def _completed(record: dict[str, Any]) -> bool:
    artifacts = record.get("artifacts") or {}
    return bool(record.get("code") and artifacts.get("source") and artifacts.get("image")
                and record.get("sourceHash") and record.get("pixelHash"))


def is_execution_repair(record: dict[str, Any]) -> bool:
    generation = record.get("generation") or {}
    if generation and generation.get("initialDraft") != "natural":
        return False
    trace = record.get("trace") or []
    return (_completed(record)
            and any(not (t.get("feedback") or {}).get("ok")
                    and (t.get("feedback") or {}).get("failureKind") == "invalid_code" for t in trace)
            and bool(trace) and (trace[-1].get("feedback") or {}).get("ok") is True)


def is_natural_repair(record: dict[str, Any]) -> bool:
    if (record.get("selectionAudit") or {}).get("eligible") is False:
        return False
    generation = record.get("generation") or {}
    if generation and generation.get("initialDraft") != "natural":
        return False
    trace = record.get("trace") or []
    return (is_execution_repair(record) or bool(
        _completed(record) and generation.get("visualFailures")
        and any((t.get("observation") or {}).get("image") for t in trace)
        and trace and (trace[-1].get("feedback") or {}).get("ok")))


def _usable(record: dict[str, Any]) -> bool:
    return record.get("status") in ("accepted", "review")


def _duplicate_only(record: dict[str, Any]) -> bool:
    return (record.get("status") == "rejected"
            and str(record.get("reason", "")).startswith("Duplicate source or rendered pixels of "))


def select_warmup_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repairs = [r for r in records if is_execution_repair(r) and (_usable(r) or _duplicate_only(r))]
    if len(repairs) != 4:
        raise ValueError(f"Expected exactly 4 existing execution repairs; found {len(repairs)}")
    repair_prompts = {r["spec"]["id"] for r in repairs}
    prompts: set[str] = set()
    pixels: set[str] = set()
    sources: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if (not _usable(record) or not _completed(record) or len(record.get("trace", [])) != 1
                or not (record["trace"][0].get("feedback") or {}).get("ok")
                or record["spec"]["id"] in repair_prompts or record["spec"]["id"] in prompts
                or record.get("pixelHash") in pixels or record.get("sourceHash") in sources):
            continue
        prompts.add(record["spec"]["id"])
        pixels.add(record["pixelHash"])
        sources.add(record["sourceHash"])
        groups.setdefault(record["spec"]["family"], []).append(record)
    one_shot: list[dict[str, Any]] = []
    ordered_groups = list(groups.values())
    while len(one_shot) < 80:
        advanced = False
        for group in ordered_groups:
            if group and len(one_shot) < 80:
                one_shot.append(group.pop(0))
                advanced = True
        if not advanced:
            raise ValueError(f"Need 80 distinct existing one-shot examples; found {len(one_shot)}")
    return ([{"kind": "one-shot", "record": r} for r in one_shot]
            + [{"kind": "repair", "record": r} for r in repairs])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_bytes(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2 if pretty else None,
                      separators=None if pretty else (",", ":"))


def _atomic(path: Path, value: Any, *, lines: bool = False) -> None:
    data = "".join(_json_bytes(row) + "\n" for row in value) if lines else _json_bytes(value, pretty=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _inside(root: Path, path: str) -> Path:
    absolute = (root / path).resolve()
    try:
        absolute.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact escapes dataset: {path}") from error
    return absolute


async def warmup_plan(source: str | Path, seed: int = 42, pool_size: int = 128) -> dict[str, Any]:
    source_root = Path(source).resolve()
    raw_bytes = (source_root / "candidates.jsonl").read_bytes()
    raw = raw_bytes.decode("utf-8")
    selected = select_warmup_records([json.loads(line) for line in raw.splitlines() if line.strip()])
    return {"version": WARMUP_VERSION, "sourceRoot": str(source_root), "sourceHash": sha256(raw_bytes),
            "selected": selected, "specs": generate_repair_prompts(seed, pool_size), "seed": seed}


def _example(record: dict[str, Any], kind: str, origin: str, source_dataset: str) -> dict[str, Any]:
    result = copy.deepcopy(record)
    result["spec"].pop("requirements", None)  # Strip legacy rubric metadata from warmup exports.
    result["warmup"] = {"kind": kind, "split": "train", "origin": origin, "sourceDataset": source_dataset,
                        "sourceRecordId": record["id"], "sourceSplit": record["spec"]["split"],
                        "sourceStatus": record["status"], "duplicateException": _duplicate_only(record)}
    return result


def _rebase_record_paths(row: dict[str, Any], source: Path, directory: Path, target_rel: str) -> None:
    artifacts = row.get("artifacts") or {}
    for key, path in list(artifacts.items()):
        if path is None:
            continue
        file = _inside(source, path)
        try:
            suffix = file.relative_to(directory)
        except ValueError as error:
            raise ValueError(f"Artifact is outside its episode: {path}") from error
        artifacts[key] = str(Path(target_rel) / suffix)
    for trace in row.get("trace", []):
        observation = trace.get("observation")
        if observation and observation.get("image"):
            observation["image"] = str(Path(target_rel) / _inside(source, observation["image"]).relative_to(directory))
    for evaluation in row.get("evaluations", []):
        if evaluation.get("artifact"):
            evaluation["artifact"] = str(Path(target_rel) / _inside(source, evaluation["artifact"]).relative_to(directory))


async def _retain(selected: dict[str, Any], source: Path, output: Path) -> dict[str, Any]:
    record, kind = selected["record"], selected["kind"]
    directory = _inside(source, record["artifacts"]["directory"])
    target = Path("artifacts") / "retained" / record["id"]
    code = _inside(source, record["artifacts"]["source"]).read_text(encoding="utf-8")
    if sha256(code.replace("\r\n", "\n").strip()) != record["sourceHash"]:
        raise ValueError(f"Source hash mismatch: {record['id']}")
    if sha256(_inside(source, record["artifacts"]["image"]).read_bytes()) != record["imageHash"]:
        raise ValueError(f"Image hash mismatch: {record['id']}")
    shutil.copytree(directory, output / target, dirs_exist_ok=True)
    row = _example(record, kind, "retained", os.path.relpath(source, output))
    _rebase_record_paths(row, source, directory, str(target))
    return row


async def generate_warmup_dataset(options: dict[str, Any]) -> dict[str, Any]:
    if options.get("imageObservations") and not options.get("judge"):
        raise ValueError("Image-aware warmup requires a vision reviewer")
    output = Path(options["output"]).resolve()
    source = Path(options["source"]).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Warmup output and source must be separate directories")
    plan = await warmup_plan(source, options.get("seed", 42), options.get("poolSize", 128))
    import_root = Path(options["naturalRepairsFrom"]).resolve() if options.get("naturalRepairsFrom") else None
    imports: list[dict[str, Any]] = []
    if import_root:
        for file in sorted((import_root / "records").iterdir()):
            if file.suffix != ".json":
                continue
            record = json.loads(file.read_text(encoding="utf-8"))
            if not _usable(record) or (record.get("generation") or {}).get("initialDraft") != "natural" or not is_natural_repair(record):
                continue
            teacher = options["teacher"]
            teacher_model, teacher_url = getattr(teacher, "model"), getattr(teacher, "baseUrl", getattr(teacher, "base_url", None))
            if record["teacher"]["model"] != teacher_model or record["teacher"]["baseUrl"] != teacher_url:
                raise ValueError("Imported repairs must use the same teacher")
            for trace in record.get("trace", []):
                if "executedCode" in trace:
                    reply = parse_json_reply(trace["completion"]["text"])
                    if not isinstance(reply, dict) or reply.get("code") != trace["executedCode"]:
                        raise ValueError(f"Imported model output was modified: {record['id']}")
            imports.append(record)
            if len(imports) == 16:
                break
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".warmup.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError(f"Warmup dataset is locked: {lock_path}") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock:
            lock.write(_json_bytes({"pid": os.getpid()}))
        manifest: dict[str, Any] = {"version": WARMUP_VERSION, "sourceDataset": os.path.relpath(source, output),
            "sourceHash": plan["sourceHash"], "seed": plan["seed"], "repairCurriculum": REPAIR_CURRICULUM_VERSION,
            "target": {"oneShot": 80, "retainedRepairs": 4, "generatedRepairs": 16, "total": 100},
            "retained": [{"id": row["record"]["id"], "kind": row["kind"], "sourceSplit": row["record"]["spec"]["split"],
                "sourceStatus": row["record"]["status"], "sourceHash": row["record"].get("sourceHash"),
                "imageHash": row["record"].get("imageHash"), "duplicateException": _duplicate_only(row["record"])} for row in plan["selected"]],
            "generatedSpecs": plan["specs"],
            "modelContract": {"student": "Qwen/Qwen3.5-0.8B", "observations": "text-and-rendered-png" if options.get("imageObservations") else "execution-text",
                "inspectImageBeforeSubmission": bool(options.get("imageObservations"))},
            "visualReview": ({"model": options["judge"].model, "baseUrl": options["judge"].baseUrl, "version": EVALUATOR_VERSION,
                "threshold": options.get("threshold", 0.85)} if options.get("imageObservations") else None),
            "importedNaturalRepairs": ({"sourceDataset": os.path.relpath(import_root, output), "ids": [r["id"] for r in imports],
                "recordsHash": sha256(_json_bytes(imports))} if import_root else None),
            "splitPolicy": "All selected examples are for training. Original spec.split values are provenance only; former validation/test examples are no longer held out. Use a new evaluation set.",
            "reviewPolicy": "Selection does not promote visual/safety approval. Original statuses and humanReviewed flags are preserved. Review final drawings before SFT.",
            "repairPolicy": "Retain the four natural repairs, including three duplicate exceptions. Execute model code verbatim. Select sixteen naturally occurring execution or visual failures followed by successful corrections. With image observations enabled, review actual previews, attach their PNGs to visual correction requests, and require fresh final image approval. Never request bugs or mutate source. Failed actions are context, not positive SFT targets."}
        manifest_path = output / "warmup-manifest.json"
        old = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else None
        if old and _json_bytes(json.loads(old)) != _json_bytes(manifest):
            raise ValueError("Warmup source or plan changed; use a new output directory")
        if not old:
            _atomic(manifest_path, manifest)
        retained = [await _retain(row, source, output) for row in plan["selected"]]
        _atomic(output / "retained.jsonl", retained, lines=True)
        imported: list[dict[str, Any]] = []
        for record in imports:
            approved_review = None
            if options.get("imageObservations"):
                review_path = output / "import-reviews" / f"{record['id']}.json"
                review_path.parent.mkdir(parents=True, exist_ok=True)
                if review_path.exists():
                    review = json.loads(review_path.read_text(encoding="utf-8"))
                else:
                    completion = await evaluate(options["judge"], record["spec"], None,
                        _inside(import_root, record["artifacts"]["image"]).read_bytes(), record.get("code"))
                    review = {"imageHash": record["imageHash"], "completion": completion}
                    _atomic(review_path, review)
                if review["imageHash"] != record["imageHash"]:
                    raise ValueError(f"Imported review image changed: {record['id']}")
                result = parse_evaluation(review["completion"]["text"])
                if (review["completion"].get("finishReason") == "length" or result["safety"] != "safe"
                        or not result["correct"] or result["score"] < options.get("threshold", 0.85)):
                    continue
                approved_review = {"completion": review["completion"], "result": result}
            row = await _retain({"kind": "repair", "record": record}, import_root, output)
            if approved_review:
                row["evaluations"].append({"artifact": row["artifacts"]["image"], **approved_review})
                judge = options["judge"]
                row["evaluator"] = {"model": judge.model, "baseUrl": judge.baseUrl, "version": EVALUATOR_VERSION,
                    "reviewMode": "self-review" if judge.model == options["teacher"].model and judge.baseUrl == options["teacher"].baseUrl else "independent-model"}
                row["status"] = "accepted"
                row["generation"] = {**row["generation"], "imageObservations": True, "finalImagePassed": True}
            row["warmup"]["origin"] = "generated"
            imported.append(row)
        if imported:
            _atomic(output / "imported-natural-repairs.jsonl", imported, lines=True)
        _atomic(output / "repair-prompts.jsonl", plan["specs"], lines=True)
        generated_root = output / "generated"
        needed = 16 - len(imported)
        generation = None
        if needed:
            gen_options = {**options, "output": str(generated_root), "specs": plan["specs"], "mode": "repair", "repairTarget": needed}
            generation = await generate_dataset(gen_options)
        natural_repairs = []
        if needed:
            natural_repairs = [r for r in _jsonl(generated_root / "candidates.jsonl")
                if _usable(r) and is_natural_repair(r) and (r.get("generation") or {}).get("initialDraft") == "natural"
                and (not options.get("imageObservations") or (r.get("generation") or {}).get("finalImagePassed"))]
        generated = natural_repairs[:needed]
        seen_source = {r.get("sourceHash") for r in [*retained, *imported]}
        seen_pixels = {r.get("pixelHash") for r in [*retained, *imported]}
        for record in generated:
            if record.get("sourceHash") in seen_source or record.get("pixelHash") in seen_pixels:
                raise ValueError(f"New repair duplicates a retained example: {record['id']}")
            seen_source.add(record.get("sourceHash")); seen_pixels.add(record.get("pixelHash"))
        all_rows = [*retained, *imported]
        for record in generated:
            row = _example(record, "repair", "generated", "generated")
            row["artifacts"] = {key: str(Path("generated") / value) if value else value for key, value in row["artifacts"].items()}
            for trace in row["trace"]:
                if trace.get("observation"):
                    trace["observation"]["image"] = str(Path("generated") / trace["observation"]["image"])
            for evaluation in row.get("evaluations", []):
                if evaluation.get("artifact"):
                    evaluation["artifact"] = str(Path("generated") / evaluation["artifact"])
            all_rows.append(row)
        summary = {"version": WARMUP_VERSION, "total": len(all_rows), "oneShot": 80, "retainedRepairs": 4,
            "generatedRepairs": len(generated) + len(imported), "importedNaturalRepairs": len(imported),
            "freshNaturalRepairs": len(generated), "importedReviewExcluded": len(imports) - len(imported),
            "repairs": 4 + len(imported) + len(generated), "complete": len(all_rows) == 100,
            "remainingRepairs": needed - len(generated), "extraNaturalRepairs": len(natural_repairs) - len(generated),
            "requiresVisualReview": any(r.get("status") != "accepted" or not r.get("humanReviewed") for r in all_rows),
            "generation": generation}
        _atomic(output / "examples.jsonl", all_rows, lines=True)
        _atomic(output / "one-shot.jsonl", [r for r in all_rows if r["warmup"]["kind"] == "one-shot"], lines=True)
        _atomic(output / "repair.jsonl", [r for r in all_rows if r["warmup"]["kind"] == "repair"], lines=True)
        _atomic(output / "summary.json", summary)
        if summary["complete"]:
            _atomic(output / "train.jsonl", all_rows, lines=True)
        else:
            (output / "train.jsonl").unlink(missing_ok=True)
        return summary
    finally:
        lock_path.unlink(missing_ok=True)
