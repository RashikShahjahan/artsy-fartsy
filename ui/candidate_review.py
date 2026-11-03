"""Local manual inspection and curation of complete Artsy SFT candidates.

Mounted in the unified UI by ui.server.
The generator's candidate file is never modified. Only reviewed, structurally
trainable episodes can be exported to accepted.jsonl for training.
"""

from __future__ import annotations

import copy
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from training.data import convert_record


PAGE = (Path(__file__).with_name('templates') / 'candidate-review.html').read_text(encoding='utf-8')


def load_candidates(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    records = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL record at line {line_number}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not isinstance(record.get("spec"), dict):
            raise ValueError(f"Malformed candidate at line {line_number}")
        if not isinstance(record["spec"].get("prompt"), str):
            raise ValueError(f"Missing prompt at line {line_number}")
        records.append(record)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Candidate file is empty or contains duplicate IDs")
    return records, hashlib.sha256(raw).hexdigest()


def validate_review(value: dict, sha: str, records: list[dict]) -> None:
    if not isinstance(value, dict) or value.get("dataset_sha256") != sha:
        raise ValueError("Review is for a different candidate dataset revision")
    if not isinstance(value.get("reviewer"), str) or not value["reviewer"].strip():
        raise ValueError("A reviewer name is required")
    decisions = value.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) - {r["id"] for r in records}:
        raise ValueError("Review contains unknown candidate IDs")
    for d in decisions.values():
        if not isinstance(d, dict) or d.get("decision") not in ("pending", "approved", "rejected") or not isinstance(d.get("notes", ""), str):
            raise ValueError("Invalid manual review decision")


def approved_record(record: dict, reviewer: str, decision: dict) -> dict:
    result = copy.deepcopy(record)
    result["automaticStatus"] = result["status"]
    result["automaticReason"] = result.get("reason")
    result["status"] = "accepted"
    result["humanReviewed"] = True
    result["manualReview"] = {"reviewer": reviewer, "notes": decision.get("notes", "")}
    return result


def export_approved(records: list[dict], review: dict, root: Path, output: Path) -> int:
    if output.resolve() == (root / "candidates.jsonl").resolve():
        raise ValueError("Refusing to overwrite the unfiltered candidate file")
    selected = []
    for record in records:
        decision = review["decisions"].get(record["id"], {})
        if decision.get("decision") != "approved":
            continue
        curated = approved_record(record, review["reviewer"], decision)
        try:
            convert_record(curated, root)
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            raise ValueError(f"{record['id']} cannot be exported as SFT: {exc}") from exc
        selected.append(curated)
    if not selected:
        raise ValueError("No approved candidates to export")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in selected), encoding="utf-8")
    os.replace(temp, output)
    return len(selected)


def image_paths(record: dict, root: Path) -> list[tuple[str, Path]]:
    images = []
    artifacts = record.get("artifacts") or {}
    def add(label: str, relative: str | None, expected_hash: str | None = None) -> None:
        if not isinstance(relative, str) or not relative:
            return
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or path.suffix.lower() != ".png":
            return
        if expected_hash:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected_hash:
                return
            from drawing.renderer import validate_png
            try:
                validate_png(data)
            except ValueError:
                return
        if path not in (p for _, p in images):
            images.append((label, path))
    for i, trace in enumerate(record.get("trace") or []):
        feedback = trace.get("feedback") or {}
        if feedback.get("ok") and feedback.get("artifact") and artifacts.get("directory"):
            add(f"Attempt {i + 1} preview", str(Path(artifacts["directory"]) / Path(feedback["artifact"]).name))
        observation = trace.get("observation") or {}
        add(f"Attempt {i + 1} observation", observation.get("image"))
    if record.get('trajectoryFormat') in ('inspect-bash-v1', 'drawing-tools-v1'):
        count = 0
        for message in (record.get('drawingTrajectory') or record.get('nativeTrajectory') or {}).get('messages') or []:
            if not isinstance(message, dict) or message.get('role') != 'tool':
                continue
            for part in message.get('content') or []:
                if isinstance(part, dict) and part.get('type') == 'image':
                    count += 1
                    expected_hash = part.get('imageHash')
                    if isinstance(expected_hash, str) and expected_hash:
                        add(f'Native preview {count}', part.get('image'), expected_hash)
    for i, evaluation in enumerate(record.get("evaluations") or []):
        add(f"Judge render {i + 1}", evaluation.get("artifact"))
    add("Final rendered image", artifacts.get("image"))
    return images


def make_handler(path: Path, port: int = 4173, review_path: Path | None = None,
                 accepted_output: Path | None = None, *, base_path="", navigation_html=None):
    path = path.resolve()
    root = path.parent
    records, sha = load_candidates(path)
    by_id = {record["id"]: record for record in records}
    review_path = (review_path or root / "manual-review.json").resolve()
    accepted_output = (accepted_output or root / "accepted.jsonl").resolve()
    if review_path == path or accepted_output == path or review_path == accepted_output:
        raise ValueError("Candidate, review and accepted-output paths must be different")

    def read_review() -> dict:
        if not review_path.exists():
            return {"dataset_sha256": sha, "reviewer": "", "decisions": {}}
        value = json.loads(review_path.read_text(encoding="utf-8"))
        if value.get("dataset_sha256") != sha:
            raise ValueError("Existing manual review is for a different candidate file")
        validate_review(value, sha, records)
        return value

    review = read_review()
    summaries = []
    for record in records:
        score = next((evaluation.get("correctness", {}).get("result", {}).get("score")
                      for evaluation in reversed(record.get("evaluations") or [])
                      if evaluation.get("correctness", {}).get("result")), None)
        summaries.append({
            "id": record["id"], "prompt": record["spec"]["prompt"],
            "family": record["spec"].get("family", ""),
            "status": record.get("status", "unknown"), "reason": record.get("reason", ""),
            "score": score,
        })

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, status: int, value: dict) -> None:
            self.respond(status, json.dumps(value, ensure_ascii=False).encode("utf-8"))

        def do_GET(self) -> None:
            target = urlsplit(self.path)
            assets = {
                "/assets/candidate-review.js": ("candidate-review.js", "application/javascript"),
                "/assets/dom.js": ("dom.js", "application/javascript"),
                "/assets/review.css": ("review.css", "text/css"),
                "/assets/style.css": ("style.css", "text/css"),
            }
            if target.path in assets:
                filename, mime = assets[target.path]
                return self.respond(200, Path(__file__).with_name(filename).read_bytes(), mime)
            if target.path == "/":
                from ui.shell import navigation
                payload = {"items": summaries, "review": review, "sha256": sha,
                           "reviewPath": str(review_path), "basePath": base_path}
                page = (PAGE.replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
                        .replace("__BASE_PATH__", base_path)
                        .replace("__NAVIGATION__", navigation_html or navigation("candidates")))
                return self.respond(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if target.path == "/api/record":
                rid = parse_qs(target.query).get("id", [""])[0]
                record = by_id.get(rid)
                if record is None:
                    return self.json_response(404, {"error": "Unknown candidate ID"})
                pairs = image_paths(record, root)
                images = [{"label": label, "url": base_path + "/image/" + rid + "/" + str(i)}
                          for i, (label, _) in enumerate(pairs)]
                try:
                    example = convert_record(approved_record(record, "inspection", {}), root)
                    error = None
                except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
                    example, error = None, str(exc)
                return self.json_response(200, {"record": record, "images": images,
                                                "example": example, "conversionError": error})
            if target.path.startswith("/image/"):
                pieces = target.path.split("/")
                if len(pieces) != 4 or pieces[2] not in by_id or not pieces[3].isdigit():
                    return self.json_response(404, {"error": "Unknown image"})
                pairs = image_paths(by_id[pieces[2]], root)
                index = int(pieces[3])
                if index >= len(pairs):
                    return self.json_response(404, {"error": "Unknown image"})
                return self.respond(200, pairs[index][1].read_bytes(), "image/png")
            return self.json_response(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                return self.json_response(403, {"error": "Invalid origin"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4_000_000:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict) or value.get("dataset_sha256") != sha:
                    raise ValueError("Candidate dataset changed")
                if self.path == "/api/review":
                    validate_review(value, sha, records)
                    temp = review_path.with_name(review_path.name + ".tmp")
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    os.replace(temp, review_path)
                    review.clear()
                    review.update(value)
                    return self.json_response(200, {"saved": True})
                if self.path == "/api/export":
                    latest = read_review()
                    count = export_approved(records, latest, root, accepted_output)
                    return self.json_response(200, {"count": count, "path": str(accepted_output)})
                return self.json_response(404, {"error": "Not found"})
            except (ValueError, TypeError, OSError, KeyError) as exc:
                return self.json_response(400, {"error": str(exc)})

    return Handler

