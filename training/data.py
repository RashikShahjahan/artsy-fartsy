"""Convert recorded Python attempts into bash-tool demonstrations.

The shell wrapper is synthesized; the Python source and execution feedback are
recorded evidence. Each record produces one complete conversation. Only its final
successful write/preview action and submission are selected for language-model
loss; earlier attempts, tool feedback, and images remain masked context.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command in the working directory.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]},
}}]
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact(root: Path, name: str, expected_hash: str | None = None) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing artifact or path outside dataset: {name}")
    if expected_hash and digest(path.read_bytes()) != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {name}")
    return path


def message(role: str, text: str, image: Path | None = None, **kwargs) -> dict:
    content = [{"type": "text", "text": text}]
    if image:
        content.append({"type": "image", "image": str(image)})
    return {"role": role, "content": content, **kwargs}


def action(command: str, call_id: str) -> dict:
    return message("assistant", "", tool_calls=[{
        "id": call_id, "type": "function",
        "function": {"name": "bash", "arguments": {"command": command}},
    }])


def write_command(code: str, seed: int) -> str:
    # A quoted heredoc preserves source bytes, including shell metacharacters.
    delimiter = "ARTSY_SOURCE"
    while delimiter in code.splitlines():
        delimiter += "_"
    separator = "" if code.endswith("\n") else "\n"
    return (f"cat > /workspace/program.py <<'{delimiter}'\n{code}{separator}{delimiter}\n"
            f"python /opt/artsy/render.py --seed {seed} "
            "--output /workspace/preview --preview < /workspace/program.py")


def parse_source(text: str) -> str:
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"code"} or not isinstance(value["code"], str):
        raise ValueError("Expected a teacher JSON completion containing code")
    return value["code"]


def system_prompt() -> str:
    return (
        "Fulfill the drawing request by writing /workspace/program.py. "
        "Use the installed ArtCanvas module to save output.png at 1920×1200. "
        "Use bash commands to write, run, and fix your program. "
        "Use the preview command to receive both execution feedback and the actual PNG image. "
        "The final program is executed again in a fresh directory with no other files. "
        "A text answer or a pre-existing PNG does not count. "
        'If you cannot fulfill the request, write only refusal.json containing {"reason":"..."} '
        "with a concise reason and remove program.py. "
        "You have at most sixteen model turns and 32768 output tokens.\n\n"
        + (ROOT / "docs/artcanvas.md").read_text()
        + "\n\nEvery response must contain at least one bash tool call with "
        '{"command":"..."}. Shell state does not persist between commands; files do persist. '
        f"Finish by calling bash with exactly: {SUBMIT}"
    )


def convert_record(record: dict, root: Path) -> dict:
    rid = record["id"]
    if record.get("selectionAudit", {}).get("eligible") is False:
        raise ValueError(f"{rid}: excluded by selection audit")
    if record.get("runtimeVersion") != "artcanvas-v2":
        raise ValueError(f"{rid}: incompatible drawing runtime")
    if record.get("apiDocHash") != digest((ROOT / "docs/artcanvas.md").read_bytes()):
        raise ValueError(f"{rid}: API documentation changed since generation")
    warmup = record.get("warmup", {})
    if warmup.get("split", record["spec"].get("split")) != "train":
        raise ValueError(f"{rid}: expected a training record")
    duplicate = (warmup.get("duplicateException") is True and
                 record.get("reason", "").startswith("Duplicate source or rendered pixels of "))
    if record.get("status") not in {"accepted", "review"} and not (duplicate and record.get("status") == "rejected"):
        raise ValueError(f"{rid}: record is not selected for training")
    generation = record.get("generation", {})
    if generation and generation.get("initialDraft") != "natural":
        raise ValueError(f"{rid}: synthetic repair excluded")
    if generation.get("imageObservations") and not generation.get("finalImagePassed"):
        raise ValueError(f"{rid}: final image did not pass review")

    source = artifact(root, record["artifacts"]["source"]).read_text()
    normalized = lambda code: code.replace("\r\n", "\n").strip()
    if normalized(source) != normalized(record["code"]) or digest(normalized(source).encode()) != record["sourceHash"]:
        raise ValueError(f"{rid}: final source does not match recorded code/hash")
    artifact(root, record["artifacts"]["image"], record["imageHash"])
    trace = record["trace"]
    if not trace or not trace[-1].get("feedback", {}).get("ok"):
        raise ValueError(f"{rid}: no final successful execution")

    seed = record["spec"]["seed"]
    preview_command = (f"python /opt/artsy/render.py --seed {seed} "
                       "--output /workspace/preview --preview < /workspace/program.py")
    messages = [message("system", system_prompt()), message(
        "user", record["spec"]["prompt"] + "\n\nPreview with: " + preview_command)]
    trainable_message_indices = []
    for index, step in enumerate(trace):
        if step.get("observation"):
            observation = step["observation"]
            image = artifact(root, observation["image"], observation["imageHash"])
            # The preceding tool result already carries this preview. Never pass
            # judge scores, verdicts, or privileged critique to the student.
            if not any(p.get("image") == str(image) for p in messages[-1]["content"]):
                messages.append(message("user", "Inspect this preview against the drawing request.", image))
        feedback = step.get("feedback")
        if feedback is None:
            # A malformed/truncated JSON reply was never executed. Preserve it
            # as masked context without inventing a successful tool invocation.
            messages.extend([message("assistant", step["completion"]["text"]), message(
                "user", "The previous reply could not be executed. Return a complete program using the bash tool.")])
            continue
        code = parse_source(step["completion"]["text"])
        if "executedCode" in step and code != step["executedCode"]:
            raise ValueError(f"{rid}: executed code was modified")
        final = index == len(trace) - 1
        if final and normalized(code) != normalized(source):
            raise ValueError(f"{rid}: final completion and artifact differ")
        call_id = f"call_{index}"
        target = action(write_command(code, seed), call_id)
        if final:
            trainable_message_indices.append(len(messages))
        messages.append(target)
        visible = {key: feedback[key] for key in ("ok", "stdout", "stderr", "error", "exitCode", "width", "height")
                   if key in feedback}
        image = None
        if feedback.get("ok"):
            # Feedback paths reference the original machine/dataset. Resolve the
            # preview basename within the copied portable episode directory.
            name = Path(feedback["artifact"]).name
            if not name.startswith("preview-") or not name.endswith(".png"):
                raise ValueError(f"{rid}: invalid preview reference")
            image = artifact(root, str(Path(record["artifacts"]["directory"]) / name),
                             record["imageHash"] if final else None)
        messages.append(message("tool", "Return code: " + str(feedback.get("exitCode", 0))
                                + "\n" + json.dumps(visible), image, tool_call_id=call_id))
    trainable_message_indices.append(len(messages))
    messages.append(action(SUBMIT, "call_submit"))
    return {"id": rid, "messages": messages,
            "trainable_message_indices": trainable_message_indices, "tools": TOOLS}


def load_examples(path: Path) -> tuple[list[dict], dict]:
    path = path.resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Training dataset is empty or contains duplicate record IDs")
    examples = [convert_record(record, path.parent) for record in records]
    return examples, {
        "dataset": str(path), "dataset_sha256": digest(path.read_bytes()),
        "records": len(records), "examples": len(examples),
        "one_shot": sum(r.get("warmup", {}).get("kind") == "one-shot" for r in records),
        "repairs": sum(r.get("warmup", {}).get("kind") == "repair" for r in records),
        "unreviewed_records": sum(r.get("status") != "accepted" for r in records),
        "examples_with_images": sum(any(p["type"] == "image" for m in e["messages"] for p in m["content"]) for e in examples),
        "api_doc_sha256": digest((ROOT / "docs/artcanvas.md").read_bytes()),
        "conversion": "bash-conversation-selective-loss-v2",
    }
