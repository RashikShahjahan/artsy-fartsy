"""Convert recorded Python attempts into bash-tool demonstrations.

The shell wrapper is synthesized; the Python source and execution feedback are
recorded evidence. Each record produces one complete conversation. All converted
assistant tool-call actions receive loss, including unsuccessful attempts in a
successful episode. Instructions, tool feedback, and preview images are context.
Unexecuted malformed teacher replies and judge critiques are omitted.
"""

import hashlib
import json
from pathlib import Path

from drawing.dataset_specs import DIFFICULTIES
from drawing.judges import parse_code_safety_judgment

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command in the working directory.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]},
}}]
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
INPUT_MODE = "multimodal-previews"
CONVERSION = "bash-trajectory-separate-write-preview-refusal-only-assistant-only-v9"
LOSS_POLICY = "assistant-only"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact(root: Path, name: str, expected_hash: str | None = None) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing artifact or path outside dataset: {name}")
    if expected_hash and digest(path.read_bytes()) != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {name}")
    return path


def message(role: str, text: str, **kwargs) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}], **kwargs}


def action(command: str, call_id: str) -> dict:
    return message("assistant", "", tool_calls=[{
        "id": call_id, "type": "function",
        "function": {"name": "bash", "arguments": {"command": command}},
    }])


def write_command(code: str) -> str:
    # A quoted heredoc preserves source bytes, including shell metacharacters.
    delimiter = "ARTSY_SOURCE"
    while delimiter in code.splitlines():
        delimiter += "_"
    separator = "" if code.endswith("\n") else "\n"
    return f"cat > /workspace/program.py <<'{delimiter}'\n{code}{separator}{delimiter}"


def refusal_command(reason: str) -> str:
    payload = json.dumps({"reason": reason}, ensure_ascii=False)
    delimiter = "ARTSY_REFUSAL"
    while delimiter in payload.splitlines():
        delimiter += "_"
    return (f"cat > /workspace/refusal.json <<'{delimiter}'\n"
            f"{payload}\n{delimiter}")


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
        "Write the program and run its preview in separate bash calls on separate turns. "
        "Use the preview command to receive execution feedback and the actual PNG image. "
        "Inspect each preview against the drawing request and fix visible mistakes before submitting. "
        "The final program is executed again in a fresh directory with no other files. "
        "A text answer or a pre-existing PNG does not count. "
        'If you cannot fulfill the request, write only refusal.json containing {"reason":"..."} '
        "with a concise reason and do not create program.py. "
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
    duplicate = (warmup.get("duplicateException") is True and
                 record.get("reason", "").startswith("Duplicate source or rendered pixels of "))
    if record.get("status") not in {"accepted", "review"} and not (duplicate and record.get("status") == "rejected"):
        raise ValueError(f"{rid}: record is not selected for training")
    spec = record["spec"]
    seed = spec["seed"]
    preview_command = (f"python /opt/artsy/render.py --seed {seed} "
                       "--output /workspace/preview --preview < /workspace/program.py")
    messages = [message("system", system_prompt()),
                message("user", spec["prompt"] + "\n\nPreview with: " + preview_command)]
    if spec.get("kind") == "refusal":
        if (record.get("status") != "accepted" or record.get("refusal") != spec["refusal"]
                or record.get("refusalSource") != "prompt_plan"
                or record.get("trace") or record.get("code")
                or record.get("artifacts")):
            raise ValueError(f"{rid}: invalid authored refusal pair")
        first = len(messages)
        messages.append(action(refusal_command(spec["refusal"]), "call_refusal"))
        messages.append(message("tool", "Return code: 0\nRefusal recorded", tool_call_id="call_refusal"))
        second = len(messages)
        messages.append(action(SUBMIT, "call_submit"))
        return {"id": rid, "messages": messages,
                "trainable_message_indices": [first, second], "tools": TOOLS}

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

    trainable_message_indices = []
    for index, step in enumerate(trace):
        review = step.get("codeSafety")
        if review:
            judgment = parse_code_safety_judgment(review["completion"]["text"])
            if judgment != review["result"] or judgment["malicious"]:
                raise ValueError(f"{rid}: code safety review is unsafe or inconsistent")
            if review["sourceHash"] != digest(step.get("executedCode", "").encode()):
                raise ValueError(f"{rid}: code safety review does not match executed source")
        if step.get("observation"):
            observation = step["observation"]
            # Validate recorded observations, but do not expose privileged
            # judge feedback. Images are attached to their originating tool call.
            artifact(root, observation["image"], observation["imageHash"])
        feedback = step.get("feedback")
        if feedback is None:
            # Teacher JSON is a collection protocol, not an agent action. Omit
            # unexecuted replies and their synthetic correction instruction;
            # only actual attempted programs become bash demonstrations.
            continue
        code = parse_source(step["completion"]["text"])
        if "executedCode" in step and code != step["executedCode"]:
            raise ValueError(f"{rid}: executed code was modified")
        final = index == len(trace) - 1
        if final and normalized(code) != normalized(source):
            raise ValueError(f"{rid}: final completion and artifact differ")
        write_id = f"call_{index}_write"
        call_id = f"call_{index}_preview"
        trainable_message_indices.append(len(messages))
        messages.append(action(write_command(code), write_id))
        messages.append(message(
            "tool", "Synthesized protocol acknowledgment: program.py source supplied for writing; "
            "no recorded write execution result.", tool_call_id=write_id))
        trainable_message_indices.append(len(messages))
        messages.append(action(preview_command, call_id))
        visible = {key: feedback[key] for key in ("ok", "stdout", "stderr", "error", "exitCode", "width", "height")
                   if key in feedback}
        observation_message = message("tool", "Return code: " + str(feedback.get("exitCode", 0))
                                      + "\n" + json.dumps(visible), tool_call_id=call_id)
        if feedback.get("ok"):
            # Feedback paths reference the original machine/dataset. Resolve the
            # preview basename within the copied portable episode directory.
            name = Path(feedback["artifact"]).name
            if not name.startswith("preview-") or not name.endswith(".png"):
                raise ValueError(f"{rid}: invalid preview reference")
            preview = artifact(root, str(Path(record["artifacts"]["directory"]) / name),
                               record["imageHash"] if final else None)
            # Keep exports JSON-serializable; the encoder loads these exact pixels
            # locally and checks the hash again before building image chunks.
            observation_message["content"].append({"type": "image", "image": str(preview),
                                                    "imageHash": digest(preview.read_bytes())})
        messages.append(observation_message)
    trainable_message_indices.append(len(messages))
    messages.append(action(SUBMIT, "call_submit"))
    return {"id": rid, "messages": messages,
            "trainable_message_indices": trainable_message_indices, "tools": TOOLS}


def is_observed_repair(record: dict) -> bool:
    """A revised source recovered from a recorded execution or visual failure."""
    trace = record.get("trace") or []
    if len(trace) < 2 or not (trace[-1].get("feedback") or {}).get("ok"):
        return False
    final_source = trace[-1].get("executedCode")
    if not final_source:
        return False
    for step in trace[:-1]:
        source = step.get("executedCode")
        if not source or source == final_source:
            continue
        feedback = step.get("feedback") or {}
        evaluation = step.get("evaluation") or {}
        correctness = evaluation.get("correctness") or evaluation
        score = correctness.get("score")
        if (feedback.get("ok") is False or evaluation.get("correct") is False
                or (type(score) in (int, float) and score < 0.85)):
            return True
    return False


def load_examples(path: Path) -> tuple[list[dict], dict]:
    path = path.resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Training dataset is empty or contains duplicate record IDs")
    examples = [convert_record(record, path.parent) for record in records]
    return examples, {
        "dataset": str(path), "dataset_sha256": digest(path.read_bytes()),
        "records": len(records), "examples": len(examples),
        "batch_unit": "trajectory",
        "loss_policy": LOSS_POLICY,
        "supervised_actions": sum(len(e["trainable_message_indices"]) for e in examples),
        "omitted_unexecuted_replies": sum(t.get("feedback") is None for r in records for t in r.get("trace", [])),
        "trajectories_with_omitted_replies": sum(any(t.get("feedback") is None for t in r.get("trace", [])) for r in records),
        "one_shot": sum(r["spec"].get("kind") != "refusal"
                        and sum(t.get("feedback") is not None for t in r.get("trace", [])) == 1
                        for r in records),
        "repairs": sum(is_observed_repair(r) for r in records),
        "refusal_pairs": sum(r["spec"].get("kind") == "refusal" for r in records),
        "difficulty_counts": {level: sum(r["spec"].get("difficulty") == level for r in records)
                              for level in DIFFICULTIES},
        "unreviewed_records": sum(r.get("status") != "accepted" for r in records),
        "examples_with_images": sum(any(p["type"] == "image" for m in e["messages"] for p in m["content"]) for e in examples),
        "input_mode": INPUT_MODE,
        "api_doc_sha256": digest((ROOT / "docs/artcanvas.md").read_bytes()),
        "conversion": CONVERSION,
    }
