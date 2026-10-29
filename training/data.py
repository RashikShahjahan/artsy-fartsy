"""Convert recorded Python attempts into bash-tool demonstrations.

The shell wrapper is synthesized; the Python source and execution feedback are
recorded evidence. Each record produces one complete conversation. All converted
assistant reasoning and tool-call actions receive loss, including unsuccessful attempts in a
successful episode. Instructions, tool feedback, and preview images are context.
Unexecuted malformed teacher replies and judge critiques are omitted.
"""

import copy
import hashlib
import json
from pathlib import Path

from drawing.config import LIMITS, MODEL_CONTEXT_WINDOW
from drawing.dataset_client import parse_json_reply
from drawing.dataset_specs import DIFFICULTIES

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command in the working directory.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]},
}}]
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
INPUT_MODE = "multimodal-previews"
CONVERSION = "bash-trajectory-human-curated-reasoning-assistant-only-v11"
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


def action(command: str, call_id: str, reasoning: str | None = None) -> dict:
    result = message("assistant", "", tool_calls=[{
        "id": call_id, "type": "function",
        "function": {"name": "bash", "arguments": {"command": command}},
    }])
    if isinstance(reasoning, str) and reasoning.strip():
        result["content"].insert(0, {"type": "thinking", "thinking": reasoning})
    return result


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
        f"You have at most {LIMITS['maxCalls']} tool calls and {MODEL_CONTEXT_WINDOW} total context tokens.\n\n"
        + (ROOT / "docs/artcanvas.md").read_text()
        + "\n\nEvery response must contain at least one bash tool call with "
        '{"command":"..."}. Shell state does not persist between commands; files do persist. '
        f"Finish by calling bash with exactly: {SUBMIT}"
    )


def native_trajectory_hash(trajectory: dict) -> str:
    return digest(json.dumps(trajectory, sort_keys=True, separators=(",", ":")).encode())


def native_prefix(spec: dict, config: dict) -> tuple[str, str]:
    """Reproduce the recorded host-agent framing without importing Inspect."""
    import yaml
    from drawing.agent_contract import drawing_system_prompt
    limit, kind = config.get("token_limit"), config.get("token_limit_type", "output")
    if (limit is not None and (type(limit) is not int or limit < 1)) or kind not in ("output", "all"):
        raise ValueError("Invalid native trajectory token budget")
    system = yaml.safe_load((ROOT / "evaluation/mini.yaml").read_text())["agent"]["system_template"]
    user = ("System instructions:\n" + drawing_system_prompt(limit, kind) + "\n\nTask:\n"
            + spec["prompt"] + "\n\nPreview with: "
            + f"python /opt/artsy/render.py --seed {spec['seed']} "
              "--output /workspace/preview --preview < /workspace/program.py")
    return system, user


def convert_native_record(record: dict, root: Path) -> dict:
    """Validate and preserve actual Inspect assistant actions and observations."""
    rid = record["id"]
    if record["spec"].get("kind") == "refusal" or record.get("refusal"):
        raise ValueError(f"{rid}: native visual trajectories must be drawings")
    trajectory = record.get("nativeTrajectory")
    if not isinstance(trajectory, dict) or native_trajectory_hash(trajectory) != record.get("nativeTrajectoryHash"):
        raise ValueError(f"{rid}: native trajectory hash mismatch")
    if trajectory.get("tools") != TOOLS:
        raise ValueError(f"{rid}: native trajectory requires the bash tool schema")
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError(f"{rid}: missing native conversation")
    system, user = native_prefix(record["spec"], record.get("nativeTrajectoryConfig") or {})
    if messages[:2] != [message("system", system), message("user", user)]:
        raise ValueError(f"{rid}: native prompt or system framing mismatch")
    indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]
    mask = trajectory.get("trainable_message_indices")
    if not indices or mask != indices or any(type(i) is not int for i in mask):
        raise ValueError(f"{rid}: native assistant supervision mask mismatch")
    source = artifact(root, record["artifacts"]["source"]).read_text()
    normalized = lambda value: value.replace("\r\n", "\n").strip()
    if normalized(source) != normalized(record["code"]) or digest(normalized(source).encode()) != record["sourceHash"]:
        raise ValueError(f"{rid}: final source does not match recorded code/hash")
    artifact(root, record["artifacts"]["image"], record["imageHash"])
    if not (record.get("submission") or {}).get("feedback", {}).get("ok"):
        raise ValueError(f"{rid}: missing successful native submission render")
    native_result = record.get("nativeResult") or {}
    if (native_result.get("outcome") != "submitted" or native_result.get("submitted") is not True
            or native_result.get("scores", {}).get("render_valid") != 1
            or native_result.get("sourceHash") != digest(source.encode())
            or native_result.get("imageHash") != record["imageHash"]):
        raise ValueError(f"{rid}: native scorer did not record a matching successful submission")
    result = copy.deepcopy(messages)
    pending, seen = [], set()
    submitted = False
    preview_count = 0
    correction = 'Every response must contain a bash tool call with {"command":"..."}.'
    for index, m in enumerate(result[2:], 2):
        if not isinstance(m, dict) or m.get("role") not in ("assistant", "tool", "user"):
            raise ValueError(f"{rid}: unexpected native message role")
        role = m["role"]
        content = m.get("content")
        if not isinstance(content, list):
            raise ValueError(f"{rid}: native content must use explicit parts")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError(f"{rid}: invalid native content")
            kind = part.get("type")
            if kind == "text" and isinstance(part.get("text"), str):
                continue
            if kind == "thinking" and role == "assistant" and isinstance(part.get("thinking"), str):
                continue
            if kind == "image" and role == "tool":
                image = artifact(root, part["image"], part["imageHash"])
                part["image"] = str(image)
                preview_count += 1
                continue
            raise ValueError(f"{rid}: unsupported native content part")
        if submitted:
            raise ValueError(f"{rid}: messages after terminal submission")
        if role == "assistant":
            if pending:
                raise ValueError(f"{rid}: native tool calls lack observations")
            calls = m.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError(f"{rid}: invalid native tool calls")
            for call in calls:
                function = call.get("function", {})
                arguments = function.get("arguments")
                call_id = call.get("id")
                if (not isinstance(call_id, str) or not call_id or call_id in seen
                        or call.get("type") != "function" or function.get("name") != "bash"
                        or not isinstance(arguments, dict) or set(arguments) != {"command"}
                        or not isinstance(arguments["command"], str) or not arguments["command"].strip()):
                    raise ValueError(f"{rid}: invalid native bash action")
                seen.add(call_id)
                command = arguments["command"]
                if command.strip() == SUBMIT and (len(calls) != 1 or index != len(result) - 2):
                    raise ValueError(f"{rid}: submission must be the final single action")
                pending.append((call_id, command))
        elif role == "tool":
            if not pending or m.get("tool_call_id") != pending[0][0] or m.get("tool_calls"):
                raise ValueError(f"{rid}: native tool observation ID/order mismatch")
            _, command = pending.pop(0)
            if command.strip() == SUBMIT:
                text = "\n".join(p["text"] for p in content if p["type"] == "text")
                if not text.startswith("Return code: 0\n") or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in text:
                    raise ValueError(f"{rid}: terminal submission did not succeed")
                submitted = True
        else:
            if pending or m != message("user", correction) or index == 2 or result[index - 1].get("role") != "assistant" or result[index - 1].get("tool_calls"):
                raise ValueError(f"{rid}: unexpected native user intervention")
    if pending or not submitted or not preview_count:
        raise ValueError(f"{rid}: native trajectory requires observed previews and terminal submission")
    return {"id": rid, "messages": result, "trainable_message_indices": indices, "tools": TOOLS}


def convert_record(record: dict, root: Path) -> dict:
    rid = record["id"]
    if record.get("runtimeVersion") != "artcanvas-v2":
        raise ValueError(f"{rid}: incompatible drawing runtime")
    if record.get("apiDocHash") != digest((ROOT / "docs/artcanvas.md").read_bytes()):
        raise ValueError(f"{rid}: API documentation changed since generation")
    if record.get("trajectoryFormat") == "inspect-bash-v1":
        return convert_native_record(record, root)
    spec = record["spec"]
    seed = spec["seed"]
    preview_command = (f"python /opt/artsy/render.py --seed {seed} "
                       "--output /workspace/preview --preview < /workspace/program.py")
    messages = [message("system", system_prompt()),
                message("user", spec["prompt"] + "\n\nPreview with: " + preview_command)]
    if spec.get("kind") == "refusal":
        if (record.get("refusal") != spec["refusal"]
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

    trace = record["trace"]
    final_index = next((i for i in range(len(trace) - 1, -1, -1) if trace[i].get("feedback") is not None), None)
    if final_index is None or not trace[final_index]["feedback"].get("ok"):
        raise ValueError(f"{rid}: no final successful execution")
    refused = trace[final_index]["feedback"].get("outcome") == "refused"
    source = None
    normalized = lambda code: code.replace("\r\n", "\n").strip()
    if refused:
        target = parse_json_reply(trace[final_index]["completion"]["text"])
        if (not isinstance(record.get("refusal"), str) or not record["refusal"].strip()
                or target != {"refusal": record["refusal"]}
                or trace[final_index]["feedback"].get("reason") != record["refusal"]):
            raise ValueError(f"{rid}: recorded refusal does not match teacher completion/feedback")
    else:
        if record.get("generation", {}).get("imagePreviews"):
            submission = record.get("submission") or {}
            preview = trace[final_index].get("previewObservation")
            if (not preview or submission.get("preview") != preview
                    or parse_json_reply(submission.get("completion", {}).get("text", "null")) != {"submit": True}
                    or not submission.get("feedback", {}).get("ok")):
                raise ValueError(f"{rid}: missing teacher confirmation of the final preview")
            artifact(root, preview["image"], preview["imageHash"])
        source = artifact(root, record["artifacts"]["source"]).read_text()
        if normalized(source) != normalized(record["code"]) or digest(normalized(source).encode()) != record["sourceHash"]:
            raise ValueError(f"{rid}: final source does not match recorded code/hash")
        artifact(root, record["artifacts"]["image"], record["imageHash"])

    trainable_message_indices = []
    for index, step in enumerate(trace):
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
        if refused and index == final_index:
            call_id = f"call_{index}_refusal"
            trainable_message_indices.append(len(messages))
            command = "rm -f /workspace/program.py\n" + refusal_command(record["refusal"])
            messages.append(action(command, call_id, step["completion"].get("reasoning")))
            messages.append(message("tool", "Return code: 0\nRefusal recorded", tool_call_id=call_id))
            continue
        code = parse_source(step["completion"]["text"])
        if "executedCode" in step and code != step["executedCode"]:
            raise ValueError(f"{rid}: executed code was modified")
        final = index == final_index
        if final and normalized(code) != normalized(source):
            raise ValueError(f"{rid}: final completion and artifact differ")
        write_id = f"call_{index}_write"
        call_id = f"call_{index}_preview"
        trainable_message_indices.append(len(messages))
        messages.append(action(write_command(code), write_id, step["completion"].get("reasoning")))
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
    messages.append(action(SUBMIT, "call_submit", (record.get("submission") or {}).get("completion", {}).get("reasoning")))
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
