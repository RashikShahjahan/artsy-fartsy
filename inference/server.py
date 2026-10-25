"""OpenAI-compatible vision and tool-call bridge for Qwen3.5."""
import base64
import binascii
import io
import json
import os
import re
import time
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from jinja2 import TemplateError

from drawing.config import IMAGE_SIZE, MODEL_ID, MODEL_REVISION
from .model_catalog import model_catalog

SAMPLER_LOCK = threading.Lock()

CONTEXT_WINDOW = 32768
MAX_TOKENS = 2048
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def parse_tools(text, tools=None):
    """Decode Qwen's function/parameter tags without evaluating generated code."""
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    # Thinking is disabled in the prompt, but never execute calls inside reasoning.
    if text.lstrip().startswith("<think>"):
        _, end, answer = text.partition("</think>")
        if not end:
            return text, []
        text = answer.lstrip("\n")
    schemas = {tool["function"]["name"]: tool["function"].get("parameters", {}).get("properties", {})
               for tool in tools or []}
    calls, prose = [], []
    while "<tool_call>" in text:
        before, rest = text.split("<tool_call>", 1)
        call_text, end, text = rest.partition("</tool_call>")
        if not end:
            raise ValueError("Incomplete tool call")
        function = re.fullmatch(r"\s*<function=([A-Za-z_]\w*)>(.*?)</function>\s*", call_text, re.DOTALL)
        if not function:
            raise ValueError("Expected a named function")
        name, parameters = function.groups()
        args = {}
        while parameters.strip():
            parameter = re.match(r"\s*<parameter=([A-Za-z_]\w*)>(.*?)</parameter>", parameters, re.DOTALL)
            if not parameter:
                raise ValueError("Invalid or incomplete parameter")
            key, value = parameter.groups()
            if key in args:
                raise ValueError("Duplicate parameter")
            # Remove only the protocol's framing newlines; preserve Python indentation.
            value = value.removeprefix("\n").removesuffix("\n")
            kind = schemas.get(name, {}).get(key, {}).get("type", "string")
            args[key] = value if kind == "string" else json.loads(value)
            parameters = parameters[parameter.end():]
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}})
        prose.append(before)
    prose.append(text)
    return "".join(prose).strip(), calls


def image_content(part):
    """Decode an inline image locally before passing it to the processor."""
    from PIL import Image, UnidentifiedImageError
    source = part.get("image_url", {})
    url = source.get("url", "") if isinstance(source, dict) else source
    if not isinstance(url, str) or not re.match(r"^data:image/[\w.+-]+;base64,", url):
        raise ValueError("Images must use inline base64 image data URLs")
    encoded = url.split(",", 1)[1]
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("Image exceeds 8 MiB")
    try:
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds 8 MiB")
        with Image.open(io.BytesIO(data)) as image:
            if not (1 <= image.width <= 4096 and 1 <= image.height <= 4096):
                raise ValueError("Image dimensions exceed 4096×4096")
            return {"type": "image", "image": image.convert("RGB")}
    except (binascii.Error, UnidentifiedImageError, OSError) as error:
        raise ValueError("Invalid inline image") from error


def template_messages(messages):
    """Qwen's template expects tool argument objects, not OpenAI JSON strings."""
    result = []
    for original in messages:
        message = dict(original)
        # OpenAI-compatible clients may send instructions as `developer`.
        # Qwen's template uses `system` for the same instruction role.
        if message.get("role") == "developer":
            message["role"] = "system"
        content = message.get("content") or ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "image_url":
                    parts.append(image_content(part))
                elif part.get("type") == "text":
                    parts.append({"type": "text", "text": part["text"]})
                else:
                    raise ValueError("Unsupported content type")
            content = parts
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                args = function.get("arguments", {})
                function["arguments"] = json.loads(args) if isinstance(args, str) else args
                calls.append({**call, "function": function})
            message["tool_calls"] = calls
        message["content"] = content
        result.append(message)
    return result


@asynccontextmanager
async def lifespan(app):
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if os.getenv("ARTSY_BACKEND", "tinker") != "tinker":
        raise ValueError("ARTSY_BACKEND must be tinker")
    from .tinker_backend import TinkerBackend
    app.state.backend = "tinker"
    app.state.tinker = TinkerBackend(MODEL_ID, MODEL_REVISION, IMAGE_SIZE,
                                    os.getenv("ARTSY_TINKER_MODEL_PATH") or None)
    app.state.samplers = {}
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"model": MODEL_ID, "tokenizer_revision": MODEL_REVISION, "ready": True,
            "model_path": os.getenv("ARTSY_TINKER_MODEL_PATH") or None,
            "backend": app.state.backend, "input": ["text", "image"], "models": model_catalog()}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{**model, "object": "model", "owned_by": "Qwen"} for model in model_catalog()]}


@app.post("/v1/chat/completions")
def complete(body: dict):
    model_id = body.get("model")
    selected = next((model for model in model_catalog() if model['id'] == model_id), None)
    if not selected:
        raise HTTPException(400, "Unknown model. Choose a model from /v1/models.")
    backend = app.state.tinker
    if model_id != MODEL_ID:
        # Keep a small cache; every request retains its own backend reference.
        with SAMPLER_LOCK:
            samplers = getattr(app.state, 'samplers', {})
            app.state.samplers = samplers
            path = selected['model_path']
            if path not in samplers:
                candidate = backend.with_model_path(path)
                if len(samplers) >= 3:
                    samplers.pop(next(iter(samplers)))
                samplers[path] = candidate
            backend = samplers[path]
    budget = max(1, min(int(body.get("max_tokens") or MAX_TOKENS), MAX_TOKENS))
    try:
        messages = template_messages(body["messages"])
        raw, count, generated = backend.generate(
            messages, body.get("tools"), budget, CONTEXT_WINDOW)
    except (ValueError, TemplateError) as error:
        raise HTTPException(400, str(error)) from error
    try:
        text, calls = parse_tools(raw, body.get("tools"))
    except (ValueError, SyntaxError):
        # Preserve malformed output for the harness to ask the model to correct.
        text, calls = raw, []
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    finish = "tool_calls" if calls else "length" if generated >= budget else "stop"
    common = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": model_id}
    usage = {"prompt_tokens": count, "completion_tokens": generated, "total_tokens": count + generated}
    if not body.get("stream"):
        return {**common, "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}

    def chunks():
        delta = dict(message)
        if calls:
            delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(calls)]
        for part, reason in [(delta, None), ({}, finish)]:
            yield "data: " + json.dumps({**common, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": part, "finish_reason": reason}], "usage": usage if reason else None}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(chunks(), media_type="text/event-stream")
