"""OpenAI-compatible vision and tool-call bridge for Qwen3.5."""
import base64
import binascii
import io
import json
import math
import os
import re
import time
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from jinja2 import TemplateError

from drawing.config import (IMAGE_SIZE, MODEL_CONTEXT_WINDOW, MODEL_ID, MODEL_REVISION,
                            TINKER_SAMPLING_MODEL_ID)
from .model_catalog import HOSTED_MODEL_PROFILES, ModelCatalogCache, model_catalog

SAMPLER_LOCK = threading.Lock()

CONTEXT_WINDOW = MODEL_CONTEXT_WINDOW
MAX_TOKENS = MODEL_CONTEXT_WINDOW
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def native_message(message):
    """Translate the renderer's parsed tool calls without applying Qwen syntax."""
    content = message.get('content') or ''
    text = content if isinstance(content, str) else ''.join(
        part.get('text', '') for part in content if part.get('type') == 'text')
    calls = []
    for original in message.get('tool_calls', []):
        call = original.model_dump() if hasattr(original, 'model_dump') else original
        function = call['function']
        arguments = function['arguments']
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        if not isinstance(json.loads(arguments), dict):
            raise ValueError('Tool arguments must be a JSON object')
        calls.append({'id': call.get('id') or 'call_' + uuid.uuid4().hex, 'type': 'function',
                      'function': {'name': function['name'], 'arguments': arguments}})
    return text, calls


def parse_tools(text, tools=None):
    """Decode Qwen's function/parameter tags without evaluating generated code."""
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    # Never execute calls inside reasoning.
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


def image_content(part, preprocess_image=None):
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
            rgb = image.convert("RGB")
            return {"type": "image", "image": preprocess_image(rgb) if preprocess_image else rgb}
    except (binascii.Error, UnidentifiedImageError, OSError) as error:
        raise ValueError("Invalid inline image") from error


def template_messages(messages, preprocess_image=None):
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
                    parts.append(image_content(part, preprocess_image))
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
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if message.get("role") == "assistant" and isinstance(reasoning, str) and reasoning.strip():
            content = ([{"type": "thinking", "thinking": reasoning}]
                       + (content if isinstance(content, list) else [{"type": "text", "text": content}]))
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
                                    os.getenv("ARTSY_TINKER_MODEL_PATH") or None,
                                    sampling_model_id=TINKER_SAMPLING_MODEL_ID)
    app.state.samplers = {}
    yield

app = FastAPI(lifespan=lifespan)
app.state.model_catalog_cache = ModelCatalogCache()


def cached_model_catalog():
    return app.state.model_catalog_cache.get(model_catalog)


@app.get("/health")
def health():
    return {"model": MODEL_ID, "tokenizer_revision": MODEL_REVISION, "ready": True,
            "context_window": CONTEXT_WINDOW, "max_output_tokens": MAX_TOKENS,
            "sampling_model": TINKER_SAMPLING_MODEL_ID,
            "model_path": os.getenv("ARTSY_TINKER_MODEL_PATH") or None,
            "backend": app.state.backend, "input": ["text", "image"], "streaming": False, "models": cached_model_catalog()}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{**model, "object": "model", "owned_by": "Qwen"} for model in cached_model_catalog()]}


@app.post("/v1/chat/completions")
def complete(body: dict):
    if body.get("stream"):
        raise HTTPException(400, "Streaming unsupported by the hosted sampling SDK; use stream=false.")
    temperature = body.get("temperature", 0.0)
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or not 0 <= temperature <= 2):
        raise HTTPException(400, "temperature must be a finite number from 0 to 2")
    model_id = body.get("model")
    selected = next((model for model in cached_model_catalog() if model['id'] == model_id), None)
    if not selected:
        raise HTTPException(400, "Unknown model. Choose a model from /v1/models.")
    backend = app.state.tinker
    context_window = CONTEXT_WINDOW
    if model_id != MODEL_ID:
        # Keep a small cache; every request retains its own backend reference.
        with SAMPLER_LOCK:
            samplers = getattr(app.state, 'samplers', {})
            app.state.samplers = samplers
            base_model = selected.get('base_model') or model_id
            profile = HOSTED_MODEL_PROFILES.get(base_model)
            model_path = selected.get('model_path')
            key = (base_model, model_path) if profile else model_path
            if key not in samplers:
                if profile:
                    from .tinker_backend import TinkerBackend
                    candidate = TinkerBackend(base_model, profile['revision'], IMAGE_SIZE,
                        **({'model_path': model_path} if model_path else {}),
                        sampling_model_id=profile['sampling_model'], renderer_name=profile['renderer'])
                else:
                    candidate = backend.with_model_path(model_path)
                if len(samplers) >= 3:
                    samplers.pop(next(iter(samplers)))
                samplers[key] = candidate
            backend = samplers[key]
            if profile:
                context_window = profile['context_window']
    requested_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    budget = max(1, min(int(requested_tokens or context_window), context_window))
    try:
        messages = template_messages(body["messages"], backend.prepare_image)
        raw, count, generated = backend.generate(
            messages, body.get("tools"), budget, context_window, temperature=temperature)
    except (ValueError, TemplateError) as error:
        raise HTTPException(400, str(error)) from error
    try:
        text, calls = native_message(raw) if isinstance(raw, dict) else parse_tools(raw, body.get("tools"))
    except (ValueError, SyntaxError, KeyError, TypeError):
        # Preserve malformed output for the harness to ask the model to correct.
        text, calls = (raw if isinstance(raw, str) else json.dumps(raw, default=str)), []
    message = {"role": "assistant", "content": text or None}
    if isinstance(raw, dict) and isinstance(raw.get("content"), list):
        reasoning = "\n".join(part["thinking"] for part in raw["content"]
                              if part.get("type") == "thinking" and isinstance(part.get("thinking"), str))
        if reasoning:
            message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    effective_budget = min(budget, context_window - count)
    finish = "tool_calls" if calls else "length" if generated >= effective_budget else "stop"
    common = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": model_id}
    usage = {"prompt_tokens": count, "completion_tokens": generated, "total_tokens": count + generated}
    return {**common, "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
