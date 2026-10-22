"""Local OpenAI-compatible bridge for Liquid's native Python-style tool calls."""
import ast
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

MODEL_ID = "LiquidAI/LFM2.5-230M"
MODEL_REVISION = "40cb2ad3b3044d5a41eee083a6103c8b523afa45"
MAX_TOKENS = 2048
lock = Lock()


def parse_tools(text):
    """Decode literal call arguments, never execute model-generated Python."""
    if "<|tool_call_start|>" not in text:
        return text.replace("<|im_end|>", "").replace("<|endoftext|>", ""), []
    before, rest = text.split("<|tool_call_start|>", 1)
    calls_text, separator, after = rest.partition("<|tool_call_end|>")
    if not separator:
        raise ValueError("Incomplete tool call")
    tree = ast.parse(calls_text.strip(), mode="eval").body
    if not isinstance(tree, ast.List):
        raise ValueError("Expected a list of tool calls")
    calls = []
    for item in tree.elts:
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.args:
            raise ValueError("Expected named tool with keyword arguments")
        args = {}
        for keyword in item.keywords:
            if keyword.arg is None or keyword.arg in args:
                raise ValueError("Invalid tool keyword")
            args[keyword.arg] = ast.literal_eval(keyword.value)
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                      "function": {"name": item.func.id, "arguments": json.dumps(args)}})
    return (before + after).replace("<|im_end|>", "").strip(), calls


def template_messages(messages):
    """Convert the OpenAI wire transcript back to Liquid's native tool format."""
    result = []
    for original in messages:
        message = dict(original)
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content)
        if message.get("tool_calls"):
            parts = []
            for call in message["tool_calls"]:
                args = json.loads(call["function"]["arguments"])
                parts.append(call["function"]["name"] + "(" + ", ".join(f"{key}={value!r}" for key, value in args.items()) + ")")
            content += "<|tool_call_start|>[" + ", ".join(parts) + "]<|tool_call_end|>"
        result.append({"role": message["role"], "content": content})
    return result


@asynccontextmanager
async def lifespan(app):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    app.state.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32).eval()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "ready": True}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "LiquidAI"}]}


@app.post("/v1/chat/completions")
def complete(body: dict):
    import torch
    if body.get("model") != MODEL_ID:
        raise HTTPException(400, "This server only serves " + MODEL_ID)
    budget = max(1, min(int(body.get("max_tokens") or MAX_TOKENS), MAX_TOKENS))
    tokenizer, model = app.state.tokenizer, app.state.model
    with lock, torch.inference_mode():
        inputs = tokenizer.apply_chat_template(template_messages(body["messages"]), tools=body.get("tools") or None,
                                               tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        count = inputs["input_ids"].shape[-1]
        if count + budget > 32768:
            raise HTTPException(400, "Context exceeds model capacity")
        output = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(output[0, count:], skip_special_tokens=False)
    try:
        text, calls = parse_tools(raw)
    except (ValueError, SyntaxError):
        # Preserve malformed output for the harness to ask the model to correct.
        text, calls = raw, []
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    finish = "tool_calls" if calls else "length" if output.shape[-1] - count >= budget else "stop"
    common = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": MODEL_ID}
    usage = {"prompt_tokens": count, "completion_tokens": int(output.shape[-1] - count), "total_tokens": int(output.shape[-1])}
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
