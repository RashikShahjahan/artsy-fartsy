"""Small OpenAI-compatible chat client used by dataset generation."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, TypedDict


class Message(TypedDict):
    role: str
    content: str | list[dict[str, Any]]


class Completion(TypedDict, total=False):
    text: str
    model: str
    usage: dict[str, int]
    finishReason: str


@dataclass
class ChatClient:
    model: str
    baseUrl: str
    api_key: str | None = field(default=None, repr=False)
    timeout_ms: int = 60_000

    async def complete(self, messages: list[Message], max_tokens: int, *, temperature: float | None = None) -> Completion:
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._complete, messages, max_tokens, temperature), self.timeout_ms / 1000)
        except asyncio.TimeoutError:
            raise RuntimeError("Chat endpoint request timed out") from None

    def _complete(self, messages: list[Message], max_tokens: int, temperature: float | None = None) -> Completion:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            request_body["temperature"] = temperature
        payload = json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.baseUrl}/chat/completions", payload, headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_ms / 1000) as response:
                status = getattr(response, "status", 200)
                body_bytes = response.read()
        except urllib.error.HTTPError as error:
            # Provider bodies can echo credentials; never include them in exceptions.
            raise RuntimeError(f"Chat endpoint returned HTTP {error.code}") from None
        if not 200 <= status < 300:
            raise RuntimeError(f"Chat endpoint returned HTTP {status}")
        try:
            body = json.loads(body_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("Chat endpoint returned invalid JSON") from None
        choices = body.get("choices") if isinstance(body, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        content = choice.get("message", {}).get("content") if isinstance(choice, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("Chat endpoint returned no text completion")
        usage_value = body.get("usage")
        usage: dict[str, int] | None = None
        if usage_value is not None:
            if not isinstance(usage_value, dict):
                raise RuntimeError("Invalid usage counts from chat endpoint")
            prompt = usage_value.get("prompt_tokens")
            completion = usage_value.get("completion_tokens")
            if any(type(n) is not int or n < 0 or n > 9_007_199_254_740_991 for n in (prompt, completion)):
                raise RuntimeError("Invalid usage counts from chat endpoint")
            usage = {"prompt_tokens": prompt, "completion_tokens": completion}
        return {
            "text": content,
            "model": body.get("model") if isinstance(body.get("model"), str) else self.model,
            **({"usage": usage} if usage is not None else {}),
            "finishReason": choice.get("finish_reason") or "unknown",
        }


def chat_client(options: dict[str, Any]):
    base_url, model = options.get("baseUrl"), options.get("model")
    if base_url == "tinker://native":
        from .tinker_client import TinkerChatClient, DEFAULT_RENDERER, DEFAULT_MAX_TOKENS
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A model identifier is required")
        timeout = options.get("timeoutMs", 600_000)
        budget = options.get("maxTokens", DEFAULT_MAX_TOKENS)
        if type(timeout) is not int or timeout < 1 or type(budget) is not int or not 1 <= budget < 65536:
            raise ValueError("Invalid native judge timeout or token budget")
        return TinkerChatClient(model=model,
            renderer_name=options.get("renderer", DEFAULT_RENDERER),
            revision=options.get("revision"), max_tokens=budget, timeout_ms=timeout)
    if not isinstance(base_url, str):
        raise ValueError("Endpoint must be an HTTP(S) base URL without credentials, query, or fragment")
    # Match URL validation while keeping endpoint credentials out of all errors/provenance.
    from urllib.parse import urlsplit, urlunsplit
    try:
        parsed = urlsplit(base_url)
        valid = (parsed.scheme in ("http", "https") and bool(parsed.hostname)
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment)
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Endpoint must be an HTTP(S) base URL without credentials, query, or fragment")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A model identifier is required")
    base = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")).rstrip("/")
    timeout = options.get("timeoutMs", 60_000)
    if type(timeout) is not int or timeout < 1:
        raise ValueError("timeoutMs must be a positive integer")
    key = options.get("apiKey")
    return ChatClient(model=model, baseUrl=base, api_key=key if isinstance(key, str) else None,
                      timeout_ms=timeout)


def parse_json_reply(text: str) -> Any:
    trimmed = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", trimmed)
    return json.loads(fenced.group(1) if fenced else trimmed)
