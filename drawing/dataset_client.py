"""Small OpenAI-compatible chat client used by dataset generation."""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, TypedDict

TINKER_MODEL_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
DEFAULT_DATA_MODEL = "thinkingmachines/Inkling:peft:262144"


def supports_image_input(model: str, base_url: str) -> bool:
    # Tinker's current GLM and DeepSeek sampling profiles reject image chunks.
    return not (base_url.rstrip("/") == TINKER_MODEL_URL
                and model.startswith(("zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4.1")))


def endpoint_api_key(base_url: str, configured_key: str | None = None) -> str | None:
    """Resolve provider credentials only for their matching endpoint."""
    if base_url.rstrip("/") == "https://openrouter.ai/api/v1":
        key = configured_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OpenRouter inference requires OPENROUTER_API_KEY")
        return key
    if base_url.rstrip("/") == TINKER_MODEL_URL:
        key = configured_key or os.getenv("TINKER_API_KEY")
        if not key:
            raise ValueError("Tinker inference requires TINKER_API_KEY")
        return key
    return configured_key


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
    max_tokens: int = 2048

    async def complete(self, messages: list[Message], max_tokens: int | None = None, *, temperature: float | None = None) -> Completion:
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._complete, messages, max_tokens, temperature), self.timeout_ms / 1000)
        except asyncio.TimeoutError:
            raise RuntimeError("Chat endpoint request timed out") from None

    def _complete(self, messages: list[Message], max_tokens: int | None, temperature: float | None = None) -> Completion:
        if not supports_image_input(self.model, self.baseUrl) and any(
                part.get("type") == "image_url" for message in messages
                if isinstance(message["content"], list) for part in message["content"]):
            raise ValueError("This Tinker model does not support image input; configure a vision model")
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            request_body["max_tokens"] = max_tokens
        if temperature is not None:
            request_body["temperature"] = temperature
        if self.baseUrl == TINKER_MODEL_URL:
            request_body["separate_reasoning"] = True
            if self.model.startswith(("thinkingmachines/Inkling", "zai-org/GLM-5.3")):
                request_body["reasoning_effort"] = "low"
        payload = json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.baseUrl}/chat/completions", payload, headers)
        if self.baseUrl == TINKER_MODEL_URL:
            # Tinker's edge rejects urllib's default client. Match its SDK's
            # HTTP transport, retaining the same sanitized error handling.
            import httpx
            response = httpx.post(request.full_url, content=payload, headers=headers,
                                  timeout=self.timeout_ms / 1000)
            status, body_bytes = response.status_code, response.content
        else:
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
        if self.baseUrl == TINKER_MODEL_URL and isinstance(choice, dict) and choice.get("finish_reason") == "length":
            raise RuntimeError("Tinker reply truncated; increase max tokens")
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
    budget = options.get("maxTokens", 2048)
    if type(budget) is not int or budget < 1:
        raise ValueError("maxTokens must be a positive integer")
    key = options.get("apiKey")
    return ChatClient(model=model, baseUrl=base, api_key=endpoint_api_key(base, key if isinstance(key, str) else None),
                      timeout_ms=timeout, max_tokens=budget)


def parse_json_reply(text: str) -> Any:
    trimmed = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", trimmed)
    return json.loads(fenced.group(1) if fenced else trimmed)
