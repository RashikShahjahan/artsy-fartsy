"""Small OpenAI-compatible chat client used by dataset generation."""

from __future__ import annotations

import asyncio
import json
import os
import re
import httpx
import weakref
from dataclasses import dataclass, field
from typing import Any, TypedDict

TINKER_MODEL_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
OPENROUTER_MODEL_URL = "https://openrouter.ai/api/v1"
DEFAULT_DATA_MODEL = "thinkingmachines/Inkling:peft:262144"
_OWNED_CLIENTS = weakref.WeakValueDictionary()


async def close_chat_clients() -> None:
    """Close pools owned by the current run before its event loop shuts down."""
    loop = asyncio.get_running_loop()
    await asyncio.gather(*(client.aclose() for client in list(_OWNED_CLIENTS.values())
                           if client._loop is loop))


def supports_image_input(model: str, base_url: str) -> bool:
    # Tinker's current GLM and DeepSeek sampling profiles reject image chunks.
    return not (base_url.rstrip("/") == TINKER_MODEL_URL
                and model.startswith(("zai-org/GLM-5.3", "deepseek-ai/DeepSeek-V4.1")))


def endpoint_api_key(base_url: str, configured_key: str | None = None) -> str | None:
    """Resolve provider credentials only for their matching endpoint."""
    if base_url.rstrip("/") == OPENROUTER_MODEL_URL:
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
    reasoning: str
    reasoning_details: list[dict[str, Any]]
    model: str
    usage: dict[str, int]
    finishReason: str
    tool_calls: list[dict[str, Any]]


@dataclass
class ChatClient:
    model: str
    baseUrl: str
    api_key: str | None = field(default=None, repr=False)
    timeout_ms: int = 60_000
    max_tokens: int = 2048
    reasoning_enabled: bool = True

    http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    _owns_client: bool = field(default=False, init=False, repr=False)
    _loop: Any = field(default=None, init=False, repr=False)

    async def aclose(self) -> None:
        if self.http_client is not None and self._owns_client:
            await self.http_client.aclose()
            self.http_client = None
            self._owns_client = False
            _OWNED_CLIENTS.pop(id(self), None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def complete(self, messages: list[Message], max_tokens: int | None = None, *, temperature: float | None = None,
                       tools: list[dict] | None = None) -> Completion:
        payload = self._request_body(messages, max_tokens, temperature)
        if tools is not None:
            payload.update(tools=tools, tool_choice="required", parallel_tool_calls=False)
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=self.timeout_ms / 1000, follow_redirects=False,
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8))
            self._owns_client = True
            self._loop = asyncio.get_running_loop()
            _OWNED_CLIENTS[id(self)] = self
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            # The deadline encloses connection pooling, headers and every body
            # chunk. Cancellation closes the stream instead of abandoning a thread.
            async with asyncio.timeout(self.timeout_ms / 1000):
                async with self.http_client.stream(
                    "POST", f"{self.baseUrl}/chat/completions", json=payload, headers=headers,
                ) as response:
                    if not 200 <= response.status_code < 300:
                        raise RuntimeError(f"Chat endpoint returned HTTP {response.status_code}")
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > 16 * 1024 * 1024:
                            raise RuntimeError("Chat endpoint response exceeds 16 MiB")
                        chunks.append(chunk)
        except (TimeoutError, httpx.TimeoutException):
            raise RuntimeError("Chat endpoint request timed out") from None
        except httpx.HTTPError:
            # Transport errors can contain request URLs or provider text.
            raise RuntimeError("Chat endpoint transport failed") from None
        return self._parse_completion(b"".join(chunks))

    def _request_body(self, messages: list[Message], max_tokens: int | None, temperature: float | None = None) -> dict[str, Any]:
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
        if self.baseUrl == OPENROUTER_MODEL_URL:
            request_body["reasoning"] = {"enabled": self.reasoning_enabled, "exclude": False}
        if self.baseUrl == TINKER_MODEL_URL:
            request_body["separate_reasoning"] = True
            if self.model.startswith(("thinkingmachines/Inkling", "zai-org/GLM-5.3")):
                request_body["reasoning_effort"] = "low"
        return request_body

    def _parse_completion(self, body_bytes: bytes) -> Completion:
        try:
            body = json.loads(body_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("Chat endpoint returned invalid JSON") from None
        choices = body.get("choices") if isinstance(body, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        if self.baseUrl == TINKER_MODEL_URL and isinstance(choice, dict) and choice.get("finish_reason") == "length":
            raise RuntimeError("Tinker reply truncated; increase max tokens")
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if content is None and isinstance(tool_calls, list) and tool_calls:
            content = ""
        if not isinstance(content, str):
            raise RuntimeError("Chat endpoint returned no text completion")
        if tool_calls is not None and (not isinstance(tool_calls, list) or any(
                not isinstance(call, dict) or not isinstance(call.get("id"), str)
                or not call["id"] or call.get("type") != "function"
                or not isinstance(call.get("function"), dict)
                or not isinstance(call["function"].get("name"), str)
                or not isinstance(call["function"].get("arguments"), str) for call in tool_calls)):
            raise RuntimeError("Chat endpoint returned invalid tool calls")
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
        # Keep provider-exposed reasoning separate from the strict JSON answer.
        # Details can also contain opaque/encrypted blocks: preserve them for
        # inspection, but only plain text is suitable as an SFT target.
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        details = message.get("reasoning_details")
        if not isinstance(reasoning, str) and isinstance(details, list):
            reasoning = "\n".join(part["text"] for part in details
                                  if isinstance(part, dict) and part.get("type") == "reasoning.text"
                                  and isinstance(part.get("text"), str))
        return {
            "text": content,
            **({"tool_calls": tool_calls} if tool_calls else {}),
            **({"reasoning": reasoning} if isinstance(reasoning, str) and reasoning.strip() else {}),
            **({"reasoning_details": details} if isinstance(details, list) and details else {}),
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
                      timeout_ms=timeout, max_tokens=budget,
                      reasoning_enabled=options.get("reasoningEnabled", True))


def parse_json_reply(text: str) -> Any:
    trimmed = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", trimmed)
    return json.loads(fenced.group(1) if fenced else trimmed)
