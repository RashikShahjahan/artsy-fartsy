"""Sequential, model-independent drawing episode operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import LIMITS
from .renderer import LocalRenderer, RUNTIME_VERSION, validate_png

ENVIRONMENT_CONFIG = {
    "version": 1,
    "runtimeVersion": RUNTIME_VERSION,
    "width": 1920,
    "height": 1200,
    "maxCalls": LIMITS["maxCalls"],
    "maxAttempts": LIMITS["maxAttempts"],
    "renderSeconds": LIMITS["renderSeconds"],
    "episodeMs": LIMITS["episodeMs"],
    "sourceBytes": LIMITS["sourceBytes"],
    "maxPolicyTokens": 32768,
    "pngBytes": 8 * 1024 * 1024,
    "feedbackBytes": 8192,
}

OUTCOMES = {"active", "submitted", "refused", "budget_exhausted", "timed_out"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class DrawingEnvironment:
    """Store episode state and enforce calls, previews, tokens, and deadlines."""

    def __init__(self, directory: str | os.PathLike[str], renderer: Any | None = None,
                 config: dict[str, Any] | None = None):
        self.artifact_root = Path(directory)
        self.config = dict(ENVIRONMENT_CONFIG if config is None else config)
        if any(not isinstance(self.config.get(key), int) or isinstance(self.config.get(key), bool)
               or not 1 <= self.config[key] <= 4096 for key in ("width", "height")):
            raise ValueError("Invalid configured canvas dimensions")
        self.renderer = renderer if renderer is not None else LocalRenderer(seconds=self.config["renderSeconds"])
        self.source = ""
        self.seed = 0
        self.prompt = ""
        self.started = 0.0
        self.calls = 0
        self.attempts = 0
        self.tokens = 0
        self.outcome = "active"
        self.directory: Path | None = None
        self.sequence = 0
        self._busy = False

    @property
    def state(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "prompt": self.prompt,
            "seed": self.seed,
            "source": self.source,
            "calls": self.calls,
            "attempts": self.attempts,
            "tokens": self.tokens,
            "runtimeVersion": self.config["runtimeVersion"],
        }

    async def _exclusive_enter(self) -> None:
        if self._busy:
            raise RuntimeError("Episode operations must be sequential")
        self._busy = True

    def _exclusive_exit(self) -> None:
        self._busy = False

    def _require_directory(self) -> Path:
        if self.directory is None:
            raise RuntimeError("Call reset first")
        return self.directory

    async def reset(self, prompt: str, seed: int = 0) -> dict[str, Any]:
        await self._exclusive_enter()
        try:
            if (not isinstance(prompt, str) or not prompt.strip() or not isinstance(seed, int)
                    or isinstance(seed, bool) or not 0 <= seed <= 2**53 - 1):
                raise ValueError("Expected a prompt and nonnegative integer seed")
            root = self.artifact_root.resolve()
            await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
            self.directory = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="episode-", dir=root))
            self.source = ""
            self.seed = seed
            self.prompt = prompt
            self.calls = self.attempts = self.tokens = self.sequence = 0
            self.started = time.monotonic()
            self.outcome = "active"
            await self._write_json(self.directory / "episode.json", {"prompt": prompt, "seed": seed, "config": self.config}, pretty=True)
            await self._record("reset", {"prompt": prompt, "seed": seed}, {"ok": True, "outcome": self.outcome})
            return {"prompt": prompt, "seed": seed, "config": self.config.copy(), "directory": str(self.directory)}
        finally:
            self._exclusive_exit()

    async def _write_json(self, path: Path, value: Any, *, pretty: bool = False) -> None:
        text = json.dumps(value, ensure_ascii=False, indent=2 if pretty else None,
                          separators=None if pretty else (",", ":"))
        await asyncio.to_thread(path.write_text, text, encoding="utf-8")

    async def _record(self, operation: str, input_value: Any, feedback: dict[str, Any]) -> None:
        directory = self._require_directory()
        entry = {
            "sequence": self.sequence,
            "operation": operation,
            "input": input_value,
            "feedback": feedback,
            "seed": self.seed,
            "runtimeVersion": self.config["runtimeVersion"],
            "calls": self.calls,
            "attempts": self.attempts,
            "tokens": self.tokens,
        }
        self.sequence += 1
        line = _canonical_json(entry) + "\n"
        await asyncio.to_thread(_append_text, directory / "trajectory.jsonl", line)
        await self._write_json(directory / "state.json", self.state)

    async def charge_tokens(self, count: int) -> dict[str, Any]:
        await self._exclusive_enter()
        try:
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Invalid token count")
            self._require_directory()
            if self.outcome != "active":
                return {"ok": False, "outcome": self.outcome, "error": "Episode is terminal"}
            self.tokens += count
            if self.tokens >= self.config["maxPolicyTokens"]:
                self.outcome = "budget_exhausted"
            feedback = {"ok": self.outcome == "active", "outcome": self.outcome}
            await self._record("tokens", {"count": count}, feedback)
            return feedback
        finally:
            self._exclusive_exit()

    async def _operation(self, name: str, input_value: Any, action: Any) -> dict[str, Any]:
        await self._exclusive_enter()
        try:
            self._require_directory()
            if self.outcome != "active":
                result = {"ok": False, "outcome": self.outcome, "error": "Episode is terminal"}
            elif (time.monotonic() - self.started) * 1000 >= self.config["episodeMs"]:
                self.outcome = "timed_out"
                result = {"ok": False, "outcome": self.outcome, "error": "Episode deadline exceeded"}
            elif self.calls >= self.config["maxCalls"] or self.tokens >= self.config["maxPolicyTokens"]:
                self.outcome = "budget_exhausted"
                result = {"ok": False, "outcome": self.outcome, "error": "Episode budget exhausted"}
            else:
                self.calls += 1
                result = await action()
                if (time.monotonic() - self.started) * 1000 >= self.config["episodeMs"] and result.get("outcome") == "active":
                    self.outcome = "timed_out"
                elif self.calls >= self.config["maxCalls"] and result.get("outcome") == "active":
                    self.outcome = "budget_exhausted"
                result["outcome"] = self.outcome
            await self._record(name, input_value, result)
            return result
        finally:
            self._exclusive_exit()

    async def write_code(self, source: str) -> dict[str, Any]:
        async def action() -> dict[str, Any]:
            if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > self.config["sourceBytes"]:
                return {"ok": False, "outcome": self.outcome, "error": "Expected 1–65536 bytes of source"}
            self.source = source
            return {"ok": True, "outcome": self.outcome}
        return await self._operation("write_code", {"source": source}, action)

    async def _render(self, final: bool) -> dict[str, Any]:
        if not self.source:
            return {"ok": False, "outcome": self.outcome, "error": "Write code first"}
        if not final and self.attempts >= self.config["maxAttempts"]:
            return {"ok": False, "outcome": self.outcome, "error": "Preview budget exhausted; submit final code"}
        if not final:
            self.attempts += 1
        directory = self._require_directory()
        try:
            result = await self.renderer.render(self.source, self.seed)
        except Exception as exc:
            result = {"ok": False, "stdout": "", "stderr": "", "error": str(exc), "failureKind": "infrastructure"}
        feedback: dict[str, Any] = {
            "ok": bool(result.get("ok")),
            "outcome": self.outcome,
            "stdout": result.get("stdout", "")[: self.config["feedbackBytes"]],
            "stderr": result.get("stderr", "")[: self.config["feedbackBytes"]],
        }
        for key in ("error", "exitCode", "elapsedMs", "failureKind"):
            if result.get(key) is not None:
                feedback[key] = result[key]
        png = result.get("png")
        if not result.get("ok") or png is None:
            feedback["ok"] = False
            feedback.setdefault("error", "No valid PNG")
            return feedback
        try:
            if len(png) > self.config["pngBytes"]:
                raise ValueError("PNG exceeds configured byte limit")
            feedback.update(validate_png(png, self.config))
        except (TypeError, ValueError) as exc:
            feedback.update({"ok": False, "error": str(exc), "failureKind": "invalid_code"})
            return feedback
        if (time.monotonic() - self.started) * 1000 >= self.config["episodeMs"]:
            self.outcome = "timed_out"
            return {**feedback, "ok": False, "outcome": self.outcome, "error": "Episode deadline exceeded"}
        artifact_name = "image.png" if final else f"preview-{self.attempts}.png"
        artifact = directory / artifact_name
        await asyncio.to_thread(artifact.write_bytes, png)
        feedback["artifact"] = str(artifact)
        if final:
            await asyncio.to_thread((directory / "program.py").write_text, self.source, encoding="utf-8")
            submission = {
                "sourceHash": _sha256(self.source.encode("utf-8")),
                "imageHash": _sha256(png),
                "seed": self.seed,
                "runtimeVersion": self.config["runtimeVersion"],
            }
            await self._write_json(directory / "submission.json", submission)
            self.outcome = "submitted"
            feedback["outcome"] = self.outcome
        return feedback

    async def run_code(self) -> dict[str, Any]:
        return await self._operation("run_code", {}, lambda: self._render(False))

    async def submit(self) -> dict[str, Any]:
        return await self._operation("submit", {}, lambda: self._render(True))

    async def refuse(self, reason: str) -> dict[str, Any]:
        async def action() -> dict[str, Any]:
            if not isinstance(reason, str) or not reason.strip() or len(reason.encode("utf-8")) > 2048:
                return {"ok": False, "outcome": self.outcome, "error": "Expected a refusal reason of 1–2048 bytes"}
            self.outcome = "refused"
            return {"ok": True, "outcome": self.outcome, "reason": reason}
        return await self._operation("refuse", {"reason": reason}, action)


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text)
