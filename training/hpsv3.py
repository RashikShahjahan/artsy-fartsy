"""HPSv3 reward contract and lightweight HTTP client (no GPU dependencies)."""
from __future__ import annotations

import base64
import math
import os

import httpx

REWARD_VERSION = "artsy-final-image-hpsv3-linear-v2"
DEFAULT_MODEL = "MizzenAI/HPSv3"
DEFAULT_LOWER = -11.0
DEFAULT_UPPER = 3.0
DEFAULT_MINIMUM = 0.02


def normalized_reward(score: float, lower: float = DEFAULT_LOWER,
                      upper: float = DEFAULT_UPPER, minimum: float = DEFAULT_MINIMUM) -> float:
    """Bounded linear reward for valid images; failures are assigned zero by callers."""
    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in (score, lower, upper, minimum))
            or lower >= upper or not math.isfinite(upper - lower)
            or not 0 < minimum < 1):
        raise ValueError("HPSv3 requires finite scores/bounds, lower < upper, and 0 < minimum < 1")
    # Branch before subtraction so extreme finite scores still clamp safely.
    fraction = 0.0 if score <= lower else 1.0 if score >= upper else (score - lower) / (upper - lower)
    return minimum + (1 - minimum) * fraction


class HPSv3Client:
    def __init__(self, url: str, model: str = DEFAULT_MODEL, *, lower: float = DEFAULT_LOWER,
                 upper: float = DEFAULT_UPPER, minimum: float = DEFAULT_MINIMUM,
                 api_key: str | None = None, timeout: float | None = None):
        normalized_reward(0.0, lower, upper, minimum)
        self.url, self.model = url.rstrip("/"), model
        self.lower, self.upper, self.minimum, self.api_key = lower, upper, minimum, api_key
        # Requests can wait behind many GPU microbatches before inference starts.
        # Resolve configuration once at construction; explicit callers win.
        timeout = float(os.getenv("ARTSY_HPSV3_TIMEOUT_SECONDS", "1800")) if timeout is None else timeout
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("HPSv3 timeout must be positive and finite")
        self.timeout = float(timeout)

    async def score(self, prompt: str, png: bytes) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url + "/score", headers=headers, json={
                "prompt": prompt, "image": base64.b64encode(png).decode("ascii"),
                "model": self.model})
            response.raise_for_status()
            result = response.json()
        if result.get("model") != self.model:
            raise ValueError("HPSv3 service model does not match the training configuration")
        score = result["score"]
        reward = normalized_reward(score, self.lower, self.upper, self.minimum)
        return {"model": self.model, "hpsv3_score": score, "preference": reward,
                "reward_version": REWARD_VERSION, "lower": self.lower,
                "upper": self.upper, "minimum": self.minimum}
