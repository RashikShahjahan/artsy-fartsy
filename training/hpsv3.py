"""HPSv3 reward contract and lightweight HTTP client (no GPU dependencies)."""
from __future__ import annotations

import base64
import math

import httpx

REWARD_VERSION = "artsy-final-image-hpsv3-v1"
DEFAULT_MODEL = "MizzenAI/HPSv3"


def normalized_reward(score: float, center: float = 0.0, scale: float = 5.0) -> float:
    """Fixed sigmoid calibration; zero remains reserved for invalid submissions."""
    if (type(score) not in (int, float) or not math.isfinite(score)
            or not math.isfinite(center) or not math.isfinite(scale) or scale <= 0):
        raise ValueError("HPSv3 requires a finite score/center and positive finite scale")
    z = (score - center) / scale
    # Stable sigmoid, bounded away from zero even for extreme finite scores.
    if z >= 0:
        reward = 1 / (1 + math.exp(-z))
    else:
        exp = math.exp(z)
        reward = exp / (1 + exp)
    return max(1e-6, min(1 - 1e-6, reward))


class HPSv3Client:
    def __init__(self, url: str, model: str = DEFAULT_MODEL, *, center: float = 0.0,
                 scale: float = 5.0, api_key: str | None = None, timeout: float = 120):
        normalized_reward(0.0, center, scale)
        self.url, self.model = url.rstrip("/"), model
        self.center, self.scale, self.api_key = center, scale, api_key
        self.timeout = timeout

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
        reward = normalized_reward(score, self.center, self.scale)
        return {"model": self.model, "hpsv3_score": score, "preference": reward,
                "center": self.center, "scale": self.scale}
