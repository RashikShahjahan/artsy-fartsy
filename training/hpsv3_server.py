"""GPU scoring service. Install in a separate environment; see docs/hpsv3.md."""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import hashlib
import io
import math
import os
from pathlib import Path
import secrets
import tempfile
import threading

from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from training.hpsv3 import DEFAULT_MODEL


@asynccontextmanager
async def lifespan(app):
    from hpsv3 import HPSv3RewardInferencer
    from huggingface_hub import hf_hub_download
    revision = os.environ["ARTSY_HPSV3_REVISION"]
    checkpoint = hf_hub_download(DEFAULT_MODEL, "HPSv3.safetensors", revision=revision)
    # Clients pin the actual weights, not a mutable branch or local filename.
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    app.state.model_id = DEFAULT_MODEL + "@sha256:" + digest.hexdigest()
    app.state.inferencer = HPSv3RewardInferencer(
        checkpoint_path=checkpoint, device=os.getenv("ARTSY_HPSV3_DEVICE", "cuda"))
    app.state.lock = threading.Lock()
    yield


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100000)
    image: str = Field(max_length=12 * 1024 * 1024)
    model: str


@app.post("/score")
def score(request: ScoreRequest, authorization: str | None = Header(default=None)):
    key = os.getenv("ARTSY_HPSV3_API_KEY")
    if key and not secrets.compare_digest(authorization or "", "Bearer " + key):
        raise HTTPException(401, "Invalid scorer credentials")
    if request.model != app.state.model_id:
        raise HTTPException(409, "Configured model differs from loaded HPSv3 weights")
    try:
        png = base64.b64decode(request.image, validate=True)
        if len(png) > 8 * 1024 * 1024:
            raise ValueError("Image exceeds 8 MiB")
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (1920, 1200):
                raise ValueError("Expected trusted 1920x1200 PNG")
            image.verify()
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error
    with tempfile.TemporaryDirectory(prefix="artsy-hpsv3-") as directory:
        path = Path(directory) / "image.png"
        path.write_bytes(png)
        # Serialize GPU access across concurrent rollout requests.
        with app.state.lock:
            values = app.state.inferencer.reward([request.prompt], image_paths=[str(path)])
            value = float(values[0][0].item())
    if not math.isfinite(value):
        raise HTTPException(502, "HPSv3 returned a nonfinite score")
    return {"model": app.state.model_id, "score": value}


@app.get("/health")
def health():
    return {"model": app.state.model_id}
