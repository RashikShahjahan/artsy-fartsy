"""GPU scoring service. Install in a separate environment; see docs/hpsv3.md."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
import io
import math
import os
from pathlib import Path
import secrets
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from training.hpsv3 import DEFAULT_MODEL


@asynccontextmanager
async def lifespan(app):
    if os.getenv("ARTSY_HPSV3_BACKEND") == "mlx":
        from training.hpsv3_mlx import MLXRewardInferencer
        cache = Path(os.environ["ARTSY_HPSV3_MLX_CACHE"])
        digest = hashlib.sha256()
        with (cache / "mlx-4bit.safetensors").open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        app.state.model_id = DEFAULT_MODEL + "@sha256:" + digest.hexdigest() + ":mlx-4bit-g64-v1"
        app.state.inferencer = MLXRewardInferencer(cache)
        app.state.backend = "mlx"
        app.state.batcher = MicroBatcher(app.state.inferencer, "mlx")
        try:
            yield
        finally:
            app.state.batcher.close()
        return
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
    app.state.backend = "pytorch"
    app.state.batcher = MicroBatcher(app.state.inferencer, "pytorch")
    try:
        yield
    finally:
        app.state.batcher.close()


class MicroBatcher:
    """Single inference thread, with a short bounded coalescing window."""
    def __init__(self, inferencer, backend, max_batch=None, wait_ms=None):
        self.inferencer, self.backend = inferencer, backend
        self.max_batch = int(os.getenv("ARTSY_HPSV3_BATCH_SIZE", "4")) if max_batch is None else max_batch
        self.wait = (float(os.getenv("ARTSY_HPSV3_BATCH_WAIT_MS", "15")) if wait_ms is None else wait_ms) / 1000
        if self.max_batch < 1 or not math.isfinite(self.wait) or self.wait < 0:
            raise ValueError("Invalid HPSv3 microbatch settings")
        self.queue = Queue()
        self._state_lock = threading.Lock()
        self._closed = False
        self.completed_batches = 0
        self.completed_requests = 0
        self.last_batch_size = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hpsv3")
        self.executor.submit(self._run)

    def submit(self, prompt, image):
        from concurrent.futures import Future
        future = Future()
        with self._state_lock:
            if self._closed:
                raise RuntimeError("HPSv3 batcher is closed")
            self.queue.put((prompt, image, future))
        return future

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self.queue.put(None)
        self.executor.shutdown(wait=True)

    def _run(self):
        stopping = False
        while not stopping:
            first = self.queue.get()
            if first is None:
                break
            batch = [first]
            deadline = time.monotonic() + self.wait
            while len(batch) < self.max_batch:
                try:
                    item = self.queue.get(timeout=max(0, deadline - time.monotonic()))
                except Empty:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            try:
                prompts = [item[0] for item in batch]
                images = [item[1] for item in batch]
                if self.backend == "mlx":
                    values = self.inferencer.score_batch(prompts, images)
                else:
                    values = [float(row[0].item()) for row in
                              self.inferencer.reward(prompts, image_paths=images)]
                if len(values) != len(batch):
                    raise ValueError("HPSv3 returned the wrong batch length")
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("HPSv3 returned a nonfinite score")
            except Exception as error:
                for _, _, future in batch:
                    if not future.cancelled():
                        future.set_exception(error)
            else:
                with self._state_lock:
                    self.completed_batches += 1
                    self.completed_requests += len(batch)
                    self.last_batch_size = len(batch)
                for (_, _, future), value in zip(batch, values):
                    if not future.cancelled():
                        future.set_result(float(value))


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100000)
    image: str = Field(max_length=12 * 1024 * 1024)
    model: str


@app.post("/score")
async def score(request: ScoreRequest, authorization: str | None = Header(default=None)):
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
    directory = tempfile.TemporaryDirectory(prefix="artsy-hpsv3-")
    path = Path(directory.name) / "image.png"
    try:
        path.write_bytes(png)
        future = app.state.batcher.submit(request.prompt, str(path))
    except Exception:
        directory.cleanup()
        raise
    # The worker owns file lifetime, even if the HTTP request is cancelled.
    future.add_done_callback(lambda _: directory.cleanup())
    try:
        value = await asyncio.shield(asyncio.wrap_future(future))
    except Exception as error:
        raise HTTPException(502, "HPSv3 inference failed") from error
    if not math.isfinite(value):
        raise HTTPException(502, "HPSv3 returned a nonfinite score")
    return {"model": app.state.model_id, "score": value}


@app.get("/health")
def health():
    return {"model": app.state.model_id, "backend": app.state.backend,
            "batch_size": app.state.batcher.max_batch,
            "batch_wait_ms": app.state.batcher.wait * 1000,
            "completed_batches": app.state.batcher.completed_batches,
            "completed_requests": app.state.batcher.completed_requests,
            "last_batch_size": app.state.batcher.last_batch_size}
