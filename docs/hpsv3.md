# HPSv3 scoring

Inspect evaluations use deferred batch scoring by default in the comparison
workflow: generate every final image with `--execution-only`, then invoke
`evaluation.score_hpsv3` in the MLX environment. It loads weights only after
all generation runs pass validation and exits after scoring. See the
[workflow commands](benchmark.md#deferred-hpsv3-scoring). No HTTP server is
required for this path. The service below remains available for RL and
optional inline evaluation.

The host-side client in `training/hpsv3.py` is shared by RL and Inspect.
The service exposes `GET /health` and `POST /score` with a prompt, base64 PNG,
and exact pinned model identifier. Set `ARTSY_HPSV3_API_KEY` on both sides to
require bearer authentication. The scorer accepts only 1920×1200 PNGs and
serializes GPU inference.

## Apple GPU backend

`training/hpsv3_mlx.py` preserves the Experiment 1 MLX implementation: language
weights at 4 bits with group size 64, bfloat16 vision weights, float32 reward
head, 200,704 image pixels, and sequential rotary positions. It requires an
isolated Python environment with `mlx-vlm`, `transformers`, `qwen-vl-utils`,
`fastapi`, `uvicorn`, `httpx`, and the cached tokenizer/processor metadata.

Set `ARTSY_HPSV3_BACKEND=mlx` and `ARTSY_HPSV3_MLX_CACHE` to the prepared cache
directory containing `mlx-4bit.safetensors`, `config.json`, and processor and
tokenizer files. This backend consumes the previously converted checkpoint;
it does not convert or download weights on startup.

```sh
ARTSY_HPSV3_BACKEND=mlx ARTSY_HPSV3_MLX_CACHE="$PWD/.cache/hpsv3" \
  .inference-venv/bin/python -m uvicorn training.hpsv3_server:app \
  --host 127.0.0.1 --port 8091
```

The Experiment 3 local launch metadata records the dedicated environment and
additional dependency path used by this workspace. The service was checked
against a saved Experiment 1 image with an identical raw reward. Quantization
validation remains limited; do not equate these rewards with full precision.

## PyTorch backend

The default backend uses `hpsv3.HPSv3RewardInferencer`. Set
`ARTSY_HPSV3_REVISION` to the intended pinned Hugging Face revision and optionally
`ARTSY_HPSV3_DEVICE` (default `cuda`). Install the upstream HPSv3 package and
its dependencies in the GPU environment, then start the same service module.

Both backends include a weight SHA-256 in the `/health` model ID. The MLX ID
also identifies its quantization and scoring pipeline version. Pass this exact
ID as Inspect's `--hpsv3-model` or RL's configured HPSv3 model. Model mismatches
and nonfinite scores fail rather than silently receiving zero reward.
