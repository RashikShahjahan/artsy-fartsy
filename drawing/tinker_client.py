"""Native Tinker vision completions for the correctness judge."""

from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass, field

from PIL import Image

from .dataset_client import Completion, Message

TINKER_URL = "tinker://native"
DEFAULT_RENDERER = "qwen3_8_medium_reasoning"
DEFAULT_MAX_TOKENS = 16384


@dataclass
class TinkerChatClient:
    model: str
    baseUrl: str = TINKER_URL
    renderer_name: str = DEFAULT_RENDERER
    revision: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_ms: int = 600_000
    _ready: bool = field(default=False, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def _initialize(self):
        import tinker
        from transformers import AutoImageProcessor, AutoTokenizer
        from tinker_cookbook import renderers

        # Only tokenizer and processor metadata are local. All weights stay hosted.
        tokenizer = AutoTokenizer.from_pretrained(self.model, revision=self.revision)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model, revision=self.revision)
        self.renderer = renderers.get_renderer(
            self.renderer_name, tokenizer, image_processor=self.image_processor)
        self.service = tinker.ServiceClient()
        self.sampler = self.service.create_sampling_client(base_model=self.model)
        self._ready = True

    def _prompt(self, messages: list[Message]):
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        from drawing.config import IMAGE_SIZE

        native = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part["type"] == "text":
                        parts.append({"type": "text", "text": part["text"]})
                        continue
                    if part["type"] != "image_url":
                        raise ValueError("Native judge accepts only text and inline images")
                    source = part["image_url"]
                    url = source["url"] if isinstance(source, dict) else source
                    if not isinstance(url, str) or not url.startswith("data:image/png;base64,"):
                        raise ValueError("Native judge requires inline base64 PNG images")
                    encoded = url.split(",", 1)[1]
                    if len(encoded) > 4 * ((8 * 1024 * 1024 + 2) // 3):
                        raise ValueError("Judge image exceeds 8 MiB")
                    png = base64.b64decode(encoded, validate=True)
                    with Image.open(io.BytesIO(png)) as image:
                        if (image.format != "PNG" or len(png) > 8 * 1024 * 1024
                                or not 1 <= image.width <= 4096
                                or not 1 <= image.height <= 4096):
                            raise ValueError("Invalid judge PNG")
                        height, width = smart_resize(
                            image.height, image.width,
                            factor=self.image_processor.patch_size * self.image_processor.merge_size,
                            min_pixels=IMAGE_SIZE["shortest_edge"],
                            max_pixels=IMAGE_SIZE["longest_edge"])
                        rgb = image.convert("RGB").resize((width, height))
                    parts.append({"type": "image", "image": rgb})
                content = parts
            native.append({"role": message["role"], "content": content})
        return self.renderer.build_generation_prompt(native)

    async def complete(self, messages: list[Message], max_tokens: int, *,
                       temperature: float | None = None) -> Completion:
        import tinker
        from tinker_cookbook.renderers import get_text_content

        async def generate():
            async with self._lock:
                if not self._ready:
                    await asyncio.to_thread(self._initialize)
            prompt = await asyncio.to_thread(self._prompt, messages)
            if prompt.length + max_tokens > 65536:
                raise ValueError("Judge request exceeds the Tinker model's 64K context")
            response = await self.sampler.sample_async(
                prompt=prompt, num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens, temperature=0 if temperature is None else temperature,
                    stop=self.renderer.get_stop_sequences()))
            sequence = response.sequences[0]
            if sequence.stop_reason == "length":
                # Fail before a partial answer can become a reward.
                raise ValueError("Correctness judge reply truncated; increase judge max tokens")
            message, termination = self.renderer.parse_response(sequence.tokens)
            if not termination.is_clean or message.get("tool_calls") or message.get("unparsed_tool_calls"):
                raise ValueError("Malformed correctness judge response")
            text = get_text_content(message)
            if not text.strip():
                raise ValueError("Correctness judge returned no final answer")
            content = message.get("content", [])
            reasoning = "\n".join(part["thinking"] for part in content
                                  if isinstance(part, dict) and part.get("type") == "thinking"
                                  and isinstance(part.get("thinking"), str)) if isinstance(content, list) else ""
            return {"text": text, "model": self.model, "finishReason": "stop",
                    **({"reasoning": reasoning} if reasoning.strip() else {}),
                    "usage": {"prompt_tokens": prompt.length,
                              "completion_tokens": len(sequence.tokens)}}

        try:
            return await asyncio.wait_for(generate(), self.timeout_ms / 1000)
        except asyncio.TimeoutError:
            raise RuntimeError("Native Tinker judge request timed out") from None
