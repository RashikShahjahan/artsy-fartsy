"""Native vision judge transport, answer extraction, and failure handling."""

import base64
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from drawing.dataset_client import chat_client
from drawing.tinker_client import TinkerChatClient


class TinkerJudgeTests(unittest.IsolatedAsyncioTestCase):
    def client(self, *, stop_reason="stop", clean=True):
        client = TinkerChatClient(model="Qwen/Qwen3.8-27B")
        client._ready = True
        client._prompt = MagicMock(return_value=SimpleNamespace(length=100))
        client.sampler = SimpleNamespace(sample_async=AsyncMock(return_value=SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[1, 2], stop_reason=stop_reason)])))
        client.renderer = SimpleNamespace(
            get_stop_sequences=lambda: ["<|im_end|>"],
            parse_response=lambda _: ({"content": [
                {"type": "thinking", "thinking": "Check the drawing carefully."},
                {"type": "text", "text": '{"score":0.8,"reason":"Matches"}'},
            ]}, SimpleNamespace(is_clean=clean)))
        return client

    async def test_only_final_answer_becomes_judgment(self):
        client = self.client()
        result = await client.complete([{"role": "user", "content": "Grade"}], 16384, temperature=0)
        self.assertEqual(result["text"], '{"score":0.8,"reason":"Matches"}')
        self.assertEqual(result["reasoning"], "Check the drawing carefully.")
        self.assertEqual(result["usage"]["prompt_tokens"], 100)
        params = client.sampler.sample_async.call_args.kwargs["sampling_params"]
        self.assertEqual(params.max_tokens, 16384)
        self.assertEqual(params.temperature, 0)

    async def test_truncated_or_malformed_answers_fail_instead_of_becoming_rewards(self):
        for client, error in ((self.client(stop_reason="length"), "truncated"),
                              (self.client(clean=False), "Malformed")):
            with self.assertRaisesRegex(ValueError, error):
                await client.complete([], 16384)

    def test_inline_png_pixels_reach_native_renderer(self):
        client = TinkerChatClient(model="Qwen/Qwen3.8-27B")
        client.image_processor = SimpleNamespace(patch_size=16, merge_size=2)
        client.renderer = SimpleNamespace(build_generation_prompt=lambda messages: messages)
        output = BytesIO()
        Image.new("RGB", (320, 240), "red").save(output, "PNG")
        url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Grade"},
            {"type": "image_url", "image_url": {"url": url}},
        ]}]
        native = client._prompt(messages)
        image = native[0]["content"][1]["image"]
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(image.mode, "RGB")
        messages[0]["content"][1]["image_url"]["url"] = "https://example.com/image.png"
        with self.assertRaisesRegex(ValueError, "inline base64 PNG"):
            client._prompt(messages)

    def test_native_factory_does_not_use_http_credentials(self):
        client = chat_client({"model": "Qwen/Qwen3.8-27B", "baseUrl": "tinker://native",
                              "apiKey": "old-provider-secret", "revision": "pinned"})
        self.assertIsInstance(client, TinkerChatClient)
        self.assertEqual(client.revision, "pinned")
        self.assertEqual(client.timeout_ms, 600_000)
        self.assertEqual(TinkerChatClient(model=client.model).timeout_ms, 600_000)
        override = chat_client({"model": client.model, "baseUrl": client.baseUrl,
                                "timeoutMs": 123_000})
        self.assertEqual(override.timeout_ms, 123_000)
        self.assertNotIn("old-provider-secret", repr(client))
        with self.assertRaisesRegex(ValueError, "token budget"):
            chat_client({"model": client.model, "baseUrl": client.baseUrl, "maxTokens": 65536})


if __name__ == "__main__":
    unittest.main()
