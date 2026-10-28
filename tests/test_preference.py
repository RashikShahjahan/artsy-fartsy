"""Offline preference-data and two-image judge tests; no Tinker API calls."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from preference.data import load_pairs
from preference.judge import (
    TinkerPairwiseJudge, native_messages, pair_messages, parse_winner, validate_png,
)


def png(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


class PreferenceTests(unittest.TestCase):
    def test_strict_decisions(self):
        for label in ("A", "B", "Tie"):
            self.assertEqual(parse_winner(json.dumps({"winner": label})), label)
        for text in ('A', '{"winner":"C"}', '{"winner":"A","reason":"x"}', '[]'):
            with self.assertRaises(ValueError):
                parse_winner(text)

    def test_invalid_image(self):
        with self.assertRaises(ValueError):
            validate_png(b"not a png")

    def test_two_image_request(self):
        parts = pair_messages(png("red"), png("blue"))[0]["content"]
        images = [part for part in parts if part["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        self.assertNotEqual(images[0], images[1])

    def test_native_two_images(self):
        with patch("transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize",
                   return_value=(64, 64)):
            messages = native_messages(png("red"), png("blue"),
                                       SimpleNamespace(patch_size=14, merge_size=2))
        images = [part["image"] for part in messages[0]["content"] if part["type"] == "image"]
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(images[1].getpixel((0, 0)), (0, 0, 255))

    def test_compare_mock(self):
        judge = TinkerPairwiseJudge(renderer="mock", sampler_path="tinker://saved")
        self.assertEqual(judge.client.sampler_path, "tinker://saved")
        judge.client.complete = AsyncMock(return_value={"text": '{"winner":"B"}'})
        self.assertEqual(asyncio.run(judge.compare(png("red"), png("blue"))), "B")
        self.assertEqual(judge.client.complete.await_count, 1)
        judge.client.complete = AsyncMock(return_value={"text": "malformed"})
        with self.assertRaises(ValueError):
            asyncio.run(judge.compare(png("red"), png("blue")))

    def test_dataset_validation_and_reverse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, color in (("a", "red"), ("b", "blue"), ("c", "green")):
                (root / f"{name}.png").write_bytes(png(color))
            data = root / "labels.jsonl"
            row = dict(pair_id="1", group_id="prompt-1", image_a="a.png",
                       image_b="b.png", winner="A", split="train")
            data.write_text(json.dumps(row))
            pair = load_pairs(data)[0]
            with patch("preference.data.native_messages", return_value=[]):
                self.assertEqual(json.loads(pair.conversation(None, reverse=True)[0]["content"]),
                                 {"winner": "B"})
            other = dict(row, pair_id="2", image_b="c.png", split="test")
            data.write_text(json.dumps(row) + "\n" + json.dumps(other))
            with self.assertRaisesRegex(ValueError, "leaks"):
                load_pairs(data)
            data.write_text(json.dumps(dict(row, image_a="../outside.png")))
            with self.assertRaisesRegex(ValueError, "inside"):
                load_pairs(data)
            data.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, pair_id="2")))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_pairs(data)


if __name__ == "__main__":
    unittest.main()
