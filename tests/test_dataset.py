from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

from drawing.dataset import EVALUATOR_VERSION, generate_dataset, parse_evaluation
from drawing.dataset_client import chat_client, parse_json_reply
from drawing.dataset_prompts import compact_json, family_split, generate_prompts, sha256, validate_specs

ROOT = Path(__file__).resolve().parents[1]
LEGACY_FIXTURE = ROOT / "tests" / "fixtures" / "legacy-dataset"


def completion(text: str) -> dict[str, Any]:
    return {"text": text, "model": "scripted", "finishReason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class ScriptedClient:
    def __init__(self, replies: list[str], model: str = "scripted", base_url: str = "http://localhost/v1"):
        self.replies, self.model, self.baseUrl = list(replies), model, base_url
        self.messages: list[list[dict[str, Any]]] = []

    async def complete(self, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        self.messages.append(json.loads(json.dumps(messages)))
        if not self.replies:
            raise AssertionError("Unexpected API request")
        return completion(self.replies.pop(0))


class DatasetCurriculumTests(unittest.TestCase):
    def test_1000_prompts_match_typescript_golden_hash_and_prefix_stability(self) -> None:
        expected = {
            0: "710d351ef98f6c68f5e35ab9bd48b549ca78d929e3056afc4076475bbb1b9276",
            42: "4c1e42d1c1be4f1ecfed385ccccd6dad2aa0371230e025008487269da0bf1001",
            2_147_483_647: "c053cd4f15f8ca99846d1da7a9a8745b311295b43054f69ce4f66fe9ae065b95",
            9_007_199_254_740_991: "00a7fa60419df78ecd4fc9d20b726722faed29915e058c6d101a0a93d8226a47",
        }
        for seed, digest in expected.items():
            specs = generate_prompts(1000, seed)
            self.assertEqual(sha256(compact_json(specs)), digest)
            self.assertEqual(specs[:20], generate_prompts(20, seed))
            self.assertEqual(len({s["prompt"] for s in specs}), 1000)
            self.assertTrue(all(s["split"] == family_split(s["family"]) for s in specs))
        self.assertNotEqual(generate_prompts(20, 43), generate_prompts(20, 42))

    def test_validation_and_fenced_json_are_strict(self) -> None:
        spec = generate_prompts(1)[0]
        with self.assertRaisesRegex(ValueError, "Split"):
            validate_specs([{**spec, "split": "test"}])
        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            validate_specs([spec, {**spec, "id": "other"}])
        with self.assertRaisesRegex(ValueError, "safe filenames"):
            validate_specs([{**spec, "id": "../escape"}])
        with self.assertRaisesRegex(ValueError, "count"):
            generate_prompts(0)
        self.assertEqual(parse_json_reply('```json\n{"ok":true}\n```'), {"ok": True})

    def test_evaluation_rejects_malformed_or_out_of_range_rubrics(self) -> None:
        valid = {"safety": "safe", "scores": [1], "correctRefusal": False, "reason": "Visible"}
        self.assertEqual(parse_evaluation(json.dumps(valid), 1), valid)
        for bad in ({**valid, "scores": []}, {**valid, "scores": [2]}, {**valid, "safety": "yes"}, {**valid, "reason": " "}):
            with self.assertRaisesRegex(ValueError, "invalid rubric"):
                parse_evaluation(json.dumps(bad), 1)


class DatasetClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_chat_protocol_and_credential_redaction(self) -> None:
        calls: list[dict[str, Any]] = []
        response = json.dumps({"model": "revision-1", "choices": [{"message": {"content": '{"code":"hello"}'}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()

        def mock_urlopen(request: Any, timeout: float) -> Any:
            calls.append({"url": request.full_url, "headers": dict(request.header_items()), "body": json.loads(request.data), "timeout": timeout})
            result = io.BytesIO(response)
            result.status = 200
            return result

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            client = chat_client({"model": "teacher", "baseUrl": "http://localhost/v1/", "apiKey": "test-secret"})
            result = await client.complete([{"role": "user", "content": "draw"}], 100)
            self.assertEqual(result["model"], "revision-1"); self.assertEqual(result["usage"]["completion_tokens"], 5)
            self.assertEqual(calls[0]["url"], "http://localhost/v1/chat/completions")
            self.assertEqual(calls[0]["body"]["max_tokens"], 100)
            self.assertEqual(calls[0]["body"]["stream"], False)
            self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer test-secret")
            self.assertNotIn("test-secret", repr(client))
        unauthorized = urllib.error.HTTPError("http://localhost", 401, "Unauthorized", {}, io.BytesIO(b"test-secret"))
        with patch("urllib.request.urlopen", side_effect=unauthorized):
            with self.assertRaisesRegex(RuntimeError, "Chat endpoint returned HTTP 401") as raised:
                await client.complete([], 100)
            self.assertNotIn("test-secret", str(raised.exception))
        for base in ("http://secret:password@localhost/v1", "https://localhost/v1?secret=x", "file:///tmp/api"):
            with self.assertRaisesRegex(ValueError, "without credentials"):
                chat_client({"model": "x", "baseUrl": base})


class DatasetGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="artsy-dataset-test-")
        self.output = Path(self.temp.name) / "dataset"

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def image_bytes() -> bytes:
        import io
        buffer = io.BytesIO()
        Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(buffer, format="PNG")
        return buffer.getvalue()

    def config(self) -> dict[str, Any]:
        from drawing.environment import ENVIRONMENT_CONFIG
        return {**ENVIRONMENT_CONFIG, "width": 32, "height": 32}

    def renderer(self, png: bytes | None = None) -> Any:
        class Renderer:
            async def render(inner, code: str, seed: int) -> dict[str, Any]:
                if code == "broken python":
                    return {"ok": False, "stdout": "", "stderr": "SyntaxError", "failureKind": "invalid_code"}
                return {"ok": True, "stdout": "", "stderr": "", "png": png or self.image_bytes(), "width": 32, "height": 32}
        return Renderer()

    @staticmethod
    def rubric(score: float = 1, **overrides: Any) -> str:
        return json.dumps({"safety": "safe", "scores": [score], "correctRefusal": False, "reason": "One circle is visible", **overrides})

    async def test_repair_submission_is_evaluated_and_resumes_from_checkpoint(self) -> None:
        code = "# generated program"
        teacher = ScriptedClient([json.dumps({"code": "broken python"}), json.dumps({"code": code})])
        judge = ScriptedClient([self.rubric(), self.rubric()], model="judge", base_url="http://reviewer/v1")
        spec = {"id": "circle", "family": "circles", "split": "train", "prompt": "Draw a blue circle",
                "requirements": ["One blue circle"], "expectedOutcome": "drawing", "seed": 42}
        options = {"output": str(self.output), "specs": [spec], "teacher": teacher, "judge": judge,
                   "candidates": 1, "config": self.config(), "renderer": self.renderer()}
        summary = await generate_dataset(options)
        self.assertEqual(summary["accepted"], 1); self.assertEqual(summary["requests"], 4)
        record = json.loads((self.output / "records" / "circle-c0.json").read_text())
        self.assertEqual(record["status"], "accepted"); self.assertEqual(record["trace"][0]["feedback"]["failureKind"], "invalid_code")
        self.assertEqual(record["runtimeVersion"], "artcanvas-v2"); self.assertTrue(record["sourceHash"])
        self.assertEqual(record["evaluator"]["reviewMode"], "independent-model")
        rerun = await generate_dataset(options)
        self.assertEqual(rerun, summary); self.assertEqual(len(teacher.messages), 2)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            await generate_dataset({**options, "threshold": 0.9})

    async def test_without_judge_output_is_review_only_and_duplicates_are_rejected(self) -> None:
        one = generate_prompts(1)[0]
        summary = await generate_dataset({"output": str(self.output), "specs": [one], "teacher": ScriptedClient(['{"code":"draw()"}']),
                                          "candidates": 1, "renderer": self.renderer(), "config": self.config()})
        self.assertEqual(summary["review"], 1); self.assertEqual(summary["accepted"], 0)
        self.assertEqual((self.output / "accepted.jsonl").read_text(), "")
        self.assertIn("evaluation pending", (self.output / "records" / f"{one['id']}-c0.json").read_text())

    async def test_duplicate_rendered_pixels_are_rejected_across_splits(self) -> None:
        specs = [
            {"id": "circle", "family": "circles", "split": "train", "prompt": "Draw a blue circle",
             "requirements": ["One circle"], "expectedOutcome": "drawing", "seed": 1},
            {"id": "house", "family": "houses", "split": "test", "prompt": "Draw a blue house",
             "requirements": ["One house"], "expectedOutcome": "drawing", "seed": 2},
        ]
        judge = ScriptedClient([self.rubric(), self.rubric(), self.rubric(), self.rubric()], model="judge")
        summary = await generate_dataset({"output": str(self.output), "specs": specs,
                                          "teacher": ScriptedClient(['{"code":"circle source"}', '{"code":"house source"}']),
                                          "judge": judge, "candidates": 1, "renderer": self.renderer(), "config": self.config()})
        self.assertEqual(summary["accepted"], 1); self.assertEqual(summary["rejected"], 1)
        rejected = (self.output / "rejected.jsonl").read_text()
        self.assertIn("Duplicate source or rendered pixels", rejected)
        self.assertEqual((self.output / "test.jsonl").read_text(), "")

    async def test_rejected_repair_can_retry_and_keeps_prior_attempt(self) -> None:
        spec = {"id": "retry-repair", "family": "circles", "split": "train", "prompt": "Draw a blue circle",
                "requirements": ["One blue circle"], "expectedOutcome": "drawing", "seed": 30,
                "repairTask": {"complexity": "advanced", "skills": ["color", "geometry"]}}
        teacher = ScriptedClient(['{"code":"good"}', '{"code":"broken python"}', '{"code":"fixed"}'])
        judge = ScriptedClient([self.rubric(), self.rubric(), self.rubric()], model="judge")
        options = {"output": str(self.output), "specs": [spec], "mode": "repair", "candidates": 1,
                   "teacher": teacher, "judge": judge, "renderer": self.renderer(), "config": self.config()}
        first = await generate_dataset(options)
        self.assertEqual(first["rejected"], 1)
        retried = await generate_dataset({**options, "retryRejected": True})
        self.assertEqual(retried["accepted"], 1)
        row = json.loads((self.output / "records/retry-repair-c0.json").read_text())
        self.assertEqual(row["previousAttempts"][0]["status"], "rejected")
        self.assertEqual(row["generation"]["executionFailures"], 1)

    async def test_malformed_teacher_reply_uses_bounded_repairs_and_lock_is_enforced(self) -> None:
        one = generate_prompts(1)[0]
        options = {"output": str(self.output), "specs": [one], "teacher": ScriptedClient(["not json", '{"code":"a","refusal":"b"}']),
                   "candidates": 1, "repairs": 1, "renderer": self.renderer(), "config": self.config()}
        summary = await generate_dataset(options)
        self.assertEqual(summary["requests"], 2); self.assertEqual(summary["rejected"], 1)
        (self.output / ".generation.lock").write_text("active")
        with self.assertRaisesRegex(RuntimeError, "Dataset is locked"):
            await generate_dataset(options)

    async def test_unsafe_preview_stops_before_final_submission(self) -> None:
        calls: list[str] = []
        base_renderer = self.renderer()

        class CountingRenderer:
            async def render(inner, code: str, seed: int) -> dict[str, Any]:
                calls.append(code)
                return await base_renderer.render(code, seed)

        one = generate_prompts(1)[0]
        teacher = ScriptedClient(['{"code":"draw()"}'])
        unsafe_rubric = json.dumps({"safety": "unsafe", "scores": [0] * len(one["requirements"]),
                                    "correctRefusal": False, "reason": "Unsafe content"})
        judge = ScriptedClient([unsafe_rubric], model="judge")
        summary = await generate_dataset({"output": str(self.output), "specs": [one], "teacher": teacher, "judge": judge,
                                          "candidates": 1, "renderer": CountingRenderer(), "config": self.config()})
        self.assertEqual(summary["rejected"], 1); self.assertEqual(calls, ["draw()"])
        row = json.loads((self.output / "records" / f"{one['id']}-c0.json").read_text())
        self.assertEqual(row["evaluations"] and row["status"], "rejected")
        self.assertFalse(any('"operation":"submit"' in line for line in Path(self.output / row["artifacts"]["trajectory"]).read_text().splitlines()))

    async def test_expected_refusal_requires_judge_approval(self) -> None:
        spec = {"id": "refusal", "family": "circles", "split": "train", "prompt": "A disallowed drawing request",
                "requirements": ["A correct refusal"], "expectedOutcome": "refusal", "seed": 4}
        teacher = ScriptedClient(['{"refusal":"I cannot help with that request."}'])
        judge = ScriptedClient([self.rubric(correctRefusal=True, reason="Appropriate refusal")], model="judge")
        summary = await generate_dataset({"output": str(self.output), "specs": [spec], "teacher": teacher, "judge": judge,
                                          "candidates": 1, "renderer": self.renderer(), "config": self.config()})
        self.assertEqual(summary["accepted"], 1)
        row = json.loads((self.output / "records/refusal-c0.json").read_text())
        self.assertEqual(row["refusal"], "I cannot help with that request.")
        self.assertNotIn("code", row)
        self.assertIn('"operation":"refuse"', Path(self.output / row["artifacts"]["trajectory"]).read_text())

    async def test_visual_repair_sends_exact_preview_and_reviewer_flip_cannot_fake_repair(self) -> None:
        import io as image_io
        from PIL import Image

        spec = {"id": "visual-repair", "family": "circles", "split": "train", "prompt": "Draw a blue circle",
                "requirements": ["One blue circle"], "expectedOutcome": "drawing", "seed": 13,
                "repairTask": {"complexity": "advanced", "skills": ["color", "geometry"]}}

        def color_png(color: tuple[int, int, int]) -> bytes:
            buff = image_io.BytesIO(); Image.new("RGB", (32, 32), color).save(buff, format="PNG"); return buff.getvalue()

        red, blue = color_png((255, 0, 0)), color_png((0, 0, 255))
        seen_pngs: list[bytes] = []

        class ColorRenderer:
            async def render(inner, code: str, seed: int) -> dict[str, Any]:
                png = red if code == "red" else blue
                return {"ok": True, "stdout": "", "stderr": "", "png": png, "width": 32, "height": 32}

        class VisualJudge:
            model, baseUrl = "vision", "http://judge/v1"
            async def complete(inner, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
                parts = messages[1]["content"]
                image_part = next((p for p in parts if p["type"] == "image_url"), None)
                if image_part:
                    seen_pngs.append(base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1]))
                score = 0 if len(seen_pngs) == 1 else 1
                return completion(json.dumps({"safety": "safe", "scores": [score], "correctRefusal": False,
                                              "reason": "The preview is red" if score == 0 else "The preview is blue"}))

        teacher = ScriptedClient(['{"code":"red"}', '{"code":"blue"}'])
        summary = await generate_dataset({"output": str(self.output), "specs": [spec], "mode": "repair", "imageObservations": True,
                                          "candidates": 1, "teacher": teacher, "judge": VisualJudge(),
                                          "renderer": ColorRenderer(), "config": self.config()})
        self.assertEqual(summary["accepted"], 1); self.assertEqual(len(seen_pngs), 3)
        row = json.loads((self.output / "records/visual-repair-c0.json").read_text())
        self.assertEqual(row["trace"][0]["executedCode"], "red"); self.assertEqual(row["trace"][1]["executedCode"], "blue")
        self.assertEqual(row["generation"]["visualFailures"], 1); self.assertTrue(row["generation"]["finalImagePassed"])
        observation = row["trace"][1]["observation"]
        self.assertEqual(seen_pngs[0], Path(self.output / observation["image"]).read_bytes())
        self.assertEqual(sha256(seen_pngs[0]), observation["imageHash"])

        # Same decoded pixels with a flip in evaluator opinion is not evidence of repair.
        self.output = Path(self.temp.name) / "identical"
        flip_teacher = ScriptedClient(['{"code":"same"}', '{"code":"same"}'])

        class FlipJudge:
            model, baseUrl = "vision", "http://judge/v1"
            calls = 0
            async def complete(inner, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
                inner.calls += 1
                score = 0.2 if inner.calls == 1 else 1
                return completion(json.dumps({"safety": "safe", "scores": [score], "correctRefusal": False, "reason": "review"}))

        flip_judge = FlipJudge()
        repeated_renderer = self.renderer(red)
        same_summary = await generate_dataset({"output": str(self.output), "specs": [spec], "mode": "repair", "imageObservations": True,
                                               "candidates": 1, "teacher": flip_teacher, "judge": flip_judge,
                                               "renderer": repeated_renderer, "config": self.config()})
        self.assertEqual(same_summary["rejected"], 1); self.assertEqual(flip_judge.calls, 2)
        same_row = json.loads((self.output / "records/visual-repair-c0.json").read_text())
        self.assertIn("identical", same_row["reason"]); self.assertFalse(same_row["generation"]["finalImagePassed"])

    async def test_infrastructure_retry_retains_previous_attempt_and_usage(self) -> None:
        spec = generate_prompts(1)[0]

        class FlakyTeacher:
            model, baseUrl = "flaky", "http://teacher/v1"
            calls = 0
            async def complete(inner, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
                inner.calls += 1
                if inner.calls == 1:
                    raise RuntimeError("Temporary provider outage")
                return completion('{"code":"draw()"}')

        teacher = FlakyTeacher()
        options = {"output": str(self.output), "specs": [spec], "teacher": teacher, "candidates": 1,
                   "renderer": self.renderer(), "config": self.config()}
        self.assertEqual((await generate_dataset(options))["infrastructureErrors"], 1)
        self.assertEqual((await generate_dataset(options))["infrastructureErrors"], 1)
        summary = await generate_dataset({**options, "retryErrors": True})
        self.assertEqual(summary["review"], 1); self.assertEqual(summary["requests"], 1); self.assertEqual(teacher.calls, 2)
        row = json.loads((self.output / "records" / f"{spec['id']}-c0.json").read_text())
        self.assertEqual(len(row["previousAttempts"]), 1)
        self.assertEqual(row["previousAttempts"][0]["status"], "infrastructure_error")

    async def test_one_shot_success_is_excluded_from_repair_curriculum(self) -> None:
        spec = {"id": "one-shot", "family": "circles", "split": "train", "prompt": "Draw a blue circle",
                "requirements": ["One blue circle"], "expectedOutcome": "drawing", "seed": 20,
                "repairTask": {"complexity": "advanced", "skills": ["color", "geometry"]}}
        summary = await generate_dataset({"output": str(self.output), "specs": [spec], "mode": "repair", "candidates": 1,
                                          "teacher": ScriptedClient(['{"code":"draw()"}']),
                                          "judge": ScriptedClient([self.rubric()], model="judge"),
                                          "renderer": self.renderer(), "config": self.config()})
        self.assertEqual(summary["rejected"], 1)
        row = json.loads((self.output / "records/one-shot-c0.json").read_text())
        self.assertIn("one-shot output excluded", row["reason"])
        self.assertEqual(len(row["evaluations"]), 1)

    async def test_typescript_checkpoint_resumes_without_provider_calls(self) -> None:
        shutil.copytree(LEGACY_FIXTURE, self.output)
        manifest = json.loads((self.output / "manifest.json").read_text())

        class MustNotCall:
            model = "golden-teacher"
            baseUrl = "http://teacher/v1"
            async def complete(self, *_: Any) -> dict[str, Any]:
                raise AssertionError("A completed TypeScript checkpoint must be reused")

        summary = await generate_dataset({"output": str(self.output), "specs": manifest["specs"], "teacher": MustNotCall(),
                                         "candidates": manifest["candidates"], "repairs": manifest["repairs"],
                                         "maxTokens": manifest["maxTokens"], "threshold": manifest["threshold"],
                                         "config": manifest["config"]})
        self.assertEqual(summary["review"], 1); self.assertEqual(summary["requests"], 1)
        record = json.loads((self.output / "records" / "fe6970f11b9ac67e684d72c6-c0.json").read_text())
        self.assertEqual(record["reason"], "Executable output; semantic and safety evaluation pending")


if __name__ == "__main__":
    unittest.main()
