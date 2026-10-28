from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

from drawing.dataset import generate_dataset
from drawing.judges import JUDGE_MAX_TOKENS, parse_correctness_judgment, parse_safety_judgment, correctness_judge_prompt, safety_judge_prompt
from drawing.dataset_client import chat_client, parse_json_reply
from drawing.dataset_specs import compact_json, sha256, validate_specs
from drawing.remove_splits import remove_splits

ROOT = Path(__file__).resolve().parents[1]


def example_spec() -> dict[str, Any]:
    """Small hand-authored spec for generation tests; no legacy templates needed."""
    return validate_specs([{"family": "circles", "prompt": "Draw a blue circle on a plain background", "seed": 42}])[0]


def completion(text: str) -> dict[str, Any]:
    return {"text": text, "model": "scripted", "finishReason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class ScriptedClient:
    def __init__(self, replies: list[str], model: str = "scripted", base_url: str = "http://localhost/v1"):
        self.replies, self.model, self.baseUrl = list(replies), model, base_url
        self.messages: list[list[dict[str, Any]]] = []
        self.temperatures: list[float | None] = []

    async def complete(self, messages: list[dict[str, Any]], max_tokens: int, *, temperature: float | None = None) -> dict[str, Any]:
        self.messages.append(json.loads(json.dumps(messages)))
        self.temperatures.append(temperature)
        if not self.replies:
            raise AssertionError("Unexpected API request")
        return completion(self.replies.pop(0))


class DatasetSpecTests(unittest.TestCase):
    def test_spec_identity_is_stable_without_split_assignment(self) -> None:
        spec = example_spec()
        self.assertEqual(validate_specs([spec]), [spec])
        self.assertEqual(spec["id"], sha256(compact_json([spec["family"], spec["prompt"], spec["seed"]]))[:24])
        self.assertNotIn("split", spec)
        self.assertNotEqual(spec["id"], validate_specs([{**spec, "seed": 43, "id": None}])[0]["id"])

    def test_validation_and_fenced_json_are_strict(self) -> None:
        spec = example_spec()
        self.assertEqual(validate_specs([{**spec, "split": "test"}]), [spec])
        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            validate_specs([spec, {**spec, "id": "other"}])
        with self.assertRaisesRegex(ValueError, "safe filenames"):
            validate_specs([{**spec, "id": "../escape"}])
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_specs([])
        self.assertNotIn("requirements", validate_specs([{**spec, "requirements": ["legacy"]}])[0])
        self.assertNotIn("expectedOutcome", validate_specs([{**spec, "expectedOutcome": "refusal"}])[0])
        self.assertEqual(parse_json_reply('```json\n{"ok":true}\n```'), {"ok": True})

    def test_inspect_judge_parsing_is_shared_with_datagen(self) -> None:
        correct = {"score": 0.95, "reason": "Matches prompt"}
        safe = {"harmful": False, "reason": "Benign image"}
        self.assertEqual(parse_correctness_judgment(json.dumps(correct)), correct)
        self.assertEqual(parse_safety_judgment(json.dumps(safe)), safe)
        self.assertIn("Request: Draw a circle", correctness_judge_prompt("Draw a circle"))
        self.assertIn("Request: Draw a circle", safety_judge_prompt("Draw a circle"))
        for invalid in ({"score": float("nan"), "reason": "bad"},
                        {"score": True, "reason": "bad"},
                        {"score": 2, "reason": "bad"}):
            with self.assertRaises(ValueError):
                parse_correctness_judgment(json.dumps(invalid))
        with self.assertRaises(ValueError):
            parse_safety_judgment('{"harmful":"no","reason":"bad"}')


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
            result = await client.complete([{"role": "user", "content": "draw"}], 100, temperature=0)
            self.assertEqual(result["model"], "revision-1"); self.assertEqual(result["usage"]["completion_tokens"], 5)
            self.assertEqual(calls[0]["url"], "http://localhost/v1/chat/completions")
            self.assertEqual(calls[0]["body"]["max_tokens"], 100)
            self.assertEqual(calls[0]["body"]["stream"], False)
            self.assertEqual(calls[0]["body"]["temperature"], 0)
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


class SplitMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = {"id": "house", "family": "houses", "prompt": "Draw a house",
                     "seed": 1, "split": "test"}
        self.record = {"id": "house-c0", "spec": self.spec, "status": "accepted",
                       "sourceHash": "unchanged", "trace": [{"completion": {"text": "raw reply"}}]}
        specs_hash = sha256(compact_json([self.spec]))
        self.write("manifest.json", {"specs": [self.spec], "promptGenerator": {"specsHash": specs_hash}})
        self.write("prompt-plan.json", {"specs": [self.spec], "specsHash": specs_hash})
        self.write("records/house-c0.json", self.record)
        self.write("prompt-records/000000.json", {"index": 0, "spec": self.spec})
        for name in ("accepted", "candidates", "test"):
            self.write(f"{name}.jsonl", [self.record])
        self.write("prompts.jsonl", [self.spec])
        self.write("train.jsonl", [])
        self.write("validation.jsonl", [])

    def write(self, name: str, value: Any) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(compact_json(row) + "\n" for row in value)
                        if path.suffix == ".jsonl" else json.dumps(value))

    def test_migration_preserves_records_updates_hashes_and_is_idempotent(self) -> None:
        summary = remove_splits(self.root)
        self.assertEqual(summary["removedExports"], 3)
        self.assertEqual(summary["acceptedRecords"], 1)
        record = json.loads((self.root / "accepted.jsonl").read_text())
        self.assertEqual(record, {**self.record, "spec": {k: v for k, v in self.spec.items() if k != "split"}})
        self.assertEqual(json.loads((self.root / "records/house-c0.json").read_text()), record)
        plan = json.loads((self.root / "prompt-plan.json").read_text())
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(plan["specsHash"], sha256(compact_json(plan["specs"])))
        self.assertEqual(manifest["promptGenerator"]["specsHash"], plan["specsHash"])
        self.assertEqual(manifest["specs"], plan["specs"])
        self.assertNotIn("split", json.loads((self.root / "prompt-records/000000.json").read_text())["spec"])
        for name in ("train", "validation", "test"):
            self.assertFalse((self.root / f"{name}.jsonl").exists())
        self.assertEqual(remove_splits(self.root), {"updatedFiles": 0, "removedExports": 0, "acceptedRecords": 1})

    def test_mismatched_exports_or_corrupt_plan_are_rejected_without_edits(self) -> None:
        for corrupt in ("exports", "plan"):
            with self.subTest(corrupt=corrupt):
                if corrupt == "exports":
                    self.write("test.jsonl", [])
                else:
                    self.write("test.jsonl", [self.record])
                    self.write("prompt-plan.json", {"specs": [self.spec], "specsHash": "bad"})
                before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
                with self.assertRaises(ValueError):
                    remove_splits(self.root)
                self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)

    def test_active_generation_lock_is_preserved(self) -> None:
        lock = self.root / ".prompt-generation.lock"
        lock.write_text("active")
        with self.assertRaisesRegex(RuntimeError, "locked"):
            remove_splits(self.root)
        self.assertEqual(lock.read_text(), "active")
        self.assertFalse((self.root / ".generation.lock").exists())


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
    def correctness(score: float = 1, reason: str = "One circle is visible") -> str:
        return json.dumps({"score": score, "reason": reason})

    @staticmethod
    def safety(harmful: bool = False, reason: str = "Benign drawing") -> str:
        return json.dumps({"harmful": harmful, "reason": reason})

    async def test_dual_judges_are_queried_in_parallel(self) -> None:
        from drawing.dataset import evaluate

        both_started = asyncio.Event()
        active = 0

        class BarrierJudge:
            def __init__(inner, role: str) -> None:
                inner.model, inner.baseUrl, inner.role = role, "http://judge/v1", role

            async def complete(inner, messages: list[dict[str, Any]], max_tokens: int, *,
                               temperature: float | None = None) -> dict[str, Any]:
                nonlocal active
                self.assertIsNone(temperature)
                self.assertEqual(max_tokens, JUDGE_MAX_TOKENS)
                self.assertEqual(messages[0]["content"][0]["text"],
                                 correctness_judge_prompt("Draw a circle") if inner.role == "correctness"
                                 else safety_judge_prompt("Draw a circle"))
                active += 1
                if active == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), 0.5)
                return completion(self.correctness() if inner.role == "correctness" else self.safety())

        result = await evaluate(BarrierJudge("correctness"), BarrierJudge("safety"),
                                "Draw a circle", self.image_bytes())
        self.assertEqual(set(result), {"correctness", "safety"})

    async def test_repair_submission_is_evaluated_and_resumes_from_checkpoint(self) -> None:
        code = "# generated program"
        teacher = ScriptedClient([json.dumps({"code": "broken python"}), json.dumps({"code": code})])
        correctness = ScriptedClient([self.correctness(), self.correctness()], model="correctness", base_url="http://reviewer/v1")
        safety = ScriptedClient([self.safety(), self.safety()], model="safety", base_url="http://reviewer/v1")
        code_safety = ScriptedClient(['{"malicious":false,"reason":"Drawing only"}'] * 2, model="code-safety")
        spec = {"id": "circle", "family": "circles", "prompt": "Draw a blue circle", "seed": 42}
        options = {"output": str(self.output), "specs": [spec], "teacher": teacher,
                   "correctnessJudge": correctness, "safetyJudge": safety,
                   "codeSafetyJudge": code_safety,
                   "candidates": 1, "config": self.config(), "renderer": self.renderer()}
        summary = await generate_dataset(options)
        self.assertEqual(summary["accepted"], 1); self.assertEqual(summary["requests"], 8)
        record = json.loads((self.output / "records" / "circle-c0.json").read_text())
        self.assertEqual(record["status"], "accepted"); self.assertEqual(record["trace"][0]["feedback"]["failureKind"], "invalid_code")
        self.assertEqual(record["runtimeVersion"], "artcanvas-v2"); self.assertTrue(record["sourceHash"])
        self.assertEqual(record["evaluator"]["correctness"]["model"], "correctness")
        self.assertEqual(record["evaluator"]["safety"]["model"], "safety")
        self.assertEqual(correctness.temperatures, [None, None])
        self.assertEqual(safety.temperatures, [None, None])
        self.assertEqual(correctness.messages[0][0]["content"][0]["text"], correctness_judge_prompt(spec["prompt"]))
        self.assertEqual(safety.messages[0][0]["content"][0]["text"], safety_judge_prompt(spec["prompt"]))
        self.assertNotIn("requirements", json.dumps(correctness.messages))
        self.assertNotIn("expectedOutcome", json.dumps(safety.messages))
        self.assertEqual(set(record["evaluations"][0]), {"artifact", "correctness", "safety"})
        self.assertNotIn("requirements", record["spec"])
        self.assertNotIn("expectedOutcome", record["spec"])
        # A legacy family-split checkpoint can resume after metadata migration,
        # with no new teacher or judge calls and unchanged recorded trajectory.
        record["spec"]["split"] = "train"
        (self.output / "records/circle-c0.json").write_text(json.dumps(record))
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["specs"][0]["split"] = "train"
        manifest_path.write_text(json.dumps(manifest))
        (self.output / "train.jsonl").write_text(compact_json(record) + "\n")
        remove_splits(self.output)
        rerun = await generate_dataset(options)
        self.assertEqual(rerun, summary); self.assertEqual(len(teacher.messages), 2)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            await generate_dataset({**options, "threshold": 0.9})

    async def test_without_judge_output_is_retained_for_manual_review(self) -> None:
        one = example_spec()
        summary = await generate_dataset({"output": str(self.output), "specs": [one], "teacher": ScriptedClient(['{"code":"draw()"}']),
                                          "candidates": 1, "renderer": self.renderer(), "config": self.config()})
        self.assertEqual(summary["review"], 1); self.assertEqual(summary["accepted"], 0)
        self.assertFalse((self.output / "accepted.jsonl").exists())
        full = [json.loads(line) for line in (self.output / "candidates.jsonl").read_text().splitlines()]
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0]["status"], "review")
        self.assertIn("evaluation pending", (self.output / "records" / f"{one['id']}-c0.json").read_text())

    async def test_authored_harmful_refusal_pair_is_trainable_without_renderer(self) -> None:
        spec = validate_specs([{
            "id": "harmful-1", "family": "safety_refusal",
            "prompt": "Draw a poster glorifying violence and encouraging attacks on people.",
            "seed": 21, "skills": ["refusal"], "kind": "refusal",
            "refusal": "I can't make content that promotes violence.",
        }])[0]
        summary = await generate_dataset({"output": str(self.output), "specs": [spec],
                                          "teacher": None, "candidates": 3})
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["refusalPairs"], 1)
        self.assertEqual(summary["requests"], 0)
        self.assertEqual(len(list((self.output / "records").glob("*.json"))), 1)
        row = json.loads((self.output / "candidates.jsonl").read_text())
        self.assertEqual(row["refusal"], spec["refusal"])
        self.assertEqual(row["artifacts"], {})
        self.assertEqual(row["generation"]["initialDraft"], "authored_refusal")

    async def test_duplicate_rendered_pixels_are_retained_for_manual_review(self) -> None:
        specs = [
            {"id": "circle", "family": "circles", "prompt": "Draw a blue circle", "seed": 1},
            {"id": "house", "family": "houses", "prompt": "Draw a blue house", "seed": 2},
        ]
        correctness = ScriptedClient([self.correctness()] * 4, model="correctness")
        safety = ScriptedClient([self.safety()] * 4, model="safety")
        summary = await generate_dataset({"output": str(self.output), "specs": specs,
                                          "teacher": ScriptedClient(['{"code":"circle source"}', '{"code":"house source"}']),
                                          "correctnessJudge": correctness, "safetyJudge": safety,
                                          "codeSafetyJudge": ScriptedClient(['{"malicious":false,"reason":"Drawing only"}'] * 2, model="code-safety"),
                                          "candidates": 1, "renderer": self.renderer(), "config": self.config()})
        self.assertEqual(summary["accepted"], 2); self.assertEqual(summary["rejected"], 0)
        candidates = [json.loads(line) for line in (self.output / "candidates.jsonl").read_text().splitlines()]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["pixelHash"], candidates[1]["pixelHash"])
        self.assertTrue(all(r["status"] == "accepted" for r in candidates))
        self.assertNotIn("split", candidates[0]["spec"])
        for name in ("accepted", "review", "rejected", "train", "validation", "test"):
            self.assertFalse((self.output / f"{name}.jsonl").exists())

    async def test_malformed_teacher_reply_uses_bounded_repairs_and_lock_is_enforced(self) -> None:
        one = example_spec()
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

        one = example_spec()
        teacher = ScriptedClient(['{"code":"draw()"}'])
        correctness = ScriptedClient([self.correctness()], model="correctness")
        safety = ScriptedClient([self.safety(True, "Unsafe content")], model="safety")
        summary = await generate_dataset({"output": str(self.output), "specs": [one], "teacher": teacher,
                                          "correctnessJudge": correctness, "safetyJudge": safety,
                                          "candidates": 1, "renderer": CountingRenderer(), "config": self.config()})
        self.assertEqual(summary["rejected"], 1); self.assertEqual(calls, ["draw()"])
        row = json.loads((self.output / "records" / f"{one['id']}-c0.json").read_text())
        self.assertEqual(row["evaluations"] and row["status"], "rejected")
        self.assertFalse(any('"operation":"submit"' in line for line in Path(self.output / row["artifacts"]["trajectory"]).read_text().splitlines()))

    async def test_refusals_cannot_pass_image_only_inspect_judges(self) -> None:
        spec = {"id": "refusal", "family": "circles", "split": "train",
                "prompt": "Draw a blue circle", "seed": 4}
        teacher = ScriptedClient(['{"refusal":"I cannot draw circles."}'])
        correctness = ScriptedClient([], model="correctness")
        safety = ScriptedClient([], model="safety")
        summary = await generate_dataset({"output": str(self.output), "specs": [spec],
                                          "teacher": teacher, "correctnessJudge": correctness,
                                          "safetyJudge": safety, "candidates": 1,
                                          "renderer": self.renderer(), "config": self.config()})
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(summary["requests"], 1)
        row = json.loads((self.output / "records/refusal-c0.json").read_text())
        self.assertNotIn("code", row)
        self.assertEqual(row["evaluations"], [])
        self.assertIn('"operation":"refuse"',
                      Path(self.output / row["artifacts"]["trajectory"]).read_text())

    async def test_both_judges_required_when_grading(self) -> None:
        spec = example_spec()
        for judges in ({"correctnessJudge": ScriptedClient([], model="correctness")},
                       {"safetyJudge": ScriptedClient([], model="safety")}):
            with self.assertRaisesRegex(ValueError, "both be configured"):
                await generate_dataset({"output": str(self.output), "specs": [spec],
                                        "teacher": ScriptedClient([]), **judges})

    async def test_visual_repair_sends_exact_preview_and_reviewer_flip_cannot_fake_repair(self) -> None:
        import io as image_io
        from PIL import Image

        spec = {"id": "visual-repair", "family": "circles", "prompt": "Draw a blue circle", "seed": 13}

        def color_png(color: tuple[int, int, int]) -> bytes:
            buff = image_io.BytesIO(); Image.new("RGB", (32, 32), color).save(buff, format="PNG"); return buff.getvalue()

        red, blue = color_png((255, 0, 0)), color_png((0, 0, 255))
        seen_pngs: list[bytes] = []

        class ColorRenderer:
            async def render(inner, code: str, seed: int) -> dict[str, Any]:
                png = red if code == "red" else blue
                return {"ok": True, "stdout": "", "stderr": "", "png": png, "width": 32, "height": 32}

        class VisualCorrectness:
            model, baseUrl = "vision-correct", "http://judge/v1"
            async def complete(inner, messages: list[dict[str, Any]], max_tokens: int, *,
                               temperature: float | None = None) -> dict[str, Any]:
                self.assertIsNone(temperature)
                self.assertEqual(max_tokens, JUDGE_MAX_TOKENS)
                parts = messages[0]["content"]
                image_part = next((p for p in parts if p["type"] == "image_url"), None)
                if image_part:
                    seen_pngs.append(base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1]))
                score = 0 if len(seen_pngs) == 1 else 1
                return completion(self.correctness(score, "The preview is red" if score == 0 else "The preview is blue"))

        teacher = ScriptedClient(['{"code":"red"}', '{"code":"blue"}'])
        summary = await generate_dataset({"output": str(self.output), "specs": [spec], "imageObservations": True,
                                          "candidates": 1, "teacher": teacher,
                                          "correctnessJudge": VisualCorrectness(),
                                          "codeSafetyJudge": ScriptedClient(['{"malicious":false,"reason":"Drawing only"}'] * 2, model="code-safety"),
                                          "safetyJudge": ScriptedClient([self.safety()] * 3, model="vision-safety"),
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

        class FlipCorrectness:
            model, baseUrl = "vision-correct", "http://judge/v1"
            calls = 0
            async def complete(inner, messages: list[dict[str, Any]], max_tokens: int, *,
                               temperature: float | None = None) -> dict[str, Any]:
                inner.calls += 1
                score = 0.2 if inner.calls == 1 else 1
                return completion(self.correctness(score, "review"))

        flip_judge = FlipCorrectness()
        repeated_renderer = self.renderer(red)
        same_summary = await generate_dataset({"output": str(self.output), "specs": [spec], "imageObservations": True,
                                               "candidates": 1, "teacher": flip_teacher,
                                               "correctnessJudge": flip_judge,
                                               "safetyJudge": ScriptedClient([self.safety()] * 2, model="vision-safety"),
                                               "renderer": repeated_renderer, "config": self.config()})
        self.assertEqual(same_summary["rejected"], 1); self.assertEqual(flip_judge.calls, 2)
        same_row = json.loads((self.output / "records/visual-repair-c0.json").read_text())
        self.assertIn("identical", same_row["reason"]); self.assertFalse(same_row["generation"]["finalImagePassed"])

    async def test_infrastructure_retry_retains_previous_attempt_and_usage(self) -> None:
        spec = example_spec()

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

    async def test_one_shot_success_is_valid_training_data(self) -> None:
        spec = {"id": "one-shot", "family": "circles", "prompt": "Draw a blue circle", "seed": 20}
        summary = await generate_dataset({"output": str(self.output), "specs": [spec], "candidates": 1,
                                          "teacher": ScriptedClient(['{"code":"draw()"}']),
                                          "correctnessJudge": ScriptedClient([self.correctness()] * 2, model="correctness"),
                                          "codeSafetyJudge": ScriptedClient(['{"malicious":false,"reason":"Drawing only"}'], model="code-safety"),
                                          "safetyJudge": ScriptedClient([self.safety()] * 2, model="safety"),
                                          "renderer": self.renderer(), "config": self.config()})
        self.assertEqual(summary["accepted"], 1)
        row = json.loads((self.output / "records/one-shot-c0.json").read_text())
        self.assertEqual(row["generation"]["executionFailures"], 0)


if __name__ == "__main__":
    unittest.main()
