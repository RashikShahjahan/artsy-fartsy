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
            if isinstance(messages[-1]["content"], list):
                return completion('{"submit":true}')
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
        self.assertEqual(len(validate_specs([spec, {**spec, "id": "other"}])), 2)
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
    async def test_openrouter_preserves_reasoning_without_changing_json_answer(self):
        import httpx
        from drawing.dataset_client import ChatClient, OPENROUTER_MODEL_URL
        details = [{"type": "reasoning.text", "text": "The circle needs to be larger."},
                   {"type": "reasoning.encrypted", "data": "opaque"}]
        for fields in ({"reasoning": "Review the preview.", "reasoning_details": details},
                       {"reasoning_details": details}, {}):
            requests = []
            def respond(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={"choices": [{"message": {
                    "content": '{"submit":true}', **fields}}]})
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
                result = await ChatClient("teacher", OPENROUTER_MODEL_URL, http_client=http).complete([], 100)
            self.assertEqual(result["text"], '{"submit":true}')
            self.assertEqual(requests[0]["reasoning"], {"enabled": True, "exclude": False})
            if fields:
                self.assertEqual(result["reasoning"], fields.get("reasoning", details[0]["text"]))
                self.assertEqual(result["reasoning_details"], details)
            else:
                self.assertNotIn("reasoning", result)

    async def test_openai_chat_protocol_and_credential_redaction(self):
        import httpx
        calls = []
        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={"model": "revision-1", "choices": [{"message": {
                "content": '{"code":"hello"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = chat_client({"model": "teacher", "baseUrl": "http://localhost/v1/", "apiKey": "test-secret"})
            client.http_client = http
            result = await client.complete([{"role": "user", "content": "draw"}], 100, temperature=0)
            self.assertEqual(result["model"], "revision-1")
            self.assertEqual(result["usage"]["completion_tokens"], 5)
            self.assertEqual(str(calls[0].url), "http://localhost/v1/chat/completions")
            body = json.loads(calls[0].content)
            self.assertEqual((body["max_tokens"], body["stream"], body["temperature"]), (100, False, 0))
            self.assertEqual(calls[0].headers["Authorization"], "Bearer test-secret")
            self.assertNotIn("test-secret", repr(client))
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(401, text="test-secret"))) as http:
            client.http_client = http
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


class ToolTeacher:
    model, baseUrl = 'scripted-vision', 'http://localhost/v1'

    def __init__(self, actions):
        self.actions, self.messages, self.schemas = list(actions), [], []

    async def complete(self, messages, max_tokens=None, *, tools):
        self.messages.append(json.loads(json.dumps(messages)))
        self.schemas.append(tools)
        action = self.actions.pop(0) if self.actions else None
        if isinstance(action, Exception):
            raise action
        result = completion('')
        result['reasoning'] = f'Thought {len(self.messages)}'
        if action is not None:
            name, args = action
            result['tool_calls'] = [{'id': f'call_{len(self.messages)}', 'type': 'function',
                                     'function': {'name': name, 'arguments': json.dumps(args)}}]
        return result


def update(source='draw()'):
    return ('update_drawing', {'source': source})


def preview():
    return ('preview_drawing', {})


def submit():
    return ('submit_drawing', {})


class DatasetGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from drawing.environment import ENVIRONMENT_CONFIG
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'dataset'
        self.config = {**ENVIRONMENT_CONFIG, 'width': 32, 'height': 32}
        self.render_calls = []
        owner = self
        class Renderer:
            async def render(self, code, seed):
                owner.render_calls.append((code, seed))
                if code == 'broken':
                    return {'ok': False, 'stderr': 'SyntaxError', 'failureKind': 'invalid_code'}
                buf = io.BytesIO()
                Image.new('RGB', (32, 32), 'red' if code == 'first' else 'blue').save(buf, format='PNG')
                return {'ok': True, 'png': buf.getvalue()}
        self.renderer = Renderer()

    def options(self, teacher):
        return {'output': str(self.output), 'specs': [example_spec()], 'teacher': teacher,
                'candidates': 1, 'renderer': self.renderer, 'config': self.config}

    def record(self):
        return json.loads((self.output / 'candidates.jsonl').read_text())

    async def test_native_tools_preview_pixels_export_and_loss_mask(self):
        from training.data import convert_record
        from drawing.agent_contract import DRAWING_TOOLS
        from ui.candidate_review import export_approved, image_paths
        teacher = ToolTeacher([update('first'), preview(), update('fixed'), preview(), submit()])
        summary = await generate_dataset(self.options(teacher))
        self.assertEqual((summary['review'], summary['accepted'], summary['requests']), (1, 0, 5))
        self.assertTrue(all(schema == DRAWING_TOOLS for schema in teacher.schemas))
        self.assertEqual(len(self.render_calls), 3)  # two previews plus fresh final render
        row = self.record()
        self.assertNotIn('data:image', json.dumps(row))
        self.assertEqual(row['code'], 'fixed')
        self.assertEqual(row['generation']['modelTurns'], 5)
        for call_index, color in ((2, (255, 0, 0)), (4, (0, 0, 255))):
            part = teacher.messages[call_index][-1]['content'][1]
            png = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
            self.assertEqual(Image.open(io.BytesIO(png)).getpixel((0, 0)), color)
        converted = convert_record(row, self.output)
        names = [converted['messages'][i]['tool_calls'][0]['function']['name']
                 for i in converted['trainable_message_indices']]
        self.assertEqual(names, ['update_drawing', 'preview_drawing', 'submit_drawing'])
        self.assertEqual(converted['messages'][converted['trainable_message_indices'][0]]['tool_calls'][0]['function']['arguments'], {'source': 'fixed'})
        self.assertEqual(sum(m['role'] == 'assistant' for m in converted['messages']), 5)
        self.assertEqual(len(convert_record(row, self.output, 'assistant-only')['trainable_message_indices']), 5)
        self.assertEqual(len(image_paths(row, self.output)), 3)
        review = {'reviewer': 'Human', 'decisions': {row['id']: {'decision': 'approved'}}}
        self.assertEqual(export_approved([row], review, self.output, self.output/'accepted.jsonl'), 1)

    async def test_repairs_resume_and_no_automatic_judges(self):
        from training.data import is_observed_repair, load_examples
        teacher = ToolTeacher([update('broken'), preview(), update(), preview(), submit()])
        judges = {name: ToolTeacher([]) for name in ('correctnessJudge', 'safetyJudge', 'codeSafetyJudge')}
        options = {**self.options(teacher), **judges}
        summary = await generate_dataset(options)
        self.assertEqual(summary['review'], 1)
        self.assertEqual(self.record()['generation']['executionFailures'], 1)
        self.assertTrue(is_observed_repair(self.record()))
        examples, manifest = load_examples(self.output / 'candidates.jsonl')
        self.assertEqual((len(examples), manifest['repairs'], manifest['one_shot'],
                          manifest['omitted_unexecuted_replies']), (1, 1, 0, 0))
        self.assertTrue(all(not j.messages for j in judges.values()))
        self.assertEqual(await generate_dataset(options), summary)
        self.assertEqual(len(teacher.messages), 5)
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            await generate_dataset({**options, 'candidates': 2})
        (self.output/'.generation.lock').write_text('active')
        with self.assertRaisesRegex(RuntimeError, 'locked'):
            await generate_dataset(options)

    async def test_submission_on_twelfth_turn_accepted_and_thirteenth_not_requested(self):
        from training.data import convert_record
        teacher = ToolTeacher([update(), preview()] + [('read_drawing', {})]*9 + [submit()])
        summary = await generate_dataset(self.options(teacher))
        self.assertEqual(summary['review'], 1)
        self.assertEqual(self.record()['generation']['modelTurns'], 12)
        self.assertEqual(sum(m['role'] == 'assistant' for m in convert_record(self.record(), self.output)['messages']), 12)
        other = self.output.parent/'over-budget'
        late = ToolTeacher([update(), preview()] + [('read_drawing', {})]*10 + [submit()])
        summary = await generate_dataset({**self.options(late), 'output': str(other)})
        self.assertEqual((summary['rejected'], summary['review'], len(late.messages)), (1, 0, 12))
        row = json.loads((other/'candidates.jsonl').read_text())
        self.assertIn('12 model turns', row['reason'])
        self.assertNotIn('submission', row)
        with self.assertRaisesRegex(ValueError, 'no completed submission'):
            convert_record(row, other)

    async def test_export_cannot_override_thirteen_turn_rejection(self):
        from drawing.tool_dataset import trajectory_hash, text_message
        from ui.candidate_review import export_approved
        teacher = ToolTeacher([update(), preview(), submit()])
        await generate_dataset(self.options(teacher))
        row = self.record()
        messages = row['drawingTrajectory']['messages']
        for _ in range(10):
            messages.insert(2, text_message('assistant', 'invalid reply'))
        row['generation']['modelTurns'] = 13
        row['trace'] = row['trace'] * 4 + row['trace'][:1]
        row['drawingTrajectoryHash'] = trajectory_hash(row['drawingTrajectory'])
        review = {'reviewer': 'Human', 'decisions': {row['id']: {'decision': 'approved'}}}
        with self.assertRaisesRegex(ValueError, 'exceeds 12'):
            export_approved([row], review, self.output, self.output/'accepted.jsonl')
        self.assertFalse((self.output/'accepted.jsonl').exists())

    async def test_invalid_and_empty_replies_count_toward_turn_limit(self):
        teacher = ToolTeacher([None, ('unknown_tool', {}), ('update_drawing', {'bad': 'arg'})] * 4)
        summary = await generate_dataset(self.options(teacher))
        self.assertEqual((summary['requests'], summary['rejected'], summary['infrastructureErrors']), (12, 1, 0))
        self.assertEqual(self.record()['generation']['modelTurns'], 12)
        self.assertEqual(self.render_calls, [])

    async def test_submit_requires_current_preview(self):
        teacher = ToolTeacher([submit(), update(), preview(), update('fixed'), submit(), preview(), submit()])
        summary = await generate_dataset(self.options(teacher))
        self.assertEqual(summary['review'], 1)
        self.assertIn('before submitting', teacher.messages[1][-1]['content'][0]['text'])
        self.assertIn('before submitting', teacher.messages[5][-1]['content'][0]['text'])

    async def test_refusals_are_single_terminal_named_tool(self):
        from training.data import convert_record
        teacher = ToolTeacher([('refuse_drawing', {'reason': 'Cannot fulfill this request.'})])
        summary = await generate_dataset(self.options(teacher))
        self.assertEqual((summary['review'], summary['requests']), (1, 1))
        example = convert_record(self.record(), self.output)
        self.assertEqual(example['trainable_message_indices'], [2])
        self.assertEqual(example['messages'][2]['tool_calls'][0]['function']['name'], 'refuse_drawing')
        self.assertFalse(self.render_calls)
        spec = validate_specs([{'id': 'refusal', 'family': 'safety_refusal', 'prompt': 'Unsafe request',
                               'seed': 1, 'kind': 'refusal', 'refusal': 'Cannot fulfill.'}])[0]
        output = self.output.parent/'authored'
        summary = await generate_dataset({'output': str(output), 'specs': [spec], 'teacher': None})
        row = json.loads((output/'candidates.jsonl').read_text())
        self.assertEqual(summary['requests'], 0)
        self.assertEqual(convert_record(row, output)['messages'][2]['tool_calls'][0]['function']['name'], 'refuse_drawing')

    async def test_infrastructure_retry_preserves_attempts(self):
        teacher = ToolTeacher([RuntimeError('Provider down'), update(), preview(), submit()])
        options = self.options(teacher)
        self.assertEqual((await generate_dataset(options))['infrastructureErrors'], 1)
        self.assertEqual((await generate_dataset(options))['infrastructureErrors'], 1)
        summary = await generate_dataset({**options, 'retryErrors': True})
        self.assertEqual((summary['review'], summary['requests']), (1, 3))
        self.assertEqual(len(self.record()['previousAttempts']), 1)

    async def test_artifact_and_transcript_integrity(self):
        from training.data import convert_record
        from drawing.tool_dataset import trajectory_hash
        teacher = ToolTeacher([update(), preview(), submit()])
        await generate_dataset(self.options(teacher))
        row = self.record()
        row['drawingTrajectory']['messages'][2]['tool_calls'][0]['function']['arguments']['source'] = 'modified'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            convert_record(row, self.output)
        row['drawingTrajectoryHash'] = trajectory_hash(row['drawingTrajectory'])
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            convert_record(row, self.output)
        row = self.record()
        (self.output/row['artifacts']['source']).write_text('modified')
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            convert_record(row, self.output)

    async def test_vision_preflight_and_no_legacy_draft_reuse(self):
        from drawing.dataset_client import TINKER_MODEL_URL
        teacher = ToolTeacher([])
        teacher.model, teacher.baseUrl = 'deepseek-ai/DeepSeek-V4.1', TINKER_MODEL_URL
        with self.assertRaisesRegex(ValueError, 'image input'):
            await generate_dataset(self.options(teacher))
        teacher = ToolTeacher([])
        with self.assertRaisesRegex(ValueError, 'Legacy JSON drafts'):
            await generate_dataset({**self.options(teacher), 'draftsFrom': ['old']})


if __name__ == '__main__':
    unittest.main()
