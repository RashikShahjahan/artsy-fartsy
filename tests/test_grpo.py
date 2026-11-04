"""Cookbook RL integration tests (no Docker/Tinker service required)."""

import asyncio
import base64
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from training.cookbook_env import (ArtsyGroupBuilder, DrawingTool, DockerSession,
                                   _preview_observation, configure_concurrency)
from training.grpo import (artsy_final_checkpoint, make_cookbook_config,
                           parser, write_final_checkpoint)


def response(code=0, output=b"", stderr=b""):
    return SimpleNamespace(returncode=code, stdout=output, stderr=stderr)


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_reads_current_bounded_files_in_one_exec(self):
        import subprocess
        session = DockerSession({}, Path('/tmp'), {})
        session.container = 'workspace'
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / 'program.py').write_bytes(b'x' * 70000)
            (directory / 'refusal.json').write_text(json.dumps({'reason': 'Declined'}))

            async def docker(argv, **kwargs):
                self.assertEqual(argv[:4], ['exec', 'workspace', 'python', '-c'])
                self.assertEqual(kwargs['timeout'], 10)
                script = argv[-1].replace('/workspace/', root + '/')
                return await asyncio.to_thread(
                    subprocess.run, [sys.executable, '-c', script], capture_output=True, check=True)

            with patch('training.cookbook_env._docker', new=AsyncMock(side_effect=docker)) as mock:
                source, refusal = await session._read_submission()
                self.assertEqual(len(source.encode()), 65537)
                self.assertEqual(refusal, 'Declined')
                mock.assert_awaited_once()
                (directory / 'program.py').write_bytes(b'changed')
                (directory / 'refusal.json').write_bytes(b'{broken')
                self.assertEqual(await session._read_submission(), ('changed', ''))
                for invalid in (b'\xff', json.dumps({'reason': 'x' * 2049}).encode(),
                                b' ' * 5000 + b'{}'):
                    (directory / 'refusal.json').write_bytes(invalid)
                    self.assertEqual(await session._read_submission(), ('changed', ''))
                (directory / 'refusal.json').unlink()
                self.assertEqual(await session._read_submission(), ('changed', None))
                (directory / 'program.py').write_bytes(b'\xff')
                with self.assertRaises(UnicodeDecodeError):
                    await session._read_submission()

    async def test_preview_processing_runs_outside_event_loop_thread(self):
        import threading
        session = DockerSession({'seed': 1}, Path('/tmp'), {})
        session.container = 'workspace'
        session.start = AsyncMock()
        session._read_source = AsyncMock(return_value='draw()')
        loop_thread = threading.get_ident()

        def preview(*args):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            return 'observation', {'ok': True}

        with patch('training.cookbook_env._docker', new=AsyncMock(return_value=response())), \
             patch('training.cookbook_env._preview_observation', side_effect=preview):
            self.assertEqual(await session.execute_tool('preview_drawing', {}), ('observation', False))
        self.assertEqual(session.previews, [{'ok': True}])

    async def test_dataset_builds_only_final_image_client(self):
        from training.cookbook_dataset import ArtsyDatasetBuilder
        builder = ArtsyDatasetBuilder(
            specs=({"id": "draw", "prompt": "Draw a circle", "seed": 1},),
            output_dir="/tmp/artsy-dataset-test", batch_size=1, group_size=2,
            epochs=1, seed=42, max_tokens=128, max_length=32768,
            model="Qwen/Qwen3.8-27B", renderer="qwen3_8_disable_thinking",
            revision=None,
            normalize_rewards=False, hpsv3_model="weights", hpsv3_url="https://judge.example")
        with patch("training.cookbook_dataset.HPSv3Client") as client:
            dataset, evaluation = await builder()
        self.assertEqual(set(dataset.judge), {"hpsv3"})
        self.assertEqual(client.call_count, 1)
        self.assertIsNone(evaluation)

    async def test_tools_use_shared_schemas_and_propagate_termination(self):
        from drawing.agent_contract import DRAWING_TOOLS
        session = DockerSession({}, Path('/tmp'), {})
        session.execute_tool = AsyncMock(side_effect=[('updated', False), ('submitted', True)])
        tools = {definition['function']['name']: DrawingTool(session, definition['function'])
                 for definition in DRAWING_TOOLS}
        self.assertNotIn('bash', tools)
        for definition in DRAWING_TOOLS:
            self.assertEqual(tools[definition['function']['name']].to_spec(), definition['function'])
        result = await tools['update_drawing'].run(SimpleNamespace(arguments={'source': 'draw()'}, call_id='first'))
        self.assertFalse(result.should_stop)
        self.assertEqual(result.messages[0]['tool_call_id'], 'first')
        result = await tools['submit_drawing'].run(SimpleNamespace(arguments={}, call_id='last'))
        self.assertTrue(result.should_stop)
        self.assertEqual(result.messages[0]['name'], 'submit_drawing')
        self.assertEqual(session.execute_tool.call_args.args, ('submit_drawing', {}))

    async def test_combined_rewards_adjust_existing_totals(self):
        spec = {"id": "draw", "prompt": "Draw a circle", "seed": 1}
        judge = {"visual": SimpleNamespace(compare=AsyncMock(side_effect=["A", "B"]))}
        with tempfile.TemporaryDirectory() as root:
            builder = ArtsyGroupBuilder(spec, 2, 1, Path(root), judge, 128, 32768, False)
            for index in range(2):
                session = DockerSession(spec, Path(root) / str(index), judge)
                session.submitted = True
                session.png = b"image" + bytes([index])
                session.result = {"scores": {"render_valid": 1.0, "preference": 0.625 + index * 0.125}}
                builder.sessions.append(session)
            trajectories = [SimpleNamespace(transitions=[SimpleNamespace(reward=0.2)]) for _ in range(2)]
            adjustments = await builder.compute_group_rewards(trajectories, [])
            self.assertEqual([round(0.2 + delta, 4) for delta, _ in adjustments], [0.625, 0.75])
            judge["visual"].compare.assert_not_awaited()
            self.assertTrue(all("visual" not in metrics for _, metrics in adjustments))
            self.assertTrue(all("visual" not in session.result["scores"] for session in builder.sessions))

    async def test_refusal_has_no_image_reward_or_judge_calls(self):
        with tempfile.TemporaryDirectory() as root:
            spec = {"id": "r", "prompt": "request", "seed": 1, "kind": "refusal"}
            judge = {"visual": SimpleNamespace(complete=AsyncMock())}
            session = DockerSession(spec, Path(root), judge)
            session.submitted = True
            with patch.object(session, "_read_submission", new=AsyncMock(return_value=("", "I cannot help"))):
                await session.grade([])
            builder = ArtsyGroupBuilder(spec, 1, 1, Path(root), judge, 128, 32768, False)
            builder.sessions = [session]
            result, = await builder.compute_group_rewards([SimpleNamespace(transitions=[])], [])
            self.assertEqual(result[0], 0.0)
            judge["visual"].complete.assert_not_awaited()

    async def test_unsubmitted_source_never_reaches_renderer(self):
        with tempfile.TemporaryDirectory() as root:
            session = DockerSession({"id": "x", "prompt": "Draw", "seed": 1}, Path(root), {})
            with patch.object(session, "_read_submission", new=AsyncMock(return_value=("draw()", None))), \
                 patch("training.cookbook_env._docker", new=AsyncMock()) as docker:
                await session.grade([])
            docker.assert_not_awaited()
            self.assertEqual(session.result["scores"]["render_valid"], 0.0)
            self.assertIsNone(session.png)

    async def test_unsubmitted_image_gets_zero_reward_without_visual_judge(self):
        spec = {"id": "x", "prompt": "Draw", "seed": 1}
        judge = {"visual": SimpleNamespace(compare=AsyncMock(return_value="Tie"))}
        with tempfile.TemporaryDirectory() as root:
            builder = ArtsyGroupBuilder(spec, 2, 1, Path(root), judge, 128, 32768, False)
            for index in range(2):
                session = DockerSession(spec, Path(root) / str(index), judge)
                session.submitted = index == 0
                session.png = b"image"
                session.result = {"scores": {"render_valid": 1., "preference": 1.}}
                builder.sessions.append(session)
            adjustments = await builder.compute_group_rewards(
                [SimpleNamespace(transitions=[]) for _ in range(2)], [])
            self.assertEqual([reward for reward, _ in adjustments], [1., 0.])
            judge["visual"].compare.assert_not_awaited()

    async def test_fresh_trusted_render_and_correctness_grading(self):
        spec = {"id": "draw", "prompt": "Draw a circle", "seed": 1}
        png_file = BytesIO()
        Image.new("RGB", (1920, 1200)).save(png_file, "PNG")
        png = png_file.getvalue()
        judge = {"hpsv3": SimpleNamespace(score=AsyncMock(return_value={
            "hpsv3_score": 2.5, "preference": 0.625, "model": "weights"}))}
        with tempfile.TemporaryDirectory() as root:
            session = DockerSession(spec, Path(root), judge)
            session.container = "fake-container"
            session.submitted = True

            async def docker(argv, **kwargs):
                if argv[0] == "exec":
                    return response(output=json.dumps([
                        base64.b64encode(b"print('draw')\n").decode(), ""]).encode())
                if argv[0] == "run":
                    self.assertIn("/opt/artsy/score_rl.py", argv)
                    self.assertEqual(kwargs["stdin"], b"print('draw')\n")
                    return response(output=json.dumps({
                        "feedback": {"ok": True}, "image": base64.b64encode(png).decode()
                    }).encode())
                return response()

            with patch("training.cookbook_env._docker", new=AsyncMock(side_effect=docker)):
                reward, metrics = await session.grade([])
            self.assertEqual(reward, 0.0)
            self.assertEqual(metrics["render_valid"], 1.0)
            self.assertEqual((Path(root) / "image.png").read_bytes(), png)
            self.assertEqual(json.loads((Path(root) / "result.json").read_text())
                             ["scores"]["hpsv3"], 2.5)
            self.assertEqual(session.result["scores"]["preference"], 0.625)
            builder = ArtsyGroupBuilder(spec, 1, 1, Path(root), judge, 128, 32768, False)
            builder.sessions = [session]
            adjustment, = await builder.compute_group_rewards([SimpleNamespace(transitions=[])], [])
            self.assertEqual(adjustment[0], 0.625)
            self.assertEqual(json.loads((Path(root) / "result.json").read_text())["scores"]["reward"], 0.625)
            judge["hpsv3"].score.assert_awaited_once_with(spec["prompt"], png)



class PreviewTests(unittest.TestCase):
    def test_png_preview_becomes_image_part_with_feedback(self):
        output = BytesIO()
        Image.new("RGB", (24, 16), "navy").save(output, "PNG")
        encoded = base64.b64encode(output.getvalue()).decode()
        marker = ("<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>"
                  f"data:image/png;base64,{encoded}</MSWEA_MULTIMODAL_CONTENT>")
        payload = (json.dumps({"ok": True, "sourceHash": "f" * 64}) + "\n" + marker).encode()
        resize = ModuleType("transformers.models.qwen2_vl.image_processing_qwen2_vl")
        resize.smart_resize = lambda height, width, **kwargs: (height, width)
        modules = {name: ModuleType(name) for name in (
            "transformers", "transformers.models", "transformers.models.qwen2_vl")}
        modules[resize.__name__] = resize
        with patch.dict(sys.modules, modules), patch("training.cookbook_env.encoder") as get_encoder:
            get_encoder.return_value.image_processor.patch_size = 14
            get_encoder.return_value.image_processor.merge_size = 2
            parts, feedback = _preview_observation(payload, b"", 0)
        self.assertEqual(feedback["sourceHash"], "f" * 64)
        self.assertEqual([part["type"] for part in parts], ["text", "image"])
        self.assertEqual(parts[1]["image"].size, (24, 16))
        self.assertNotIn("data:image/png", parts[0]["text"])

    def test_text_feedback_without_preview(self):
        observation, feedback = _preview_observation(b'{"ok":false}\n', b"SyntaxError", 1)
        self.assertIn("Return code: 1", observation)
        self.assertIn("SyntaxError", observation)
        self.assertIsNone(feedback)

    def test_invalid_image_payload_is_rejected(self):
        marker = ("<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>"
                  "data:image/png;base64,AAAA</MSWEA_MULTIMODAL_CONTENT>")
        # The renderer's marker is not arbitrary text or a valid PNG.
        with self.assertRaises(ValueError):
            _preview_observation(marker.encode(), b"", 0)


class ConfigurationTests(unittest.TestCase):
    def test_recover_final_json_without_rerunning_training(self):
        record = SimpleNamespace(name="final", final=None, batch=1,
                                 state_path="tinker://run/weights/final",
                                 sampler_path="tinker://run/sampler_weights/final")
        package = ModuleType("tinker_cookbook")
        package.checkpoint_utils = SimpleNamespace(get_last_checkpoint=lambda _: record)
        plan = {"algorithm": "tinker-cookbook-ppo-v1", "model": "Qwen/Qwen3.5-4B"}
        with tempfile.TemporaryDirectory() as root, patch.dict(
                sys.modules, {"tinker_cookbook": package}):
            output_dir = Path(root)
            result = write_final_checkpoint(output_dir, plan)
            self.assertEqual(json.loads((output_dir / "final.json").read_text()), result)
            self.assertEqual(write_final_checkpoint(output_dir, plan), result)
            with self.assertRaisesRegex(ValueError, "incompatible final.json"):
                write_final_checkpoint(output_dir, {**plan, "model": "other"})
            self.assertEqual(json.loads((output_dir / "final.json").read_text()), result)

    def test_cookbook_final_record_does_not_require_optional_final_flag(self):
        record = SimpleNamespace(name="final", final=None, batch=1,
                                 state_path="tinker://run/weights/final",
                                 sampler_path="tinker://run/sampler_weights/final")
        plan = {"model": "Qwen/Qwen3.5-4B"}
        result = artsy_final_checkpoint(record, plan)
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["sampler_path"], record.sampler_path)
        self.assertIs(result["config"], plan)
        with self.assertRaisesRegex(RuntimeError, "final state"):
            artsy_final_checkpoint(SimpleNamespace(
                name="step-000001", batch=1, state_path=record.state_path,
                sampler_path=record.sampler_path), plan)

    def test_defaults_match_existing_agent_protocol(self):
        args = parser().parse_args([])
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.max_samples, 32)
        from training.data import ROOT
        self.assertEqual(args.dataset, ROOT / "datasets/parti-prompts/rl/prompts.jsonl")
        self.assertIsNone(args.max_tokens)
        self.assertIsNone(parser().parse_args(['--max-tokens', '0']).max_tokens)
        self.assertEqual(args.max_length, 262144)
        self.assertEqual(args.init_checkpoint.name, 'final.json')
        self.assertEqual(args.init_checkpoint.parent.name,
                         'sft-v2-qwen3.8-27b-final-preview-masked-cosine')

    def test_sft_checkpoint_initializes_rl_and_other_stage_is_rejected(self):
        from drawing.config import MODEL_ID, MODEL_REVISION
        from training.data import ROOT, digest
        from training.sft import RENDERER
        from training.grpo import load_run
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            dataset = directory / 'prompts.jsonl'
            dataset.write_text(json.dumps({'id': 'draw', 'prompt': 'Draw a circle', 'seed': 1}) + '\n')
            checkpoint = directory / 'final.json'
            config = {'model': MODEL_ID, 'tokenizer_revision': MODEL_REVISION,
                      'renderer': RENDERER, 'lora_rank': 32,
                      'api_doc_sha256': digest((ROOT / 'docs/artcanvas.md').read_bytes())}
            initial = {'config': config, 'state_path': 'tinker://sft/state',
                       'sampler_path': 'tinker://sft/sampler'}
            checkpoint.write_text(json.dumps(initial))
            args = parser().parse_args(['--dataset', str(dataset), '--init-checkpoint', str(checkpoint)])
            _, plan, _ = load_run(args)
            self.assertEqual(plan['initial_state_path'], initial['state_path'])
            self.assertEqual(plan['reference_sampler_path'], initial['sampler_path'])
            self.assertEqual(plan['reference_model'], MODEL_ID + ':peft:262144')
            config['model'] = MODEL_ID + ':peft:262144'
            checkpoint.write_text(json.dumps(initial))
            _, plan, _ = load_run(args)
            self.assertEqual(plan['model'], MODEL_ID)
            self.assertEqual(plan['reference_model'], config['model'])
            checkpoint.write_text(json.dumps({**initial, 'config': {**config, 'algorithm': 'another-policy-stage'}}))
            with self.assertRaisesRegex(ValueError, 'compatible SFT'):
                load_run(args)

    @unittest.skipUnless(importlib.util.find_spec("tinker_cookbook"),
                         "Install the pinned training dependencies")
    def test_cookbook_uses_ppo_kl_and_sft_initialization(self):
        args = parser().parse_args(["--max-steps", "1", "--clip-epsilon", "0.1"])
        args.output_dir = Path("/tmp/artsy-rl-example")
        plan = {"initial_state_path": "tinker://initial",
                "reference_model": "Qwen/Qwen3.8-27B:peft:262144",
                "reference_sampler_path": "tinker://reference",
                "model": "Qwen/Qwen3.8-27B",
                "renderer": "qwen3_8_disable_thinking",
                "tokenizer_revision": None,
                "lora_rank": 32, "hpsv3_model": "weights",
                "hpsv3_url": "https://judge.example", "hpsv3_center": 0., "hpsv3_scale": 5.,
                }
        config = make_cookbook_config(
            args, [{"id": "a", "prompt": "Draw a circle", "seed": 1}], plan, 1)
        self.assertEqual(config.loss_fn, "ppo")
        self.assertEqual(config.loss_fn_config["clip_low_threshold"], 0.9)
        self.assertEqual(config.loss_fn_config["clip_high_threshold"], 1.1)
        self.assertEqual(config.load_checkpoint_path, "tinker://initial")
        self.assertEqual(config.kl_reference_config.load_checkpoint_path, "tinker://reference")
        self.assertEqual(config.kl_reference_config.base_model, "Qwen/Qwen3.8-27B:peft:262144")
        self.assertEqual(config.max_tokens, args.max_length)
        self.assertIsNone(config.dataset_builder.max_tokens)
        self.assertTrue(config.termination.zero_reward_on_limit)


class NativeContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_cookbook_runner_accepts_long_prompt_and_clips_near_context(self):
        import tinker
        from tinker_cookbook.completers import TokensWithLogprobs
        from tinker_cookbook.rl.rollouts import do_single_rollout
        from drawing.config import MODEL_CONTEXT_WINDOW
        for length, expected in ((40000, MODEL_CONTEXT_WINDOW - 40000), (MODEL_CONTEXT_WINDOW - 13, 13)):
            configure_concurrency(1, 1)
            renderer = MagicMock()
            renderer.get_stop_sequences.return_value = [2]
            renderer.build_generation_prompt.return_value = tinker.ModelInput.from_ints([1] * length)
            builder = ArtsyGroupBuilder({'id': 'x', 'prompt': 'disk', 'seed': 1},
                                       1, 1, Path('/tmp'), MagicMock(),
                                       None, MODEL_CONTEXT_WINDOW, False)
            calls = []

            async def policy(observation, stop, *, max_tokens=None):
                calls.append((observation.length, max_tokens))
                return TokensWithLogprobs([1] * max_tokens, [0.] * max_tokens, 'length')

            with patch('training.cookbook_env.encoder', return_value=SimpleNamespace(renderer=renderer)), \
                 patch('training.cookbook_env.initial_messages', return_value=[]):
                env, = await builder.make_envs()
                self.assertEqual(env.max_generation_tokens, 1)
                self.assertFalse(env._exceeds_context_limit(262143))
                self.assertTrue(env._exceeds_context_limit(262144))
                self.assertTrue(env._exceeds_context_limit(262145))
                self.assertIsNone(env.rollout_limits.max_sampled_tokens)
                self.assertIsNone(env.rollout_limits.max_turn_tokens)
                self.assertIsNone(env.rollout_limits.max_tool_calls)
                await do_single_rollout(policy, env)
            self.assertEqual(calls, [(length, expected)])


class ToolBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_countdown_tracks_tool_and_no_tool_turns_and_stops_on_last(self):
        from tinker_cookbook.renderers import ToolCall
        configure_concurrency(1, 1)
        builder = ArtsyGroupBuilder({'id': 'x', 'prompt': 'disk', 'seed': 1},
                                   1, 1, Path('/tmp'), MagicMock(), None, 262144, False)
        settings = {'agent': {'step_limit': 3}}
        with patch('training.cookbook_env.encoder', return_value=SimpleNamespace(renderer=MagicMock())), \
             patch('training.cookbook_env.initial_messages', return_value=[]), \
             patch('training.cookbook_env.agent_settings', return_value=settings):
            wrapped, = await builder.make_envs()
        env = wrapped.message_env
        env.reward_fn = AsyncMock(return_value=(0.0, {}))
        builder.sessions[0].execute_tool = AsyncMock(return_value=('ok', False))
        history = await env.initial_observation()
        self.assertEqual(history[-1]['content'], 'Turn 1 of 3.')
        await env.initial_observation()
        self.assertEqual(len(history), 1)
        result = await env.step({'role': 'assistant', 'content': '', 'tool_calls': [
            ToolCall(id='one', function=ToolCall.FunctionBody(
                name='read_drawing', arguments='{}'))]})
        self.assertFalse(result.episode_done)
        self.assertEqual(result.next_messages[-1]['content'], 'Turn 2 of 3.')
        result = await env.step({'role': 'assistant', 'content': 'thinking'})
        self.assertFalse(result.episode_done)
        self.assertIn('Every response must contain a drawing tool call', result.next_messages[-2]['content'])
        self.assertEqual(result.next_messages[-1]['content'],
                         'Turn 3 of 3. This is your final turn; submit your final output now.')
        builder.sessions[0].execute_tool.return_value = ('submitted', True)
        result = await env.step({'role': 'assistant', 'content': '', 'tool_calls': [
            ToolCall(id='last', function=ToolCall.FunctionBody(name='submit_drawing', arguments='{}'))]})
        self.assertTrue(result.episode_done)
        self.assertFalse(any('Turn 4' in str(m.get('content')) for m in result.next_messages))
        env.reward_fn.assert_awaited_once()

    async def test_training_allows_more_than_twenty_calls_until_submission(self):
        from training.cookbook_env import DockerSession
        with tempfile.TemporaryDirectory() as root:
            session = DockerSession({'prompt': 'Draw a disk', 'seed': 1}, Path(root), {})
            session.container = 'fake-container'
            reply = SimpleNamespace(stdout=b'draw()', stderr=b'', returncode=0)
            with patch('training.cookbook_env._docker', new=AsyncMock(return_value=reply)) as docker:
                for index in range(40):
                    _, done = await session.execute_tool('read_drawing', {})
                    self.assertFalse(done)
                _, done = await session.execute_tool('submit_drawing', {})
                self.assertTrue(done)
                self.assertEqual(docker.await_count, 41)
                self.assertTrue(session.submitted)


class RolloutPromptTests(unittest.TestCase):
    def test_initial_messages_announce_turn_budget_without_token_or_call_caps(self):
        from training.cookbook_env import initial_messages
        renderer = SimpleNamespace(create_conversation_prefix_with_tools=lambda *a, **k: [])
        with patch('training.cookbook_env.encoder', return_value=SimpleNamespace(renderer=renderer)):
            messages = initial_messages({'prompt': 'Draw a disk', 'seed': 1}, [])
        content = messages[-1]['content']
        from drawing.agent_contract import drawing_tools_prompt, host_user_prompt
        expected = host_user_prompt(drawing_tools_prompt(), 'Draw a disk', 20)
        self.assertEqual(content, expected)
        self.assertIn('20 model turns', content)
        self.assertIn('Submit your final output by the last turn.', content)
        self.assertNotIn('bash', content)
        self.assertNotIn('/opt/artsy/render.py', content)
        self.assertNotIn('model-turn limit', content)
        self.assertNotIn('32768', content)
        self.assertNotIn('total input and output', content)


if __name__ == "__main__":
    unittest.main()

class RolloutResourceTests(unittest.IsolatedAsyncioTestCase):
    def test_invalid_capacity_is_rejected(self):
        for capacity, group in ((0, 1), (1, 0), (1, 2)):
            with self.subTest(capacity=capacity, group=group), self.assertRaises(ValueError):
                configure_concurrency(capacity, group)

    async def test_cancelled_start_waits_for_docker_then_removes_container(self):
        import threading
        from training.cookbook_env import _docker
        configure_concurrency(1, 1)
        started, finish = threading.Event(), threading.Event()
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            if argv[1] == 'run':
                started.set()
                finish.wait(2)
            return response()
        session = DockerSession({'id': 'x'}, Path('/tmp'), MagicMock())
        with patch('training.cookbook_env._run', side_effect=run):
            task = asyncio.create_task(session.start())
            await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual([call[1] for call in calls], ['run', 'rm'])
        self.assertIsNone(session.container)
        self.assertFalse(session._owns_slot)

    async def test_failed_start_removes_named_container_and_releases_slot(self):
        configure_concurrency(1, 1)
        session = DockerSession({'id': 'x'}, Path('/tmp'), MagicMock())
        with patch('training.cookbook_env._docker', new=AsyncMock(side_effect=[
                response(1, stderr=b'failure'), response()])) as docker:
            with self.assertRaises(RuntimeError):
                await session.start()
            self.assertEqual(docker.await_args_list[1].args[0][0], 'rm')
            self.assertFalse(session._owns_slot)

    def test_subprocess_output_is_bounded_and_drained(self):
        from training.cookbook_env import _run, MAX_OUTPUT
        result = _run([sys.executable, '-c',
                       'import sys; sys.stdout.buffer.write(b"x" * 13000000); '
                       'sys.stderr.buffer.write(b"y" * 13000000)'])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(result.stdout), MAX_OUTPUT)
        self.assertEqual(len(result.stderr), MAX_OUTPUT)

    async def test_failed_group_construction_releases_reservation(self):
        configure_concurrency(2, 2)
        spec = {'id': 'x', 'prompt': 'disk', 'seed': 1}
        first = ArtsyGroupBuilder(spec, 2, 1, Path('/tmp'), MagicMock(), 128, 32768, False)
        second = ArtsyGroupBuilder(spec, 2, 1, Path('/tmp'), MagicMock(), 128, 32768, False)
        with patch('training.cookbook_env.initial_messages', side_effect=ValueError('bad prompt')):
            for builder in (first, second):
                with self.assertRaisesRegex(ValueError, 'bad prompt'):
                    await asyncio.wait_for(builder.make_envs(), timeout=1)
                self.assertFalse(builder._owns_group_slot)
                self.assertEqual(builder.sessions, [])

    async def test_group_size_must_match_reservation(self):
        configure_concurrency(2, 2)
        builder = ArtsyGroupBuilder({'id': 'x'}, 1, 1, Path('/tmp'), MagicMock(), 128, 32768, False)
        with self.assertRaisesRegex(ValueError, 'Group size'):
            await builder.make_envs()

    async def test_cancelled_scorer_is_removed(self):
        configure_concurrency(1, 1)
        session = DockerSession({'id': 'x', 'prompt': 'draw', 'seed': 1}, Path('/tmp'), MagicMock())
        session.container = 'workspace'
        session.submitted = True
        session._read_submission = AsyncMock(return_value=('print(1)', None))
        session._judge = AsyncMock(return_value={'malicious': False})
        started = asyncio.Event()
        calls = []
        async def docker(argv, **kwargs):
            calls.append(argv)
            if argv[0] == 'run':
                started.set()
                await asyncio.Future()
            return response()
        with tempfile.TemporaryDirectory() as root, patch('training.cookbook_env._docker', side_effect=docker):
            session.directory = Path(root)
            task = asyncio.create_task(session.grade([]))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        name = calls[0][calls[0].index('--name') + 1]
        self.assertEqual(calls[1], ['rm', '-f', name])
