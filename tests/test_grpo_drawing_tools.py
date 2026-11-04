"""RL drawing-tool behavior without Docker or hosted model calls."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from drawing.agent_contract import DRAWING_TOOLS, DRAWING_TOOLS_CONTRACT_VERSION
from training.cookbook_env import ArtsyGroupBuilder, DockerSession, configure_concurrency


def response(output=b'', code=0):
    return SimpleNamespace(stdout=output, stderr=b'', returncode=code)


class DrawingToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = DockerSession({'id': 'x', 'prompt': 'Draw a circle', 'seed': 7},
                                     Path(self.tmp.name), {})
        self.session.container = 'fake-container'

    async def test_write_read_preview_submit_and_post_submission_guard(self):
        # Shell-looking model content must remain stdin data in a fixed command.
        source = "# `touch /tmp/injected` $(echo injected)\nprint('draw')"
        session = self.session
        with patch('training.cookbook_env._docker', new=AsyncMock(return_value=response())) as docker:
            observation, done = await session.execute_tool('update_drawing', {'source': source})
            self.assertFalse(done)
            self.assertIn('Drawing updated', observation)
            argv = docker.call_args.args[0]
            self.assertNotIn('bash', argv)
            self.assertNotIn(source, argv)
            self.assertEqual(docker.call_args.kwargs['stdin'], source.encode())
            compile(argv[-1], '<fixed-write>', 'exec')
        with patch.object(session, '_read_source', new=AsyncMock(return_value=source)):
            observation, done = await session.execute_tool('read_drawing', {})
            self.assertEqual(observation, 'Return code: 0\n' + source)
            self.assertFalse(done)
            preview = [{'type': 'text', 'text': 'render feedback'}, {'type': 'image', 'image': 'native-image'}]
            with patch('training.cookbook_env._docker', new=AsyncMock(return_value=response(b'preview'))) as docker, \
                 patch('training.cookbook_env._preview_observation', return_value=(preview, {'ok': True})):
                observation, done = await session.execute_tool('preview_drawing', {})
                self.assertEqual(observation, preview)
                self.assertFalse(done)
                self.assertEqual(session.previews, [{'ok': True}])
                argv = docker.call_args.args[0]
                self.assertIn('/opt/artsy/render.py', argv)
                self.assertEqual(argv[argv.index('--seed') + 1], '7')
                self.assertEqual(docker.call_args.kwargs['stdin'], source.encode())
            observation, done = await session.execute_tool('submit_drawing', {})
            self.assertTrue(done)
            self.assertTrue(session.submitted)
        with patch('training.cookbook_env._docker', new=AsyncMock()) as docker:
            observation, done = await session.execute_tool('update_drawing', {'source': 'replacement'})
            self.assertTrue(done)
            self.assertIn('already ended', observation)
            docker.assert_not_awaited()
        self.assertEqual([call['name'] for call in session.tool_calls],
                         ['update_drawing', 'read_drawing', 'preview_drawing', 'submit_drawing'])

    async def test_missing_source_can_be_repaired_after_failed_submit(self):
        with patch.object(self.session, '_read_source', new=AsyncMock(return_value='')):
            for name in ('read_drawing', 'preview_drawing', 'submit_drawing'):
                observation, done = await self.session.execute_tool(name, {})
                self.assertFalse(done)
                self.assertIn('No drawing yet', observation)
                self.assertFalse(self.session.submitted)

    async def test_strict_arguments_and_utf8_byte_limits(self):
        cases = [('bash', {'command': 'true'}), ('read_drawing', {'extra': ''}),
                 ('submit_drawing', None), ('update_drawing', {'source': 1}),
                 ('update_drawing', {'source': ' '}), ('update_drawing', {'source': 'é' * 32769}),
                 ('refuse_drawing', {'reason': ''}), ('refuse_drawing', {'reason': 'é' * 1025})]
        with patch('training.cookbook_env._docker', new=AsyncMock()) as docker:
            for name, arguments in cases:
                with self.subTest(name=name, arguments=str(arguments)[:60]):
                    observation, done = await self.session.execute_tool(name, arguments)
                    self.assertTrue(observation.startswith('Return code: 1'))
                    self.assertFalse(done)
            docker.assert_not_awaited()

    async def test_refusal_removes_source_and_ends_without_image_judgment(self):
        reason = 'Cannot fulfill this request'
        with patch('training.cookbook_env._docker', new=AsyncMock(return_value=response())) as docker:
            _, done = await self.session.execute_tool('refuse_drawing', {'reason': reason})
            self.assertTrue(done)
            argv = docker.call_args.args[0]
            self.assertIn("Path('/workspace/program.py').unlink(missing_ok=True)", argv[-1])
            self.assertEqual(json.loads(docker.call_args.kwargs['stdin']), {'reason': reason})
            self.assertNotIn('bash', argv)
        with patch.object(self.session, '_read_source', new=AsyncMock(return_value='')), \
             patch.object(self.session, '_read_refusal', new=AsyncMock(return_value=reason)):
            await self.session.grade([])
        self.assertEqual(self.session.result['outcome'], 'refused')
        self.assertEqual(self.session.result['trajectoryContractVersion'], DRAWING_TOOLS_CONTRACT_VERSION)
        self.assertEqual(self.session.result['toolCalls'][0]['name'], 'refuse_drawing')
        self.assertNotIn('judgment', self.session.result)

    async def test_write_failure_and_preview_timeout_do_not_submit(self):
        with patch('training.cookbook_env._docker', new=AsyncMock(return_value=response(code=1))):
            with self.assertRaisesRegex(RuntimeError, 'Cannot write drawing'):
                await self.session.execute_tool('update_drawing', {'source': 'draw()'})
        with patch.object(self.session, '_read_source', new=AsyncMock(return_value='draw()')), \
             patch('training.cookbook_env._docker', new=AsyncMock(side_effect=subprocess.TimeoutExpired('render', 65))):
            observation, done = await self.session.execute_tool('preview_drawing', {})
            self.assertIn('124', observation)
            self.assertFalse(done)
        self.assertFalse(self.session.submitted)

    async def test_real_cookbook_dispatches_only_shared_tools_in_order(self):
        from tinker_cookbook.renderers import ToolCall
        configure_concurrency(1, 1)
        builder = ArtsyGroupBuilder(self.session.spec, 1, 1, Path(self.tmp.name), {}, None, 262144, False)
        with patch('training.cookbook_env.encoder', return_value=SimpleNamespace(renderer=MagicMock())), \
             patch('training.cookbook_env.initial_messages', return_value=[]):
            wrapped, = await builder.make_envs()
        env = wrapped.message_env
        self.assertEqual([tool.to_spec() for tool in env.tools], [t['function'] for t in DRAWING_TOOLS])
        self.assertEqual(env.tool_execution, 'sequential')
        session = builder.sessions[0]
        session.container = 'fake-container'
        env.reward_fn = AsyncMock(return_value=(0.0, {}))
        calls = [ToolCall(id=str(i), function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)))
                 for i, (name, args) in enumerate([
                     ('update_drawing', {'source': 'draw()'}), ('submit_drawing', {}),
                     ('update_drawing', {'source': 'must not run'})])]
        with patch('training.cookbook_env._docker', new=AsyncMock(return_value=response())) as docker, \
             patch.object(session, '_read_source', new=AsyncMock(return_value='draw()')):
            result = await env.step({'role': 'assistant', 'content': '', 'tool_calls': calls})
            self.assertTrue(result.episode_done)
            # Only the first write may mutate the sandbox, even in a multi-call turn.
            self.assertEqual(docker.await_count, 1)
        env.reward_fn.assert_awaited_once()
        session.container = None
        await builder.cleanup()
