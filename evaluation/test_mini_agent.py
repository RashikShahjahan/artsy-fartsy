"""mini's real loop through Inspect, with fixed local test commands and rendering."""

import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import (ChatCompletionChoice, ChatMessageAssistant, ContentImage,
                              GenerateConfig, ModelOutput, ModelUsage, get_model)
from inspect_ai.tool import ToolCall

from drawing.mini_agent import CONTRACT_VERSION, PREVIEW, SUBMIT
from drawing.renderer import LocalRenderer
from evaluation.mini_agent import mini_drawing_agent
from tests.test_mini_agent import CODE, EDIT, TestWorkspace


class MiniInspectTests(unittest.TestCase):
    def test_reference_source_targeted_edit_preview_and_tool_logging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = TestWorkspace()
            self.addCleanup(workspace.cleanup)
            model = get_model('mockllm/mini-test', memoize=False, custom_outputs=[
                ModelOutput(model='mockllm/mini-test', choices=[ChatCompletionChoice(
                    message=ChatMessageAssistant(content='Edit and inspect.', tool_calls=[
                        ToolCall(id=f'call-{i}', function='bash', arguments={'command': command})]),
                    stop_reason='tool_calls')],
                    usage=ModelUsage(input_tokens=10, output_tokens=10, total_tokens=20))
                for i, command in enumerate([EDIT, PREVIEW, SUBMIT])])

            async def write_file(path, source):
                (workspace.directory / Path(path).name).write_text(source)

            async def execute(cmd, **kwargs):
                if cmd[0] == 'python':
                    import shlex
                    result = workspace.execute({'command': 'python -c ' + shlex.quote(cmd[2]) + ''.join(' ' + shlex.quote(a) for a in cmd[3:])})
                    return SimpleNamespace(success=result['returncode'] == 0, returncode=result['returncode'],
                                           stdout=result['output'], stderr='')
                result = workspace.execute({'command': cmd[-1]})
                return SimpleNamespace(success=result['returncode'] == 0, returncode=result['returncode'],
                                       stdout=result['output'], stderr='')

            sbox = SimpleNamespace(exec=execute, write_file=write_file)
            with patch('evaluation.mini_agent.sandbox', return_value=sbox), \
                 patch('inspect_ai._util.appdirs.user_data_path', side_effect=lambda n: root / 'data' / n), \
                 patch('inspect_ai._util.appdirs.user_cache_path', side_effect=lambda n: root / 'cache' / n):
                logs = eval(Task(dataset=[Sample(id='edit', input='Make it blue', metadata={
                    'seed': 7, 'prompt': 'Make it blue', 'reference_source': CODE})],
                    solver=mini_drawing_agent(GenerateConfig(max_tokens=1024), max_turns=4)),
                    model=model, log_dir=str(root / 'logs'), display='none')
            self.assertEqual(logs[0].status, 'success', logs[0].error)
            sample = read_eval_log(logs[0].location, resolve_attachments=True).samples[0]
            self.assertIsNone(sample.error)
            self.assertTrue(sample.metadata['submitted'])
            self.assertEqual(sample.metadata['trajectoryContractVersion'], CONTRACT_VERSION)
            self.assertEqual(sample.metadata['harnessTiming']['model_calls'], 3)
            self.assertEqual(sum(e.event == 'tool' for e in sample.events), 3)
            from evaluation.harness import trajectory_metrics
            self.assertEqual(trajectory_metrics(sample.messages, True)['previewAttempts'], 1)
            self.assertTrue(any(isinstance(p, ContentImage) for m in sample.messages
                                if isinstance(m.content, list) for p in m.content))
            self.assertIn('0,0,1', (workspace.directory / 'program.py').read_text())
