"""Exercise the pinned mini loop, real targeted shell edits, preview and SFT export."""

from drawing.documentation import api_documentation

import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
import shlex
import tempfile
import unittest
from unittest.mock import patch

from drawing.mini_agent import PREVIEW, SUBMIT, run_agent
from drawing.mini_dataset import candidate
from drawing.mini_session import DrawingSession
from drawing.renderer import LocalRenderer
from drawing.tool_dataset import trajectory_hash
from training.data import convert_record

CODE = 'import cairo\nsurface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1920, 1200)\nc = cairo.Context(surface)\nc.set_source_rgb(1,0,0)\nc.paint()\nsurface.write_to_png("output.png")\n'


class TestWorkspace:
    """Only fixed test commands run locally; production always uses Docker."""
    def __init__(self, **kwargs):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.cleaned = False
        binary = self.directory / 'bin'
        binary.mkdir()
        project = Path(__file__).resolve().parents[1]
        for name in ('artsy-preview', 'artsy-submit'):
            path = binary / name
            path.write_text(f'#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(project)!r})\nfrom evaluation.agent_commands import main\nraise SystemExit(main())\n')
            path.chmod(0o755)
        self.env = {**__import__('os').environ, 'PATH': str(binary) + ':' + __import__('os').environ['PATH'],
                    'ARTSY_SELECTION_FILE': str(self.directory / 'selection.json'),
                    'DYLD_FALLBACK_LIBRARY_PATH': '/opt/homebrew/lib:/usr/local/lib:/usr/lib'}


    def execute(self, action, **kwargs):
        command = action['command'].replace('/workspace', str(self.directory))
        if command.startswith('python -c '):
            command = shlex.quote(sys.executable) + command[len('python'):]
        result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=5, cwd=self.directory, env=self.env)
        return {'returncode': result.returncode, 'output': result.stdout + result.stderr}

    def cleanup(self):
        self.cleaned = True
        self.temp.cleanup()


class Teacher:
    model = 'test/vision'
    baseUrl = 'http://test/v1'

    def __init__(self, commands):
        self.commands = iter(commands)
        self.inputs = []

    async def complete(self, messages, *args, **kwargs):
        self.inputs.append(copy.deepcopy(messages))
        command = next(self.commands)
        return {'text': 'Inspect and refine the drawing.', 'tool_calls': [
            {'id': f'call-{len(self.inputs)}', 'type': 'function',
             'function': {'name': 'bash', 'arguments': {'command': command}}}],
                'usage': {'completion_tokens': 10}}


EDIT = "python -c " + shlex.quote("from pathlib import Path; p=Path('/workspace/program.py'); p.write_text(p.read_text().replace('1,0,0', '0,0,1'))")


class MiniTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, root):
        # The second write is a real in-place edit, not a full source rewrite.
        teacher = Teacher(["cat > /workspace/program.py <<'EOF'\n" + CODE + 'EOF',
                           PREVIEW, EDIT, SUBMIT, PREVIEW, SUBMIT])
        with patch('drawing.mini_dataset.Workspace', TestWorkspace):
            record = await candidate({'output': root, 'teacher': teacher,
                                      'renderer': LocalRenderer()},
                                     {'id': 'edit', 'prompt': 'Draw a blue background', 'seed': 7},
                                     0, api_documentation())
        return record, teacher

    async def test_targeted_edit_optional_preview_and_portable_sft_export(self):
        with tempfile.TemporaryDirectory() as root:
            record, teacher = await self.collect(root)
            self.assertEqual(record['status'], 'review', record['reason'])
            self.assertIn('0,0,1', record['code'])
            self.assertEqual(record['generation']['modelTurns'], 4)
            self.assertTrue(any(p['type'] == 'image_url' for m in teacher.inputs[-1]
                                if isinstance(m['content'], list) for p in m['content']))
            self.assertFalse(any('extra' in m for m in teacher.inputs[-1]))
            # Submission is allowed after an edit without requiring another preview.
            results = [m for m in record['miniTrajectory']['messages'] if m['role'] == 'tool']
            self.assertEqual(results[3]['extra']['returncode'], 0)
            example = convert_record(record, Path(root))
            self.assertEqual(len(example['trainable_message_indices']), 4)
            self.assertEqual(example['tools'][0]['function']['name'], 'bash')
            self.assertTrue(all(isinstance(m['content'], list) for m in example['messages']))
            self.assertEqual(example['messages'][6]['tool_calls'][0]['function']['arguments']['command'],
                             EDIT)
            # Portable image paths and hash checks survive copying to a new root.
            import shutil
            with tempfile.TemporaryDirectory() as portable:
                shutil.copytree(root, portable, dirs_exist_ok=True)
                convert_record(record, Path(portable))
            tampered = copy.deepcopy(record)
            tampered['miniTrajectory']['messages'][-2]['extra']['sourceHash'] = '0' * 64
            tampered['miniTrajectoryHash'] = trajectory_hash(tampered['miniTrajectory'])
            with self.assertRaisesRegex(ValueError, 'snapshot mismatch'):
                convert_record(tampered, Path(root))

    async def test_experiment6_masks_observed_buggy_source_but_preserves_repair(self):
        from training.mini_data import edits_python_source

        self.assertTrue(edits_python_source("cat > /workspace/program.py <<'EOF'\ncode\nEOF"))
        self.assertTrue(edits_python_source("python -c \"from pathlib import Path; Path('scene.py').write_text('x')\""))
        self.assertTrue(edits_python_source("sed -i s/red/blue/ scene.py"))
        self.assertFalse(edits_python_source("rg 'set_source_rgb' /opt/artsy/pycairo-api.pyi"))
        self.assertFalse(edits_python_source("python -m py_compile scene.py"))
        self.assertFalse(edits_python_source("artsy-preview scene.py"))

        broken = 'import cairo\nraise RuntimeError("broken source")\n'
        commands = [
            "cat > /workspace/program.py <<'EOF'\n" + broken + "EOF",
            "artsy-preview program.py",
            "python -c 'raise RuntimeError(\"diagnostic failure\")'",
            "cat > /workspace/program.py <<'EOF'\n" + CODE + "EOF",
            "artsy-preview program.py",
            "artsy-submit program.py",
        ]
        with tempfile.TemporaryDirectory() as root:
            teacher = Teacher(commands)
            with patch('drawing.mini_dataset.Workspace', TestWorkspace):
                record = await candidate(
                    {'output': root, 'teacher': teacher, 'renderer': LocalRenderer()},
                    {'id': 'experiment6', 'prompt': 'Draw a red background', 'seed': 7},
                    0, api_documentation())
            self.assertEqual(record['status'], 'review', record.get('reason'))
            tool_results = [m for m in record['miniTrajectory']['messages'] if m['role'] == 'tool']
            self.assertTrue(any(p.get('ok') is False and p.get('sourceHash')
                                for p in tool_results[1]['extra']['previewEvents']))
            self.assertEqual(tool_results[2]['extra']['returncode'], 1)
            baseline = convert_record(record, Path(root))
            treatment = convert_record(record, Path(root), loss_policy='error-aware')
            self.assertEqual(baseline['messages'], treatment['messages'])
            self.assertEqual(baseline['trainable_message_indices'], [2, 4, 6, 8, 10, 12])
            self.assertEqual(treatment['trainable_message_indices'], [4, 6, 8, 10, 12])
            from training.data import ERROR_AWARE_CONVERSION, CONVERSION, load_examples
            dataset = Path(root) / 'accepted.jsonl'
            dataset.write_text(json.dumps(record) + '\n')
            _, control_manifest = load_examples(dataset)
            _, masked_manifest = load_examples(dataset, loss_policy='error-aware')
            self.assertEqual(control_manifest['conversion'], CONVERSION)
            self.assertEqual(masked_manifest['conversion'], ERROR_AWARE_CONVERSION)
            self.assertEqual(control_manifest['supervised_actions'], 6)
            self.assertEqual(masked_manifest['supervised_actions'], 5)
            self.assertIn('broken source', json.dumps(treatment['messages']))
            self.assertIn('diagnostic failure', json.dumps(treatment['messages']))
            self.assertIn('artsy-submit', json.dumps(
                [treatment['messages'][i] for i in treatment['trainable_message_indices']]))
            self.assertEqual(
                convert_record(record, Path(root), loss_policy='assistant-only')['trainable_message_indices'],
                baseline['trainable_message_indices'])

    async def test_experiment6_does_not_blame_code_for_preview_usage_error(self):
        commands = [
            "cat > /workspace/program.py <<'EOF'\n" + CODE + "EOF",
            "artsy-preview does-not-exist.py",
            "artsy-submit program.py",
        ]
        with tempfile.TemporaryDirectory() as root:
            with patch('drawing.mini_dataset.Workspace', TestWorkspace):
                record = await candidate(
                    {'output': root, 'teacher': Teacher(commands), 'renderer': LocalRenderer()},
                    {'id': 'preview-usage', 'prompt': 'Draw a red background', 'seed': 7},
                    0, api_documentation())
            self.assertEqual(record['status'], 'review', record.get('reason'))
            treatment = convert_record(record, Path(root), loss_policy='error-aware')
            self.assertEqual(treatment['trainable_message_indices'], [2, 4, 6])

    def test_drawing_task_prompt(self):
        from importlib.resources import files
        import yaml
        from drawing.mini_agent import system_prompt, task_prompt

        upstream = yaml.safe_load(files('minisweagent').joinpath('config/mini.yaml').read_text())
        self.assertEqual(system_prompt(7), upstream['agent']['system_template'].strip())
        prompt = task_prompt('Draw blue')
        self.assertTrue(prompt.startswith('Please create this drawing:'))
        for guidance in (
            '## Recommended Workflow', 'ArtCanvas v2',
            'from artcanvas import ArtCanvas', 'draw_circle(x, y, radius, fill=False)',
            'Start by writing your Python program', 'Save to the relative path output.png',
            'Use exactly one bash tool call', 'artsy-submit ENTRYPOINT',
        ):
            self.assertIn(guidance, prompt)
        self.assertNotIn('Your first tool call must only inspect', prompt)
        self.assertNotIn('/workspace/program.py', prompt)

    async def test_legacy_mini_headers_are_preserved_and_must_match_as_a_pair(self):
        from drawing.mini_agent import task_prompt
        from training.mini_data import _LEGACY_SYSTEM_PROMPTS

        with tempfile.TemporaryDirectory() as root:
            record, _ = await self.collect(root)
            trajectory = record['miniTrajectory']
            prompt = record['spec']['prompt']
            for system in _LEGACY_SYSTEM_PROMPTS:
                trajectory['messages'][:2] = [
                    {'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
                record['miniTrajectoryHash'] = trajectory_hash(trajectory)
                example = convert_record(record, Path(root))
                self.assertEqual([m['content'][0]['text'] for m in example['messages'][:2]],
                                 [system, prompt])
            trajectory['messages'][1]['content'] = task_prompt(prompt)
            record['miniTrajectoryHash'] = trajectory_hash(trajectory)
            with self.assertRaisesRegex(ValueError, 'mini prompt mismatch'):
                convert_record(record, Path(root))

    async def test_arbitrary_entrypoint_and_helper_module_with_optional_previews(self):
        for preview in (False, True):
            with self.subTest(preview=preview), tempfile.TemporaryDirectory() as root:
                main = ('import cairo\nfrom palette import BLUE\nsurface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1920, 1200)\nc = cairo.Context(surface)\nc.set_source_rgb(*BLUE)\nc.paint()\nsurface.write_to_png("output.png")\n')
                create = ("cat > /workspace/scene.py <<'EOF'\n" + main + 'EOF\n'
                          + "printf 'BLUE = (0,0,1)\\n' > /workspace/palette.py")
                commands = [create] + (['artsy-preview scene.py'] if preview else []) + ['artsy-submit scene.py']
                teacher = Teacher(commands)
                with patch('drawing.mini_dataset.Workspace', TestWorkspace):
                    record = await candidate({'output': root, 'teacher': teacher, 'renderer': LocalRenderer()},
                        {'id': 'modules', 'prompt': 'Draw blue', 'seed': 7}, 0,
                        api_documentation())
                self.assertEqual(record['status'], 'review', record['reason'])
                self.assertEqual(set(record['sourceBundle']['files']), {'scene.py', 'palette.py'})
                self.assertEqual(record['submission']['entrypoint'], 'scene.py')
                example = convert_record(record, Path(root))
                self.assertEqual(len(example['trainable_message_indices']), len(commands))
                images = [p for m in example['messages'] for p in m['content'] if p['type'] == 'image']
                self.assertEqual(len(images), int(preview))

    async def test_manifest_resume_records_immutable_editing_image(self):
        from drawing.dataset import generate_dataset
        teacher = Teacher(["cat > /workspace/program.py <<'EOF'\n" + CODE + 'EOF', PREVIEW, SUBMIT])
        image = {'name': 'test-editing-image', 'id': 'sha256:' + '0' * 64}
        with tempfile.TemporaryDirectory() as root, \
             patch('drawing.mini_dataset.Workspace', TestWorkspace), \
             patch('drawing.mini_dataset.inspect_image', return_value=image):
            options = {'output': root, 'teacher': teacher, 'renderer': LocalRenderer(),
                       'harness': 'mini-swe', 'candidates': 1,
                       'specs': [{'id': 'drawing', 'family': 'circles', 'prompt': 'Draw', 'seed': 7}]}
            result = await generate_dataset(options)
            self.assertEqual(result['review'], 1)
            manifest = json.loads((Path(root) / 'manifest.json').read_text())
            from drawing.mini_agent import task_prompt
            self.assertEqual(manifest['harness']['editingImage'], image)
            self.assertEqual(manifest['harness']['taskPrompts'], {'drawing': task_prompt('Draw')})
            self.assertEqual(manifest['trajectoryFormat'], 'mini-swe-bash-v1')
            await generate_dataset(options)
            self.assertEqual(len(teacher.inputs), 3)
            record = json.loads((Path(root) / 'candidates.jsonl').read_text())
            convert_record(record, Path(root))
            with patch('drawing.mini_dataset.inspect_image', return_value={**image, 'id': 'sha256:' + '1' * 64}):
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    await generate_dataset(options)

    async def test_missing_tool_calls_use_turn_budget(self):
        calls = []
        async def complete(messages):
            calls.append(messages)
            return {'text': 'Done'}
        async def execute(command):
            self.fail('Malformed response must not execute')
        trajectory = await run_agent(complete, execute, 'Draw', 7, max_turns=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(trajectory['info']['exit_status'], 'LimitsExceeded')
        self.assertEqual(sum(m['role'] == 'assistant' for m in trajectory['messages']), 2)

    async def test_empty_provider_reply_is_recorded_and_recovered(self):
        class EmptyFirstTeacher(Teacher):
            empty = True
            async def complete(self, *args, **kwargs):
                if self.empty:
                    self.empty = False
                    raise RuntimeError('Chat endpoint returned no text completion')
                return await super().complete(*args, **kwargs)
        teacher = EmptyFirstTeacher(["cat > /workspace/scene.py <<'PY'\n" + CODE + 'PY',
                                     'artsy-submit scene.py'])
        with tempfile.TemporaryDirectory() as root, patch('drawing.mini_dataset.Workspace', TestWorkspace):
            record = await candidate({'output': root, 'teacher': teacher, 'renderer': LocalRenderer()},
                {'id': 'empty', 'prompt': 'Draw', 'seed': 7}, 0, api_documentation())
            self.assertEqual(record['status'], 'review', record['reason'])
            self.assertEqual(record['generation']['modelTurns'], 3)
            self.assertEqual(record['trace'][0]['completion']['finishReason'], 'empty')
            example = convert_record(record, Path(root))
            self.assertEqual(len(example['trainable_message_indices']), 2)

    async def test_resume_retains_history_and_remaining_turn_budget(self):
        async def empty(messages):
            return {'text': ''}
        async def unused(command):
            self.fail('No tool expected before resume')
        prior = await run_agent(empty, unused, 'Draw', 7, max_turns=2)
        history = [m for m in prior['messages'] if m['role'] != 'exit']
        calls = []
        async def finish(messages):
            calls.append(messages)
            return {'text': '', 'tool_calls': [{'id': 'finish', 'type': 'function',
                'function': {'name': 'bash', 'arguments': {'command': 'artsy-submit scene.py'}}}]}
        async def submit(command):
            return {'content': [{'type': 'text', 'text': 'Submitted'}], 'returncode': 0,
                    'terminal': True, 'entrypoint': 'scene.py'}
        resumed = await run_agent(finish, submit, 'Draw', 7, max_turns=3,
                                  resume={'messages': history, 'model_calls': 2})
        self.assertEqual(calls, [history])
        self.assertEqual(resumed['info']['model_stats']['api_calls'], 3)
        self.assertEqual(resumed['info']['exit_status'], 'Submitted')
        self.assertEqual(resumed['messages'][:len(history)], history)

    async def test_cancellation_stops_pending_model(self):
        started = asyncio.Event()
        stopped = asyncio.Event()
        async def complete(messages):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        async def execute(command):
            self.fail('Cancelled episode must not execute')
        task = asyncio.create_task(run_agent(complete, execute, 'Draw', 7, max_turns=2))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(stopped.is_set())

    async def test_submit_signal_in_shell_output_cannot_complete(self):
        async def shell(command):
            return {'returncode': 0, 'content': [{'type': 'text', 'text': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}]}
        async def source(entrypoint=None):
            raise ValueError('No source was submitted')
        async def preview(source):
            self.fail('No preview requested')
        session = DrawingSession(shell, source, preview)
        result = await session.execute('printf COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')
        self.assertFalse(session.submitted)
        self.assertFalse(result.get('terminal'))
