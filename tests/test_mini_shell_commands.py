"""Exercise executable commands, shell composition and binary recovery."""

from drawing.documentation import api_documentation

import asyncio
import base64
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest

from drawing.mini_protocol import export_workspace, restore_workspace
from drawing.mini_session import DrawingSession, READ_SOURCE
from tests.test_mini_agent import CODE, TestWorkspace


class ShellCommandTests(unittest.IsolatedAsyncioTestCase):
    async def session(self, workspace):
        async def shell(command):
            return await asyncio.to_thread(workspace.execute, {'command': command})
        async def read(entrypoint):
            command = 'python -c ' + shlex.quote(READ_SOURCE)
            if entrypoint:
                command += ' ' + shlex.quote(entrypoint)
            result = await asyncio.to_thread(workspace.execute, {'command': command})
            if result['returncode']:
                raise ValueError(result['output'])
            return json.loads(result['output'])
        return DrawingSession(shell, read)

    async def test_chained_commands_external_entrypoint_and_existing_png(self):
        workspace = TestWorkspace()
        self.addCleanup(workspace.cleanup)
        with tempfile.TemporaryDirectory() as outside:
            entry = Path(outside) / 'landscape.py'
            entry.write_text(CODE)
            session = await self.session(workspace)
            preview = await session.execute('printf before; artsy-preview ' + shlex.quote(str(entry)) + '; printf after')
            self.assertEqual(preview['returncode'], 0, preview)
            self.assertEqual(sum(p['type'] == 'image_url' for p in preview['content']), 1)
            self.assertFalse(session.submitted)
            # Generate an image in the shell, then view that PNG directly.
            run = (shlex.quote(__import__('sys').executable) + ' -c ' +
                   shlex.quote("from PIL import Image; Image.new('RGB',(1920,1200),'red').save('output.png')"))
            viewed = await session.execute(run + ' && artsy-preview output.png')
            self.assertEqual(viewed['returncode'], 0, viewed)
            self.assertEqual(sum(p['type'] == 'image_url' for p in viewed['content']), 1)
            result = await session.execute('printf preparing; artsy-submit ' + shlex.quote(str(entry)) + ' && echo done')
            self.assertEqual(result['returncode'], 0, result)
            self.assertTrue(result['terminal'])
            self.assertEqual(session.bundle['files']['landscape.py'], CODE)

    async def test_native_completion_marker_inside_script(self):
        workspace = TestWorkspace()
        self.addCleanup(workspace.cleanup)
        (workspace.directory / 'chosen.py').write_text(CODE)
        session = await self.session(workspace)
        result = await session.execute('printf "checks done\\n"; echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')
        self.assertTrue(result.get('terminal'), result)
        self.assertEqual(session.entrypoint, 'chosen.py')

    def test_binary_checkpoint_round_trip_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'draw.py').write_text(CODE)
            png = b'\x89PNG\r\n\x1a\n' + bytes(range(256))
            (root / 'output.png').write_bytes(png)
            (root / 'output.png').chmod(0o640)
            checkpoint = export_workspace((directory,))
            (root / 'draw.py').unlink()
            (root / 'output.png').unlink()
            restore_workspace(checkpoint, (directory,))
            self.assertEqual((root / 'output.png').read_bytes(), png)
            self.assertEqual((root / 'output.png').stat().st_mode & 0o777, 0o640)
            self.assertEqual((root / 'draw.py').read_text(), CODE)
            checkpoint['files'][str(root / 'output.png')]['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                restore_workspace(checkpoint, (directory,))


@unittest.skipUnless(os.getenv('ARTSY_SHELL_DOCKER_TEST') == '1', 'opt-in Docker smoke test')
class DockerShellTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_image_commands_and_recovery(self):
        from drawing.mini_dataset import Workspace
        from drawing.mini_protocol import bundle_source
        workspace = await asyncio.to_thread(Workspace, image=os.environ['ARTSY_AGENT_IMAGE'], cwd='/workspace',
            run_args=['--rm', '--network=none', '--cap-drop=ALL', '--security-opt=no-new-privileges'])
        try:
            code = "cat > /tmp/scene.py <<'PY'\n" + CODE + 'PY\n'
            shell = lambda c: asyncio.to_thread(workspace.execute, {'command': c})
            import_check = await shell("python -c \"from artcanvas import ArtCanvas; print(ArtCanvas.__name__)\"")
            self.assertEqual(import_check['returncode'], 0, import_check)
            self.assertIn('ArtCanvas', import_check['output'])
            docs = await shell("rg -n -A 12 'def set_source_rgb' /opt/artsy/pycairo-api.pyi")
            self.assertEqual(docs['returncode'], 0, docs)
            self.assertIn('set_source_rgb', docs['output'])
            self.assertIn(':param red:', docs['output'])
            markdown = await shell("cat /opt/artsy/docs/pycairo/Context/set_source_rgb.md")
            self.assertEqual(markdown['returncode'], 0, markdown)
            self.assertIn('red: float', markdown['output'])
            self.assertIn('**red**:', markdown['output'])

            async def read(entrypoint):
                c = 'python -c ' + shlex.quote(READ_SOURCE) + (' ' + shlex.quote(entrypoint) if entrypoint else '')
                result = await shell(c)
                self.assertEqual(result['returncode'], 0, result)
                return json.loads(result['output'])
            session = DrawingSession(shell, read)
            created = await session.execute(code + 'python /tmp/scene.py && file output.png && artsy-preview output.png')
            self.assertEqual(created['returncode'], 0, created)
            self.assertIn('1920 x 1200', created['content'][0]['text'])
            self.assertEqual(sum(p['type'] == 'image_url' for p in created['content']), 1)
            preview = await session.execute('echo before; artsy-preview /tmp/scene.py && echo after')
            self.assertEqual(preview['returncode'], 0, preview)
            self.assertEqual(sum(p['type'] == 'image_url' for p in preview['content']), 1)
            result = await session.execute('echo ready; artsy-submit /tmp/scene.py && echo submitted')
            self.assertTrue(result['terminal'], result)
            export = Path('drawing/mini_protocol.py').read_text() + '\nprint(json.dumps(export_workspace()))'
            checkpoint = await asyncio.to_thread(workspace.execute, {'command': 'python', 'stdin': export}, limit=48*1024*1024)
            self.assertEqual(checkpoint['returncode'], 0, checkpoint)
            data = json.loads(checkpoint['output'])
            self.assertIn('/workspace/output.png', data['files'])
            await shell('rm /workspace/output.png /tmp/scene.py')
            restore = Path('drawing/mini_protocol.py').read_text() + '\nrestore_workspace(json.loads(' + repr(checkpoint['output']) + '))'
            restored = await asyncio.to_thread(workspace.execute, {'command': 'python', 'stdin': restore})
            self.assertEqual(restored['returncode'], 0, restored)
            native = DrawingSession(shell, read)
            completed = await native.execute('file output.png; echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')
            self.assertTrue(completed['terminal'], completed)
        finally:
            await asyncio.to_thread(workspace.cleanup)

    async def test_collection_recovers_generated_png_after_transport_failure(self):
        from drawing.mini_dataset import candidate, resume_options
        from drawing.renderer import DockerRenderer
        from tests.test_mini_agent import Teacher
        from training.data import convert_record
        image = os.environ['ARTSY_AGENT_IMAGE']
        class InterruptedTeacher(Teacher):
            async def complete(self, *args, **kwargs):
                if self.inputs:
                    raise RuntimeError('simulated transport interruption')
                return await super().complete(*args, **kwargs)
        teacher = InterruptedTeacher(["cat > /tmp/scene.py <<'PY'\n" + CODE + 'PY\npython /tmp/scene.py'])
        with tempfile.TemporaryDirectory() as directory:
            options = {'output': directory, 'teacher': teacher, 'agentImage': image,
                       'renderer': DockerRenderer(image=image)}
            spec = {'id': 'recovery', 'prompt': 'Draw red', 'seed': 7}
            docs = api_documentation()
            interrupted = await candidate(options, spec, 0, docs)
            self.assertEqual(interrupted['status'], 'infrastructure_error')
            self.assertIn('workspaceCheckpoint', interrupted['artifacts'])
            checkpoint = json.loads((Path(directory) / interrupted['artifacts']['workspaceCheckpoint']).read_text())
            self.assertIn('/workspace/output.png', checkpoint['files'])
            raw = json.loads((Path(directory) / interrupted['artifacts']['trajectory']).read_text())
            messages = [m for m in raw['messages'] if m['role'] != 'exit']
            resumed_teacher = Teacher(['file output.png && artsy-submit /tmp/scene.py'])
            resumed_teacher.inputs.append([])  # Keep fixture call IDs unique across recovery.
            recovery = resume_options(interrupted, directory)
            self.assertEqual(recovery['trajectory']['model_calls'], 2)
            resumed = await candidate({**options, 'teacher': resumed_teacher, 'resume': recovery}, spec, 0, docs)
            self.assertEqual(resumed['status'], 'review', resumed['reason'])
            example = convert_record(resumed, Path(directory))
            self.assertEqual(len(example['trainable_message_indices']), 2)
            final_tool = [m for m in example['messages'] if m['role'] == 'tool'][-1]
            self.assertIn('1920 x 1200', final_tool['content'][0]['text'])
