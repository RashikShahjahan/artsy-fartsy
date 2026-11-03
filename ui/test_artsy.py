"""Consumer request policy, edit integrity, routing and worker handoff; no paid calls."""
import hashlib
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import server, worker


class ArtsyTests(unittest.TestCase):
    def test_consumer_policy_cannot_be_overridden_by_the_client(self):
        request = server.artsy_request({'prompt': '  A friendly fox  ', 'seed': 0, 'max_turns': 900})
        self.assertEqual(request, {'prompt': 'A friendly fox', 'model': 'openai/gpt-6-luna',
                                  'seed': 42, 'experience': 'artsy', 'max_turns': 10, 'execution_mode': 'auto'})
        for body in ([], {}, {'prompt': ''}, {'prompt': 'a' * 4001}, {'prompt': 20}):
            with self.subTest(body=str(body)[:30]), self.assertRaises(ValueError):
                server.artsy_request(body)

    def test_execution_mode_is_validated_and_saved(self):
        self.assertEqual(server.artsy_request({'prompt': 'Fox', 'execution_mode': 'approval'})['execution_mode'], 'approval')
        for mode in (None, '', 'manual', [], True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                server.artsy_request({'prompt': 'Fox', 'execution_mode': mode})

    def test_approval_only_releases_the_current_pending_action(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(server, 'JOBS', Path(temporary)):
            folder = Path(temporary) / ('a' * 32)
            folder.mkdir()
            job = {'execution_mode': 'approval', 'state': 'running'}
            (folder / 'job.json').write_text(json.dumps(job))
            (folder / 'progress.json').write_text(json.dumps({'phase': 'awaiting_approval', 'steps': [
                {'id': 'action-1', 'status': 'completed'}, {'id': 'action-2', 'status': 'pending'}]}))
            for body in ([], {}, {'step_id': 'action-1'}, {'step_id': 'action-3'}):
                with self.subTest(body=body), self.assertRaises(ValueError):
                    server.approve_action(folder.name, body)
            self.assertFalse((folder / 'approval.json').exists())
            server.approve_action(folder.name, {'step_id': 'action-2'})
            self.assertEqual(json.loads((folder / 'approval.json').read_text()), {'step_id': 'action-2'})
            for change in ({'state': 'completed'}, {'execution_mode': 'auto'}):
                (folder / 'job.json').write_text(json.dumps({**job, **change}))
                with self.assertRaises(ValueError):
                    server.approve_action(folder.name, {'step_id': 'action-2'})

    def completed(self, root):
        ident = 'a' * 32
        folder = root / ident
        output = folder / 'output'
        output.mkdir(parents=True)
        (folder / 'job.json').write_text(json.dumps({'state': 'completed', 'prompt': 'Draw a fox'}))
        (output / 'program.py').write_text('print("fox")\n')
        (output / 'image.png').write_bytes(b'fixture image')
        (output / 'result.json').write_text(json.dumps({'outcome': 'submitted',
            'sourceHash': hashlib.sha256((output / 'program.py').read_bytes()).hexdigest(),
            'imageHash': hashlib.sha256((output / 'image.png').read_bytes()).hexdigest()}))
        return ident, folder

    def test_edits_only_accept_completed_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ident, folder = self.completed(root)
            with patch.object(server, 'JOBS', root):
                self.assertEqual(server.artsy_request({'prompt': 'Make it blue', 'parent_id': ident})['parent_id'], ident)
                for invalid in ('../secret', 1, '', 'b' * 32):
                    with self.subTest(parent=invalid), self.assertRaises(ValueError):
                        server.artsy_request({'prompt': 'Edit', 'parent_id': invalid})
                (folder / 'output/program.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    server.artsy_request({'prompt': 'Edit', 'parent_id': ident})

    def test_missing_image_or_incomplete_parent_cannot_be_edited(self):
        for broken in ('missing_image', 'running', 'refused'):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                ident, folder = self.completed(root)
                if broken == 'missing_image':
                    (folder / 'output/image.png').unlink()
                    (folder / 'output/result.json').write_text(json.dumps({'outcome': 'submitted',
                        'sourceHash': server.content_hash(folder / 'output/program.py')}))
                elif broken == 'running':
                    (folder / 'job.json').write_text(json.dumps({'state': 'running'}))
                else:
                    (folder / 'output/result.json').write_text(json.dumps({'outcome': 'refused'}))
                with patch.object(server, 'JOBS', root), self.assertRaises(ValueError):
                    server.artsy_request({'prompt': 'Edit', 'parent_id': ident})

    def test_home_is_consumer_ui_and_existing_studio_is_preserved(self):
        handler = object.__new__(server.Handler)
        handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
        for route, text in (('/', 'What should we draw?'), ('/studio', 'Drawing studio'),
                            ('/drawings', 'From dataset')):
            handler.path = route
            with patch.object(handler, 'send_page') as page:
                handler.do_GET()
                self.assertIn(text, page.call_args.args[0])
        handler.path = '/api/artsy/runs'
        handler.headers['Origin'] = 'https://untrusted.example'
        with patch.object(handler, 'send_json') as send:
            handler.do_POST()
            self.assertEqual(send.call_args.args[1], 403)

    def test_worker_passes_edit_context_ten_turns_and_atomic_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ident, parent = self.completed(root)
            child = root / ('b' * 32)
            child.mkdir()
            (child / 'job.json').write_text(json.dumps({'experience': 'artsy', 'parent_id': ident}))
            def draw(*, progress):
                import sys
                argv = sys.argv
                self.assertIn('Previous request: Draw a fox', argv[1])
                self.assertIn('Requested change: Make it blue', argv[1])
                self.assertEqual(argv[argv.index('--max-turns') + 1], '10')
                self.assertEqual(argv[argv.index('--reference-source') + 1], str(parent / 'output/program.py'))
                progress({'phase': 'previewing', 'turn': 3, 'max_turns': 10})
                self.assertEqual(json.loads((child / 'progress.json').read_text())['turn'], 3)
                self.assertFalse((child / 'progress.tmp').exists())
                progress({'phase': 'drawing', 'step': {'id': 'one', 'status': 'running', 'command': 'draw'}})
                png = b'\x89PNG\r\n\x1a\nfixture'
                progress({'phase': 'previewing', 'step': {'id': 'one', 'status': 'completed',
                    'previews': ['data:image/png;base64,' + base64.b64encode(png).decode()]}})
                progress({'phase': 'thinking', 'step': {'id': 'two', 'status': 'running'}})
                saved = json.loads((child / 'progress.json').read_text())
                self.assertEqual([step['id'] for step in saved['steps']], ['one', 'two'])
                self.assertEqual(saved['steps'][0]['command'], 'draw')
                self.assertEqual(saved['steps'][0]['previews'], ['preview-0-0.png'])
                self.assertEqual((child / 'output/preview-0-0.png').read_bytes(), png)
                self.assertNotIn('base64', json.dumps(saved))
                return 0
            with patch.object(server, 'JOBS', root), \
                 patch('sys.argv', ['ui.worker', str(child), 'Make it blue', '42', 'openai/gpt-6-luna']), \
                 patch('evaluation.draw.main', side_effect=draw):
                self.assertEqual(worker.main(), 0)

    def test_worker_waits_for_the_matching_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'job.json').write_text(json.dumps({'experience': 'artsy', 'execution_mode': 'approval'}))

            def draw(*, progress, approval):
                async def exercise():
                    task = asyncio.create_task(approval('action-2'))
                    await asyncio.sleep(0.01)
                    self.assertFalse(task.done())
                    (folder / 'approval.json').write_text(json.dumps({'step_id': 'action-1'}))
                    await asyncio.sleep(0.21)
                    self.assertFalse(task.done())
                    (folder / 'approval.json').write_text(json.dumps({'step_id': 'action-2'}))
                    await asyncio.wait_for(task, 1)
                asyncio.run(exercise())
                return 0

            with patch('sys.argv', ['ui.worker', str(folder), 'Fox', '42', 'openai/gpt-6-luna']), \
                 patch('evaluation.draw.main', side_effect=draw):
                self.assertEqual(worker.main(), 0)

    def test_status_and_preview_routes_preserve_live_steps(self):
        from io import BytesIO
        with tempfile.TemporaryDirectory() as temporary, patch.object(server, 'JOBS', Path(temporary)):
            folder = Path(temporary) / ('a' * 32)
            (folder / 'output').mkdir(parents=True)
            (folder / 'job.json').write_text(json.dumps({'state': 'running', 'execution_mode': 'approval'}))
            progress = {'phase': 'awaiting_approval', 'steps': [{'id': 'action-1', 'status': 'pending'}]}
            (folder / 'progress.json').write_text(json.dumps(progress))
            (folder / 'output/preview-0-0.png').write_bytes(b'fixture PNG')
            handler = object.__new__(server.Handler)
            handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
            handler.path = f'/api/runs/{folder.name}'
            with patch.object(handler, 'send_json') as send:
                handler.do_GET()
                self.assertEqual(send.call_args.args[0]['progress'], progress)
            handler.path += '/preview-0-0.png'
            handler.wfile = BytesIO()
            with patch.object(handler, 'send_response'), patch.object(handler, 'send_header') as header, patch.object(handler, 'end_headers'):
                handler.do_GET()
                self.assertIn(('Content-Type', 'image/png'), [call.args for call in header.call_args_list])
            self.assertEqual(handler.wfile.getvalue(), b'fixture PNG')
            for suffix in ('../job.json', 'preview-0-0.png.tmp', 'imageXpng'):
                handler.path = f'/api/runs/{folder.name}/{suffix}'
                with patch.object(handler, 'send_json') as send:
                    handler.do_GET()
                    self.assertEqual(send.call_args.args[1], 404)

    def test_reference_source_reaches_model_input(self):
        from evaluation.harness import drawing_task
        task = drawing_task([{'id': 'edit', 'prompt': 'Make it blue', 'seed': 42,
                              'reference_source': '# original drawing\n'}], '/tmp/artsy-test', max_turns=10)
        self.assertIn('# original drawing', task.dataset[0].input)
        self.assertIn('Make it blue', task.dataset[0].input)
        self.assertEqual(task.metadata['max_model_turns'], 10)


if __name__ == '__main__':
    unittest.main()
