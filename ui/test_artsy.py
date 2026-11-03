"""Consumer request policy, edit integrity, routing and worker handoff; no paid calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import server, worker


class ArtsyTests(unittest.TestCase):
    def test_consumer_policy_cannot_be_overridden_by_the_client(self):
        request = server.artsy_request({'prompt': '  A friendly fox  ', 'model': 'other/model',
                                        'seed': 0, 'max_turns': 900})
        self.assertEqual(request, {'prompt': 'A friendly fox', 'model': 'openai/gpt-6-luna',
                                  'seed': 42, 'experience': 'artsy', 'max_turns': 10})
        for body in ([], {}, {'prompt': ''}, {'prompt': 'a' * 4001}, {'prompt': 20}):
            with self.subTest(body=str(body)[:30]), self.assertRaises(ValueError):
                server.artsy_request(body)

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
                return 0
            with patch.object(server, 'JOBS', root), \
                 patch('sys.argv', ['ui.worker', str(child), 'Make it blue', '42', 'openai/gpt-6-luna']), \
                 patch('evaluation.draw.main', side_effect=draw):
                self.assertEqual(worker.main(), 0)

    def test_reference_source_reaches_model_input(self):
        from evaluation.harness import drawing_task
        task = drawing_task([{'id': 'edit', 'prompt': 'Make it blue', 'seed': 42,
                              'reference_source': '# original drawing\n'}], '/tmp/artsy-test', max_turns=10)
        self.assertIn('# original drawing', task.dataset[0].input)
        self.assertIn('Make it blue', task.dataset[0].input)
        self.assertEqual(task.metadata['max_model_turns'], 10)


if __name__ == '__main__':
    unittest.main()
