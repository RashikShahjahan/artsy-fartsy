"""Consumer API boundary checks; no model, Docker, or provider calls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from client.backend import server


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def parent(self, paused=False):
        ident = 'a' * 32
        directory = self.root / ident
        output = directory / 'output'
        output.mkdir(parents=True)
        (directory / 'job.json').write_text(json.dumps({'state': 'completed', 'prompt': 'A circle'}))
        (output / 'program.py').write_text('print("drawing")')
        if not paused:
            (output / 'image.png').write_bytes(b'fixture')
        result = {'outcome': 'incomplete' if paused else 'submitted', 'turnLimitReached': paused,
                  'sourceHash': server.content_hash(output / 'program.py'),
                  'imageHash': server.content_hash(output / 'image.png')}
        (output / 'result.json').write_text(json.dumps(result))
        return ident, output

    def test_generation_uses_consumer_model_and_no_studio_configuration(self):
        request = server.generation_request({'prompt': ' A circle ', 'model': 'untrusted', 'max_turns': 900}, self.root)
        self.assertEqual(request['model'], server.MODEL)
        self.assertEqual(request['prompt'], 'A circle')
        self.assertEqual(request['max_turns'], 10)
        for body in (None, {}, {'prompt': ''}, {'prompt': 'a' * 4001}, {'prompt': 'Edit', 'parent_id': '../escape'}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                server.generation_request(body, self.root)

    def test_edit_requires_unchanged_completed_artifacts(self):
        ident, output = self.parent()
        request = server.generation_request({'prompt': 'Make it blue', 'parent_id': ident}, self.root)
        self.assertEqual(request['parent_id'], ident)
        (output / 'program.py').write_text('modified')
        with self.assertRaisesRegex(ValueError, 'changed'):
            server.validate_parent(self.root, ident)

    def test_paused_drawing_can_continue_without_an_image(self):
        ident, _ = self.parent(paused=True)
        self.assertEqual(server.validate_parent(self.root, ident), self.root / ident)

    def test_restart_reconciles_a_previously_running_job(self):
        ident = 'b' * 32
        directory = self.root / ident
        directory.mkdir()
        (directory / 'job.json').write_text('{"state":"running"}')
        self.assertEqual(server.DrawingJobs(self.root).status(ident)['state'], 'interrupted')

    def test_job_uses_its_own_worker_storage_and_bridge(self):
        ident = 'c' * 32
        directory = self.root / ident
        directory.mkdir()
        request = server.generation_request({'prompt': 'A circle'}, self.root)
        server.write_json(directory / 'job.json', {'state': 'queued', **request})
        jobs = server.DrawingJobs(self.root)
        jobs.active.add(ident)
        process = Mock(pid=123)
        process.wait.return_value = 0
        with patch.object(server.subprocess, 'Popen', return_value=process) as spawn:
            jobs.run(ident, request)
        self.assertIn('client.backend.worker', spawn.call_args.args[0])
        self.assertEqual(spawn.call_args.kwargs['env']['ARTSY_CLIENT_JOBS'], str(self.root))
        self.assertEqual(jobs.status(ident)['state'], 'completed')
        self.assertFalse(jobs.active)

    def test_queued_cancellation_does_not_spawn_a_worker(self):
        ident = 'd' * 32
        directory = self.root / ident
        directory.mkdir()
        request = server.generation_request({'prompt': 'A circle'}, self.root)
        server.write_json(directory / 'job.json', {'state': 'queued', **request})
        jobs = server.DrawingJobs(self.root)
        jobs.active.add(ident)
        jobs.cancel(ident)
        with patch.object(server.subprocess, 'Popen') as spawn:
            jobs.run(ident, request)
        spawn.assert_not_called()
        self.assertEqual(jobs.status(ident)['state'], 'cancelled')


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        self.http.jobs = server.DrawingJobs(Path(self.folder.name))
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.http.server_port}'

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        self.folder.cleanup()

    def request(self, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(self.url + path, data=data, headers=headers or {})
        try:
            with urlopen(request, timeout=2) as response:
                return response.status, response.read(), response.headers
        except HTTPError as error:
            with error:
                return error.code, error.read(), error.headers

    def test_health_and_consumer_only_routes(self):
        self.assertEqual(json.loads(self.request('/health')[1])['service'], 'artsy-client')
        for path in ('/', '/studio', '/api/launches', '/api/runs', '/api/runs/' + 'a' * 32 + '/program.py'):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)
        self.assertEqual(self.request('/api/launches', {})[0], 404)
        self.assertEqual(self.request('/health', headers={'Host': 'attacker.example'})[0], 403)
        self.assertEqual(self.request('/api/artsy/runs', {'prompt': 'A circle'}, {'Origin': 'https://attacker.example'})[0], 403)

    def test_submission_polling_images_and_cancellation(self):
        # Keep the job queued without starting a real inference process.
        with patch.object(self.http.jobs, 'run'):
            status, data, _ = self.request('/api/artsy/runs', {'prompt': 'A circle'})
        self.assertEqual(status, 202)
        ident = json.loads(data)['id']
        self.assertEqual(self.request('/api/artsy/runs', {'prompt': 'Another'})[0], 409)
        self.assertEqual(json.loads(self.request('/api/runs/' + ident)[1])['state'], 'queued')
        output = self.http.jobs.root / ident / 'output'
        output.mkdir()
        (output / 'image.png').write_bytes(b'PNG fixture')
        status, data, headers = self.request('/api/runs/' + ident + '/image.png')
        self.assertEqual((status, data, headers['Content-Type']), (200, b'PNG fixture', 'image/png'))
        self.assertEqual(self.request('/api/runs/' + ident + '/cancel', {})[0], 200)
        self.assertEqual(json.loads(self.request('/api/runs/' + ident)[1])['state'], 'stopping')

    def test_missing_artifact_and_optional_speech(self):
        self.assertEqual(self.request('/api/runs/' + 'f' * 32 + '/image.png')[0], 404)
        with patch.object(server.speech, 'synthesize', return_value=b'WAV fixture'):
            status, data, headers = self.request('/api/artsy/speech', {'text': 'Ready'})
        self.assertEqual((status, data, headers['Content-Type']), (200, b'WAV fixture', 'audio/wav'))


class IndependenceTests(unittest.TestCase):
    def test_consumer_modules_import_with_internal_ui_blocked(self):
        code = '''
import importlib.abc, sys
class BlockUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ui' or fullname.startswith('ui.'):
            raise RuntimeError('Internal UI dependency: ' + fullname)
sys.meta_path.insert(0, BlockUI())
from client.backend import server, start, worker, speech, setup_tts, tts_worker
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=server.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
