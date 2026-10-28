"""Job lifecycle checks without model requests or Docker."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import server
from drawing.config import MODEL_ID as LARGE_MODEL_ID


class JobTests(unittest.TestCase):
    def test_drawing_page_is_allowed_without_exposing_arbitrary_files(self):
        handler = object.__new__(server.Handler)
        handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
        handler.path = '/drawings'
        with patch.object(handler, 'send_page') as serve:
            handler.do_GET()
            page = serve.call_args.args[0]
            self.assertIn('Drawing studio', page)
            self.assertIn('From dataset', page)
            self.assertIn('href="/review"', page)
            self.assertNotIn('__NAVIGATION__', page)
        handler.path = '/server.py'
        with patch.object(handler, 'send_json') as response:
            handler.do_GET()
            self.assertEqual(response.call_args.args[1], 404)

    def test_review_mount_saves_to_the_configured_file_and_rejects_other_origins(self):
        from io import BytesIO
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'review.json'
            handler = object.__new__(server.Handler)
            handler.command = 'GET'
            handler.path = '/review'
            handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
            handler.wfile = BytesIO()
            handler.send_response = lambda *args: None
            handler.send_header = lambda *args: None
            handler.end_headers = lambda: None
            with patch.object(server, 'PROMPT_REVIEW_OUTPUT', output):
                handler.do_GET()
                page = handler.wfile.getvalue().decode()
                self.assertIn('aria-current="page">Prompt review', page)
                self.assertIn("fetch('/review/review'", page)
                self.assertIn('href="/style.css"', page)
                self.assertNotIn('<style>', page)
                manifest, _ = __import__('evaluation.benchmark', fromlist=['suite']).suite()
                body = json.dumps({'dataset_sha256': manifest['sha256'], 'reviewer': 'Test',
                                   'approved': False, 'decisions': {}}).encode()
                handler.command = 'POST'
                handler.path = '/review/review'
                handler.headers['Content-Length'] = str(len(body))
                handler.rfile = BytesIO(body)
                handler.do_POST()
                self.assertEqual(json.loads(output.read_text())['reviewer'], 'Test')
                handler.headers['Origin'] = 'https://untrusted.example'
                with patch.object(handler, 'send_json') as response:
                    handler.do_POST()
                    self.assertEqual(response.call_args.args[1], 403)

    def test_safety_scope_reuses_approval_rules_and_excludes_ordinary_drawings(self):
        from io import BytesIO
        handler = object.__new__(server.Handler)
        handler.command = 'GET'
        handler.path = '/review?scope=safety'
        handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
        handler.wfile = BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        with patch.object(server, 'DRAWING_DATASET', None):
            handler.do_GET()
        page = handler.wfile.getvalue().decode()
        self.assertIn('24 prompts', page)
        self.assertIn('"scope": "safety"', page)
        self.assertIn('href="/review?scope=safety" aria-current="true"', page)
        self.assertNotIn('<option>drawing</option>', page)

    def test_candidate_mount_uses_scoped_record_and_review_urls(self):
        from io import BytesIO
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'candidates.jsonl'
            path.write_text(json.dumps({'id': 'one', 'spec': {'prompt': 'Draw a circle.'},
                                        'status': 'review'}) + '\n')
            handler = object.__new__(server.Handler)
            handler.command = 'GET'
            handler.path = '/candidates'
            handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
            handler.wfile = BytesIO()
            handler.send_response = lambda *args: None
            handler.send_header = lambda *args: None
            handler.end_headers = lambda: None
            with patch.object(server, 'CANDIDATE_DATASET', str(path)):
                handler.do_GET()
                page = handler.wfile.getvalue().decode()
                self.assertIn("fetch('/candidates/api/record?id='", page)
                self.assertIn("post('/candidates/api/export'", page)
                self.assertIn('aria-current="page">SFT candidates', page)
                handler.path = '/candidates/api/record?id=one'
                handler.wfile = BytesIO()
                handler.do_GET()
                record = json.loads(handler.wfile.getvalue())
                self.assertEqual(record['record']['id'], 'one')
                handler.command = 'POST'
                handler.path = '/candidates/api/review'
                import hashlib
                body = json.dumps({'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'reviewer': 'Test', 'decisions': {'one': {'decision': 'rejected'}}}).encode()
                handler.headers['Content-Length'] = str(len(body))
                handler.rfile = BytesIO(body)
                handler.do_POST()
                self.assertEqual(json.loads((path.parent / 'manual-review.json').read_text())
                                 ['decisions']['one']['decision'], 'rejected')

    def test_dataset_runs_use_canonical_drawing_and_seed_with_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prompts.jsonl'
            rows = [
                {'id': 'drawing', 'family': 'circles', 'prompt': 'Draw two circles.', 'seed': 123, 'benchmarkSlice': 'drawing'},
                {'id': 'boundary', 'family': 'circles', 'prompt': 'Draw a fictional SAMPLE ID.', 'seed': 321, 'benchmarkSlice': 'benign_boundary'},
                {'id': 'unsafe', 'family': 'safety_refusal', 'prompt': 'Unsafe test request', 'seed': 7,
                 'kind': 'refusal', 'refusal': 'I cannot help with that request.', 'safetyCategory': 'malicious_code', 'benchmarkSlice': 'malicious_code'},
            ]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with patch.object(server, 'DRAWING_DATASET', str(path)):
                dataset = server.drawing_dataset()
                self.assertEqual([row['id'] for row in dataset['items']], ['drawing'])
                body = {'prompt_id': 'drawing', 'dataset_sha256': dataset['sha256'], 'model': LARGE_MODEL_ID,
                        'prompt': 'Forged text', 'seed': 999}
                request = server.generation_request(body)
                self.assertEqual((request['prompt'], request['seed']), ('Draw two circles.', 123))
                self.assertEqual(request['dataset_path'], str(path.resolve()))
                self.assertEqual(request['dataset_sha256'], dataset['sha256'])
                for changes in ({'prompt_id': 'unsafe'}, {'prompt_id': 'boundary'},
                                {'dataset_sha256': 'stale'}, {'model': 'not/a-supported-model'}):
                    with self.assertRaises(ValueError):
                        server.generation_request({**body, **changes})
            target = Path(folder) / 'run'
            target.mkdir()
            with patch.object(server.subprocess, 'run') as run:
                run.return_value.returncode = 0
                server.run_job(target, request)
            environment = run.call_args.kwargs['env']
            self.assertEqual(environment['ARTSY_MODEL_URL'], 'http://127.0.0.1:8080/v1')
            self.assertEqual(json.loads((target/'job.json').read_text())['prompt_id'], 'drawing')

    def test_history_is_scoped_to_the_dataset_revision(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'JOBS', Path(folder)):
            for ident, sha in (('a'*32, 'current'), ('b'*32, 'older')):
                target = Path(folder) / ident
                target.mkdir()
                (target/'job.json').write_text(json.dumps({'dataset_sha256': sha, 'prompt_id': 'one',
                    'model': LARGE_MODEL_ID, 'state': 'completed', 'seed': 12}))
            runs = server.drawing_runs({'sha256': 'current'})
            self.assertEqual([run['id'] for run in runs], ['a'*32])

    def test_bridge_capabilities_distinguish_legacy_and_finetune_routing(self):
        with patch.object(server, 'urlopen') as request:
            response = request.return_value.__enter__.return_value
            response.read.return_value = json.dumps({'model': 'Qwen/Qwen3.5-4B'}).encode()
            self.assertEqual(server.bridge_model_catalog(), [{'id': 'Qwen/Qwen3.5-4B', 'model_path': None}])
            response.read.return_value = json.dumps({'models': [{'id': 'artsy/trained'}]}).encode()
            self.assertEqual(server.bridge_model_catalog(), [{'id': 'artsy/trained'}])
            response.read.return_value = b'invalid json'
            self.assertIsNone(server.bridge_model_catalog())
        with patch.object(server, 'urlopen', side_effect=OSError('offline')):
            self.assertIsNone(server.bridge_model_catalog())

    def test_command_and_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            request = {'prompt': "draw 'a'; $(anything)", 'seed': 42, 'model': 'Qwen/Qwen3.5-4B'}
            with patch.object(server.subprocess, 'run') as run:
                run.return_value.returncode = 0
                server.ACTIVE = 'test'
                server.run_job(directory, request)
            args = run.call_args.args[0]
            self.assertEqual(args[-3:], [request['prompt'], '42', request['model']])
            self.assertFalse(run.call_args.kwargs.get('shell', False))
            self.assertEqual(json.loads((directory / 'job.json').read_text())['state'], 'completed')
            self.assertIsNone(server.ACTIVE)

    def test_studio_models_and_direct_routing(self):
        self.assertEqual([model['label'] for model in server.model_catalog()],
                         ['Qwen3.5 · 4B', 'Inkling', 'Inkling Small'])
        for model in server.INKLING_MODELS:
            with tempfile.TemporaryDirectory() as temp, patch.object(server.subprocess, 'run') as run:
                run.return_value.returncode = 0
                server.run_job(Path(temp), {'prompt': 'circle', 'seed': 42, 'model': model['id']})
                env = run.call_args.kwargs['env']
                self.assertEqual(env['ARTSY_MODEL_URL'], server.TINKER_MODEL_URL)
                self.assertNotIn('ARTSY_MODEL_API_KEY', env)
        with self.assertRaisesRegex(ValueError, 'available Studio model'):
            server.generation_request({'prompt': 'circle', 'seed': 42, 'model': 'artsy/retired'})

    def test_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(server.subprocess, 'run', side_effect=OSError('missing Python')):
                server.ACTIVE = 'test'
                server.run_job(directory, {'prompt': 'circle', 'seed': 0, 'model': 'Qwen/Qwen3.5-4B'})
            status = json.loads((directory / 'job.json').read_text())
            self.assertEqual(status['state'], 'failed')
            self.assertEqual(status['error'], 'missing Python')
            self.assertIsNone(server.ACTIVE)


if __name__ == '__main__':
    unittest.main()
