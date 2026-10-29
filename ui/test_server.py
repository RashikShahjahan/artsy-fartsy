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
        from evaluation.suite import suite
        _, specs = suite()
        self.assertIn(f'{sum(row["benchmarkSlice"] != "drawing" for row in specs)} prompts', page)
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

    def test_collection_selects_only_compatible_local_sft_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / 'checkpoints' / 'sft-test' / 'final.json'
            file.parent.mkdir(parents=True)
            (root / 'docs').mkdir()
            (root / 'docs' / 'artcanvas.md').write_text('API')
            checkpoint = {'config': {'model': LARGE_MODEL_ID, 'tokenizer_revision': server.MODEL_REVISION,
                'renderer': 'qwen3_5_preserve_thinking', 'lora_rank': 32,
                'api_doc_sha256': server.content_hash(root / 'docs' / 'artcanvas.md')},
                'state_path': 'tinker://state', 'sampler_path': 'tinker://sft'}
            file.write_text(json.dumps(checkpoint))
            catalog = [{'id': LARGE_MODEL_ID, 'model_path': None}, {'id': 'artsy/sft-test', 'model_path': 'tinker://sft'}]
            with patch.object(server, 'ROOT', root), patch.object(server, 'bridge_model_catalog', return_value=catalog):
                self.assertEqual(server.collection_profile(), {'model': 'artsy/sft-test', 'model_path': 'tinker://sft'})
                self.assertEqual(server.collection_checkpoint(), 'tinker://sft')
                for change in ({'algorithm': 'artsy-visual-dpo-v1'}, {'algorithm': 'rl'}, {'model': 'other/model'}, {'api_doc_sha256': 'stale'}):
                    file.write_text(json.dumps({**checkpoint, 'config': {**checkpoint['config'], **change}}))
                    self.assertIsNone(server.collection_profile())
                file.write_text(json.dumps(checkpoint))
                catalog[1]['model_path'] = 'tinker://different'
                self.assertIsNone(server.collection_profile())

    def test_collection_uses_canonical_specs_and_requires_current_checkpoint(self):
        spec = {'id': 'training', 'family': 'circles', 'prompt': 'Draw a unique training scene.', 'seed': 7}
        dataset = {'sha256': 'current', 'items': [spec]}
        body = {'dataset_sha256': 'current', 'prompt_ids': ['training'], 'candidates': 2, 'model_path': 'tinker://sft'}
        with patch.object(server, 'collection_dataset', return_value=dataset), patch.object(server, 'collection_profile', return_value={'model': LARGE_MODEL_ID, 'model_path': 'tinker://sft'}):
            request = server.collection_request({**body, 'prompt': 'forged', 'seed': 99})
            self.assertEqual(request['specs'], [spec])
            for change in ({'dataset_sha256': 'old'}, {'prompt_ids': ['unknown']}, {'prompt_ids': ['training', 'training']}, {'candidates': True}, {'model_path': 'tinker://other'}):
                with self.assertRaises(ValueError):
                    server.collection_request({**body, **change})

    def test_collection_excludes_frozen_prompts_even_with_new_ids(self):
        frozen = server.drawing_dataset()['items'][0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prompts.jsonl'
            path.write_text(json.dumps({**frozen, 'id': 'renamed'}) + '\n')
            with patch.object(server, 'DRAWING_DATASET', str(path)):
                with self.assertRaisesRegex(ValueError, 'training prompts'):
                    server.collection_dataset()
        with patch.object(server, 'DRAWING_DATASET', None):
            with self.assertRaisesRegex(ValueError, 'training prompts'):
                server.collection_dataset()

    def test_collection_worker_records_failure_and_cleans_shared_active(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / 'request.json').write_text('{}')
            server.ACTIVE = 'busy'
            with patch.object(server.subprocess, 'run', side_effect=OSError('worker unavailable')) as run:
                server.run_dpo_collection(directory, {'dataset_sha256': 'current'})
            self.assertIsNone(server.ACTIVE)
            self.assertEqual(json.loads((directory / 'job.json').read_text())['state'], 'failed')
            self.assertEqual(run.call_args.args[0][-2:], ['ui.dpo_collect', str(directory / 'request.json')])

    def test_collection_post_queues_once_and_respects_shared_lock(self):
        from io import BytesIO
        handler = object.__new__(server.Handler)
        handler.path = '/dpo/api/collect'
        handler.headers = {'Host': f'127.0.0.1:{server.PORT}', 'Content-Length': '2'}
        handler.rfile = BytesIO(b'{}')
        request = {'specs': [], 'dataset_sha256': 'current'}
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'COLLECTION_JOBS', Path(folder)), patch.object(server, 'collection_request', return_value=request), patch.object(server.threading, 'Thread') as thread, patch.object(handler, 'send_json') as send:
            server.ACTIVE = 'studio'
            handler.do_POST()
            self.assertEqual(send.call_args.args[1], 409)
            thread.assert_not_called()
            server.ACTIVE = None
            handler.rfile = BytesIO(b'{}')
            handler.do_POST()
            self.assertEqual(send.call_args.args[1], 202)
            directory = Path(folder) / send.call_args.args[0]['id']
            self.assertEqual(json.loads((directory / 'request.json').read_text())['output'], str(directory / 'dataset'))
            thread.return_value.start.assert_called_once()
            server.ACTIVE = None

    def test_collection_mount_scopes_candidates_and_export_directory(self):
        from io import BytesIO
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ident = 'a' * 32
            directory = root / 'jobs' / ident
            (directory / 'dataset').mkdir(parents=True)
            source = directory / 'dataset' / 'candidates.jsonl'
            source.write_text('')
            (directory / 'job.json').write_text(json.dumps({'state': 'failed'}))
            handler = object.__new__(server.Handler)
            handler.command = 'GET'
            handler.path = '/dpo?collection=' + ident
            handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
            handler.wfile = BytesIO()
            class Mounted:
                def do_GET(self):
                    self.wfile.write(self.path.encode())
            with patch.object(server, 'COLLECTION_JOBS', root / 'jobs'), patch.object(server, 'DPO_OUTPUT', root / 'exports'), patch('ui.dpo_review.make_handler', return_value=Mounted) as factory:
                server.MOUNTED_HANDLERS.clear()
                handler.do_GET()
                args, kwargs = factory.call_args
                self.assertEqual(args[0], source)
                self.assertEqual(args[2], root / 'exports' / 'exports' / ident)
                self.assertEqual(kwargs['base_path'], '/dpo/collections/' + ident)
                self.assertEqual(handler.wfile.getvalue(), b'/')
                handler.path = '/dpo/collections/' + ident + '/api/image?id=x'
                handler.wfile = BytesIO()
                handler.do_GET()
                self.assertEqual(handler.wfile.getvalue(), b'/api/image?id=x')
                source.write_text('changed')
                handler.do_GET()
                self.assertEqual(factory.call_count, 2)
                self.assertEqual(len(server.MOUNTED_HANDLERS), 1)

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
                         ['Qwen3.5 · 4B', 'GPT-6 Luna'])
        for model in server.DIRECT_MODELS:
            with tempfile.TemporaryDirectory() as temp, patch.object(server.subprocess, 'run') as run:
                run.return_value.returncode = 0
                server.run_job(Path(temp), {'prompt': 'circle', 'seed': 42, 'model': model['id']})
                env = run.call_args.kwargs['env']
                self.assertEqual(env['ARTSY_MODEL_URL'], server.OPENROUTER_MODEL_URL)
                self.assertNotIn('ARTSY_MODEL_API_KEY', env)
        with self.assertRaisesRegex(ValueError, 'available Studio model'):
            server.generation_request({'prompt': 'circle', 'seed': 42, 'model': 'artsy/retired'})
        for retired in ('thinkingmachines/Inkling:peft:262144', 'thinkingmachines/Inkling-Small:peft:262144'):
            with self.assertRaisesRegex(ValueError, 'available Studio model'):
                server.generation_request({'prompt': 'circle', 'seed': 42, 'model': retired})

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
