"""Saved comparison access must stay within verified evaluation artifacts."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import comparisons, server


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'paired-run'
        (self.folder / 'records').mkdir(parents=True)
        image = self.root / 'base' / 'artifacts' / 'prompt-1' / 'image.png'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'verified final image')
        self.image = image
        self.data = {'fingerprint': 'current-inputs', 'model': 'openai/gpt-6-luna',
                     'source_models': ['base-model', 'sft-model'],
                     'pairs': [{'id': 'prompt', 'prompt': '<untrusted prompt>', 'members': {
                         'base': {'image': str(image), 'outcome': 'submitted',
                                  'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()},
                         'sft': {'image': None, 'outcome': 'incomplete', 'image_sha256': None}}}]}
        self.save_manifest()
        self.record = {'fingerprint': 'current-inputs', 'scores': {
            'base': {'adherence_score': 3, 'visual_quality_score': 2, 'comments': 'Visible content.'},
            'sft': {'adherence_score': None, 'visual_quality_score': None, 'comments': 'No final image.'}},
            'preferred': 'base', 'comparison': 'Only B has an output.',
            'label_mapping': {'A': 'sft', 'B': 'base'}, 'raw_response': {'private': 'not exposed'}}
        self.save_record()

    def save_manifest(self):
        (self.folder / 'manifest.json').write_text(json.dumps(self.data))

    def save_record(self):
        (self.folder / 'records/prompt.json').write_text(json.dumps(self.record))

    def test_packet_keeps_missing_images_and_labels_without_exposing_paths_or_raw_response(self):
        result = comparisons.packet(self.root, 'paired-run')
        pair = result['pairs'][0]
        self.assertEqual(pair['labels'], {'A': 'sft', 'B': 'base'})
        self.assertIsNone(pair['members']['sft']['image_url'])
        self.assertIsNone(pair['members']['sft']['scores']['adherence_score'])
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertNotIn('raw_response', json.dumps(result))
        self.assertEqual(comparisons.image(self.root, 'paired-run', 'prompt', 'base'), self.image.read_bytes())

    def test_changed_image_is_rejected(self):
        self.image.write_bytes(b'replaced')
        with self.assertRaises(ValueError):
            comparisons.image(self.root, 'paired-run', 'prompt', 'base')

    def test_manifest_cannot_expose_files_outside_eval_root_even_via_symlink(self):
        outside = self.root.parent / 'outside-image.png'
        self.data['pairs'][0]['members']['base']['image'] = str(outside)
        self.save_manifest()
        with self.assertRaises(ValueError):
            comparisons.image(self.root, 'paired-run', 'prompt', 'base')
        self.image.unlink()
        self.image.symlink_to(outside)
        self.data['pairs'][0]['members']['base']['image'] = str(self.image)
        self.save_manifest()
        with self.assertRaises(ValueError):
            comparisons.image(self.root, 'paired-run', 'prompt', 'base')
        with self.assertRaises(ValueError):
            comparisons.packet(self.root, '../paired-run')

    def test_stale_judgments_are_pending(self):
        self.record['fingerprint'] = 'old-inputs'
        self.save_record()
        pair = comparisons.packet(self.root, 'paired-run')['pairs'][0]
        self.assertIsNone(pair['preferred'])
        self.assertIsNone(pair['members']['base']['scores'])

    def test_four_model_comparison_preserves_every_role(self):
        self.data['roles'] = ['base', 'sft', 'rank64', 'rank128']
        self.data['source_models'] += ['rank64-model', 'rank128-model']
        for role in ['rank64', 'rank128']:
            self.data['pairs'][0]['members'][role] = dict(self.data['pairs'][0]['members']['base'])
            self.record['scores'][role] = dict(self.record['scores']['base'])
        self.save_manifest()
        self.save_record()
        result = comparisons.packet(self.root, 'paired-run')
        self.assertEqual(list(result['models']), self.data['roles'])
        self.assertEqual(set(result['pairs'][0]['members']), set(self.data['roles']))
        self.assertEqual(comparisons.image(self.root, 'paired-run', 'prompt', 'rank128'), self.image.read_bytes())
        with self.assertRaises(ValueError):
            comparisons.image(self.root, 'paired-run', 'prompt', 'unknown')

    def test_waiting_pipeline_is_visible_without_fabricating_judgments(self):
        source = self.root / 'base'
        (source / 'summary.json').write_text(json.dumps({'model': 'base-model', 'ids': ['prompt']}))
        waiting = self.root / 'new-comparison'
        waiting.mkdir()
        (waiting / 'pipeline.json').write_text(json.dumps({
            'runs': [f'base={source}'], 'status': 'waiting_for_evals'}))
        self.assertEqual(comparisons.list_comparisons(self.root)[0]['id'], 'new-comparison')
        data = comparisons.packet(self.root, 'new-comparison')
        self.assertTrue(data['pending'])
        self.assertEqual(data['progress']['base'], 0)
        self.assertEqual(data['pairs'], [])
        self.assertNotIn(str(self.root), json.dumps(data))

    def test_routes_serve_comparison_but_do_not_allow_arbitrary_files(self):
        handler = object.__new__(server.Handler)
        handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
        handler.path = '/comparisons'
        with patch.object(handler, 'send_page') as send:
            handler.do_GET()
        self.assertIn('aria-current="page">Eval comparison', send.call_args.args[0])
        self.assertNotIn('__NAVIGATION__', send.call_args.args[0])
        for path in ('/api/comparisons/../outside', '/api/comparisons/paired-run/images/prompt/program.py'):
            handler.path = path
            with patch.object(handler, 'send_json') as send:
                handler.do_GET()
            self.assertEqual(send.call_args.args[1], 404)
        handler.path = '/api/comparisons/paired-run/images/prompt/base'
        handler.wfile = BytesIO()
        with patch.object(server, 'ROOT', self.root.parent), \
                patch.object(comparisons, 'image', return_value=b'png') as get_image, \
                patch.object(handler, 'send_response') as status, \
                patch.object(handler, 'send_header'), patch.object(handler, 'end_headers'):
            handler.do_GET()
        status.assert_called_once_with(200)
        get_image.assert_called_once()
        self.assertEqual(handler.wfile.getvalue(), b'png')

    def collection(self):
        folder = self.root / 'combined'
        folder.mkdir()
        config = {'label': 'Combined evaluation', 'prompts': [
            {'id': 'prompt', 'prompt': '<untrusted prompt>', 'seed': 42},
            {'id': 'new', 'prompt': 'a new drawing', 'seed': 43}],
            'original_comparison': 'paired-run', 'original_ids': ['prompt'],
            'added_runs': {'base': 'new-base', 'sft': 'new-sft'}, 'added_comparison': 'new-judge'}
        (folder / 'collection.json').write_text(json.dumps(config))
        directory = self.root / 'new-base/artifacts/new-1'
        directory.mkdir(parents=True)
        png = b'new image'
        (directory / 'image.png').write_bytes(png)
        result = {'prompt': 'a new drawing', 'seed': 43, 'model': 'openai-api/artsy/base-model',
                  'scores': {'render_valid': 1.0}, 'outcome': 'submitted',
                  'imageHash': hashlib.sha256(png).hexdigest()}
        (directory / 'result.json').write_text(json.dumps(result))
        return folder, directory, result

    def test_combined_view_reuses_original_scores_and_shows_pending_new_output(self):
        _, directory, _ = self.collection()
        packet = comparisons.packet(self.root, 'combined')
        original, added = packet['pairs']
        self.assertEqual([p['subset'] for p in packet['pairs']], ['original', 'added'])
        self.assertEqual(original['members']['base']['scores'], self.record['scores']['base'])
        self.assertTrue(original['members']['sft']['finished'])
        self.assertFalse(added['members']['sft']['finished'])
        self.assertTrue(added['members']['base']['finished'])
        self.assertIsNone(added['members']['base']['scores'])
        self.assertEqual(packet['progress'], {'base': 2, 'sft': 1})
        self.assertEqual(comparisons.image(self.root, 'combined', 'new', 'base'), b'new image')
        self.assertEqual(comparisons.image(self.root, 'combined', 'prompt', 'base'), self.image.read_bytes())
        self.assertNotIn(str(self.root), json.dumps(packet))
        directory.joinpath('image.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            comparisons.image(self.root, 'combined', 'new', 'base')

    def test_combined_view_attaches_new_scores_only_when_image_inputs_match(self):
        _, _, result = self.collection()
        # The second model has a completed refusal, with no final image.
        second = self.root / 'new-sft/artifacts/new-1'
        second.mkdir(parents=True)
        second.joinpath('result.json').write_text(json.dumps({**result, 'model': 'sft-model',
            'scores': {'render_valid': 0.0}, 'outcome': 'refused', 'imageHash': None}))
        judge = self.root / 'new-judge'
        judge.joinpath('records').mkdir(parents=True)
        manifest = {**self.data, 'pairs': [{'id': 'new', 'prompt': 'a new drawing', 'members': {
            'base': {'outcome': 'submitted', 'image': 'unused', 'image_sha256': result['imageHash']},
            'sft': {'outcome': 'refused', 'image': None, 'image_sha256': None}}}]}
        judge.joinpath('manifest.json').write_text(json.dumps(manifest))
        judge.joinpath('records/new.json').write_text(json.dumps(self.record))
        packet = comparisons.packet(self.root, 'combined')
        self.assertEqual(packet['pairs'][1]['preferred'], 'base')
        manifest['pairs'][0]['members']['base']['image_sha256'] = 'stale'
        judge.joinpath('manifest.json').write_text(json.dumps(manifest))
        self.assertIsNone(comparisons.packet(self.root, 'combined')['pairs'][1]['preferred'])

    def test_combined_view_rejects_changed_prompt_seed_or_model(self):
        _, directory, result = self.collection()
        for key, value in [('seed', 99), ('prompt', 'wrong prompt'), ('model', 'wrong model')]:
            directory.joinpath('result.json').write_text(json.dumps({**result, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                comparisons.packet(self.root, 'combined')


if __name__ == '__main__':
    unittest.main()
