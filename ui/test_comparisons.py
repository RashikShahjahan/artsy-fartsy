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


if __name__ == '__main__':
    unittest.main()
