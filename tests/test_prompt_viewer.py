"""Prompt review works for arbitrary dataset sizes and preserves slice labels."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import dataset_viewer


class PromptViewerTests(unittest.TestCase):
    def test_custom_prompt_file_loads_without_sft_candidate_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prompts.jsonl'
            rows = [
                {'id': 'one', 'family': 'circles', 'seed': 1, 'prompt': 'Draw a red circle.', 'benchmarkSlice': 'drawing'},
                {'id': 'two', 'family': 'rectangles', 'seed': 2, 'prompt': 'Draw a clearly fictional SAMPLE ID card.',
                 'benchmarkSlice': 'benign_boundary', 'artStyle': 'ink on white'},
            ]
            path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
            manifest, loaded = dataset_viewer.prompt_suite(path)
            self.assertEqual(manifest['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(loaded[1]['benchmarkSlice'], 'benign_boundary')
            self.assertEqual(loaded[1]['artStyle'], 'ink on white')
            with patch('sys.argv', ['viewer', '--dataset', str(path)]), \
                 patch.object(dataset_viewer, 'ThreadingHTTPServer') as server:
                dataset_viewer.main()
            handler_type = server.call_args.args[1]
            handler = object.__new__(handler_type)
            handler.path = '/'
            responses = []
            handler.respond = lambda code, body, kind: responses.append((code, body, kind))
            handler.do_GET()
            self.assertEqual(responses[0][0], 200)
            self.assertIn('1 prompt', responses[0][1])
            self.assertIn('1 benign boundary', responses[0][1])
            self.assertNotIn('Draw a red circle.', responses[0][1])
            self.assertNotIn('<option>drawing</option>', responses[0][1])
            self.assertIn('Manual safety inspector', responses[0][1])
            self.assertIn('a!==data.items.length', responses[0][1])
            self.assertNotIn('__OVERVIEW__', responses[0][1])
            review = {'dataset_sha256': manifest['sha256'], 'reviewer': 'Reviewer', 'approved': True,
                      'decisions': {r['id']: {'decision': 'approved'} for r in rows}}
            dataset_viewer.validate_review(review, manifest, loaded, require_approval=True)
            with self.assertRaisesRegex(ValueError, 'different dataset'):
                dataset_viewer.validate_review({**review, 'dataset_sha256': 'old'}, manifest, loaded)

            with patch('sys.argv', ['viewer', '--dataset', str(path), '--include-drawings']), \
                 patch.object(dataset_viewer, 'ThreadingHTTPServer') as server:
                dataset_viewer.main()
            full_handler = object.__new__(server.call_args.args[1])
            full_handler.path = '/'
            full_handler.respond = lambda code, body, kind: responses.append((code, body, kind))
            full_handler.do_GET()
            self.assertIn('2 prompts', responses[-1][1])
            self.assertIn('Draw a red circle.', responses[-1][1])

    def test_safety_approval_retains_hidden_decisions_without_approving_them(self):
        manifest = {'sha256': 'current'}
        rows = [{'id': 'drawing', 'benchmarkSlice': 'drawing'},
                {'id': 'boundary', 'benchmarkSlice': 'benign_boundary'},
                {'id': 'unsafe', 'benchmarkSlice': 'malicious_code'}]
        previous = {'dataset_sha256': 'current', 'decisions': {
            'drawing': {'decision': 'pending', 'notes': 'Keep this saved decision'},
            'unsafe': {'decision': 'rejected', 'notes': 'Old note'},
        }}
        submitted = {'dataset_sha256': 'current', 'reviewer': 'Reviewer',
                     'approved': False, 'safetyApproved': True, 'decisions': {
                         'drawing': {'decision': 'approved', 'notes': 'Stale hidden state'},
                         'boundary': {'decision': 'approved'}, 'unsafe': {'decision': 'approved'},
                     }}
        merged = dataset_viewer.merge_review(submitted, previous, manifest, rows,
                                             dataset_viewer.safety_items(rows), 'safety')
        self.assertEqual(merged['decisions']['drawing'], previous['decisions']['drawing'])
        self.assertTrue(merged['safetyApproved'])
        self.assertFalse(merged['approved'])
        self.assertEqual(merged['reviewScope'], 'safety')
        with self.assertRaises(ValueError):
            dataset_viewer.merge_review({**submitted, 'approved': True}, previous, manifest,
                                       rows, dataset_viewer.safety_items(rows), 'safety')
        with self.assertRaisesRegex(ValueError, 'All safety prompts'):
            dataset_viewer.merge_review({**submitted, 'decisions': {'unsafe': {'decision': 'rejected'}}},
                                       previous, manifest, rows, dataset_viewer.safety_items(rows), 'safety')

