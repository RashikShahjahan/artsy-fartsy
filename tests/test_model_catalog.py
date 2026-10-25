"""Checkpoint discovery needs no inference credentials or hosted calls."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from drawing.config import MODEL_ID, MODEL_REVISION
from inference.model_catalog import model_catalog


class ModelCatalogTests(unittest.TestCase):
    def test_discovers_only_completed_compatible_sampler_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'ARTSY_TINKER_MODEL_PATH': ''}):
            root = Path(temp)
            valid = {'step': 29, 'config': {'model': MODEL_ID, 'tokenizer_revision': MODEL_REVISION},
                     'sampler_path': 'tinker://test/sampler_weights/final'}
            records = {'trained': valid, 'wrong-model': {**valid, 'config': {'model': 'other'}},
                       'state-only': {**valid, 'sampler_path': None}, 'malformed': []}
            for name, data in records.items():
                directory = root / 'checkpoints' / name
                directory.mkdir(parents=True)
                (directory / 'final.json').write_text(json.dumps(data))
            unfinished = root / 'checkpoints' / 'in-progress'
            unfinished.mkdir()
            (unfinished / 'checkpoint-000001.json').write_text(json.dumps(valid))
            catalog = model_catalog(root)
            self.assertEqual([entry['id'] for entry in catalog], [MODEL_ID, 'artsy/trained'])
            self.assertEqual(catalog[1]['model_path'], valid['sampler_path'])
            self.assertIn('step 29', catalog[1]['label'])

    def test_legacy_configured_checkpoint_is_labeled(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'ARTSY_TINKER_MODEL_PATH': 'tinker://legacy'}):
            base = model_catalog(Path(temp))[0]
            self.assertEqual(base['model_path'], 'tinker://legacy')
            self.assertIn('configured checkpoint', base['label'])


if __name__ == '__main__':
    unittest.main()
