"""Checkpoint discovery needs no inference credentials or hosted calls."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from drawing.config import MODEL_ID, MODEL_REVISION
from inference.model_catalog import FRONTIER_MODEL_ID, LARGE_MODEL_ID, QWEN36_MODEL_ID, ModelCatalogCache, model_catalog


class ModelCatalogTests(unittest.TestCase):
    def test_large_qwen_profile_enables_thinking(self):
        from inference.model_catalog import HOSTED_MODEL_PROFILES
        self.assertEqual(HOSTED_MODEL_PROFILES[LARGE_MODEL_ID]['renderer'], 'qwen3_8_disable_thinking')

    def test_qwen36_profile_uses_native_vision_renderer_and_extended_context(self):
        from inference.model_catalog import HOSTED_MODEL_PROFILES
        profile = HOSTED_MODEL_PROFILES[QWEN36_MODEL_ID]
        self.assertEqual(profile['renderer'], 'qwen3_5_disable_thinking')
        self.assertEqual(profile['sampling_model'], QWEN36_MODEL_ID + ':peft:262144')
        self.assertEqual(profile['context_window'], 262144)
        self.assertEqual(len(profile['revision']), 40)

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
            self.assertEqual([entry['id'] for entry in catalog], [MODEL_ID, 'artsy/trained', LARGE_MODEL_ID, QWEN36_MODEL_ID, FRONTIER_MODEL_ID])
            self.assertEqual(catalog[1]['model_path'], valid['sampler_path'])
            self.assertIn('step 29', catalog[1]['label'])

    def test_discovers_qwen36_and_qwen38_sft_sampler_checkpoints(self):
        from inference.model_catalog import HOSTED_MODEL_PROFILES, LARGE_MODEL_ID, QWEN36_MODEL_ID
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                ('sft-qwen38', LARGE_MODEL_ID, 'qwen3_8_disable_thinking', None),
                ('sft-qwen36', QWEN36_MODEL_ID, 'qwen3_5_disable_thinking', None),
                ('invalid-renderer', QWEN36_MODEL_ID, 'qwen3_8_disable_thinking', None),
                ('invalid-revision', LARGE_MODEL_ID, 'qwen3_8_disable_thinking', 'wrong'),
                ('unsupported', 'Qwen/Unrecognized', 'qwen3_8_disable_thinking', None),
            ]
            for name, model, renderer, revision in rows:
                folder = root / 'checkpoints' / name
                folder.mkdir(parents=True)
                (folder / 'final.json').write_text(json.dumps({
                    'step': 42,
                    'config': {'model': model, 'renderer': renderer,
                               'tokenizer_revision': revision},
                    'sampler_path': f'tinker://{name}/sampler',
                    'state_path': f'tinker://{name}/state',
                }))
            records = {entry['id']: entry for entry in model_catalog(root)}
            self.assertTrue({'artsy/sft-qwen38', 'artsy/sft-qwen36'} <= records.keys())
            self.assertNotIn('artsy/invalid-renderer', records)
            self.assertNotIn('artsy/invalid-revision', records)
            self.assertNotIn('artsy/unsupported', records)
            for name, model in [('sft-qwen38', LARGE_MODEL_ID),
                                ('sft-qwen36', QWEN36_MODEL_ID)]:
                record = records['artsy/' + name]
                self.assertEqual(record['base_model'], model)
                self.assertEqual(record['model_path'], f'tinker://{name}/sampler')
                for key in ('sampling_model', 'revision', 'renderer', 'context_window'):
                    self.assertEqual(record[key], HOSTED_MODEL_PROFILES[model][key])
                self.assertIn('step 42', record['label'])

    def test_legacy_configured_checkpoint_is_labeled(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'ARTSY_TINKER_MODEL_PATH': 'tinker://legacy'}):
            base = model_catalog(Path(temp))[0]
            self.assertEqual(base['model_path'], 'tinker://legacy')
            self.assertIn('configured checkpoint', base['label'])


class ModelCatalogCacheTests(unittest.TestCase):
    def test_reuses_snapshot_refreshes_and_isolates_mutations(self):
        cache = ModelCatalogCache(ttl=5)
        entries = [{"id": "first", "nested": {"value": 1}}]
        with patch("time.monotonic", return_value=10) as clock, patch.dict(os.environ, {"ARTSY_TINKER_MODEL_PATH": ""}):
            from unittest.mock import Mock
            loader = Mock(return_value=entries)
            first = cache.get(loader)
            first[0]["nested"]["value"] = 9
            entries[0]["id"] = "changed"
            self.assertEqual(cache.get(loader)[0], {"id": "first", "nested": {"value": 1}})
            loader.assert_called_once()
            clock.return_value = 15
            self.assertEqual(cache.get(loader)[0]["id"], "changed")
            self.assertEqual(loader.call_count, 2)
            os.environ["ARTSY_TINKER_MODEL_PATH"] = "tinker://new"
            cache.get(loader)
            self.assertEqual(loader.call_count, 3)

    def test_concurrent_readers_share_one_refresh(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import Mock
        cache = ModelCatalogCache()
        loader = Mock(return_value=[{"id": "base"}])
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: cache.get(loader), range(16)))
        loader.assert_called_once()
        self.assertEqual(results, [[{"id": "base"}]] * 16)


if __name__ == '__main__':
    unittest.main()
