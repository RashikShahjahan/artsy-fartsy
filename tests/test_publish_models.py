"""Verify portable Hugging Face adapter export.
"""
import unittest


class PublishModelTests(unittest.TestCase):
    def test_export_preserves_output_only_lora_with_tied_embeddings(self):
        import json
        from pathlib import Path
        import tempfile
        import torch
        from safetensors.torch import load_file, save_file
        from scripts.publish_models import normalize_output_adapter
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            tensor = torch.ones(2, 3)
            save_file({'base_model.model.model.language_model.embed_tokens.lora_A.weight': tensor},
                      adapter / 'adapter_model.safetensors')
            (adapter / 'adapter_config.json').write_text(json.dumps({'target_modules': ['embed_tokens', 'q_proj']}))
            normalize_output_adapter(adapter, 'Qwen/base', 'pinned')
            normalize_output_adapter(adapter, 'Qwen/base', 'pinned')
            loaded = load_file(adapter / 'adapter_model.safetensors')
            self.assertTrue(torch.equal(loaded['base_model.model.lm_head.lora_A.weight'], tensor))
            self.assertEqual(json.loads((adapter / 'adapter_config.json').read_text())['target_modules'],
                             ['lm_head', 'q_proj'])


