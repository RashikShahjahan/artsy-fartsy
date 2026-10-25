"""Discover completed, compatible fine-tunes without loading hosted samplers."""
import json
import os
from pathlib import Path

from drawing.config import MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]


def model_catalog(root=ROOT):
    configured = os.getenv('ARTSY_TINKER_MODEL_PATH') or None
    entries = [{'id': MODEL_ID, 'label': 'Qwen3.5 · 4B' + (' · configured checkpoint' if configured else ' · base'),
                'model_path': configured}]
    for file in sorted((root / 'checkpoints').glob('*/final.json')):
        try:
            data = json.loads(file.read_text())
            config = data.get('config', {})
            path = data.get('sampler_path')
            if (config.get('model') != MODEL_ID or config.get('tokenizer_revision') != MODEL_REVISION
                    or not isinstance(path, str) or not path.startswith('tinker://')):
                continue
            name = file.parent.name
            entries.append({'id': f'artsy/{name}', 'label': f'{name} · step {data.get("step", "final")}',
                            'model_path': path})
        except (OSError, ValueError, AttributeError, TypeError):
            continue
    return entries
