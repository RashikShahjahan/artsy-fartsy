"""Discover completed, compatible fine-tunes without loading hosted samplers."""
import json
import os
from pathlib import Path
import re

from drawing.config import MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
LARGE_MODEL_ID = 'Qwen/Qwen3.8-27B'
QWEN36_MODEL_ID = 'Qwen/Qwen3.6-35B-A3B'
FRONTIER_MODEL_ID = 'moonshotai/Kimi-K2.6'
# Verified against Tinker's sampling catalog; processors remain local, weights hosted.
HOSTED_MODEL_PROFILES = {
    LARGE_MODEL_ID: {
        'revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
        'renderer': 'qwen3_8_disable_thinking',
        'sampling_model': LARGE_MODEL_ID + ':peft:262144',
        'context_window': 262144,
    },
    QWEN36_MODEL_ID: {
        'revision': '995ad96eacd98c81ed38be0c5b274b04031597b0',
        'renderer': 'qwen3_5_disable_thinking',
        'sampling_model': QWEN36_MODEL_ID + ':peft:262144',
        'context_window': 262144,
    },
    FRONTIER_MODEL_ID: {
        'revision': 'b5aabbfb20227ed42becbf5541dbffd213942c58',
        'renderer': 'kimi_k26_disable_thinking',
        'sampling_model': FRONTIER_MODEL_ID + ':peft:131072',
        'context_window': 131072,
    },
}



def _compatible_checkpoint_model(config):
    """Resolve checkpoint model metadata without constructing a hosted client."""
    if not isinstance(config, dict):
        return None
    model = config.get('model')
    if not isinstance(model, str):
        return None
    base_model = model.split(':peft:', 1)[0]
    profile = HOSTED_MODEL_PROFILES.get(base_model)
    if base_model == MODEL_ID:
        if config.get('tokenizer_revision') != MODEL_REVISION:
            return None
    elif (base_model not in (LARGE_MODEL_ID, QWEN36_MODEL_ID)
          or config.get('renderer') != profile['renderer']
          or config.get('tokenizer_revision') not in (None, profile['revision'])):
        return None
    return base_model, profile


def _fine_tune_entry(model_id, label, model_path, config):
    compatible = _compatible_checkpoint_model(config)
    if (compatible is None or not isinstance(model_path, str)
            or not model_path.startswith('tinker://')):
        return None
    base_model, profile = compatible
    entry = {'id': model_id, 'label': label, 'model_path': model_path,
             'base_model': base_model}
    if profile:
        entry.update(profile)
    return entry


def _grpo_step_entries(root):
    """Discover published Cookbook sampler checkpoints in unfinished GRPO runs.

    Intermediate checkpoints are referenced by checkpoints.jsonl; they do not
    have final.json files. Only named, complete saved steps are selectable.
    """
    entries = []
    for manifest in sorted((root / 'checkpoints').glob('*/training-manifest.json')):
        name = manifest.parent.name
        # Studio model aliases must match the same validation as final models.
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,115}', name):
            continue
        try:
            plan = json.loads(manifest.read_text(encoding='utf-8'))
            config = plan.get('config') if isinstance(plan, dict) else None
            if (not isinstance(config, dict)
                    or not str(config.get('algorithm', '')).startswith('tinker-cookbook-')
                    or _compatible_checkpoint_model(config) is None):
                continue
            records_path = manifest.parent / 'checkpoints.jsonl'
            if not records_path.is_file():
                continue
            steps = {}
            with records_path.open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            continue
                        checkpoint_name = record.get('name')
                        step = record.get('batch')
                        path = record.get('sampler_path')
                        state = record.get('state_path')
                        if (not isinstance(checkpoint_name, str)
                                or not re.fullmatch(r'(?:step-|checkpoint-)?[0-9]{1,9}', checkpoint_name)
                                or type(step) is not int or step < 0
                                or not isinstance(state, str) or not state.startswith('tinker://')):
                            continue
                        entry = _fine_tune_entry(
                            f'artsy/{name}--grpo-step-{step:06d}',
                            f'{name} · GRPO · step {step} · {config["model"].split(":peft:", 1)[0]}',
                            path, config)
                        if entry is not None:
                            steps[step] = entry
                    except (ValueError, TypeError, AttributeError):
                        # Ignore incomplete lines while training appends records.
                        continue
            # Offer the newest saved policy first in large GRPO runs.
            entries.extend(entry for _, entry in sorted(steps.items(), reverse=True))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return entries


def model_catalog(root=ROOT):
    configured = os.getenv('ARTSY_TINKER_MODEL_PATH') or None
    entries = [{'id': MODEL_ID, 'label': 'Qwen3.5 · 4B' + (' · configured checkpoint' if configured else ' · base'),
                'model_path': configured}]
    for file in sorted((root / 'checkpoints').glob('*/final.json')):
        try:
            data = json.loads(file.read_text())
            if not isinstance(data, dict):
                continue
            config = data.get('config')
            compatible = _compatible_checkpoint_model(config)
            if compatible is None:
                continue
            base_model, _ = compatible
            name = file.parent.name
            stage = 'GRPO' if config.get('algorithm') else 'SFT'
            entry = _fine_tune_entry(
                f'artsy/{name}',
                f'{name} · {stage} · {base_model.rsplit("/", 1)[-1]} · step {data.get("step", "final")}',
                data.get('sampler_path'), config)
            if entry is not None:
                entries.append(entry)
        except (OSError, ValueError, AttributeError, TypeError):
            continue
    entries.extend(_grpo_step_entries(root))
    entries.append({'id': LARGE_MODEL_ID, 'label': 'Qwen3.8 · 27B · Tinker base',
                    'model_path': None, **HOSTED_MODEL_PROFILES[LARGE_MODEL_ID]})
    entries.append({'id': QWEN36_MODEL_ID, 'label': 'Qwen3.6 · 35B-A3B · Tinker base',
                    'model_path': None, **HOSTED_MODEL_PROFILES[QWEN36_MODEL_ID]})
    entries.append({'id': FRONTIER_MODEL_ID, 'label': 'Kimi K2.6 · 1T MoE · Tinker base',
                    'model_path': None, **HOSTED_MODEL_PROFILES[FRONTIER_MODEL_ID]})
    return entries


class ModelCatalogCache:
    """Refresh checkpoint discovery periodically, returning isolated snapshots."""

    def __init__(self, ttl=5.0):
        import threading
        self.ttl = ttl
        self._lock = threading.Lock()
        self._entries = None
        self._expires = 0.0
        self._configured = None

    def get(self, loader=model_catalog):
        import copy
        import time
        # Serialize discovery as well as publication so concurrent requests do not
        # all scan the checkpoint directory when the snapshot expires.
        with self._lock:
            configured = os.getenv('ARTSY_TINKER_MODEL_PATH') or None
            now = time.monotonic()
            if self._entries is None or now >= self._expires or configured != self._configured:
                entries = loader()
                self._entries = copy.deepcopy(entries)
                self._configured = configured
                self._expires = time.monotonic() + self.ttl
            return copy.deepcopy(self._entries)
