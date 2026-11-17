"""Export completed Tinker fine-tunes as public, pinned Hugging Face adapters."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize_output_adapter(adapter, model_id, revision):
    """Keep Tinker's output-only LoRA on lm_head, even with tied base weights.

    Cookbook's merge-oriented Qwen remap targets embed_tokens when the base
    safetensors omit the tied lm_head. PEFT needs the runtime Linear module,
    rather than an Embedding adapter with different tensor names/semantics.
    """
    from safetensors.torch import load_file, save_file
    config_path = adapter / 'adapter_config.json'
    config = json.loads(config_path.read_text())
    if 'embed_tokens' in config['target_modules']:
        tensors = load_file(adapter / 'adapter_model.safetensors')
        remapped = {}
        for key, tensor in tensors.items():
            if '.embed_tokens.lora_' in key:
                key = key.replace('model.language_model.embed_tokens.', 'lm_head.')
            remapped[key] = tensor
        save_file(remapped, adapter / 'adapter_model.safetensors')
        config['target_modules'] = ['lm_head' if name == 'embed_tokens' else name
                                    for name in config['target_modules']]
    config.update(base_model_name_or_path=model_id, revision=revision)
    config_path.write_text(json.dumps(config, indent=2) + '\n')


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    from huggingface_hub import get_token
    token = get_token()
    if token:
        os.environ.setdefault('HF_TOKEN', token)
    from huggingface_hub import HfApi, snapshot_download
    from tinker_cookbook import weights
    from drawing.config import MODEL_ID, MODEL_REVISION
    from inference.model_catalog import _compatible_checkpoint_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', help='Defaults to the authenticated Hugging Face user')
    parser.add_argument('--checkpoint', action='append', help='Export only these checkpoint directory names')
    args = parser.parse_args()
    api = HfApi(token=token)
    namespace = args.namespace or api.whoami()['name']
    manifest_path = ROOT / 'inference/huggingface_models.json'
    entries = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    print(f'Downloading pinned base {MODEL_ID}…', flush=True)
    base = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    for final in sorted((ROOT / 'checkpoints').glob('*/final.json')):
        if args.checkpoint and final.parent.name not in args.checkpoint:
            continue
        record = json.loads(final.read_text())
        config = record.get('config', {})
        compatible = _compatible_checkpoint_model(config)
        if not compatible or compatible[0] != MODEL_ID:
            continue
        name = final.parent.name
        repo_id = f'{namespace}/{name}'
        directory = ROOT / '.cache/model-exports' / name
        adapter = directory / 'peft'
        if not (adapter / 'adapter_config.json').is_file():
            raw = directory / 'raw'
            if not (raw / 'adapter_model.safetensors').is_file():
                weights.download(tinker_path=record['sampler_path'], output_dir=str(raw))
            weights.build_lora_adapter(base_model=base, adapter_path=str(raw), output_path=str(adapter))
        # Conversion uses the pinned local snapshot for exact tensor naming;
        # published metadata must refer to the portable Hub base, not that path.
        normalize_output_adapter(adapter, MODEL_ID, MODEL_REVISION)
        stage = 'RLHF' if config.get('reward_mode') == 'rlhf' else 'RLVR' if config.get('algorithm') else 'SFT'
        card_path = ROOT / 'docs/huggingface' / f'{name}.md'
        (adapter / 'README.md').write_text(card_path.read_text() if card_path.is_file() else (
            f'---\nbase_model: {MODEL_ID}\nlibrary_name: peft\ntags:\n- lora\n- image-text-to-text\n---\n\n'
            f'# Artsy {stage}\n\nQwen3.5-4B LoRA adapter for the Artsy drawing agent.\n\n'
            f'Base revision: `{MODEL_REVISION}`. Training step: {record["step"]}. '
            'Use with the multimodal processor, tools, and `enable_thinking=False`.\n'))
        api.create_repo(repo_id, private=False, exist_ok=True)
        commit = api.upload_folder(repo_id=repo_id, folder_path=str(adapter),
                                   commit_message=f'Export completed Artsy {stage} adapter')
        entry = {'id': f'artsy/{name}', 'label': f'Qwen3.5 · 4B · {stage}',
                 'repo_id': repo_id, 'revision': commit.oid, 'base_model': MODEL_ID,
                 'base_revision': MODEL_REVISION, 'context_window': 262144}
        entries = [item for item in entries if item['id'] != entry['id']] + [entry]
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(entries, indent=2) + '\n')
        temporary.replace(manifest_path)
        print(f'Published https://huggingface.co/{repo_id} @ {commit.oid}', flush=True)


if __name__ == '__main__':
    main()
