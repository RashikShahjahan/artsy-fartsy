"""Pinned local metadata and hosted capabilities for supported vision models."""
from drawing.config import MODEL_ID, MODEL_REVISION, MODEL_CONTEXT_WINDOW, TINKER_SAMPLING_MODEL_ID

LARGE_MODEL_ID = 'Qwen/Qwen3.8-27B'
QWEN36_MODEL_ID = 'Qwen/Qwen3.6-35B-A3B'
FRONTIER_MODEL_ID = 'moonshotai/Kimi-K2.6'

MODEL_PROFILES = {
    MODEL_ID: {'revision': MODEL_REVISION, 'renderer': 'qwen3_5_disable_thinking',
               'sampling_model': TINKER_SAMPLING_MODEL_ID, 'context_window': MODEL_CONTEXT_WINDOW,
               'vision': 'qwen', 'checkpoint': True},
    LARGE_MODEL_ID: {'revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                     'renderer': 'qwen3_8_disable_thinking', 'sampling_model': LARGE_MODEL_ID + ':peft:262144',
                     'context_window': 262144, 'vision': 'qwen', 'checkpoint': True},
    QWEN36_MODEL_ID: {'revision': '995ad96eacd98c81ed38be0c5b274b04031597b0',
                      'renderer': 'qwen3_5_disable_thinking', 'sampling_model': QWEN36_MODEL_ID + ':peft:262144',
                      'context_window': 262144, 'vision': 'qwen', 'checkpoint': True},
    FRONTIER_MODEL_ID: {'revision': 'b5aabbfb20227ed42becbf5541dbffd213942c58',
                        'renderer': 'kimi_k26_disable_thinking', 'sampling_model': FRONTIER_MODEL_ID + ':peft:131072',
                        'context_window': 131072, 'vision': 'kimi', 'checkpoint': False},
}


def resolve_model(model: str) -> dict:
    """Resolve a base or hosted PEFT model ID; reject unpinned metadata."""
    base = model.split(':peft:', 1)[0]
    try:
        return {'base_model': base, **MODEL_PROFILES[base]}
    except KeyError:
        raise ValueError(f'Unsupported model: {model}') from None


def load_metadata(model: str, revision: str | None = None):
    """Load the pinned tokenizer and processor without loading model weights."""
    from transformers import AutoImageProcessor, AutoTokenizer
    profile = resolve_model(model)
    base = profile['base_model']
    revision = revision or profile['revision']
    if profile['vision'] == 'kimi':
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module('tokenization_kimi.TikTokenTokenizer', base, revision=revision)
        tokenizer = cls.from_pretrained(base, revision=revision)
        processor = AutoImageProcessor.from_pretrained(base, revision=revision, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
        processor = AutoImageProcessor.from_pretrained(base, revision=revision)
    return tokenizer, processor
