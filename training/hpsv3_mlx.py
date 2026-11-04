"""Optional Apple GPU HPSv3 backend, matching the Experiment 1 scoring path.

Requires mlx-vlm and qwen-vl-utils in the dedicated inference environment.
Cached weights are the converted 4-bit language / bfloat16 vision checkpoint.
"""
from pathlib import Path
import json

INSTRUCTION = '\nYou are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. \n\n**Visual Quality:**  \nEvaluate the overall visual quality of the image. The following sub-dimensions should be considered:\n- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.\n- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.\n- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).\n- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.\n- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. \n\n**Text Alignment:**  \nAssess how well the image matches the textual prompt across the following sub-dimensions:\n- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.\n- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.\n- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.\n- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.\n- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.\nTextual prompt - {text_prompt}\n\n\n'
REWARD_SUFFIX = '\nPlease provide the overall ratings of this image: <|Reward|>\n\nEND\n'


class MLXRewardInferencer:
    def __init__(self, cache: Path):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models.qwen2_vl.qwen2_vl import Model
        from mlx_vlm.models.qwen2_vl.config import ModelConfig, TextConfig, VisionConfig
        from transformers import AutoProcessor

        config = json.loads((cache / 'config.json').read_text())
        config['vocab_size'] = 151658
        args = ModelConfig.from_dict(config)
        args.text_config = TextConfig.from_dict(args.text_config)
        args.vision_config = VisionConfig.from_dict(args.vision_config)
        self.model = Model(args)
        nn.quantize(self.model, group_size=64, bits=4, class_predicate=lambda p, m:
                    p.startswith('language_model.') and hasattr(m, 'to_quantized'))
        weights = mx.load(str(cache / 'mlx-4bit.safetensors'))
        self.head = {k: v.astype(mx.float32) for k, v in weights.items() if k.startswith('rm_head.')}
        self.model.load_weights([(k, v) for k, v in weights.items() if not k.startswith('rm_head.')], strict=False)
        mx.eval(self.model.parameters(), self.head)
        self.processor = AutoProcessor.from_pretrained(cache, local_files_only=True)
        self.processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<|Reward|>']})

    def score(self, prompt: str, image: str) -> float:
        import numpy as np
        import mlx.core as mx
        import mlx.nn as nn
        from qwen_vl_utils import process_vision_info

        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image, 'min_pixels': 200704, 'max_pixels': 200704},
            {'type': 'text', 'text': INSTRUCTION.format(text_prompt=prompt) + REWARD_SUFFIX}]}]
        images, _ = process_vision_info(messages)
        inputs = self.processor(text=[self.processor.apply_chat_template(messages, tokenize=False,
                                add_generation_prompt=True)], images=images, padding=True, return_tensors='np')
        ids = mx.array(inputs['input_ids'])
        features = self.model.get_input_embeddings(ids, mx.array(inputs['pixel_values']),
                                                   image_grid_thw=mx.array(inputs['image_grid_thw']))
        # Sequential positions on all three rotary axes match the HPSv3 forward.
        positions = mx.broadcast_to(mx.arange(ids.shape[1])[None, None, :], (3, 1, ids.shape[1]))
        hidden = self.model.language_model.model(ids, inputs_embeds=features.inputs_embeds, position_ids=positions)
        indices = np.where(inputs['input_ids'][0] == self.processor.tokenizer.convert_tokens_to_ids('<|Reward|>'))[0]
        if len(indices) != 1:
            raise ValueError('Expected exactly one HPSv3 reward token')
        value = hidden[:, int(indices[0]), :].astype(mx.float32)
        for layer in (0, 3, 5):
            value = value @ self.head[f'rm_head.{layer}.weight'].T + self.head[f'rm_head.{layer}.bias']
            if layer != 5:
                value = nn.relu(value)
        mx.eval(value)
        return float(value.tolist()[0][0])
