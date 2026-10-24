"""MLX-VLM inference using the original, pinned vision-language checkpoint."""


class MLXBackend:
    def __init__(self, model_id, revision, image_size):
        import mlx.core as mx
        from mlx_vlm import load

        if not mx.metal.is_available():
            raise RuntimeError("MLX inference requires an Apple Silicon GPU")
        self.model, self.processor = load(model_id, revision=revision)
        self.processor.image_processor.size = dict(image_size)

    def generate(self, messages, tools, budget, context_window):
        from mlx_vlm import generate
        from mlx_vlm.utils import prepare_inputs

        images = [part["image"] for message in messages
                  if isinstance(message["content"], list)
                  for part in message["content"] if part["type"] == "image"]
        prompt = self.processor.apply_chat_template(
            messages, tools=tools or None, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = prepare_inputs(self.processor, images=images or None, prompts=prompt)
        count = inputs["input_ids"].shape[-1]
        if count + budget > context_window:
            raise ValueError("Context exceeds model capacity")
        # Pass the prepared tensors so image expansion is counted before generation.
        inputs["mask"] = inputs.pop("attention_mask", None)
        output = generate(self.model, self.processor, prompt, max_tokens=budget,
                          temperature=0, verbose=False, **inputs)
        return output.text, count, output.generation_tokens
