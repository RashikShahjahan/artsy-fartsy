"""Hosted Qwen sampling with the same renderer used for Tinker training."""
import json

RENDERER = "qwen3_5_disable_thinking"


class TinkerBackend:
    def __init__(self, model_id, revision, image_size, model_path=None):
        import tinker
        from transformers import AutoImageProcessor, AutoTokenizer
        from tinker_cookbook import renderers

        self.service = tinker.ServiceClient()
        self.client = self.service.create_sampling_client(
            **({"model_path": model_path} if model_path else {"base_model": model_id}))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.image_processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.image_size = image_size
        self.renderer = renderers.get_renderer(
            RENDERER, self.tokenizer, image_processor=self.image_processor)

    def render(self, messages, tools):
        from tinker_cookbook.renderers import ToolCall

        native = []
        for original in messages:
            message = dict(original)
            if isinstance(message["content"], list):
                parts = []
                for part in message["content"]:
                    if part["type"] == "image":
                        # Qwen3-VL reuses the Qwen2-VL image resize utility;
                        # transformers 5.x does not ship image_processing_qwen3_vl.
                        # Import only for images so text/tool requests stay independent.
                        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
                        image = part["image"]
                        height, width = smart_resize(
                            image.height, image.width,
                            factor=self.image_processor.patch_size * self.image_processor.merge_size,
                            min_pixels=self.image_size["shortest_edge"],
                            max_pixels=self.image_size["longest_edge"])
                        part = {"type": "image", "image": image.resize((width, height))}
                    parts.append(part)
                message["content"] = parts
            if message.get("tool_calls"):
                message["tool_calls"] = [ToolCall.model_validate({
                    **call, "function": {**call["function"],
                        "arguments": json.dumps(call["function"]["arguments"])}})
                    for call in message["tool_calls"]]
            native.append(message)

        # Merge leading system/developer instructions into the tool declaration.
        system = []
        while native and native[0]["role"] == "system":
            content = native.pop(0)["content"]
            if isinstance(content, list):
                if any(part["type"] != "text" for part in content):
                    raise ValueError("System instructions must contain text only")
                content = "".join(part["text"] for part in content)
            system.append(content)
        # Inspect/OpenAI describes functions as {"type": "function", "function": {...}};
        # Tinker's renderer expects flat ToolSpec objects with name/description/parameters.
        tool_specs = [
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            }
            for tool in tools or []
            for function in [tool.get("function", tool)]
        ]
        prefix = self.renderer.create_conversation_prefix_with_tools(
            tool_specs, system_prompt="\n\n".join(system))
        return self.renderer.build_generation_prompt(prefix + native)

    def generate(self, messages, tools, budget, context_window):
        import tinker

        prompt = self.render(messages, tools)
        if prompt.length + budget > context_window:
            raise ValueError("Context exceeds model capacity")
        result = self.client.sample(prompt=prompt, num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=budget, temperature=0, stop=self.renderer.get_stop_sequences())).result()
        tokens = result.sequences[0].tokens
        # Keep the bridge's schema-aware parser: code arguments must stay strings.
        return self.tokenizer.decode(tokens, skip_special_tokens=False), prompt.length, len(tokens)
