"""Native hosted sampling with model-specific tokenization, vision and tools."""
import json

RENDERER = "qwen3_5_disable_thinking"


class TinkerBackend:
    def __init__(self, model_id, revision, image_size, model_path=None, *, sampling_model_id=None,
                 renderer_name=RENDERER):
        import tinker
        from transformers import AutoImageProcessor, AutoTokenizer
        from tinker_cookbook import renderers

        self.model_id = model_id
        self.native_tools = renderer_name.startswith('kimi_')
        self.sampling_model_id = sampling_model_id or model_id
        self.service = tinker.ServiceClient()
        self.client = self.service.create_sampling_client(
            base_model=self.sampling_model_id,
            **({"model_path": model_path} if model_path else {}))
        if self.native_tools:
            # Use the pinned vendor tokenizer rather than HF's generic backend;
            # Kimi's tool tokens and TypeScript declarations require this class.
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            tokenizer_class = get_class_from_dynamic_module(
                'tokenization_kimi.TikTokenTokenizer', model_id, revision=revision)
            self.tokenizer = tokenizer_class.from_pretrained(model_id, revision=revision)
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_id, revision=revision, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            self.image_processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.image_size = image_size
        self.renderer = renderers.get_renderer(
            renderer_name, self.tokenizer, image_processor=self.image_processor)

    def with_model_path(self, model_path):
        """Reuse local processors while selecting a different hosted sampler."""
        from copy import copy
        backend = copy(self)
        backend.client = self.service.create_sampling_client(
            base_model=self.sampling_model_id,
            **({"model_path": model_path} if model_path else {}))
        return backend

    def prepare_image(self, image):
        """Apply the training-compatible Qwen resize before retaining request pixels."""
        if getattr(self, 'native_tools', False):
            image = image.convert('RGB')
            edge = int(self.image_size['longest_edge'] ** 0.5)
            image.thumbnail((edge, edge))
            return image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        height, width = smart_resize(
            image.height, image.width,
            factor=self.image_processor.patch_size * self.image_processor.merge_size,
            min_pixels=self.image_size["shortest_edge"],
            max_pixels=self.image_size["longest_edge"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image if image.size == (width, height) else image.resize((width, height))

    def render(self, messages, tools):
        from tinker_cookbook.renderers import ToolCall

        native = []
        tool_names = {}
        for original in messages:
            message = dict(original)
            if message.get('role') == 'tool' and not message.get('name'):
                name = tool_names.get(message.get('tool_call_id'))
                if name:
                    message['name'] = name
            if isinstance(message["content"], list):
                parts = []
                for part in message["content"]:
                    if part["type"] == "image":
                        part = {"type": "image", "image": self.prepare_image(part["image"])}
                    parts.append(part)
                message["content"] = parts
            if message.get("tool_calls"):
                tool_names.update({call['id']: call['function']['name'] for call in message['tool_calls']})
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

    def generate(self, messages, tools, budget, context_window, *, temperature=0.0):
        import tinker

        prompt = self.render(messages, tools)
        remaining = context_window - prompt.length
        if remaining < 1:
            raise ValueError("Context exceeds model capacity")
        budget = min(budget, remaining)
        result = self.client.sample(prompt=prompt, num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=budget, temperature=temperature, stop=self.renderer.get_stop_sequences())).result()
        tokens = result.sequences[0].tokens
        if getattr(self, 'native_tools', False):
            message, termination = self.renderer.parse_response(tokens)
            if termination.is_clean and not message.get('unparsed_tool_calls'):
                return message, prompt.length, len(tokens)
        # Keep the bridge's schema-aware parser: code arguments must stay strings.
        return self.tokenizer.decode(tokens, skip_special_tokens=False), prompt.length, len(tokens)
