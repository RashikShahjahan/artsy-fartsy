import base64
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from inference.server import CONTEXT_WINDOW, MAX_TOKENS, MODEL_ID, app, complete, health, image_content, models, parse_tools, template_messages


TOOLS = [{"type": "function", "function": {"name": "write_code", "parameters": {
    "type": "object", "properties": {"source": {"type": "string"}}}}}]


def tool_call(name, parameters=""):
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def parameter(name, value):
    return f"<parameter={name}>\n{value}\n</parameter>\n"


def inline_image(size=(32, 32), mode="RGB", patterned=False):
    from PIL import Image
    data = io.BytesIO()
    image = Image.new(mode, size, "blue")
    if patterned:
        from PIL import ImageDraw
        ImageDraw.Draw(image).rectangle((size[0] // 3, size[1] // 3, size[0] - 1, size[1] - 1), fill="red")
    image.save(data, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()}}


class ToolProtocolTest(unittest.TestCase):
    def test_extended_context_selector_is_preserved_when_loading_saved_weights(self):
        from drawing.config import IMAGE_SIZE, MODEL_REVISION, TINKER_SAMPLING_MODEL_ID
        from inference.tinker_backend import TinkerBackend

        with patch("tinker.ServiceClient") as service, \
             patch("transformers.AutoTokenizer.from_pretrained"), \
             patch("transformers.AutoImageProcessor.from_pretrained"), \
             patch("tinker_cookbook.renderers.get_renderer"):
            backend = TinkerBackend(MODEL_ID, MODEL_REVISION, IMAGE_SIZE,
                                   sampling_model_id=TINKER_SAMPLING_MODEL_ID)
            service.return_value.create_sampling_client.assert_called_once_with(
                base_model="Qwen/Qwen3.5-4B:peft:262144")
            backend.with_model_path("tinker://saved/checkpoint")
            service.return_value.create_sampling_client.assert_called_with(
                base_model="Qwen/Qwen3.5-4B:peft:262144", model_path="tinker://saved/checkpoint")
            TinkerBackend(MODEL_ID, MODEL_REVISION, IMAGE_SIZE, "tinker://saved/checkpoint",
                          sampling_model_id=TINKER_SAMPLING_MODEL_ID)
            service.return_value.create_sampling_client.assert_called_with(
                base_model="Qwen/Qwen3.5-4B:peft:262144", model_path="tinker://saved/checkpoint")

    def test_output_budget_uses_remaining_context_and_rejects_a_full_prompt(self):
        from inference.tinker_backend import TinkerBackend

        backend = TinkerBackend.__new__(TinkerBackend)
        backend.render = Mock(return_value=SimpleNamespace(length=24))
        backend.renderer = Mock()
        backend.renderer.get_stop_sequences.return_value = ["stop"]
        backend.client = Mock()
        backend.client.sample.return_value.result.return_value = SimpleNamespace(
            sequences=[SimpleNamespace(tokens=list(range(8)))])
        backend.tokenizer = Mock()
        backend.tokenizer.decode.return_value = "done"
        self.assertEqual(backend.generate([], None, 16, 32), ("done", 24, 8))
        self.assertEqual(backend.client.sample.call_args.kwargs["sampling_params"].max_tokens, 8)
        backend.client.sample.reset_mock()
        backend.render.return_value = SimpleNamespace(length=32)
        with self.assertRaisesRegex(ValueError, "Context exceeds"):
            backend.generate([], None, 16, 32)
        backend.client.sample.assert_not_called()

    def test_native_tinker_renderer_receives_flat_tool_specs_and_preview_pixels(self):
        from inference.tinker_backend import TinkerBackend
        from tinker_cookbook.renderers import ToolCall

        backend = TinkerBackend.__new__(TinkerBackend)
        backend.image_processor = SimpleNamespace(patch_size=14, merge_size=2)
        backend.image_size = {"shortest_edge": 28 * 28, "longest_edge": 56 * 56}
        backend.renderer = Mock()
        backend.renderer.create_conversation_prefix_with_tools.return_value = [
            {"role": "system", "content": "tools available"}]
        messages = template_messages([
            {"role": "developer", "content": "Use bash to inspect drawings."},
            {"role": "user", "content": [inline_image(), {"type": "text", "text": "Inspect this"}]},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "bash", "arguments": '{"command":"pwd"}'}}]},
        ])
        tools = [{"type": "function", "function": {
            "name": "bash", "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]

        backend.render(messages, tools)

        specs = backend.renderer.create_conversation_prefix_with_tools.call_args.args[0]
        self.assertEqual(specs, [{"name": "bash", "description": "Run a command",
                                  "parameters": tools[0]["function"]["parameters"]}])
        self.assertEqual(backend.renderer.create_conversation_prefix_with_tools.call_args.kwargs,
                         {"system_prompt": "Use bash to inspect drawings."})
        native = backend.renderer.build_generation_prompt.call_args.args[0]
        self.assertEqual(native[1]["content"][0]["image"].size, (28, 28))
        self.assertEqual(native[1]["content"][0]["image"].getpixel((0, 0)), (0, 0, 255))
        self.assertIsInstance(native[2]["tool_calls"][0], ToolCall)
        self.assertEqual(messages[1]["content"][0]["image"].size, (32, 32))

    def test_developer_instructions_are_compatible_with_qwen_system_role(self):
        messages = [{"role": "developer", "content": "Use the bash tool to draw."},
                    {"role": "user", "content": "Draw a circle."}]
        native = template_messages(messages)
        self.assertEqual(native, [
            {"role": "system", "content": "Use the bash tool to draw."}, messages[1]])
        self.assertEqual(messages[0]["role"], "developer")

    def test_multiline_source_and_transcript_replay(self):
        source = 'from artcanvas import ArtCanvas\nif 1 < 2:\n    print("hi & bye")\n'
        text, calls = parse_tools(tool_call("write_code", parameter("source", source)) + "<|im_end|>", TOOLS)
        self.assertEqual(text, "")
        self.assertEqual(calls[0]["function"]["name"], "write_code")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["source"], source)
        messages = [{"role": "assistant", "content": None, "tool_calls": calls},
                    {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"written": true}'}]
        native = template_messages(messages)
        self.assertEqual(native[0]["content"], "")
        self.assertEqual(native[0]["tool_calls"][0]["function"]["arguments"], {"source": source})
        self.assertEqual(native[1], messages[1])
        # Normalizing for the template must not mutate the persisted wire transcript.
        self.assertIsInstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)

    def test_source_is_always_a_literal_string(self):
        for source in ['123', 'true', '{"code": 1}', "__import__('os').getcwd()"]:
            with self.subTest(source=source):
                _, calls = parse_tools(tool_call("write_code", parameter("source", source)), TOOLS)
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"source": source})

    def test_prose_multiple_calls_and_empty_arguments(self):
        text, calls = parse_tools("Running now.\n" + tool_call("run_code") + "\n" + tool_call("submit"))
        self.assertEqual(text, "Running now.")
        self.assertEqual([call["function"]["name"] for call in calls], ["run_code", "submit"])
        self.assertTrue(all(json.loads(call["function"]["arguments"]) == {} for call in calls))

    def test_truncated_calls_are_rejected(self):
        for raw in ['<tool_call>\n<function=submit>\n</function>',
                    tool_call("write_code", '<parameter=source>\nincomplete'),
                    tool_call("run_code") + '<tool_call>\n<function=submit>']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_tools(raw, TOOLS)

    def test_duplicate_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_tools(tool_call("write_code", parameter("source", "one") + parameter("source", "two")), TOOLS)

    def test_reasoning_cannot_invoke_tools(self):
        self.assertEqual(parse_tools("<think>" + tool_call("submit") + "</think>\nHello<|im_end|>"), ("Hello", []))
        raw = "<think>" + tool_call("submit")
        self.assertEqual(parse_tools(raw), (raw, []))

    def test_schema_decodes_nonstring_arguments(self):
        tools = [{"function": {"name": "test", "parameters": {"properties": {"count": {"type": "integer"}}}}}]
        _, calls = parse_tools(tool_call("test", parameter("count", "3")), tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"count": 3})

    def test_image_blocks_reach_the_template_with_pixels_and_text(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Inspect this preview"}, inline_image()]}]
        native = template_messages(messages)
        self.assertEqual(native[0]["content"][0], messages[0]["content"][0])
        picture = native[0]["content"][1]
        self.assertEqual(picture["type"], "image")
        self.assertEqual(picture["image"].size, (32, 32))
        self.assertEqual(picture["image"].getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(messages[0]["content"][1]["type"], "image_url")

    def test_invalid_images_are_rejected_without_fetching_paths_or_urls(self):
        for url in ["https://example.com/image.png", "/tmp/image.png", "data:image/png;base64,???",
                    "data:image/png;base64," + base64.b64encode(b"not an image").decode()]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                image_content({"image_url": {"url": url}})


class CompletionTest(unittest.TestCase):
    def setUp(self):
        self.raw = tool_call("submit") + "<|im_end|>"
        self.state_patch = patch.object(app, "state")
        state = self.state_patch.start()
        self.addCleanup(self.state_patch.stop)
        from inference.model_catalog import ModelCatalogCache
        from inference.tinker_backend import TinkerBackend
        state.model_catalog_cache = ModelCatalogCache()
        state.backend = "tinker"
        preprocessor = TinkerBackend.__new__(TinkerBackend)
        preprocessor.image_processor = SimpleNamespace(patch_size=14, merge_size=2)
        preprocessor.image_size = {"shortest_edge": 28 * 28, "longest_edge": 56 * 56}
        state.tinker.prepare_image.side_effect = preprocessor.prepare_image
        state.tinker.generate.return_value = (self.raw, 3, 2)
        self.body = {"model": MODEL_ID, "messages": [{"role": "user", "content": "Draw a circle"}], "tools": TOOLS}

    def test_completion_uses_tinker_and_openai_tool_response(self):
        response = complete({**self.body, "max_tokens": MAX_TOKENS + 1})
        messages, tools, budget, context = app.state.tinker.generate.call_args.args
        self.assertEqual(messages, self.body["messages"])
        self.assertEqual((tools, budget, context), (TOOLS, MAX_TOKENS, CONTEXT_WINDOW))
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"], {"name": "submit", "arguments": "{}"})
        self.assertEqual(response["usage"], {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        self.assertEqual(models()["data"][0]["id"], "Qwen/Qwen3.5-4B")

    def test_completion_honors_modern_and_legacy_token_limits(self):
        for limits, expected in (({"max_tokens": 4096}, 4096),
                                 ({"max_completion_tokens": 4096}, 4096),
                                 ({"max_tokens": 8192, "max_completion_tokens": 2048}, 2048)):
            with self.subTest(limits=limits):
                complete({**self.body, **limits})
                self.assertEqual(app.state.tinker.generate.call_args.args[2], expected)

    def test_temperature_reaches_sampler_and_invalid_values_fail_before_sampling(self):
        complete({**self.body, "temperature": 0.8})
        self.assertEqual(app.state.tinker.generate.call_args.kwargs, {"temperature": 0.8})
        complete(self.body)
        self.assertEqual(app.state.tinker.generate.call_args.kwargs, {"temperature": 0.0})
        for value in (-1, 3, True, "0.8", None, float("nan"), float("inf")):
            app.state.tinker.generate.reset_mock()
            with self.subTest(value=value), self.assertRaises(HTTPException):
                complete({**self.body, "temperature": value})
            app.state.tinker.generate.assert_not_called()

    def test_completion_health_and_models_reuse_catalog(self):
        entries = [{"id": MODEL_ID, "label": "base", "model_path": None}]
        with patch("inference.server.model_catalog", return_value=entries) as catalog:
            complete(self.body)
            listed = models()
            listed["data"][0]["label"] = "mutated"
            self.assertEqual(health()["models"][0]["label"], "base")
            self.assertEqual(models()["data"][0]["label"], "base")
        catalog.assert_called_once()

    def test_finetune_selection_routes_to_its_sampler_and_reuses_it(self):
        entry = {"id": "artsy/trained", "label": "trained", "model_path": "tinker://trained/final"}
        app.state.samplers = {}
        sampler = app.state.tinker.with_model_path.return_value
        sampler.generate.return_value = (self.raw, 10, 4)
        with patch("inference.server.model_catalog", return_value=[entry]):
            for _ in range(2):
                response = complete({**self.body, "model": entry['id']})
                self.assertEqual(response['model'], entry['id'])
                self.assertEqual(models()['data'][0]['id'], entry['id'])
        app.state.tinker.with_model_path.assert_called_once_with(entry['model_path'])
        self.assertEqual(sampler.generate.call_count, 2)
        app.state.tinker.generate.assert_not_called()

    def test_malformed_call_is_returned_as_text(self):
        self.raw = tool_call("run_code") + '<tool_call>\n<function=submit>'
        app.state.tinker.generate.return_value = (self.raw, 3, 2)
        choice = complete(self.body)["choices"][0]
        self.assertEqual(choice["message"]["content"], self.raw)
        self.assertNotIn("tool_calls", choice["message"])

    def test_tinker_preserves_images_tool_calls_and_usage(self):
        app.state.tinker.generate.return_value = (tool_call("submit"), 123, 12)
        response = complete({**self.body, "messages": [
            {"role": "developer", "content": "Draw with tools."},
            {"role": "user", "content": [inline_image(), {"type": "text", "text": "Fix it"}]}]})
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "submit")
        self.assertEqual(response["usage"],
                         {"prompt_tokens": 123, "completion_tokens": 12, "total_tokens": 135})
        messages, tools, budget, context = app.state.tinker.generate.call_args.args
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["content"][0]["image"].size, (28, 28))
        self.assertEqual((tools, budget, context), (TOOLS, MAX_TOKENS, CONTEXT_WINDOW))

    def test_streaming_is_rejected_before_catalog_or_sampler_work(self):
        with patch("inference.server.model_catalog") as catalog:
            with self.assertRaises(HTTPException) as error:
                complete({**self.body, "stream": True, "model": "artsy/new"})
        self.assertEqual(error.exception.status_code, 400)
        self.assertIn("stream=false", error.exception.detail)
        catalog.assert_not_called()
        app.state.model_catalog_cache.get = Mock(side_effect=AssertionError("catalog accessed"))
        with self.assertRaises(HTTPException):
            complete({**self.body, "stream": True})
        app.state.tinker.generate.assert_not_called()
        app.state.tinker.with_model_path.assert_not_called()

    def test_large_image_is_bounded_before_generation_with_matching_pixels(self):
        from PIL import Image
        from inference.tinker_backend import TinkerBackend
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        for size, factor in [((4096, 2048), 28), ((33, 29), 16), ((8, 8), 28)]:
            with self.subTest(size=size, factor=factor):
                backend = TinkerBackend.__new__(TinkerBackend)
                backend.image_processor = SimpleNamespace(patch_size=factor // 2, merge_size=2)
                backend.image_size = {"shortest_edge": factor ** 2, "longest_edge": (factor * 2) ** 2}
                app.state.tinker.prepare_image.side_effect = backend.prepare_image
                part = inline_image(size, "RGBA", patterned=True)
                messages = [{"role": "user", "content": [part]}]
                complete({**self.body, "messages": messages})
                actual = app.state.tinker.generate.call_args.args[0][0]["content"][0]["image"]
                raw = image_content(part)["image"]
                height, width = smart_resize(size[1], size[0], factor=factor,
                    min_pixels=factor ** 2, max_pixels=(factor * 2) ** 2)
                expected = raw.resize((width, height))
                self.assertEqual(actual.mode, "RGB")
                self.assertEqual(actual.size, expected.size)
                self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertIs(backend.prepare_image(actual), actual)
                self.assertEqual(messages[0]["content"][0], part)

    def test_wrong_model_and_excess_context_are_rejected(self):
        with self.assertRaises(HTTPException) as error:
            complete({**self.body, "model": "LiquidAI/LFM2.5-230M"})
        self.assertEqual(error.exception.status_code, 400)
        app.state.tinker.generate.side_effect = ValueError("Context exceeds model capacity")
        with self.assertRaises(HTTPException) as error:
            complete(self.body)
        self.assertEqual(error.exception.status_code, 400)

    def test_context_clipped_completion_reports_length(self):
        app.state.tinker.generate.return_value = ("partial answer", CONTEXT_WINDOW - 5, 5)
        result = complete({**self.body, "max_tokens": 8192})
        self.assertEqual(result["choices"][0]["finish_reason"], "length")

    def test_native_context_accepts_history_above_the_old_bridge_limit(self):
        from inference.tinker_backend import TinkerBackend

        backend = TinkerBackend.__new__(TinkerBackend)
        prompt_length = 262144 - 8192
        backend.render = Mock(return_value=SimpleNamespace(length=prompt_length))
        backend.renderer = Mock()
        backend.renderer.get_stop_sequences.return_value = ["stop"]
        backend.client = Mock()
        backend.client.sample.return_value.result.return_value = SimpleNamespace(
            sequences=[SimpleNamespace(tokens=[1])])
        backend.tokenizer = Mock()
        backend.tokenizer.decode.return_value = "done"
        app.state.tinker.generate.side_effect = backend.generate

        result = complete({**self.body, "max_tokens": 8192})

        self.assertEqual(result["usage"]["prompt_tokens"], prompt_length)
        self.assertEqual(backend.client.sample.call_args.kwargs["sampling_params"].max_tokens, 8192)
        self.assertEqual(health()["context_window"], 262144)

    def test_image_input_is_passed_to_model_generation(self):
        response = complete({**self.body, "messages": [{"role": "user", "content": [inline_image(), {"type": "text", "text": "Describe"}]}]})
        messages = app.state.tinker.generate.call_args.args[0]
        self.assertEqual(messages[0]["content"][0]["image"].getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(response["model"], MODEL_ID)

    def test_bad_image_is_an_http_400(self):
        with self.assertRaises(HTTPException) as error:
            complete({**self.body, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "bad"}}]}]})
        self.assertEqual(error.exception.status_code, 400)

    def test_invalid_chat_template_is_an_http_400(self):
        from jinja2 import TemplateError
        app.state.tinker.generate.side_effect = TemplateError("Unexpected message role.")
        with self.assertRaises(HTTPException) as error:
            complete(self.body)
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(error.exception.detail, "Unexpected message role.")


if __name__ == "__main__":
    unittest.main()
