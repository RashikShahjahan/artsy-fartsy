import asyncio
import base64
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from inference.server import CONTEXT_WINDOW, MAX_TOKENS, MODEL_ID, app, complete, image_content, models, parse_tools, template_messages


TOOLS = [{"type": "function", "function": {"name": "write_code", "parameters": {
    "type": "object", "properties": {"source": {"type": "string"}}}}}]


def tool_call(name, parameters=""):
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


def parameter(name, value):
    return f"<parameter={name}>\n{value}\n</parameter>\n"


def inline_image():
    from PIL import Image
    data = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(data, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()}}


class ToolProtocolTest(unittest.TestCase):
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
        state.backend = "tinker"
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
        self.assertEqual(messages[1]["content"][0]["image"].size, (32, 32))
        self.assertEqual((tools, budget, context), (TOOLS, MAX_TOKENS, CONTEXT_WINDOW))

    def test_streaming_tool_response_has_index_and_done(self):
        response = complete({**self.body, "stream": True})

        async def collect():
            return [chunk async for chunk in response.body_iterator]

        chunks = asyncio.run(collect())
        first = json.loads(chunks[0].removeprefix("data: "))
        self.assertEqual(first["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
        self.assertEqual(json.loads(chunks[1].removeprefix("data: "))["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

    def test_wrong_model_and_excess_context_are_rejected(self):
        with self.assertRaises(HTTPException) as error:
            complete({**self.body, "model": "LiquidAI/LFM2.5-230M"})
        self.assertEqual(error.exception.status_code, 400)
        app.state.tinker.generate.side_effect = ValueError("Context exceeds model capacity")
        with self.assertRaises(HTTPException) as error:
            complete(self.body)
        self.assertEqual(error.exception.status_code, 400)

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
