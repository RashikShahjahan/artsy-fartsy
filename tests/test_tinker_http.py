"""Tinker-compatible transport and model capability checks."""
import json
import os
import unittest
from unittest.mock import patch
from contextlib import asynccontextmanager


@asynccontextmanager
async def endpoint(client, response):
    requests = []
    def respond(request):
        requests.append(request)
        return response
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client.http_client = http
        yield requests


import httpx

from drawing.dataset_client import DEFAULT_DATA_MODEL, TINKER_MODEL_URL, chat_client


class TinkerHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_drawing_calls_accept_null_content_and_send_tools(self):
        from drawing.agent_contract import DRAWING_TOOLS
        call = {'id': 'draw_1', 'type': 'function', 'function': {
            'name': 'update_drawing', 'arguments': '{"source":"draw()"}'}}
        response = httpx.Response(200, json={'choices': [{'message': {
            'content': None, 'tool_calls': [call], 'reasoning': 'Plan the drawing'},
            'finish_reason': 'tool_calls'}]})
        client = chat_client({'model': 'vision-teacher', 'baseUrl': 'http://localhost/v1'})
        async with endpoint(client, response) as requests:
            result = await client.complete([{'role': 'user', 'content': 'Draw a circle'}],
                                           tools=DRAWING_TOOLS)
        body = json.loads(requests[0].content)
        self.assertEqual(body['tools'], DRAWING_TOOLS)
        self.assertEqual(body['tool_choice'], 'required')
        self.assertFalse(body['parallel_tool_calls'])
        self.assertEqual(result['text'], '')
        self.assertEqual(result['tool_calls'], [call])
        self.assertEqual(result['reasoning'], 'Plan the drawing')

    async def test_malformed_tool_call_envelope_rejected(self):
        response = httpx.Response(200, json={'choices': [{'message': {
            'content': None, 'tool_calls': [{'id': 'bad', 'function': {'arguments': None}}]}}]})
        client = chat_client({'model': 'teacher', 'baseUrl': 'http://localhost/v1'})
        async with endpoint(client, response):
            with self.assertRaisesRegex(RuntimeError, 'invalid tool calls'):
                await client.complete([])

    async def test_inkling_preserves_image_parts(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Grade this drawing"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,pixels"}},
        ]}]
        response = httpx.Response(200, json={"choices": [{
            "message": {"content": '{"score":1,"reason":"Matches"}'}, "finish_reason": "stop"}]})
        client = chat_client({"model": DEFAULT_DATA_MODEL, "baseUrl": TINKER_MODEL_URL, "apiKey": "test"})
        async with endpoint(client, response) as requests:
            await client.complete(messages, 2048)
        self.assertEqual(json.loads(requests[0].content)["messages"], messages)

    async def test_tinker_auth_reasoning_and_final_answer(self):
        response = httpx.Response(200, json={
            "choices": [{"message": {"content": '{"ok":true}',
                                     "reasoning_content": "private reasoning"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}})
        with patch.dict(os.environ, {"TINKER_API_KEY": "tinker-test"}):
            client = chat_client({"model": DEFAULT_DATA_MODEL, "baseUrl": TINKER_MODEL_URL})
        async with endpoint(client, response) as requests:
            result = await client.complete([{"role": "user", "content": "Check code"}], 8192)
        self.assertEqual(result["text"], '{"ok":true}')
        self.assertEqual(result["reasoning"], "private reasoning")
        body = json.loads(requests[0].content)
        self.assertTrue(body["separate_reasoning"])
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer tinker-test")
        self.assertNotIn("tinker-test", repr(client))

    async def test_truncation_and_provider_errors_are_sanitized(self):
        with patch.dict(os.environ, {"TINKER_API_KEY": "tinker-test"}):
            client = chat_client({"model": DEFAULT_DATA_MODEL, "baseUrl": TINKER_MODEL_URL})
        for response, error in (
            (httpx.Response(200, json={"choices": [{"message": {"content": "partial"},
                                                   "finish_reason": "length"}]}), "truncated"),
            (httpx.Response(401, text="echoed-secret"), "HTTP 401"),
        ):
            async with endpoint(client, response):
                with self.assertRaisesRegex(RuntimeError, error) as raised:
                    await client.complete([], 10)
            self.assertNotIn("echoed-secret", str(raised.exception))

    async def test_unsupported_images_fail_before_request(self):
        for model in ("zai-org/GLM-5.3-Flash:peft:262144", "deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144"):
            client = chat_client({"model": model, "baseUrl": TINKER_MODEL_URL, "apiKey": "test"})
            with patch("httpx.AsyncClient") as post, self.assertRaisesRegex(ValueError, "does not support image"):
                await client.complete([{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}]}], 100)
            post.assert_not_called()

    def test_tinker_requires_key_without_changing_other_endpoints(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TINKER_API_KEY"):
                chat_client({"model": DEFAULT_DATA_MODEL, "baseUrl": TINKER_MODEL_URL})
            local = chat_client({"model": "local", "baseUrl": "http://localhost/v1"})
            self.assertIsNone(local.api_key)

    def test_http_judges_keep_the_configured_rl_token_budget(self):
        client = chat_client({"model": DEFAULT_DATA_MODEL, "baseUrl": TINKER_MODEL_URL,
                              "apiKey": "test", "maxTokens": 16384})
        self.assertEqual(client.max_tokens, 16384)


if __name__ == "__main__":
    unittest.main()
