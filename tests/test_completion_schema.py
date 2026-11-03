"""Malformed requests fail before catalog loading, processing, or sampling."""
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from inference.server import complete


class CompletionSchemaTests(unittest.TestCase):
    def test_invalid_requests_never_reach_backend(self):
        base = {"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "draw"}]}
        invalid = [None, {}, {**base, "messages": []}, {**base, "messages": [None]},
                   {**base, "messages": [{"role": "invalid", "content": "x"}]},
                   {**base, "messages": [{"role": "user", "content": [None]}]},
                   {**base, "messages": [{"role": "assistant", "tool_calls": [{"id": "a", "function": {
                       "name": "bash", "arguments": "[]"}}]}]},
                   {**base, "tools": [{"type": "function", "function": {"name": 4}}]}]
        invalid.extend({**base, "max_tokens": value} for value in ("invalid", True, 0, -1, 1.5))
        with patch("inference.server.cached_model_catalog") as catalog:
            for body in invalid:
                with self.subTest(body=body), self.assertRaises(HTTPException) as error:
                    complete(body)
                self.assertEqual(error.exception.status_code, 400)
            catalog.assert_not_called()

    def test_http_validation_without_starting_backend(self):
        from fastapi.testclient import TestClient
        from inference.server import app
        with patch("inference.server.cached_model_catalog") as catalog:
            client = TestClient(app)
            response = client.post("/v1/chat/completions", json={"model": "x", "messages": [None]})
            self.assertEqual(response.status_code, 422)
            catalog.assert_not_called()
