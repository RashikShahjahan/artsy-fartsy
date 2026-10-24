"""Verify the direct-Tinker preflight checks the wire features Artsy needs."""

import base64
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.check_tinker import main
from evaluation.harness import TINKER_MODEL_URL


class TinkerPreflightTests(unittest.TestCase):
    def test_requests_function_call_and_inline_png(self):
        tool = SimpleNamespace(function=SimpleNamespace(
            name="bash", arguments='{"command":"echo READY"}'))
        tool_reply = SimpleNamespace(choices=[SimpleNamespace(message=
            SimpleNamespace(tool_calls=[tool]))])
        image_reply = SimpleNamespace(choices=[SimpleNamespace(message=
            SimpleNamespace(content="red"))])
        with patch.dict(os.environ, {
            "ARTSY_MODEL_URL": TINKER_MODEL_URL,
            "TINKER_API_KEY": "test-only-key",
            "ARTSY_MODEL_API_KEY": "",
        }, clear=True), patch("evaluation.check_tinker.load_dotenv"), \
             patch("evaluation.check_tinker.OpenAI") as client_class, \
             patch("sys.argv", ["check_tinker"]):
            client = client_class.return_value.__enter__.return_value
            client.chat.completions.create.side_effect = [tool_reply, image_reply]
            self.assertEqual(main(), 0)
            self.assertEqual(client_class.call_args.kwargs["api_key"], "test-only-key")
            tool_request, vision_request = [call.kwargs for call in
                                            client.chat.completions.create.call_args_list]
        self.assertEqual(tool_request["tool_choice"]["function"]["name"], "bash")
        self.assertEqual(tool_request["reasoning_effort"], "none")
        self.assertEqual(vision_request["reasoning_effort"], "none")
        image_url = vision_request["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertTrue(base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
