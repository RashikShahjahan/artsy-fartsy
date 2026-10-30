"""Verify the native-Tinker preflight checks the wire features Artsy needs."""

import base64
import httpx
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.check_tinker import main
from evaluation.harness import (BASE_MODEL, LOCAL_MODEL_URL, TINKER_MODEL_URL,
                                check_native_bridge)


class TinkerPreflightTests(unittest.TestCase):
    def test_larger_model_requires_matching_native_profile(self):
        from inference.model_catalog import LARGE_MODEL_ID, model_catalog
        entry = next(row for row in model_catalog() if row['id'] == LARGE_MODEL_ID)
        status = {'backend': 'tinker', 'model': BASE_MODEL, 'input': ['text', 'image'], 'models': [entry]}
        with patch.dict(os.environ, {'ARTSY_MODEL_URL': ''}), patch('evaluation.harness.httpx.get') as request:
            request.return_value.json.return_value = status
            check_native_bridge(LARGE_MODEL_ID)
            request.return_value.json.return_value = {**status, 'models': []}
            with self.assertRaisesRegex(RuntimeError, 'Restart `make serve`'):
                check_native_bridge(LARGE_MODEL_ID)
    def test_qwen36_requires_matching_native_profile(self):
        from inference.model_catalog import QWEN36_MODEL_ID, model_catalog
        entry = next(row for row in model_catalog() if row['id'] == QWEN36_MODEL_ID)
        status = {'backend': 'tinker', 'model': BASE_MODEL, 'input': ['text', 'image'], 'models': [entry]}
        with patch.dict(os.environ, {'ARTSY_MODEL_URL': ''}), patch('evaluation.harness.httpx.get') as request:
            request.return_value.json.return_value = status
            check_native_bridge(QWEN36_MODEL_ID)
            request.return_value.json.return_value = {**status, 'models': []}
            with self.assertRaisesRegex(RuntimeError, 'Restart `make serve`'):
                check_native_bridge(QWEN36_MODEL_ID)

    def test_default_endpoint_requires_running_native_image_bridge(self):
        status = {"backend": "tinker", "model": BASE_MODEL, "model_path": None,
                  "input": ["text", "image"]}
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": "", "ARTSY_TINKER_MODEL_PATH": ""}), \
             patch("evaluation.harness.httpx.get") as request:
            request.return_value.json.return_value = status
            check_native_bridge(BASE_MODEL)
            request.assert_called_once_with("http://127.0.0.1:8080/health", timeout=5)
            for changed in ({"backend": "torch"}, {"model": "other/model"},
                            {"input": ["text"]}, {"model_path": "tinker://other"}):
                request.return_value.json.return_value = {**status, **changed}
                with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, "native Tinker"):
                    check_native_bridge(BASE_MODEL)
            request.side_effect = httpx.ConnectError("offline")
            with self.assertRaisesRegex(RuntimeError, "Start it with `make serve`"):
                check_native_bridge(BASE_MODEL)

    def test_finetune_requires_matching_sampler_advertised_by_bridge(self):
        entry = {"id": "artsy/trained", "label": "trained", "model_path": "tinker://trained/final"}
        status = {"backend": "tinker", "model": BASE_MODEL, "input": ["text", "image"], "models": [entry]}
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": ""}), \
             patch("inference.model_catalog.model_catalog", return_value=[entry]), \
             patch("evaluation.harness.httpx.get") as request:
            request.return_value.json.return_value = status
            check_native_bridge(entry['id'])
            for advertised in ([], [{**entry, "model_path": "tinker://wrong/final"}]):
                request.return_value.json.return_value = {**status, "models": advertised}
                with self.assertRaisesRegex(RuntimeError, "Restart `make serve`"):
                    check_native_bridge(entry['id'])

    def test_custom_endpoint_skips_native_bridge_check(self):
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": "https://example.com/v1"}), \
             patch("evaluation.harness.httpx.get") as request:
            check_native_bridge()
            request.assert_not_called()

    def test_qwen_direct_beta_endpoint_is_rejected_before_evaluation(self):
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": TINKER_MODEL_URL}), \
             patch("evaluation.harness.httpx.get") as request, \
             self.assertRaisesRegex(RuntimeError, "Unset ARTSY_MODEL_URL"):
            check_native_bridge(BASE_MODEL)
        request.assert_not_called()

    def test_requests_function_call_and_inline_png(self):
        tool = SimpleNamespace(function=SimpleNamespace(
            name="bash", arguments='{"command":"echo READY"}'))
        tool_reply = SimpleNamespace(choices=[SimpleNamespace(message=
            SimpleNamespace(tool_calls=[tool]))])
        image_reply = SimpleNamespace(choices=[SimpleNamespace(message=
            SimpleNamespace(content="red"))])
        with patch.dict(os.environ, {
            "ARTSY_MODEL_URL": LOCAL_MODEL_URL,
            "ARTSY_MODEL_API_KEY": "",
        }, clear=True), patch("evaluation.check_tinker.load_dotenv"), \
             patch("evaluation.check_tinker.check_native_bridge") as health, \
             patch("evaluation.check_tinker.OpenAI") as client_class, \
             patch("sys.argv", ["check_tinker"]):
            client = client_class.return_value.__enter__.return_value
            client.chat.completions.create.side_effect = [tool_reply, image_reply]
            self.assertEqual(main(), 0)
            self.assertEqual(client_class.call_args.kwargs["api_key"], "local-no-auth")
            tool_request, vision_request = [call.kwargs for call in
                                            client.chat.completions.create.call_args_list]
        health.assert_called_once_with("Qwen/Qwen3.5-4B")
        self.assertEqual(tool_request["tool_choice"]["function"]["name"], "bash")
        self.assertNotIn("reasoning_effort", tool_request)
        self.assertNotIn("reasoning_effort", vision_request)
        image_url = vision_request["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertTrue(base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\x89PNG"))

    def test_direct_beta_endpoint_is_not_a_native_preflight(self):
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": TINKER_MODEL_URL}, clear=True), \
             patch("evaluation.check_tinker.load_dotenv"), \
             patch("sys.argv", ["check_tinker"]), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
