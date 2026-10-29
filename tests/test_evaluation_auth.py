"""The host agent configures endpoint credentials without changing judge keys."""

import os
import unittest
from unittest.mock import patch

from evaluation.harness import TINKER_MODEL_URL, OPENROUTER_MODEL_URL, drawing_task, model_from_env


class HostModelAuthTests(unittest.TestCase):
    def test_openrouter_luna_uses_provider_key_and_preserves_wire_model(self):
        with patch.dict(os.environ, {
            'ARTSY_MODEL_URL': OPENROUTER_MODEL_URL,
            'OPENROUTER_API_KEY': 'test-router-key',
            'OPENAI_API_KEY': 'independent-judge-key',
        }, clear=True):
            before = dict(os.environ)
            model = model_from_env('openai/gpt-6-luna')
            self.assertEqual(model.api.api_key, 'test-router-key')
            self.assertEqual(model.api.base_url, OPENROUTER_MODEL_URL)
            self.assertEqual(model.api.service_model_name(), 'openai/gpt-6-luna')
            self.assertFalse(model.model_args['responses_api'])
            self.assertEqual(dict(os.environ), before)
        with patch.dict(os.environ, {'ARTSY_MODEL_URL': OPENROUTER_MODEL_URL}, clear=True):
            with self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
                model_from_env('openai/gpt-6-luna')

    def test_local_model_initializes_without_openai_credentials(self):
        with patch.dict(os.environ, {
            "ARTSY_MODEL_API_KEY": "",
            "ARTSY_MODEL_URL": "http://127.0.0.1:8080/v1",
        }, clear=True):
            model = model_from_env()
            self.assertEqual(model.api.api_key, "local-no-auth")
            self.assertEqual(model.api.base_url, "http://127.0.0.1:8080/v1")
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_long_retry_timeout_reaches_task_and_http_client(self):
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": "http://127.0.0.1:8080/v1"}, clear=True):
            model = model_from_env(timeout=600)
            self.assertEqual(model.config.timeout, 600)
            self.assertEqual(model.api.client_timeout, 600)
            task = drawing_task([{"id": "retry", "prompt": "Draw a circle", "seed": 1}],
                                "/tmp/artsy-test-retry", time_limit=600)
            self.assertEqual(task.time_limit, 600)
            self.assertEqual(task.config.timeout, 600)
            self.assertEqual(task.metadata["environment_limits"]["episodeMs"], 600000)
            for invalid in (0, -1, True, 1.5):
                with self.subTest(value=invalid), self.assertRaises(ValueError):
                    drawing_task([], "/tmp/artsy-test-retry", time_limit=invalid)

    def test_local_alias_uses_generic_provider_and_preserves_wire_model(self):
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": "http://127.0.0.1:8080/v1"}, clear=True):
            model = model_from_env("artsy/sft-v1")
            self.assertEqual(str(model), "openai-api/artsy/artsy/sft-v1")
            self.assertEqual(model.api.service_model_name(), "artsy/sft-v1")
            from inspect_ai.model import GenerateConfig
            params = model.api.completion_params(GenerateConfig(temperature=0.8), tools=True)
            self.assertEqual(params["temperature"], 0.8)
            self.assertEqual(params["model"], "artsy/sft-v1")

    def test_remote_endpoint_key_leaves_judge_credentials_unchanged(self):
        with patch.dict(os.environ, {
            "ARTSY_MODEL_API_KEY": "drawing-endpoint-key",
            "ARTSY_MODEL_URL": "https://drawing.example/v1",
            "OPENAI_API_KEY": "independent-judge-key",
        }, clear=True):
            before = dict(os.environ)
            model = model_from_env()
            self.assertEqual(model.api.api_key, "drawing-endpoint-key")
            self.assertEqual(model.api.base_url, "https://drawing.example/v1")
            self.assertEqual(dict(os.environ), before)
            with patch("evaluation.harness.get_model", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    model_from_env()
            self.assertEqual(dict(os.environ), before)

    def test_direct_tinker_uses_tinker_key_and_disables_thinking(self):
        with patch.dict(os.environ, {
            "ARTSY_MODEL_URL": TINKER_MODEL_URL,
            "TINKER_API_KEY": "test-tinker-key",
            "OPENAI_API_KEY": "independent-judge-key",
        }, clear=True):
            model = model_from_env()
            self.assertEqual(model.api.api_key, "test-tinker-key")
            self.assertEqual(model.api.base_url, TINKER_MODEL_URL)
            self.assertFalse(model.model_args["responses_api"])
            self.assertEqual(os.environ["OPENAI_API_KEY"], "independent-judge-key")
            task = drawing_task([{"id": "smoke", "prompt": "Draw a circle", "seed": 0}],
                                "/tmp/artsy-test-artifacts")
            self.assertEqual(task.config.reasoning_effort, "none")

    def test_direct_tinker_requires_credentials_and_supports_explicit_override(self):
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": TINKER_MODEL_URL}, clear=True):
            with self.assertRaisesRegex(ValueError, "TINKER_API_KEY"):
                model_from_env()
        with patch.dict(os.environ, {
            "ARTSY_MODEL_URL": TINKER_MODEL_URL + "/",
            "TINKER_API_KEY": "test-tinker-key",
            "ARTSY_MODEL_API_KEY": "override-key",
        }, clear=True):
            self.assertEqual(model_from_env().api.api_key, "override-key")


if __name__ == "__main__":
    unittest.main()
