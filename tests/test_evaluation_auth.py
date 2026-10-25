"""Regression tests for Inspect SWE's OpenAI-compatible model bridge."""

import os
import unittest
from unittest.mock import patch

from inspect_swe._mini_swe_agent.mini_swe_agent import _model_without_responses_api

from evaluation.harness import (TINKER_MODEL_URL, drawing_task,
                                inspect_swe_model_auth, model_from_env)


class InspectSWEModelAuthTests(unittest.TestCase):
    def test_local_model_bridge_initializes_without_openai_credentials(self):
        with patch.dict(os.environ, {
            "ARTSY_MODEL_API_KEY": "",
            "ARTSY_MODEL_URL": "http://127.0.0.1:8080/v1",
        }, clear=True):
            model = model_from_env()
            self.assertEqual(model.api.api_key, "local-no-auth")
            with inspect_swe_model_auth():
                bridged = _model_without_responses_api(model)
                self.assertEqual(bridged.api.api_key, "local-no-auth")
                self.assertEqual(bridged.api.base_url, "http://127.0.0.1:8080/v1")
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_local_alias_uses_generic_provider_and_preserves_wire_model(self):
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": "http://127.0.0.1:8080/v1"}, clear=True):
            model = model_from_env("artsy/sft-v1")
            self.assertEqual(str(model), "openai-api/artsy/artsy/sft-v1")
            self.assertEqual(model.api.service_model_name(), "artsy/sft-v1")
            self.assertIs(_model_without_responses_api(model), model)
            from inspect_ai.model import GenerateConfig
            params = model.api.completion_params(GenerateConfig(temperature=0.8), tools=True)
            self.assertEqual(params["temperature"], 0.8)
            self.assertEqual(params["model"], "artsy/sft-v1")

    def test_remote_key_is_scoped_and_existing_openai_key_is_restored(self):
        with patch.dict(os.environ, {
            "ARTSY_MODEL_API_KEY": "drawing-endpoint-key",
            "OPENAI_API_KEY": "independent-judge-key",
        }, clear=True):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with inspect_swe_model_auth():
                    self.assertEqual(os.environ["OPENAI_API_KEY"], "drawing-endpoint-key")
                    raise RuntimeError("test failure")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "independent-judge-key")

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
            with inspect_swe_model_auth():
                bridged = _model_without_responses_api(model)
                self.assertEqual(bridged.api.api_key, "test-tinker-key")
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
