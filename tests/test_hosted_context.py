import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from training.hosted_context import ServiceClient, TrainingClient, cookbook_hosted_context

HOSTED = 'Qwen/Qwen3.5-4B:peft:262144'


class HostedContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_weights_and_optimizer_restore_select_same_context(self):
        raw = SimpleNamespace(create_training_client_from_state_async=AsyncMock(return_value=object()),
            create_training_client_from_state_with_optimizer_async=AsyncMock(return_value=object()),
            create_lora_training_client_async=AsyncMock(return_value=object()))
        service = ServiceClient(raw, HOSTED)
        await service.create_training_client_from_state_async('weights', user_metadata={'renderer': 'qwen'})
        await service.create_training_client_from_state_with_optimizer_async('optimizer')
        await service.create_lora_training_client_async('base', rank=8)
        raw.create_training_client_from_state_async.assert_awaited_once_with('weights', base_model=HOSTED, user_metadata={'renderer': 'qwen'})
        raw.create_training_client_from_state_with_optimizer_async.assert_awaited_once_with('optimizer', base_model=HOSTED)
        raw.create_lora_training_client_async.assert_awaited_once_with(HOSTED, rank=8)

    async def test_periodic_and_between_checkpoint_samplers_use_explicit_context(self):
        service = SimpleNamespace(create_sampling_client=Mock(return_value='periodic'),
                                  create_sampling_client_async=AsyncMock(return_value='fresh'))
        future = SimpleNamespace(result_async=AsyncMock(return_value=SimpleNamespace(path='new-sampler')))
        raw = SimpleNamespace(save_weights_for_sampler_async=AsyncMock(return_value=future))
        client = TrainingClient(raw, service, HOSTED)
        self.assertEqual(client.create_sampling_client('checkpoint2'), 'periodic')
        service.create_sampling_client.assert_called_once_with(model_path='checkpoint2', base_model=HOSTED)
        self.assertEqual(await client.save_weights_and_get_sampling_client_async(), 'fresh')
        self.assertEqual(raw.save_weights_for_sampler_async.await_args.kwargs['ttl_seconds'], 86400)
        service.create_sampling_client_async.assert_awaited_once_with(model_path='new-sampler', base_model=HOSTED)

    def test_proxy_scope_does_not_modify_sdk(self):
        sdk = SimpleNamespace(ServiceClient=Mock(return_value=object()), types='types')
        cookbook = SimpleNamespace(tinker=sdk)
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            with cookbook_hosted_context(cookbook, HOSTED):
                self.assertIsInstance(cookbook.tinker.ServiceClient(), ServiceClient)
                self.assertEqual(cookbook.tinker.types, 'types')
                self.assertIsNot(cookbook.tinker, sdk)
                raise RuntimeError('boom')
        self.assertIs(cookbook.tinker, sdk)


class BudgetBoundaryTests(unittest.TestCase):
    def test_native_context_remainder_clamps_then_stops_gracefully(self):
        from tinker_cookbook.rl.rollout_runner import _RolloutRun
        from tinker_cookbook.rl.rollout_limits import RolloutLimits
        from tinker_cookbook.rl.types import StopReason
        run = object.__new__(_RolloutRun)
        run.limits = RolloutLimits(max_trajectory_tokens=262144, max_turn_tokens=None)
        run.sampled_tokens = 0
        run.ob = SimpleNamespace(length=65536)
        self.assertEqual(run._per_call_budget(), 196608)
        self.assertIsNone(run._budget_stop_reason())
        run.ob = SimpleNamespace(length=262140)
        self.assertEqual(run._per_call_budget(), 4)
        for length in (262143, 262144, 262145):
            run.ob = SimpleNamespace(length=length)
            self.assertEqual(run._budget_stop_reason(), StopReason.MAX_TOKENS)
