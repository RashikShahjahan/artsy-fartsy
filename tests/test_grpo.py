"""Verify on-policy evidence, masked GRPO gradients, and checkpoint continuation."""

import asyncio
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import tinker
import torch

from training.grpo import (NativeRollouts, ReferenceLogprobs, advantages, build_training_data, collect_rollouts,
                           grpo_loss, main, parser, train)


def turn(tokens=(10, 11), prompt=None):
    prompt = prompt or tinker.ModelInput.from_ints([1, 2, 3])
    return {"prompt": prompt.model_dump(mode="json"), "tokens": list(tokens),
            "logprobs": [-1.0] * len(tokens), "stop_reason": "stop"}


def groups():
    return [{"id": "a", "attempts": [{"epoch": 1, "reward": 0.0, "turns": [turn()]},
                                      {"epoch": 2, "reward": 1.0, "turns": [turn(), turn((12,))]}]}]


class ObjectiveTests(unittest.TestCase):
    def test_group_baselines_do_not_mix_prompts_and_reject_invalid_rewards(self):
        self.assertEqual(advantages([0, 1]), [-0.5, 0.5])
        self.assertEqual(advantages([1, 1]), [0, 0])
        self.assertAlmostEqual(advantages([0, 1], normalize=True)[1], 1.0)
        for rewards in ([], [1], [0, float("nan")], [-1, 1], [True, 1]):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                advantages(rewards)
        second = {"id": "b", "attempts": [{"reward": 0.2, "turns": [turn()]},
                                            {"reward": 0.2, "turns": [turn()]}]}
        data = build_training_data(groups() + [second])
        self.assertTrue(all(v == 0 for d in data[-2:] for v in d.loss_fn_inputs["advantages"].data))

    def test_exact_tokens_shift_images_and_equal_trajectory_weight(self):
        group = groups()
        image = tinker.types.ImageChunk(data=b"\x89PNG\x00\xff", format="png", expected_tokens=4)
        prompt = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=[1]), image,
                                         tinker.EncodedTextChunk(tokens=[2, 3])])
        group[0]["attempts"][1]["turns"][1] = turn((12,), prompt)
        original = copy.deepcopy(group)
        data = build_training_data(group)
        self.assertEqual(group, original)
        self.assertEqual(data[0].model_input.to_ints(), [1, 2, 3, 10])
        self.assertEqual(data[0].loss_fn_inputs["target_tokens"].data, [2, 3, 10, 11])
        self.assertEqual(data[0].loss_fn_inputs["logprobs"].data, [0, 0, -1, -1])
        self.assertEqual(data[0].loss_fn_inputs["advantages"].data, [0, 0, -0.5, -0.5])
        self.assertAlmostEqual(sum(data[0].loss_fn_inputs["weights"].data), 0.5)
        self.assertAlmostEqual(sum(sum(d.loss_fn_inputs["weights"].data) for d in data[1:]), 0.5)
        self.assertIn(image, data[-1].model_input.chunks)
        self.assertEqual(data[-1].model_input.length, prompt.length)
        self.assertTrue(all(w == 0 for w in data[-1].loss_fn_inputs["weights"].data[:-1]))

    def test_masked_gradient_signs_clipping_and_reference_kl(self):
        data = build_training_data(groups())
        currents = [d.loss_fn_inputs["logprobs"].to_torch().clone().requires_grad_(True) for d in data]
        for d in data:
            d.loss_fn_inputs["reference_logprobs"] = d.loss_fn_inputs["logprobs"]
        value, metrics = grpo_loss(0.2, 0.01)(data, currents)
        self.assertAlmostEqual(float(value.detach()), 0)
        self.assertEqual(metrics["reference_kl"], 0)
        value.backward()
        self.assertTrue(torch.all(currents[0].grad[:2] == 0))
        self.assertTrue(torch.all(currents[0].grad[2:] > 0))
        self.assertTrue(torch.all(currents[1].grad[2:] < 0))
        # A positive-advantage action outside the upper clip has zero policy gradient.
        d = data[1]
        current = d.loss_fn_inputs["logprobs"].to_torch().clone()
        current[2:] += math.log(1.5)
        current.requires_grad_(True)
        clipped, _ = grpo_loss(0.2, 0)([d], [current])
        clipped.backward()
        self.assertTrue(torch.all(current.grad == 0))
        current = current.detach().requires_grad_(True)
        regularized, metrics = grpo_loss(0.2, 0.1)([d], [current])
        regularized.backward()
        self.assertGreater(metrics["reference_kl"], 0)
        self.assertTrue(torch.all(current.grad[2:] > 0))

    def test_reference_scores_exact_full_sequences_and_ignores_context_nans(self):
        samples = groups()
        reference = SimpleNamespace(compute_logprobs_async=AsyncMock())

        async def compute(full):
            return [None, float("nan"), None] + [-2.0] * (full.length - 3)

        reference.compute_logprobs_async.side_effect = compute
        async def annotate(directory):
            prefetch = ReferenceLogprobs(reference, 2)
            try:
                for index, record in enumerate(t for a in samples[0]["attempts"] for t in a["turns"]):
                    prefetch.submit(tinker.ModelInput.model_validate(record["prompt"]), record,
                                    directory / f"turn-{index}.json")
                await prefetch.finish()
            finally:
                await prefetch.close()
        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(annotate(Path(temp)))
            self.assertEqual(json.loads((Path(temp) / "turn-0.json").read_text())["reference_logprobs"], [-2, -2])
        data = build_training_data(samples)
        first_full = reference.compute_logprobs_async.call_args_list[0].args[0]
        self.assertEqual(first_full.to_ints(), [1, 2, 3, 10, 11])
        self.assertEqual(data[0].loss_fn_inputs["reference_logprobs"].data, [0, 0, -2, -2])


class ReferencePrefetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_requests_keep_out_of_order_results_with_their_actions(self):
        active, peak = 0, 0
        second_done = asyncio.Event()

        async def compute(full):
            nonlocal active, peak
            token = full.to_ints()[-1]
            active += 1
            peak = max(peak, active)
            try:
                if token == 10:
                    await second_done.wait()
                else:
                    await asyncio.sleep(0)
                if token == 11:
                    second_done.set()
                return [None, float("nan"), None, -float(token)]
            finally:
                active -= 1

        with tempfile.TemporaryDirectory() as temp:
            prefetch = ReferenceLogprobs(SimpleNamespace(compute_logprobs_async=compute), 2)
            records = [turn((token,)) for token in range(10, 16)]
            try:
                for index, record in enumerate(records):
                    prefetch.submit(tinker.ModelInput.model_validate(record["prompt"]), record,
                                    Path(temp) / f"turn-{index}.json")
                await asyncio.wait_for(prefetch.finish(), timeout=5)
                self.assertEqual(peak, 2)
                self.assertEqual(prefetch.completed, 6)
                self.assertEqual([r["reference_logprobs"] for r in records], [[-float(t)] for t in range(10, 16)])
            finally:
                await prefetch.close()

    async def test_failed_reference_cancels_other_requests_and_rejects_invalid_action_scores(self):
        for values in ([None, 0.0, 0.0, None], [None, 0.0, 0.0, float("nan")], [None]):
            started = asyncio.Event()
            active = 0

            async def compute(full):
                nonlocal active
                active += 1
                try:
                    if full.to_ints()[-1] == 10:
                        await started.wait()
                        return values
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    active -= 1

            with self.subTest(values=values), tempfile.TemporaryDirectory() as temp:
                prefetch = ReferenceLogprobs(SimpleNamespace(compute_logprobs_async=compute), 2)
                try:
                    for index, token in enumerate((10, 11, 12)):
                        record = turn((token,))
                        prefetch.submit(tinker.ModelInput.model_validate(record["prompt"]), record,
                                        Path(temp) / f"turn-{index}.json")
                    with self.assertRaisesRegex(ValueError, "invalid action log probabilities"):
                        await asyncio.wait_for(prefetch.finish(), timeout=5)
                finally:
                    await prefetch.close()
                self.assertEqual(active, 0)
                self.assertEqual(list(Path(temp).glob("turn-*.json")), [])

    async def test_reference_overlaps_collection_and_judging_and_is_awaited_before_return(self):
        args = parser().parse_args(["--group-size", "2"])
        reference_started = asyncio.Event()
        release_reference = asyncio.Event()
        records = {("a", 1): [turn()], ("a", 2): [turn((12,))]}

        async def compute(full):
            reference_started.set()
            await release_reference.wait()
            return [None] * 3 + [-2.0] * (full.length - 3)

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def recorder(backend, options, step, output, prefetch):
                for (name, epoch), turns in records.items():
                    artifact = output / "artifacts" / f"{name}-{epoch}"
                    artifact.mkdir(parents=True)
                    (artifact / "result.json").write_text("{}")
                    for record in turns:
                        prefetch.submit(tinker.ModelInput.model_validate(record["prompt"]), record,
                                        artifact / "native.json")
                return SimpleNamespace(turns=records)

            async def evaluate(*args, **kwargs):
                await asyncio.wait_for(reference_started.wait(), timeout=5)
                return [SimpleNamespace(status="success", samples=[])]

            async def score(*args, **kwargs):
                self.assertTrue(reference_started.is_set())
                self.assertTrue(all("reference_logprobs" not in t for turns in records.values() for t in turns))
                release_reference.set()
                return [{"id": "a", "attempts": [{"epoch": 1, "reward": 0}, {"epoch": 2, "reward": 1}]}]

            with patch("training.grpo.NativeRollouts", side_effect=recorder), \
                 patch("inspect_ai.eval_async", side_effect=evaluate), \
                 patch("evaluation.harness.drawing_task"), \
                 patch("evaluation.sample_grpo.score_outputs", side_effect=score):
                result = await asyncio.wait_for(collect_rollouts(
                    args, [{"id": "a"}], MagicMock(), MagicMock(), 1, directory, MagicMock(),
                    SimpleNamespace(compute_logprobs_async=compute)), timeout=5)
            data = build_training_data(result)
            self.assertEqual(data[0].loss_fn_inputs["reference_logprobs"].data, [0, 0, -2, -2])
            self.assertEqual(data[1].loss_fn_inputs["reference_logprobs"].data, [0, 0, -2])
            metrics = json.loads((directory / "phase-metrics.json").read_text())
            self.assertEqual(metrics["reference_requests"], 2)
            self.assertEqual(metrics["reference_completed_during_rollouts"], 0)


class NativeSamplingTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_native_samples_and_keeps_group_members_and_seeds_distinct(self):
        from inspect_ai.model import ChatMessageUser, GenerateConfig
        from inspect_ai.tool import ToolInfo

        with tempfile.TemporaryDirectory() as temp:
            backend = MagicMock()
            backend.render.return_value = tinker.ModelInput.from_ints([1, 2, 3])
            backend.renderer.get_stop_sequences.return_value = [99]
            backend.tokenizer.decode.return_value = (
                '<tool_call><function=bash><parameter=command>echo hello</parameter></function></tool_call>')
            backend.client.sample_async = AsyncMock(return_value=SimpleNamespace(sequences=[
                SimpleNamespace(tokens=[10, 11], logprobs=[-0.3, -0.4], stop_reason="stop")]))
            args = parser().parse_args([])
            prefetch = MagicMock()
            recorder = NativeRollouts(backend, args, 1, Path(temp), prefetch)
            state = SimpleNamespace(sample_id="sample", epoch=1)
            tools = [ToolInfo(name="bash", description="Run bash", parameters={
                "type": "object", "properties": {"command": {"type": "string"}}})]
            with patch("inspect_ai.solver._task_state.sample_state", return_value=state):
                result = await recorder(None, [ChatMessageUser(content="Draw a disk")], tools, None, GenerateConfig())
                state.epoch = 2
                await recorder(None, [ChatMessageUser(content="Draw a disk")], tools, None, GenerateConfig())
            self.assertEqual(result.choices[0].message.tool_calls[0].arguments, {"command": "echo hello"})
            self.assertEqual(result.usage.output_tokens, 2)
            self.assertEqual(recorder.turns[("sample", 1)][0]["logprobs"], [-0.3, -0.4])
            self.assertTrue((Path(temp) / "native/sample-1/turn-000.json").is_file())
            self.assertEqual(prefetch.submit.call_count, 2)
            self.assertEqual(prefetch.submit.call_args_list[0].args[1], recorder.turns[("sample", 1)][0])
            self.assertEqual(prefetch.submit.call_args_list[0].args[2], Path(temp) / "native/sample-1/turn-000.json")
            calls = backend.client.sample_async.call_args_list
            self.assertEqual(calls[0].kwargs["sampling_params"].temperature, 0.8)
            self.assertNotEqual(calls[0].kwargs["sampling_params"].seed, calls[1].kwargs["sampling_params"].seed)
            backend.render.return_value = tinker.ModelInput.from_ints([1] * args.max_length)
            with patch("inspect_ai.solver._task_state.sample_state", return_value=state), self.assertRaisesRegex(ValueError, "context exhausted"):
                await recorder(None, [], [], None, GenerateConfig())
            self.assertEqual(backend.client.sample_async.call_count, 2)

    async def test_collection_stops_on_sample_errors_before_judging(self):
        args = parser().parse_args([])
        with tempfile.TemporaryDirectory() as temp, \
             patch("inspect_ai.eval_async", AsyncMock(return_value=[SimpleNamespace(
                 status="success", samples=[SimpleNamespace(error="provider failure")])])), \
             patch("evaluation.harness.drawing_task"), \
             patch("evaluation.sample_grpo.score_outputs", new_callable=AsyncMock) as score:
            with self.assertRaisesRegex(RuntimeError, "collection failed"):
                await collect_rollouts(args, [{"id": "a"}], MagicMock(), MagicMock(), 1, Path(temp), MagicMock())
            score.assert_not_called()

    async def test_truncation_zeroes_terminal_reward_and_updates_group_statistics(self):
        args = parser().parse_args(["--group-size", "2"])
        samples = groups()
        samples[0]["attempts"][0]["reward"] = 0.7
        native = {("a", 1): [{**turn(), "stop_reason": "length"}], ("a", 2): [turn()]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for epoch in (1, 2):
                artifact = directory / "artifacts" / f"a-{epoch}"
                artifact.mkdir(parents=True)
                (artifact / "result.json").write_text("{}")
            with patch("training.grpo.NativeRollouts", return_value=SimpleNamespace(turns=native)), \
                 patch("inspect_ai.eval_async", AsyncMock(return_value=[SimpleNamespace(status="success", samples=[])])), \
                 patch("evaluation.harness.drawing_task"), \
                 patch("evaluation.sample_grpo.score_outputs", AsyncMock(return_value=samples)):
                result = await collect_rollouts(args, [{"id": "a"}], MagicMock(), MagicMock(), 1, directory, MagicMock())
            self.assertEqual(result[0]["rewards"], [0.0, 1.0])
            self.assertEqual(result[0]["mean"], 0.5)
            self.assertTrue(result[0]["attempts"][0]["truncated"])
            self.assertTrue((directory / "groups.json").is_file())


class TrainerTests(unittest.TestCase):
    def test_on_policy_refresh_and_exact_resume_with_optimizer(self):
        specs = [{"id": str(i)} for i in range(4)]
        config = {"initial_state_path": "tinker://sft/state", "reference_sampler_path": "tinker://sft/sampler",
                  "correctness_judge_model": "judge", "correctness_judge_url": "https://example.com/v1"}
        client, service = MagicMock(), MagicMock()
        service.create_training_client_from_state.return_value = client
        service.create_training_client_from_state_with_optimizer.return_value = client
        client.forward_backward_custom.return_value.result.return_value = SimpleNamespace(metrics={"loss": 0.2})
        client.save_state.return_value.result.return_value = SimpleNamespace(path="tinker://grpo/state")
        client.save_weights_for_sampler.return_value.result.return_value = SimpleNamespace(path="tinker://grpo/sampler")
        with tempfile.TemporaryDirectory() as temp:
            args = parser().parse_args(["--output-dir", temp, "--max-steps", "1", "--kl-coef", "0"])
            with patch("tinker.ServiceClient", return_value=service), \
                 patch("training.grpo.create_encoder", return_value=MagicMock()), \
                 patch("training.grpo.collect_rollouts", new_callable=AsyncMock, return_value=groups()) as collect, \
                 contextlib.redirect_stdout(io.StringIO()):
                train(args, specs, config, 1)
                service.create_training_client_from_state.assert_called_once_with("tinker://sft/state")
                self.assertEqual(client.optim_step.call_count, 1)
                checkpoint = Path(temp) / "checkpoint-000001.json"
                args.resume_from_checkpoint = checkpoint
                train(args, specs, config, 2)
                service.create_training_client_from_state_with_optimizer.assert_called_once_with("tinker://grpo/state")
                self.assertEqual(client.save_weights_and_get_sampling_client.call_count, 2)
                self.assertEqual(client.optim_step.call_count, 2)
                first, second = collect.call_args_list
                self.assertNotEqual(first.args[1], second.args[1])
                self.assertEqual(second.args[4], 2)
                rows = [json.loads(line) for line in (Path(temp) / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in rows], [1, 2])
                final = json.loads((Path(temp) / "final.json").read_text())
                self.assertEqual(final["sampler_path"], "tinker://grpo/sampler")
                with self.assertRaisesRegex(ValueError, "configuration does not match"):
                    train(args, specs, {**config, "changed": True}, 2)
                self.assertEqual(client.optim_step.call_count, 2)

    def test_no_variation_skips_optimizer_but_saves_progress(self):
        flat = groups()
        flat[0]["attempts"][1]["reward"] = 0
        config = {"initial_state_path": "tinker://sft/state", "reference_sampler_path": "tinker://sft/sampler",
                  "correctness_judge_model": "judge", "correctness_judge_url": "https://example.com/v1"}
        client = MagicMock()
        client.save_state.return_value.result.return_value = SimpleNamespace(path="tinker://grpo/state")
        client.save_weights_for_sampler.return_value.result.return_value = SimpleNamespace(path="tinker://grpo/sampler")
        with tempfile.TemporaryDirectory() as temp:
            args = parser().parse_args(["--output-dir", temp])
            with patch("tinker.ServiceClient") as service, patch("training.grpo.create_encoder"), \
                 patch("training.grpo.collect_rollouts", new_callable=AsyncMock, return_value=flat), \
                 patch("training.grpo.ReferenceLogprobs") as reference, \
                 contextlib.redirect_stdout(io.StringIO()):
                service.return_value.create_training_client_from_state.return_value = client
                train(args, [{"id": "a"}], config, 1)
                client.optim_step.assert_not_called()
                client.forward_backward_custom.assert_not_called()
                reference.assert_not_called()
                self.assertTrue((Path(temp) / "checkpoint-000001.json").is_file())

    def test_dry_run_requires_no_service_or_encoder(self):
        with patch("training.grpo.load_run", return_value=([{"id": "a"}], {}, 1)), \
             patch("training.grpo.train") as run, patch("training.grpo.create_encoder") as encoder, \
             patch("tinker.ServiceClient") as service, contextlib.redirect_stdout(io.StringIO()):
            main(["--dry-run"])
        run.assert_not_called()
        encoder.assert_not_called()
        service.assert_not_called()

    def test_judge_or_collection_failure_never_reaches_optimizer(self):
        config = {"initial_state_path": "tinker://sft/state", "reference_sampler_path": "tinker://sft/sampler",
                  "correctness_judge_model": "judge", "correctness_judge_url": "https://example.com/v1"}
        with tempfile.TemporaryDirectory() as temp:
            args = parser().parse_args(["--output-dir", temp])
            with patch("tinker.ServiceClient") as service, patch("training.grpo.create_encoder"), \
                 patch("training.grpo.collect_rollouts", new_callable=AsyncMock, side_effect=RuntimeError("Judge unavailable")):
                client = service.return_value.create_training_client_from_state.return_value
                with self.assertRaisesRegex(RuntimeError, "Judge unavailable"):
                    train(args, [{"id": "a"}], config, 1)
                client.forward_backward_custom.assert_not_called()
                client.optim_step.assert_not_called()
                client.save_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
