"""Visual DPO data, objective, checkpoint handoff and resume without hosted calls."""
import copy
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from drawing.config import MODEL_ID, MODEL_REVISION, TINKER_SAMPLING_MODEL_ID
from training import dpo, grpo
from training.data import ROOT, digest
from training.sft import RENDERER


class DPOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records, self.labels = [], []
        self.data = self.root / "labels.jsonl"
        self.trajectories = self.root / "accepted.jsonl"
        self.initial = self.root / "sft" / "final.json"
        self.initial.parent.mkdir()
        self.initial.write_text(json.dumps({
            "state_path": "tinker://sft/state", "sampler_path": "tinker://sft/sampler",
            "config": {"model": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
                       "renderer": RENDERER, "lora_rank": 32,
                       "api_doc_sha256": digest((ROOT / "docs/artcanvas.md").read_bytes())}}))
        self.add_pair("train", "blue", "navy")
        self.write()

    def add_pair(self, split, color_a, color_b):
        number = len(self.labels)
        ids = []
        for side, color in (("a", color_a), ("b", color_b)):
            rid = f"{number}-{side}"
            ids.append(rid)
            directory = self.root / rid
            directory.mkdir()
            code = f"draw({color!r})\n"
            (directory / "program.py").write_text(code)
            for image in ("image.png", "preview-1.png"):
                Image.new("RGB", (24, 16), color).save(directory / image)
            self.records.append({
                "id": rid, "status": "accepted", "spec": {"prompt": f"Draw circle {number}", "seed": 1},
                "runtimeVersion": "artcanvas-v2", "apiDocHash": digest((ROOT / "docs/artcanvas.md").read_bytes()),
                "artifacts": {"directory": rid, "source": f"{rid}/program.py", "image": f"{rid}/image.png"},
                "code": code, "sourceHash": digest(code.strip().encode()),
                "imageHash": digest((directory / "image.png").read_bytes()),
                "trace": [{"completion": {"text": json.dumps({"code": code})}, "executedCode": code,
                           "feedback": {"ok": True, "exitCode": 0, "artifact": "/old/preview-1.png"}}]})
        self.labels.append({"pair_id": f"pair-{number}", "group_id": f"family-{number}",
                            "image_a": f"{ids[0]}/image.png", "image_b": f"{ids[1]}/image.png",
                            "record_a": ids[0], "record_b": ids[1], "winner": "B", "split": split})

    def write(self):
        self.data.write_text("".join(json.dumps(r) + "\n" for r in self.labels))
        self.trajectories.write_text("".join(json.dumps(r) + "\n" for r in self.records))

    def arguments(self, *extra):
        return ["--data", str(self.data), "--trajectories", str(self.trajectories),
                "--init-checkpoint", str(self.initial), "--output-dir", str(self.root / "dpo"), *extra]

    def plan(self, *extra):
        args = dpo.parser().parse_args(self.arguments(*extra))
        pairs, manifest = dpo.load_preferences(self.data, self.trajectories)
        return args, pairs, dpo.make_plan(args, manifest, dpo.load_initial_checkpoint(self.initial))

    def test_pairs_train_drawing_trajectories_and_exclude_ties_and_held_out(self):
        self.add_pair("validation", "red", "pink")
        self.add_pair("train", "green", "lime")
        self.labels[-1]["winner"] = "Tie"
        self.write()
        _, pairs, plan = self.plan()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"]["id"], "0-b")
        self.assertEqual(pairs[0]["rejected"]["id"], "0-a")
        self.assertEqual(plan["config"]["reference_sampler_path"], "tinker://sft/sampler")
        self.assertEqual(plan["config"]["lora_rank"], 32)
        self.assertIn("tool_calls", pairs[0]["chosen"]["messages"][2])

    def test_mismatched_prompt_seed_image_unknown_record_and_unsafe_records_rejected(self):
        originals = copy.deepcopy((self.records, self.labels))
        for mutation, error in (
                (lambda: self.records[1]["spec"].update(prompt="Other request"), "same prompt and seed"),
                (lambda: self.records[1]["spec"].update(seed=2), "same prompt and seed"),
                (lambda: self.labels[0].update(record_b="0-a"), "label image does not match"),
                (lambda: self.labels[0].update(record_b="missing"), "recorded trajectory"),
                (lambda: self.records[1].update(status="review"), "approved drawing"),
                (lambda: self.records[1]["trace"][0].update(evaluation={"safety": {"harmful": True}}), "unsafe image")):
            with self.subTest(error=error):
                self.records, self.labels = copy.deepcopy(originals)
                mutation()
                self.write()
                with self.assertRaisesRegex(ValueError, error):
                    dpo.load_preferences(self.data, self.trajectories)

    def test_prompt_cannot_leak_across_splits_with_different_images(self):
        self.add_pair("validation", "red", "pink")
        for record in self.records[2:]:
            record["spec"]["prompt"] = self.records[0]["spec"]["prompt"]
        self.write()
        with self.assertRaisesRegex(ValueError, "leaks across splits"):
            dpo.load_preferences(self.data, self.trajectories)

    def test_ties_are_not_fabricated_preferences(self):
        self.labels[0]["winner"] = "Tie"
        self.write()
        with self.assertRaisesRegex(ValueError, "non-tie training pairs"):
            self.plan()

    def test_dry_run_and_check_batch_never_start_training_or_write(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(dpo, "train") as train, \
                patch.object(dpo, "create_encoder") as encoder:
            self.assertEqual(dpo.main(self.arguments("--dry-run")), 0)
            encoder.assert_not_called()
            encoder.return_value.side_effect = lambda example: [example["id"]]
            self.assertEqual(dpo.main(self.arguments("--check-batch")), 0)
            self.assertEqual(encoder.return_value.call_count, 2)
            train.assert_not_called()
        self.assertFalse((self.root / "dpo").exists())

    def test_stage_handoff_requires_sft_then_dpo(self):
        args, _, plan = self.plan()
        prompts = self.root / "prompts.jsonl"
        prompts.write_text(json.dumps({"id": "x", "prompt": "Draw", "seed": 1}) + "\n")
        rl_args = grpo.parser().parse_args(["--dataset", str(prompts), "--init-checkpoint", str(self.initial)])
        with self.assertRaisesRegex(ValueError, "compatible DPO"):
            grpo.load_run(rl_args)
        final = self.root / "dpo-final.json"
        final.write_text(json.dumps({"config": plan["config"], "state_path": "tinker://dpo/state",
                                     "sampler_path": "tinker://dpo/sampler"}))
        rl_args.init_checkpoint = final
        _, rl_plan, _ = grpo.load_run(rl_args)
        self.assertEqual(rl_plan["initial_state_path"], "tinker://dpo/state")
        self.assertEqual(rl_plan["reference_sampler_path"], "tinker://dpo/sampler")
        self.assertEqual(rl_plan["reward"].split("benign=")[1], "adherence")
        self.assertFalse(any(k.startswith("visual_") for k in rl_plan))
        self.assertEqual(grpo.parser().parse_args([]).init_checkpoint, ROOT / "checkpoints/dpo-v1/final.json")
        with self.assertRaisesRegex(ValueError, "compatible SFT"):
            dpo.load_initial_checkpoint(final)

    @staticmethod
    def datum():
        import tinker
        return tinker.Datum(model_input=tinker.ModelInput.from_ints([1, 2, 3, 4]),
                            loss_fn_inputs={"target_tokens": [2, 3, 4, 5], "weights": [0., 1., 0., 1.]})

    def test_dpo_loss_direction_masking_reference_and_pair_alignment(self):
        import torch
        data = [self.datum(), self.datum()]
        refs = [[None, -1., None, -1.]] * 2
        policy = [torch.tensor([float("nan"), -1., float("nan"), -1.], requires_grad=True) for _ in data]
        loss, _ = dpo.dpo_loss(data, policy, refs, 0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertLess(policy[0].grad[1], 0)
        self.assertGreater(policy[1].grad[1], 0)
        self.assertEqual(policy[0].grad[0], 0)
        preferred = [torch.tensor([0., -0.5, 0., -0.5]), policy[1]]
        self.assertLess(dpo.dpo_loss(data, preferred, refs, 0.1)[0], loss)
        with self.assertRaisesRegex(ValueError, "assistant target"):
            dpo.dpo_loss(data, policy, [[None] * 4] * 2, 0.1)
        with self.assertRaisesRegex(ValueError, "aligned chosen/rejected"):
            dpo.dpo_loss(data[:1], policy, refs, 0.1)

    def test_fresh_and_resumed_training_keep_sft_reference_and_export_rl_checkpoint(self):
        import torch
        self.add_pair("train", "red", "pink")
        self.write()
        for start in (0, 1):
            with self.subTest(start=start):
                args, pairs, plan = self.plan("--batch-size", "1")
                args.output_dir = self.root / f"dpo-{start}"
                args.output_dir.mkdir()
                if start:
                    checkpoint = args.output_dir / "checkpoint-000001.json"
                    checkpoint.write_text(json.dumps({"config": plan["config"], "step": start,
                                                      "state_path": "tinker://dpo/resume-state"}))
                    args.resume_from_checkpoint = checkpoint
                service = MagicMock()
                client = (service.create_training_client_from_state_with_optimizer.return_value if start
                          else service.create_training_client_from_state.return_value)
                client.save_state.return_value.result.return_value.path = "tinker://dpo/final-state"
                client.save_weights_for_sampler.return_value.result.return_value.path = "tinker://dpo/final-sampler"
                reference = service.create_sampling_client.return_value
                reference.compute_logprobs.return_value.result.return_value = [None, None, -1., None, -1.]

                def backward(data, loss_fn):
                    _, metrics = loss_fn(data, [torch.tensor([0., -1., 0., -1.], requires_grad=True) for _ in data])
                    return SimpleNamespace(result=lambda: SimpleNamespace(metrics=metrics))

                client.forward_backward_custom.side_effect = backward
                with patch("tinker.ServiceClient", return_value=service), contextlib.redirect_stdout(io.StringIO()):
                    dpo.train(args, [(self.datum(), self.datum()) for _ in pairs], plan)
                if start:
                    service.create_training_client_from_state_with_optimizer.assert_called_once_with("tinker://dpo/resume-state")
                    service.create_training_client_from_state.assert_not_called()
                else:
                    service.create_training_client_from_state.assert_called_once_with("tinker://sft/state")
                    service.create_training_client_from_state_with_optimizer.assert_not_called()
                service.create_sampling_client.assert_called_once_with(
                    base_model=TINKER_SAMPLING_MODEL_ID, model_path="tinker://sft/sampler")
                self.assertEqual(client.optim_step.call_count, 2 - start)
                self.assertEqual(reference.compute_logprobs.call_args.args[0].to_ints(), [1, 2, 3, 4, 5])
                final = json.loads((args.output_dir / "final.json").read_text())
                self.assertEqual(final["step"], 2)
                self.assertEqual(final["config"], plan["config"])
                self.assertEqual(final["sampler_path"], "tinker://dpo/final-sampler")

    def test_resume_drift_and_invalid_hyperparameters_fail_before_training(self):
        args, _, plan = self.plan()
        checkpoint = self.root / "checkpoint.json"
        checkpoint.write_text(json.dumps({"config": {**plan["config"], "beta": 999},
                                          "step": 0, "state_path": "tinker://resume"}))
        for options in (("--beta", "nan"), ("--learning-rate", "0"), ("--max-steps", "0"),
                        ("--resume-from-checkpoint", str(checkpoint))):
            with contextlib.redirect_stderr(io.StringIO()), patch.object(dpo, "train") as train:
                with self.assertRaises(SystemExit):
                    dpo.main(self.arguments(*options))
                train.assert_not_called()
