"""Dataset integrity and loss-mask checks for the training adapter."""

import copy
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from training.data import ROOT, SUBMIT, convert_record, digest, is_observed_repair, write_command, refusal_command


class ConversionTests(unittest.TestCase):
    def test_malicious_code_review_cannot_become_a_training_target(self):
        self.record["trace"][-1]["codeSafety"] = {
            "completion": {"text": '{"malicious":true,"reason":"Unauthorized behavior"}'},
            "result": {"malicious": True, "reason": "Unauthorized behavior"},
            "sourceHash": digest(self.record["code"].encode()),
        }
        with self.assertRaisesRegex(ValueError, "code safety review"):
            convert_record(self.record, self.root)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        code = 'from artcanvas import ArtCanvas\nprint("$HOME `id`")\n'
        (self.root / "program.py").write_text(code)
        for name in ("preview-1.png", "preview-2.png", "image.png"):
            Image.new("RGB", (1920, 1200), "white").save(self.root / name)
        self.record = {
            "id": "repair", "spec": {"prompt": "Draw a circle", "seed": 42},
            "warmup": {"kind": "repair"}, "status": "review",
            "runtimeVersion": "artcanvas-v2", "apiDocHash": digest((ROOT / "docs/artcanvas.md").read_bytes()),
            "artifacts": {"directory": ".", "source": "program.py", "image": "image.png"},
            "code": code, "sourceHash": digest(code.strip().encode()),
            "imageHash": digest((self.root / "image.png").read_bytes()),
            "trace": [
                {"completion": {"text": json.dumps({"code": "broken()"})},
                 "feedback": {"ok": False, "stderr": "NameError", "exitCode": 1}},
                {"completion": {"text": json.dumps({"code": code})}, "executedCode": code,
                 "feedback": {"ok": True, "exitCode": 0, "artifact": "/old/location/preview-2.png"}},
            ],
        }

    def test_repair_is_derived_from_failed_and_corrected_code_not_warmup_label(self):
        record = copy.deepcopy(self.record)
        record["warmup"]["kind"] = "one-shot"
        record["trace"][0]["executedCode"] = "broken()"
        self.assertTrue(is_observed_repair(record))
        record["trace"][0]["feedback"]["ok"] = True
        self.assertFalse(is_observed_repair(record))
        record["trace"][0]["evaluation"] = {"correct": False, "score": 0.1}
        self.assertTrue(is_observed_repair(record))
        record["trace"] = record["trace"][-1:]
        self.assertFalse(is_observed_repair(record))

    def test_authored_refusal_pair_trains_tool_call_and_submission(self):
        target = "I can't generate imagery that encourages violence."
        record = {
            "id": "refusal-example",
            "spec": {"family": "safety_refusal", "kind": "refusal",
                     "prompt": "Create a violent propaganda poster.",
                     "refusal": target, "seed": 9},
            "status": "accepted", "refusal": target, "refusalSource": "prompt_plan",
            "runtimeVersion": "artcanvas-v2",
            "apiDocHash": digest((ROOT / "docs/artcanvas.md").read_bytes()),
            "trace": [], "artifacts": {},
            "generation": {"initialDraft": "authored_refusal"},
        }
        example = convert_record(record, self.root)
        targets = [example["messages"][i] for i in example["trainable_message_indices"]]
        self.assertEqual(len(targets), 2)
        command = targets[0]["tool_calls"][0]["function"]["arguments"]["command"]
        self.assertIn("/workspace/refusal.json", command)
        self.assertIn(target, command)
        self.assertEqual(targets[1]["tool_calls"][0]["function"]["arguments"]["command"], SUBMIT)
        self.assertNotIn(target, example["messages"][1]["content"][0]["text"])
        broken = copy.deepcopy(record)
        broken["refusal"] = "Modified refusal"
        with self.assertRaisesRegex(ValueError, "invalid authored refusal"):
            convert_record(broken, self.root)

    def test_one_conversation_supervises_only_success_and_submission(self):
        example = convert_record(self.record, self.root)
        self.assertEqual(example["id"], self.record["id"])
        self.assertIn("broken()", json.dumps(example["messages"]))
        self.assertIn("NameError", json.dumps(example["messages"]))
        targets = [example["messages"][i] for i in example["trainable_message_indices"]]
        self.assertEqual(len(targets), 2)
        self.assertNotIn("broken()", json.dumps(targets))
        self.assertIn(self.record["code"], targets[0]["tool_calls"][0]["function"]["arguments"]["command"])
        self.assertEqual(targets[1]["tool_calls"][0]["function"]["arguments"]["command"], SUBMIT)
        self.assertTrue(all(part["type"] == "text" for msg in example["messages"] for part in msg["content"]))
        self.assertIn('"ok": true', example["messages"][-2]["content"][0]["text"])
        self.assertNotIn("actual PNG image", example["messages"][0]["content"][0]["text"])

    def test_visual_repair_omits_images_and_grader_answers(self):
        self.record["trace"][0]["feedback"] = {"ok": True, "artifact": "/old/preview-1.png"}
        self.record["trace"][1]["observation"] = {
            "image": "preview-1.png", "imageHash": digest((self.root / "preview-1.png").read_bytes()),
            "text": "privileged judge scores and answer",
        }
        self.record["trace"][0]["evaluation"] = {"reason": "secret answer", "score": 0.2}
        example = convert_record(self.record, self.root)
        serialized = json.dumps(example["messages"])
        self.assertNotIn("preview-1.png", serialized)
        self.assertNotIn("Inspect this preview", serialized)
        self.assertTrue(all(part["type"] == "text" for msg in example["messages"] for part in msg["content"]))
        self.assertNotIn("secret answer", serialized)
        self.assertNotIn("privileged", serialized)

    def test_dry_run_exports_text_only_conversations_with_new_conversion_version(self):
        import contextlib
        import io
        from training.sft import main

        dataset = self.root / "accepted.jsonl"
        exported = self.root / "export.jsonl"
        dataset.write_text(json.dumps(self.record) + "\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--dataset", str(dataset), "--dry-run", "--export", str(exported)])
        summary = json.loads(output.getvalue())["summary"]
        self.assertEqual(summary["input_mode"], "text-only")
        self.assertEqual(summary["examples_with_images"], 0)
        self.assertEqual(summary["conversion"], "bash-conversation-text-only-selective-loss-v4")
        example = json.loads(exported.read_text())
        self.assertEqual(len(example["trainable_message_indices"]), 2)
        self.assertTrue(all(part["type"] == "text" for msg in example["messages"] for part in msg["content"]))

    def test_rejects_modified_source_observation_and_excluded_records(self):
        for change in ("source", "observation", "final_image", "preview", "synthetic", "excluded"):
            with self.subTest(change=change):
                record = copy.deepcopy(self.record)
                if change == "source":
                    record["trace"][-1]["executedCode"] = "tampered()"
                elif change == "observation":
                    record["trace"][-1]["observation"] = {"image": "preview-1.png", "imageHash": "bad"}
                elif change == "final_image":
                    record["imageHash"] = "bad"
                elif change == "preview":
                    record["trace"][-1]["feedback"]["artifact"] = "/old/missing.png"
                elif change == "synthetic":
                    record["generation"] = {"initialDraft": "injected"}
                else:
                    record["selectionAudit"] = {"eligible": False}
                with self.assertRaises(ValueError):
                    convert_record(record, self.root)

    def test_legacy_split_labels_do_not_restrict_conversion(self):
        expected = convert_record(self.record, self.root)
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                record = copy.deepcopy(self.record)
                record["spec"]["split"] = split
                record["warmup"]["split"] = split
                self.assertEqual(convert_record(record, self.root), expected)

    def test_explicit_duplicate_exception(self):
        self.record.update(status="rejected", reason="Duplicate source or rendered pixels of other")
        with self.assertRaises(ValueError):
            convert_record(self.record, self.root)
        self.record["warmup"]["duplicateException"] = True
        self.assertEqual(convert_record(self.record, self.root)["id"], "repair")

    def test_unexecuted_malformed_output_stays_masked_context(self):
        self.record["trace"].insert(0, {"completion": {"text": "truncated JSON"}, "error": "truncated"})
        example = convert_record(self.record, self.root)
        self.assertIn("truncated JSON", json.dumps(example["messages"]))
        targets = [example["messages"][i] for i in example["trainable_message_indices"]]
        self.assertNotIn("truncated JSON", json.dumps(targets))

    def test_shell_payload_is_literal_and_heredoc_collision_is_handled(self):
        import subprocess
        code = 'ARTSY_SOURCE\nprint("$HOME `id`")\n'
        command = write_command(code, 42)
        write_only = command.split("\npython /opt/artsy/render.py", 1)[0].replace(
            "/workspace/program.py", str(self.root / "literal.py"))
        subprocess.run(["bash", "-c", write_only], check=True)
        self.assertEqual((self.root / "literal.py").read_text(), code)


@unittest.skipUnless(importlib.util.find_spec("tinker_cookbook"), "Requires training/requirements.txt")
class TinkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from tinker_cookbook import renderers
        from training.sft import MODEL_ID, MODEL_REVISION, RENDERER

        path = ROOT / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION
        if not path.is_dir():
            raise unittest.SkipTest("Requires a cached Qwen3.5 tokenizer; no automatic download")
        # Use the same model metadata as inference and training, without weights
        # or a hosted training session.
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cls.renderer = renderers.get_renderer(RENDERER, cls.tokenizer)

    def example(self):
        from training.data import TOOLS, action, message
        return {"id": "repair", "tools": TOOLS, "trainable_message_indices": [3, 5],
                "messages": [message("system", "Draw using ArtCanvas."), message("user", "Draw a circle."),
                             action("broken()", "bad"), action("printf fixed", "fixed"),
                             message("tool", "Execution succeeded", tool_call_id="fixed"),
                             action(SUBMIT, "submit")]}

    def targets(self, datum):
        return self.tokenizer.decode([token for token, weight in zip(
            datum.loss_fn_inputs["target_tokens"].data, datum.loss_fn_inputs["weights"].data) if weight])

    def test_rendering_preserves_native_tool_schema_and_chat_text(self):
        from tinker_cookbook.renderers import TrainOnWhat
        from training.sft import DrawingEncoder
        example = self.example()
        messages, selected = DrawingEncoder(self.renderer).messages(example)
        for index in selected:
            model_input, _ = self.renderer.build_supervised_example(
                messages[:index + 1], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            native = self.tokenizer.apply_chat_template(
                example["messages"][:index + 1], tools=example["tools"], tokenize=False,
                enable_thinking=False, add_generation_prompt=False)
            # Cookbook ends at im_end; HF also emits a trailing separator newline.
            self.assertEqual(self.tokenizer.decode(model_input.to_ints()), native.removesuffix("\n"))

    def test_real_renderer_has_only_text_chunks_and_masks_context(self):
        import tinker
        from training.sft import DrawingEncoder
        example = self.example()
        original = copy.deepcopy(example)
        encoder = DrawingEncoder(self.renderer)
        data = encoder(example)
        self.assertEqual(example, original)
        self.assertEqual(len(data), 2)
        self.assertIn("printf fixed", self.targets(data[0]))
        self.assertNotIn(SUBMIT, self.targets(data[0]))
        self.assertIn(SUBMIT, self.targets(data[1]))
        self.assertNotIn("printf fixed", self.targets(data[1]))
        for datum in data:
            selected = self.targets(datum)
            for excluded in ("broken()", "Execution succeeded", "<|im_start|>", "<|image_pad|>", "<think>"):
                self.assertNotIn(excluded, selected)
            self.assertEqual(datum.model_input.length, len(datum.loss_fn_inputs["target_tokens"].data))
            self.assertEqual(datum.model_input.length, len(datum.loss_fn_inputs["weights"].data))
            self.assertTrue(all(isinstance(chunk, tinker.types.EncodedTextChunk) for chunk in datum.model_input.chunks))
        context = self.tokenizer.decode(data[1].model_input.to_ints())
        self.assertIn("broken()", context)
        self.assertIn("Execution succeeded", context)
        self.assertIn('"name": "bash"', context)
        self.assertIn("<think>\n\n</think>", context)
        self.assertNotIn("<|vision_start|>", context)
        self.assertNotIn("<|image_pad|>", context)
        with self.assertRaisesRegex(ValueError, "never truncated"):
            DrawingEncoder(self.renderer, max_length=5)(example)

    def test_rejects_invalid_targets_and_nontext_content(self):
        from training.sft import DrawingEncoder
        encoder = DrawingEncoder(self.renderer)
        for indices in ([], [3, 3], [1], [True], [-1], [99]):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "invalid trainable"):
                encoder({**self.example(), "trainable_message_indices": indices})
        for index in (0, 1, 4, 5):
            example = self.example()
            example["messages"][index]["content"].append({"type": "image", "image": "/unused/preview.png"})
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "only text content"):
                encoder(example)

@unittest.skipUnless(importlib.util.find_spec("tinker"), "Requires training/requirements.txt")
class TinkerAPITests(unittest.TestCase):
    def test_training_api_normalization_checkpoints_and_exact_resume_order(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        import tinker
        from training.sft import MODEL_ID, parser, train

        data = [tinker.Datum(model_input=tinker.ModelInput.from_ints([10, token]),
                             loss_fn_inputs={"target_tokens": tinker.TensorData(data=[token, 12], dtype="int64", shape=[2]),
                                             "weights": tinker.TensorData(data=[0.0, 1.0], dtype="float32", shape=[2])})
                for token in (11, 21)]
        manifest = {"dataset_sha256": "dataset", "api_doc_sha256": "docs",
                    "conversion": "bash-conversation-text-only-selective-loss-v4"}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            args = parser().parse_args(["--dataset", "accepted.jsonl", "--output-dir", temp,
                                       "--batch-size", "1", "--max-steps", "1"])
            client = MagicMock()
            client.forward_backward.return_value.result.return_value = SimpleNamespace(metrics={"loss:sum": 1.25})
            client.save_state.return_value.result.return_value = SimpleNamespace(path="tinker://run/weights/state")
            client.save_weights_for_sampler.return_value.result.return_value = SimpleNamespace(path="tinker://run/sampler_weights/final")
            service = MagicMock()
            service.get_server_capabilities.return_value = SimpleNamespace(supported_models=[SimpleNamespace(model_name=MODEL_ID)])
            service.create_lora_training_client.return_value = client
            service.create_training_client_from_state_with_optimizer.return_value = client
            with patch("tinker.ServiceClient", return_value=service), contextlib.redirect_stdout(io.StringIO()):
                train(args, data, dict(manifest))
                service.create_lora_training_client.assert_called_once_with(base_model=MODEL_ID, rank=32, seed=42)
                self.assertEqual(client.optim_step.call_count, 1)
                forwarded, loss = client.forward_backward.call_args.args
                self.assertEqual(loss, "cross_entropy")
                self.assertAlmostEqual(sum(sum(d.loss_fn_inputs["weights"].data) for d in forwarded), 1)
                first_input = forwarded[0].model_input
                checkpoint = output / "checkpoint-000001.json"
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(json.loads((output / "final.json").read_text())["sampler_path"],
                                 "tinker://run/sampler_weights/final")
                args.resume_from_checkpoint = checkpoint
                args.max_steps = 2
                legacy = json.loads(checkpoint.read_text())
                legacy["config"]["conversion"] = "bash-conversation-selective-loss-v2"
                legacy_path = output / "legacy-checkpoint.json"
                legacy_path.write_text(json.dumps(legacy))
                args.resume_from_checkpoint = legacy_path
                with self.assertRaisesRegex(ValueError, "configuration does not match"):
                    train(args, data, dict(manifest))
                service.create_training_client_from_state_with_optimizer.assert_not_called()
                args.resume_from_checkpoint = checkpoint
                train(args, data, dict(manifest))
                service.create_training_client_from_state_with_optimizer.assert_called_once_with("tinker://run/weights/state")
                self.assertEqual(client.optim_step.call_count, 2)
                second_input = client.forward_backward.call_args.args[0][0].model_input
                self.assertNotEqual(first_input, second_input)
                self.assertEqual(json.loads((output / "checkpoint-000002.json").read_text())["step"], 2)
                args.learning_rate *= 2
                with self.assertRaisesRegex(ValueError, "configuration does not match"):
                    train(args, data, dict(manifest))
                self.assertEqual(client.optim_step.call_count, 2)


class TrainingCLITests(unittest.TestCase):
    def test_preflight_does_not_create_a_service_client(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        from training.sft import main
        from training.data import message

        example = {"id": "sample", "messages": [message("system", "Instructions")], "tools": [],
                   "trainable_message_indices": []}
        manifest = {"unreviewed_records": 1}
        datum = SimpleNamespace(model_input=SimpleNamespace(length=10),
                                loss_fn_inputs={"weights": SimpleNamespace(data=[0, 1])})
        with patch("training.sft.load_examples", return_value=([example], manifest)), \
             patch("training.sft.create_encoder", return_value=lambda e: [datum]) as encoder, \
             patch("training.sft.train") as train, contextlib.redirect_stdout(io.StringIO()):
            main(["--dataset", "accepted.jsonl", "--dry-run"])
            encoder.assert_not_called()
            train.assert_not_called()
            main(["--dataset", "accepted.jsonl", "--check-batch"])
            encoder.assert_called_once()
            train.assert_not_called()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(["--dataset", "accepted.jsonl"])
            train.assert_not_called()

    def test_batch_order_is_deterministic_across_epochs(self):
        from training.sft import batches
        first = list(batches(list(range(10)), 3, 2, 42))
        self.assertEqual(first, list(batches(list(range(10)), 3, 2, 42)))
        for epoch in (first[:4], first[4:]):
            self.assertEqual(sorted(item for batch in epoch for item in batch), list(range(10)))
        self.assertNotEqual(first[:4], first[4:])


if __name__ == "__main__":
    unittest.main()
