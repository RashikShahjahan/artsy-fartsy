"""Dataset integrity and loss-mask checks for the training adapter."""

import copy
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from training.data import ROOT, SUBMIT, convert_record, digest, write_command


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        code = 'from artcanvas import ArtCanvas\nprint("$HOME `id`")\n'
        (self.root / "program.py").write_text(code)
        for name in ("preview-1.png", "preview-2.png", "image.png"):
            Image.new("RGB", (1920, 1200), "white").save(self.root / name)
        self.record = {
            "id": "repair", "spec": {"prompt": "Draw a circle", "seed": 42, "split": "test"},
            "warmup": {"split": "train", "kind": "repair"}, "status": "review",
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
        self.assertEqual(example["messages"][-2]["content"][-1]["image"], str(self.root / "preview-2.png"))

    def test_visual_repair_uses_png_without_grader_answers(self):
        self.record["trace"][0]["feedback"] = {"ok": True, "artifact": "/old/preview-1.png"}
        self.record["trace"][1]["observation"] = {
            "image": "preview-1.png", "imageHash": digest((self.root / "preview-1.png").read_bytes()),
            "text": "privileged judge scores and answer",
        }
        self.record["trace"][0]["evaluation"] = {"reason": "secret answer", "score": 0.2}
        example = convert_record(self.record, self.root)
        serialized = json.dumps(example["messages"])
        self.assertIn("preview-1.png", serialized)
        self.assertNotIn("secret answer", serialized)
        self.assertNotIn("privileged", serialized)

    def test_rejects_modified_source_observation_and_holdout(self):
        for change in ("source", "observation", "holdout", "synthetic", "excluded"):
            with self.subTest(change=change):
                record = copy.deepcopy(self.record)
                if change == "source":
                    record["trace"][-1]["executedCode"] = "tampered()"
                elif change == "observation":
                    record["trace"][-1]["observation"] = {"image": "preview-1.png", "imageHash": "bad"}
                elif change == "holdout":
                    record["warmup"]["split"] = "test"
                elif change == "synthetic":
                    record["generation"] = {"initialDraft": "injected"}
                else:
                    record["selectionAudit"] = {"eligible": False}
                with self.assertRaises(ValueError):
                    convert_record(record, self.root)

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


@unittest.skipUnless(importlib.util.find_spec("trl"), "Requires training/requirements.txt")
class CollatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        from training.sft import MODEL_REVISION, IMAGE_SIZE

        path = ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots" / MODEL_REVISION
        if not path.is_dir():
            raise unittest.SkipTest("Requires the cached Qwen3.5 processor; no automatic download")
        cls.processor = AutoProcessor.from_pretrained(path, local_files_only=True, size=IMAGE_SIZE)

    def test_real_qwen_selective_masks_across_turns_images_and_padding(self):
        import torch
        from training.sft import DrawingCollator
        from training.data import action, message

        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "preview.png"
            Image.new("RGB", (1920, 1200), "white").save(image)
            text_example = {"id": "text", "tools": [], "trainable_message_indices": [3, 5],
                            "messages": [message("user", "Draw a circle"),
                                         action("broken()", "call_bad"),
                                         message("tool", "Execution failed."),
                                         action("printf fixed", "call_fixed"),
                                         message("tool", "Execution succeeded."),
                                         action(SUBMIT, "call_submit")]}
            image_example = copy.deepcopy(text_example)
            image_example["id"] = "image"
            image_example["messages"][2] = message("tool", "Execution failed. Inspect this image.", image)
            image_example["messages"][4] = message("tool", "Execution succeeded. Inspect this image.", image)
            collator = DrawingCollator(self.processor)
            batch = collator([{"payload": json.dumps(e)} for e in (text_example, image_example)])
            self.assertIn("pixel_values", batch)
            self.assertEqual(batch["input_ids"].shape[0], 2)
            for i, example in enumerate((text_example, image_example)):
                rendered = collator.render(example)
                tokenized = self.processor.tokenizer.apply_chat_template(
                    rendered["messages"], chat_template=collator.chat_template,
                    tools=example["tools"], tokenize=True, return_dict=True,
                    return_assistant_tokens_mask=True, enable_thinking=False, add_generation_prompt=False)
                expected = [token for token, mask in zip(tokenized["input_ids"], tokenized["assistant_masks"]) if mask]
                actual = batch["labels"][i][batch["labels"][i] != -100].tolist()
                self.assertEqual(actual, expected)
                decoded = self.processor.tokenizer.decode(actual)
                self.assertIn("printf fixed", decoded)
                self.assertIn(SUBMIT, decoded)
                self.assertNotIn("broken()", decoded)
                self.assertNotIn("Execution", decoded)
                self.assertNotIn("<|im_start|>", decoded)
                self.assertTrue(torch.all(batch["labels"][i][batch["attention_mask"][i] == 0] == -100))
            image_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            self.assertTrue(torch.all(batch["labels"][batch["input_ids"] == image_id] == -100))
            with self.assertRaisesRegex(ValueError, "never truncated"):
                DrawingCollator(self.processor, max_length=5)([{"payload": json.dumps(image_example)}])

    def test_tiny_qwen_optimizer_step_with_real_processor(self):
        import torch
        from datasets import Dataset
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from trl import SFTConfig, SFTTrainer
        from training.sft import DrawingCollator
        from training.data import message

        torch.set_num_threads(2)
        config = Qwen3_5Config(
            text_config={"vocab_size": len(self.processor.tokenizer), "hidden_size": 32,
                         "intermediate_size": 64, "num_hidden_layers": 1,
                         "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 32,
                         "layer_types": ["full_attention"], "use_cache": False,
                         "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                             "mrope_section": [4, 4, 8], "partial_rotary_factor": 1.0}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 1, "out_hidden_size": 32, "num_position_embeddings": 16},
        )
        model = Qwen3_5ForConditionalGeneration(config)
        example = {"id": "smoke", "tools": [], "trainable_message_indices": [1],
                   "messages": [message("user", "Hi"), message("assistant", "Hello")]}
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "preview.png"
            Image.new("RGB", (1920, 1200), "blue").save(image)
            vision_example = copy.deepcopy(example)
            vision_example["messages"][0] = message("user", "Describe this", image)
            processor = copy.deepcopy(self.processor)
            processor.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": 32 * 32}
            dataset = Dataset.from_list([{"payload": json.dumps(e), "images": []}
                                         for e in (example, vision_example)])
            trainer = SFTTrainer(
                model=model, processing_class=processor, train_dataset=dataset,
                data_collator=DrawingCollator(processor),
                args=SFTConfig(output_dir=temp, max_steps=1, per_device_train_batch_size=2,
                               use_cpu=True, bf16=False, report_to="none", save_strategy="no",
                               gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                               max_length=None, packing=False, completion_only_loss=False,
                               dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False),
            )
            before = model.get_input_embeddings().weight.detach().clone()
            vision_before = model.model.visual.patch_embed.proj.weight.detach().clone()
            result = trainer.train()
            self.assertEqual(result.global_step, 1)
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
            self.assertFalse(torch.equal(before, model.get_input_embeddings().weight))
            self.assertFalse(torch.equal(vision_before, model.model.visual.patch_embed.proj.weight))


if __name__ == "__main__":
    unittest.main()
