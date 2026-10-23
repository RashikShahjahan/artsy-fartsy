"""TRL SFT for the Qwen3.5 vision-language drawing policy.

Run from the repository root: python -m training.sft --dry-run
"""

import argparse
import copy
import json
from pathlib import Path

from training.data import ROOT, load_examples

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
IMAGE_SIZE = {"shortest_edge": 256 * 256, "longest_edge": 1024 * 1024}


class DrawingCollator:
    """Process whole conversations and supervise only explicitly selected actions."""

    def __init__(self, processor, max_length: int = 32768):
        self.processor = processor
        self.max_length = max_length
        self.chat_template = selective_loss_template(processor.tokenizer.chat_template)

    def render(self, example: dict) -> dict:
        from PIL import Image

        messages = copy.deepcopy(example["messages"])
        selected = example["trainable_message_indices"]
        if not selected or len(set(selected)) != len(selected) or any(
            type(i) is not int or not 0 <= i < len(messages) or messages[i]["role"] != "assistant"
            for i in selected
        ):
            raise ValueError(f"{example['id']}: invalid trainable assistant message indices")
        for index, msg in enumerate(messages):
            msg["train"] = index in selected
            for part in msg["content"]:
                if part["type"] == "image":
                    if msg["train"]:
                        raise ValueError("Image observations cannot be supervised targets")
                    with Image.open(part["image"]) as image:
                        if image.format != "PNG" or image.size != (1920, 1200):
                            raise ValueError(f"Invalid preview image: {part['image']}")
                        part["image"] = image.convert("RGB")

        template = self.processor.tokenizer.apply_chat_template
        kwargs = {"tools": example["tools"], "tokenize": False,
                  "enable_thinking": False, "add_generation_prompt": False}
        native = template(messages, **kwargs)
        rendered = template(messages, chat_template=self.chat_template, **kwargs)
        if rendered != native:
            raise ValueError(f"{example['id']}: loss annotations change the native chat template output")
        return {"messages": messages, "text": native}

    def __call__(self, features: list[dict]) -> dict:
        examples = [json.loads(f["payload"]) for f in features]
        if any(e["tools"] != examples[0]["tools"] for e in examples):
            raise ValueError("All conversations in a batch must use the same tool schema")
        rendered = [self.render(example) for example in examples]
        batch = self.processor.apply_chat_template(
            [r["messages"] for r in rendered], chat_template=self.chat_template,
            tools=examples[0]["tools"], enable_thinking=False, add_generation_prompt=False,
            tokenize=True, return_dict=True, return_tensors="pt", return_assistant_tokens_mask=True,
            processor_kwargs={"padding": True, "padding_side": "right", "truncation": False,
                              "add_special_tokens": False},
        )
        labels = batch["input_ids"].clone()
        labels[(batch.pop("assistant_masks") == 0) | (batch["attention_mask"] == 0)] = -100
        batch["labels"] = labels
        lengths = batch["attention_mask"].sum(dim=1)
        if (lengths > self.max_length).any():
            raise ValueError(f"Sequence exceeds {self.max_length} tokens: {lengths.tolist()}. "
                             "Use a larger supported --max-length; images/targets are never truncated.")
        if (batch["labels"] != -100).sum(dim=1).eq(0).any():
            raise ValueError("An example has no supervised tokens")
        return batch


def selective_loss_template(template: str) -> str:
    """Add generation spans to selected Qwen assistant bodies without changing text.

    Assistant headers and the disabled-thinking prefix stay outside the spans.
    Transformers carries the span offsets through image-token expansion, avoiding
    text-only token counts that would misalign masks after a rendered preview.
    Fail explicitly if a different template cannot be annotated safely.
    """
    start = '{%- elif message.role == "assistant" %}'
    end = '{%- elif message.role == "tool" %}'
    if template.count(start) != 1 or template.count(end) != 1:
        raise ValueError("Unsupported Qwen chat template: assistant/tool branches not found")
    before, assistant = template.split(start)
    assistant, after = assistant.split(end)
    headers = (
        "'\\n</think>\\n\\n' + content }}",
        "'\\n' + content }}",
    )
    for header in headers:
        if assistant.count(header) != 1:
            raise ValueError("Unsupported Qwen chat template: assistant header changed")
        assistant = assistant.replace(header, header.replace(" + content", ""))
    boundary = "{%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}"
    if assistant.count(boundary) != 1:
        raise ValueError("Unsupported Qwen chat template: tool-call rendering changed")
    header, body = assistant.split(boundary)
    body = "{{- content }}" + boundary + body
    assistant = (header + '{%- if message.train | default(false) %}{% generation %}'
                 + body + '{%- endgeneration %}{%- else %}' + body + '{%- endif %}')
    return before + start + assistant + end + after


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets/sft-warmup-100/train.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/sft-warmup")
    p.add_argument("--model", default=MODEL_ID, help="Qwen3.5 model ID or local Qwen3.5 checkpoint")
    p.add_argument("--revision", help="Defaults to the inference revision for the default base model")
    p.add_argument("--dry-run", action="store_true", help="Validate artifacts and print a converted example; no model load")
    p.add_argument("--export", type=Path, help="Write the converted conversation JSONL with trainable message indices")
    p.add_argument("--sample-id", help="Record ID to display in dry-run")
    p.add_argument("--allow-unreviewed", action="store_true", help="Include selected records whose visual review is pending")
    p.add_argument("--check-batch", action="store_true", help="Check every example's tokens/labels/images without loading weights")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume-from-checkpoint", type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.epochs <= 0 or args.max_steps == 0 or args.max_steps < -1 or args.batch_size < 1
        or args.gradient_accumulation_steps < 1 or args.learning_rate <= 0 or args.max_length < 1):
        p.error("epochs, batch sizes, learning rate, and max length must be positive; max steps is -1 or positive")
    examples, manifest = load_examples(args.dataset)
    if args.export:
        if args.export.resolve() == args.dataset.resolve():
            p.error("Export path must differ from the source dataset")
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("w") as out:
            for example in examples:
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
    if args.dry_run:
        selected = next((e for e in examples if not args.sample_id or e["id"] == args.sample_id), None)
        if selected is None:
            p.error(f"Sample ID not found: {args.sample_id}")
        shown = copy.deepcopy(selected)
        shown["messages"][0]["content"][0]["text"] = "[System instructions and full docs/artcanvas.md omitted from preview]"
        print(json.dumps({"summary": manifest, "sample": shown}, indent=2, ensure_ascii=False))
        return
    if manifest["unreviewed_records"] and not args.allow_unreviewed and not args.check_batch:
        p.error(f"{manifest['unreviewed_records']} selected records are pending visual review. "
                "Review them or use --allow-unreviewed for an experimental run.")
    if not args.check_batch and not args.resume_from_checkpoint and args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Output directory is not empty; choose a fresh directory or --resume-from-checkpoint")

    import torch
    from transformers import AutoProcessor

    revision = args.revision or (MODEL_REVISION if args.model == MODEL_ID else None)
    processor = AutoProcessor.from_pretrained(args.model, revision=revision, size=IMAGE_SIZE)
    collator = DrawingCollator(processor, args.max_length)
    # The images column identifies a vision dataset to SFTTrainer. Actual PNGs
    # are loaded lazily from the serialized conversation by DrawingCollator.
    payloads = [{"payload": json.dumps(example), "images": []} for example in examples]
    # Check every row before allocating model weights or starting optimization.
    # This catches long repair histories, template drift, and broken PNGs early.
    max_tokens = max_supervised_tokens = supervised = 0
    for payload in payloads:
        batch = collator([payload])
        max_tokens = max(max_tokens, int(batch["attention_mask"].sum()))
        target_tokens = int((batch["labels"] != -100).sum())
        max_supervised_tokens = max(max_supervised_tokens, target_tokens)
        supervised += target_tokens
    manifest.update(max_tokens=max_tokens, max_supervised_tokens=max_supervised_tokens, supervised_tokens=supervised,
                    model=args.model, revision=revision, image_size=IMAGE_SIZE)
    print(json.dumps(manifest, indent=2))
    if args.check_batch:
        return

    from datasets import Dataset
    from transformers import Qwen3_5ForConditionalGeneration, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    use_cpu = args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    bf16 = args.dtype == "bfloat16" or (args.dtype == "auto" and not use_cpu and torch.cuda.is_bf16_supported())
    if bf16 and (use_cpu or not torch.cuda.is_bf16_supported()):
        p.error("bfloat16 training requires a CUDA device with bfloat16 support; use --dtype float32")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, revision=revision, dtype=torch.bfloat16 if bf16 else torch.float32,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    config = SFTConfig(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_steps=0.05, lr_scheduler_type="cosine",
        weight_decay=0.01, max_grad_norm=1.0, seed=args.seed, data_seed=args.seed,
        bf16=bf16, fp16=False, use_cpu=use_cpu, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None, packing=False, completion_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
        logging_steps=1, save_strategy="epoch", save_total_limit=2,
        report_to="none", dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, args=config, processing_class=processor,
                         train_dataset=Dataset.from_list(payloads), data_collator=collator)
    if trainer.is_world_process_zero():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        from importlib.metadata import version
        manifest.update(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        packages={name: version(name) for name in ("torch", "transformers", "trl", "datasets", "accelerate")})
        (args.output_dir / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir / "final"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir / "final")
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
