"""Tinker LoRA SFT for Qwen vision-language drawing policies.

Run from the repository root: python -m training.sft --dataset DATA/accepted.jsonl --dry-run
"""

import argparse
import copy
import io
import json
import math
from pathlib import Path
import random

from training.data import LOSS_POLICIES, LOSS_POLICY, ROOT, digest, load_examples
from drawing.config import IMAGE_SIZE, MODEL_ID, MODEL_REVISION, MODEL_CONTEXT_WINDOW

RENDERER = "qwen3_5_disable_thinking"


def renderer_for_model(model: str) -> str:
    if model.split(":", 1)[0] == "Qwen/Qwen3.8-27B":
        return "qwen3_8_disable_thinking"
    return RENDERER


class DrawingEncoder:
    """Encode a complete trajectory once, supervising only selected assistant turns."""

    def __init__(self, renderer, max_length: int = MODEL_CONTEXT_WINDOW, image_processor=None):
        self.renderer = renderer
        self.max_length = max_length
        self.image_processor = image_processor

    def preview(self, part: dict):
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        if self.image_processor is None:
            raise ValueError("Preview images require an image processor")
        path = Path(part["image"])
        if not path.is_absolute():
            raise ValueError("Preview images must use absolute local artifact paths")
        data = path.read_bytes()
        if digest(data) != part.get("imageHash"):
            raise ValueError(f"Preview image hash mismatch: {path}")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or len(data) > 8 * 1024 * 1024 or not (
                1 <= image.width <= 4096 and 1 <= image.height <= 4096
            ):
                raise ValueError(f"Invalid preview PNG: {path}")
            # Match inference's RGB conversion and bounded Qwen image resolution.
            height, width = smart_resize(
                image.height, image.width,
                factor=self.image_processor.patch_size * self.image_processor.merge_size,
                min_pixels=IMAGE_SIZE["shortest_edge"], max_pixels=IMAGE_SIZE["longest_edge"])
            return {"type": "image", "image": image.convert("RGB").resize((width, height))}

    def messages(self, example: dict) -> tuple[list[dict], list[int]]:
        from tinker_cookbook.renderers import ToolCall

        messages = copy.deepcopy(example["messages"])
        selected = example["trainable_message_indices"]
        if not selected or any(type(i) is not int for i in selected) or len(set(selected)) != len(selected) or any(
            not 0 <= i < len(messages) or messages[i]["role"] != "assistant" for i in selected
        ):
            raise ValueError(f"{example['id']}: invalid trainable assistant message indices")
        for msg in messages:
            parts = []
            for part in msg["content"]:
                if part["type"] == "image" and msg["role"] == "tool":
                    parts.append(self.preview(part))
                elif part["type"] == "text":
                    parts.append(part)
                elif part["type"] == "thinking" and msg["role"] == "assistant" and isinstance(part.get("thinking"), str):
                    parts.append({"type": "text", "text": part["thinking"]})
                else:
                    raise ValueError(f"{example['id']}: only text, assistant thinking, and tool preview images are supported")
            msg["content"] = parts
            if "tool_calls" in msg:
                for call in msg["tool_calls"]:
                    arguments = call["function"]["arguments"]
                    if not isinstance(arguments, str):
                        call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
                msg["tool_calls"] = [ToolCall.model_validate(call) for call in msg["tool_calls"]]
        if messages[0]["role"] != "system":
            raise ValueError(f"{example['id']}: expected leading system instructions")
        system = "".join(part["text"] for part in messages[0]["content"] if part["type"] == "text")
        # Match inference: the Qwen renderer expects flat ToolSpec objects,
        # while exported demonstrations use OpenAI function wrappers.
        tools = [tool.get("function", tool) for tool in example["tools"]]
        prefix = self.renderer.create_conversation_prefix_with_tools(tools, system_prompt=system)
        return prefix + messages[1:], [i + len(prefix) - 1 for i in selected]

    def __call__(self, example: dict) -> list:
        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        messages, selected = self.messages(example)
        selected = set(selected)
        for index, msg in enumerate(messages):
            msg["trainable"] = index in selected
        # Keep masked assistant turns in the causal context. Headers and
        # observations are also masked. Render the full trajectory only once.
        model_input, weights = self.renderer.build_supervised_example(
            messages, train_on_what=TrainOnWhat.CUSTOMIZED,
        )
        if model_input.length > self.max_length:
            raise ValueError(f"{example['id']}: sequence exceeds {self.max_length} tokens "
                             f"({model_input.length}). Context/targets are never truncated.")
        datum = datum_from_model_input_weights(model_input, weights)
        if not any(datum.loss_fn_inputs["weights"].data):
            raise ValueError(f"{example['id']}: trajectory has no supervised tokens")
        return [datum]


def create_encoder(model: str, max_length: int, renderer_name: str | None = None,
                   revision: str | None = None) -> DrawingEncoder:
    from transformers import AutoImageProcessor, AutoTokenizer
    from tinker_cookbook import renderers

    # Only tokenizer/processor metadata is local; model weights stay on Tinker.
    if revision is None and model == MODEL_ID:
        revision = MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    image_processor = AutoImageProcessor.from_pretrained(model, revision=revision)
    renderer = renderers.get_renderer(renderer_name or renderer_for_model(model), tokenizer,
                                      image_processor=image_processor)
    return DrawingEncoder(renderer, max_length, image_processor=image_processor)


def batches(data: list, batch_size: int, epochs: int, seed: int):
    """Deterministic ordering lets a checkpoint resume at the next optimizer step."""
    for epoch in range(epochs):
        order = list(range(len(data)))
        random.Random(seed + epoch).shuffle(order)
        for offset in range(0, len(order), batch_size):
            yield [data[i] for i in order[offset:offset + batch_size]]


def save_checkpoint(client, output_dir: Path, step: int, config: dict) -> dict:
    checkpoint = {"step": step, "config": config,
                  "state_path": client.save_state(f"step-{step:06d}").result().path}
    path = output_dir / f"checkpoint-{step:06d}.json"
    path.write_text(json.dumps(checkpoint, indent=2) + "\n")
    return checkpoint


def train(args, data: list, manifest: dict):
    import tinker
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    total_steps = math.ceil(len(data) / args.batch_size) * args.epochs
    if args.max_steps != -1:
        total_steps = min(total_steps, args.max_steps)
    config = {"dataset_sha256": manifest["dataset_sha256"], "api_doc_sha256": manifest["api_doc_sha256"],
              "conversion": manifest["conversion"], "renderer": renderer_for_model(args.model), "input_mode": manifest["input_mode"],
              "model": args.model, "tokenizer_revision": MODEL_REVISION if args.model.split(":", 1)[0] == MODEL_ID else None,
              "lora_rank": args.lora_rank, "batch_size": args.batch_size,
              "batch_unit": "trajectory", "image_size": IMAGE_SIZE,
              "loss_policy": manifest["loss_policy"],
              "learning_rate": args.learning_rate, "lr_schedule": args.lr_schedule,
              "seed": args.seed, "max_length": args.max_length,
              "training_examples": len(data)}
    if args.lr_schedule != "constant":
        # Resuming must not change the decay curve for already-completed steps.
        config["schedule_total_steps"] = total_steps
    start_step = 0
    checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = json.loads(args.resume_from_checkpoint.read_text())
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint dataset/model/training configuration does not match this run")
        start_step = checkpoint["step"]
    if not 0 <= start_step < total_steps:
        raise ValueError("No remaining optimizer steps; increase --epochs or --max-steps")

    service = tinker.ServiceClient()
    supported = {model.model_name for model in service.get_server_capabilities().supported_models}
    if args.model not in supported:
        raise ValueError(f"Tinker does not offer {args.model}; choose a supported Qwen vision model with --model")
    if checkpoint:
        client = service.create_training_client_from_state_with_optimizer(checkpoint["state_path"])
    else:
        client = service.create_lora_training_client(base_model=args.model, rank=args.lora_rank, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version
    manifest.update(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    packages={name: version(name) for name in ("tinker", "tinker-cookbook", "transformers")},
                    config=config, total_steps=total_steps)
    (args.output_dir / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output_dir / "metrics.jsonl").open("a") as metrics:
        for step, batch in enumerate(batches(data, args.batch_size, args.epochs, args.seed), 1):
            if step <= start_step:
                continue
            if step > total_steps:
                break
            # cross_entropy sums token losses. Normalize across the entire batch
            # so gradient magnitude does not grow with program/preview length.
            token_count = sum(sum(d.loss_fn_inputs["weights"].data) for d in batch)
            normalized = []
            for datum in batch:
                inputs = dict(datum.loss_fn_inputs)
                inputs["weights"] = tinker.TensorData(
                    data=[w / token_count for w in inputs["weights"].data], dtype="float32",
                    shape=inputs["weights"].shape,
                )
                normalized.append(tinker.Datum(model_input=datum.model_input, loss_fn_inputs=inputs))
            learning_rate = args.learning_rate * compute_schedule_lr_multiplier(
                args.lr_schedule, step=step - 1, total_steps=total_steps)
            result = client.forward_backward(normalized, "cross_entropy").result()
            client.optim_step(tinker.AdamParams(learning_rate=learning_rate)).result()
            row = {**result.metrics, "step": step, "supervised_tokens": token_count,
                   "learning_rate": learning_rate, "batch_size": len(batch)}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == total_steps:
                checkpoint = save_checkpoint(client, args.output_dir, step, config)
    sampler_path = client.save_weights_for_sampler("final").result().path
    final = {**checkpoint, "sampler_path": sampler_path}
    (args.output_dir / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True, help="Recorded training dataset JSONL")
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/sft-warmup")
    p.add_argument("--model", default=MODEL_ID, help="Hosted Qwen3.5 or Qwen3.8 vision model ID")
    p.add_argument("--dry-run", action="store_true", help="Validate artifacts and print a converted example; no API calls")
    p.add_argument("--export", type=Path, help="Write converted conversation JSONL with trainable message indices")
    p.add_argument("--sample-id", help="Record ID to display in dry-run")
    p.add_argument("--allow-unreviewed", action="store_true", help="Include selected records whose visual review is pending")
    p.add_argument("--check-batch", action="store_true", help="Check text/image tokens and loss masks locally; no API calls")
    p.add_argument("--loss-policy", choices=LOSS_POLICIES, default=LOSS_POLICY,
                   help="Supervise the final write, final preview and submission (default), or all assistant turns")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=8, help="Complete trajectories per optimizer step")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine",
                   help="Learning-rate schedule; --learning-rate is the initial/peak rate")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-length", type=int, default=MODEL_CONTEXT_WINDOW)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50, help="Save optimizer state every N steps and at the end")
    p.add_argument("--resume-from-checkpoint", type=Path, help="Local checkpoint-NNNNNN.json containing a Tinker state path")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.epochs < 1 or args.max_steps == 0 or args.max_steps < -1 or args.batch_size < 1
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 1 <= args.max_length <= MODEL_CONTEXT_WINDOW
        or args.lora_rank < 1 or args.save_every < 1):
        p.error("epochs, batch size, learning rate, max length, rank and save interval must be positive; max steps is -1 or positive; max length cannot exceed the model context window")
    examples, manifest = load_examples(args.dataset, loss_policy=args.loss_policy)
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
    if not (args.model.startswith("Qwen/Qwen3.5-") or args.model.split(":", 1)[0] == "Qwen/Qwen3.8-27B") or "Base" in args.model:
        p.error("Use a Qwen3.5 or Qwen3.8 vision chat model with a disabled-thinking renderer")

    encoder = create_encoder(args.model.split(":", 1)[0], args.max_length)
    data = [datum for example in examples for datum in encoder(example)]
    manifest.update(training_examples=len(data), max_tokens=max(d.model_input.length + 1 for d in data),
                    supervised_tokens=sum(sum(d.loss_fn_inputs["weights"].data) for d in data),
                    model=args.model, renderer=renderer_for_model(args.model),
                    batch_size=args.batch_size, steps_per_epoch=math.ceil(len(data) / args.batch_size),
                    total_steps=min(math.ceil(len(data) / args.batch_size) * args.epochs,
                                    args.max_steps if args.max_steps != -1 else math.inf))
    print(json.dumps(manifest, indent=2))
    if not args.check_batch:
        train(args, data, manifest)


if __name__ == "__main__":
    main()
