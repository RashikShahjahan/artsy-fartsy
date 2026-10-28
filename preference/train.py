"""Train a Tinker-hosted Qwen VLM on human A/B/Tie labels with LoRA."""
from __future__ import annotations

import argparse
import math
import random

from .data import load_pairs
from .judge import DEFAULT_MODEL


def train(args) -> None:
    pairs = load_pairs(args.data)
    training_pairs = [pair for pair in pairs if pair.split == "train"]
    if not training_pairs:
        raise ValueError("Preference dataset has no training split")
    if args.validate_only:
        print(f"Validated {len(pairs)} pairs; {len(training_pairs)} training pairs")
        return
    if (min(args.epochs, args.batch_size, args.rank, args.max_length) < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Training parameters must be finite and positive")

    import tinker
    from transformers import AutoImageProcessor, AutoTokenizer
    from tinker_cookbook import renderers
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    processor = AutoImageProcessor.from_pretrained(args.model, revision=args.revision)
    renderer = renderers.get_renderer(
        args.renderer or get_recommended_renderer_name(args.model),
        tokenizer, image_processor=processor)

    # Check all examples for length and multimodal serialization before paying for training.
    datums = []
    for pair in training_pairs:
        for reverse in (False, True):
            conversation = pair.conversation(processor, reverse=reverse)
            model_input, _ = renderer.build_supervised_example(
                conversation, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
            if model_input.length > args.max_length:
                raise ValueError(f"{pair.pair_id} exceeds max_length; no truncation permitted")
            datums.append(conversation_to_datum(
                conversation, renderer, args.max_length, TrainOnWhat.ALL_ASSISTANT_MESSAGES))

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=args.model, rank=args.rank)
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(datums)
        for offset in range(0, len(datums), args.batch_size):
            client.forward_backward(
                datums[offset:offset + args.batch_size], loss_fn="cross_entropy").result()
            client.optim_step(tinker.AdamParams(learning_rate=args.learning_rate)).result()
        label = f"{args.name}-epoch-{epoch + 1}"
        client.save_state(label).result()
        checkpoint = client.save_weights_for_sampler(label).result()
        print(f"epoch={epoch + 1} sampler_path={checkpoint.path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--renderer")
    parser.add_argument("--revision")
    parser.add_argument("--name", default="artsy-preference")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
