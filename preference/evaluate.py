"""Held-out human agreement and A/B order-consistency of a visual preference judge."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from .data import load_pairs
from .judge import DEFAULT_MODEL, TinkerPairwiseJudge


async def evaluate(args) -> None:
    pairs = [pair for pair in load_pairs(args.data) if pair.split == args.split]
    if not pairs:
        raise ValueError("Held-out split is empty")
    judge = TinkerPairwiseJudge(
        model=args.model, sampler_path=args.sampler_path, renderer=args.renderer)
    labels = ("A", "B", "Tie")
    confusion = {label: {pred: 0 for pred in labels} for label in labels}
    correct = consistent = 0
    for pair in pairs:
        a, b = pair.image_a.read_bytes(), pair.image_b.read_bytes()
        prediction = await judge.compare(a, b)
        reverse = await judge.compare(b, a)
        consistent += prediction == {"A": "B", "B": "A", "Tie": "Tie"}[reverse]
        correct += prediction == pair.winner
        confusion[pair.winner][prediction] += 1
    print(json.dumps({
        "count": len(pairs),
        "agreement": correct / len(pairs),
        "order_consistency": consistent / len(pairs),
        "confusion": confusion,
        "per_class_recall": {
            label: counts[label] / sum(counts.values())
            if sum(counts.values()) else None
            for label, counts in confusion.items()
        },
    }, indent=2))


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sampler-path")
    parser.add_argument("--renderer")
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
