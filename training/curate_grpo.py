"""Select 200 diverse GRPO prompts with a 10% refusal mix, without model calls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from drawing.dataset_specs import DIFFICULTIES
from drawing.eval_specs import load_specs

ROOT = Path(__file__).resolve().parents[1]


def normalized(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def curate(rows: list[dict], seed: int = 42) -> list[dict]:
    """Enforce difficulty/category quotas; greedily spread skills and motifs."""
    rng = random.Random(seed)
    pools = {}
    for category, per_difficulty in (("drawing", 45), ("harmful_image", 3),
                                     ("malicious_code", 2)):
        for difficulty in DIFFICULTIES:
            pool = [r for r in rows if r["difficulty"] == difficulty and
                    (r.get("safetyCategory", "harmful_image")
                     if r.get("kind") == "refusal" else "drawing") == category]
            rng.shuffle(pool)
            if len(pool) < per_difficulty:
                raise ValueError(f"Insufficient unique prompts for {category}/{difficulty}")
            pools[category, difficulty] = (pool, per_difficulty)

    selected = []
    counts = Counter()
    tokens = {r["id"]: set(re.findall(r"\w+", normalized(r["prompt"]))) for r in rows}
    selected_tokens = {category: [] for category, _ in pools}
    diversity_fields = ("family", "contentTheme", "format", "artStyle", "voice")
    while any(quota for _, quota in pools.values()):
        # Round-robin difficulty avoids giving one difficulty first pick of skills.
        for (category, difficulty), (pool, quota) in pools.items():
            if not quota:
                continue

            def score(row):
                words = tokens[row["id"]]
                similarity = max((len(words & other) / len(words | other)
                                  for other in selected_tokens[category]), default=0)
                return (counts[category, "primarySkill", row.get("primarySkill")],
                        counts[category, difficulty, row.get("primarySkill")],
                        sum(counts[category, field, row.get(field)]
                            for field in diversity_fields) + 10 * similarity)

            choice = min(pool, key=score)
            pool.remove(choice)
            pools[category, difficulty] = (pool, quota - 1)
            selected.append(choice)
            selected_tokens[category].append(tokens[choice["id"]])
            for field in (*diversity_fields, "primarySkill"):
                counts[category, field, choice.get(field)] += 1
            counts[category, difficulty, choice.get("primarySkill")] += 1
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/grpo-v2/prompts.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/grpo-v2-subset-200")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = load_specs(args.source)
    seen = set()
    unique = []
    for row in source:
        key = normalized(row["prompt"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    selected = curate(unique, args.seed)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row in selected)
    breakdown = {}
    for category in ("drawing", "harmful_image", "malicious_code"):
        group = [r for r in selected if
                 (r.get("safetyCategory", "harmful_image")
                  if r.get("kind") == "refusal" else "drawing") == category]
        breakdown[category] = {"count": len(group), **{
            field: dict(sorted(Counter(r.get(field, "missing") for r in group).items()))
            for field in ("difficulty", "primarySkill", "family", "contentTheme", "format")}}
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_count": len(source), "unique_source_count": len(unique),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "count": len(selected), "refusal_count": 20, "refusal_fraction": 0.1,
        "seed": args.seed,
        "method": "Exact difficulty/category quotas; greedy primary-skill balance, then "
                  "family/theme/format/style/voice diversity and token-Jaccard dissimilarity. "
                  "Original records preserved; deterministic shuffled output. No manual semantic review.",
        "curator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "breakdown": breakdown, "selected_ids": [r["id"] for r in selected],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prompts.jsonl").write_text(payload, encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(selected),
                      "refusals": 20, "breakdown": breakdown}, indent=2))


if __name__ == "__main__":
    main()
