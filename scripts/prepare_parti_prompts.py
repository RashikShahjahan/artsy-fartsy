"""Prepare disjoint PartiPrompts inputs for Artsy SFT generation, evaluation and RL.

Uses only the standard library. Download the pinned TSV first; see the generated
dataset README for commands. This does not generate SFT answers or call models.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drawing.dataset_specs import validate_specs
from drawing.eval_specs import load_specs

REVISION = "944b156abfdad7627c3221b5ec4f6a6fb060a197"
SOURCE_SHA256 = "fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb"
SOURCE = "https://huggingface.co/datasets/nateraw/parti-prompts"
SPLITS = ("sft", "evaluation", "rl")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def apportion(sizes: dict, target: int) -> dict:
    """Proportional integer allocation using largest remainders and stable ties."""
    total = sum(sizes.values())
    if not 0 <= target <= total:
        raise ValueError("Allocation target exceeds available prompts")
    counts = {key: size * target // total for key, size in sizes.items()}
    ranked = sorted(sizes, key=lambda key: (-(sizes[key] * target % total), key))
    for key in ranked[:target - sum(counts.values())]:
        counts[key] += 1
    return counts


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(source: Path, output: Path, seed: int) -> dict:
    if digest(source.read_bytes()) != SOURCE_SHA256:
        raise ValueError("Source TSV does not match the pinned Hugging Face revision")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["Prompt", "Category", "Challenge", "Note"]:
            raise ValueError(f"Unexpected TSV fields: {reader.fieldnames}")
        raw = list(reader)
    groups = {}
    for index, row in enumerate(raw, 1):
        if any(row.get(key) is None for key in ("Prompt", "Category", "Challenge", "Note")) or None in row:
            raise ValueError(f"Malformed source record {index}")
        key = normalize(row["Prompt"])
        if not key or not row["Category"].strip() or not row["Challenge"].strip():
            raise ValueError(f"Empty prompt/category/challenge at source record {index}")
        if key not in groups:
            prompt_hash = digest(key.encode())
            groups[key] = {
                "id": "parti-" + prompt_hash[:24],
                "family": row["Category"],
                "prompt": row["Prompt"],
                "seed": int(prompt_hash[:8], 16),
                "source": {"dataset": "nateraw/parti-prompts", "revision": REVISION,
                           "rows": [], "category": row["Category"],
                           "challenge": row["Challenge"], "note": row["Note"]},
            }
        groups[key]["source"]["rows"].append(index)
    if len(groups) < 10:
        raise ValueError("Expected at least ten distinct prompts")
    strata = defaultdict(list)
    for row in groups.values():
        strata[(row["source"]["category"], row["source"]["challenge"])].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: digest(f"{seed}:{row['id']}".encode()))
    # Round the overall 70/10/20 quotas by largest remainder.
    targets = {name: len(groups) * weight // 100
               for name, weight in zip(SPLITS, (70, 10, 20))}
    ranked = sorted(zip(SPLITS, (70, 10, 20)),
                    key=lambda item: (-(len(groups) * item[1] % 100), item[0]))
    for name, _ in ranked[:len(groups) - sum(targets.values())]:
        targets[name] += 1
    sizes = {key: len(rows) for key, rows in strata.items()}
    sft_counts = apportion(sizes, targets["sft"])
    remaining = {key: sizes[key] - sft_counts[key] for key in sizes}
    eval_counts = apportion(remaining, targets["evaluation"])
    splits = {name: [] for name in SPLITS}
    for key in sorted(strata):
        rows = strata[key]
        sft_end = sft_counts[key]
        eval_end = sft_end + eval_counts[key]
        for name, subset in zip(SPLITS, (rows[:sft_end], rows[sft_end:eval_end], rows[eval_end:])):
            splits[name].extend(subset)

    # Keep the source order out of the training order as well as the assignment.
    for rows in splits.values():
        rows.sort(key=lambda row: digest(f"{seed}:order:{row['id']}".encode()))
        validate_specs(rows)
    seen = set()
    seen_ids = set()
    for name, rows in splits.items():
        assert len(rows) == targets[name]
        keys = {normalize(row["prompt"]) for row in rows}
        ids = {row["id"] for row in rows}
        assert not seen.intersection(keys) and len(keys) == len(rows)
        assert not seen_ids.intersection(ids) and len(ids) == len(rows)
        seen.update(keys)
        seen_ids.update(ids)
    assert len(seen) == len(groups)

    manifest = {
        "format_version": 1,
        "source": {"url": SOURCE, "revision": REVISION, "license": "Apache-2.0",
                   "file": "raw/PartiPrompts.tsv", "sha256": digest(source.read_bytes()),
                   "records": len(raw)},
        "unique_prompts": len(groups), "duplicates_collapsed": len(raw) - len(groups),
        "split_seed": seed, "requested_percentages": dict(zip(SPLITS, (70, 10, 20))),
        "method": "Largest-remainder SFT allocation per (Category, Challenge), then evaluation allocation from remainder; RL receives remainder. SHA-256 ordering with split seed.",
        "normalization": "Unicode NFKC, casefold, collapse whitespace; first source occurrence retained, all source row indices preserved (one-based data rows).",
        "leakage_checks": {"disjoint_ids": True, "disjoint_normalized_prompts": True,
                           "all_unique_prompts_assigned_once": True,
                           "near_duplicate_or_semantic_overlap_checked": False},
        "readiness": {"sft": "Prompt inputs only; generate teacher trajectories and review/export accepted.jsonl before SFT.",
                      "evaluation": "Compatible with evaluation.evaluate --dataset.",
                      "rl": "Compatible with training.grpo --dataset; requires an SFT checkpoint and configured judges."},
        "splits": {},
    }
    for name, rows in splits.items():
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "prompts.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        path.write_text(text, encoding="utf-8")
        assert len(load_specs(path)) == len(rows)
        manifest["splits"][name] = {
            "file": f"{name}/prompts.jsonl", "records": len(rows),
            "sha256": digest(path.read_bytes()),
            "categories": dict(sorted(Counter(row["source"]["category"] for row in rows).items())),
            "challenges": dict(sorted(Counter(row["source"]["challenge"] for row in rows).items())),
            "strata": {f"{category} / {challenge}": count for (category, challenge), count in sorted(
                Counter((row["source"]["category"], row["source"]["challenge"]) for row in rows).items())},
        }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/parti-prompts/raw/PartiPrompts.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/parti-prompts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.seed)
    print(json.dumps({name: entry["records"] for name, entry in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
