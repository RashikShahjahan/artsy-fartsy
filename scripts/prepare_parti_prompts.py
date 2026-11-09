"""Prepare disjoint PartiPrompts inputs for SFT, evaluation, RL and preferences.

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
from training.rl_split import (RL_TOTAL, RLVR_COUNT, RLHF_COUNT, SPLIT_VERSION,
                               partition_rl_prompts)

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


def prepare(source: Path, output: Path, seed: int, preference_count: int | None = 0,
            preference_to_evaluation: bool = False) -> dict:
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

    # Carve only from the original SFT pool, preserving evaluation/RL assignments.
    original_sft_count = targets["sft"]
    if preference_count is None:
        preference_count = (original_sft_count * 20 + 50) // 100
    if not 0 <= preference_count < original_sft_count:
        raise ValueError("Preference count must be nonnegative and leave SFT nonempty")
    if preference_to_evaluation and not preference_count:
        raise ValueError("Moving preference prompts to evaluation requires a positive preference count")
    sft_strata = defaultdict(list)
    for row in splits["sft"]:
        sft_strata[(row["source"]["category"], row["source"]["challenge"])].append(row)
    preference_counts = apportion({key: len(rows) for key, rows in sft_strata.items()},
                                  preference_count)
    splits["sft"] = []
    splits["preference"] = []
    for key in sorted(sft_strata):
        rows = sorted(sft_strata[key],
                      key=lambda row: digest(f"{seed}:preference:{row['id']}".encode()))
        boundary = preference_counts[key]
        splits["preference"].extend(rows[:boundary])
        splits["sft"].extend(rows[boundary:])
    targets["sft"] -= preference_count
    targets["preference"] = preference_count

    # Keep the source order out of the training order as well as the assignment.
    for rows in splits.values():
        rows.sort(key=lambda row: digest(f"{seed}:order:{row['id']}".encode()))
        if rows:
            validate_specs(rows)
    original_evaluation = list(splits['evaluation'])
    added_evaluation = list(splits['preference']) if preference_to_evaluation else []
    if preference_to_evaluation:
        # Keep the historical eval as a prefix so existing sample IDs and order
        # remain usable; the extension can be evaluated independently.
        splits['evaluation'].extend(added_evaluation)
        splits['preference'] = []
        targets['evaluation'] += targets['preference']
        targets['preference'] = 0
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
        "format_version": 2,
        "source": {"url": SOURCE, "revision": REVISION, "license": "Apache-2.0",
                   "file": "raw/PartiPrompts.tsv", "sha256": digest(source.read_bytes()),
                   "records": len(raw)},
        "unique_prompts": len(groups), "duplicates_collapsed": len(raw) - len(groups),
        "split_seed": seed, "original_requested_percentages": dict(zip(SPLITS, (70, 10, 20))),
        "preference_reservation": {"from_split": "sft", "original_sft_records": original_sft_count,
                                   "records": preference_count,
                                   "percent_of_original_sft": 100 * preference_count / original_sft_count},
        "method": "Original 70/10/20 allocation: largest-remainder SFT allocation per (Category, Challenge), then evaluation from remainder; RL receives remainder. Preference is carved from original SFT using largest-remainder allocation per (Category, Challenge) and SHA-256 seed:preference:id ordering. Evaluation and RL are unchanged. Final ordering uses SHA-256 with split seed.",
        "normalization": "Unicode NFKC, casefold, collapse whitespace; first source occurrence retained, all source row indices preserved (one-based data rows).",
        "leakage_checks": {"disjoint_ids": True, "disjoint_normalized_prompts": True,
                           "all_unique_prompts_assigned_once": True,
                           "near_duplicate_or_semantic_overlap_checked": False},
        "readiness": {"sft": "Prompt inputs only; generate teacher trajectories and review/export accepted.jsonl before SFT.",
                      "preference": "Reserved prompt inputs only; collect candidate outputs and preference labels before preference tuning. Exclude from SFT teacher generation and training.",
                      "evaluation": "Compatible with evaluation.evaluate --dataset.",
                      "rl": "Compatible with training.grpo --dataset; requires an SFT checkpoint and configured judges."},
        "splits": {},
    }
    if not preference_count:
        manifest['format_version'] = 3
        manifest['method'] = 'Original 70/10/20 allocation with no preference reservation. All original SFT prompts, including the formerly reserved 228, belong to SFT. Final ordering uses SHA-256 with split seed.'
        manifest['preference_reservation']['current_purpose'] = 'sft'
        manifest['readiness']['preference'] = 'Empty; former preference prompts restored to SFT.'
        manifest['historical_evaluation'] = {
            'experiment': 2,
            'archive': 'eval-results/experiment-2-extended-prompts',
            'warning': 'Restored prompts are eligible for SFT. Historical extended evaluations are not held-out evaluation for models trained on these prompts.'}
    if preference_to_evaluation:
        manifest['format_version'] = 3
        manifest['method'] += ' The complete preference reserve is then reassigned to evaluation, appended after the original evaluation prompts; SFT and RL remain unchanged.'
        manifest['preference_reservation']['current_purpose'] = 'evaluation'
        manifest['readiness']['preference'] = 'Empty: former reserved prompts are now evaluation-only and must not be used for preference tuning.'
        manifest['evaluation_extension'] = {'from_split': 'preference', 'records': len(added_evaluation),
            'original_records': len(original_evaluation), 'subsets': {}}
        for name, rows in [('original', original_evaluation), ('added-from-preference', added_evaluation)]:
            path = output / 'evaluation' / f'{name}.prompts.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows), encoding='utf-8')
            manifest['evaluation_extension']['subsets'][name] = {
                'file': str(path.relative_to(output)), 'records': len(rows),
                'sha256': digest(path.read_bytes())}
    for name, rows in splits.items():
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "prompts.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        path.write_text(text, encoding="utf-8")
        if rows:
            assert len(load_specs(path)) == len(rows)
        else:
            assert not path.read_bytes()
        manifest["splits"][name] = {
            "file": f"{name}/prompts.jsonl", "records": len(rows),
            "sha256": digest(path.read_bytes()),
            "categories": dict(sorted(Counter(row["source"]["category"] for row in rows).items())),
            "challenges": dict(sorted(Counter(row["source"]["challenge"] for row in rows).items())),
            "strata": {f"{category} / {challenge}": count for (category, challenge), count in sorted(
                Counter((row["source"]["category"], row["source"]["challenge"]) for row in rows).items())},
        }
    # Also materialize the stage-specific RL subsets for inspection and
    # explicit --dataset overrides. The trainer derives precisely the same
    # assignments from rl/prompts.jsonl when using its default dataset.
    rlvr_target = (len(splits["rl"]) * RLVR_COUNT + RL_TOTAL // 2) // RL_TOTAL
    staged = partition_rl_prompts(splits["rl"], rlvr_count=rlvr_target)
    assert set(row["id"] for row in staged["rlvr"]).isdisjoint(
        row["id"] for row in staged["rlhf"])
    manifest["rl_stage_splits"] = {
        "version": SPLIT_VERSION,
        "source_split": "rl",
        "canonical_counts": {"rlvr": RLVR_COUNT, "rlhf": RLHF_COUNT, "total": RL_TOTAL},
        "method": "Stratified largest-remainder allocation over RL Category/Challenge; "
                  "stable SHA-256 by prompt ID, independent of training RNG seeds.",
        "splits": {},
    }
    for stage in ("rlvr", "rlhf"):
        rows = staged[stage]
        path = output / "rl" / f"{stage}.prompts.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                       for row in rows)
        path.write_text(text, encoding="utf-8")
        assert len(load_specs(path)) == len(rows)
        manifest["rl_stage_splits"]["splits"][stage] = {
            "file": f"rl/{stage}.prompts.jsonl",
            "records": len(rows),
            "sha256": digest(path.read_bytes()),
            "categories": dict(sorted(Counter(row["source"]["category"] for row in rows).items())),
            "challenges": dict(sorted(Counter(row["source"]["challenge"] for row in rows).items())),
        }
    if len(splits["rl"]) == RL_TOTAL:
        assert len(staged["rlvr"]) == RLVR_COUNT and len(staged["rlhf"]) == RLHF_COUNT
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/parti-prompts/raw/PartiPrompts.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/parti-prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preference-count", type=int, default=0,
                        help="Prompts to reserve from original SFT (default: 0; all original SFT prompts retained)")
    parser.add_argument('--preference-to-evaluation', action='store_true',
                        help='Move the full preference reserve into evaluation; preserve original and added eval subsets')
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.seed, args.preference_count, args.preference_to_evaluation)
    print(json.dumps({name: entry["records"] for name, entry in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
