"""Build Experiment 9's 681 RLVR / 150 hard-evaluation prompt inputs.

Uses pinned, prompt-only GenAI-Bench metadata. No inference or image downloads.
Run with .venv/bin/python scripts/prepare_expanded_rlvr.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drawing.eval_specs import load_specs
from training.rl_split import partition_rl_prompts

REVISION = "e51e8a8070fadfbfe1f160436bf14e7664e87247"
SOURCE = "BaiqiL/GenAI-Bench-1600"
HASHES = {
    "genai_image.json": "a7e15e46786b64f5d3518d3450b2b5a86438f2b946e8d093dcaf0d5c5c094b1e",
    "genai_skills.json": "8df390cb89d0584389747c69acecd03615ffc229c1db71a90e4abd3d12d9a3d2",
}
ADVANCED = {"counting", "comparison", "differentiation", "negation", "universal"}
BUCKETS = {
    "counting_attributes": {"counting"},
    "spatial_composition": {"spatial relation"},
    "comparison_logic": {"comparison", "differentiation", "negation", "universal"},
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def signature(text: str) -> tuple[str, frozenset[str]]:
    words = re.findall(r"\w+", normalize(text))
    return " ".join(words), frozenset(words)


def near(a, b) -> bool:
    """Conservative lexical screen; not a semantic-equivalence classifier."""
    if a[0] == b[0]:
        return True
    intersection = len(a[1] & b[1])
    if intersection / max(1, len(a[1] | b[1])) >= 0.80:
        return True
    if min(len(a[0]), len(b[0])) / max(1, len(a[0]), len(b[0])) < 0.90:
        return False
    # A 90% character match also needs substantial common vocabulary.
    return (intersection / max(1, min(len(a[1]), len(b[1]))) >= 0.70
            and SequenceMatcher(None, a[0], b[0], autojunk=False).ratio() >= 0.90)


def stable(row: dict, salt: str) -> str:
    return digest(f"42:{salt}:{row['id']}".encode())


def hardness(row: dict) -> tuple[int, int]:
    tags = set(row["source"]["tags"])
    # Multiple independent skills, especially advanced ones; length breaks ties.
    return len(tags) + 2 * len(tags & ADVANCED), len(row["prompt"].split())


def stats(rows: list[dict]) -> dict:
    tags = Counter(tag for row in rows for tag in row.get("source", {}).get("tags", []))
    scores = [hardness(row)[0] for row in rows if row["id"].startswith("genai-")]
    return {"records": len(rows), "tags": dict(sorted(tags.items())),
            "primary_categories": dict(sorted(Counter(row.get("primarySkill", "parti") for row in rows).items())),
            "mean_tag_difficulty_score": sum(scores) / len(scores) if scores else None}


def prepare(output: Path) -> dict:
    raw = {}
    for name, expected in HASHES.items():
        path = output / "raw" / name
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        raw[name] = json.loads(path.read_text())
    images, skills = raw["genai_image.json"], raw["genai_skills.json"]
    if len(images) != 1600:
        raise ValueError("Expected 1,600 source prompts")
    index_tags = {int(key): [] for key in images}
    for tag, indices in skills.items():
        if tag not in {"basic", "advanced"}:
            for index in indices:
                index_tags[index].append(tag)

    # Exclude all canonical Parti splits, the selected checkpoint's exact SFT
    # input, and the newer SFT collection too. Refusal prompts are never copied.
    checkpoint = ROOT / "checkpoints/parti-sft-artcanvas-full-2026-11-07-qwen3.5-4b-r128-b32"
    sft_manifest = json.loads((checkpoint / "training-manifest.json").read_text())
    sft_path = Path(sft_manifest["dataset"])
    exclusions = [ROOT / f"datasets/parti-prompts/{split}/prompts.jsonl"
                  for split in ("sft", "evaluation", "rl")]
    exclusions += [sft_path, ROOT / "datasets/sft-v2/prompts.jsonl"]
    existing = {}
    excluded_ids = set()
    exclusion_manifest = []
    for path in exclusions:
        data = path.read_bytes()
        if path == sft_path and digest(data) != sft_manifest["dataset_sha256"]:
            raise ValueError("Selected checkpoint SFT dataset hash changed")
        records = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        for record in records:
            spec = record.get("spec", record)
            existing[normalize(spec["prompt"])] = signature(spec["prompt"])
            excluded_ids.add(spec["id"])
        exclusion_manifest.append({"file": str(path.relative_to(ROOT)),
                                   "sha256": digest(data), "records": len(records)})
    base = partition_rl_prompts(load_specs(exclusions[2]))["rlvr"]
    assert len(base) == 227
    base_path = exclusions[2]
    pool, rejected = [], []
    seen = list(existing.values())
    for index, record in sorted(images.items()):
        text = record["prompt"]
        sig = signature(text)
        if any(near(sig, other) for other in seen):
            rejected.append({"source_id": index, "reason": "existing_or_source_near_duplicate"})
            continue
        seen.append(sig)
        hashed = digest(normalize(text).encode())
        tags = sorted(index_tags[int(index)])
        pool.append({"id": "genai-" + hashed[:24], "family": "genai_bench",
                     "prompt": text, "seed": int(hashed[:8], 16),
                     "source": {"dataset": SOURCE, "revision": REVISION,
                                "source_id": index, "tags": tags}})
    selected_eval, selection_signatures = [], []
    # Scarce categories first. Reserve hard evaluation before selecting training.
    for bucket, eligible_tags in BUCKETS.items():
        candidates = [row for row in pool if set(row["source"]["tags"]) & eligible_tags
                      and len(row["source"]["tags"]) >= 3]
        candidates.sort(key=lambda row: (-hardness(row)[0], -hardness(row)[1], stable(row, "eval")))
        chosen = []
        for row in candidates:
            sig = signature(row["prompt"])
            if any(near(sig, other) for other in selection_signatures):
                continue
            chosen.append({**row, "difficulty": "hard", "primarySkill": bucket,
                           "difficultySelection": {"method": "skill-tags-v1", "score": hardness(row)[0]}})
            selection_signatures.append(sig)
            if len(chosen) == 50:
                break
        if len(chosen) != 50:
            raise ValueError(f"Insufficient hard prompts for {bucket}: {len(chosen)}")
        selected_eval.extend(chosen)

    # Balanced additions, retaining basic scenes as well as advanced challenges.
    training, allocation = [], {"counting_attributes": 114, "spatial_composition": 114,
                                "comparison_logic": 113, "scene_attributes": 113}
    for bucket, count in allocation.items():
        eligible_tags = BUCKETS.get(bucket, {"scene", "attribute", "action relation", "part relation"})
        candidates = sorted((row for row in pool if set(row["source"]["tags"]) & eligible_tags),
                            key=lambda row: stable(row, "train:" + bucket))
        chosen = []
        for row in candidates:
            sig = signature(row["prompt"])
            if any(near(sig, other) for other in selection_signatures):
                continue
            chosen.append({**row, "primarySkill": bucket})
            selection_signatures.append(sig)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Insufficient training prompts for {bucket}: {len(chosen)}")
        training.extend(chosen)
    rlvr = sorted(base + training, key=lambda row: stable(row, "order"))
    evaluation = sorted(selected_eval, key=lambda row: stable(row, "order"))
    assert len(rlvr) == 681 and len(evaluation) == 150
    assert not ({r["id"] for r in evaluation} & {r["id"] for r in rlvr})
    assert not ({r["id"] for r in training + evaluation} & excluded_ids)
    manifest = {
        "version": "artsy-expanded-rlvr-v1", "seed": 42,
        "source": {"dataset": SOURCE, "revision": REVISION, "files": HASHES,
                   "license": "Apache-2.0 (GenAI-Bench dataset card)",
                   "url": "https://huggingface.co/datasets/" + SOURCE},
        "base_rlvr": {"file": str(base_path.relative_to(ROOT)),
                      "method": "existing canonical 227/100 stage partition", "records": 227},
        "exclusion_inputs": exclusion_manifest,
        "selection": {"evaluation_first": True, "evaluation_per_bucket": 50,
                      "minimum_eval_tags": 3,
                      "counting_eval_note": "Source rarely co-tags counting with attribute; use counting plus at least two other skills, often differentiation/logic.",
                      "difficulty": "tag count + 2 * advanced tag count; word count breaks ties",
                      "training_allocation": allocation,
                      "ratings_used": False, "source_prompts_rewritten": False},
        "leakage_checks": {"disjoint_ids": True, "disjoint_normalized_text": True,
                           "lexical_near_duplicates_screened": True,
                           "method": "punctuation-normalized equality, token Jaccard >=0.80, or character ratio >=0.90 with shared vocabulary >=0.70",
                           "scope": "all additions vs exclusion inputs, each other, and held-out evaluation",
                           "semantic_equivalence_verified": False},
        "source_rejections": rejected, "eligible_source_pool": stats(pool),
        "splits": {},
    }
    for name, rows in (("rlvr", rlvr), ("evaluation-hard", evaluation), ("rlvr-added", training)):
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "prompts.jsonl"
        data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        if path.exists() and path.read_text() != data:
            raise ValueError(f"Refusing to replace a different frozen dataset: {path}")
        path.write_text(data)
        assert len(load_specs(path)) == len(rows)
        manifest["splits"][name] = {"file": str(path.relative_to(output)),
                                   "sha256": digest(path.read_bytes()), **stats(rows)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/experiment9-genai")
    result = prepare(parser.parse_args().output.resolve())
    print(json.dumps({"splits": result["splits"],
                      "source_rejections": len(result["source_rejections"])}, indent=2))
