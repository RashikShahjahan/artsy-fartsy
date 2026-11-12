"""Build 300 RLHF prompts without reusing expanded RLVR or evaluation inputs.

Uses the pinned local sources and exclusion manifest from prepare_expanded_rlvr.
No inference, preference labels, image downloads, or training are performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_expanded_rlvr import (
    BUCKETS, HASHES, REVISION, SOURCE, digest, near, normalize, signature, stable, stats,
)
from drawing.eval_specs import load_specs
from training.rl_split import partition_rl_prompts

ALLOCATION = {"counting_attributes": 50, "spatial_composition": 50,
              "comparison_logic": 50, "scene_attributes": 50}


def prepare(output: Path, rlvr_dir: Path) -> dict:
    upstream_path = rlvr_dir / "manifest.json"
    upstream_bytes = upstream_path.read_bytes()
    upstream = json.loads(upstream_bytes)
    if (upstream["version"] != "artsy-expanded-rlvr-v1"
            or upstream["source"]["revision"] != REVISION
            or upstream["source"]["files"] != HASHES):
        raise ValueError("Expected the frozen Experiment 9 RLVR source manifest")

    raw = {}
    for name, expected in HASHES.items():
        data = (rlvr_dir / "raw" / name).read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
        raw[name] = json.loads(data)
    images, skills = raw["genai_image.json"], raw["genai_skills.json"]
    if len(images) != 1600:
        raise ValueError("Expected 1,600 source prompts")
    tags = {int(key): [] for key in images}
    for tag, indices in skills.items():
        if tag not in {"basic", "advanced"}:
            for index in indices:
                tags[index].append(tag)

    # Verify every frozen input before using it, including the exact SFT input
    # used for Experiment 9. Canonical RL includes the retained RLHF reserve.
    inputs = [(ROOT / item["file"], item) for item in upstream["exclusion_inputs"]]
    for split in ("rlvr", "evaluation-hard"):
        item = upstream["splits"][split]
        inputs.append((rlvr_dir / item["file"], item))
    exclusions, seen, blocked_ids = [], [], set()
    blocked_training_eval = []
    canonical = ROOT / "datasets/parti-prompts/rl/prompts.jsonl"
    for path, expected in inputs:
        data = path.read_bytes()
        if digest(data) != expected["sha256"]:
            raise ValueError(f"Frozen exclusion input changed: {path}")
        count = 0
        # SFT records wrap the drawing spec; prompt-only inputs do not.
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            spec = record.get("spec", record)
            seen.append(signature(spec["prompt"]))
            blocked_ids.add(spec["id"])
            if path != canonical:
                blocked_training_eval.append(spec)
            count += 1
        if count != expected["records"]:
            raise ValueError(f"Frozen exclusion count changed: {path}")
        exclusions.append({"file": str(path.relative_to(ROOT)),
                           "sha256": digest(data), "records": count})
    # Deduplicate exclusion signatures to keep the quadratic screen small.
    seen = list(dict.fromkeys(seen))
    base = partition_rl_prompts(load_specs(canonical))["rlhf"]
    if len(base) != 100:
        raise ValueError("Expected the unchanged 100-prompt Parti RLHF reserve")
    if ({r["id"] for r in base} & {r["id"] for r in blocked_training_eval}
            or {normalize(r["prompt"]) for r in base}
            & {normalize(r["prompt"]) for r in blocked_training_eval}):
        raise ValueError("Canonical RLHF reserve overlaps training/evaluation inputs")

    pool, rejected = [], []
    for index, record in sorted(images.items()):
        prompt = record["prompt"]
        hashed = digest(normalize(prompt).encode())
        row_id = "genai-" + hashed[:24]
        sig = signature(prompt)
        if row_id in blocked_ids or any(near(sig, other) for other in seen):
            rejected.append({"source_id": index, "reason": "existing_or_source_near_duplicate"})
            continue
        seen.append(sig)
        pool.append({"id": row_id, "family": "genai_bench", "prompt": prompt,
                     "seed": int(hashed[:8], 16),
                     "source": {"dataset": SOURCE, "revision": REVISION,
                                "source_id": index, "tags": sorted(tags[int(index)])}})

    added, selected = [], set()
    for bucket, count in ALLOCATION.items():
        eligible = BUCKETS.get(bucket, {"scene", "attribute", "action relation", "part relation"})
        candidates = sorted((row for row in pool if set(row["source"]["tags"]) & eligible),
                            key=lambda row: stable(row, "rlhf-train:" + bucket))
        chosen = []
        for row in candidates:
            if row["id"] in selected:
                continue
            chosen.append({**row, "primarySkill": bucket})
            selected.add(row["id"])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Insufficient unused prompts for {bucket}: {len(chosen)}")
        added.extend(chosen)
    rows = sorted(base + added, key=lambda row: stable(row, "rlhf-order"))
    if len(rows) != 300 or len({r["id"] for r in rows}) != 300:
        raise ValueError("Expected 300 distinct RLHF prompts")

    manifest = {
        "version": "artsy-expanded-rlhf-v1", "seed": 42,
        "source": upstream["source"],
        "upstream_rlvr_manifest": {"file": str(upstream_path.relative_to(ROOT)),
                                   "sha256": digest(upstream_bytes)},
        "base_rlhf": {"file": str(canonical.relative_to(ROOT)), "records": 100,
                      "method": "unchanged canonical 227/100 stage partition"},
        "exclusion_inputs": exclusions,
        "selection": {"training_allocation": ALLOCATION, "ratings_used": False,
                      "source_prompts_rewritten": False,
                      "method": "seeded hash within skill buckets; scarce categories first"},
        "leakage_checks": {
            "disjoint_ids": True, "disjoint_normalized_text": True,
            "lexical_near_duplicates_screened": True,
            "method": upstream["leakage_checks"]["method"],
            "scope": "all additions vs frozen SFT inputs, all canonical Parti splits, expanded RLVR, hard evaluation, and each other",
            "base_reserve": "original assignments preserved; exact ID/text disjointness checked against SFT, RLVR and evaluation",
            "semantic_equivalence_verified": False},
        "evaluation": {name: item for name, item in (
            ("parti", exclusions[1]), ("hard", exclusions[-1]))},
        "source_rejections": rejected, "eligible_source_pool": stats(pool), "splits": {},
    }
    # Preflight every frozen output before writing any files.
    artifacts = []
    for name, records in (("rlhf", rows), ("rlhf-added", added)):
        path = output / name / "prompts.jsonl"
        data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                       for row in records).encode("utf-8")
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to replace a different frozen dataset: {path}")
        artifacts.append((path, data))
        manifest["splits"][name] = {"file": str(path.relative_to(output)),
                                   "sha256": digest(data), **stats(records)}
    for path, data in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        load_specs(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/rlhf-genai300")
    parser.add_argument("--rlvr-dir", type=Path, default=ROOT / "datasets/experiment9-genai")
    args = parser.parse_args()
    result = prepare(args.output.resolve(), args.rlvr_dir.resolve())
    print(json.dumps({"splits": result["splits"],
                      "source_rejections": len(result["source_rejections"])}, indent=2))
