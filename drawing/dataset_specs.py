"""Shared drawing specification validation, taxonomy, and hashing."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

FAMILIES = [
    ("circles", "train"), ("rectangles", "train"), ("triangles", "train"), ("lines", "train"),
    ("rows", "train"), ("columns", "train"), ("landscapes", "train"), ("flowers", "train"),
    ("grids", "validation"), ("houses", "test"), ("rings", "validation"), ("orbits", "test"),
    ("safety_refusal", "train"),
]
# Orthogonal to the scene family: a prompt can exercise several skills.
# Used to plan/stratify datasets, never as requirements sent to an evaluator.
DIFFICULTIES = ("easy", "medium", "hard", "very-hard")

SKILL_TAXONOMY = {
    "primitives": "Basic shapes, paths, fills, and strokes",
    "counting": "Exact quantities, repetition, and grouping",
    "positioning": "Absolute placement, spacing, and alignment",
    "relative_geometry": "Tangency, containment, distances, and relative sizes",
    "layering": "Occlusion and draw order",
    "transformations": "Translation, rotation, reflection, and scaling",
    "symmetry": "Mirror and rotational symmetry",
    "curves": "Arcs, Bezier paths, and smooth contours",
    "procedural_geometry": "Algorithmic patterns, randomness, and tiling",
    "composition": "Multi-object scenes and visual hierarchy",
    "constraint_following": "Simultaneous explicit visual conditions",
    "artistic_quality": "Color harmony, gradients, and visual finish",
    "negative_constraints": "Explicitly forbidden visual elements",
    "robustness": "Edge cases, clipping, density, and degeneracy",
    "refusal": "Recognize and refuse unsafe drawing requests",
}


def compact_json(value: Any) -> str:
    """Serialize like JSON.stringify for the JSON values used by this module."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256(value: str | bytes | bytearray) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def family_split(family: str) -> str:
    for name, split in FAMILIES:
        if name == family:
            return split
    bucket = int(sha256(family)[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def assign_holdout(specs: list[dict[str, Any]], eval_count: int, seed: int) -> list[dict[str, Any]]:
    """Allocate an exact, stable held-out test set within task strata.

    The whole prompt plan is validated before splitting, so near-duplicates
    rejected by the prompt writer cannot leak across train/test.
    """
    specs = validate_specs(specs)
    if type(eval_count) is not int or not 0 <= eval_count < len(specs):
        raise ValueError("eval-count must be between zero and count - 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("Holdout seed must be a nonnegative integer")
    if eval_count == 0:
        return specs

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for spec in specs:
        key = (spec.get("difficulty", "unspecified"),
               spec.get("kind", "drawing"), spec["family"])
        groups.setdefault(key, []).append(spec)

    quotas: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[int, str, tuple[str, str, str]]] = []
    for key, rows in groups.items():
        whole, remainder = divmod(len(rows) * eval_count, len(specs))
        quotas[key] = whole
        remainders.append((remainder, sha256(f"{seed}:{compact_json(key)}"), key))
    remaining = eval_count - sum(quotas.values())
    for _, _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        quotas[key] += 1

    heldout: set[str] = set()
    for key, rows in groups.items():
        ranked = sorted(rows, key=lambda spec: sha256(f"{seed}:{spec['id']}"))
        heldout.update(spec["id"] for spec in ranked[:quotas[key]])
    assert len(heldout) == eval_count
    return validate_specs([{**spec, "split": "test" if spec["id"] in heldout else "train"}
                           for spec in specs])


def _safe_int(value: Any) -> bool:
    return type(value) is int and -9_007_199_254_740_991 <= value <= 9_007_199_254_740_991


def validate_specs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Expected at least one prompt specification")
    ids: set[str] = set()
    prompts: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in value:
        if (not isinstance(row, dict) or not isinstance(row.get("family"), str) or not row["family"].strip()
                or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                or not _safe_int(row.get("seed")) or row["seed"] < 0):
            raise ValueError("Each prompt needs family, prompt, and a nonnegative integer seed")
        split = row.get("split", family_split(row["family"]))
        if split not in ("train", "validation", "test"):
            raise ValueError("Split must be train, validation, or test")
        ident = row.get("id")
        if ident is None:
            ident = sha256(compact_json([row["family"], row["prompt"], row["seed"]]))[:24]
        if not isinstance(ident, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", ident) or ident in ids:
            raise ValueError("Prompt IDs must be unique safe filenames (1–80 characters)")
        normalized = re.sub(r"\s+", " ", row["prompt"].strip()).lower()
        if normalized in prompts:
            raise ValueError("Duplicate prompt in generation plan")
        ids.add(ident)
        prompts.add(normalized)
        normalized_row = {"id": ident, "family": row["family"], "split": split, "prompt": row["prompt"],
                          "seed": row["seed"]}
        skills = row.get("skills")
        if skills is not None:
            if (not isinstance(skills, list) or not 1 <= len(skills) <= 5
                    or any(not isinstance(skill, str) or skill not in SKILL_TAXONOMY for skill in skills)
                    or len(set(skills)) != len(skills)):
                raise ValueError("Skills must be 1-5 unique taxonomy labels")
            normalized_row["skills"] = list(skills)
        difficulty = row.get("difficulty")
        if difficulty is not None:
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
            normalized_row["difficulty"] = difficulty
        kind = row.get("kind")
        if kind is not None:
            if kind != "refusal" or row["family"] != "safety_refusal":
                raise ValueError("Only safety_refusal family may specify kind=refusal")
            refusal = row.get("refusal")
            if (not isinstance(refusal, str) or not 10 <= len(refusal.strip()) <= 500
                    or "```" in refusal or any(ord(char) < 32 and char not in "\n\t" for char in refusal)):
                raise ValueError("Refusal targets must be 10-500 characters of plain text")
            if skills is not None and "refusal" not in skills:
                raise ValueError("Refusal skills must include refusal")
            normalized_row["kind"] = "refusal"
            normalized_row["refusal"] = refusal.strip()
        elif "refusal" in row:
            raise ValueError("Refusal target requires kind=refusal")
        result.append(normalized_row)
    return result
