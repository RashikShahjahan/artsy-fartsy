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
        split = family_split(row["family"])
        if "split" in row and row["split"] != split:
            raise ValueError(f"Split for family {row['family']} must be {split}")
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
        result.append(normalized_row)
    return result
