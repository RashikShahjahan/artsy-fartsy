"""Shared drawing specification validation, taxonomy, and hashing."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

FAMILIES = [
    "circles", "rectangles", "triangles", "lines", "rows", "columns",
    "landscapes", "flowers", "grids", "houses", "rings", "orbits", "safety_refusal",
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
        # Legacy split labels are ignored, like other unused input metadata.
        normalized_row = {"id": ident, "family": row["family"], "prompt": row["prompt"],
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
