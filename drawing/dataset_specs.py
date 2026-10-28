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
SAFETY_CATEGORIES = ("harmful_image", "malicious_code")

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
    "refusal": "Recognize and refuse unsafe image or malicious-code requests",
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
        ids.add(ident)
        # Legacy split labels are ignored, like other unused input metadata.
        normalized_row = {"id": ident, "family": row["family"], "prompt": row["prompt"],
                          "seed": row["seed"]}
        skills = row.get("skills")
        if skills is not None:
            if (not isinstance(skills, list) or not skills
                    or any(not isinstance(skill, str) or not skill.strip() for skill in skills)):
                raise ValueError("Skills must be nonempty string labels")
            normalized_row["skills"] = list(skills)
        difficulty = row.get("difficulty")
        if difficulty is not None:
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
            normalized_row["difficulty"] = difficulty
        if "plannedDifficulty" in row:
            if row["plannedDifficulty"] not in DIFFICULTIES:
                raise ValueError("Invalid planned difficulty")
            normalized_row["plannedDifficulty"] = row["plannedDifficulty"]
        kind = row.get("kind")
        if kind is not None:
            if kind != "refusal" or row["family"] != "safety_refusal":
                raise ValueError("Only safety_refusal family may specify kind=refusal")
            refusal = row.get("refusal")
            if not isinstance(refusal, str) or not refusal.strip():
                raise ValueError("Refusal targets must be nonempty strings")
            normalized_row["kind"] = "refusal"
            normalized_row["refusal"] = refusal.strip()
            if "safetyCategory" in row:
                if row["safetyCategory"] not in SAFETY_CATEGORIES:
                    raise ValueError(f"safetyCategory must be one of {SAFETY_CATEGORIES}")
                normalized_row["safetyCategory"] = row["safetyCategory"]
        elif "refusal" in row:
            raise ValueError("Refusal target requires kind=refusal")
        elif "safetyCategory" in row:
            raise ValueError("safetyCategory requires kind=refusal")
        # Preserve prompt planning provenance through normalization and exports.
        # It is analysis metadata, never a hidden drawing requirement.
        for key in ("benchmarkSlice", "contentTheme", "format", "primarySkill",
                    "artStyle", "voice", "generatorVersion"):
            if key in row:
                if not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > 300:
                    raise ValueError(f"Invalid prompt metadata: {key}")
                normalized_row[key] = row[key]
        if "benchmarkSlice" in row:
            expected = row.get("safetyCategory", "harmful_image") if kind == "refusal" else None
            if (expected and row["benchmarkSlice"] != expected) or (
                not expected and row["benchmarkSlice"] not in ("drawing", "benign_boundary")
            ):
                raise ValueError("Benchmark slice does not match the prompt kind/category")
        result.append(normalized_row)
    return result
