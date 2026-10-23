"""Dataset specification validation and legacy deterministic curriculum."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

PROMPT_VERSION = "artsy-curriculum-v1"
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

COLORS = ["red", "blue", "green", "orange", "purple", "teal"]
BACKGROUNDS = ["white", "light gray", "cream"]


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
        repair = row.get("repairTask")
        if repair is not None and (not isinstance(repair, dict) or repair.get("complexity") != "advanced"
                                   or not isinstance(repair.get("skills"), list) or len(repair["skills"]) < 2
                                   or not all(isinstance(skill, str) and skill.strip() for skill in repair["skills"])
                                   or "defects" in repair):
            raise ValueError("Repair tasks need advanced complexity and at least two skills; defect instructions are prohibited")
        ids.add(ident)
        prompts.add(normalized)
        normalized_row = {"id": ident, "family": row["family"], "split": split, "prompt": row["prompt"],
                          "seed": row["seed"]}
        if repair is not None:
            normalized_row["repairTask"] = copy.deepcopy(repair)
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


def generate_prompts(count: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    if not _safe_int(count) or count < 1 or count > 10_000 or not _safe_int(seed) or seed < 0:
        raise ValueError("count must be an integer from 1 to 10000 and seed a nonnegative safe integer")
    prompts: list[dict[str, Any]] = []
    for i in range(count):
        family, split = FAMILIES[i % len(FAMILIES)]
        entropy = sha256(f"{PROMPT_VERSION}:{seed}:{i}")

        def sample(offset: int, size: int) -> int:
            return int(entropy[offset:offset + 8], 16) % size

        color = COLORS[sample(0, len(COLORS))]
        accent = COLORS[(COLORS.index(color) + 1 + sample(8, 5)) % len(COLORS)]
        n = 2 + sample(16, 5)
        background = BACKGROUNDS[sample(24, len(BACKGROUNDS))]
        margin = 60 + (i // len(FAMILIES)) / 10
        tasks = {
            "circles": [f"Draw one filled {color} circle centered on the canvas", f"The circle has a radius of {100 + sample(32, 200)} pixels"],
            "rectangles": [f"Draw one filled {color} rectangle centered on the canvas", f"Its width is {300 + sample(32, 400)} pixels and its height is 200 pixels"],
            "triangles": [f"Draw one filled {color} equilateral triangle centered on the canvas", "The triangle points upward and has a side length of 400 pixels"],
            "lines": [f"Draw exactly {n} parallel {color} horizontal lines", f"Use 12-pixel-wide strokes, equal spacing, and endpoints {200 + sample(32, 200)} pixels from both sides"],
            "rows": [f"Draw exactly {n} filled {color} circles in one centered horizontal row", "Use equal radii and spacing, with no overlaps"],
            "columns": [f"Draw exactly {n} filled {color} squares in one centered vertical column", "Use equal sizes and spacing, with no overlaps"],
            "landscapes": [f"Draw a flat landscape with exactly {n} {color} triangular mountains", f"Place a single {accent} sun above the mountains and a green ground strip below them"],
            "flowers": [f"Draw one flower with exactly {n + 3} {color} petals around a {accent} circular center", "Add a green stem below the flower and exactly two green leaves"],
            "grids": [f"Draw a grid of {n} columns and 3 rows of filled {color} squares", "Use equal square sizes and equal gaps, with the grid centered"],
            "houses": [f"Draw one house with a {color} rectangular body and a {accent} triangular roof", f"Add exactly {n} blue windows in a row and one brown door below the windows"],
            "rings": [f"Draw exactly {n} concentric {color} circle outlines centered on the canvas", "Use 10-pixel strokes, equal radial gaps, and no filled circles"],
            "orbits": [f"Draw one {accent} filled circle at the center with exactly {n} small {color} filled circles around it", "Space the small circles equally around a circular orbit and draw the orbit as a thin black outline"],
        }
        instructions = [*tasks[family], f"Use a {background} background",
                        f"Leave at least {margin:g} pixels clear along every canvas edge",
                        "Do not include text, labels, borders, or extra objects"]
        lead = ["Please create this drawing:", "Make an illustration with these requirements:", "Draw the following composition:"][sample(40, 3)]
        prompts.append({"id": sha256(f"{PROMPT_VERSION}:{seed}:{i}")[:24], "family": family,
                        "split": split, "prompt": f"{lead} {'. '.join(instructions)}.",
                        "seed": int(entropy[:8], 16)})
    return validate_specs(prompts)
