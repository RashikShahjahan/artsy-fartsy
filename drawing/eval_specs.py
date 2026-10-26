"""Validate evaluation/RL prompt specs without importing Inspect AI."""

import json
from pathlib import Path
import re

from drawing.dataset_specs import DIFFICULTIES, SAFETY_CATEGORIES


def load_specs(path: Path) -> list[dict]:
    specs = []
    ids, prompts = set(), set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", row["id"])
            or not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
            or type(row.get("seed")) is not int or not 0 <= row["seed"] <= 2**53 - 1
            or ("difficulty" in row and row["difficulty"] not in DIFFICULTIES)
            or (row.get("kind") not in (None, "refusal"))
            or ("safetyCategory" in row and
                (row.get("kind") != "refusal" or row["safetyCategory"] not in SAFETY_CATEGORIES))
            or (row.get("kind") == "refusal" and
                (row.get("family") != "safety_refusal" or
                 not isinstance(row.get("refusal"), str) or not row["refusal"].strip()))):
            raise ValueError(f"{path}:{line_number}: expected a drawing spec with id, prompt, and seed")
        normalized = " ".join(row["prompt"].lower().split())
        if row["id"] in ids or normalized in prompts:
            raise ValueError(f"{path}:{line_number}: duplicate ID or prompt")
        ids.add(row["id"])
        prompts.add(normalized)
        specs.append({key: value for key, value in row.items() if key != "split"})
    if not specs:
        raise ValueError("Evaluation dataset is empty")
    return specs
