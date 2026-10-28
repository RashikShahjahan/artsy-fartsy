"""Validated human image preferences with explicit held-out splits."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .judge import native_messages, validate_png


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    group_id: str
    image_a: Path
    image_b: Path
    winner: str
    split: str

    def conversation(self, image_processor, *, reverse: bool = False) -> list[dict]:
        image_a, image_b = self.image_a.read_bytes(), self.image_b.read_bytes()
        label = self.winner
        if reverse:
            image_a, image_b = image_b, image_a
            label = {"A": "B", "B": "A", "Tie": "Tie"}[label]
        return native_messages(image_a, image_b, image_processor) + [{
            "role": "assistant", "content": json.dumps({"winner": label}, separators=(",", ":"))
        }]


def load_pairs(path: str | Path) -> list[PreferencePair]:
    path = Path(path)
    root = path.resolve().parent
    pairs: list[PreferencePair] = []
    ids: set[str] = set()
    groups: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    pair_keys: set[tuple[str, str]] = set()
    required = {"pair_id", "group_id", "image_a", "image_b", "winner", "split"}

    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError("Missing required preference fields")
            if any(not isinstance(row[key], str) or not row[key].strip() for key in required):
                raise ValueError("Preference fields must be nonempty strings")
            if row["winner"] not in ("A", "B", "Tie"):
                raise ValueError("winner must be A, B, or Tie")
            if row["split"] not in ("train", "validation", "test"):
                raise ValueError("split must be train, validation, or test")
            if row["pair_id"] in ids:
                raise ValueError("Duplicate pair_id")
            ids.add(row["pair_id"])
            group, split = row["group_id"], row["split"]
            if group in groups and groups[group] != split:
                raise ValueError("group_id leaks across splits")
            groups[group] = split
            images: list[Path] = []
            hashes: list[str] = []
            for key in ("image_a", "image_b"):
                file = (root / row[key]).resolve()
                if not file.is_relative_to(root):
                    raise ValueError("Image paths must remain inside dataset directory")
                png = file.read_bytes()
                validate_png(png)
                digest = hashlib.sha256(png).hexdigest()
                if digest in image_splits and image_splits[digest] != split:
                    raise ValueError("Identical image leaks across splits")
                image_splits[digest] = split
                images.append(file)
                hashes.append(digest)
            if hashes[0] == hashes[1]:
                raise ValueError("Pair must contain distinct images")
            key = tuple(sorted(hashes))
            if key in pair_keys:
                raise ValueError("Duplicate or reversed image pair")
            pair_keys.add(key)
            pairs.append(PreferencePair(row["pair_id"], group, *images, row["winner"], split))
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not pairs:
        raise ValueError("Preference dataset is empty")
    return pairs
