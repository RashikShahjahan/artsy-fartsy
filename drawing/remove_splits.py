"""Remove legacy split metadata/exports without regenerating dataset records.

Run once for an existing dataset: python -m drawing.remove_splits datasets/my-run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .dataset import _write_json, _write_jsonl_atomic
from .dataset_specs import compact_json, sha256


def _without_splits(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_splits(item) for key, item in value.items() if key != "split"}
    if isinstance(value, list):
        return [_without_splits(item) for item in value]
    return value


def remove_splits(directory: Path) -> dict[str, int]:
    """Migrate checkpoints and frozen-plan hashes, then remove redundant exports.

    Raw completion strings, record IDs, source/image hashes and artifacts stay
    intact. Both generation locks prevent concurrent writes during migration.
    All files are parsed and split coverage is checked before the first edit.
    """
    directory = directory.resolve()
    if not (directory / "manifest.json").is_file() and not (directory / "prompt-progress.json").is_file():
        raise ValueError("Expected a dataset manifest or prompt-progress checkpoint")
    locks: list[Path] = []
    try:
        for name in (".generation.lock", ".prompt-generation.lock"):
            lock = directory / name
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                raise RuntimeError(f"Dataset is locked: {lock}") from None
            os.close(fd)
            locks.append(lock)

        exports = [directory / f"{name}.jsonl" for name in ("train", "validation", "test")]
        existing_exports = [path for path in exports if path.exists()]
        paths = sorted(directory.glob("*.json"))
        paths += [path for path in sorted(directory.glob("*.jsonl")) if path not in exports]
        for name in ("records", "prompt-records"):
            paths += sorted((directory / name).glob("*.json"))
        original = {
            path: ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                   if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8")))
            for path in paths
        }
        accepted = original.get(directory / "accepted.jsonl", [])
        if existing_exports:
            split_rows = [json.loads(line) for path in existing_exports
                          for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if (len(split_rows) != len(accepted)
                    or {row["id"] for row in split_rows} != {row["id"] for row in accepted}):
                raise ValueError("Legacy split exports do not match accepted.jsonl; no files changed")

        migrated = {path: _without_splits(value) for path, value in original.items()}
        plan = directory / "prompt-plan.json"
        if plan in original:
            if sha256(compact_json(original[plan]["specs"])) != original[plan]["specsHash"]:
                raise ValueError("Stored prompt plan hash does not match; no files changed")
            migrated[plan]["specsHash"] = sha256(compact_json(migrated[plan]["specs"]))
        manifest = directory / "manifest.json"
        if manifest in migrated and "promptGenerator" in migrated[manifest]:
            specs_hash = sha256(compact_json(migrated[manifest]["specs"]))
            if plan in migrated and migrated[plan]["specsHash"] != specs_hash:
                raise ValueError("Manifest and prompt plan differ; no files changed")
            migrated[manifest]["promptGenerator"]["specsHash"] = specs_hash

        changed = 0
        for path, value in migrated.items():
            if value == original[path]:
                continue
            if path.suffix == ".jsonl":
                _write_jsonl_atomic(path, value)
            else:
                _write_json(path, value)
            changed += 1
        for path in existing_exports:
            path.unlink()
        return {"updatedFiles": changed, "removedExports": len(existing_exports),
                "acceptedRecords": len(accepted)}
    finally:
        for lock in reversed(locks):
            lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(remove_splits(args.directory)))


if __name__ == "__main__":
    main()
