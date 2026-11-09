"""Disjoint, stratified RLVR/RLHF assignment from the reserved Parti RL pool.

The split is independent of training RNG seeds and preserves source ordering
within each stage. The same helper is used by dataset preparation and training.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path

RL_TOTAL = 327
RLVR_COUNT = 227
RLHF_COUNT = 100
SPLIT_VERSION = "artsy-parti-rl-stages-v1"
SPLIT_SALT = "artsy-parti-rl-stages-v1"
DEFAULT_RL_DATASET = Path(__file__).resolve().parents[1] / "datasets/parti-prompts/rl/prompts.jsonl"


def _stratum(spec: dict) -> tuple[str, str]:
    source = spec.get("source")
    if isinstance(source, dict) and source.get("category") and source.get("challenge"):
        return str(source["category"]), str(source["challenge"])
    return str(spec.get("family", "")), str(spec.get("difficulty", ""))


def partition_rl_prompts(specs: list[dict], rlvr_count: int = RLVR_COUNT) -> dict[str, list[dict]]:
    """Apportion RLVR samples across source strata, leaving all others for RLHF.

    Largest-remainder quotas retain the original Category/Challenge balance.
    Stable SHA-256 within each stratum makes membership independent of the
    incoming file order. Neither stage may reuse the other stage's prompt IDs.
    """
    if not specs or not 0 < rlvr_count < len(specs):
        raise ValueError("RL stage split needs nonempty RLVR and RLHF subsets")
    ids = [spec["id"] for spec in specs]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("RL stage split requires distinct nonempty prompt IDs")

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for spec in specs:
        strata[_stratum(spec)].append(spec)

    total = len(specs)
    sizes = {key: len(rows) for key, rows in strata.items()}
    quota = {key: size * rlvr_count // total for key, size in sizes.items()}
    remainder_order = sorted(
        sizes, key=lambda key: (-(sizes[key] * rlvr_count % total), key))
    for key in remainder_order[:rlvr_count - sum(quota.values())]:
        quota[key] += 1

    selected_ids = set()
    for key, rows in strata.items():
        ordered = sorted(
            rows, key=lambda row: sha256(f"{SPLIT_SALT}:{row['id']}".encode()).hexdigest())
        selected_ids.update(row["id"] for row in ordered[:quota[key]])

    result = {
        "rlvr": [spec for spec in specs if spec["id"] in selected_ids],
        "rlhf": [spec for spec in specs if spec["id"] not in selected_ids],
    }
    if (len(result["rlvr"]) != rlvr_count
            or len(result["rlhf"]) != total - rlvr_count):
        raise ValueError("RL stage split did not cover the input dataset exactly")
    return result


def stage_specs(dataset: Path, specs: list[dict], reward_mode: str) -> tuple[list[dict], str | None]:
    """Automatically split only the canonical reserved RL pool.

    Explicit alternate dataset files are treated as already-selected inputs.
    Reject a changed canonical pool instead of silently moving prompts between
    stages or exposing RLHF examples to RLVR.
    """
    if reward_mode not in ("rlvr", "rlhf"):
        raise ValueError("Unknown RL reward mode")
    if Path(dataset).resolve() != DEFAULT_RL_DATASET.resolve():
        return specs, None
    if len(specs) != RL_TOTAL:
        raise ValueError(
            f"Canonical Parti RL dataset must contain {RL_TOTAL} prompts "
            f"for the {RLVR_COUNT}/{RLHF_COUNT} split; found {len(specs)}")
    return partition_rl_prompts(specs)[reward_mode], SPLIT_VERSION
