"""Read-only, bounded inspection of Artsy RL artifacts.

This module is never imported by the RL trainer. Reading is initiated only by an
open browser page, and no file is created or changed by the monitor.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MAX_RUNS = 30
MAX_STEPS = 3
MAX_EPISODES = 48
MAX_RESULT_BYTES = 512 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024
PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
STEP = re.compile(r"step-[0-9]{6,}\Z")
EPISODE = re.compile(r"[a-f0-9]{32}\Z")


def child(root: Path, *parts: str) -> Path:
    """Only files under the explicitly configured root can be served."""
    if not all(isinstance(p, str) and PART.fullmatch(p) and p not in {".", ".."}
               for p in parts):
        raise ValueError("Invalid RL artifact path")
    base = root.resolve()
    candidate = base.joinpath(*parts).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError("RL artifact escapes configured root")
    return candidate


def load_json(path: Path, limit: int = MAX_RESULT_BYTES):
    try:
        if not path.is_file() or path.stat().st_size > limit:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        # A result may still be being written, or a run may be interrupted.
        return None


def _run_dirs(root: Path):
    root = root.resolve()
    if (root / "training-manifest.json").is_file():
        return [root]
    try:
        return [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                and PART.fullmatch(p.name)]
    except OSError:
        return []


def runs(root: Path):
    """Discover only RL runs, without walking episode directories."""
    choices = []
    for path in _run_dirs(root):
        manifest = load_json(path / "training-manifest.json", 256 * 1024)
        if not isinstance(manifest, dict):
            continue
        config = manifest.get("config")
        if not isinstance(config, dict) or not str(config.get("algorithm", "")).startswith("tinker-cookbook-"):
            continue
        choices.append({
            "id": path.name,
            "model": config.get("model"),
            "totalSteps": manifest.get("total_steps"),
            "groupSize": config.get("group_size"),
            "batchSize": config.get("batch_size"),
            "finished": (path / "final.json").is_file(),
        })
    return sorted(choices, key=lambda x: x["id"], reverse=True)[:MAX_RUNS]


def run_path(root: Path, name: str):
    base = root.resolve()
    path = base if (base / "training-manifest.json").is_file() and name == base.name else child(base, name)
    if path.is_symlink():
        raise ValueError("RL run not found")
    manifest = load_json(path / "training-manifest.json", 256 * 1024)
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(config, dict) or not str(config.get("algorithm", "")).startswith("tinker-cookbook-"):
        raise ValueError("RL run not found")
    return path


def _recent_steps(rollouts: Path):
    try:
        names = [p.name for p in rollouts.iterdir()
                 if p.is_dir() and not p.is_symlink() and STEP.fullmatch(p.name)]
    except OSError:
        return []
    return sorted(names, reverse=True)[:MAX_STEPS]


def episodes(root: Path, name: str):
    """Read only recent completed episodes; leave active rollouts alone.

    The number of result files read is capped. No scans of the full RL history,
    image decoding, thumbnailing, or calls to model/Docker are performed.
    """
    path = run_path(root, name)
    rows = []
    steps = _recent_steps(path / "rollouts")
    for step in steps:
        directory = path / "rollouts" / step
        try:
            prompts = sorted((p for p in directory.iterdir()
                              if p.is_dir() and not p.is_symlink()), key=lambda p: p.name)
        except OSError:
            continue
        for prompt in prompts:
            if not PART.fullmatch(prompt.name):
                continue
            try:
                samples = sorted((p for p in prompt.iterdir()
                                  if p.is_dir() and not p.is_symlink()), key=lambda p: p.name)
            except OSError:
                continue
            for sample in samples:
                if not EPISODE.fullmatch(sample.name):
                    continue
                result = load_json(sample / "result.json")
                if (not isinstance(result, dict) or not isinstance(result.get("scores"), dict)
                        or not isinstance(result["scores"].get("reward"), (int, float))):
                    # grade() writes result.json before compute_group_rewards().
                    # A terminal reward marks the end of rollout scoring.
                    continue
                scores = result["scores"]
                rows.append({
                    "step": step,
                    "promptId": prompt.name,
                    "episodeId": sample.name,
                    "prompt": str(result.get("prompt", ""))[:600],
                    "outcome": result.get("outcome"),
                    "expectedRefusal": result.get("expectedRefusal") is True,
                    "unsafe": result.get("unsafe") is True,
                    "score": scores.get("reward"),
                    "adherence": scores.get("adherence"),
                    "renderValid": scores.get("render_valid") == 1,
                    "bashCalls": len(result["commands"]) if isinstance(result.get("commands"), list) else 0,
                    "previewAttempts": result.get("previewAttempts"),
                    "hasImage": (sample / "image.png").is_file(),
                    "hasSource": (sample / "program.py").is_file(),
                })
                if len(rows) >= MAX_EPISODES:
                    return {"run": name, "steps": steps, "episodes": rows,
                            "limited": True, "maxEpisodes": MAX_EPISODES}
    return {"run": name, "steps": steps, "episodes": rows,
            "limited": False, "maxEpisodes": MAX_EPISODES}


def artifact(root: Path, name: str, step: str, prompt: str, episode: str, filename: str):
    """Validate the requested artifact before reading any bytes."""
    path = run_path(root, name)
    if not STEP.fullmatch(step) or not PART.fullmatch(prompt) or not EPISODE.fullmatch(episode):
        raise ValueError("Invalid RL episode identifier")
    if filename not in ("result.json", "program.py", "image.png"):
        raise ValueError("Unsupported RL artifact")
    file = child(path, "rollouts", step, prompt, episode, filename)
    limit = {
        "result.json": MAX_RESULT_BYTES,
        "program.py": MAX_SOURCE_BYTES,
        "image.png": MAX_IMAGE_BYTES,
    }[filename]
    if not file.is_file() or file.stat().st_size > limit:
        raise ValueError("RL artifact unavailable")
    return file
