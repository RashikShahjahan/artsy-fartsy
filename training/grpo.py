"""Correctness-only multi-turn RL with Tinker Cookbook.

python -m training.grpo --dry-run
python -m training.grpo --max-steps 1 --output-dir checkpoints/grpo-pilot

The Cookbook owns rollout sampling, grouped advantages, PPO updates and
checkpointing. Artsy's Docker tool environment supplies image observations
and a separately rendered correctness reward. Inspect is used for evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import subprocess

from drawing.config import IMAGE_SIZE, MODEL_ID, MODEL_REVISION
from drawing.eval_specs import load_specs
from training.data import ROOT, digest
from training.sft import RENDERER, batches

ALGORITHM = "tinker-cookbook-ppo-v1"
PROVENANCE_FILES = (
    "docs/artcanvas.md", "drawing/config.py", "drawing/judges.py",
    "drawing/agent_contract.py", "drawing/eval_specs.py",
    "runtime/artcanvas.py", "runtime/cairo.py", "runtime/requirements.txt",
    "evaluation/harness.py", "evaluation/host_agent.py", "evaluation/mini.yaml",
    "evaluation/render.py", "evaluation/Dockerfile", "evaluation/compose.yaml",
    "evaluation/requirements.txt", "training/cookbook_env.py",
    "training/cookbook_dataset.py", "training/score_rl.py", "training/grpo.py",
    "training/sft.py", "training/requirements.txt", "training/requirements-grpo.txt",
)


def load_run(args):
    specs = load_specs(args.dataset)
    if any(row.get("kind") == "refusal" for row in specs):
        raise ValueError("Correctness-only RL requires benign drawing prompts")
    if args.limit:
        specs = specs[:args.limit]
    initial = json.loads(args.init_checkpoint.read_text())
    source = initial.get("config", {})
    if (source.get("model") != MODEL_ID or source.get("tokenizer_revision") != MODEL_REVISION
            or source.get("renderer") != RENDERER or type(source.get("lora_rank")) is not int
            or source["lora_rank"] < 1
            or source.get("api_doc_sha256") != digest((ROOT / "docs/artcanvas.md").read_bytes())
            or any(not isinstance(initial.get(key), str) or not initial[key].startswith("tinker://")
                   for key in ("state_path", "sampler_path"))):
        raise ValueError("Initial checkpoint must contain compatible SFT weights and sampler paths")
    config = {
        "algorithm": ALGORITHM, "model": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "renderer": RENDERER, "image_size": IMAGE_SIZE, "lora_rank": source["lora_rank"],
        "dataset_sha256": digest(args.dataset.read_bytes()),
        "prompt_ids": [row["id"] for row in specs],
        "initial_state_path": initial["state_path"],
        "reference_sampler_path": initial["sampler_path"],
        "api_doc_sha256": digest((ROOT / "docs/artcanvas.md").read_bytes()),
        "source_sha256": {p: digest((ROOT / p).read_bytes()) for p in PROVENANCE_FILES},
        "batch_size": args.batch_size, "group_size": args.group_size,
        "learning_rate": args.learning_rate, "clip_epsilon": args.clip_epsilon,
        "kl_coef": args.kl_coef, "normalize_rewards": args.normalize_rewards,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "max_length": args.max_length, "max_samples": args.max_samples,
        "seed": args.seed, "epochs": args.epochs, "max_steps": args.max_steps,
        "correctness_judge_model": os.getenv("ARTSY_CORRECTNESS_JUDGE_MODEL"),
        "correctness_judge_url": os.getenv("ARTSY_CORRECTNESS_JUDGE_URL"),
        "reward": "final-image correctness only",
        "loss": "Cookbook PPO; optional reference-KL reward penalty",
    }
    total = math.ceil(len(specs) / args.batch_size) * args.epochs
    if args.max_steps != -1:
        total = min(total, args.max_steps)
    return specs, config, total


def make_cookbook_config(args, specs, plan, total):
    """Build the published Tinker Cookbook trainer configuration."""
    from tinker_cookbook.rl import train as cookbook
    from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
    from training.cookbook_dataset import ArtsyDatasetBuilder

    return cookbook.Config(
        log_path=str(args.output_dir.resolve()),
        model_name=MODEL_ID,
        recipe_name="artsy_rl",
        renderer_name=RENDERER,
        load_checkpoint_path=plan["initial_state_path"],
        lora_rank=plan["lora_rank"],
        dataset_builder=ArtsyDatasetBuilder(
            specs=tuple(specs), output_dir=str((args.output_dir / "rollouts").resolve()),
            batch_size=args.batch_size, group_size=args.group_size,
            epochs=args.epochs, seed=args.seed,
            max_tokens=args.max_tokens, max_length=args.max_length,
            normalize_rewards=args.normalize_rewards,
            judge_model=plan["correctness_judge_model"],
            judge_url=plan["correctness_judge_url"],
        ),
        learning_rate=args.learning_rate,
        # The Cookbook rollout cap is cumulative; RolloutLimits separately
        # enforces the per-turn max_tokens and context-size max_length.
        max_tokens=args.max_length,
        temperature=args.temperature,
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 1 - args.clip_epsilon,
                        "clip_high_threshold": 1 + args.clip_epsilon},
        kl_penalty_coef=args.kl_coef,
        kl_reference_config=(cookbook.KLReferenceConfig(
            base_model=MODEL_ID, load_checkpoint_path=plan["reference_sampler_path"])
            if args.kl_coef else None),
        remove_constant_reward_groups=True,
        eval_every=0,
        save_every=args.save_every,
        max_steps=total,
        # Terminations from tool/turn/token limits must not earn positive reward.
        termination=TerminationRewardPolicy(zero_reward_on_limit=True),
    )


def _build_sandbox_image():
    try:
        subprocess.run(
            ["docker", "build", "--file", "evaluation/Dockerfile",
             "--tag", "artsy-rl-runtime", "."],
            cwd=ROOT, check=True, timeout=600)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Cannot build the RL Docker sandbox; start Docker and check its build output") from error


def artsy_final_checkpoint(record, plan):
    """Convert Cookbook's named final record into Artsy's model-catalog format.

    Cookbook v0.5.8 records the final checkpoint with ``name='final'``,
    while its optional ``final`` field may be None even on a successful run.
    """
    if (record is None or record.name != "final"
            or not record.state_path or not record.sampler_path):
        raise RuntimeError("Cookbook did not produce a final state and sampler checkpoint")
    return {"step": record.batch, "state_path": record.state_path,
            "sampler_path": record.sampler_path, "config": plan}


def write_final_checkpoint(output_dir: Path, plan: dict):
    """Publish the completed Cookbook checkpoint for Artsy's model catalog."""
    from tinker_cookbook import checkpoint_utils

    record = checkpoint_utils.get_last_checkpoint(str(output_dir))
    final = artsy_final_checkpoint(record, plan)
    destination = output_dir / "final.json"
    if destination.exists() and json.loads(destination.read_text()) != final:
        raise ValueError("An incompatible final.json already exists in the output directory")
    destination.write_text(json.dumps(final, indent=2) + "\n")
    return final


async def train(args, specs, plan, total):
    from tinker_cookbook.rl import train as cookbook
    from training.cookbook_env import configure_concurrency

    configure_concurrency(args.max_samples)
    config = make_cookbook_config(args, specs, plan, total)
    await cookbook.main(config)
    # Retain Artsy's model-catalog format for make serve, evaluation and the UI.
    final = write_final_checkpoint(args.output_dir, plan)
    print(json.dumps(final, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets/grpo-v1/prompts.jsonl")
    p.add_argument("--init-checkpoint", type=Path, default=ROOT / "checkpoints/sft-v1/final.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/grpo-v1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--finalize-only", action="store_true",
                   help="Recover final.json from a completed Cookbook run without training")
    p.add_argument("--limit", type=int, help="First N prompts for a pilot")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=2, help="Prompt groups per update")
    p.add_argument("--group-size", type=int, default=4, help="Independent trajectories per prompt")
    p.add_argument("--max-samples", type=int, default=4, help="Maximum concurrent Docker episodes")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    # Cookbook recommends T=1 when using a reference-policy KL penalty.
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.01)
    p.add_argument("--normalize-rewards", action="store_true")
    p.add_argument("--max-tokens", type=int, default=4096, help="Maximum generated tokens per turn")
    p.add_argument("--max-length", type=int, default=32768, help="Maximum trajectory context tokens")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1)
    return p


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    p = parser()
    args = p.parse_args(argv)
    if args.finalize_only:
        if args.dry_run:
            p.error("--finalize-only and --dry-run are mutually exclusive")
        try:
            recorded = json.loads((args.output_dir / "training-manifest.json").read_text())
            config = recorded["config"]
            if not isinstance(config, dict) or config.get("algorithm") != ALGORITHM:
                raise ValueError("Output directory does not contain a compatible Cookbook run")
            print(json.dumps(write_final_checkpoint(args.output_dir, config), indent=2))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            p.error(str(error))
        return 0
    if (any(value < 1 for value in (args.epochs, args.batch_size, args.max_samples,
                                   args.save_every, args.max_tokens, args.max_length))
            or args.group_size < 2 or args.max_steps == 0 or args.max_steps < -1
            or (args.limit is not None and args.limit < 1)
            or not 0 <= args.seed <= 2**32 - 1 or args.max_length > 32768
            or args.max_tokens > args.max_length
            or any(not math.isfinite(v) for v in
                   (args.learning_rate, args.temperature, args.clip_epsilon, args.kl_coef))
            or args.learning_rate <= 0 or not 0 < args.temperature <= 2
            or not 0 < args.clip_epsilon < 1 or args.kl_coef < 0):
        p.error("Invalid training hyperparameters")
    try:
        specs, config, total = load_run(args)
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    plan = {"config": config, "prompts": len(specs), "total_steps": total,
            "episodes": sum(len(batch) * args.group_size for i, batch in enumerate(
                batches(specs, args.batch_size, args.epochs, args.seed)) if i < total)}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    if not config["correctness_judge_model"] or not config["correctness_judge_url"]:
        p.error("Set ARTSY_CORRECTNESS_JUDGE_MODEL and ARTSY_CORRECTNESS_JUDGE_URL")
    if args.output_dir.resolve() == args.init_checkpoint.resolve().parent:
        p.error("RL output must differ from the initial SFT checkpoint directory")
    manifest = args.output_dir / "training-manifest.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not manifest.is_file() or not (args.output_dir / "checkpoints.jsonl").is_file():
            p.error("Output is not a resumable Cookbook run; choose a new directory")
        if json.loads(manifest.read_text()) != plan:
            p.error("Training configuration changed; resume with the original settings")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
    _build_sandbox_image()
    asyncio.run(train(args, specs, config, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
