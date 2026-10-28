"""Multi-signal RLHF multi-turn RL with Tinker Cookbook.

python -m training.grpo --dry-run
python -m training.grpo --max-steps 1 --output-dir checkpoints/grpo-pilot

The Cookbook owns rollout sampling, grouped advantages, PPO updates and
checkpointing. Artsy's Docker tool environment supplies image observations
and independent frozen visual, adherence, content-safety and code-safety rewards. Inspect is used for evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from drawing.config import IMAGE_SIZE, MODEL_ID, MODEL_REVISION, MODEL_CONTEXT_WINDOW
from drawing.eval_specs import load_specs
from training.data import ROOT, digest
from training.sft import RENDERER, batches

ALGORITHM = "tinker-cookbook-rlhf-v1"
PROVENANCE_FILES = (
    "docs/artcanvas.md", "drawing/config.py", "drawing/judges.py",
    "drawing/agent_contract.py", "drawing/eval_specs.py",
    "runtime/artcanvas.py", "runtime/cairo.py", "runtime/requirements.txt",
    "evaluation/harness.py", "evaluation/host_agent.py", "evaluation/mini.yaml",
    "evaluation/render.py", "evaluation/Dockerfile", "evaluation/compose.yaml",
    "evaluation/requirements.txt", "training/cookbook_env.py",
    "training/cookbook_dataset.py", "training/score_rl.py", "training/grpo.py",
    "training/sft.py", "training/requirements.txt", "training/requirements-grpo.txt",
    "drawing/dataset_client.py", "drawing/tinker_client.py", "preference/judge.py",
)


def judge_settings(prefix):
    return {"model": os.getenv(prefix + "_MODEL"), "baseUrl": os.getenv(prefix + "_URL"),
            "renderer": os.getenv(prefix + "_RENDERER", "qwen3_8_medium_reasoning"),
            "revision": os.getenv(prefix + "_REVISION"),
            "maxTokens": int(os.getenv(prefix + "_MAX_TOKENS", "8192")),
            "timeoutMs": 300000}


def load_run(args):
    specs = load_specs(args.dataset)
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
        "max_length": args.max_length, "max_sampled_tokens": None, "max_samples": args.max_samples,
        "seed": args.seed, "epochs": args.epochs, "max_steps": args.max_steps,
        "correctness_judge_model": os.getenv("ARTSY_CORRECTNESS_JUDGE_MODEL"),
        "correctness_judge_url": os.getenv("ARTSY_CORRECTNESS_JUDGE_URL"),
        "correctness_judge_renderer": os.getenv("ARTSY_CORRECTNESS_JUDGE_RENDERER", "qwen3_8_medium_reasoning"),
        "correctness_judge_revision": os.getenv("ARTSY_CORRECTNESS_JUDGE_REVISION"),
        "correctness_judge_max_tokens": int(os.getenv("ARTSY_CORRECTNESS_JUDGE_MAX_TOKENS", "16384")),
        "reward": "unsafe=-1; correct refusal=1; overrefusal/invalid=0; benign=0.5*visual+0.5*adherence",
        "reward_version": "artsy-multisignal-rlhf-v1",
        "visual_model": args.visual_model,
        "visual_sampler_path": args.visual_sampler_path,
        "visual_renderer": args.visual_renderer,
        "visual_revision": args.visual_revision,
        "visual_max_tokens": args.visual_max_tokens,
        "safety_config": judge_settings("ARTSY_SAFETY_JUDGE"),
        "code_safety_config": judge_settings("ARTSY_CODE_SAFETY_JUDGE"),
        "loss": "Cookbook PPO; optional reference-KL reward penalty",
    }
    total = math.ceil(len(specs) / args.batch_size) * args.epochs
    if args.max_steps != -1:
        total = min(total, args.max_steps)
    return specs, config, total


INHERITED_PARAMETERS = (
    "batch_size", "group_size", "learning_rate", "clip_epsilon", "kl_coef",
    "normalize_rewards", "temperature", "max_tokens", "max_length",
    "max_samples", "seed", "visual_model", "visual_sampler_path",
    "visual_renderer", "visual_revision", "visual_max_tokens",
)


def load_continuation(args):
    """Plan a bounded fork, retaining global batch indices and optimizer state."""
    origin = args.continue_from.resolve()
    if args.output_dir.resolve() == origin or origin in args.output_dir.resolve().parents:
        raise ValueError("Continuation requires a new output directory")
    manifest_bytes = (origin / "training-manifest.json").read_bytes()
    saved = json.loads(manifest_bytes)
    previous = saved["config"]
    final = json.loads((origin / "final.json").read_text())
    records = [json.loads(line) for line in (origin / "checkpoints.jsonl").read_text().splitlines()
               if line.strip()]
    record = records[-1] if records else {}
    start = final["step"]
    if (previous.get("algorithm") != ALGORITHM or final.get("config") != previous
            or type(start) is not int or start < 1 or start != saved["total_steps"]
            or record.get("name") != "final" or record.get("batch") != start
            or any(record.get(k) != final.get(k) or not isinstance(final.get(k), str)
                   or not final[k].startswith("tinker://")
                   for k in ("state_path", "sampler_path"))):
        raise ValueError("Continuation source must be a completed compatible Cookbook run")
    current_hashes = {p: digest((ROOT / p).read_bytes()) for p in PROVENANCE_FILES}
    old_hashes = previous["source_sha256"]
    if set(old_hashes) != set(current_hashes) or any(
            old_hashes[p] != h for p, h in current_hashes.items()):
        raise ValueError("Continuation source runtime or training dependencies changed")
    if (previous["model"] != MODEL_ID or previous["tokenizer_revision"] != MODEL_REVISION
            or previous["renderer"] != RENDERER or previous["image_size"] != IMAGE_SIZE
            or digest(args.dataset.read_bytes()) != previous["dataset_sha256"]):
        raise ValueError("Continuation model or dataset changed")
    all_specs = load_specs(args.dataset)
    wanted = previous["prompt_ids"]
    specs = all_specs[:len(wanted)]
    if not specs or [s["id"] for s in specs] != wanted:
        raise ValueError("Continuation prompt selection changed")
    for key in INHERITED_PARAMETERS:
        setattr(args, key, previous[key])
    total = start + args.additional_steps
    args.epochs = math.ceil(total / math.ceil(len(specs) / args.batch_size))
    args.max_steps = total
    config = {**previous, "epochs": args.epochs, "max_steps": total,
              "source_sha256": current_hashes,
              "continuation": {"source_dir": str(origin), "source_manifest_sha256": digest(manifest_bytes),
                               "start_step": start, "additional_steps": args.additional_steps,
                               "state_path": final["state_path"],
                               "sampler_path": final["sampler_path"],
                               "source_sha256": old_hashes}}
    return specs, config, total


def seed_continuation(output_dir, config):
    """Use Cookbook's resume branch, which loads weights AND optimizer state.

    load_checkpoint_path alone deliberately resets the optimizer in Cookbook.
    The synthetic checkpoint only points at the immutable source; later records
    and rollouts belong to this new directory and use cumulative batch indices.
    """
    path = output_dir / "checkpoints.jsonl"
    if "continuation" in config and not path.exists():
        source = config["continuation"]
        record = {"name": "continuation", "batch": source["start_step"],
                  "state_path": source["state_path"], "sampler_path": source["sampler_path"]}
        path.write_text(json.dumps(record) + "\n")


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
            judge_renderer=plan.get("correctness_judge_renderer", "qwen3_8_medium_reasoning"),
            judge_revision=plan.get("correctness_judge_revision"),
            judge_max_tokens=plan.get("correctness_judge_max_tokens", 16384),
            **{key: plan[key] for key in ("visual_model", "visual_sampler_path",
                "visual_renderer", "visual_revision", "visual_max_tokens",
                "safety_config", "code_safety_config")},
        ),
        learning_rate=args.learning_rate,
        # Cookbook max_tokens is a per-call cap; no aggregate sampled cap.
        max_tokens=args.max_tokens,
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
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets/grpo-v1/prompts.jsonl")
    p.add_argument("--init-checkpoint", type=Path, default=ROOT / "checkpoints/sft-v1/final.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/grpo-v1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-from", type=Path, help="Completed RL run to fork with optimizer state")
    p.add_argument("--additional-steps", type=int, help="Positive number of extra updates")
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
    p.add_argument("--normalize-rewards", action="store_true",
                   help="Deprecated compatibility flag: Tinker mean-centers without std scaling")
    p.add_argument("--visual-model", default=os.getenv("ARTSY_VISUAL_JUDGE_MODEL", "Qwen/Qwen3.5-9B"))
    p.add_argument("--visual-sampler-path", default=os.getenv("ARTSY_VISUAL_JUDGE_SAMPLER_PATH", ""))
    p.add_argument("--visual-renderer", default=os.getenv("ARTSY_VISUAL_JUDGE_RENDERER"))
    p.add_argument("--visual-revision", default=os.getenv("ARTSY_VISUAL_JUDGE_REVISION"))
    p.add_argument("--visual-max-tokens", type=int, default=int(os.getenv("ARTSY_VISUAL_JUDGE_MAX_TOKENS", "2048")))
    p.add_argument("--max-tokens", type=int, default=8192, help="Maximum generated tokens per turn")
    p.add_argument("--max-length", type=int, default=MODEL_CONTEXT_WINDOW, help="Maximum trajectory context tokens")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1)
    return p


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    p = parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(arguments)
    if args.continue_from is not None:
        forbidden = {"--" + k.replace("_", "-") for k in INHERITED_PARAMETERS}
        forbidden.update({"--epochs", "--max-steps", "--limit", "--init-checkpoint"})
        if any(token.split("=", 1)[0] in forbidden for token in arguments):
            p.error("Continuation inherits training parameters; only dataset, output and save cadence may be set")
        if args.additional_steps is None or args.additional_steps < 1 or args.finalize_only:
            p.error("--continue-from requires positive --additional-steps and cannot use --finalize-only")
    elif args.additional_steps is not None:
        p.error("--additional-steps requires --continue-from")
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
            or args.visual_max_tokens < 1 or args.group_size < 2 or args.max_steps == 0 or args.max_steps < -1
            or (args.limit is not None and args.limit < 1)
            or not 0 <= args.seed <= 2**32 - 1 or args.max_length > MODEL_CONTEXT_WINDOW
            or args.max_tokens > args.max_length
            or any(not math.isfinite(v) for v in
                   (args.learning_rate, args.temperature, args.clip_epsilon, args.kl_coef))
            or args.learning_rate <= 0 or not 0 < args.temperature <= 2
            or not 0 < args.clip_epsilon < 1 or args.kl_coef < 0):
        p.error("Invalid training hyperparameters")
    try:
        specs, config, total = (load_continuation(args) if args.continue_from else load_run(args))
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        p.error(str(error))
    start = config.get("continuation", {}).get("start_step", 0)
    plan = {"config": config, "prompts": len(specs), "total_steps": total,
            "episodes": sum(len(batch) * args.group_size for i, batch in enumerate(
                batches(specs, args.batch_size, args.epochs, args.seed)) if start <= i < total)}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    if not config["correctness_judge_model"] or not config["correctness_judge_url"]:
        p.error("Set ARTSY_CORRECTNESS_JUDGE_MODEL and ARTSY_CORRECTNESS_JUDGE_URL")
    if not config.get("visual_sampler_path", "").startswith("tinker://"):
        p.error("Set --visual-sampler-path to a trained frozen Tinker judge checkpoint")
    for role in ("safety_config", "code_safety_config"):
        if not config.get(role, {}).get("model") or not config[role].get("baseUrl"):
            p.error("Set ARTSY_SAFETY_JUDGE_MODEL/URL and ARTSY_CODE_SAFETY_JUDGE_MODEL/URL")
    if args.output_dir.resolve() == args.init_checkpoint.resolve().parent:
        p.error("RL output must differ from the initial SFT checkpoint directory")
    manifest = args.output_dir / "training-manifest.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not manifest.is_file() or (not args.continue_from and not (args.output_dir / "checkpoints.jsonl").is_file()):
            p.error("Output is not a resumable Cookbook run; choose a new directory")
        if json.loads(manifest.read_text()) != plan:
            p.error("Training configuration changed; resume with the original settings")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
    seed_continuation(args.output_dir, config)
    _build_sandbox_image()
    asyncio.run(train(args, specs, config, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
