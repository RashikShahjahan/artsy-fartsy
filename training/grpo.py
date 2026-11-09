"""Final-image RLHF multi-turn RL with Tinker Cookbook.

python -m training.grpo --dry-run
python -m training.grpo --max-steps 1 --output-dir checkpoints/grpo-pilot

The Cookbook owns rollout sampling, grouped advantages, PPO updates and
checkpointing. Artsy's Docker tool environment supplies image observations
and final-image HPSv3 preference rewards. Inspect is used for evaluation.
"""

from __future__ import annotations

from drawing.documentation import api_documentation

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from drawing.config import IMAGE_SIZE, MODEL_CONTEXT_WINDOW
from drawing.eval_specs import load_specs
from training.hpsv3 import REWARD_VERSION, normalized_reward
from training.rlvr_reward import RLVR_ALGORITHM, RLVR_REWARD_VERSION
from training.rl_split import stage_specs
from drawing.agent_contract import DRAWING_TOOLS_CONTRACT_VERSION
from drawing.mini_agent import CONTRACT_VERSION as MINI_SWE_CONTRACT_VERSION
from training.data import ROOT, digest
from training.sft import batches
from training.run_store import atomic_json, atomic_text, contents, recover_jsonl_tail, run_lock

ALGORITHM = "tinker-cookbook-rlhf-v8"
PROVENANCE_FILES = (
    "drawing/documentation.py", "drawing/config.py", "drawing/judges.py",
    "drawing/agent_contract.py", "drawing/eval_specs.py", "drawing/visual_judge.py",
    "drawing/mini_agent.py", "drawing/mini_session.py", "drawing/mini_protocol.py",
    "evaluation/agent_commands.py",
    "drawing/render_contract.py", "drawing/model_profiles.py", "drawing/vision.py",
    "training/run_store.py", "training/hpsv3.py", "training/hpsv3_server.py",
    "runtime/requirements.txt", "runtime/artcanvas.py", "docs/artcanvas.md",
    "evaluation/harness.py", "evaluation/host_agent.py", "evaluation/mini.yaml",
    "evaluation/render.py", "evaluation/Dockerfile", "evaluation/compose.yaml",
    "evaluation/requirements.txt", "training/cookbook_env.py", "training/cookbook_mini.py",
    "training/cookbook_dataset.py", "training/score_rl.py", "training/grpo.py",
    "training/sft.py", "training/rlvr.py", "training/rlhf.py", "training/rlvr_reward.py",
    "training/rl_split.py",
    "training/requirements.txt", "training/requirements-grpo.txt",
    "drawing/dataset_client.py", "drawing/tinker_client.py",
)

def checkpoint_tool_contract(path: Path, source: dict) -> str | None:
    """Recover the tool contract for checkpoints created before SFT tagged it.

    Older SFT manifests retained the number of mini-SWE records, even though
    their final.json lacked an explicit tool_contract field.
    """
    tagged = source.get("tool_contract")
    if tagged is not None:
        return tagged
    manifest_file = path.parent / "training-manifest.json"
    if not manifest_file.is_file():
        return None
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("config") != source:
        raise ValueError("SFT manifest and checkpoint configuration do not match")
    total = manifest.get("records")
    mini = manifest.get("mini_swe_trajectories")
    if type(total) is not int or total < 1 or type(mini) is not int:
        return None
    if mini == total:
        return "mini-swe"
    if mini == 0 and manifest.get("drawing_tool_trajectories") in (None, total):
        # Pre-metadata SFT manifests did not record named-tool counts.
        return "drawing-tools"
    # A checkpoint trained on heterogeneous or historical bash demonstrations
    # must not be assumed compatible with the new mini-SWE tool interface.
    return "mixed-or-legacy"


def load_run(args):
    specs, stage_split_version = stage_specs(args.dataset, load_specs(args.dataset), args.reward_mode)
    if args.limit:
        specs = specs[:args.limit]
    if args.reward_mode == "rlvr" and args.harness != "mini-swe":
        raise ValueError("RLVR requires --harness mini-swe for verified bash execution outcomes")
    if args.harness == "mini-swe" and any(spec.get("kind") == "refusal" for spec in specs):
        raise ValueError("mini-SWE GRPO requires drawing-only prompts; refusal examples have no positive reward")
    initial = json.loads(args.init_checkpoint.read_text())
    source = initial.get("config", {})
    checkpoint_model = source.get("model")
    if isinstance(checkpoint_model, str) and ":peft:" in checkpoint_model:
        checkpoint_model = checkpoint_model.split(":peft:", 1)[0]
    source_stage = source.get("algorithm")
    if (source_stage not in (None, RLVR_ALGORITHM)
            or (args.reward_mode == "rlvr" and source_stage is not None)
            or (source_stage == RLVR_ALGORITHM and source.get("harness") != args.harness)
            or not isinstance(checkpoint_model, str)
            or not checkpoint_model.startswith("Qwen/")
            or not isinstance(source.get("renderer"), str)
            or type(source.get("lora_rank")) is not int
            or source["lora_rank"] < 1
            or source.get("image_size", IMAGE_SIZE) != IMAGE_SIZE
            or source.get("api_doc_sha256") != digest(api_documentation().encode())
            or any(not isinstance(initial.get(key), str) or not initial[key].startswith("tinker://")
                   for key in ("state_path", "sampler_path"))):
        raise ValueError("Initial checkpoint must contain compatible SFT or RLVR weights and sampler paths")
    if source_stage == RLVR_ALGORITHM:
        reference_model = source.get("reference_model")
        reference_sampler_path = source.get("reference_sampler_path")
        if (not isinstance(reference_model, str) or not reference_model.startswith("Qwen/")
                or not isinstance(reference_sampler_path, str)
                or not reference_sampler_path.startswith("tinker://")):
            raise ValueError("RLVR checkpoint is missing its frozen SFT KL reference")
        recorded_contract = source["harness"]
    else:
        # Preserve the hosted context selection for the frozen SFT sampler.
        reference_model = (source["model"] if ":peft:" in source["model"]
                           else f"{checkpoint_model}:peft:{MODEL_CONTEXT_WINDOW}")
        reference_sampler_path = initial["sampler_path"]
        recorded_contract = checkpoint_tool_contract(args.init_checkpoint, source)
    if recorded_contract is None:
        print("Warning: SFT tool contract is unknown; verify its training dataset matches "
              f"--harness {args.harness} before training.", file=sys.stderr)
    if recorded_contract is not None and recorded_contract != args.harness:
        raise ValueError(
            f"SFT checkpoint uses {recorded_contract}, not {args.harness}; select the matching --harness")
    config = {
        "algorithm": RLVR_ALGORITHM if args.reward_mode == "rlvr" else ALGORITHM,
        "reward_mode": args.reward_mode, "model": checkpoint_model,
        "harness": args.harness, "checkpoint_tool_contract": recorded_contract,
        "trajectory_contract_version": (
            MINI_SWE_CONTRACT_VERSION if args.harness == "mini-swe"
            else DRAWING_TOOLS_CONTRACT_VERSION),
        "tokenizer_revision": source.get("tokenizer_revision"),
        "renderer": source["renderer"], "image_size": IMAGE_SIZE,
        "lora_rank": source["lora_rank"],
        "dataset_sha256": digest(args.dataset.read_bytes()),
        "rl_stage_split_version": stage_split_version,
        "prompt_ids": [row["id"] for row in specs],
        "initial_state_path": initial["state_path"],
        "reference_model": reference_model,
        "reference_sampler_path": reference_sampler_path,
        "api_doc_sha256": digest(api_documentation().encode()),
        "source_sha256": {p: digest((ROOT / p).read_bytes()) for p in PROVENANCE_FILES},
        "batch_size": args.batch_size, "group_size": args.group_size,
        "learning_rate": args.learning_rate, "clip_epsilon": args.clip_epsilon,
        "kl_coef": args.kl_coef, "normalize_rewards": args.normalize_rewards,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "max_length": args.max_length, "max_sampled_tokens": None, "max_samples": args.max_samples,
        "seed": args.seed, "epochs": args.epochs, "max_steps": args.max_steps,
        "hpsv3_model": os.getenv("ARTSY_HPSV3_MODEL") if args.reward_mode == "rlhf" else None,
        "hpsv3_url": os.getenv("ARTSY_HPSV3_URL", "http://127.0.0.1:8090"),
        "hpsv3_center": float(os.getenv("ARTSY_HPSV3_CENTER", "0")),
        "hpsv3_scale": float(os.getenv("ARTSY_HPSV3_SCALE", "5")),
        "reward": ("penalty-only: -0.75 missing/invalid final PNG, -0.125 per invalid bash "
                   "tool call (or no calls), -0.125 per failed command/preview; best=0"
                   if args.reward_mode == "rlvr"
                   else "submitted final image=sigmoid((hpsv3_mu-center)/scale); missing/refused/invalid=0"),
        "reward_version": RLVR_REWARD_VERSION if args.reward_mode == "rlvr" else REWARD_VERSION,
        "loss": "Cookbook PPO; optional reference-KL reward penalty",
    }
    if args.reward_mode == "rlhf":
        normalized_reward(0.0, config["hpsv3_center"], config["hpsv3_scale"])
    total = math.ceil(len(specs) / args.batch_size) * args.epochs
    if args.max_steps != -1:
        total = min(total, args.max_steps)
    return specs, config, total


INHERITED_PARAMETERS = (
    "batch_size", "group_size", "learning_rate", "clip_epsilon", "kl_coef",
    "normalize_rewards", "temperature", "max_tokens", "max_length",
    "max_samples", "seed", "harness",
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
    expected_algorithm = RLVR_ALGORITHM if args.reward_mode == "rlvr" else ALGORITHM
    if (previous.get("algorithm") != expected_algorithm or final.get("config") != previous
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
    if (not previous["model"].startswith("Qwen/")
            or previous["image_size"] != IMAGE_SIZE
            or digest(args.dataset.read_bytes()) != previous["dataset_sha256"]):
        raise ValueError("Continuation model or dataset changed")
    all_specs, stage_split_version = stage_specs(
        args.dataset, load_specs(args.dataset), args.reward_mode)
    if previous.get("rl_stage_split_version") != stage_split_version:
        raise ValueError("Continuation RL stage split changed")
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
        atomic_text(path, json.dumps(record) + "\n")


def validate_resume(output_dir, config, total, *, finalize=False):
    """Reject unrelated or malformed histories and recover a torn final append."""
    path = output_dir / "checkpoints.jsonl"
    if not finalize and (output_dir / "final.json").exists():
        raise ValueError("Run is already finalized")
    if not path.exists():
        if finalize:
            raise ValueError("Missing checkpoint history")
        if any(p.name != "training-manifest.json" for p in contents(output_dir)):
            raise ValueError("Output has artifacts but no checkpoint history")
        return

    def validate_records(records):
        continuation = config.get("continuation")
        if continuation:
            first = records[0]
            if (first.get("name") != "continuation" or first.get("batch") != continuation["start_step"]
                    or any(first.get(k) != continuation[k] for k in ("state_path", "sampler_path"))):
                raise ValueError("Continuation checkpoint does not match its source")
        previous = -1
        for record in records:
            step = record.get("batch")
            if (type(step) is not int or step < previous or step > total
                    or not isinstance(record.get("state_path"), str)
                    or not record["state_path"].startswith("tinker://")):
                raise ValueError("Invalid checkpoint history")
            previous = step

    recover_jsonl_tail(path, validate_records)


def make_cookbook_config(args, specs, plan, total):
    """Build the published Tinker Cookbook trainer configuration."""
    from tinker_cookbook.rl import train as cookbook
    from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
    from training.cookbook_dataset import ArtsyDatasetBuilder

    return cookbook.Config(
        log_path=str(args.output_dir.resolve()),
        model_name=plan["model"],
        recipe_name="artsy_rl",
        renderer_name=plan["renderer"],
        load_checkpoint_path=plan["initial_state_path"],
        lora_rank=plan["lora_rank"],
        dataset_builder=ArtsyDatasetBuilder(
            specs=tuple(specs), output_dir=str((args.output_dir / "rollouts").resolve()),
            batch_size=args.batch_size, group_size=args.group_size,
            epochs=args.epochs, seed=args.seed,
            max_tokens=args.max_tokens, max_length=args.max_length,
            normalize_rewards=args.normalize_rewards,
            revision=plan.get("tokenizer_revision"),
            hpsv3_model=plan["hpsv3_model"], hpsv3_url=plan["hpsv3_url"],
            hpsv3_center=plan["hpsv3_center"], hpsv3_scale=plan["hpsv3_scale"],
            model=plan["model"],
            renderer=plan["renderer"],
            harness=plan.get("harness", "drawing-tools"),
            reward_mode=plan.get("reward_mode", "rlhf"),
        ),
        learning_rate=args.learning_rate,
        # Cookbook requires an integer ceiling. Without an explicit cap, use
        # context capacity; the rollout runner clips to remaining context.
        max_tokens=args.max_tokens or args.max_length,
        temperature=args.temperature,
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 1 - args.clip_epsilon,
                        "clip_high_threshold": 1 + args.clip_epsilon},
        kl_penalty_coef=args.kl_coef,
        kl_reference_config=(cookbook.KLReferenceConfig(
            base_model=plan["reference_model"], load_checkpoint_path=plan["reference_sampler_path"])
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


def write_final_checkpoint(output_dir: Path, plan: dict, total_steps=None):
    """Publish the completed Cookbook checkpoint for Artsy's model catalog."""
    from tinker_cookbook import checkpoint_utils

    record = checkpoint_utils.get_last_checkpoint(str(output_dir))
    final = artsy_final_checkpoint(record, plan)
    if total_steps is not None and final["step"] != total_steps:
        raise ValueError("Final checkpoint does not match the planned total steps")
    destination = output_dir / "final.json"
    if destination.exists() and json.loads(destination.read_text()) != final:
        raise ValueError("An incompatible final.json already exists in the output directory")
    atomic_json(destination, final)
    return final


async def train(args, specs, plan, total):
    from tinker_cookbook.rl import train as cookbook
    from training.cookbook_env import configure_concurrency

    configure_concurrency(args.max_samples, args.group_size)
    config = make_cookbook_config(args, specs, plan, total)
    from drawing.dataset_client import close_chat_clients
    try:
        await cookbook.main(config)
    finally:
        await close_chat_clients()
    # Retain Artsy's model-catalog format for make serve, evaluation and the UI.
    final = write_final_checkpoint(args.output_dir, plan, total)
    print(json.dumps(final, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets/parti-prompts/rl/prompts.jsonl")
    p.add_argument("--init-checkpoint", type=Path, default=ROOT / "checkpoints/sft-v2-qwen3.8-27b-final-preview-masked-cosine/final.json")
    p.add_argument("--reward-mode", choices=("rlvr", "rlhf"), default="rlhf",
                   help="Verifier rewards for RLVR; HPSv3 preference rewards for RLHF")
    p.add_argument("--harness", choices=("mini-swe", "drawing-tools"), default="mini-swe",
                   help="Coding environment; must match the SFT checkpoint tool interface")
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/grpo-v4")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-from", type=Path, help="Completed RL run to fork with optimizer state")
    p.add_argument("--additional-steps", type=int, help="Positive number of extra updates")
    p.add_argument("--finalize-only", action="store_true",
                   help="Recover final.json from a completed Cookbook run without training")
    p.add_argument("--limit", type=int, help="First N prompts for a pilot")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=8, help="Prompt groups per update")
    p.add_argument("--group-size", type=int, default=4, help="Independent trajectories per prompt")
    p.add_argument("--max-samples", type=int, default=32, help="Maximum concurrent Docker episodes")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    # Cookbook recommends T=1 when using a reference-policy KL penalty.
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.01)
    p.add_argument("--normalize-rewards", action="store_true",
                   help="Deprecated compatibility flag: Tinker mean-centers without std scaling")
    p.add_argument("--max-tokens", type=lambda value: int(value) or None, default=None,
                   help="Optional generated tokens per turn; 0 uses remaining context (default)")
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
            with run_lock(args.output_dir):
                recorded = json.loads((args.output_dir / "training-manifest.json").read_text())
                config = recorded["config"]
                expected_algorithm = RLVR_ALGORITHM if args.reward_mode == "rlvr" else ALGORITHM
                if not isinstance(config, dict) or config.get("algorithm") != expected_algorithm:
                    raise ValueError("Output directory does not contain a compatible Cookbook run")
                validate_resume(args.output_dir, config, recorded["total_steps"], finalize=True)
                print(json.dumps(write_final_checkpoint(args.output_dir, config, recorded["total_steps"]), indent=2))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            p.error(str(error))
        return 0
    if (any(value < 1 for value in (args.epochs, args.batch_size, args.max_samples,
                                   args.save_every, args.max_length))
            or args.group_size < 2 or args.max_samples < args.group_size or args.max_steps == 0 or args.max_steps < -1
            or (args.limit is not None and args.limit < 1)
            or not 0 <= args.seed <= 2**32 - 1 or args.max_length > MODEL_CONTEXT_WINDOW
            or (args.max_tokens is not None and not 1 <= args.max_tokens <= args.max_length)
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
    if args.reward_mode == "rlhf" and (not config["hpsv3_model"] or not config["hpsv3_url"]):
        p.error("Set ARTSY_HPSV3_MODEL to the /health model ID and ARTSY_HPSV3_URL")
    if args.output_dir.resolve() == args.init_checkpoint.resolve().parent:
        p.error("RL output must differ from the initial SFT checkpoint directory")
    try:
        with run_lock(args.output_dir):
            manifest = args.output_dir / "training-manifest.json"
            if contents(args.output_dir):
                if not manifest.is_file():
                    raise ValueError("Output is not a resumable Cookbook run; choose a new directory")
                if json.loads(manifest.read_text()) != plan:
                    raise ValueError("Training configuration changed; resume with the original settings")
                validate_resume(args.output_dir, config, total)
            else:
                atomic_json(manifest, plan)
            seed_continuation(args.output_dir, config)
            _build_sandbox_image()
            asyncio.run(train(args, specs, config, total))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        p.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
