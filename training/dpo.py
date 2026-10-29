"""Visual preference DPO for the drawing policy: SFT -> DPO -> RL.

python -m training.dpo --data DATA/labels.jsonl --trajectories DATA/accepted.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from drawing.config import (IMAGE_SIZE, MODEL_ID, MODEL_REVISION, MODEL_CONTEXT_WINDOW,
                            TINKER_SAMPLING_MODEL_ID)
from preference.data import load_pairs
from training.data import ROOT, digest, load_examples
from training.sft import RENDERER, batches, create_encoder, save_checkpoint

ALGORITHM = "artsy-visual-dpo-v1"
PROVENANCE_FILES = (
    "training/dpo.py", "training/data.py", "training/sft.py",
    "training/requirements.txt", "drawing/config.py", "preference/data.py",
    "preference/judge.py",
)


def load_preferences(data: Path, trajectories: Path):
    """Bind image labels to verified, same-context drawing trajectories."""
    pairs = load_pairs(data)
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    examples, manifest = load_examples(trajectories)
    records = {r["id"]: r for r in map(
        json.loads, filter(str.strip, trajectories.read_text().splitlines()))}
    by_id = {e["id"]: e for e in examples}
    splits = {}
    selected = []
    for pair, row in zip(pairs, rows, strict=True):
        candidates = []
        for key, image in (("record_a", pair.image_a), ("record_b", pair.image_b)):
            record_id = row.get(key)
            if not isinstance(record_id, str) or record_id not in by_id:
                raise ValueError(f"{pair.pair_id}: {key} must identify a recorded trajectory")
            record = records[record_id]
            if record.get("status") != "accepted" or record["spec"].get("kind") == "refusal":
                raise ValueError(f"{pair.pair_id}: visual DPO requires approved drawing trajectories")
            if record["imageHash"] != digest(image.read_bytes()):
                raise ValueError(f"{pair.pair_id}: label image does not match {record_id}")
            example = by_id[record_id]
            # The system/user prefix includes the original request and preview seed.
            context = json.dumps(example["messages"][:2], sort_keys=True)
            for identity in ("prompt:" + record["spec"]["prompt"], "record:" + record_id,
                             "source:" + record["sourceHash"]):
                if identity in splits and splits[identity] != pair.split:
                    raise ValueError(f"{pair.pair_id}: prompt or trajectory leaks across splits")
                splits[identity] = pair.split
            candidates.append((context, example))
        if candidates[0][0] != candidates[1][0]:
            raise ValueError(f"{pair.pair_id}: trajectories must have the same prompt and seed")
        if pair.split == "train" and pair.winner != "Tie":
            chosen, rejected = (candidates if pair.winner == "A" else candidates[::-1])
            selected.append({"id": pair.pair_id, "chosen": chosen[1], "rejected": rejected[1]})
    if not selected:
        raise ValueError("DPO requires non-tie training pairs")
    manifest.update(labels_sha256=digest(data.read_bytes()), pairs=len(pairs),
                    training_pairs=len(selected), ties=sum(p.winner == "Tie" for p in pairs),
                    examples_sha256=digest(json.dumps(examples, sort_keys=True).encode()))
    return selected, manifest


def load_initial_checkpoint(path: Path):
    checkpoint = json.loads(path.read_text())
    config = checkpoint.get("config", {})
    if (config.get("model") != MODEL_ID or config.get("tokenizer_revision") != MODEL_REVISION
            or config.get("renderer") != RENDERER
            or type(config.get("lora_rank")) is not int or config["lora_rank"] < 1
            or config.get("api_doc_sha256") != digest((ROOT / "docs/artcanvas.md").read_bytes())
            or config.get("algorithm") is not None
            or any(not isinstance(checkpoint.get(k), str) or not checkpoint[k].startswith("tinker://")
                   for k in ("state_path", "sampler_path"))):
        raise ValueError("DPO initialization requires a compatible SFT state and sampler checkpoint")
    return checkpoint


def make_plan(args, manifest, initial):
    total = math.ceil(manifest["training_pairs"] / args.batch_size) * args.epochs
    if args.max_steps != -1:
        total = min(total, args.max_steps)
    config = {
        "algorithm": ALGORITHM, "model": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "renderer": RENDERER, "image_size": IMAGE_SIZE,
        "lora_rank": initial["config"]["lora_rank"],
        "api_doc_sha256": manifest["api_doc_sha256"],
        "initial_state_path": initial["state_path"],
        "reference_sampler_path": initial["sampler_path"],
        "dataset_sha256": manifest["dataset_sha256"],
        "labels_sha256": manifest["labels_sha256"],
        "examples_sha256": manifest["examples_sha256"],
        "conversion": manifest["conversion"], "input_mode": manifest["input_mode"],
        "source_sha256": {p: digest((ROOT / p).read_bytes()) for p in PROVENANCE_FILES},
        "loss": "DPO on summed assistant-action logprobs; frozen SFT reference",
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "beta": args.beta, "epochs": args.epochs, "seed": args.seed,
        "max_length": args.max_length,
    }
    return {"config": config, "total_steps": total, "training_pairs": manifest["training_pairs"]}


def dpo_loss(data, logprobs, reference_logprobs, beta):
    """Sum only assistant targets; tool observations/images never receive loss."""
    import torch
    from tinker_cookbook.preference.train_dpo import compute_dpo_loss

    if not data or len(data) % 2 or len(logprobs) != len(data) or len(reference_logprobs) != len(data):
        raise ValueError("DPO batch must contain aligned chosen/rejected pairs")
    policy, reference = [], []
    for datum, lp, ref in zip(data, logprobs, reference_logprobs, strict=True):
        weights = torch.tensor(datum.loss_fn_inputs["weights"].data, device=lp.device)
        mask = weights > 0
        if len(lp) != len(weights) or len(ref) != len(weights) or not mask.any():
            raise ValueError("DPO logprobs and assistant masks must align")
        # Image/prompt positions may have None logprobs. They are context only.
        if any(w and (v is None or not math.isfinite(v)) for w, v in zip(weights.tolist(), ref)):
            raise ValueError("Missing or non-finite reference logprob for an assistant target")
        ref_tensor = torch.tensor([v if v is not None else 0.0 for v in ref], device=lp.device)
        if not torch.isfinite(lp[mask]).all():
            raise ValueError("Non-finite policy logprob for an assistant target")
        policy.append((lp[mask].float() * weights[mask]).sum())
        reference.append((ref_tensor[mask].float() * weights[mask]).sum())
    return compute_dpo_loss(policy[::2], policy[1::2], reference[::2], reference[1::2], beta)


def train(args, encoded_pairs, plan):
    import tinker

    config = plan["config"]
    start = 0
    checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = json.loads(args.resume_from_checkpoint.read_text())
        if checkpoint["config"] != config:
            raise ValueError("DPO checkpoint configuration changed")
        start = checkpoint["step"]
    if type(start) is not int or not 0 <= start < plan["total_steps"]:
        raise ValueError("No remaining DPO optimizer steps")
    service = tinker.ServiceClient()
    client = (service.create_training_client_from_state_with_optimizer(checkpoint["state_path"])
              if checkpoint else service.create_training_client_from_state(config["initial_state_path"]))
    # Always reload the original SFT sampler, including on resume. Taking a
    # snapshot of resumed DPO weights would silently change the objective.
    reference = service.create_sampling_client(
        base_model=TINKER_SAMPLING_MODEL_ID, model_path=config["reference_sampler_path"])
    with (args.output_dir / "metrics.jsonl").open("a") as metrics:
        for step, batch in enumerate(batches(encoded_pairs, args.batch_size, args.epochs, args.seed), 1):
            if step <= start:
                continue
            if step > plan["total_steps"]:
                break
            data = [datum for pair in batch for datum in pair]
            futures = [reference.compute_logprobs(d.model_input.append_int(
                int(d.loss_fn_inputs["target_tokens"].data[-1]))) for d in data]
            refs = [future.result()[1:] for future in futures]
            result = client.forward_backward_custom(
                data, lambda datums, lp: dpo_loss(datums, lp, refs, args.beta)).result()
            client.optim_step(tinker.AdamParams(learning_rate=args.learning_rate)).result()
            row = {**result.metrics, "step": step, "pairs": len(batch)}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == plan["total_steps"]:
                checkpoint = save_checkpoint(client, args.output_dir, step, config)
    final = {**checkpoint, "sampler_path": client.save_weights_for_sampler("final").result().path}
    (args.output_dir / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--data", type=Path, required=True, help="Image preference JSONL with record_a/record_b IDs")
    p.add_argument("--trajectories", type=Path, required=True, help="Approved recorded drawing dataset JSONL")
    p.add_argument("--init-checkpoint", type=Path, default=ROOT / "checkpoints/sft-v1/final.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/dpo-v1")
    p.add_argument("--dry-run", action="store_true", help="Validate labels/artifacts/checkpoint without API calls or writes")
    p.add_argument("--check-batch", action="store_true", help="Also encode all training pairs locally; no API calls")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4, help="Preference pairs per update")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=MODEL_CONTEXT_WINDOW)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--resume-from-checkpoint", type=Path)
    return p


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    p = parser()
    args = p.parse_args(argv)
    if (min(args.epochs, args.batch_size, args.max_length, args.save_every) < 1
            or args.max_length > MODEL_CONTEXT_WINDOW or not 0 <= args.seed <= 2**32 - 1
            or args.max_steps == 0 or args.max_steps < -1
            or any(not math.isfinite(v) or v <= 0 for v in (args.learning_rate, args.beta))):
        p.error("Invalid DPO training hyperparameters")
    try:
        pairs, manifest = load_preferences(args.data, args.trajectories)
        initial = load_initial_checkpoint(args.init_checkpoint)
        plan = make_plan(args, manifest, initial)
        if args.output_dir.resolve() == args.init_checkpoint.resolve().parent:
            raise ValueError("DPO output must differ from the SFT checkpoint directory")
        if args.resume_from_checkpoint:
            saved = json.loads(args.resume_from_checkpoint.read_text())
            if (saved.get("config") != plan["config"] or type(saved.get("step")) is not int
                    or not 0 <= saved["step"] < plan["total_steps"]
                    or not isinstance(saved.get("state_path"), str)
                    or not saved["state_path"].startswith("tinker://")):
                raise ValueError("Incompatible DPO resume checkpoint or no remaining steps")
        print(json.dumps(plan, indent=2))
        if args.dry_run and not args.check_batch:
            return 0
        encoder = create_encoder(MODEL_ID, args.max_length)
        encoded = [(encoder(pair["chosen"])[0], encoder(pair["rejected"])[0]) for pair in pairs]
        if args.check_batch:
            print(f"Encoded {len(encoded)} complete preference pairs; no truncation")
            return 0
        destination = args.output_dir / "training-manifest.json"
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            if (not args.resume_from_checkpoint or not destination.is_file()
                    or json.loads(destination.read_text()) != plan):
                raise ValueError("Output is not this resumable DPO run; choose a new directory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plan, indent=2) + "\n")
        train(args, encoded, plan)
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
