"""On-policy, correctness-only GRPO for the hosted ArtCanvas drawing policy.

python -m training.grpo --dry-run
python -m training.grpo --max-steps 1 --output-dir checkpoints/grpo-pilot

Requires Docker, Tinker authentication, and ARTSY_CORRECTNESS_JUDGE_{MODEL,URL}.
Rollouts use the host-side evaluation agent and Docker tools; no inference server is required.
"""

import argparse
import asyncio
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
from time import monotonic
from uuid import uuid4

from drawing.config import IMAGE_SIZE, MODEL_ID, MODEL_REVISION
from training.data import ROOT, digest
from training.sft import RENDERER, batches, create_encoder, save_checkpoint

ALGORITHM = "correctness-grpo-v1"


def advantages(rewards, normalize=False):
    if len(rewards) < 2 or any(type(r) not in (int, float) or not math.isfinite(r)
                              or not 0 <= r <= 1 for r in rewards):
        raise ValueError("A group needs at least two finite correctness rewards in [0, 1]")
    mean = statistics.mean(rewards)
    scale = statistics.pstdev(rewards) + 1e-8 if normalize else 1.0
    return [(reward - mean) / scale for reward in rewards]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_run(args):
    from evaluation.harness import load_specs

    specs = load_specs(args.dataset)
    if any(spec.get("kind") == "refusal" for spec in specs):
        raise ValueError("Correctness-only GRPO requires benign drawing prompts")
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
        raise ValueError("Initial checkpoint must contain compatible SFT state and sampler paths")
    files = ["docs/artcanvas.md", "drawing/config.py", "drawing/judges.py",
             "runtime/artcanvas.py", "runtime/cairo.py", "runtime/requirements.txt",
             "evaluation/harness.py", "evaluation/host_agent.py", "evaluation/render.py", "evaluation/mini.yaml",
             "evaluation/Dockerfile", "evaluation/compose.yaml", "evaluation/requirements.txt",
             "evaluation/sample_grpo.py", "inference/tinker_backend.py", "inference/server.py",
             "training/grpo.py", "training/sft.py", "training/requirements.txt",
             "training/requirements-grpo.txt"]
    config = {"algorithm": ALGORITHM, "model": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
              "renderer": RENDERER, "image_size": IMAGE_SIZE, "lora_rank": source["lora_rank"],
              "dataset_sha256": digest(args.dataset.read_bytes()), "prompt_ids": [s["id"] for s in specs],
              "initial_state_path": initial["state_path"], "reference_sampler_path": initial["sampler_path"],
              "api_doc_sha256": digest((ROOT / "docs/artcanvas.md").read_bytes()),
              "source_sha256": {name: digest((ROOT / name).read_bytes()) for name in files},
              "batch_size": args.batch_size, "group_size": args.group_size,
              "learning_rate": args.learning_rate, "clip_epsilon": args.clip_epsilon,
              "kl_coef": args.kl_coef, "normalize_rewards": args.normalize_rewards,
              "temperature": args.temperature, "max_tokens": args.max_tokens,
              "max_length": args.max_length, "max_samples": args.max_samples, "seed": args.seed,
              "correctness_judge_model": os.getenv("ARTSY_CORRECTNESS_JUDGE_MODEL"),
              "correctness_judge_url": os.getenv("ARTSY_CORRECTNESS_JUDGE_URL"),
              "reward": "final-image correctness only", "loss_reduction": "mean of trajectory token means"}
    total = math.ceil(len(specs) / args.batch_size) * args.epochs
    if args.max_steps != -1:
        total = min(total, args.max_steps)
    return specs, config, total


class ReferenceLogprobs:
    """Fetch frozen-reference action scores while rollout episodes continue."""

    def __init__(self, reference, concurrency):
        self.reference = reference
        self.semaphore = asyncio.Semaphore(concurrency)
        self.tasks = []

    def submit(self, prompt, record, path):
        self.tasks.append(asyncio.create_task(self._annotate(prompt, record, path)))

    async def _annotate(self, prompt, record, path):
        import tinker
        full = tinker.ModelInput(chunks=[*prompt.chunks, tinker.EncodedTextChunk(tokens=record["tokens"])])
        async with self.semaphore:
            values = await self.reference.compute_logprobs_async(full)
        action = values[prompt.length:]
        if (len(values) != full.length or len(action) != len(record["tokens"])
                or any(value is None or not math.isfinite(value) for value in action)):
            raise ValueError("Reference returned invalid action log probabilities")
        record["reference_logprobs"] = action
        write_json(path, record)

    @property
    def completed(self):
        return sum(task.done() and not task.cancelled() and task.exception() is None for task in self.tasks)

    async def finish(self):
        await asyncio.gather(*self.tasks)

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class NativeRollouts:
    """Sample host-agent requests and retain the exact native sampling evidence."""

    def __init__(self, backend, args, step, directory, reference_prefetch=None):
        self.backend, self.args, self.step, self.directory = backend, args, step, directory
        self.turns = defaultdict(list)
        self.reference_prefetch = reference_prefetch

    async def __call__(self, model, messages, tools, tool_choice, config):
        import tinker
        from inspect_ai.model import ChatMessageAssistant, ChatCompletionChoice, ModelOutput, ModelUsage
        # These adapters are version-bound to evaluation/requirements.txt. They
        # preserve Inspect's image observations and per-sample epoch context.
        from inspect_ai.model._openai import messages_to_openai, openai_chat_tools
        from inspect_ai.solver._task_state import sample_state
        from inspect_ai.tool import ToolCall
        from inference.server import parse_tools, template_messages

        state = sample_state()
        if state is None:
            raise RuntimeError("Native sampling requires an active Inspect sample")
        key = (str(state.sample_id), state.epoch)
        turn = len(self.turns[key])
        tool_specs = openai_chat_tools(tools)
        wire = await messages_to_openai(messages)
        prompt = self.backend.render(template_messages(wire, self.backend.prepare_image), tool_specs)
        budget = min(self.args.max_tokens, self.args.max_length - prompt.length)
        if budget < 1:
            # Abort collection instead of dropping a member from its reward group.
            raise ValueError("Rollout context exhausted; no room for a sampled action")
        sampling_seed = int(digest(f"{self.args.seed}:{self.step}:{key}:{turn}".encode())[:8], 16)
        response = await self.backend.client.sample_async(
            prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(
                max_tokens=budget, temperature=self.args.temperature, seed=sampling_seed,
                stop=self.backend.renderer.get_stop_sequences()))
        sequence = response.sequences[0]
        tokens, logprobs = sequence.tokens, sequence.logprobs
        if (not tokens or logprobs is None or len(tokens) != len(logprobs)
                or any(not math.isfinite(value) for value in logprobs)):
            raise ValueError("Sampling returned missing or invalid action log probabilities")
        record = {"prompt": prompt.model_dump(mode="json"), "tokens": tokens,
                  "logprobs": logprobs, "stop_reason": sequence.stop_reason, "seed": sampling_seed}
        self.turns[key].append(record)
        directory = self.directory / "native" / f"{key[0]}-{key[1]}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"turn-{turn:03d}.json"
        write_json(path, record)
        if self.reference_prefetch is not None:
            self.reference_prefetch.submit(prompt, record, path)
        raw = self.backend.tokenizer.decode(tokens, skip_special_tokens=False)
        try:
            text, calls = parse_tools(raw, tool_specs)
        except (ValueError, SyntaxError):
            text, calls = raw, []
        output_calls = [ToolCall(id=call["id"], function=call["function"]["name"],
                                 arguments=json.loads(call["function"]["arguments"])) for call in calls]
        reason = "max_tokens" if sequence.stop_reason == "length" else "tool_calls" if calls else "stop"
        return ModelOutput(model=MODEL_ID, choices=[ChatCompletionChoice(
            message=ChatMessageAssistant(content=text, tool_calls=output_calls or None), stop_reason=reason)],
            usage=ModelUsage(input_tokens=prompt.length, output_tokens=len(tokens),
                             total_tokens=prompt.length + len(tokens)))


async def collect_rollouts(args, specs, sampler, encoder, step, directory, judge, reference=None):
    from inspect_ai import eval_async
    from inspect_ai.model import GenerateConfig, get_model
    from evaluation.harness import drawing_task
    from evaluation.sample_grpo import score_outputs, summarize_rewards
    from inference.tinker_backend import TinkerBackend

    # Reuse the serving renderer and image preparation without loading a second
    # tokenizer or creating a sampler for the base policy.
    backend = object.__new__(TinkerBackend)
    backend.client, backend.renderer = sampler, encoder.renderer
    backend.tokenizer, backend.image_processor = encoder.renderer.tokenizer, encoder.image_processor
    backend.image_size = IMAGE_SIZE
    prefetch = ReferenceLogprobs(reference, args.max_samples) if reference is not None else None
    recorder = NativeRollouts(backend, args, step, directory, prefetch)
    task = drawing_task(specs, str(directory / "artifacts"), temperature=args.temperature,
                        max_connections=args.max_samples, generation_filter=recorder)
    # Every request is handled by the filter. This local placeholder is never
    # called and does not require a running inference bridge.
    placeholder = get_model("openai-api/artsy/grpo", base_url="http://127.0.0.1:8080/v1",
                            api_key="local-no-auth", responses_api=False, stream=False,
                            config=GenerateConfig(max_retries=0))
    try:
        began = monotonic()
        logs = await eval_async(task, model=placeholder, epochs=args.group_size,
                                max_samples=args.max_samples, log_dir=str(directory / "logs"))
        rollout_seconds = monotonic() - began
        completed_during_rollouts = prefetch.completed if prefetch is not None else 0
        if (not logs or any(log.status != "success" for log in logs)
                or any(sample.error is not None for log in logs for sample in (log.samples or []))):
            raise RuntimeError("Rollout collection failed; no policy update was made")
        for spec in specs:
            for epoch in range(1, args.group_size + 1):
                if (not recorder.turns[(spec["id"], epoch)] or not
                        (directory / "artifacts" / f"{spec['id']}-{epoch}" / "result.json").is_file()):
                    raise RuntimeError("Incomplete rollout group; no policy update was made")
        began = monotonic()
        groups = await score_outputs(specs, directory, args.group_size, judge, args.max_samples)
        judge_seconds = monotonic() - began
        began = monotonic()
        if prefetch is not None:
            await prefetch.finish()
        reference_wait_seconds = monotonic() - began
    finally:
        if prefetch is not None:
            await prefetch.close()
    for group in groups:
        for attempt in group["attempts"]:
            attempt["turns"] = recorder.turns[(group["id"], attempt["epoch"])]
            # A clipped action is still sampled evidence, but cannot earn a
            # positive terminal reward just because an earlier file renders.
            attempt["truncated"] = any(t["stop_reason"] == "length" for t in attempt["turns"])
            if attempt["truncated"]:
                attempt["reward"] = 0.0
        group.update(summarize_rewards([attempt["reward"] for attempt in group["attempts"]]))
    write_json(directory / "groups.json", groups)
    write_json(directory / "phase-metrics.json", {
        "rollout_seconds": rollout_seconds, "judge_seconds": judge_seconds,
        "reference_wait_seconds": reference_wait_seconds,
        "reference_requests": len(prefetch.tasks) if prefetch is not None else 0,
        "reference_completed_during_rollouts": completed_during_rollouts})
    return groups


def build_training_data(groups, normalize=False):
    """One datum per sampled action; previous actions and pixels are context only."""
    import tinker
    from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets

    data = []
    trajectories = sum(len(group["attempts"]) for group in groups)
    for group in groups:
        centered = advantages([attempt["reward"] for attempt in group["attempts"]], normalize)
        for attempt, advantage in zip(group["attempts"], centered, strict=True):
            count = sum(len(turn["tokens"]) for turn in attempt["turns"])
            if not count:
                raise ValueError("Rollout has no sampled actions")
            for turn in attempt["turns"]:
                prompt = tinker.ModelInput.model_validate(turn["prompt"])
                tokens, logprobs = turn["tokens"], turn["logprobs"]
                if (prompt.length < 1 or not tokens or len(tokens) != len(logprobs)
                        or any(not math.isfinite(value) for value in logprobs)):
                    raise ValueError("Invalid recorded action or log probabilities")
                full = tinker.ModelInput(chunks=[*prompt.chunks, tinker.EncodedTextChunk(tokens=tokens)])
                inputs, targets = create_rightshifted_model_input_and_leftshifted_targets(list(full.chunks))
                prefix = prompt.length - 1
                fields = {"target_tokens": (targets, "int64"),
                          "logprobs": ([0.0] * prefix + logprobs, "float32"),
                          "advantages": ([0.0] * prefix + [advantage] * len(tokens), "float32"),
                          "weights": ([0.0] * prefix + [1 / (count * trajectories)] * len(tokens), "float32")}
                if "reference_logprobs" in turn:
                    reference = turn["reference_logprobs"]
                    if (len(reference) != len(tokens)
                            or any(value is None or not math.isfinite(value) for value in reference)):
                        raise ValueError("Invalid recorded reference log probabilities")
                    fields["reference_logprobs"] = ([0.0] * prefix + reference, "float32")
                data.append(tinker.Datum(model_input=inputs, loss_fn_inputs={
                    name: tinker.TensorData(data=values, dtype=dtype, shape=[len(targets)])
                    for name, (values, dtype) in fields.items()}))
    return data


def grpo_loss(clip_epsilon, kl_coef):
    """Clipped policy objective plus k3 KL, averaged equally over trajectories."""
    def loss(data, logprobs):
        import torch
        total, policy_metric, kl_metric, clipped_metric = None, 0.0, 0.0, 0.0
        for datum, current in zip(data, logprobs, strict=True):
            fields = datum.loss_fn_inputs
            weights = fields["weights"].to_torch().to(current.device)
            active = weights > 0
            current = current[active]
            weights = weights[active]
            old = fields["logprobs"].to_torch().to(current.device)[active]
            advantage = fields["advantages"].to_torch().to(current.device)[active]
            ratio = (current - old).exp()
            objective = torch.minimum(ratio * advantage,
                                      ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
            policy = -(weights * objective).sum()
            kl = current.sum() * 0.0
            if kl_coef:
                ref = fields["reference_logprobs"].to_torch().to(current.device)[active]
                delta = ref - current
                kl = (weights * (delta.exp() - delta - 1)).sum()
            value = policy + kl_coef * kl
            if not torch.isfinite(value):
                raise ValueError("Non-finite GRPO loss; no optimizer step was made")
            total = value if total is None else total + value
            policy_metric += float(policy.detach())
            kl_metric += float(kl.detach())
            clipped_metric += float((weights * ((ratio - 1).abs() > clip_epsilon)).sum().detach())
        if total is None:
            raise ValueError("No GRPO training data")
        return total, {"policy_loss": policy_metric, "reference_kl": kl_metric,
                       "clip_fraction": clipped_metric}
    return loss


def train(args, specs, config, total_steps):
    import tinker
    from importlib.metadata import version
    from drawing.dataset_client import chat_client

    judge = chat_client({"model": config["correctness_judge_model"],
                         "baseUrl": config["correctness_judge_url"],
                         "apiKey": os.getenv("ARTSY_CORRECTNESS_JUDGE_API_KEY"), "timeoutMs": 120000})
    start = 0
    checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = json.loads(args.resume_from_checkpoint.read_text())
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint dataset/model/training configuration does not match this run")
        start = checkpoint["step"]
    if not 0 <= start < total_steps:
        raise ValueError("No remaining steps; increase --epochs or --max-steps")
    encoder = create_encoder(MODEL_ID, args.max_length)
    service = tinker.ServiceClient()
    client = (service.create_training_client_from_state_with_optimizer(checkpoint["state_path"])
              if checkpoint else service.create_training_client_from_state(config["initial_state_path"]))
    reference = (service.create_sampling_client(model_path=config["reference_sampler_path"])
                 if args.kl_coef else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "training-manifest.json", {
        "config": config, "total_steps": total_steps,
        "packages": {name: version(name) for name in ("tinker", "tinker-cookbook", "inspect-ai", "inspect-swe")}})
    metrics_path = args.output_dir / "metrics.jsonl"
    # Rows after the chosen checkpoint are uncommitted when resuming an
    # interrupted run. Preserve rollout artifacts, but discard stale metrics.
    if metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= start))
    with metrics_path.open("a") as metrics:
        for step, batch in enumerate(batches(specs, args.batch_size, args.epochs, args.seed), 1):
            if step <= start:
                continue
            if step > total_steps:
                break
            began = monotonic()
            sampler = client.save_weights_and_get_sampling_client()
            directory = args.output_dir / "rollouts" / f"step-{step:06d}" / uuid4().hex
            directory.mkdir(parents=True)
            groups = asyncio.run(collect_rollouts(args, batch, sampler, encoder, step, directory, judge, reference))
            data = build_training_data(groups, args.normalize_rewards)
            signal = sum(len(set(a["reward"] for a in g["attempts"])) > 1 for g in groups)
            result_metrics = {}
            # No relative signal means no update, including no KL-only drift.
            if signal:
                result = client.forward_backward_custom(data, grpo_loss(args.clip_epsilon, args.kl_coef)).result()
                client.optim_step(tinker.AdamParams(learning_rate=args.learning_rate)).result()
                result_metrics = result.metrics
            rewards = [a["reward"] for g in groups for a in g["attempts"]]
            phase_path = directory / "phase-metrics.json"
            phase_metrics = json.loads(phase_path.read_text()) if phase_path.exists() else {}
            row = {**result_metrics, **phase_metrics, "step": step, "optimizer_updated": bool(signal),
                   "prompts": len(batch), "episodes": len(rewards), "reward_mean": statistics.mean(rewards),
                   "valid_renders": sum(a.get("render_valid", False) for g in groups for a in g["attempts"]),
                   "groups_with_learning_signal": signal,
                   "reward_std_mean": statistics.mean(statistics.pstdev([a["reward"] for a in g["attempts"]])
                                                      for g in groups),
                   "truncated": sum(a.get("truncated", False) for g in groups for a in g["attempts"]),
                   "sampled_tokens": sum(len(t["tokens"]) for g in groups for a in g["attempts"] for t in a["turns"]),
                   "learning_rate": args.learning_rate, "seconds": monotonic() - began,
                   "rollouts": str(directory.resolve())}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == total_steps:
                checkpoint = save_checkpoint(client, args.output_dir, step, config)
    final = {**checkpoint, "sampler_path": client.save_weights_for_sampler("final").result().path}
    write_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets/grpo-v1/prompts.jsonl")
    p.add_argument("--init-checkpoint", type=Path, default=ROOT / "checkpoints/sft-v1/final.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/grpo-v1")
    p.add_argument("--resume-from-checkpoint", type=Path)
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and print plan without API calls or Docker")
    p.add_argument("--limit", type=int, help="Use the first N dataset prompts for a smoke run")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1, help="Total prompt batches, including groups with no signal")
    p.add_argument("--batch-size", type=int, default=2, help="Prompt groups per update")
    p.add_argument("--group-size", type=int, default=4, help="Independent trajectories per prompt")
    p.add_argument("--max-samples", type=int, default=4, help="Concurrent sandbox episodes, correctness judgments, and reference requests")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.01, help="KL coefficient against the frozen initial SFT policy")
    p.add_argument("--normalize-rewards", action="store_true", help="Divide centered rewards by group population std")
    p.add_argument("--max-tokens", type=int, default=4096, help="Maximum sampled tokens per action")
    p.add_argument("--max-length", type=int, default=32768, help="Maximum native context including each sampled action")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1)
    return p


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    p = parser()
    args = p.parse_args(argv)
    if (any(value < 1 for value in (args.epochs, args.batch_size, args.max_samples, args.save_every,
                                   args.max_tokens, args.max_length)) or args.group_size < 2
            or args.max_steps == 0 or args.max_steps < -1 or (args.limit is not None and args.limit < 1)
            or not 0 <= args.seed <= 2**32 - 1 or args.max_length > 32768
            or args.max_tokens > args.max_length
            or any(not math.isfinite(value) for value in (args.learning_rate, args.temperature,
                                                         args.clip_epsilon, args.kl_coef))
            or args.learning_rate <= 0 or not 0 < args.temperature <= 2
            or not 0 < args.clip_epsilon < 1 or args.kl_coef < 0):
        p.error("Invalid counts, token limits, seed, learning rate, temperature, clipping or KL coefficient")
    try:
        specs, config, total = load_run(args)
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    plan = {"config": config, "prompts": len(specs), "total_steps": total,
            "episodes": sum(len(batch) * args.group_size for i, batch in enumerate(
                batches(specs, args.batch_size, args.epochs, args.seed)) if i < total)}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    if not config["correctness_judge_model"] or not config["correctness_judge_url"]:
        p.error("Set ARTSY_CORRECTNESS_JUDGE_MODEL and ARTSY_CORRECTNESS_JUDGE_URL")
    if not args.resume_from_checkpoint and args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Output directory is not empty; choose a fresh directory or --resume-from-checkpoint")
    if args.output_dir.resolve() == args.init_checkpoint.resolve().parent:
        p.error("GRPO output must differ from the initial SFT checkpoint directory")
    train(args, specs, config, total)


if __name__ == "__main__":
    main()
