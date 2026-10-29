# Running Artsy SFT → DPO → RL

Visual quality is learned in a DPO stage between SFT and RL. RL then optimizes instruction adherence with safety and refusal outcomes. A trained aesthetic judge is optional for offline labeling or evaluation and is not required to launch either policy stage.

## Prepare visual preferences for DPO

Collect at least two approved, safe, render-valid drawing trajectories for each training request, preferably sampled from the SFT policy. Retain their recorded dataset JSONL and all source/preview/final-image artifacts in the existing dataset format used by `training.sft`. Judge the final images on composition, coherence, color, balance, craftsmanship and expressive appeal. Both candidates should satisfy the request so aesthetic labels do not encode adherence or safety differences.

Store A/B/Tie labels in JSONL. Image paths are relative to this labels file. `record_a` and `record_b` identify the corresponding entries in the recorded trajectory dataset:

```json
{"pair_id":"pair-0001","group_id":"cabin-family","image_a":"artifacts/cabin-a/image.png","image_b":"artifacts/cabin-b/image.png","record_a":"cabin-c0","record_b":"cabin-c1","winner":"B","split":"train"}
{"pair_id":"pair-0002","group_id":"skyline-family","image_a":"artifacts/skyline-a/image.png","image_b":"artifacts/skyline-b/image.png","record_a":"skyline-c0","record_b":"skyline-c1","winner":"A","split":"validation"}
```

`winner` is `A`, `B`, or `Tie`; `split` is `train`, `validation`, or `test`. Keep entire prompt families in one split and keep held-out evaluation data out of all training. The validator rejects duplicate or reversed image pairs, identical images, paths outside the labels directory, unknown record IDs, mismatched image hashes, unapproved/refusal trajectories, different prompt/seed contexts, and prompt/source/trajectory/image leakage across splits. Standard trajectory conversion checks source and artifact integrity. Historical automated judge verdicts do not override human selections. Accepted records represent dataset curation; audit them before labeling.

Ties are retained in the label file but omitted from DPO updates. Only non-tie `train` pairs are encoded for training. Validation/test pairs remain held out. There are no bundled DPO labels or trained DPO checkpoint; collect these inputs before starting RL with the new defaults.

## Train the drawing policy with DPO

Use the completed SFT checkpoint as both initialization and the frozen reference:

```bash
# Validate locally, without API calls, metadata downloads or output writes:
uv run python -m training.dpo \
  --data datasets/dpo-v1/labels.jsonl \
  --trajectories datasets/dpo-v1/accepted.jsonl \
  --init-checkpoint checkpoints/sft-v1/final.json \
  --dry-run

# Check full text/image encoding and context lengths without hosted calls:
make dpo ARGS="--data datasets/dpo-v1/labels.jsonl --trajectories datasets/dpo-v1/accepted.jsonl --check-batch"

# Configure TINKER_API_KEY in the environment or local .env, then train:
uv run python -m training.dpo \
  --data datasets/dpo-v1/labels.jsonl \
  --trajectories datasets/dpo-v1/accepted.jsonl \
  --init-checkpoint checkpoints/sft-v1/final.json \
  --output-dir checkpoints/dpo-v1 \
  --epochs 1 --batch-size 4 --learning-rate 1e-5 --beta 0.1
```

DPO inherits the SFT LoRA rank and uses the same pinned model metadata, Qwen renderer and multimodal encoder. Each pair contains two complete drawing-agent trajectories. The [DPO objective](https://tinker-docs.thinkingmachines.ai/cookbook/preferences/dpo-guide/) uses summed assistant-action log probabilities relative to the frozen SFT sampler. User/system text, tool feedback and preview pixels are context only; neither preference labels nor judge explanations enter the policy conversation. The 262,144-token default limit applies to each trajectory, with no truncation. `--max-steps 1` caps a live smoke run at one update.

Outputs include a provenance manifest, metrics, periodic state checkpoints and `final.json` containing state and sampler paths compatible with the model catalog and RL trainer. Resume an interrupted run with its original arguments plus `--resume-from-checkpoint checkpoints/dpo-v1/checkpoint-NNNNNN.json`. Configuration changes are rejected. Resume restores optimizer state and keeps the original SFT sampler as the reference. Existing output directories require a matching manifest and an explicit resume checkpoint.

## Configure RL judges

Set the following in the environment or local `.env`:

- `ARTSY_CORRECTNESS_JUDGE_MODEL`, `ARTSY_CORRECTNESS_JUDGE_URL`: instruction adherence rubric judge.
- `ARTSY_SAFETY_JUDGE_MODEL`, `ARTSY_SAFETY_JUDGE_URL`: independent image safety judge.
- `ARTSY_CODE_SAFETY_JUDGE_MODEL`, `ARTSY_CODE_SAFETY_JUDGE_URL`: independent full-code/action-history safety judge.
- Corresponding `*_API_KEY` variables when the provider requires authentication.

For native Tinker judges use `*_URL=tinker://native` and a supported VLM/renderer. No `ARTSY_VISUAL_JUDGE_*` settings or visual checkpoint are used by RL.

## Train the policy with RL

Initialize RL from the completed **DPO checkpoint**. The bundled prompt set contains only drawing tasks; supply an audited mixed prompt JSONL with trusted `kind="refusal"` examples to train refusal behavior:

```bash
uv run python -m training.grpo \
  --dataset datasets/grpo-v1/prompts.jsonl \
  --init-checkpoint checkpoints/dpo-v1/final.json \
  --output-dir checkpoints/grpo-v2 \
  --group-size 4 --max-samples 4 --epochs 1 --max-steps 1 \
  --dry-run
```

Remove `--dry-run` to launch. RL uses trusted Docker rendering and native Cookbook multi-turn rollouts. A safe, valid, submitted drawing earns normalized adherence alone. Unsafe code/actions or images earn −1, correct refusals earn +1, and invalid drawings or over-refusals earn 0. Cookbook mean-centers group rewards; `--normalize-rewards` is a compatibility flag with no effect. KL regularization references the initial DPO sampler.

Start a fresh output directory for this reward version. Legacy visual-reward runs cannot be continued under the new objective. Interrupted new runs can resume with their original arguments; completed compatible runs can be extended as described in [continuation and checkpoint selection](grpo-continuation.md).

## Optional visual comparison judge

The existing `preference.train` tool fine-tunes an independent Qwen3.5-9B aesthetic classifier using the image-label fields above. It predicts A/B/Tie with supervised label loss and reversed image order; it does not perform policy DPO. `record_a`/`record_b` are only required by the policy trainer.

```bash
uv run python -m preference.train --data datasets/dpo-v1/labels.jsonl --validate-only
uv run python -m preference.train --data datasets/dpo-v1/labels.jsonl \
  --model Qwen/Qwen3.5-9B --epochs 1 --batch-size 4 --rank 16
uv run python -m preference.evaluate --data datasets/dpo-v1/labels.jsonl \
  --split validation --sampler-path tinker://YOUR_VISUAL_JUDGE_SAMPLER
```

Inspect remains the independent evaluation harness. Measure human visual preference against SFT and DPO alongside adherence, safety, over-refusal and render success. Hosted DPO training and full Docker/RL smoke tests require credentials and a working container runtime.
