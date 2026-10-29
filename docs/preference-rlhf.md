# Running Artsy SFT → DPO → RL

Visual quality is learned in a DPO stage between SFT and RL. RL then optimizes instruction adherence with safety and refusal outcomes. A trained aesthetic judge is optional for offline labeling or evaluation and is not required to launch either policy stage.

## Prepare visual preferences for DPO

### Collection size for the first experiment

There is no required DPO-to-SFT dataset ratio. Count SFT data in trajectories and
DPO data in non-tie preference pairs; a pair contains two trajectories and may
share a candidate with another comparison. Unique prompts, coverage and label
quality matter alongside the pair count.

For this project's first experiment, collect **100 drawing prompts with two
candidates each: 200 generated trajectories and at most 100 preference pairs**.
Use a **50/50 mix: 50 reused SFT prompts and 50 freshly authored prompts**.
Fresh prompts follow the same drawing skills and difficulty distribution and are
checked for exact, normalized and close lexical overlap with SFT and benchmark
prompts. Reused prompts preserve their original wording and rendering seeds.
This is the agreed collection budget, not an established sample-efficiency
result. The current SFT v2 collection has
224 structurally convertible records, including 179 drawings and 45 refusals;
human approval is still required. Use the drawing count as the relevant baseline
for visual DPO.

Keep entire prompt families in one split before labeling; aim for approximately
80 training prompts and 20 held-out prompts, adjusting for uneven family sizes.
This gives at most about 80 training pairs and 20 held-out comparisons. Failed
eligibility checks and ties reduce those counts; do not force a preference or
expand the agreed collection automatically. Reusing SFT training prompts is
allowed; keep evaluation prompts held out.

Train a small pilot, compare it blindly against SFT on held-out prompts, and
expand the collection only if visual preference improves without regressions in
adherence, safety or render success. The preference-pair formulation follows the
[DPO paper](https://arxiv.org/abs/2305.18290); the counts above are our project plan.

### Create the collection in the UI

Start the unified UI with training prompts and an output directory:

```bash
.venv/bin/python -m ui.server --dataset datasets/sft-v2/prompts.jsonl \
  --candidates datasets/sft-v2/candidates.jsonl --dpo-output datasets/dpo-v1
```

Open **DPO dataset**. Configure the native bridge to serve your SFT sampler
checkpoint, or select a verified saved SFT alias from its catalog, before generating
candidates. Prepare the mixed prompt file with
`python -m ui.prepare_dpo_mix --output datasets/dpo-mixed-v1`, then start the UI
with `--dataset datasets/dpo-mixed-v1/prompts.jsonl`. The preparation step uses
the configured prompt-writing model and saves its provenance. Use **Select 100 prompts** to apply the
100-prompt, two-candidate plan, then click **Generate candidates**. Custom prompt
selections and 2–4 candidates per prompt remain available. Generation uses model
credits. The collection worker records the checkpoint and runs the native
Inspect bash-tool drawing agent in Docker. It retains actual tool actions,
reasoning, preview observations and final-render results; it does not ask the
SFT policy to produce teacher-format JSON. Frozen benchmark prompts are excluded.

When collection finishes, use **Approve candidates** to inspect trajectories in
the existing candidate reviewer and save approvals. Then use **Compare candidates**
to view same-prompt, same-seed images side by side. Confirm both are safe,
render-valid and prompt-adherent before choosing A, B or Tie. Set the prompt-family
group and train/validation/test split, save decisions, and export the dataset.
Skip pairs that cannot support an aesthetic comparison.

Exports contain `labels.jsonl`, `accepted.jsonl` and copied artifacts. Each
collection has its own export directory under the configured DPO output, and the
UI displays the resulting path. The exporter validates the files with the policy
trainer's loader before publication. It does not train a model or change the
source candidate file. Existing candidate JSONL files can also be opened using
`--dpo-candidates`; their referenced artifacts must be present beside the file.

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
