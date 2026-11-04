# Experiment evaluation

The current experiment dataset is `datasets/parti-prompts/evaluation/prompts.jsonl`
with 391 held-out prompts. The original 163 and added 228 are also saved as
`original.prompts.jsonl` and `added-from-preference.prompts.jsonl` alongside it.
Keep all 391 prompts out of SFT, RL, and preference training. The setup, hashes,
results, and artifact locations for the rank comparison are in
[Experiment 1](../EXPERIMENT_LOG.md).

## Prepare prompts

From the repository root, with the pinned upstream TSV available locally:

```sh
python3 scripts/prepare_parti_prompts.py --preference-count 228 --preference-to-evaluation
```

This deterministically assigns 914 SFT prompts, 327 RL prompts, and 391 evaluation
prompts. The preference reserve is empty. Dataset preparation makes no model
calls; datasets and generated results are Git-ignored.

## Generate checkpoint outputs

Start Docker and the native bridge (`make serve`). Leave
`ARTSY_TINKER_MODEL_PATH` empty to keep the base alias pointing to the base model.
Choose a new output directory for each checkpoint:

```sh
make eval ARGS="--dataset datasets/parti-prompts/evaluation/prompts.jsonl --model Qwen/Qwen3.5-4B --execution-only --output eval-results/parti-base-new"
make eval ARGS="--dataset datasets/parti-prompts/evaluation/prompts.jsonl --model artsy/sft-parti-qwen3.5-4b-rank64-2026-11-03 --execution-only --output eval-results/parti-rank64-new"
```

Use an available checkpoint alias from the bridge. `--execution-only` collects
outputs without the independent adherence, image-safety, and code-safety judges.
Generation uses paid inference. Defaults are temperature 0, 25 model turns,
32,768 tokens per response, and ten concurrent samples/connections/sandboxes,
with no aggregate token limit or episode deadline. Keep the same settings and
prompt seeds across models. See the [README](../README.md#evaluate-models) for
budget controls and full safety scoring.

The agent uses `update_drawing`, `read_drawing`, `preview_drawing`,
`submit_drawing`, and `refuse_drawing`. Submitted source is rendered afresh and
validated as a 1920×1200 PNG. A preview or textual completion claim cannot pass.
Inspect `summary.json` and resolve sample errors before comparing models.

## Judge and review

Configure the judge in `.env.example`, then compare final outputs:

```sh
.venv/bin/python -m evaluation.judge_pairs \
  --run base=eval-results/parti-base-new \
  --run rank64=eval-results/parti-rank64-new \
  --output eval-results/parti-comparison-new
```

Repeat `--run ROLE=DIRECTORY` for 2–8 models. One request per prompt includes all
available final images with shuffled anonymous labels. The judge assigns separate
0–4 adherence and visual-quality scores, comments, and a preferred output.
Missing images remain unscored. Successful saved judgments are reused only after
input hashes match. Judging uses paid API calls.

Open `/comparisons` in the UI (`make ui`) to review images, comments, failures,
and matched score averages. The combined 391-prompt collection retains the
original and added subsets; its definition and verification rules are documented
in the [UI guide](../ui/README.md). Human calibration and independent training
seeds remain follow-up work recorded in the experiment log.

## Regression suite

Without `--dataset`, evaluation and the drawing/prompt-review UI still use the
frozen 50-prompt suite at `evaluation/generated-50-2026-10-28/prompts.jsonl`,
validated by `evaluation/benchmark-v3/manifest.json`. It includes 35 drawings and
five each of benign boundary, harmful-image refusal, and malicious-code refusal
prompts. Retain it for safety and refusal regression coverage; it is separate from
the PartiPrompts experiment results.

`evaluation.benchmark` validates the frozen suite and produces blind human-review
packets and reports. `evaluation.run_benchmark` compares available checkpoints on
that suite with shared budgets and independent safety judges. Both support local
planning; use `--help` for their explicit suite, checkpoint, and review options.
