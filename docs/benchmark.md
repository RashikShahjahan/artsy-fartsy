# Experiment evaluation

The active held-out dataset is `datasets/parti-prompts/evaluation/prompts.jsonl`
with **163 prompts**. The former 228-prompt preference reserve is now in SFT
training. Its historical Experiment 2 outputs remain archived, but those 228
prompts are not held out for Experiment 3. Current allocation and experiment
provenance are recorded in [the experiment log](../EXPERIMENT_LOG.md).

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

## Deferred HPSv3 scoring

Run Inspect generation with `--execution-only`. The model writes, previews and
submits drawings as before, and the trusted scorer validates and saves final
PNGs. Generation needs neither HPSv3 weights nor a scoring server.

After launching generation for each rank, start the comparison pipeline:

```sh
.venv/bin/python -m evaluation.compare_pipeline \
  --judge hpsv3 \
  --run rank32=eval-results/experiment3-rank32-generation \
  --run rank64=eval-results/experiment3-rank64-generation \
  --run rank128=eval-results/experiment3-rank128-generation \
  --hpsv3-python .inference-venv/bin/python \
  --hpsv3-cache .cache/hpsv3 \
  --output eval-results/experiment3-hpsv3
```

The pipeline waits for every generation summary to report success, then starts
one short-lived MLX HPSv3 batch process. It validates all image hashes before
loading weights, scores final images sequentially, and exits to release GPU and
model memory. A dedicated environment must contain the dependencies listed in
[HPSv3 setup](hpsv3.md); `--hpsv3-python-path` can append an existing dependency
directory when needed.

For already completed generation runs, invoke the batch directly:

```sh
.inference-venv/bin/python -m evaluation.score_hpsv3 \
  --run rank32=eval-results/experiment3-rank32-generation \
  --run rank64=eval-results/experiment3-rank64-generation \
  --cache .cache/hpsv3 --output eval-results/experiment3-hpsv3
```

Scoring checkpoints are saved under `records/`, separately from immutable
Inspect generation logs and artifacts. Matching saved scores are reused; a
changed prompt, image, scorer or reward calibration is rejected. Existing inline scores with the
same pinned model can be reused when migrating a run. Scoring failure leaves
generated images intact, so scoring can resume without regenerating drawings.
Use `evaluation.resume_eval --output RUN --execution-only` to migrate an
interrupted inline run while preserving its completed outputs.

`comparison.json` and `scores.csv` retain raw HPSv3 rewards for valid images.
A bounded linear transform `0.02 + 0.98 * clip((raw + 11) / 14, 0, 1)`
produces normalized preference; invalid or
missing submissions receive zero preference and no raw reward. The report
separates image delivery, raw image-only means, normalized all-episode means,
and pairwise wins on matched delivered images. Historical sigmoid scores are
not comparable to the new normalized scores: rescore into a new output directory
using the saved images or raw scores. These are not calibrated human
preference probabilities or separate adherence and safety judgments.

The optional inline `evaluation.evaluate --hpsv3-url URL --hpsv3-model ID` mode
remains available for callers that already operate a persistent scorer. The
Experiment 3 pipeline now uses deferred scoring instead.

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
