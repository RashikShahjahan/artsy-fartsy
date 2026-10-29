# Artsy generation UI

From the repository root:

```sh
.venv/bin/python -m ui.server
```

Open http://127.0.0.1:4173. Start Docker and the native bridge (`make serve`) before generating. The service uses `.eval-venv/bin/python`, falling back to `.venv/bin/python`; override with `ARTSY_EVAL_PYTHON`. Install the evaluation dependencies documented in the repository README. Change the port with `ARTSY_UI_PORT`.

The app has four workspaces with shared navigation and styling:

- **Studio** writes a freeform prompt or selects an ordinary drawing from a dataset, using one model selector, output panel, timeline, and artifact downloads.
- **Prompt review** searches and reviews drawing and safety prompts, saves notes and decisions, and records named dataset approval. Choose Safety prompts to retain the separate safety-only approval workflow.
- **SFT candidates** inspects full trajectories and images, saves manual decisions, and exports approved, structurally trainable examples.
- **DPO dataset** collects multiple candidates per training prompt, compares approved drawings side by side, saves A/B/Tie preferences and family splits, and exports portable, validated policy-training data.

Run all four with one command:

```sh
.venv/bin/python -m ui.server \
  --dataset datasets/sft-v2/prompts.jsonl \
  --candidates datasets/sft-v2/candidates.jsonl \
  --dpo-output datasets/dpo-v1
```

Open http://127.0.0.1:4173. No separate inspector servers are needed. Without
`--dataset`, Studio and Prompt review use the frozen benchmark. Without
`--candidates`, the candidate workspace explains how to configure a file.
`ARTSY_DRAWING_DATASET` and `ARTSY_CANDIDATE_DATASET` can supply these paths.

For DPO collection, configure the bridge with an SFT sampler checkpoint. The
agreed first collection is **100 prompts × 2 candidates = 200 trajectories**, with
**50 reused SFT prompts and 50 fresh prompts**, and at most 100 preference pairs
before filtering and ties. Prepare `datasets/dpo-mixed-v1/prompts.jsonl` with
`.venv/bin/python -m ui.prepare_dpo_mix --output datasets/dpo-mixed-v1` and pass
that file to `--dataset`. Use **Select 100 prompts**
to apply this plan, then click Generate candidates. Manual prompt selection and
2–4 candidates per prompt remain available.
The UI shows the number of generations before starting. Each collection uses a
new directory under `episodes/dpo-ui/`, records its checkpoint, and runs the native
Inspect drawing agent inside Docker. Actual bash actions, reasoning, preview
observations and final render artifacts are preserved. Collection shares the single-job limit with Studio. Benchmark
prompts are excluded. No generation starts merely by opening the workspace.

Completed collection links open candidate approval and DPO comparison for that
collection. Save manual candidate approvals before labeling; both images must
be safe, render-valid and satisfy the same prompt and seed. Label visual quality
with A/B/Tie, or skip a pair. Keep entire prompt families in one split. Save
decisions, then export `labels.jsonl`, `accepted.jsonl` and copied artifacts under
`datasets/dpo-v1/exports/<collection-id>/`. The trainer validates the export.
Use `--dpo-candidates path/to/candidates.jsonl` for an existing collection; its
default export is the `--dpo-output` directory. Original candidate data is preserved.
See the [DPO collection plan and format](../docs/preference-rlhf.md).

Prompt decisions use the existing `prompts.review.json` beside a custom prompt
file, or `eval-results/dataset-review/approval.json` for historical benchmark v1. The validated default suite is the October 28 80-prompt set; its recorded validation does not require repeat approval for evaluation.
Candidate decisions default to `manual-review.json` and approved exports to
`accepted.jsonl` beside the candidates. Override with `--review-output`,
`--candidate-review`, and `--accepted-output`. The original data is preserved.

The old `/drawings` URLs open the same Studio in dataset mode; saved `#run=...`
and `#prompt=...` links remain usable. Standalone viewer launchers have been removed; review handlers live inside `ui`.
Only clicking Generate or Run starts sampling; switching prompt source, model,
or workspace does not. Dataset runs preserve canonical prompt text, seed,
dataset hash, and prompt ID. Saved runs are scoped to the dataset revision.

Studio Generate launches the existing `evaluation.draw` CLI. DPO Generate uses the
same native Inspect drawing agent and Docker execution boundary. Model calls use credits. No
generated code is executed by the web server. Only one UI job runs at a time. The Studio model dropdown contains Qwen3.5 4B and GPT-6 Luna.
Qwen uses the local native Tinker sampling bridge. GPT-6 Luna uses OpenRouter's
OpenAI-compatible endpoint with `OPENROUTER_API_KEY` from the environment or `.env`.
Both use the same Docker drawing harness. GPT-6 Luna does not require the local bridge.
Removed models are not restored to the dropdown by saved runs.


Runs are stored in `episodes/ui/<id>/`. The UI shows process output while running. After the CLI finishes, it exports the full Inspect logs with resolved attachments to `output/trace.json`. The completed-run timeline pairs each command with its recorded tool results, shows failed renders and inline previews, and distinguishes submission requests from final render validation. Source rewrites show a line diff against the previous successful write, with full commands and source available to expand. Model response timing and token counts appear when recorded. The Transcript view preserves all messages; the Events view searches full event data and loads raw records when expanded. Failed actions open automatically. Downloads include the final PNG, Python source, result JSON, and full trace when present. Refusals and failures are displayed without substituting artwork. Reloading the same browser tab resumes polling its last run. Run URLs include `#run=<id>` so a saved run can be reopened directly.

The trace is the recorded execution history, not inferred or invented model reasoning. Full agent events currently appear after completion, not live. Runs interrupted by server shutdown are not resumed. Keep the server running until the job ends. This service binds only to loopback, checks Host and Origin, and serves only the UI and explicitly allowed run artifacts.

Validation: `node --check ui/app.js`, `node --test ui/test_trace.cjs`,
`.venv/bin/python -m unittest ui.test_server ui.test_dpo_review ui.test_dpo_collect`,
and browser checks. No paid model generation was run during implementation.
