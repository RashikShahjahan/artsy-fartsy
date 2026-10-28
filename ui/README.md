# Artsy generation UI

From the repository root:

```sh
python3 -m ui.server
```

Open http://127.0.0.1:4173. Start Docker and the native bridge (`make serve`) before generating. The service uses `.eval-venv/bin/python`, falling back to `.venv/bin/python`; override with `ARTSY_EVAL_PYTHON`. Install the evaluation dependencies documented in the repository README. Change the port with `ARTSY_UI_PORT`.

The app has three workspaces with shared navigation and styling:

- **Studio** writes a freeform prompt or selects an ordinary drawing from a dataset, using one model selector, output panel, timeline, and artifact downloads.
- **Prompt review** searches and reviews drawing and safety prompts, saves notes and decisions, and records named dataset approval. Choose Safety prompts to retain the separate safety-only approval workflow.
- **SFT candidates** inspects full trajectories and images, saves manual decisions, and exports approved, structurally trainable examples.

Run all three with one command:

```sh
python3 -m ui.server \
  --dataset datasets/validation-v8-50-2026-10-28/prompts.jsonl \
  --candidates datasets/sft-v1/candidates.jsonl
```

Open http://127.0.0.1:4173. No separate inspector servers are needed. Without
`--dataset`, Studio and Prompt review use the frozen benchmark. Without
`--candidates`, the candidate workspace explains how to configure a file.
`ARTSY_DRAWING_DATASET` and `ARTSY_CANDIDATE_DATASET` can supply these paths.

Prompt decisions use the existing `prompts.review.json` beside a custom prompt
file, or `eval-results/dataset-review/approval.json` for the frozen benchmark.
Candidate decisions default to `manual-review.json` and approved exports to
`accepted.jsonl` beside the candidates. Override with `--review-output`,
`--candidate-review`, and `--accepted-output`. The original data is preserved.

The old `/drawings` URLs open the same Studio in dataset mode; saved `#run=...`
and `#prompt=...` links remain usable. The standalone inspector CLIs remain
available for compatibility and reuse the same handlers and validation logic.
Only clicking Generate or Run starts sampling; switching prompt source, model,
or workspace does not. Dataset runs preserve canonical prompt text, seed,
dataset hash, and prompt ID. Saved runs are scoped to the dataset revision.

Generate launches the existing `evaluation.draw` CLI. Model calls use credits. No generated code is executed by the web server: the drawing CLI retains its Docker execution boundary. Only one UI job runs at a time. The model dropdown discovers completed fine-tunes from `checkpoints/*/final.json` alongside Qwen3.5-4B. Only sampler checkpoints matching the base model and pinned tokenizer revision are listed. The selected fine-tune is routed to its saved Tinker sampler through a stable `artsy/<training-run>` model ID; samplers load on first use and reuse the bridge’s tokenizer and image processor. Restart the UI server and bridge once after updating to enable discovery and routing. If the running bridge does not advertise a selected fine-tune, the UI asks for a bridge restart and disables generation for that selection. Newly completed compatible training runs then appear on page reload without another bridge restart. Reopened runs retain their recorded model in the dropdown. Generation uses seed 42 automatically. The legacy `ARTSY_TINKER_MODEL_PATH` setting still configures the default model entry, labeled “configured checkpoint” when set.

Runs are stored in `episodes/ui/<id>/`. The UI shows process output while running. After the CLI finishes, it exports the full Inspect logs with resolved attachments to `output/trace.json`. The completed-run timeline pairs each command with its recorded tool results, shows failed renders and inline previews, and distinguishes submission requests from final render validation. Source rewrites show a line diff against the previous successful write, with full commands and source available to expand. Model response timing and token counts appear when recorded. The Transcript view preserves all messages; the Events view searches full event data and loads raw records when expanded. Failed actions open automatically. Downloads include the final PNG, Python source, result JSON, and full trace when present. Refusals and failures are displayed without substituting artwork. Reloading the same browser tab resumes polling its last run. Run URLs include `#run=<id>` so a saved run can be reopened directly.

The trace is the recorded execution history, not inferred or invented model reasoning. Full agent events currently appear after completion, not live. Runs interrupted by server shutdown are not resumed. Keep the server running until the job ends. This service binds only to loopback, checks Host and Origin, and serves only the UI and explicitly allowed run artifacts.

Validation: `node --check ui/app.js`, `node --test ui/test_trace.cjs`, `python3 -m unittest ui.test_server`, and browser checks. No paid model generation was run during implementation.
