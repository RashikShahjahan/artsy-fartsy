# Artsy generation UI

From the repository root:

```sh
.venv/bin/python -m ui.server
```

Open http://127.0.0.1:4173 for the voice-first Artsy experience. It introduces
Artsy, asks what to draw, shows live drawing progress, then presents the finished
image with voice or text follow-ups. The home screen offers voice and keyboard input. Open **Drawing settings**
to choose a model or require approval for each action. Saved drawings live in an
on-demand history drawer. Image download and new-drawing controls appear with
the artwork; a compact version selector appears after edits. Selecting a version
makes it the basis of the next edit; **New drawing** starts a separate idea. Active jobs
resume polling after a reload, and completed drawings remain in the sketchbook.

The model selector offers **Base**, **SFT**, **GRPO**, and **GPT-6 Luna**. Base uses
`Qwen/Qwen3.8-27B`; SFT uses `artsy/sft-v2-qwen3.8-27b-final-preview-masked-cosine`;
GRPO uses `artsy/grpo-v4-subset200-b16-c64`. These three require the native bridge
(`make serve`) with the completed checkpoints available. GPT-6 Luna uses
`openai/gpt-6-luna` through OpenRouter, requires `OPENROUTER_API_KEY` in `.env`,
and works without the bridge. Luna is the default; the browser remembers your
selection. The selected model handles both new drawings and subsequent edits,
so you can switch models between versions.

The consumer endpoint (`POST /api/artsy/runs`) accepts the selected model ID and
enforces **10 internal model turns per response**, including edits. Start Docker
before drawing.
Edits reuse the completed drawing's verified pycairo source. Choose **Auto** to
let Artsy continue drawing, or **Approve steps** to pause before every drawing
action, review its details, and click **Approve action** to continue. Each approval
releases only that action, including when one model turn proposes several actions.
The mode is fixed for the current drawing and can be changed before the next request.

The drawing canvas shows only the current stage during generation. Each new visible model
update, action, or preview replaces the previous stage instead of building a timeline.
Both modes show the same current stage; Auto continues without pausing, while
Approve steps adds an approval button for the pending action. Each action uses one plain-language message, changing from an action label to its result when it finishes.
Raw output, source code, and arguments are not displayed in the consumer UI.
Assistant messages and tool actions appear one at a time. Saved history uses
Previous and Next controls to inspect one recorded step at a time.
New runs record tool arguments live, including the drawing source.
Named harness tools
provide short drawing labels such as “Refining your drawing” and “Checking the preview.” While a model turn has no text,
the latest action or preview stays visible; before the first action, the canvas shows “Thinking…”. Once the final image passes validation, it takes the place
of the live steps. Recorded steps and previews remain visible after completion, with Previous and Next
buttons shown by default. Activity survives
reloads and can be reopened with a saved drawing or version; older drawings use
their exported trace when available. Progress comes from recorded execution events.
The browser stores sketchbook metadata and the selected mode; run artifacts and
step history stay on the local server under `episodes/ui/`.

Voice input uses the browser's Web Speech API and starts only when the user taps
the mic. Microphone permission and a supported browser are required. Typing is
always available and opens automatically when recognition is unsupported or
fails. **Kokoro** generates the greeting and spoken updates locally, using the
`af_heart` voice through [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx).
Install it once with `make tts-setup` (Python 3.12, isolated
`.tts-venv`, approximately 120 MB of model/voice files in `.cache/kokoro/`). Then
start the UI normally; it needs no separate TTS server or API key. The same-origin
`POST /api/artsy/speech` endpoint returns 24 kHz mono WAV, with a bounded cache
for repeated phrases. `ARTSY_KOKORO_VOICE` selects the voice; optional
`ARTSY_TTS_PYTHON`, `ARTSY_KOKORO_MODEL`, and `ARTSY_KOKORO_VOICES` override paths.
These settings load from `.env`; exported variables take precedence.

Browser Web Audio plays the Kokoro audio. Autoplay policies may require an
initial interaction before the greeting is audible. The mute control stops
playback, and tapping the mic cancels speech and any pending speech request.
Recognition still uses the browser's speech-to-text service, which may process
microphone audio remotely. Browser-native text-to-speech is not used.

The developer **Studio** is now at `/studio`; `/drawings` remains an alias.
Start the native bridge (`make serve`) for Studio's Qwen models. The service uses
`.eval-venv/bin/python`, falling back to `.venv/bin/python`; override with
`ARTSY_EVAL_PYTHON`. Install the evaluation dependencies documented in the
repository README. Change the port with `ARTSY_UI_PORT`.

The developer workspaces retain their shared navigation and styling:

- **Studio** (`/studio`) writes a freeform prompt or selects an ordinary drawing from a dataset, using one model selector, output panel, timeline, and artifact downloads.
- **Prompt review** searches and reviews drawing and safety prompts, saves notes and decisions, and records named dataset approval. Choose Safety prompts to retain the separate safety-only approval workflow.
- **SFT candidates** inspects full trajectories and images, saves manual decisions, and exports approved, structurally trainable examples.
- **Runs** (`/runs`) launches evaluations, supervised fine-tuning, and GRPO jobs from validated CLI settings, with status, console output, history, and Stop controls.
- **Eval comparison** (`/comparisons`) shows saved Luna judgments for 2–8 models with final images, individual 0–4 adherence and visual-quality scores, comments, and preferences. Search prompts/comments, filter by preferred model or missing outputs, sort by adherence spread, and click images to enlarge. Summary averages use only prompts with scored images from every model. It discovers judge manifests under `eval-results/`; browsing makes no model calls. `evaluation.judge_pairs --run base=RUN --run rank32=RUN --run rank64=RUN --run rank128=RUN --output DIR` sends all available final images per prompt in one request. `evaluation.compare_pipeline` accepts the same run/output arguments, waits for launched evals, then runs the judge; `pipeline.json` records progress and failures.

The **Combined eval · 391 prompts** collection joins the original 163 scored
prompts with the 228 added prompts. Its Prompt subset selector updates both the
gallery and score averages. New final images appear on Refresh results while
generation continues; unfinished samples and completed missing outputs have
separate filters. Unscored images never count as zero in score averages. A
`collection.json` under `eval-results/<collection>/` defines the prompt snapshot,
original comparison, per-model added eval directories, and optional added
judgments directory. Saved judgments are attached only when model identities,
prompt text, and final image hashes match. Browsing or refreshing never starts
generation or judging. The historical comparison remains independently available.
- **RL prompts** (`/rl/prompts`) browses the complete manifest-selected training prompt set, with search and category filters.
- **RL rollouts** (`/rl/rollouts`) passively inspects completed RL episode artifacts and rewards, without launching or modifying training. The legacy `/rl` URL opens this workspace.

Run all workspaces with one command:

```sh
.venv/bin/python -m ui.server \
  --dataset datasets/sft-v2/prompts.jsonl \
  --candidates datasets/sft-v2/candidates.jsonl
```

Open http://127.0.0.1:4173. No separate inspector servers are needed. Without
`--dataset`, Studio and Prompt review use the frozen benchmark. Without
`--candidates`, the candidate workspace explains how to configure a file.
`ARTSY_DRAWING_DATASET` and `ARTSY_CANDIDATE_DATASET` can supply these paths.

Prompt decisions use the existing `prompts.review.json` beside a custom prompt
file, or `eval-results/dataset-review/approval.json` for historical benchmark v1. The validated default suite is the October 28 50-prompt set; its recorded validation does not require repeat approval for evaluation.
Candidate decisions default to `manual-review.json` and approved exports to
`accepted.jsonl` beside the candidates. Override with `--review-output`,
`--candidate-review`, and `--accepted-output`. The original data is preserved.

The old `/drawings` URLs open the same Studio in dataset mode; saved `#run=...`
and `#prompt=...` links remain usable. Standalone viewer launchers have been removed; review handlers live inside `ui`.
Only clicking Generate or Run starts sampling; switching prompt source, model,
or workspace does not. Dataset runs preserve canonical prompt text, seed,
dataset hash, and prompt ID. Saved runs are scoped to the dataset revision.

Studio Generate launches the existing `evaluation.draw` CLI. Model calls use credits. No
generated code is executed by the web server. Only one UI job runs at a time. The Studio
dropdown includes the SFT base-model choices (Qwen3.5 4B, Qwen3.6
35B-A3B and Qwen3.8 27B), GPT-6 Luna, plus saved SFT models,
completed GRPO models, and intermediate GRPO step checkpoints advertised by
the native bridge. Final checkpoints come from `checkpoints/*/final.json`;
intermediate GRPO checkpoints are discovered via
`checkpoints/*/training-manifest.json` and `checkpoints/*/checkpoints.jsonl`.
Each checkpoint uses its own sampler path and model-specific tokenizer,
renderer, and context profile. The bridge refreshes its model catalog
periodically; reload Studio to see new options.
Qwen uses the local native Tinker sampling bridge. GPT-6 Luna uses OpenRouter's
OpenAI-compatible endpoint with `OPENROUTER_API_KEY` from the environment or `.env`.
Both use the same Docker drawing harness with explicit update, read, preview, submit,
and refusal tools. GPT-6 Luna does not require the local bridge. Existing fine-tunes
were trained on the historical bash tool schema and have not been retrained for these tools.
Removed models are not restored to the dropdown by saved runs.


Runs are stored in `episodes/ui/<id>/`. The UI shows process output while running. After the CLI finishes, it exports the full Inspect logs with resolved attachments to `output/trace.json`. The completed-run timeline pairs each named tool call with its recorded results, shows failed renders and inline previews, and distinguishes submission requests from final render validation. Source rewrites show a line diff against the previous successful write, with full tool arguments and source available to expand. Historical bash runs remain readable. Model response timing and token counts appear when recorded. The Transcript view preserves all messages; the Events view searches full event data and loads raw records when expanded. Failed actions open automatically. Downloads include the final PNG, Python source, result JSON, and full trace when present. Refusals and failures are displayed without substituting artwork. Reloading the same browser tab resumes polling its last run. Run URLs include `#run=<id>` so a saved run can be reopened directly.

The trace is recorded execution history, not inferred or invented model reasoning. Studio's full Inspect events appear after completion; the consumer view shows only the current stage live and keeps history in the completed drawing's disclosure. Runs interrupted by server shutdown are not resumed. Keep the server running until the job ends. This service binds only to loopback, checks Host and Origin, and serves only the UI and explicitly allowed run artifacts.

Validation: `node --check ui/app.js`, `node --test ui/test_trace.cjs`,
`.venv/bin/python -m unittest ui.test_server`,
and browser checks. No paid model generation was run during implementation.


## Passive RL prompts and rollouts

Open http://127.0.0.1:4173/rl/prompts for the complete prompt browser or
http://127.0.0.1:4173/rl/rollouts for the completed rollout gallery
(the old `/rl` URL remains an alias for rollouts). Each workspace has its own
HTML page, so the other workspace cannot appear even if old CSS is cached.
They share a run selector but fetch only their own data. Training proceeds in a separate process.
After updating, restart `python -m ui.server` so the new routes load.
By default it looks for RL run manifests under `checkpoints/`. To monitor a
different output directory, start the UI with `--rl-root /path/to/checkpoints`
(or point it directly at a single RL run). `ARTSY_RL_ROOT` also sets this root.

The **RL prompts** workspace lists every prompt selected in the run
manifest, even prompts absent from recent rollouts. It offers full-text search,
category filters (including safety/refusal categories), expandable original prompt
text, and 50-prompt pagination. The viewer finds a matching prompt JSONL in
`datasets/*/prompts.jsonl`, preferring `datasets/parti-prompts/rl/prompts.jsonl`.
For an external or nonstandard dataset path, pass
`--rl-dataset /path/to/original/prompts.jsonl` or set `ARTSY_RL_DATASET`.
The source file must match the **SHA-256 recorded when RL training began**;
only manifest-selected IDs are displayed, respecting `--limit`. Missing or
modified files produce a visible explanation instead of incorrect prompts.
The full list loads only when a run is selected or Refresh now is clicked;
it is not re-read on each 30-second rollout poll.

The two RL workspaces make **no changes to the trainer**, launch no Docker containers,
and invoke no models or judges. The rollout viewer reads at most 48 completed
`result.json` files from the latest three `rollouts/step-*/` directories,
refreshing every 30 seconds while the browser tab is visible. Intermediate
previews and complete model messages are not saved by the current RL trainer,
so the monitor shows terminal artifacts and recorded bash commands only.
Rollouts sharing one prompt and optimizer step are grouped under a single prompt
heading, with each sampled variant selectable independently. Search retains
original variant numbering. Every saved final PNG appears as a lazy-loaded
thumbnail alongside terminal reward, adherence, and outcome, so variants can be
compared at a glance. Refusals or failed renders show a no-image placeholder.
Selecting a tile opens the existing full-size image and detailed judgments;
final source is read when expanded. Unchanged groups retain their DOM nodes
between periodic polls so their images are not re-requested every 30 seconds.
If training is on another host, mount/sync the output directory to the UI host.

The UI is a separate read-only process, but filesystem reads are not literally
zero-cost. Disable the monitor or run it on another host if your training
machine has constrained disk bandwidth. No data is indexed or written
by the monitor. Prompt text is read from the original dataset on demand; runs can appear before they complete because the manifest is
created at training startup.

RL prompt browser tests: `python -m unittest ui.test_rl_monitor` and `node --test ui/test_rl_monitor.cjs`.

## Launch evaluation, SFT, and RL from the browser

Open **Runs** at http://127.0.0.1:4173/runs. Choose a workflow, edit
its CLI settings, and click **Launch**. A confirmation is shown before every
non-dry-run launch because these jobs can consume model credits and compute.

The model selectors query the existing Artsy inference catalog and local
checkpoint metadata. **Evaluation** lists base models, completed SFT/GRPO
models, and saved intermediate GRPO sampler checkpoints (for example,
`artsy/grpo-run--grpo-step-000023`) for supported Qwen3.5, Qwen3.6 35B-A3B,
and Qwen3.8 27B models. GRPO steps can be evaluated before RL finishes; only
records with a published Tinker sampler are listed. The dropdown distinguishes
models currently advertised by the native bridge.
Choices not served by a connected bridge are disabled; if the bridge is
offline, known options are still shown with an offline warning. **SFT** defaults to `Qwen/Qwen3.6-35B-A3B:peft:262144` (256K-context Tinker training model) and also offers
Qwen3.5 4B and Qwen3.8 27B vision base models. The suffix is sent unchanged to Tinker; the tokenizer and image processor are loaded using the unsuffixed Hugging Face model ID. The older Qwen3.5 35B-A3B chat checkpoint is retired on Tinker; select the Qwen3.6 vision model instead.
**RL** offers completed SFT checkpoint files as its initialization choices;
RL final checkpoints are deliberately excluded. A **Refresh models** button
finds newly saved checkpoints without restarting the UI, and each selector
retains a custom-entry option. These selectors do not change which models
the underlying bridge can actually serve or which models Tinker supports.

- **Evaluation** calls `evaluation.evaluate` using `.eval-venv/bin/python` when
  available, falling back to `.venv`. Set an evaluation model that is actually
  served by your configured bridge/provider; evaluation is not limited to
  Studio's static model dropdown. The default prompt set is the frozen benchmark.
- **SFT** calls `training.sft` using `.venv/bin/python` when available.
  It requires a reviewed, accepted trajectory JSONL (for example
  `datasets/sft-v2/accepted.jsonl`). Base model, LoRA rank, epochs, batch
  size, learning rate and maximum updates can be changed.
- **RL** calls `training.grpo` using `.venv/bin/python` when available.
  Configure the prompt dataset, SFT initialization checkpoint, updates, group
  size, concurrency and hyperparameters. The optional **Continue from**
  directory uses the trainer's existing continuation mode, inheriting previous
  training settings; enter the extra update count instead of overriding them.

The UI creates a unique output for each launch: `eval-results/ui-eval-<id>`
or `checkpoints/ui-{sft,rl}-<id>`. Its own job metadata and bounded console
log live in `episodes/ui-launches/<id>/`. Polling the run detail shows the
latest 100 KB of stdout/stderr, job exit state, and the exact argv. The passive
RL monitor discovers completed rollouts in the new RL checkpoint directories.
Use the existing CLIs to resume a failed trainer run that requires its original
output directory; the UI creates a new directory for each launch.

Only one optimizer (SFT or RL) launch can be active at a time; evaluation can
run concurrently. Run cancellation sends SIGTERM to the subprocess group
on POSIX (or terminates the direct child on Windows). Cancellation is available
only while the current UI server owns the child process; if the UI server
restarts, an already running child can outlive its tracking thread. The UI
then displays the saved PID/state, but cannot cancel that orphaned process.
A run whose PID has vanished without a recorded exit is marked interrupted.

The UI only accepts recognized flags and repository-local input paths. It does
not provide arbitrary shell command execution or expose environment/API keys.
Existing `.env` variables still configure Tinker, the inference bridge, and
the evaluation judges. Set `ARTSY_TRAIN_PYTHON` or `ARTSY_EVAL_PYTHON`
to explicitly choose an interpreter.

Local checks: `python -m unittest ui.test_launch ui.test_server`,
`node --check ui/runs.js`.

## Blind ranking

Open `/preferences` to rank Base, Parent SFT, and New GRPO outputs for all 40
unique drawing prompts, once each (including benign boundary cases). Enter a
reviewer name to start or resume. For each model, select the earliest repetition
with a matching prompt/seed, verified image hash, valid render, and no automatic
harmful-image or malicious-code flag. If none exists, show a no-image outcome
rather than omitting the prompt. This uses existing outputs without generation.

Images appear as A/B/C in a stable, independently shuffled order per case and
reviewer. Model names, paths, and automatic scores are not included in ranking
responses. Rank 1 is best; equal ranks are ties. Save all cases before revealing
scores. Revealing permanently locks that reviewer's votes. Each pairwise win earns
1, a tie 0.5, and a loss 0; each model's two comparisons are averaged, then unique prompts are averaged and multiplied by 100.
Scores are separate per reviewer, not an automatic quality/safety metric.

Rankings persist in `eval-results/human-preferences/rankings.json`; per-reviewer
score JSON files are exported there on reveal. The private mapping stays on the
server. Changing image bytes creates new case IDs rather than reusing old votes.
This is a local trusted-reviewer workflow, not authenticated multi-user access.
