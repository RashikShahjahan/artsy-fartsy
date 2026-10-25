# Artsy generation UI

From the repository root:

```sh
python3 -m ui.server
```

Open http://127.0.0.1:4173. Start Docker and the native bridge (`make serve`) before generating. The service uses `.eval-venv/bin/python`, falling back to `.venv/bin/python`; override with `ARTSY_EVAL_PYTHON`. Install the evaluation dependencies documented in the repository README. Change the port with `ARTSY_UI_PORT`.

Generate launches the existing `evaluation.draw` CLI. Model calls use credits. No generated code is executed by the web server: the drawing CLI retains its Docker execution boundary. Only one UI job runs at a time. The model dropdown uses the Qwen3.5-4B model served by the bundled bridge. Reopened runs retain their recorded model in the dropdown. Generation uses seed 42 automatically. To use a trained checkpoint, set `ARTSY_TINKER_MODEL_PATH` in the environment or repository `.env`, and restart the bridge with the same configuration.

Runs are stored in `episodes/ui/<id>/`. The UI shows process output while running. After the CLI finishes, it exports the full Inspect logs with resolved attachments to `output/trace.json`. The completed-run timeline pairs each command with its recorded tool results, shows failed renders and inline previews, and distinguishes submission requests from final render validation. Source rewrites show a line diff against the previous successful write, with full commands and source available to expand. Model response timing and token counts appear when recorded. The Transcript view preserves all messages; the Events view searches full event data and loads raw records when expanded. Failed actions open automatically. Downloads include the final PNG, Python source, result JSON, and full trace when present. Refusals and failures are displayed without substituting artwork. Reloading the same browser tab resumes polling its last run. Run URLs include `#run=<id>` so a saved run can be reopened directly.

The trace is the recorded execution history, not inferred or invented model reasoning. Full agent events currently appear after completion, not live. Runs interrupted by server shutdown are not resumed. Keep the server running until the job ends. This service binds only to loopback, checks Host and Origin, and serves only the UI and explicitly allowed run artifacts.

Validation: `node --check ui/app.js`, `node --test ui/test_trace.cjs`, `python3 -m unittest ui.test_server`, and browser checks. No paid model generation was run during implementation.
