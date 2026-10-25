# Artsy generation UI

From the repository root:

```sh
python3 -m ui.server
```

Open http://127.0.0.1:4173. Start Docker and the native bridge (`make serve`) before generating. The service uses `.eval-venv/bin/python`, falling back to `.venv/bin/python`; override with `ARTSY_EVAL_PYTHON`. Install the evaluation dependencies documented in the repository README. Change the port with `ARTSY_UI_PORT`.

Generate launches the existing `evaluation.draw` CLI. Model calls use credits. No generated code is executed by the web server: the drawing CLI retains its Docker execution boundary. Only one UI job runs at a time. The model field must match the configured bridge. To use a trained checkpoint, set `ARTSY_TINKER_MODEL_PATH` in the environment or repository `.env`, and restart the bridge with the same configuration.

Runs are stored in `episodes/ui/<id>/`. The UI shows process output while running. After the CLI finishes, it exports the full Inspect logs with resolved attachments to `output/trace.json`. The UI displays recorded messages, tool calls, image previews, errors, and raw events. Downloads include the final PNG, Python source, result JSON, and full trace when present. Refusals and failures are displayed without substituting artwork. Reloading the same browser tab resumes polling its last run.

The trace is the recorded execution history, not inferred or invented model reasoning. Full agent events currently appear after completion, not live. Runs interrupted by server shutdown are not resumed. Keep the server running until the job ends. This service binds only to loopback, checks Host and Origin, and serves only the UI and explicitly allowed run artifacts.

Validation: `node --check ui/app.js`, `python3 -m unittest ui.test_server`, and browser checks. No paid model generation was run during implementation.
