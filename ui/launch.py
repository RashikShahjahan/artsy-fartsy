"""Local, allowlisted launch service for the Artsy evaluation/SFT/GRPO CLIs.

Jobs run as child processes of the UI server, not inside HTTP request threads.
No shell, arbitrary commands, or user-selected output directories are supported.
"""
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "episodes" / "ui-launches"
LOCK = threading.RLock()
PROCESSES = {}

# Each job receives its own directory; the actual trainer writes to checkpoints/
# or eval-results/ so existing tooling and the passive RL monitor still work.
OPTIONS = {
    "eval": {
        "model": ("--model", "model"),
        "dataset": ("--dataset", "file"),
        "limit": ("--limit", "positive"),
        "epochs": ("--epochs", "positive"),
        "max_samples": ("--max-samples", "positive"),
        "max_connections": ("--max-connections", "positive"),
        "max_turns": ("--max-turns", "positive"),
        "execution_only": ("--execution-only", "bool"),
        "dry_run": ("--dry-run", "bool"),
    },
    "sft": {
        "dataset": ("--dataset", "file"),
        "model": ("--model", "model"),
        "epochs": ("--epochs", "positive"),
        "max_steps": ("--max-steps", "steps"),
        "batch_size": ("--batch-size", "positive"),
        "learning_rate": ("--learning-rate", "rate"),
        "lora_rank": ("--lora-rank", "positive"),
        "save_every": ("--save-every", "positive"),
        "allow_unreviewed": ("--allow-unreviewed", "bool"),
        "dry_run": ("--dry-run", "bool"),
    },
    "rl": {
        "dataset": ("--dataset", "file"),
        "init_checkpoint": ("--init-checkpoint", "file"),
        "continue_from": ("--continue-from", "directory"),
        "additional_steps": ("--additional-steps", "positive"),
        "epochs": ("--epochs", "positive"),
        "max_steps": ("--max-steps", "steps"),
        "batch_size": ("--batch-size", "positive"),
        "group_size": ("--group-size", "group"),
        "max_samples": ("--max-samples", "positive"),
        "learning_rate": ("--learning-rate", "rate"),
        "kl_coef": ("--kl-coef", "nonnegative"),
        "save_every": ("--save-every", "positive"),
        "dry_run": ("--dry-run", "bool"),
    },
}
MODULES = {"eval": "evaluation.evaluate", "sft": "training.sft", "rl": "training.grpo"}


def model_choices(bridge_models=None):
    """Existing inference models and trainable SFT checkpoints for run forms.

    A served model is selectable for evaluation. Checkpoint discovery is local
    and metadata-only; it neither loads sampler weights nor starts a model.
    """
    from drawing.config import MODEL_ID
    from inference.model_catalog import LARGE_MODEL_ID, model_catalog as inference_models

    served = {row["id"] for row in bridge_models or []
              if isinstance(row, dict) and isinstance(row.get("id"), str)}
    known = {}
    for entry in inference_models(ROOT):
        model_id = entry.get("id")
        if isinstance(model_id, str) and model_id:
            known[model_id] = entry.get("label") or model_id
    # Show bridge-advertised models even when they were not in the local cache.
    for entry in bridge_models or []:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            known.setdefault(entry["id"], entry.get("label") or entry["id"])
    evaluation = [
        {"value": model_id, "label": label, "available": model_id in served}
        for model_id, label in known.items()
    ]
    # The SFT trainer validates these Qwen base model families; its --model
    # argument does not accept an existing fine-tune sampler checkpoint.
    sft = [
        {"value": MODEL_ID, "label": "Qwen3.5 · 4B · base"},
        {"value": LARGE_MODEL_ID, "label": "Qwen3.8 · 27B · base"},
    ]
    checkpoints = []
    for file in (ROOT / "checkpoints").glob("*/final.json"):
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
            config = record.get("config")
            if not isinstance(config, dict) or config.get("algorithm") is not None:
                continue  # GRPO final.json is not a valid SFT initialization.
            base_model = config.get("model")
            if (not isinstance(base_model, str) or not base_model.startswith("Qwen/")
                    or not isinstance(config.get("renderer"), str)
                    or type(config.get("lora_rank")) is not int or config["lora_rank"] < 1
                    or any(not isinstance(record.get(key), str)
                           or not record[key].startswith("tinker://")
                           for key in ("state_path", "sampler_path"))):
                continue
            relative = file.relative_to(ROOT).as_posix()
            checkpoints.append((file.stat().st_mtime, {
                "value": relative,
                "label": f"{file.parent.name} · {base_model} · step {record.get('step', 'final')}",
                "model": base_model,
            }))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    checkpoints.sort(key=lambda entry: entry[0], reverse=True)
    return {
        "eval": evaluation, "sft": sft,
        "rl": [entry for _, entry in checkpoints],
        "bridge_online": bridge_models is not None,
    }
IDENTIFIER = re.compile(r"[a-f0-9]{32}\Z")
MODEL = re.compile(r"[a-zA-Z0-9_./:+-]{1,200}\Z")


def _python(kind):
    key = "ARTSY_EVAL_PYTHON" if kind == "eval" else "ARTSY_TRAIN_PYTHON"
    if os.getenv(key):
        return os.environ[key]
    names = (".eval-venv", ".venv") if kind == "eval" else (".venv", ".eval-venv")
    for name in names:
        executable = ROOT / name / "bin" / "python"
        if executable.is_file():
            return str(executable)
    return "python3"


def _path(value, directory=False):
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValueError("Expected a repository-relative input path.")
    candidate = (ROOT / value).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError("Inputs must be inside the Artsy repository.")
    if directory:
        if not candidate.is_dir():
            raise ValueError(f"Directory not found: {value}")
    elif not candidate.is_file():
        raise ValueError(f"File not found: {value}")
    return str(candidate)


def _value(name, kind, value):
    if kind == "bool":
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean.")
        return value
    if kind in ("file", "directory"):
        return _path(value, directory=kind == "directory")
    if kind == "model":
        if not isinstance(value, str) or not MODEL.fullmatch(value):
            raise ValueError(f"Invalid {name} model ID.")
        return value
    if kind in ("positive", "group", "steps"):
        if type(value) is not int or (kind == "steps" and value != -1 and value < 1) or (
                kind != "steps" and value < (2 if kind == "group" else 1)) or value > 1_000_000:
            raise ValueError(f"Invalid {name}.")
        return value
    if kind in ("rate", "nonnegative"):
        if type(value) not in (int, float) or not math.isfinite(value) or (
                value <= 0 if kind == "rate" else value < 0):
            raise ValueError(f"Invalid {name}.")
        return value
    raise ValueError(f"Unrecognized option: {name}")


def command(kind, settings, run_id):
    """Validate inputs then build the exact argument vector used by the CLI."""
    if kind not in OPTIONS or not isinstance(settings, dict):
        raise ValueError("Choose evaluation, SFT, or RL and provide settings.")
    unknown = set(settings) - set(OPTIONS[kind])
    if unknown:
        raise ValueError("Unsupported setting: " + sorted(unknown)[0])
    selected = {key: value for key, value in settings.items() if value is not None and value != ""}
    if kind == "sft" and not selected.get("dataset"):
        raise ValueError("SFT requires an accepted training dataset JSONL.")
    if kind == "rl":
        continuation = bool(selected.get("continue_from"))
        if continuation:
            forbidden = set(selected) - {"continue_from", "additional_steps", "dataset", "save_every", "dry_run"}
            if forbidden:
                raise ValueError("RL continuation inherits training settings; remove " + sorted(forbidden)[0])
            if not selected.get("additional_steps"):
                raise ValueError("RL continuation requires additional steps.")
        elif "additional_steps" in selected:
            raise ValueError("Additional steps requires a continuation directory.")
    output = (ROOT / ("eval-results" if kind == "eval" else "checkpoints") /
              ("ui-" + kind + "-" + run_id)).resolve()
    argv = [_python(kind), "-u", "-m", MODULES[kind]]
    for name, (flag, category) in OPTIONS[kind].items():
        if name not in selected:
            continue
        parsed = _value(name, category, selected[name])
        if category == "bool":
            if parsed:
                argv.append(flag)
        else:
            argv.extend((flag, str(parsed)))
    argv.extend(("--output" if kind == "eval" else "--output-dir", str(output)))
    return argv, output


def _write(folder, payload):
    tmp = folder / "job.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(folder / "job.json")


def _load(run_id):
    if not IDENTIFIER.fullmatch(run_id):
        raise ValueError("Invalid run ID.")
    file = RUNS / run_id / "job.json"
    if not file.is_file():
        raise FileNotFoundError("Run not found.")
    return json.loads(file.read_text(encoding="utf-8"))


def _alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _state(record):
    # A new UI server can still display an orphaned run. It cannot cancel a
    # child it no longer owns, so a vanished PID is marked interrupted.
    if record["state"] in ("running", "stopping") and not _alive(record.get("pid")):
        record = {**record, "state": "interrupted",
                  "error": "The process exited without a recorded result (UI server may have restarted)."}
    return record


def list_runs():
    if not RUNS.is_dir():
        return []
    files = sorted(RUNS.glob("*/job.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = []
    for file in files[:80]:
        try:
            record = _state(json.loads(file.read_text(encoding="utf-8")))
            result.append({k: record.get(k) for k in
                           ("id", "kind", "state", "created_at", "finished_at", "output", "pid")})
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return result


def get_run(run_id):
    record = _state(_load(run_id))
    console = RUNS / run_id / "console.txt"
    if console.exists():
        with console.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 100_000))
            record["console"] = stream.read().decode("utf-8", errors="replace")
    else:
        record["console"] = ""
    with LOCK:
        process = PROCESSES.get(run_id)
        record["can_cancel"] = process is not None and process.poll() is None
    return record


def _finish(run_id, process):
    try:
        returncode = process.wait()
        with LOCK:
            record = _load(run_id)
            record["state"] = "cancelled" if record["state"] == "stopping" else (
                "completed" if returncode == 0 else "failed")
            record["exit_code"] = returncode
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write(RUNS / run_id, record)
    finally:
        with LOCK:
            PROCESSES.pop(run_id, None)


def start(kind, settings):
    run_id = uuid4().hex
    argv, output = command(kind, settings, run_id)
    with LOCK:
        # Prevent overlapping optimizer jobs; eval may run alongside training.
        if kind in ("sft", "rl") and (
                any(p.poll() is None and _load(ident).get("kind") in ("sft", "rl")
                    for ident, p in PROCESSES.items())
                or any(row.get("kind") in ("sft", "rl")
                       and row.get("state") in ("running", "stopping") for row in list_runs())):
            raise RuntimeError("An SFT or RL training run is already active.")
        folder = RUNS / run_id
        folder.mkdir(parents=True, exist_ok=False)
        record = {
            "id": run_id, "kind": kind, "state": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": argv, "output": str(output.relative_to(ROOT)),
        }
        _write(folder, record)
        try:
            with (folder / "console.txt").open("wb") as log:
                process = subprocess.Popen(argv, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, env=os.environ.copy(),
                                           start_new_session=(os.name == "posix"))
            record.update(state="running", pid=process.pid)
            _write(folder, record)
            PROCESSES[run_id] = process
            threading.Thread(target=_finish, args=(run_id, process), daemon=True).start()
        except Exception as error:
            record.update(state="failed", error=str(error),
                          finished_at=datetime.now(timezone.utc).isoformat())
            _write(folder, record)
            raise RuntimeError(str(error)) from error
    return {"id": run_id, "output": record["output"]}


def cancel(run_id):
    with LOCK:
        process = PROCESSES.get(run_id)
        if process is None or process.poll() is not None:
            raise ValueError("This run is no longer controlled by the current UI server.")
        record = _load(run_id)
        record["state"] = "stopping"
        _write(RUNS / run_id, record)
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    return {"id": run_id, "state": "stopping"}
