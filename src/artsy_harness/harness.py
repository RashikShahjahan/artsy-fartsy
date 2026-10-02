"""Episode storage, local Python execution, and PNG validation (not a sandbox)."""

import ast
import hashlib
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .models import PROMPT_TEMPLATE, PROMPT_VERSION

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024
LIMITS = {
    "wall_seconds": 30,
}


def validate_png(data):
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("PNG exceeds byte limit")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("Renderer output is not PNG")
            width, height = image.size
            if not (0 < width <= 4096 and 0 < height <= 4096):
                raise ValueError("PNG dimensions outside allowed range")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    return {
        "width": width,
        "height": height,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _collect(stream, target):
    # Drain pipes continuously, but never keep unbounded attacker output.
    while chunk := stream.read(8192):
        target.extend(chunk[: max(0, MAX_LOG_BYTES - len(target))])
    stream.close()


def _render(program, timeout):
    result = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False}
    with tempfile.TemporaryDirectory(prefix="artsy-") as temporary:
        root = Path(temporary)
        source = root / "program.py"
        source.write_text(program)
        output = root / "output"
        output.mkdir()
        command = [sys.executable, "-B", str(source)]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
        result["command"] = command
        with subprocess.Popen(command, cwd=output, env=env, start_new_session=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            stdout, stderr = bytearray(), bytearray()
            threads = [
                threading.Thread(target=_collect, args=(process.stdout, stdout)),
                threading.Thread(target=_collect, args=(process.stderr, stderr)),
            ]
            for thread in threads:
                thread.start()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
            finally:
                # Also stop children that inherited the output pipes.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            for thread in threads:
                thread.join()
            result.update(
                returncode=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        if result["returncode"] != 0 or result["timed_out"]:
            return result, None
        # Refuse symlinks, directories, and special files produced by the program.
        try:
            fd = os.open(output / "output.png", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError("Output must be a regular file")
                data = handle.read(MAX_IMAGE_BYTES + 1)
        except (OSError, ValueError) as error:
            result["output_error"] = str(error)
            return result, None
        return result, data


def run_episode(program, episodes, *, timeout=LIMITS["wall_seconds"], model=None, prompt=None):
    if not 0 < timeout <= LIMITS["wall_seconds"]:
        raise ValueError("Timeout must be between 0 and 30 seconds")
    episode_id = uuid.uuid4().hex
    directory = Path(episodes) / episode_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 2,
        "episode_id": episode_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "model_generated" if model else "hand_written",
        "program": None,
        "program_sha256": None,
        "execution": {"backend": "local_python", "sandboxed": False,
                      "python": sys.executable, "python_version": sys.version},
        "limits": {**LIMITS, "wall_seconds": timeout},
        "status": "execution_failed",
        "renderer": None,
        "image": None,
    }
    try:
        if model:
            record["status"] = "model_call_failed"
            full_prompt = PROMPT_TEMPLATE + prompt
            payload = model.request(full_prompt)
            record["generation"] = {
                "provider": model.provider, "endpoint": model.endpoint,
                "timeout_seconds": model.timeout, "requested_model": model.model,
                "prompt_version": PROMPT_VERSION, "user_prompt": prompt,
                "request": payload, "response_contract": "verbatim-python-v1",
            }
            response = model.generate(payload)
            generation = record["generation"]
            with (directory / "response.body").open("xb") as handle:
                handle.write(response.raw)
            generation.update(raw_response="response.body", http_status=response.http_status,
                              returned_model=response.model, returned_digest=response.digest)
            record["status"] = "protocol_failed"
            if response.error:
                raise ValueError(response.error)
            program = response.text
            with (directory / "generated.txt").open("x", encoding="utf-8", newline="") as handle:
                handle.write(program)
            generation["generated_text"] = "generated.txt"

        with (directory / "program.py").open("x", encoding="utf-8", newline="") as handle:
            handle.write(program)
        record.update(program="program.py", program_sha256=hashlib.sha256(program.encode()).hexdigest())
        if model:
            record["status"] = "syntax_failed"
            record["generation"]["syntax"] = "failed"
            if not program.strip():
                raise ValueError("Empty model output")
            ast.parse(program, filename="program.py")
            record["generation"]["syntax"] = "valid"
        record["status"] = "execution_failed"
        result, data = _render(program, timeout)
        record["renderer"] = result
        if result.get("output_error"):
            record["status"] = "validation_failed"
            raise ValueError(result["output_error"])
        if data is not None:
            record["status"] = "validation_failed"
            metadata = validate_png(data)
            with (directory / "image.png").open("xb") as handle:
                handle.write(data)
            record.update(status="success", image={"path": "image.png", **metadata})
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        record["error"] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            record["error"] += ": " + (error.stderr or "")[-MAX_LOG_BYTES:]
    with (directory / "record.json").open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return directory, record
