"""Local subprocess renderer for ArtCanvas Python programs."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


from .config import LIMITS

from .render_contract import RUNTIME_VERSION, MAX_PNG_BYTES, MAX_LOG_BYTES, validate_png, RenderBackend

RUNTIME_DIR = Path(__file__).resolve().parent.parent / "runtime"

def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    # The leader can exit while children remain in its session.
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


class LocalRenderer:
    """Runs drawing code in a fresh process and validates its output artifact."""

    def __init__(self, python: str | None = None, seconds: int | None = None):
        if python is None:
            python = os.environ.get("ARTSY_PYTHON")
            if python is None:
                venv_python = Path(".venv/bin/python")
                python = str(venv_python.absolute()) if venv_python.exists() else sys.executable
        self.python = python
        self.seconds = seconds if seconds is not None else LIMITS["renderSeconds"]

    async def render(
        self,
        code: str,
        seed: int,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._render_sync, code, seed)

    def _render_sync(
        self,
        code: str,
        seed: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**53 - 1
                or len(code.encode("utf-8")) > LIMITS["sourceBytes"]):
            return {"ok": False, "stdout": "", "stderr": "", "error": "Invalid seed or source size",
                    "failureKind": "invalid_code", "runtimeVersion": RUNTIME_VERSION}

        temporary = tempfile.TemporaryDirectory(prefix="artsy-render-")
        work = Path(temporary.name)
        try:
            (work / "program.py").write_text(code, encoding="utf-8")
            runner = "import random, runpy; "
            if os.name == "posix":
                runner += (
                    "import resource; "
                    f"resource.setrlimit(resource.RLIMIT_FSIZE, ({MAX_PNG_BYTES}, {MAX_PNG_BYTES})); "
                    f"resource.setrlimit(resource.RLIMIT_CPU, ({max(1, int(self.seconds) + 1)}, {max(2, int(self.seconds) + 2)})); "
                )
            runner += f"random.seed({seed}); runpy.run_path('program.py', run_name='__main__')"
            env = os.environ.copy()
            env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": str(seed % (2**32))})
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(RUNTIME_DIR), existing]))
            if sys.platform == "darwin":
                env.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib:/usr/local/lib:/usr/lib")
            try:
                with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                    process = subprocess.Popen(
                        [self.python, "-c", runner], cwd=work, env=env,
                        stdin=subprocess.DEVNULL, stdout=stdout_file, stderr=stderr_file,
                        start_new_session=(os.name == "posix"),
                    )
                    timed_out = False
                    try:
                        process.wait(timeout=self.seconds)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        _kill_process_group(process)
                        process.wait()
                    finally:
                        # Also terminate descendants left behind by a completed leader.
                        _kill_process_group(process)
                    stdout_file.seek(0)
                    stderr_file.seek(0)
                    stdout = stdout_file.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
                    stderr = stderr_file.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
                    code_value = process.returncode
                    result = {"ok": code_value == 0 and not timed_out, "stdout": stdout, "stderr": stderr}
                    if code_value is not None:
                        result["exitCode"] = code_value
                    if timed_out:
                        result["error"] = "Rendering timed out"
            except OSError as exc:
                result = {"ok": False, "stdout": "", "stderr": "", "error": str(exc),
                          "failureKind": "infrastructure"}

            result["elapsedMs"] = int((time.monotonic() - started) * 1000)
            result["runtimeVersion"] = RUNTIME_VERSION
            if not result["ok"]:
                result.setdefault("failureKind", "invalid_code")
                return result

            try:
                output = work / "output.png"
                fd = os.open(output, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                with os.fdopen(fd, "rb") as file:
                    stat = os.fstat(file.fileno())
                    if not __import__("stat").S_ISREG(stat.st_mode) or not 24 <= stat.st_size <= MAX_PNG_BYTES:
                        raise ValueError("Expected a regular PNG of at most 8 MiB")
                    png = file.read(MAX_PNG_BYTES + 1)
                if len(png) > MAX_PNG_BYTES:
                    raise ValueError("Expected a regular PNG of at most 8 MiB")
                dimensions = validate_png(png)
                return {**result, "png": png, **dimensions}
            except (OSError, ValueError) as exc:
                return {**result, "ok": False, "error": str(exc), "failureKind": "invalid_code"}
        finally:
            temporary.cleanup()


class DockerRenderer:
    """Render untrusted programs in a disposable, network-isolated container."""

    def __init__(self, image: str | None = None, seconds: int | None = None,
                 docker: str = "docker"):
        self.image = image or os.environ.get("ARTSY_RENDER_IMAGE", "artsy-rl-runtime:latest")
        self.seconds = seconds if seconds is not None else LIMITS["renderSeconds"]
        self.docker = docker

    async def render(self, code: str, seed: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._render_sync, code, seed)

    def _render_sync(self, code: str, seed: int) -> dict[str, Any]:
        import base64
        import json
        import uuid

        started = time.monotonic()
        failure = {"ok": False, "stdout": "", "stderr": "", "runtimeVersion": RUNTIME_VERSION}
        if (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**53 - 1
                or not isinstance(code, str) or not code.strip()
                or len(code.encode("utf-8")) > LIMITS["sourceBytes"]):
            return {**failure, "error": "Invalid seed or source size", "failureKind": "invalid_code"}
        name = "artsy-render-" + uuid.uuid4().hex
        command = [self.docker, "run", "--rm", "--interactive", "--pull", "never", "--name", name,
                   "--network", "none", "--read-only", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--pids-limit", "64",
                   "--memory", "1g", "--cpus", "2", "--user", "65534:65534",
                   "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m", "--workdir", "/tmp",
                   self.image, "python", "/opt/artsy/render.py", "--seed", str(seed),
                   "--seconds", str(self.seconds), "--output", "/tmp/result", "--preview"]
        try:
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                           start_new_session=(os.name == "posix"))
                try:
                    process.communicate(code.encode("utf-8"), timeout=self.seconds + 15)
                except subprocess.TimeoutExpired:
                    _kill_process_group(process)
                    process.wait()
                    return {**failure, "error": "Rendering timed out", "failureKind": "infrastructure"}
                finally:
                    # Killing the client does not stop a container; remove it explicitly.
                    try:
                        subprocess.run([self.docker, "rm", "-f", name], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                stdout.seek(0)
                stderr.seek(0)
                raw = stdout.read(MAX_PNG_BYTES * 2 + MAX_LOG_BYTES * 4)
                diagnostic = stderr.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
                if process.returncode != 0:
                    return {**failure, "stderr": diagnostic, "error": "Docker renderer failed to start or exited unexpectedly", "failureKind": "infrastructure"}
            line, _, preview = raw.partition(b"\n")
            result = json.loads(line)
            if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
                raise ValueError("Invalid renderer response")
            result["runtimeVersion"] = RUNTIME_VERSION
            result["elapsedMs"] = int((time.monotonic() - started) * 1000)
            if not result["ok"]:
                result["failureKind"] = "invalid_code"
                return result
            prefix = b"<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>data:image/png;base64,"
            suffix = b"</MSWEA_MULTIMODAL_CONTENT>"
            preview = preview.strip()
            if not preview.startswith(prefix) or not preview.endswith(suffix):
                raise ValueError("Missing renderer image")
            png = base64.b64decode(preview[len(prefix):-len(suffix)], validate=True)
            dimensions = validate_png(png)
            return {**result, "png": png, **dimensions}
        except (OSError, ValueError, UnicodeError) as exc:
            return {**failure, "error": f"Docker renderer error: {exc}", "failureKind": "infrastructure"}
