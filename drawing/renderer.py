"""Local subprocess renderer for ArtCanvas Python programs."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
import io
from pathlib import Path
from typing import Any

from PIL import Image

from .config import LIMITS

RUNTIME_VERSION = "artcanvas-v2"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 8192
RUNTIME_DIR = Path(__file__).resolve().parent.parent / "runtime"


def validate_png(png: bytes, expected: dict[str, int] | None = None) -> dict[str, int]:
    """Validate PNG framing, chunk CRCs, dimensions, and compressed image data."""
    if len(png) < 24 or len(png) > MAX_PNG_BYTES or not png.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid or oversized PNG")
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ValueError("PNG dimensions exceed 4096×4096")
    if expected and (width != expected["width"] or height != expected["height"]):
        raise ValueError(f"Expected {expected['width']}×{expected['height']} PNG")

    try:
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise ValueError("Invalid PNG dimensions or format")
            image.verify()
        with Image.open(io.BytesIO(png)) as image:
            image.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Invalid PNG image data") from exc
    return {"width": width, "height": height}


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        # The leader can exit while children remain in its session.
        pass
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
