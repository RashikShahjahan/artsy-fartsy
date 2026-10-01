"""Host-side episode storage, Docker execution, and untrusted PNG validation."""

import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024
LIMITS = {
    "wall_seconds": 30,
    "cpu_seconds": 10,
    "memory_bytes": 256 * 1024 * 1024,
    "cpus": 1,
    "pids": 32,
    "file_bytes": MAX_IMAGE_BYTES,
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


def _render(program, image, timeout):
    name = "artsy-" + uuid.uuid4().hex
    result = {"returncode": None, "stdout": "", "stderr": "", "timed_out": False}
    with tempfile.TemporaryDirectory(prefix="artsy-") as temporary:
        root = Path(temporary)
        source = root / "program.py"
        source.write_text(program)
        source.chmod(0o444)
        output = root / "output"
        output.mkdir()
        # mkdir's mode is masked by the host umask; the container UID needs access.
        output.chmod(0o777)
        command = [
            "docker", "run", "--name", name, "--pull=never", "--network=none",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=65534:65534", f"--pids-limit={LIMITS['pids']}",
            f"--memory={LIMITS['memory_bytes']}",
            f"--memory-swap={LIMITS['memory_bytes']}", f"--cpus={LIMITS['cpus']}",
            "--ulimit", f"cpu={LIMITS['cpu_seconds']}:{LIMITS['cpu_seconds']}",
            "--ulimit", f"fsize={LIMITS['file_bytes']}:{LIMITS['file_bytes']}",
            "--ulimit", "nofile=64:64", "--log-driver=none",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--mount", f"type=bind,src={source},dst=/input/program.py,readonly",
            "--mount", f"type=bind,src={output},dst=/output",
            image,
        ]
        result["command"] = command
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
                process.kill()
                process.wait()
            for thread in threads:
                thread.join()
            result.update(
                returncode=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        finally:
            # Killing the Docker client alone does not stop its container.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        if result["returncode"] != 0 or result["timed_out"]:
            return result, None
        # Refuse symlinks, directories, and special files produced by the program.
        fd = os.open(output / "output.png", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("Output must be a regular file")
            data = handle.read(MAX_IMAGE_BYTES + 1)
        return result, data


def run_episode(program, episodes, *, image="artsy-renderer:0.1.0", timeout=LIMITS["wall_seconds"]):
    if not 0 < timeout <= LIMITS["wall_seconds"]:
        raise ValueError("Timeout must be between 0 and 30 seconds")
    episode_id = uuid.uuid4().hex
    directory = Path(episodes) / episode_id
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "program.py").open("x") as handle:
        handle.write(program)
    record = {
        "schema_version": 1,
        "episode_id": episode_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "hand_written",
        "program": "program.py",
        "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
        "renderer_image": image,
        "limits": {**LIMITS, "wall_seconds": timeout},
        "status": "execution_failed",
        "renderer": None,
        "image": None,
    }
    try:
        inspection = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        record["renderer_image_id"] = inspection.stdout.strip()
        result, data = _render(program, record["renderer_image_id"], timeout)
        record["renderer"] = result
        if data is not None:
            record["status"] = "validation_failed"
            metadata = validate_png(data)
            with (directory / "image.png").open("xb") as handle:
                handle.write(data)
            record.update(status="success", image={"path": "image.png", **metadata})
    except (OSError, ValueError, subprocess.SubprocessError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        record["error"] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            record["error"] += ": " + (error.stderr or "")[-MAX_LOG_BYTES:]
    with (directory / "record.json").open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return directory, record
