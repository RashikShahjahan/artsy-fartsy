"""Fresh, bounded rendering used inside the Mini SWE sandbox.

Usage: python render.py --seed 42 --output /tmp/submission < program.py
Only the scoring process calls this; previews can run `python program.py`.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from PIL import Image

SOURCE_BYTES = 64 * 1024
PNG_BYTES = 8 * 1024 * 1024


def render(source: str, seed: int, output: Path, runtime: Path, seconds: int = 10) -> dict:
    if not source.strip() or len(source.encode()) > SOURCE_BYTES:
        return {"ok": False, "error": "Expected 1–65536 bytes of Python source"}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="artsy-render-") as directory:
        work = Path(directory)
        (work / "program.py").write_text(source, encoding="utf-8")
        # File-backed output avoids unbounded captured logs in memory. Limit each
        # file the child writes, including its PNG and logs, to 8 MiB.
        runner = (
            "import random, resource, runpy; "
            f"resource.setrlimit(resource.RLIMIT_FSIZE, ({PNG_BYTES}, {PNG_BYTES})); "
            f"random.seed({seed}); runpy.run_path('program.py', run_name='__main__')"
        )
        env = {**os.environ, "PYTHONPATH": str(runtime.resolve()),
               "PYTHONHASHSEED": str(seed % 2**32), "PYTHONDONTWRITEBYTECODE": "1"}
        if sys.platform == "darwin":
            env.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib:/usr/local/lib:/usr/lib")
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen([sys.executable, "-c", runner], cwd=work,
                                       env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            timed_out = False
            try:
                process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            stdout.seek(0)
            stderr.seek(0)
            feedback = {"ok": False, "exitCode": process.returncode,
                        "stdout": stdout.read(8192).decode(errors="replace"),
                        "stderr": stderr.read(8192).decode(errors="replace")}
        if timed_out or process.returncode:
            return {**feedback, "error": "Rendering timed out" if timed_out else "Python execution failed"}
        try:
            fd = os.open(work / "output.png", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as file:
                import stat
                info = os.fstat(file.fileno())
                if not stat.S_ISREG(info.st_mode) or not 24 <= info.st_size <= PNG_BYTES:
                    raise ValueError("Expected a regular PNG of at most 8 MiB")
                png = file.read(PNG_BYTES + 1)
            import io
            with Image.open(io.BytesIO(png)) as image:
                if image.format != "PNG" or image.size != (1920, 1200):
                    raise ValueError("Expected a 1920×1200 PNG")
                image.verify()
            (output / "image.png").write_bytes(png)
            return {**feedback, "ok": True, "width": 1920, "height": 1200}
        except (OSError, ValueError, SyntaxError) as error:
            return {**feedback, "error": str(error)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Return the rendered PNG to Mini SWE as an image observation")
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**53 - 1:
        parser.error("seed must be a nonnegative safe integer")
    source = sys.stdin.buffer.read(SOURCE_BYTES + 1).decode("utf-8")
    result = render(source, args.seed, args.output, Path(__file__).parent / "runtime")
    print(json.dumps(result))
    if args.preview and result["ok"]:
        encoded = base64.b64encode((args.output / "image.png").read_bytes()).decode()
        print("<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>"
              f"data:image/png;base64,{encoded}</MSWEA_MULTIMODAL_CONTENT>")
