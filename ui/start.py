"""Start the UI and its native Tinker bridge; Ctrl-C stops owned services."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
HEALTH_URL = 'http://127.0.0.1:8080/health'


def bridge_ready():
    try:
        with urlopen(HEALTH_URL, timeout=1) as response:
            health = json.loads(response.read(1024 * 1024))
        return (isinstance(health, dict) and health.get('ready') is True
                and health.get('backend') == 'tinker')
    except (OSError, ValueError):
        return False


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main(args=None):
    children = []
    environment = dict(os.environ)
    environment.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        bridge = None
        if bridge_ready():
            print('Reusing native Tinker bridge on port 8080.', flush=True)
        else:
            print('Starting native Tinker bridge on port 8080…', flush=True)
            bridge = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'inference.server:app',
                 '--host', '127.0.0.1', '--port', '8080'],
                cwd=ROOT, env=environment, start_new_session=True)
            children.append(bridge)
            deadline = time.monotonic() + 180
            while not bridge_ready():
                if bridge.poll() is not None:
                    raise RuntimeError('Native bridge exited during startup. See its output above.')
                if time.monotonic() >= deadline:
                    raise RuntimeError('Native bridge did not become ready within 180 seconds.')
                time.sleep(0.25)
            if bridge.poll() is not None:
                raise RuntimeError('Native bridge exited during startup. See its output above.')

        ui = subprocess.Popen(
            [sys.executable, '-m', 'ui.server', *(sys.argv[1:] if args is None else args)],
            cwd=ROOT, env=environment, start_new_session=True)
        children.append(ui)
        while ui.poll() is None:
            if bridge is not None and bridge.poll() is not None:
                raise RuntimeError('Native bridge stopped. Stopping the UI.')
            time.sleep(0.25)
        return ui.returncode
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        for process in reversed(children):
            stop(process)
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    sys.exit(main())
