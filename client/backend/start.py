"""Launch the consumer API, inference bridge, and Next.js as one application."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

from .server import stop as stop_process

ROOT = Path(__file__).resolve().parents[2]


def stop(process):
    # Give the API time to stop its independently supervised drawing worker.
    stop_process(process, grace_seconds=15)


def health(url):
    try:
        with urlopen(url, timeout=1) as response:
            value = json.loads(response.read(1024 * 1024))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def bridge_ready():
    value = health('http://127.0.0.1:8080/health')
    return value.get('ready') is True and value.get('backend') == 'tinker'


def wait_ready(process, ready, label, seconds):
    deadline = time.monotonic() + seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'{label} exited during startup. See its output above.')
        if ready():
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'{label} did not become ready within {seconds} seconds.')
        time.sleep(0.25)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-inference', action='store_true', help='Browse without starting the inference bridge')
    parser.add_argument('--production', action='store_true', help='Serve a previously built Next.js client')
    options = parser.parse_args(args)
    children = []
    environment = dict(os.environ)
    environment.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    api_port = int(environment.get('ARTSY_CLIENT_API_PORT', '3001'))
    port = int(environment.get('ARTSY_CLIENT_PORT', '3000'))
    api_url = f'http://127.0.0.1:{api_port}'
    environment['ARTSY_CLIENT_API_URL'] = api_url
    environment['ARTSY_CLIENT_API_PORT'] = str(api_port)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    def spawn(command):
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   start_new_session=(os.name == 'posix'))
        children.append(process)
        return process

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        if not (ROOT / 'client/node_modules/next/package.json').is_file():
            print('Installing client dependencies…', flush=True)
            install = spawn(['npm', 'ci', '--prefix', 'client'])
            if install.wait() != 0:
                raise RuntimeError('Client dependency installation failed.')
        if health(api_url + '/health'):
            raise RuntimeError(f'Port {api_port} already has a service. Stop it or set ARTSY_CLIENT_API_PORT.')
        api = spawn([sys.executable, '-m', 'client.backend.server'])
        wait_ready(api, lambda: health(api_url + '/health').get('service') == 'artsy-client', 'Consumer API', 15)
        bridge = None
        if not options.no_inference:
            if bridge_ready():
                print('Reusing the native inference bridge on port 8080.', flush=True)
            else:
                print('Starting the native inference bridge…', flush=True)
                bridge = spawn([sys.executable, '-m', 'uvicorn', 'inference.server:app',
                                '--host', '127.0.0.1', '--port', '8080'])
                wait_ready(bridge, bridge_ready, 'Inference bridge', 180)
        frontend = spawn(['npm', '--prefix', 'client', 'run', 'start' if options.production else 'dev',
                          '--', '--port', str(port)])
        print(f'Artsy: http://127.0.0.1:{port} — Ctrl-C stops the app.', flush=True)
        while frontend.poll() is None:
            if api.poll() is not None:
                raise RuntimeError('Consumer API stopped. Stopping the app.')
            if bridge is not None and bridge.poll() is not None:
                raise RuntimeError('Inference bridge stopped. Stopping the app.')
            time.sleep(0.25)
        return frontend.returncode
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        for process in reversed(children):
            stop(process)
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    sys.exit(main())
