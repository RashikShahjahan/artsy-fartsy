"""Consumer drawing API. No internal workspaces, datasets, or training endpoints."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from urllib.parse import urlsplit
from uuid import uuid4

from . import speech

ROOT = Path(__file__).resolve().parents[2]
JOBS = Path(os.getenv('ARTSY_CLIENT_JOBS', str(ROOT / 'episodes/client'))).expanduser().resolve()
PORT = int(os.getenv('ARTSY_CLIENT_API_PORT', '3001'))
MODEL = os.getenv('ARTSY_CLIENT_MODEL', 'artsy/rlhf-experiment10-genai1000-linear-epoch1-kl-budget-recovery')
RUN_ID = re.compile(r'[a-f0-9]{32}')
ACTIVE = {'queued', 'running', 'stopping'}


def content_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def validate_parent(root, ident):
    if not isinstance(ident, str) or not RUN_ID.fullmatch(ident):
        raise ValueError('Choose a completed drawing to edit.')
    directory = root / ident
    try:
        job = json.loads((directory / 'job.json').read_text())
        result = json.loads((directory / 'output/result.json').read_text())
        source, image = directory / 'output/program.py', directory / 'output/image.png'
        paused = result.get('outcome') == 'incomplete' and result.get('turnLimitReached') is True
        if (job.get('state') != 'completed'
                or not (result.get('outcome') == 'submitted' or paused)
                or not source.is_file() or source.stat().st_size > 64 * 1024
                or (not paused and not image.is_file())
                or content_hash(source) != result.get('sourceHash')
                or content_hash(image) != result.get('imageHash')):
            raise ValueError('The original drawing is incomplete or has changed. Start a new drawing.')
    except (OSError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError('That drawing is unavailable. Start a new drawing.') from error
    return directory


def generation_request(body, root):
    prompt = body.get('prompt') if isinstance(body, dict) else None
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError('Enter a prompt of 1–4000 characters.')
    request = {'prompt': prompt.strip(), 'seed': 42, 'model': MODEL,
               'experience': 'artsy', 'max_turns': 10}
    if body.get('parent_id') is not None:
        validate_parent(root, body['parent_id'])
        request['parent_id'] = body['parent_id']
    return request


def terminate(process, force=False):
    try:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        pass


def stop(process, grace_seconds=5):
    if process.poll() is None:
        terminate(process)
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            terminate(process, force=True)
            process.wait()


class DrawingJobs:
    def __init__(self, root):
        self.root = root
        self.lock = threading.RLock()
        self.active = set()
        self.processes = {}

    def status(self, ident):
        with self.lock:
            directory = self.root / ident
            status = json.loads((directory / 'job.json').read_text())
            if status.get('state') in ACTIVE and ident not in self.active:
                status.update(state='interrupted', error='The drawing service restarted. Please try again.')
                write_json(directory / 'job.json', status)
            return status

    def create(self, body):
        request = generation_request(body, self.root)
        with self.lock:
            if self.active:
                raise RuntimeError('A drawing is already running.')
            ident = uuid4().hex
            directory = self.root / ident
            directory.mkdir(parents=True)
            write_json(directory / 'job.json', {'state': 'queued', **request})
            self.active.add(ident)
            threading.Thread(target=self.run, args=(ident, request), daemon=True).start()
            return ident

    def run(self, ident, request):
        directory = self.root / ident
        process = None
        try:
            environment = dict(os.environ)
            environment['ARTSY_MODEL_URL'] = 'http://127.0.0.1:8080/v1'
            environment['ARTSY_MODEL_API_KEY'] = 'local-no-auth'
            environment['ARTSY_CLIENT_JOBS'] = str(self.root)
            with (directory / 'console.txt').open('w') as output:
                with self.lock:
                    if self.status(ident)['state'] == 'stopping':
                        write_json(directory / 'job.json', {'state': 'cancelled', **request})
                        return
                    python = os.getenv('ARTSY_EVAL_PYTHON') or str(
                        ROOT / ('.eval-venv' if (ROOT / '.eval-venv/bin/python').is_file() else '.venv') / 'bin/python')
                    process = subprocess.Popen(
                        [python, '-u', '-m', 'client.backend.worker', str(directory),
                         request['prompt'], str(request['seed']), request['model']],
                        cwd=ROOT, env=environment, stdout=output, stderr=output,
                        start_new_session=(os.name == 'posix'))
                    self.processes[ident] = process
                    write_json(directory / 'job.json', {'state': 'running', 'pid': process.pid, **request})
                code = process.wait()
            with self.lock:
                state = 'cancelled' if self.status(ident)['state'] == 'stopping' else 'completed' if code == 0 else 'failed'
                write_json(directory / 'job.json', {'state': state, 'exit_code': code, **request})
        except Exception as error:
            if process is not None:
                stop(process)
            with self.lock:
                write_json(directory / 'job.json', {'state': 'failed', 'error': str(error), **request})
        finally:
            with self.lock:
                self.active.discard(ident)
                self.processes.pop(ident, None)

    def cancel(self, ident):
        with self.lock:
            if ident not in self.active:
                raise RuntimeError('This drawing is no longer running.')
            status = self.status(ident)
            status['state'] = 'stopping'
            write_json(self.root / ident / 'job.json', status)
            if process := self.processes.get(ident):
                terminate(process)
                threading.Thread(target=stop, args=(process,), daemon=True).start()

    def close(self):
        with self.lock:
            for ident in self.active:
                status = self.status(ident)
                status['state'] = 'stopping'
                write_json(self.root / ident / 'job.json', status)
            processes = list(self.processes.values())
        for process in processes:
            stop(process)


class Handler(BaseHTTPRequestHandler):
    def respond(self, data, status=200, content_type='application/json'):
        if content_type == 'application/json':
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def trusted(self):
        host = self.headers.get('Host')
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        origin = self.headers.get('Origin')
        return host in allowed and (not origin or origin in {f'http://{value}' for value in allowed})

    def body(self):
        size = int(self.headers.get('Content-Length', '0'))
        if not 0 < size <= 20000:
            raise ValueError('Invalid request size.')
        return json.loads(self.rfile.read(size))

    def do_GET(self):
        if not self.trusted():
            return self.respond({'error': 'Local requests only.'}, 403)
        path = urlsplit(self.path).path
        if path == '/health':
            return self.respond({'service': 'artsy-client', 'ready': True})
        match = re.fullmatch(r'/api/runs/([a-f0-9]{32})(?:/(image\.png|preview-\d+-\d+\.png))?', path)
        if not match:
            return self.respond({'error': 'Not found.'}, 404)
        ident, artifact = match.groups()
        directory = self.server.jobs.root / ident
        try:
            status = self.server.jobs.status(ident)
            if artifact:
                return self.respond((directory / 'output' / artifact).read_bytes(), content_type='image/png')
            output = directory / 'output'
            status['artifacts'] = [name for name in ('image.png', 'program.py') if (output / name).is_file()]
            result = output / 'result.json'
            if result.is_file() and status['state'] not in ACTIVE:
                status['result'] = json.loads(result.read_text())
            progress = directory / 'progress.json'
            if progress.is_file():
                status['progress'] = json.loads(progress.read_text())
            return self.respond(status)
        except FileNotFoundError:
            return self.respond({'error': 'Drawing or image not found.'}, 404)
        except (OSError, ValueError):
            return self.respond({'error': 'Couldn’t read this drawing. Try again.'}, 503)

    def do_POST(self):
        if not self.trusted():
            return self.respond({'error': 'Local requests only.'}, 403)
        path = urlsplit(self.path).path
        try:
            if path == '/api/artsy/runs':
                return self.respond({'id': self.server.jobs.create(self.body())}, 202)
            if path == '/api/artsy/speech':
                text = speech.validate_text(self.body())
                return self.respond(speech.synthesize(text), content_type='audio/wav')
            if match := re.fullmatch(r'/api/runs/([a-f0-9]{32})/cancel', path):
                self.server.jobs.cancel(match[1])
                return self.respond({'id': match[1], 'state': 'stopping'})
            return self.respond({'error': 'Not found.'}, 404)
        except (json.JSONDecodeError, TypeError):
            return self.respond({'error': 'Enter a valid drawing or speech request.'}, 400)
        except ValueError as error:
            return self.respond({'error': str(error)}, 400)
        except RuntimeError as error:
            return self.respond({'error': str(error)}, 409)
        except OSError:
            return self.respond({'error': 'Drawing service unavailable. Please try again.'}, 503)


def main():
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    server.jobs = DrawingJobs(JOBS)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    print(f'Artsy consumer API: http://127.0.0.1:{PORT}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.jobs.close()
        server.server_close()


if __name__ == '__main__':
    main()
