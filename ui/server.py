"""Loopback-only UI and drawing job service. Run: python3 -m ui.server."""
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'episodes' / 'ui'
LOCK = threading.Lock()
ACTIVE = None
PORT = int(os.getenv('ARTSY_UI_PORT', '4173'))


def interpreter():
    explicit = os.getenv('ARTSY_EVAL_PYTHON')
    if explicit:
        return explicit
    for name in ('.eval-venv', '.venv'):
        path = ROOT / name / 'bin/python'
        if path.exists():
            return str(path)
    return 'python3'


def run_job(directory, request):
    global ACTIVE
    status = {'state': 'running', **request}
    initial = directory / 'starting.tmp'
    initial.write_text(json.dumps(status))
    initial.replace(directory / 'job.json')
    try:
        with (directory / 'console.txt').open('w') as output:
            process = subprocess.run(
                [interpreter(), '-u', '-m', 'ui.worker', str(directory), request['prompt'],
                 str(request['seed']), request['model']], cwd=ROOT, stdout=output,
                stderr=subprocess.STDOUT, check=False,
            )
        status['state'] = 'completed' if process.returncode == 0 else 'failed'
        status['exit_code'] = process.returncode
    except Exception as error:
        status.update(state='failed', error=str(error))
    finally:
        temp = directory / 'job.tmp'
        temp.write_text(json.dumps(status))
        temp.replace(directory / 'job.json')
        with LOCK:
            ACTIVE = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'ui'), **kwargs)

    def trusted(self):
        return (self.headers.get('Host') in (f'127.0.0.1:{PORT}', f'localhost:{PORT}')
                and self.headers.get('Origin') in (None, f'http://127.0.0.1:{PORT}', f'http://localhost:{PORT}'))

    def send_json(self, value, status=200):
        content = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        global ACTIVE
        if not self.trusted():
            return self.send_json({'error': 'Local requests only.'}, 403)
        if self.path != '/api/runs':
            return self.send_json({'error': 'Not found.'}, 404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 20000:
                raise ValueError('Invalid request size.')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Invalid request.')
            prompt, seed, model = body.get('prompt'), body.get('seed'), body.get('model')
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
                raise ValueError('Enter a prompt of 1–4000 characters.')
            if type(seed) is not int or not 0 <= seed <= 2**53 - 1:
                raise ValueError('Seed must be a nonnegative safe integer.')
            if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_./:-]{1,200}', model):
                raise ValueError('Enter a valid model ID.')
            request = {'prompt': prompt.strip(), 'seed': seed, 'model': model}
        except (ValueError, TypeError) as error:
            return self.send_json({'error': str(error)}, 400)
        with LOCK:
            if ACTIVE:
                return self.send_json({'error': 'A generation is already running.'}, 409)
            run_id = uuid4().hex
            directory = JOBS / run_id
            directory.mkdir(parents=True)
            (directory / 'job.json').write_text(json.dumps({'state': 'queued', **request}))
            ACTIVE = run_id
            threading.Thread(target=run_job, args=(directory, request), daemon=True).start()
        self.send_json({'id': run_id}, 202)

    def do_GET(self):
        if not self.trusted():
            return self.send_json({'error': 'Local requests only.'}, 403)
        match = re.fullmatch(r'/api/runs/([a-f0-9]{32})(?:/(image.png|program.py|trace.json|result.json))?', self.path)
        if match:
            directory = JOBS / match[1]
            if not (directory / 'job.json').exists():
                return self.send_json({'error': 'Run not found.'}, 404)
            if match[2]:
                file = directory / 'output' / match[2]
                if not file.is_file():
                    return self.send_json({'error': 'Artifact not available.'}, 404)
                data = file.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/png' if match[2] == 'image.png' else 'text/plain; charset=utf-8')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            status = json.loads((directory / 'job.json').read_text())
            console = directory / 'console.txt'
            status['console'] = console.read_text(errors='replace') if console.exists() else ''
            for name in ('trace.json', 'result.json'):
                file = directory / 'output' / name
                if file.exists() and status['state'] not in ('queued', 'running'):
                    status[name.split('.')[0]] = json.loads(file.read_text())
            status['artifacts'] = [name for name in ('image.png', 'program.py', 'trace.json', 'result.json') if (directory / 'output' / name).exists()]
            return self.send_json(status)
        if self.path.startswith('/api/'):
            return self.send_json({'error': 'Not found.'}, 404)
        if self.path not in ('/', '/index.html', '/app.js', '/trace.js', '/style.css'):
            return self.send_json({'error': 'Not found.'}, 404)
        super().do_GET()


if __name__ == '__main__':
    print(f'Artsy: http://127.0.0.1:{PORT}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
