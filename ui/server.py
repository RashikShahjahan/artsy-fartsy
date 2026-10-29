"""Loopback-only UI and drawing job service. Run: python3 -m ui.server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4
from urllib.parse import urlsplit, parse_qs
from ui.shell import navigation, review_page, empty_page
from urllib.request import urlopen

from drawing.config import MODEL_ID
from drawing.dataset_client import OPENROUTER_MODEL_URL
from evaluation.suite import dataset_path

DIRECT_MODELS = [
    {"id": "openai/gpt-6-luna", "label": "GPT-6 Luna", "direct": True},
]

def model_catalog():
    return [{"id": MODEL_ID, "label": "Qwen3.5 · 4B", "model_path": None},
            *[dict(model) for model in DIRECT_MODELS]]

from drawing.dataset_specs import validate_specs

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'episodes' / 'ui'
LOCK = threading.Lock()
ACTIVE = None
PORT = int(os.getenv('ARTSY_UI_PORT', '4173'))
DRAWING_DATASET = os.getenv('ARTSY_DRAWING_DATASET')
CANDIDATE_DATASET = os.getenv('ARTSY_CANDIDATE_DATASET')
PROMPT_REVIEW_OUTPUT = None
CANDIDATE_REVIEW_OUTPUT = None
ACCEPTED_OUTPUT = None
REVIEW_LOCK = threading.RLock()
MOUNTED_HANDLERS = {}


def drawing_dataset():
    path = Path(DRAWING_DATASET).resolve() if DRAWING_DATASET else dataset_path()
    raw = path.read_bytes()
    rows = validate_specs([json.loads(line) for line in raw.splitlines() if line.strip()])
    drawings = [row for row in rows if row.get('kind') != 'refusal'
                and row.get('benchmarkSlice', 'drawing') == 'drawing']
    return {'path': str(path), 'label': path.parent.name, 'sha256': hashlib.sha256(raw).hexdigest(),
            'items': drawings}


def drawing_runs(dataset):
    result = []
    for file in sorted(JOBS.glob('*/job.json'), key=lambda file: file.stat().st_mtime, reverse=True):
        try:
            run = json.loads(file.read_text())
            if run.get('dataset_sha256') == dataset['sha256']:
                result.append({'id': file.parent.name, **{key: run.get(key) for key in
                               ('prompt_id', 'model', 'state', 'seed')}})
        except (OSError, ValueError, AttributeError):
            continue
    return result


def generation_request(body):
    if not isinstance(body, dict):
        raise ValueError('Invalid request.')
    prompt, seed, model = body.get('prompt'), body.get('seed'), body.get('model')
    origin = {}
    if 'prompt_id' in body:
        dataset = drawing_dataset()
        if body.get('dataset_sha256') != dataset['sha256']:
            raise ValueError('The prompt dataset changed. Reload the drawing view before running.')
        spec = next((row for row in dataset['items'] if row['id'] == body['prompt_id']), None)
        if not spec:
            raise ValueError('Drawing prompt not found in this dataset.')
        prompt, seed = spec['prompt'], spec['seed']
        if model not in {item['id'] for item in model_catalog()}:
            raise ValueError('Choose an available Studio model.')
        origin = {'prompt_id': spec['id'], 'dataset_sha256': dataset['sha256'],
                  'dataset_path': dataset['path']}
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError('Enter a prompt of 1–4000 characters.')
    if type(seed) is not int or not 0 <= seed <= 2**53 - 1:
        raise ValueError('Seed must be a nonnegative safe integer.')
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_./:-]{1,200}', model):
        raise ValueError('Enter a valid model ID.')
    if model not in {item['id'] for item in model_catalog()}:
        raise ValueError('Choose an available Studio model.')
    return {'prompt': prompt.strip(), 'seed': seed, 'model': model, **origin}


def bridge_model_catalog():
    try:
        with urlopen('http://127.0.0.1:8080/health', timeout=1) as response:
            status = json.loads(response.read(1024 * 1024))
        if not isinstance(status, dict):
            return None
        return [item for item in status.get('models', []) if isinstance(item, dict) and isinstance(item.get('id'), str)] or [{'id': status.get('model'), 'model_path': status.get('model_path')}]
    except (OSError, ValueError):
        return None


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
            environment = os.environ.copy()
            direct = request['model'] in {model['id'] for model in DIRECT_MODELS}
            environment['ARTSY_MODEL_URL'] = OPENROUTER_MODEL_URL if direct else 'http://127.0.0.1:8080/v1'
            if direct:
                environment.pop('ARTSY_MODEL_API_KEY', None)
            else:
                environment['ARTSY_MODEL_API_KEY'] = 'local-no-auth'
            process = subprocess.run(
                [interpreter(), '-u', '-m', 'ui.worker', str(directory), request['prompt'],
                 str(request['seed']), request['model']], cwd=ROOT, stdout=output,
                stderr=subprocess.STDOUT, check=False, env=environment,
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

    def serve_workspace(self):
        target = urlsplit(self.path)
        mount = next((name for name in ('review', 'candidates')
                      if target.path == '/' + name or target.path.startswith('/' + name + '/')), None)
        if mount is None:
            return False
        try:
            with REVIEW_LOCK:
                safety_only = mount == 'review' and parse_qs(target.query).get('scope') == ['safety']
                key = (mount, safety_only, DRAWING_DATASET, CANDIDATE_DATASET, PROMPT_REVIEW_OUTPUT,
                       CANDIDATE_REVIEW_OUTPUT, ACCEPTED_OUTPUT)
                if key not in MOUNTED_HANDLERS:
                    def transform(page):
                        rendered = review_page(page, mount)
                        if mount == 'review':
                            switch = '<div class="review-scope" aria-label="Review scope">'
                            for url, label, selected in [('/review', 'All prompts', not safety_only),
                                    ('/review?scope=safety', 'Safety prompts', safety_only)]:
                                switch += f'<a href="{url}"' + (' aria-current="true"' if selected else '') + f'>{label}</a>'
                            rendered = rendered.replace('<h1>', switch + '</div><h1>', 1)
                        return rendered
                    if mount == 'review':
                        from ui.prompt_review import make_handler
                        handler_type = make_handler(Path(DRAWING_DATASET) if DRAWING_DATASET else None,
                            PROMPT_REVIEW_OUTPUT, not safety_only, PORT, base_path='/review', page_transform=transform)
                    elif CANDIDATE_DATASET:
                        from ui.candidate_review import make_handler
                        handler_type = make_handler(Path(CANDIDATE_DATASET), PORT,
                            CANDIDATE_REVIEW_OUTPUT, ACCEPTED_OUTPUT,
                            base_path='/candidates', page_transform=transform)
                    else:
                        handler_type = None
                    MOUNTED_HANDLERS[key] = handler_type
                handler_type = MOUNTED_HANDLERS[key]
                if handler_type is None:
                    if self.command == 'GET' and target.path in ('/candidates', '/candidates/'):
                        self.send_page(empty_page(mount, 'SFT candidates',
                            'To inspect and curate generated trajectories, start this app with '
                            '--candidates datasets/YOUR-RUN/candidates.jsonl.'))
                    else:
                        self.send_json({'error': 'No candidate dataset configured.'}, 404)
                    return True
                handler = object.__new__(handler_type)
                handler.__dict__.update(self.__dict__)
                handler.path = target.path[len('/' + mount):] or '/'
                if target.query and mount != 'review':
                    handler.path += '?' + target.query
                getattr(handler, 'do_' + self.command)()
        except (OSError, ValueError, TypeError) as error:
            if self.command == 'GET' and target.path in ('/' + mount, '/' + mount + '/'):
                self.send_page(empty_page(mount, 'Workspace unavailable', str(error)))
            else:
                self.send_json({'error': str(error)}, 400)
        return True

    def send_page(self, page):
        content = page.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        global ACTIVE
        if not self.trusted():
            return self.send_json({'error': 'Local requests only.'}, 403)
        if self.serve_workspace():
            return
        if self.path != '/api/runs':
            return self.send_json({'error': 'Not found.'}, 404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 20000:
                raise ValueError('Invalid request size.')
            body = json.loads(self.rfile.read(size))
            request = generation_request(body)
        except (OSError, ValueError, TypeError) as error:
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
        if self.serve_workspace():
            return
        if self.path == '/api/drawing-prompts':
            try:
                dataset = drawing_dataset()
                return self.send_json({**dataset, 'runs': drawing_runs(dataset)})
            except (OSError, ValueError, TypeError) as error:
                return self.send_json({'error': str(error)}, 400)
        if self.path == '/api/models':
            served = bridge_model_catalog()
            catalog = {item['id']: item for item in model_catalog()}
            for item in served or []:
                if item['id'] in catalog:
                    catalog[item['id']].update(item)
                    if item.get('model_path') and 'label' not in item:
                        catalog[item['id']]['label'] = 'Qwen3.5 · 4B · configured checkpoint'
            return self.send_json({'models': list(catalog.values()), 'bridge_models': [item['id'] for item in served] if served is not None else None})
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
        if urlsplit(self.path).path in ('/', '/index.html', '/drawings', '/drawings.html'):
            page = (ROOT / 'ui/index.html').read_text().replace('__NAVIGATION__', navigation('studio'))
            return self.send_page(page)
        if self.path not in ('/app.js', '/trace.js', '/style.css'):
            return self.send_json({'error': 'Not found.'}, 404)
        super().do_GET()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, help='Prompt JSONL for Studio and prompt review (defaults to the frozen benchmark)')
    parser.add_argument('--candidates', type=Path, help='Candidate JSONL for SFT inspection and export')
    parser.add_argument('--review-output', type=Path, help='Prompt review output JSON')
    parser.add_argument('--candidate-review', type=Path, help='Candidate review output JSON')
    parser.add_argument('--accepted-output', type=Path, help='Approved SFT export JSONL')
    args = parser.parse_args()
    CANDIDATE_DATASET = str(args.candidates.resolve()) if args.candidates else CANDIDATE_DATASET
    PROMPT_REVIEW_OUTPUT = args.review_output
    CANDIDATE_REVIEW_OUTPUT = args.candidate_review
    ACCEPTED_OUTPUT = args.accepted_output
    if args.dataset:
        DRAWING_DATASET = str(args.dataset.resolve())
    print(f'Artsy: http://127.0.0.1:{PORT}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
