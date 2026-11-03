"""Loopback-only UI and drawing job service. Run: python3 -m ui.server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4
from urllib.parse import urlsplit, parse_qs
from ui.shell import navigation, review_page, empty_page
from ui import rl_monitor, launch, preferences, speech
from ui.supervisor import SUPERVISOR, alive
from urllib.request import urlopen

from drawing.config import MODEL_ID
from inference.model_catalog import LARGE_MODEL_ID, QWEN36_MODEL_ID
from drawing.dataset_client import OPENROUTER_MODEL_URL
from evaluation.suite import dataset_path

DIRECT_MODELS = [
    {"id": "openai/gpt-6-luna", "label": "GPT-6 Luna", "direct": True},
]
ARTSY_MODELS = [
    {"id": LARGE_MODEL_ID, "label": "Base"},
    {"id": "artsy/sft-v2-qwen3.8-27b-final-preview-masked-cosine", "label": "SFT"},
    {"id": "artsy/grpo-v4-subset200-b16-c64", "label": "GRPO"},
    *[dict(model) for model in DIRECT_MODELS],
]

def model_catalog(bridge_models=None):
    """Studio's default models plus saved fine-tunes advertised by the bridge."""
    entries = [{"id": MODEL_ID, "label": "Qwen3.5 · 4B", "model_path": None},
               {"id": QWEN36_MODEL_ID, "label": "Qwen3.6 · 35B-A3B", "model_path": None},
               {"id": LARGE_MODEL_ID, "label": "Qwen3.8 · 27B", "model_path": None},
               *[dict(model) for model in DIRECT_MODELS]]
    known = {entry["id"] for entry in entries}
    for entry in bridge_models or []:
        model_id = entry.get("id")
        if (not isinstance(model_id, str)
                or not re.fullmatch(r"artsy/[A-Za-z0-9_.-]{1,150}", model_id)
                or model_id in known
                or not isinstance(entry.get("model_path"), str)
                or not entry["model_path"].startswith("tinker://")
                or entry.get("base_model") not in
                    ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.8-27B")):
            continue
        entries.append({"id": model_id, "label": str(entry.get("label") or model_id)[:200],
                        "model_path": entry["model_path"], "base_model": entry["base_model"]})
        known.add(model_id)
    return entries

from drawing.dataset_specs import validate_specs

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'episodes' / 'ui'
RL_ROOT = Path(os.getenv('ARTSY_RL_ROOT', str(ROOT / 'checkpoints'))).expanduser().resolve()
RL_DATASET = Path(os.environ['ARTSY_RL_DATASET']).expanduser().resolve() if os.getenv('ARTSY_RL_DATASET') else None
LOCK = threading.RLock()
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


def drawing_status(directory):
    with LOCK:
        file = directory / 'job.json'
        status = json.loads(file.read_text())
        if status.get('state') in ('queued', 'running', 'stopping') and not SUPERVISOR.active(directory.name):
            status.update(state='interrupted', error='The UI server lost ownership of this process.')
            temporary = directory / 'job.reconcile.tmp'
            temporary.write_text(json.dumps(status))
            temporary.replace(file)
        return status


def drawing_busy():
    for file in JOBS.glob('*/job.json'):
        try:
            status = drawing_status(file.parent)
            if (SUPERVISOR.active(file.parent.name) or
                    status.get('state') == 'interrupted' and alive(status.get('pid'))):
                return True
        except (OSError, ValueError, TypeError):
            continue
    return False


def drawing_runs(dataset):
    result = []
    for file in sorted(JOBS.glob('*/job.json'), key=lambda file: file.stat().st_mtime, reverse=True):
        try:
            run = drawing_status(file.parent)
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
    available_models = {item['id'] for item in model_catalog()}
    if isinstance(model, str) and model.startswith("artsy/"):
        available_models.update(item['id'] for item in model_catalog(bridge_model_catalog()))
    if 'prompt_id' in body:
        dataset = drawing_dataset()
        if body.get('dataset_sha256') != dataset['sha256']:
            raise ValueError('The prompt dataset changed. Reload the drawing view before running.')
        spec = next((row for row in dataset['items'] if row['id'] == body['prompt_id']), None)
        if not spec:
            raise ValueError('Drawing prompt not found in this dataset.')
        prompt, seed = spec['prompt'], spec['seed']
        if model not in available_models:
            raise ValueError('Choose an available Studio model.')
        origin = {'prompt_id': spec['id'], 'dataset_sha256': dataset['sha256'],
                  'dataset_path': dataset['path']}
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError('Enter a prompt of 1–4000 characters.')
    if type(seed) is not int or not 0 <= seed <= 2**53 - 1:
        raise ValueError('Seed must be a nonnegative safe integer.')
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_./:-]{1,200}', model):
        raise ValueError('Enter a valid model ID.')
    if model not in available_models:
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


def artsy_request(body):
    """Consumer requests select one of four models with ten turns per drawing/edit."""
    if not isinstance(body, dict):
        raise ValueError('Invalid request.')
    mode = body.get('execution_mode', 'auto')
    if mode not in ('auto', 'approval'):
        raise ValueError('Choose Auto or Approve steps.')
    model = body.get('model', DIRECT_MODELS[0]['id'])
    choice = next((entry for entry in ARTSY_MODELS if entry['id'] == model), None)
    if choice is None:
        raise ValueError('Choose Base, SFT, GRPO, or GPT-6 Luna.')
    try:
        request = generation_request({'prompt': body.get('prompt'), 'seed': 42, 'model': model})
    except ValueError as error:
        if str(error) == 'Choose an available Studio model.':
            raise ValueError(f"{choice['label']} is unavailable. Start the native bridge with its completed checkpoint, then try again.") from error
        raise
    request.update(experience='artsy', max_turns=10, execution_mode=mode)
    parent_id = body.get('parent_id')
    if parent_id is not None:
        if not isinstance(parent_id, str) or not re.fullmatch(r'[a-f0-9]{32}', parent_id):
            raise ValueError('Choose a completed drawing to edit.')
        try:
            directory = JOBS / parent_id
            parent = json.loads((directory / 'job.json').read_text())
            result = json.loads((directory / 'output/result.json').read_text())
            source = directory / 'output/program.py'
            image = directory / 'output/image.png'
            if (parent.get('state') != 'completed' or result.get('outcome') != 'submitted'
                    or not source.is_file() or not image.is_file() or source.stat().st_size > 64 * 1024
                    or content_hash(source) != result.get('sourceHash')
                    or content_hash(image) != result.get('imageHash')):
                raise ValueError('The original drawing is incomplete or has changed. Start a new drawing.')
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            raise ValueError('That drawing is unavailable. Start a new drawing.') from error
        request['parent_id'] = parent_id
    return request


def content_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None


def approve_action(run_id, body):
    if not isinstance(body, dict) or not isinstance(body.get('step_id'), str):
        raise ValueError('Choose the action awaiting approval.')
    directory = JOBS / run_id
    job = json.loads((directory / 'job.json').read_text())
    progress = json.loads((directory / 'progress.json').read_text())
    if (job.get('execution_mode') != 'approval' or job.get('state') != 'running'
            or progress.get('phase') != 'awaiting_approval'
            or not any(step.get('id') == body['step_id'] and step.get('status') == 'pending'
                       for step in progress.get('steps', []))):
        raise ValueError('This action is no longer awaiting approval.')
    temporary = directory / 'approval.tmp'
    temporary.write_text(json.dumps({'step_id': body['step_id']}))
    temporary.replace(directory / 'approval.json')
    return {'approved': body['step_id']}


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
    status = {'state': 'queued', **request}
    process = None
    try:
        with (directory / 'console.txt').open('w') as output:
            environment = os.environ.copy()
            direct = request['model'] in {model['id'] for model in DIRECT_MODELS}
            environment['ARTSY_MODEL_URL'] = OPENROUTER_MODEL_URL if direct else 'http://127.0.0.1:8080/v1'
            if direct:
                environment.pop('ARTSY_MODEL_API_KEY', None)
            else:
                environment['ARTSY_MODEL_API_KEY'] = 'local-no-auth'
            with LOCK:
                process = SUPERVISOR.spawn(
                    directory.name, [interpreter(), '-u', '-m', 'ui.worker', str(directory), request['prompt'],
                                     str(request['seed']), request['model']],
                    cwd=ROOT, stdout=output, stderr=output, env=environment)
                status.update(state='running', pid=process.pid)
                temporary = directory / 'starting.tmp'
                temporary.write_text(json.dumps(status))
                temporary.replace(directory / 'job.json')
            code = process.wait()
        status['exit_code'] = code
    except Exception as error:
        if process is not None and process.poll() is None:
            SUPERVISOR.cancel(directory.name)
            process.wait()
        status.update(state='failed', error=str(error))
    finally:
        with LOCK:
            try:
                if status.get('state') != 'failed':
                    current = json.loads((directory / 'job.json').read_text())
                    status['state'] = ('cancelled' if current.get('state') == 'stopping' else
                                       'completed' if status.get('exit_code') == 0 else 'failed')
                temporary = directory / 'job.tmp'
                temporary.write_text(json.dumps(status))
                temporary.replace(directory / 'job.json')
            finally:
                SUPERVISOR.release(directory.name)


PREFERENCE_REVIEW = None


def preference_review():
    global PREFERENCE_REVIEW
    with LOCK:
        if PREFERENCE_REVIEW is None:
            PREFERENCE_REVIEW = preferences.Review(ROOT)
        return PREFERENCE_REVIEW


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
                source = CANDIDATE_DATASET
                base = '/' + mount
                route = target.path[len(base):] or '/'
                manual_path = CANDIDATE_REVIEW_OUTPUT or (
                    Path(source).parent / 'manual-review.json' if source else None)
                key = (mount, safety_only, DRAWING_DATASET, source, PROMPT_REVIEW_OUTPUT,
                       CANDIDATE_REVIEW_OUTPUT, ACCEPTED_OUTPUT,
                       content_hash(source), content_hash(manual_path))
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
                    elif source:
                        from ui.candidate_review import make_handler
                        handler_type = make_handler(Path(source), PORT, CANDIDATE_REVIEW_OUTPUT,
                            ACCEPTED_OUTPUT, base_path=base, page_transform=transform)
                    else:
                        handler_type = None
                    for old in list(MOUNTED_HANDLERS):
                        if old[0] == mount:
                            del MOUNTED_HANDLERS[old]
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
                handler.path = route
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
        if not self.trusted():
            return self.send_json({'error': 'Local requests only.'}, 403)
        if self.serve_workspace():
            return
        approve = re.fullmatch(r'/api/runs/([a-f0-9]{32})/approve', self.path)
        if approve:
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Invalid approval request size.')
                with LOCK:
                    decision = approve_action(approve[1], json.loads(self.rfile.read(size)))
                return self.send_json(decision)
            except (OSError, ValueError, TypeError) as error:
                return self.send_json({'error': str(error)}, 409)
        if self.path == '/api/artsy/speech':
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Invalid speech request size.')
                text = speech.validate_text(json.loads(self.rfile.read(size)))
                data = speech.synthesize(text)
            except (ValueError, TypeError) as error:
                return self.send_json({'error': str(error)}, 400)
            except RuntimeError as error:
                return self.send_json({'error': str(error)}, 503)
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path in ('/api/preferences/rank', '/api/preferences/reveal'):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('Invalid request size.')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Expected an object.')
                review = preference_review()
                result = (review.results(body.get('reviewer')) if self.path.endswith('/reveal') else
                          review.save(body.get('reviewer'), body.get('case'), body.get('ranks')))
                return self.send_json(result)
            except (ValueError, TypeError) as error:
                return self.send_json({'error': str(error)}, 400)
        if self.path == '/api/launches':
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 20000:
                    raise ValueError('Invalid request size.')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Invalid launch request.')
                result = launch.start(body.get('kind'), body.get('settings'))
            except (OSError, ValueError, TypeError) as error:
                return self.send_json({'error': str(error)}, 400)
            except RuntimeError as error:
                return self.send_json({'error': str(error)}, 409)
            return self.send_json(result, 202)
        cancel = re.fullmatch(r'/api/launches/([a-f0-9]{32})/cancel', self.path)
        if cancel:
            try:
                return self.send_json(launch.cancel(cancel[1]))
            except (ValueError, OSError) as error:
                return self.send_json({'error': str(error)}, 409)
        drawing_cancel = re.fullmatch(r'/api/runs/([a-f0-9]{32})/cancel', self.path)
        if drawing_cancel:
            with LOCK:
                directory = JOBS / drawing_cancel[1]
                try:
                    status = drawing_status(directory)
                    if status.get('state') != 'running':
                        raise ValueError('This run is no longer controlled by the current UI server.')
                    SUPERVISOR.cancel(drawing_cancel[1])
                    status['state'] = 'stopping'
                    temp = directory / 'job.cancel.tmp'
                    temp.write_text(json.dumps(status))
                    temp.replace(directory / 'job.json')
                except (ValueError, OSError) as error:
                    return self.send_json({'error': str(error)}, 409)
            return self.send_json({'id': drawing_cancel[1], 'state': 'stopping'})
        if self.path not in ('/api/runs', '/api/artsy/runs'):
            return self.send_json({'error': 'Not found.'}, 404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 20000:
                raise ValueError('Invalid request size.')
            body = json.loads(self.rfile.read(size))
            request = artsy_request(body) if self.path == '/api/artsy/runs' else generation_request(body)
        except (OSError, ValueError, TypeError) as error:
            return self.send_json({'error': str(error)}, 400)
        with LOCK:
            if drawing_busy():
                return self.send_json({'error': 'A generation is already running.'}, 409)
            run_id = uuid4().hex
            directory = JOBS / run_id
            directory.mkdir(parents=True)
            (directory / 'job.json').write_text(json.dumps({'state': 'queued', **request}))
            SUPERVISOR.reserve(run_id)
            threading.Thread(target=run_job, args=(directory, request), daemon=True).start()
        self.send_json({'id': run_id}, 202)

    def do_GET(self):
        if not self.trusted():
            return self.send_json({'error': 'Local requests only.'}, 403)
        if self.serve_workspace():
            return
        target = urlsplit(self.path).path
        if target in ('/preferences', '/preferences/'):
            return self.send_page((ROOT/'ui/preferences.html').read_text().replace(
                '__NAVIGATION__', navigation('preferences')))
        if target == '/api/preferences' or target.startswith('/api/preferences/image/'):
            try:
                name = parse_qs(urlsplit(self.path).query).get('reviewer', [''])[0]
                review = preference_review()
                if target == '/api/preferences':
                    return self.send_json(review.packet(name))
                match = re.fullmatch(r'/api/preferences/image/([a-f0-9]{24})/([012])', target)
                if not match:
                    raise ValueError('Unknown image.')
                data = review.image(name, match[1], int(match[2]))
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except (ValueError, TypeError) as error:
                return self.send_json({'error': str(error)}, 400)
        if target == '/api/launches/models':
            return self.send_json(launch.model_choices(bridge_model_catalog()))
        if target == '/api/launches':
            return self.send_json({'runs': launch.list_runs()})
        launched = re.fullmatch(r'/api/launches/([a-f0-9]{32})', target)
        if launched:
            try:
                return self.send_json(launch.get_run(launched[1]))
            except (ValueError, OSError):
                return self.send_json({'error': 'Run not found.'}, 404)
        if target == '/api/rl/runs':
            return self.send_json({'runs': rl_monitor.runs(RL_ROOT)})
        match = re.fullmatch(r'/api/rl/runs/([^/]+)/prompts', target)
        if match:
            try:
                return self.send_json(rl_monitor.prompts(RL_ROOT, match[1], RL_DATASET))
            except (ValueError, OSError):
                return self.send_json({'error': 'RL run unavailable.'}, 404)
        match = re.fullmatch(r'/api/rl/runs/([^/]+)', target)
        if match:
            try:
                return self.send_json(rl_monitor.episodes(RL_ROOT, match[1]))
            except (ValueError, OSError):
                return self.send_json({'error': 'RL run unavailable.'}, 404)
        match = re.fullmatch(
            r'/api/rl/artifacts/([^/]+)/([^/]+)/([^/]+)/([^/]+)/(image\.png|program\.py|result\.json)', target)
        if match:
            try:
                file = rl_monitor.artifact(RL_ROOT, *match.groups())
                data = file.read_bytes()
            except (ValueError, OSError):
                return self.send_json({'error': 'RL artifact unavailable.'}, 404)
            self.send_response(200)
            self.send_header('Content-Type', {
                'image.png': 'image/png', 'program.py': 'text/plain; charset=utf-8',
                'result.json': 'application/json',
            }[match[5]])
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == '/api/drawing-prompts':
            try:
                dataset = drawing_dataset()
                return self.send_json({**dataset, 'runs': drawing_runs(dataset)})
            except (OSError, ValueError, TypeError) as error:
                return self.send_json({'error': str(error)}, 400)
        if self.path == '/api/models':
            served = bridge_model_catalog()
            catalog = {item['id']: item for item in model_catalog(served)}
            for item in served or []:
                if item['id'] in catalog:
                    catalog[item['id']].update(item)
                    if item.get('model_path') and 'label' not in item:
                        catalog[item['id']]['label'] = 'Qwen3.5 · 4B · configured checkpoint'
            return self.send_json({'models': list(catalog.values()), 'bridge_models': [item['id'] for item in served] if served is not None else None})
        match = re.fullmatch(r'/api/runs/([a-f0-9]{32})(?:/(image\.png|program\.py|trace\.json|result\.json|preview-\d+-\d+\.png))?', self.path)
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
                self.send_header('Content-Type', 'image/png' if match[2].endswith('.png') else 'text/plain; charset=utf-8')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            status = drawing_status(directory)
            console = directory / 'console.txt'
            status['console'] = console.read_text(errors='replace') if console.exists() else ''
            for name in ('trace.json', 'result.json'):
                file = directory / 'output' / name
                if file.exists() and status['state'] not in ('queued', 'running', 'stopping'):
                    status[name.split('.')[0]] = json.loads(file.read_text())
            status['artifacts'] = [name for name in ('image.png', 'program.py', 'trace.json', 'result.json') if (directory / 'output' / name).exists()]
            progress = directory / 'progress.json'
            if progress.exists():
                status['progress'] = json.loads(progress.read_text())
            return self.send_json(status)
        if self.path.startswith('/api/'):
            return self.send_json({'error': 'Not found.'}, 404)
        if target in ('/runs', '/runs/'):
            page = (ROOT / 'ui/runs.html').read_text().replace('__NAVIGATION__', navigation('runs'))
            return self.send_page(page)
        if target in ('/rl', '/rl/', '/rl/rollouts', '/rl/rollouts/',
                      '/rl/prompts', '/rl/prompts/'):
            view = 'prompts' if target in ('/rl/prompts', '/rl/prompts/') else 'rollouts'
            title = 'RL prompts' if view == 'prompts' else 'RL rollouts'
            description = ('Browse the full prompt selection for an RL run.'
                           if view == 'prompts'
                           else 'Inspect completed rollout images, rewards, and saved episode artifacts.')
            template = 'rl-prompts.html' if view == 'prompts' else 'rl-monitor.html'
            page = (ROOT / 'ui' / template).read_text()
            page = (page.replace('__NAVIGATION__', navigation('rl-' + view))
                        .replace('__RL_VIEW__', view)
                        .replace('__RL_TITLE__', title)
                        .replace('__RL_DESCRIPTION__', description))
            return self.send_page(page)
        if target in ('/', '/index.html'):
            from html import escape
            options = ''.join(
                f'<option value="{escape(model["id"], quote=True)}"'
                + (' selected' if model['id'] == DIRECT_MODELS[0]['id'] else '')
                + f'>{escape(model["label"])}</option>' for model in ARTSY_MODELS)
            return self.send_page((ROOT / 'ui/artsy.html').read_text().replace(
                '__ARTSY_MODEL_OPTIONS__', options))
        if target in ('/studio', '/studio/', '/drawings', '/drawings.html'):
            page = (ROOT / 'ui/index.html').read_text().replace('__NAVIGATION__', navigation('studio'))
            return self.send_page(page)
        if target not in ('/speech.js', '/artsy.js', '/activity.js', '/artsy.css', '/app.js', '/trace.js', '/style.css', '/rl-monitor.js', '/rl-monitor.css', '/runs.js', '/runs.css', '/preferences.js', '/preferences.css'):
            return self.send_json({'error': 'Not found.'}, 404)
        super().do_GET()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl-root', type=Path, help='RL checkpoints directory or one RL run; read-only monitoring')
    parser.add_argument('--rl-dataset', type=Path, help='Original RL prompt JSONL for the full prompt browser')
    parser.add_argument('--dataset', type=Path, help='Prompt JSONL for Studio and prompt review (defaults to the frozen benchmark)')
    parser.add_argument('--candidates', type=Path, help='Candidate JSONL for SFT inspection and export')
    parser.add_argument('--review-output', type=Path, help='Prompt review output JSON')
    parser.add_argument('--candidate-review', type=Path, help='Candidate review output JSON')
    parser.add_argument('--accepted-output', type=Path, help='Approved SFT export JSONL')
    args = parser.parse_args()
    if args.rl_root:
        RL_ROOT = args.rl_root.expanduser().resolve()
    if args.rl_dataset:
        RL_DATASET = args.rl_dataset.expanduser().resolve()
    CANDIDATE_DATASET = str(args.candidates.resolve()) if args.candidates else CANDIDATE_DATASET
    PROMPT_REVIEW_OUTPUT = args.review_output
    CANDIDATE_REVIEW_OUTPUT = args.candidate_review
    ACCEPTED_OUTPUT = args.accepted_output
    if args.dataset:
        DRAWING_DATASET = str(args.dataset.resolve())
    print(f'Artsy: http://127.0.0.1:{PORT}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
