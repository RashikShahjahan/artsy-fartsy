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

from drawing.config import MODEL_ID, MODEL_REVISION
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
DPO_CANDIDATE_DATASET = None
DPO_OUTPUT = ROOT / 'datasets' / 'dpo-v1'
COLLECTION_JOBS = ROOT / 'episodes' / 'dpo-ui'
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


def collection_dataset():
    if not DRAWING_DATASET:
        raise ValueError('Configure --dataset with training prompts for DPO collection. Frozen evaluation prompts cannot be collected.')
    dataset = drawing_dataset()
    raw = dataset_path().read_bytes()
    frozen = {row['prompt'].strip() for row in validate_specs([json.loads(line) for line in raw.splitlines() if line.strip()])}
    dataset['items'] = [row for row in dataset['items'] if row['prompt'].strip() not in frozen]
    if not dataset['items']:
        raise ValueError('Use training prompts for DPO collection; this dataset contains no eligible prompts outside the frozen evaluation benchmark.')
    return dataset


def collection_profile():
    """Select only a locally verified SFT sampler advertised by the bridge."""
    served = bridge_model_catalog() or []
    profiles = []
    api_hash = content_hash(ROOT / 'docs' / 'artcanvas.md')
    for file in sorted((ROOT / 'checkpoints').glob('*/final.json')):
        try:
            checkpoint = json.loads(file.read_text())
            config = checkpoint.get('config', {})
            if (config.get('model') != MODEL_ID or config.get('tokenizer_revision') != MODEL_REVISION
                    or config.get('renderer') != 'qwen3_5_preserve_thinking'
                    or type(config.get('lora_rank')) is not int or config['lora_rank'] < 1
                    or config.get('api_doc_sha256') != api_hash or config.get('algorithm') is not None
                    or any(not isinstance(checkpoint.get(k), str)
                           or not checkpoint[k].startswith('tinker://')
                           or len(checkpoint[k]) <= len('tinker://') for k in ('state_path', 'sampler_path'))):
                continue
            for item in served:
                if (item.get('id') in (MODEL_ID, 'artsy/' + file.parent.name)
                        and item.get('model_path') == checkpoint['sampler_path']):
                    profiles.append((item['id'] == MODEL_ID, file.stat().st_mtime_ns,
                                     file.parent.name, {'model': item['id'], 'model_path': checkpoint['sampler_path']}))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return max(profiles, key=lambda p: p[:3])[3] if profiles else None


def collection_checkpoint():
    profile = collection_profile()
    return profile['model_path'] if profile else None


def collection_request(body):
    if not isinstance(body, dict):
        raise ValueError('Invalid request.')
    dataset = collection_dataset()
    if body.get('dataset_sha256') != dataset['sha256']:
        raise ValueError('The prompt dataset changed. Reload before collecting.')
    ids = body.get('prompt_ids')
    if not isinstance(ids, list) or not 1 <= len(ids) <= 150 or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Choose 1–150 unique training prompt IDs.')
    lookup = {row['id']: row for row in dataset['items']}
    if any(i not in lookup for i in ids):
        raise ValueError('Training prompt not found or belongs to the frozen evaluation benchmark.')
    count = body.get('candidates')
    if type(count) is not int or not 2 <= count <= 4:
        raise ValueError('Choose 2–4 candidates per prompt.')
    profile = collection_profile()
    checkpoint = profile['model_path'] if profile else None
    if not checkpoint or body.get('model_path') != checkpoint:
        raise ValueError('Configure and select the SFT tinker:// checkpoint on the local bridge before collecting.')
    return {'specs': [lookup[i] for i in ids], 'candidates': count, 'model': profile['model'], 'model_path': checkpoint,
            'dataset_sha256': dataset['sha256']}


def collection_status(directory, console=False):
    status = json.loads((directory / 'job.json').read_text())
    result = {'id': directory.name, 'state': status['state'],
              'candidates_path': str(directory / 'dataset' / 'candidates.jsonl') if (directory / 'dataset' / 'candidates.jsonl').is_file() else None}
    if status.get('error'):
        result['error'] = status['error']
    if console or status['state'] == 'failed':
        file = directory / 'console.txt'
        result['console'] = file.read_text(errors='replace')[-100000:] if file.exists() else ''
    return result


def run_dpo_collection(directory, request):
    global ACTIVE
    status = {'state': 'running', 'dataset_sha256': request['dataset_sha256']}
    try:
        (directory / 'job.json').write_text(json.dumps(status))
        with (directory / 'console.txt').open('w') as output:
            process = subprocess.run([interpreter(), '-u', '-m', 'ui.dpo_collect', str(directory / 'request.json')],
                cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False)
        status.update(state='completed' if process.returncode == 0 else 'failed', exit_code=process.returncode)
    except Exception as error:
        status.update(state='failed', error=str(error))
    finally:
        try:
            (directory / 'job.json').write_text(json.dumps(status))
        finally:
            with LOCK:
                ACTIVE = None


def content_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None


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
        mount = next((name for name in ('review', 'candidates', 'dpo')
                      if target.path == '/' + name or target.path.startswith('/' + name + '/')), None)
        if mount is None:
            return False
        try:
            with REVIEW_LOCK:
                safety_only = mount == 'review' and parse_qs(target.query).get('scope') == ['safety']
                collection = parse_qs(target.query).get('collection', [None])[0]
                internal = re.match(r'^/(?:dpo|candidates)/collections/([a-f0-9]{32})(/.*)?$', target.path)
                if internal:
                    collection = internal[1]
                source = DPO_CANDIDATE_DATASET or CANDIDATE_DATASET if mount == 'dpo' else CANDIDATE_DATASET
                review_output = CANDIDATE_REVIEW_OUTPUT
                output = Path(DPO_OUTPUT)
                base = '/' + mount
                route = target.path[len(base):] or '/'
                if collection is not None:
                    if mount == 'review' or not re.fullmatch(r'[a-f0-9]{32}', collection):
                        raise ValueError('Invalid collection ID.')
                    directory = COLLECTION_JOBS / collection
                    status = collection_status(directory)
                    if status['state'] not in ('completed', 'failed') or not status['candidates_path']:
                        raise ValueError('Collection candidates are not available.')
                    source = status['candidates_path']
                    review_output = directory / 'dataset' / 'manual-review.json'
                    output = output / 'exports' / collection
                    base += '/collections/' + collection
                    route = internal[2] or '/' if internal else '/'
                manual_path = review_output or (Path(source).parent / 'manual-review.json' if source else None)
                key = (mount, safety_only, DRAWING_DATASET, source, PROMPT_REVIEW_OUTPUT,
                       str(review_output), ACCEPTED_OUTPUT, str(output), base,
                       content_hash(source), content_hash(manual_path), content_hash(output / 'dpo-review.json'))
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
                    elif mount == 'dpo':
                        from ui.dpo_review import make_handler
                        handler_type = make_handler(Path(source) if source else None, PORT, output,
                            review_path=review_output, base_path=base, page_transform=transform)
                    elif source:
                        from ui.candidate_review import make_handler
                        handler_type = make_handler(Path(source), PORT,
                            review_output, None if collection else ACCEPTED_OUTPUT,
                            base_path=base, page_transform=transform)
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
                    query = '&'.join(part for part in target.query.split('&') if not part.startswith('collection='))
                    if query:
                        handler.path += '?' + query
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
        if urlsplit(self.path).path == '/dpo/api/collect':
            return self.collect()
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

    def collect(self):
        global ACTIVE
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 20000:
                raise ValueError('Invalid request size.')
            request = collection_request(json.loads(self.rfile.read(size)))
        except (OSError, ValueError, TypeError) as error:
            return self.send_json({'error': str(error)}, 400)
        with LOCK:
            if ACTIVE:
                return self.send_json({'error': 'A generation is already running.'}, 409)
            ident = uuid4().hex
            directory = COLLECTION_JOBS / ident
            directory.mkdir(parents=True)
            worker_request = {**request, 'output': str(directory / 'dataset')}
            (directory / 'request.json').write_text(json.dumps(worker_request))
            (directory / 'job.json').write_text(json.dumps({'state': 'queued'}))
            ACTIVE = ident
            try:
                threading.Thread(target=run_dpo_collection, args=(directory, request), daemon=True).start()
            except Exception:
                ACTIVE = None
                (directory / 'job.json').write_text(json.dumps({'state': 'failed'}))
                raise
        self.send_json({'id': ident}, 202)

    def do_GET(self):
        if not self.trusted():
            return self.send_json({'error': 'Local requests only.'}, 403)
        if self.path == '/dpo/api/collection':
            try:
                dataset = collection_dataset()
                error = None
            except (OSError, ValueError, TypeError) as failure:
                dataset = {'sha256': None, 'items': []}
                error = str(failure)
            profile = collection_profile()
            checkpoint = profile['model_path'] if profile else None
            completed = []
            active = None
            for file in sorted(COLLECTION_JOBS.glob('*/job.json'), key=lambda p: p.stat().st_mtime, reverse=True):
                if not re.fullmatch(r'[a-f0-9]{32}', file.parent.name):
                    continue
                try:
                    status = collection_status(file.parent, console=file.parent.name == ACTIVE)
                    if file.parent.name == ACTIVE:
                        active = status
                    elif status['state'] in ('completed', 'failed'):
                        completed.append(status)
                except (OSError, ValueError, KeyError):
                    continue
            return self.send_json({'dataset': dataset, 'error': error, 'model_path': checkpoint,
                                   'checkpoint_ready': bool(checkpoint), 'collection_model': profile['model'] if profile else None, 'active': active, 'completed': completed})
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
    parser.add_argument('--dpo-output', type=Path, default=DPO_OUTPUT, help='DPO labels and exports directory')
    parser.add_argument('--dpo-candidates', type=Path, help='Separate DPO candidate JSONL')
    parser.add_argument('--review-output', type=Path, help='Prompt review output JSON')
    parser.add_argument('--candidate-review', type=Path, help='Candidate review output JSON')
    parser.add_argument('--accepted-output', type=Path, help='Approved SFT export JSONL')
    args = parser.parse_args()
    CANDIDATE_DATASET = str(args.candidates.resolve()) if args.candidates else CANDIDATE_DATASET
    DPO_OUTPUT = args.dpo_output.resolve()
    DPO_CANDIDATE_DATASET = str(args.dpo_candidates.resolve()) if args.dpo_candidates else None
    PROMPT_REVIEW_OUTPUT = args.review_output
    CANDIDATE_REVIEW_OUTPUT = args.candidate_review
    ACCEPTED_OUTPUT = args.accepted_output
    if args.dataset:
        DRAWING_DATASET = str(args.dataset.resolve())
    print(f'Artsy: http://127.0.0.1:{PORT}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
