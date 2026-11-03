"""Local prompt review UI. No inference or external requests."""
from collections import Counter
import hashlib
import html
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from evaluation.benchmark import suite, ROOT, write_json
from drawing.eval_specs import load_specs

PAGE = (Path(__file__).with_name('templates') / 'prompt-review.html').read_text(encoding='utf-8')

def validate_review(value, manifest, items, require_approval=False):
    if value.get('dataset_sha256')!=manifest['sha256']:raise ValueError('Approval belongs to a different dataset revision')
    decisions=value.get('decisions',{})
    if not isinstance(decisions,dict) or set(decisions)-{x['id'] for x in items}:raise ValueError('Unknown prompt IDs')
    for record in decisions.values():
        if not isinstance(record,dict) or record.get('decision') not in ('approved','rejected','pending') or not isinstance(record.get('notes',''),str):raise ValueError('Invalid review decision')
    if require_approval or value.get('approved'):
        if value.get('approved') is not True or not isinstance(value.get('reviewer'),str) or not value['reviewer'].strip():raise ValueError('Explicit named approval is required')
        if any(decisions.get(x['id'],{}).get('decision')!='approved' for x in items):raise ValueError('All prompts must be approved')
    if value.get('safetyApproved'):
        if value['safetyApproved'] is not True or not isinstance(value.get('reviewer'),str) or not value['reviewer'].strip():raise ValueError('Explicit named approval is required')
        safety = safety_items(items)
        if not safety or any(decisions.get(x['id'],{}).get('decision')!='approved' for x in safety):raise ValueError('All safety prompts must be approved')


def safety_items(items):
    return [x for x in items if x.get('benchmarkSlice') in ('benign_boundary','harmful_image','malicious_code')]


def merge_review(value, previous, manifest, all_items, visible_items, scope):
    """Update only this inspector's scope while retaining hidden decisions."""
    validate_review(value, manifest, all_items)
    if scope == 'safety' and value.get('approved'):
        raise ValueError('This inspector approves safety prompts only, not the full dataset')
    visible_ids = {x['id'] for x in visible_items}
    retained = previous.get('decisions', {}) if previous.get('dataset_sha256') == manifest['sha256'] else {}
    decisions = {**retained, **{key: decision for key, decision in value.get('decisions',{}).items() if key in visible_ids}}
    merged = {**value, 'decisions': decisions, 'reviewScope': scope}
    validate_review(merged, manifest, all_items)
    return merged


def prompt_suite(path):
    path = path.resolve()
    items = load_specs(path)
    for item in items:
        item.setdefault('benchmarkSlice', item.get('safetyCategory', 'harmful_image')
                        if item.get('kind') == 'refusal' else 'drawing')
        item['benchmarkSource'] = str(path)
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, items


def make_handler(prompts=None, output=None, include_drawings=True, port=4173, *,
                 base_path='', navigation_html=None):
    if prompts:
        manifest, items = prompt_suite(prompts)
        output = output or prompts.with_suffix('.review.json')
    else:
        manifest, items = suite()
        output = output or ROOT/'eval-results/dataset-review/approval.json'
    all_items = items
    scope = 'all' if include_drawings else 'safety'
    items = all_items if include_drawings else safety_items(all_items)
    counts = Counter(x['benchmarkSlice'] for x in items)
    overview = f"{len(items)} {'prompt' if len(items) == 1 else 'prompts'} · " + ' · '.join(f"{n} {name.replace('_', ' ')}" for name,n in counts.items())
    from ui.shell import navigation
    title = ('Safety prompt review' if scope == 'safety' else
             'Validated evaluation prompts' if manifest.get('status') == 'validated' else
             'Prompt review')
    description = ('Read each displayed prompt and choose Approve or Reject. Add notes for changes. '
                   'All displayed prompts must be approved before this review can be approved. '
                   'Saving progress does not launch evaluation. This page contains text requests, '
                   'including unsafe requests intended to test refusal; no images or code are generated.')
    if manifest.get('status') == 'validated':
        description = description.replace(
            'All displayed prompts must be approved before this review can be approved.',
            'This dataset is already validated; another review is optional and is not required to run evaluation.')
    links = ''.join(
        f'<a href="{url}"' + (' aria-current="true"' if chosen else '') + f'>{label}</a>'
        for url, label, chosen in (('/review', 'All prompts', scope == 'all'),
                                   ('/review?scope=safety', 'Safety prompts', scope == 'safety')))
    page = (PAGE.replace('__NAVIGATION__', navigation_html or navigation('review'))
            .replace('__PAGE_TITLE__', html.escape(title))
            .replace('__OVERVIEW__', html.escape(overview))
            .replace('__PAGE_DESCRIPTION__', html.escape(description))
            .replace('__SCOPE_LINKS__', links)
            .replace('__SLICE_OPTIONS__',
                     ''.join(f'<option>{html.escape(name)}</option>' for name in counts))
            .replace('__BASE_PATH__', base_path))
    class Handler(BaseHTTPRequestHandler):
        def respond(self,code,body,kind='application/json'):
            self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.end_headers();self.wfile.write(body.encode())
        def do_GET(self):
            assets = {
                '/assets/prompt-review.js': ('prompt-review.js', 'application/javascript'),
                '/assets/dom.js': ('dom.js', 'application/javascript'),
                '/assets/review.css': ('review.css', 'text/css'),
                '/assets/style.css': ('style.css', 'text/css'),
            }
            if self.path in assets:
                filename, mime = assets[self.path]
                return self.respond(200, Path(__file__).with_name(filename).read_text(encoding='utf-8'), mime)
            if self.path != '/':
                return self.respond(404, '{}')
            review = json.loads(output.read_text()) if output.exists() else {}
            if review.get('dataset_sha256') != manifest['sha256']:
                review = {}
            payload = json.dumps({
                'items': items, 'sha256': manifest['sha256'], 'review': review,
                'scope': scope, 'basePath': base_path,
                'validated': manifest.get('status') == 'validated',
            }).replace('<', '\\u003c')
            rendered = page.replace('__DATA__', payload)
            self.respond(200, rendered, 'text/html')
        def do_POST(self):
            if self.path!='/review':return self.respond(404,'{}')
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}'):return self.respond(403,'{}')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<200000:raise ValueError('Invalid request size')
                value=json.loads(self.rfile.read(length))
                previous=json.loads(output.read_text()) if output.exists() else {}
                value=merge_review(value,previous,manifest,all_items,items,scope)
                output.parent.mkdir(parents=True,exist_ok=True)
                temp=output.with_suffix('.tmp');write_json(temp,value);temp.replace(output)
                self.respond(200,'{"saved":true}')
            except (ValueError,TypeError,AttributeError) as e:self.respond(400,json.dumps({'error':str(e)}))
    return Handler

