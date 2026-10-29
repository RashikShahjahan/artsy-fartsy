"""Local human DPO pairing workspace; never executes candidate programs."""
from __future__ import annotations

import copy
import fcntl
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlsplit

from ui.candidate_review import load_candidates, validate_review, approved_record
from training.data import convert_record, digest, artifact, native_trajectory_hash

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Artsy · DPO dataset</title>
<style>body{font:15px system-ui;background:#f5f4ef;color:#23342f;margin:24px}input,select,button{font:inherit;padding:9px;margin:4px}button{cursor:pointer}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;height:350px;object-fit:contain;background:white}article{border:1px solid #ccc;padding:16px;margin:12px 0}pre{white-space:pre-wrap;max-height:250px;overflow:auto}#status{white-space:pre-wrap}label{display:block}@media(max-width:700px){.pair{display:block}}</style>
<header><h1>DPO dataset workspace</h1></header><main><p>Compare approved drawing trajectories for the same exact prompt and seed. Judge visual quality only after confirming both drawings are eligible. Compare composition, color, coherence, craftsmanship and expressive appeal.</p><p><a href="/">Open Studio to generate more candidates</a> · <a href="__CANDIDATE_LINK__">Review candidate approvals</a></p>
<section id="collection"><h2>Generate comparison candidates</h2><p>Collection target: 100 prompts × 2 candidates = 200 trajectories, yielding at most 100 comparison pairs before exclusions and ties. Held-out pairs reduce the number available for training.</p><p id="collection-status">Loading available prompts…</p><div id="collection-controls"></div></section><input id="search" placeholder="Search prompt"><input id="reviewer" placeholder="Reviewer name"><button id="save">Save decisions</button><button id="export">Export validated dataset</button><p id="progress"></p><p id="status" role="status"></p><div id="groups"></div>
</main><script>const data=__DATA__, $=id=>document.getElementById(id);let decisions=data.state.decisions,revision=data.revision,dirty=false;const selectedPairs=new Map();const api=data.base+'/api/';
function el(t,s){let n=document.createElement(t);if(s!==undefined)n.textContent=s;return n}function changed(){dirty=true;$('status').textContent='Unsaved changes'}
$('reviewer').value=data.state.reviewer||'';$('reviewer').oninput=changed;window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
function render(){let q=$('search').value.toLowerCase();$('groups').replaceChildren();$('progress').textContent=data.items.length+' approved convertible candidates · '+data.groups.length+' comparison groups · '+Object.values(decisions).filter(d=>d.winner&&d.winner!=='Tie'&&d.split==='train').length+' training pairs · '+Object.values(decisions).filter(d=>d.winner&&d.split!=='train').length+' held-out pairs · '+Object.values(decisions).filter(d=>d.winner==='Tie').length+' ties · '+data.excluded.length+' unavailable candidates';if(data.excluded.length){let unavailable=el('details');unavailable.append(el('summary','Unavailable candidates and reasons'));for(let x of data.excluded)unavailable.append(el('p',x.id+': '+x.reason));$('groups').append(unavailable)}for(let g of data.groups){if(!g.prompt.toLowerCase().includes(q))continue;let card=el('article');card.append(el('h2',g.prompt),el('p','Seed '+g.seed));if(g.ids.length<2){card.append(el('p','Generate another candidate with this exact prompt and seed.'));$('groups').append(card);continue}let panels=el('div');panels.className='pair';let selects=[],images=[];for(let i=0;i<2;i++){let panel=el('section'),select=el('select');select.setAttribute('aria-label',i?'Candidate B':'Candidate A');for(let id of g.ids){let o=el('option',id);o.value=id;select.append(o)}select.selectedIndex=i;if(selectedPairs.has(JSON.stringify([g.prompt,g.seed])))select.value=selectedPairs.get(JSON.stringify([g.prompt,g.seed]))[i];let image=el('img');image.alt=i?'Candidate B final drawing':'Candidate A final drawing';image.tabIndex=0;image.style.cursor='zoom-in';function enlarge(){let dialog=el('dialog'),full=el('img'),close=el('button','Close image');full.src=image.src;full.alt=image.alt;full.style.maxWidth='90vw';full.style.maxHeight='85vh';close.onclick=()=>dialog.close();dialog.append(close,full);dialog.onclose=()=>{dialog.remove();image.focus()};document.body.append(dialog);dialog.showModal()}image.onclick=enlarge;image.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();enlarge()}};selects.push(select);images.push(image);panel.append(el('h3',i?'Candidate B':'Candidate A'),select,image);panels.append(panel)}let family=el('input');family.placeholder='Family / group ID';family.setAttribute('aria-label','Family group');let split=el('select');split.setAttribute('aria-label','Dataset split');for(let s of ['train','validation','test'])split.append(el('option',s));let check=el('input');check.type='checkbox';let confirmation=el('label');confirmation.append(check,document.createTextNode(' Both candidates are safe, render-valid, and adhere to the prompt'));let result=el('p');function key(){return [selects[0].value,selects[1].value].sort().join(' / ')}function update(){selectedPairs.set(JSON.stringify([g.prompt,g.seed]),selects.map(s=>s.value));selects.forEach((s,i)=>images[i].src=data.items.find(x=>x.id===s.value).image);let d=decisions[key()];if(d){selects[0].value=d.record_a;selects[1].value=d.record_b;selects.forEach((s,i)=>images[i].src=data.items.find(x=>x.id===s.value).image)}family.value=d?.group_id||g.family;split.value=d?.split||'train';check.checked=d?.eligible===true;result.textContent=d?'Decision: '+(d.winner||'Skipped'):'Unlabeled'}selects.forEach(s=>s.onchange=update);function metadataChanged(){let d=decisions[key()];if(d){d.group_id=family.value.trim();d.split=split.value;d.eligible=check.checked;changed()}}family.oninput=metadataChanged;split.onchange=metadataChanged;check.onchange=metadataChanged;card.append(panels,family,split,confirmation,result);for(let winner of ['A','B','Tie','Skip']){let b=el('button',winner);b.onclick=()=>{if(selects[0].value===selects[1].value){result.textContent='Choose distinct candidates';return}if(winner!=='Skip'&&!check.checked){result.textContent='Confirm eligibility first';return}decisions[key()]={record_a:selects[0].value,record_b:selects[1].value,winner:winner==='Skip'?null:winner,eligible:check.checked,group_id:family.value.trim(),split:split.value};changed();render()};card.append(b)}let existing=el('select');existing.setAttribute('aria-label','Edit labeled pair');existing.append(el('option','Select an existing decision to edit'));for(let [k,d] of Object.entries(decisions)){if(g.ids.includes(d.record_a)&&g.ids.includes(d.record_b)){let o=el('option',k+' · '+(d.winner||'Skip'));o.value=k;existing.append(o)}}existing.onchange=()=>{let d=decisions[existing.value];if(d){selects[0].value=d.record_a;selects[1].value=d.record_b;update()}};card.append(existing);update();$('groups').append(card)}}
async function post(endpoint){let r=await fetch(api+endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_sha256:data.sha256,approval_sha256:data.approval_sha256,revision,reviewer:$('reviewer').value.trim(),decisions})});let v=await r.json();if(!r.ok)throw Error(v.error);revision=v.revision;return v}
$('save').onclick=async()=>{try{await post('review');dirty=false;render();$('status').textContent='Decisions saved'}catch(e){$('status').textContent=e.message}};$('export').onclick=async()=>{try{if(dirty)throw Error('Save decisions before exporting');let v=await post('export');$('status').textContent='Exported '+v.count+' pairs to '+v.path}catch(e){$('status').textContent=e.message}};$('search').oninput=render;render();
function selectCollectionPrompts(items, limit=100){
 const buckets=new Map();
 for(const prompt of items){
  const key=JSON.stringify([prompt.family||'',prompt.difficulty||'']);
  if(!buckets.has(key))buckets.set(key,[]);
  buckets.get(key).push(prompt);
 }
 const queues=[...buckets.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([,rows])=>rows);
 const result=[],seen=new Set();
 let position=0,remaining=true;
 while(result.length<limit&&remaining){
  remaining=false;
  for(const rows of queues){
   if(position>=rows.length)continue;
   remaining=true;
   const prompt=rows[position];
   if(!seen.has(prompt.id)){seen.add(prompt.id);result.push(prompt.id)}
   if(result.length===limit)break;
  }
  position++;
 }
 return result;
}
let collectionData=null, selectedPrompts=new Set(), collectionBusy=false;
async function collection(){try{let response=await fetch('/dpo/api/collection');let c=await response.json();if(!response.ok)throw Error(c.error||'Collection unavailable');collectionData=c;let controls=$('collection-controls');if(!controls.childNodes.length){let search=el('input');search.placeholder='Search generation prompts';search.setAttribute('aria-label','Search generation prompts');let list=el('div');list.style.maxHeight='220px';list.style.overflow='auto';function prompts(){list.replaceChildren();for(let p of c.dataset.items){if(!p.prompt.toLowerCase().includes(search.value.toLowerCase()))continue;let label=el('label'),check=el('input');check.type='checkbox';check.checked=selectedPrompts.has(p.id);check.onchange=()=>{check.checked?selectedPrompts.add(p.id):selectedPrompts.delete(p.id);budget()};label.append(check,document.createTextNode(p.prompt));list.append(label)}}search.oninput=prompts;let count=el('select');count.setAttribute('aria-label','Candidates per prompt');for(let n of [2,3,4])count.append(el('option',String(n)));let estimate=el('p');function budget(){estimate.textContent=selectedPrompts.size+' prompts × '+count.value+' = '+(selectedPrompts.size*Number(count.value))+' candidate generations (uses model credits)'}count.onchange=budget;let generate=el('button','Generate candidates (uses model credits)');generate.onclick=async()=>{try{if(!selectedPrompts.size)throw Error('Choose prompts first');generate.disabled=true;let r=await fetch('/dpo/api/collect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_sha256:collectionData.dataset.sha256,prompt_ids:[...selectedPrompts],candidates:Number(count.value),model_path:collectionData.model_path})});let v=await r.json();if(!r.ok)throw Error(v.error);await collection()}catch(e){$('collection-status').textContent=e.message}finally{generate.disabled=!collectionData.checkpoint_ready||collectionBusy||!!collectionData.error||!collectionData.dataset.items.length}};generate.id='collect-button';let preset=el('button','Select 100 prompts');preset.type='button';
const available=new Set(c.dataset.items.map(p=>p.id)).size;
preset.disabled=available<100;
let presetStatus=el('p',available<100?'100-prompt preset unavailable: '+available+' prompts available; exactly 100 are required.':available+' prompts available. The preset selects 100 across family and difficulty and sets 2 candidates each.');
preset.onclick=()=>{const ids=selectCollectionPrompts(collectionData.dataset.items,100);if(ids.length!==100){presetStatus.textContent='Exactly 100 unique prompts are required; '+ids.length+' available.';return}selectedPrompts=new Set(ids);count.value='2';prompts();budget();presetStatus.textContent='Selected exactly 100 prompts with 2 candidates each. Click Generate candidates to begin.'};
controls.append(preset,presetStatus,search,list,count,estimate,generate);prompts();budget()}collectionBusy=!!c.active&&['queued','running','starting'].includes(c.active.state);$('collect-button').disabled=!c.checkpoint_ready||collectionBusy||!!c.error||!c.dataset.items.length;let status=$('collection-status');status.replaceChildren(el('span',c.error||(!c.checkpoint_ready?'Configure ARTSY_TINKER_MODEL_PATH with an SFT checkpoint and restart make serve. ':'')));status.append(el('p','Current checkpoint: '+(c.model_path||'not configured')));if(c.active){status.append(el('p','Job '+c.active.id+' · '+c.active.state));if(c.active.console)status.append(el('pre',c.active.console))}for(let job of c.completed||[]){let row=el('p',job.id+' · '+job.state+' ');if(job.error)row.append(el('span',job.error));if(job.console)row.append(el('pre',job.console));if(job.candidates_path)for(let [url,text] of [['/candidates?collection=','Approve candidates'],['/dpo?collection=','Compare candidates']]){let a=el('a',text);a.href=url+encodeURIComponent(job.id);row.append(a,document.createTextNode(' · '))}status.append(row)}if(collectionBusy)setTimeout(collection,2000)}catch(e){$('collection-status').textContent=e.message}}
collection();</script></html>'''


def make_handler(candidate_path, port, output_dir, review_path=None, base_path='/dpo', page_transform=None):
    path = Path(candidate_path).resolve() if candidate_path else None
    root = path.parent if path else Path.cwd().resolve()
    output = Path(output_dir).resolve()
    approval = Path(review_path or root / 'manual-review.json').resolve()
    state_path = output / 'dpo-review.json'
    records, sha = load_candidates(path) if path else ([], digest(b''))
    approval_bytes = approval.read_bytes() if path and approval.exists() else b''
    approval_sha = digest(approval_bytes)
    reviewed = json.loads(approval_bytes) if approval_bytes else {'decisions': {}}
    if approval_bytes:
        validate_review(reviewed, sha, records)
    for destination in (state_path, output / 'accepted.jsonl', output / 'labels.jsonl'):
        if destination in (path, approval):
            raise ValueError('DPO outputs must not overwrite input files')
    lock = threading.RLock()
    eligible, excluded = {}, []
    for original in records:
        decision = reviewed['decisions'].get(original['id'], {})
        if decision.get('decision') == 'rejected' or (decision.get('decision') != 'approved' and original.get('status') != 'accepted'):
            excluded.append({'id': original['id'], 'reason': 'Not approved'})
            continue
        record = (approved_record(original, reviewed['reviewer'], decision)
                  if decision.get('decision') == 'approved' else copy.deepcopy(original))
        try:
            if record['spec'].get('kind') == 'refusal' or record.get('refusal'):
                raise ValueError('Visual DPO requires drawings')
            convert_record(record, root)
            artifact(root, record['artifacts']['image'], record['imageHash'])
            eligible[record['id']] = record
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            excluded.append({'id': record['id'], 'reason': str(exc)})

    @contextmanager
    def write_lock():
        # File lock serializes saves across independently mounted handler instances.
        output.mkdir(parents=True, exist_ok=True)
        with (output / '.dpo-review.lock').open('a') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def fresh():
        if (path and digest(path.read_bytes()) != sha) or digest(approval.read_bytes() if path and approval.exists() else b'') != approval_sha:
            raise ValueError('Candidate dataset or manual approvals changed. Restart/reload the workspace.')

    def state():
        raw = state_path.read_bytes() if state_path.exists() else b''
        value = json.loads(raw) if raw else {'dataset_sha256': sha, 'approval_sha256': approval_sha, 'reviewer': '', 'decisions': {}}
        if value.get('dataset_sha256') != sha:
            raise ValueError('Saved DPO review belongs to a different candidate dataset')
        if value.get('approval_sha256') != approval_sha:
            try:
                validate(value)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError('Manual approvals changed and saved pairs are no longer eligible. Restore approvals or use a new output directory; saved decisions were retained. ' + str(exc)) from exc
            value['approval_sha256'] = approval_sha
        return value, digest(raw)

    def validate(value):
        if not isinstance(value.get('reviewer'), str) or not value['reviewer'].strip():
            raise ValueError('Reviewer name is required')
        if not isinstance(value.get('decisions'), dict):
            raise ValueError('Invalid decisions')
        pair_images, split_ids = set(), {}
        for key, d in value['decisions'].items():
            if not isinstance(d, dict):
                raise ValueError('Invalid pair decision')
            a, b = d.get('record_a'), d.get('record_b')
            if a not in eligible or b not in eligible or a == b or key != ' / '.join(sorted([a, b])):
                raise ValueError('Pair requires distinct approved candidates and a canonical key')
            ra, rb = eligible[a], eligible[b]
            if (ra['spec']['prompt'], ra['spec']['seed']) != (rb['spec']['prompt'], rb['spec']['seed']):
                raise ValueError('Pair must have the same exact prompt and seed')
            if d.get('winner') is None:
                continue
            if d['winner'] not in ('A', 'B', 'Tie') or d.get('eligible') is not True:
                raise ValueError('Confirm both candidates are safe, render-valid and prompt-adherent')
            if not isinstance(d.get('group_id'), str) or not d['group_id'].strip() or d.get('split') not in ('train', 'validation', 'test'):
                raise ValueError('Every label requires family/group and split')
            image_key = tuple(sorted([ra['imageHash'], rb['imageHash']]))
            if image_key[0] == image_key[1] or image_key in pair_images:
                raise ValueError('Identical, duplicate or reversed image pair')
            pair_images.add(image_key)
            identities = [('family', d['group_id']), ('prompt', ra['spec']['prompt'])]
            for r in (ra, rb):
                identities += [('record', r['id']), ('source', r['sourceHash']), ('image', r['imageHash'])]
            for identity in identities:
                if identity in split_ids and split_ids[identity] != d['split']:
                    raise ValueError('Family, prompt, source, record or image leaks across splits')
                split_ids[identity] = d['split']

    def export(value):
        from training.dpo import load_preferences
        validate(value)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.dpo-stage-', dir=output) as tmp:
            stage = Path(tmp)
            selected = {rid for d in value['decisions'].values() if d.get('winner') for rid in (d['record_a'], d['record_b'])}
            converted = {}
            # Preserve relative paths, copying only explicitly referenced artifacts and preview files.
            for rid in sorted(selected):
                r = copy.deepcopy(eligible[rid])
                names = {r['artifacts']['source'], r['artifacts']['image']}
                names.update(r['artifacts'].get('inspectLogs') or [])
                if r['artifacts'].get('result'):
                    names.add(r['artifacts']['result'])
                original_directory = r['artifacts']['directory']
                directory = str((root / original_directory).resolve().relative_to(root))
                for step in r['trace']:
                    feedback = step.get('feedback') or {}
                    if feedback.get('ok'):
                        names.add(str(Path(directory) / Path(feedback['artifact']).name))
                def references(obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k in ('image', 'source', 'artifact', 'trajectory') and isinstance(v, str):
                                p = (root / v).resolve()
                                if p.is_relative_to(root) and p.is_file():
                                    names.add(v)
                            references(v)
                    elif isinstance(obj, list):
                        for v in obj: references(v)
                references(r)
                mapping = {}
                for name in names:
                    source = artifact(root, name)
                    relative = source.relative_to(root)
                    target = stage / 'artifacts' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    mapping[name] = str(Path('artifacts') / relative)
                def rewrite(obj, key=None):
                    if isinstance(obj, dict):
                        return {k: rewrite(v, k) for k, v in obj.items()}
                    if isinstance(obj, list): return [rewrite(v, key) for v in obj]
                    if key in ('image', 'source', 'artifact', 'trajectory', 'result', 'inspectLogs') and isinstance(obj, str):
                        return mapping.get(obj, obj)
                    return obj
                r = rewrite(r)
                r['artifacts']['directory'] = str(Path('artifacts') / Path(directory))
                if r.get('trajectoryFormat') == 'inspect-bash-v1':
                    r['nativeTrajectoryHash'] = native_trajectory_hash(r['nativeTrajectory'])
                converted[rid] = r
            labels = []
            for key, d in value['decisions'].items():
                if not d.get('winner'): continue
                labels.append({**d, 'pair_id': digest(key.encode())[:24], 'image_a': converted[d['record_a']]['artifacts']['image'], 'image_b': converted[d['record_b']]['artifacts']['image'], 'reviewer': value['reviewer']})
            for name, rows in [('accepted.jsonl', list(converted.values())), ('labels.jsonl', labels)]:
                (stage / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            load_preferences(stage / 'labels.jsonl', stage / 'accepted.jsonl')
            fresh()
            # Content-addressed export namespace avoids replacing artifacts used by older readers.
            namespace = 'export-' + digest((stage / 'accepted.jsonl').read_bytes())[:24]
            final_artifacts = output / namespace
            for name in ('accepted.jsonl', 'labels.jsonl'):
                rows = [json.loads(line) for line in (stage / name).read_text().splitlines()]
                def relocate(obj, key=None):
                    if isinstance(obj, dict): return {k: relocate(v, k) for k,v in obj.items()}
                    if isinstance(obj, list): return [relocate(v, key) for v in obj]
                    if key in ('image', 'source', 'artifact', 'trajectory', 'directory', 'image_a', 'image_b', 'result', 'inspectLogs') and isinstance(obj, str) and obj.startswith('artifacts/'):
                        return namespace + obj[len('artifacts'):]
                    return obj
                relocated = [relocate(r) for r in rows]
                for r in relocated:
                    if r.get('trajectoryFormat') == 'inspect-bash-v1':
                        r['nativeTrajectoryHash'] = native_trajectory_hash(r['nativeTrajectory'])
                (stage / name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in relocated))
            (stage / 'artifacts').rename(stage / namespace)
            load_preferences(stage / 'labels.jsonl', stage / 'accepted.jsonl')
            if not final_artifacts.exists(): shutil.move(str(stage / namespace), final_artifacts)
            for name in ('accepted.jsonl', 'labels.jsonl'):
                os.replace(stage / name, output / name)
            return len(labels)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, body, mime='application/json'):
            if not isinstance(body, bytes): body = json.dumps(body).encode()
            self.send_response(code)
            for k, v in [('Content-Type', mime), ('Content-Length', str(len(body))), ('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff')]: self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def trusted(self):
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}') and self.headers.get('Origin') in (None, f'http://127.0.0.1:{port}', f'http://localhost:{port}')

        def do_GET(self):
            if not self.trusted(): return self.respond(403, {'error': 'Local requests only'})
            try:
                with lock:
                    fresh()
                    target = urlsplit(self.path)
                    if target.path == '/api/image':
                        rid = parse_qs(target.query).get('id', [''])[0]
                        if rid not in eligible: return self.respond(404, {'error': 'Unknown candidate'})
                        r = eligible[rid]
                        return self.respond(200, artifact(root, r['artifacts']['image'], r['imageHash']).read_bytes(), 'image/png')
                    if target.path not in ('/', '/api/state'): return self.respond(404, {'error': 'Not found'})
                    value, revision = state()
                    from urllib.parse import quote
                    items = [{'id': rid, 'image': base_path + '/api/image?id=' + quote(rid, safe='')} for rid in eligible]
                    groups = {}
                    for rid, r in eligible.items():
                        spec = r['spec']; key = (spec['prompt'], spec['seed'])
                        groups.setdefault(key, {'prompt': key[0], 'seed': key[1], 'family': spec.get('family') or 'drawing', 'ids': []})['ids'].append(rid)
                    payload = {'items': items, 'groups': list(groups.values()), 'excluded': excluded, 'state': value, 'revision': revision, 'sha256': sha, 'approval_sha256': approval_sha, 'base': base_path}
                    if target.path == '/api/state': return self.respond(200, payload)
                    page = PAGE.replace('__DATA__', json.dumps(payload).replace('<', '\\u003c'))
                    candidate_link = '/candidates' + base_path[len('/dpo'):] if base_path.startswith('/dpo/collections/') else '/candidates'
                    page = page.replace('__CANDIDATE_LINK__', candidate_link)
                    if page_transform: page = page_transform(page)
                    self.respond(200, page.encode(), 'text/html; charset=utf-8')
            except (OSError, ValueError, TypeError, KeyError) as exc: self.respond(400, {'error': str(exc)})

        def do_POST(self):
            if not self.trusted(): return self.respond(403, {'error': 'Local requests only'})
            if self.path not in ('/api/review', '/api/export'):
                return self.respond(404, {'error': 'Not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4_000_000: raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(length))
                with lock, write_lock():
                    fresh()
                    saved, revision = state()
                    if not isinstance(value, dict) or value.get('dataset_sha256') != sha or value.get('approval_sha256') != approval_sha or value.get('revision') != revision:
                        return self.respond(409, {'error': 'Inputs or decisions changed. Reload before saving.'})
                    if self.path == '/api/review':
                        validate(value)
                        persisted = {k: value[k] for k in ('dataset_sha256', 'approval_sha256', 'reviewer', 'decisions')}
                        output.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(mode='w', dir=output, delete=False) as stream:
                            json.dump(persisted, stream, ensure_ascii=False, indent=2); temporary = stream.name
                        os.replace(temporary, state_path)
                        return self.respond(200, {'saved': True, 'revision': state()[1]})
                    if self.path == '/api/export':
                        count = export(saved)
                        return self.respond(200, {'count': count, 'path': str(output), 'revision': revision})
                    self.respond(404, {'error': 'Not found'})
            except (OSError, ValueError, TypeError, KeyError, IndexError) as exc: self.respond(400, {'error': str(exc)})
    return Handler
