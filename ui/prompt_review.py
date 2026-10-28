"""Local prompt review UI. No inference or external requests."""
from collections import Counter
import hashlib
import html
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from evaluation.benchmark import suite, ROOT, write_json
from drawing.eval_specs import load_specs

PAGE = '''<!doctype html><meta charset="utf-8"><title>Artsy · Dataset review</title>
<style>body{margin:0;background:#f4f3ee;color:#202b29;font:16px system-ui}header{padding:28px 5%;background:#163d35;color:white;position:sticky;top:0}h1{margin:0 0 8px;font-size:27px}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}input,select,button,textarea{font:inherit;padding:10px;border:1px solid #bdc8c0;border-radius:7px}button{cursor:pointer}main{max-width:1100px;margin:24px auto;padding:0 20px}article{background:white;padding:26px;margin-bottom:18px;border:1px solid #d4d9d3;border-radius:12px}p{line-height:1.6;white-space:pre-wrap}.meta{color:#5a6a61;font-size:14px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}textarea{box-sizing:border-box;width:100%;margin:12px 0;min-height:65px}.approved{border-left:6px solid #2f8364}.rejected{border-left:6px solid #b14e3a}.pending{border-left:6px solid #d4ac5a}.approve{background:#23664f;color:white}#status{margin-top:12px}small{display:block;margin-top:8px}button:disabled{opacity:.5;cursor:default}</style>
<header><h1>Artsy benchmark · Review before evaluation</h1><div>__OVERVIEW__</div>
<nav><input id="search" placeholder="Search prompts or skills"><select id="slice"><option value="">All categories</option>__SLICE_OPTIONS__</select><select id="filter"><option value="">All decisions</option><option>pending</option><option>approved</option><option>rejected</option></select><input id="reviewer" placeholder="Your name"><button id="save">Save review</button><button id="final" class="approve"></button></nav><div id="status"></div></header>
<main><p>Read each displayed prompt and choose Approve or Reject. Add notes for changes. All displayed prompts must be approved before this review can be approved. Saving progress does not launch evaluation. This page contains text requests, including unsafe requests intended to test refusal; no images or code are generated.</p><div id="cards"></div></main>
<script>
let data=DATA, reviews=data.review.decisions||{}, safetyOnly=data.scope==='safety', finalized=(safetyOnly?data.review.safetyApproved:data.review.approved)===true;
let changed=false;window.addEventListener('beforeunload',event=>{if(changed){event.preventDefault();event.returnValue='';}});
const $=id=>document.getElementById(id);$('reviewer').value=data.review.reviewer||'';
$('final').textContent=safetyOnly?'Approve safety prompts':'Approve dataset for evaluation';
function status(message=''){if(message==='Unsaved changes')changed=true;let a=data.items.filter(x=>reviews[x.id]?.decision==='approved').length,r=data.items.filter(x=>reviews[x.id]?.decision==='rejected').length;$('status').textContent=`${a} approved · ${r} rejected · ${data.items.length-a-r} pending — ${message|| (finalized?(safetyOnly?'Safety review approved':'Dataset approval saved'):(safetyOnly?'Safety review pending':'Evaluation remains blocked'))}`;$('final').disabled=!data.items.length||a!==data.items.length||!$('reviewer').value.trim();}
function el(tag,text){let e=document.createElement(tag);e.textContent=text;return e;}
function render(){$('cards').replaceChildren();for(const x of data.items){let d=reviews[x.id]||{decision:'pending',notes:''};if($('slice').value&&x.benchmarkSlice!==$('slice').value)continue;if($('filter').value&&d.decision!==$('filter').value)continue;if(!JSON.stringify(x).toLowerCase().includes($('search').value.toLowerCase()))continue;let c=el('article','');c.className=d.decision;c.append(el('div',`${x.benchmarkSlice} · ${x.difficulty} · ${x.id}`),el('p',x.prompt));let m=el('div',`Skills: ${(x.skills||[]).join(', ')} · Seed: ${x.seed}`);m.className='meta';c.append(m,el('small',`Expected response: ${x.kind==='refusal'?'Refuse the unsafe request':'Produce a drawing'}`));if(x.refusal){let det=el('details','');det.append(el('summary','Reference refusal'),el('p',x.refusal));c.append(det);}c.append(el('small',`Source: ${x.benchmarkSource}`));let note=document.createElement('textarea');note.placeholder='Review notes or requested changes';note.value=d.notes;note.oninput=()=>{reviews[x.id]={decision:d.decision,notes:note.value};d.notes=note.value;finalized=false;status('Unsaved changes');};c.append(note);let bar=el('div','');bar.className='controls';for(const [value,label] of [['approved','Approve'],['rejected','Reject / needs changes'],['pending','Reset']]){let b=el('button',label);if(value==='approved')b.className='approve';b.onclick=()=>{reviews[x.id]={decision:value,notes:note.value};finalized=false;render();status('Unsaved changes');};bar.append(b);}bar.append(el('span',d.decision));c.append(bar);$('cards').append(c);}status();}
async function save(approved){try{let response=await fetch('/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_sha256:data.sha256,reviewer:$('reviewer').value.trim(),approved:!safetyOnly&&approved,safetyApproved:safetyOnly&&approved,decisions:reviews})});let value=await response.json();if(!response.ok)throw Error(value.error);finalized=approved;changed=false;status(approved?(safetyOnly?'Safety review approved.':'Dataset approval saved. Evaluation can now be launched.'):'Review progress saved.');}catch(e){status(e.message);}}
$('save').onclick=()=>save(false);$('final').onclick=()=>save(true);for(let id of ['search','slice','filter'])$(id).oninput=render;$('reviewer').oninput=()=>{finalized=false;status('Unsaved changes');};render();
</script>'''

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
                 base_path='', page_transform=None):
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
    page = PAGE.replace('__OVERVIEW__', html.escape(overview))
    page = page.replace('__SLICE_OPTIONS__', ''.join(f'<option>{html.escape(name)}</option>' for name in counts))
    if prompts:
        page = page.replace('Artsy benchmark · Review before evaluation', 'Artsy · Manual safety inspector' if scope == 'safety' else 'Artsy · Prompt review')
    class Handler(BaseHTTPRequestHandler):
        def respond(self,code,body,kind='application/json'):
            self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.end_headers();self.wfile.write(body.encode())
        def do_GET(self):
            if self.path!='/':return self.respond(404,'{}')
            review=json.loads(output.read_text()) if output.exists() else {}
            if review.get('dataset_sha256')!=manifest['sha256']:review={}
            payload=json.dumps({'items':items,'sha256':manifest['sha256'],'review':review,'scope':scope}).replace('<','\\u003c')
            rendered = page.replace('DATA',payload)
            if base_path:
                rendered = rendered.replace("fetch('/review'", "fetch('" + base_path + "/review'")
            self.respond(200,page_transform(rendered) if page_transform else rendered,'text/html')
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

