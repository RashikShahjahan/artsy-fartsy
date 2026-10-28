"""Local prompt review UI. No inference or external requests."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from .benchmark import suite, ROOT, write_json

PAGE = '''<!doctype html><meta charset="utf-8"><title>Artsy · Dataset review</title>
<style>body{margin:0;background:#f4f3ee;color:#202b29;font:16px system-ui}header{padding:28px 5%;background:#163d35;color:white;position:sticky;top:0}h1{margin:0 0 8px;font-size:27px}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}input,select,button,textarea{font:inherit;padding:10px;border:1px solid #bdc8c0;border-radius:7px}button{cursor:pointer}main{max-width:1100px;margin:24px auto;padding:0 20px}article{background:white;padding:26px;margin-bottom:18px;border:1px solid #d4d9d3;border-radius:12px}p{line-height:1.6;white-space:pre-wrap}.meta{color:#5a6a61;font-size:14px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}textarea{box-sizing:border-box;width:100%;margin:12px 0;min-height:65px}.approved{border-left:6px solid #2f8364}.rejected{border-left:6px solid #b14e3a}.pending{border-left:6px solid #d4ac5a}.approve{background:#23664f;color:white}#status{margin-top:12px}small{display:block;margin-top:8px}button:disabled{opacity:.5;cursor:default}</style>
<header><h1>Artsy benchmark · Review before evaluation</h1><div>80 prompts · 56 drawings · 8 benign boundary · 8 harmful image · 8 malicious code</div>
<nav><input id="search" placeholder="Search prompts or skills"><select id="slice"><option value="">All categories</option><option>drawing</option><option>benign_boundary</option><option>harmful_image</option><option>malicious_code</option></select><select id="filter"><option value="">All decisions</option><option>pending</option><option>approved</option><option>rejected</option></select><input id="reviewer" placeholder="Your name"><button id="save">Save review</button><button id="final" class="approve">Approve dataset for evaluation</button></nav><div id="status"></div></header>
<main><p>Read each prompt and choose Approve or Reject. Add notes for changes. All 80 prompts must be approved before final approval is available. Saving progress does not authorize evaluation. This page contains text requests, including unsafe requests intended to test refusal; no images or code are generated.</p><div id="cards"></div></main>
<script>
let data=DATA, reviews=data.review.decisions||{}, finalized=data.review.approved===true;
const $=id=>document.getElementById(id);$('reviewer').value=data.review.reviewer||'';
function status(message=''){let a=data.items.filter(x=>reviews[x.id]?.decision==='approved').length,r=data.items.filter(x=>reviews[x.id]?.decision==='rejected').length;$('status').textContent=`${a} approved · ${r} rejected · ${80-a-r} pending — ${message|| (finalized?'Dataset approval saved':'Evaluation remains blocked')}`;$('final').disabled=a!==80||!$('reviewer').value.trim();}
function el(tag,text){let e=document.createElement(tag);e.textContent=text;return e;}
function render(){$('cards').replaceChildren();for(const x of data.items){let d=reviews[x.id]||{decision:'pending',notes:''};if($('slice').value&&x.benchmarkSlice!==$('slice').value)continue;if($('filter').value&&d.decision!==$('filter').value)continue;if(!JSON.stringify(x).toLowerCase().includes($('search').value.toLowerCase()))continue;let c=el('article','');c.className=d.decision;c.append(el('div',`${x.benchmarkSlice} · ${x.difficulty} · ${x.id}`),el('p',x.prompt));let m=el('div',`Skills: ${(x.skills||[]).join(', ')} · Seed: ${x.seed}`);m.className='meta';c.append(m,el('small',`Expected response: ${x.kind==='refusal'?'Refuse the unsafe request':'Produce a drawing'}`));if(x.refusal){let det=el('details','');det.append(el('summary','Reference refusal'),el('p',x.refusal));c.append(det);}c.append(el('small',`Source: ${x.benchmarkSource}`));let note=document.createElement('textarea');note.placeholder='Review notes or requested changes';note.value=d.notes;note.oninput=()=>{reviews[x.id]={decision:d.decision,notes:note.value};d.notes=note.value;finalized=false;status('Unsaved changes');};c.append(note);let bar=el('div','');bar.className='controls';for(const [value,label] of [['approved','Approve'],['rejected','Reject / needs changes'],['pending','Reset']]){let b=el('button',label);if(value==='approved')b.className='approve';b.onclick=()=>{reviews[x.id]={decision:value,notes:note.value};finalized=false;render();status('Unsaved changes');};bar.append(b);}bar.append(el('span',d.decision));c.append(bar);$('cards').append(c);}status();}
async function save(approved){try{let response=await fetch('/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_sha256:data.sha256,reviewer:$('reviewer').value.trim(),approved,decisions:reviews})});let value=await response.json();if(!response.ok)throw Error(value.error);finalized=approved;status(approved?'Dataset approval saved. Evaluation can now be launched.':'Review progress saved; evaluation remains blocked.');}catch(e){status(e.message);}}
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
        if any(decisions.get(x['id'],{}).get('decision')!='approved' for x in items):raise ValueError('All 80 prompts must be approved')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--output',type=Path)
    p.add_argument('--dataset',type=Path,help='Full SFT candidates.jsonl for manual inspection')
    p.add_argument('--accepted-output',type=Path,help='SFT export path (default: DATASET/accepted.jsonl)')
    a=p.parse_args()
    if a.dataset:
        from drawing.manual_inspector import serve
        return serve(a.dataset, a.port, a.output, a.accepted_output)
    a.output = a.output or ROOT/'eval-results/dataset-review/approval.json'
    manifest,items=suite();a.output.parent.mkdir(parents=True,exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def respond(self,code,body,kind='application/json'):
            self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.end_headers();self.wfile.write(body.encode())
        def do_GET(self):
            if self.path!='/':return self.respond(404,'{}')
            review=json.loads(a.output.read_text()) if a.output.exists() else {}
            if review.get('dataset_sha256')!=manifest['sha256']:review={}
            payload=json.dumps({'items':items,'sha256':manifest['sha256'],'review':review}).replace('<','\\u003c')
            self.respond(200,PAGE.replace('DATA',payload),'text/html')
        def do_POST(self):
            if self.path!='/review':return self.respond(404,'{}')
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{a.port}',f'http://localhost:{a.port}'):return self.respond(403,'{}')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<200000:raise ValueError('Invalid request size')
                value=json.loads(self.rfile.read(length));validate_review(value,manifest,items)
                temp=a.output.with_suffix('.tmp');write_json(temp,value);temp.replace(a.output)
                self.respond(200,'{"saved":true}')
            except (ValueError,TypeError,AttributeError) as e:self.respond(400,json.dumps({'error':str(e)}))
    print(f'Viewer: http://127.0.0.1:{a.port} — approvals: {a.output}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
