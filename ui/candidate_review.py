"""Local manual inspection and curation of complete Artsy SFT candidates.

Mounted in the unified UI by ui.server.
The generator's candidate file is never modified. Only reviewed, structurally
trainable episodes can be exported to accepted.jsonl for training.
"""

from __future__ import annotations

import copy
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from training.data import convert_record


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Artsy · SFT inspector</title>
<style>
body{margin:0;background:#f5f4ef;color:#23342f;font:15px system-ui}
header{padding:18px 3%;background:#173d34;color:white;position:sticky;top:0;z-index:1}
h1{margin:0 0 8px}input,select,button,textarea{font:inherit;padding:8px;border:1px solid #b7c9bf;border-radius:7px}
button{cursor:pointer}button:disabled{opacity:.5;cursor:not-allowed}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
main{display:grid;grid-template-columns:minmax(270px,38%) minmax(0,1fr);gap:16px;padding:16px}
#items{max-height:calc(100vh - 170px);overflow:auto}#detail{min-width:0}
article,.panel{background:white;border:1px solid #d5dfd8;border-radius:10px;padding:14px;margin-bottom:10px}
article{cursor:pointer}article.selected{border-color:#277c5d;box-shadow:0 0 0 2px #277c5d}
.meta{color:#577064;font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f3;padding:12px;border-radius:7px;max-height:520px;overflow:auto}
img{max-width:100%;max-height:480px;object-fit:contain;border:1px solid #ddd;background:white}
textarea{width:100%;box-sizing:border-box;min-height:65px}.good{color:#2c805e}.bad{color:#a84d38}
button.primary{background:#276b51;color:white}small{color:#63786d}
.attempt-list{list-style:none;padding:0;margin:16px 0 0;display:grid;gap:10px}.attempt-card{border:1px solid #d5dfd8;border-radius:9px;background:#fff;overflow:hidden}.attempt-card>summary{cursor:pointer;display:flex;align-items:center;gap:10px;padding:13px 15px;list-style:none}.attempt-card>summary::-webkit-details-marker{display:none}.attempt-number{background:#e8f0eb;color:#21583f;border-radius:50%;width:25px;height:25px;line-height:25px;text-align:center;font-size:12px;font-weight:700;flex:none}.attempt-title{font-weight:650;flex:1}.attempt-state{font-size:12px;padding:3px 8px;border-radius:20px;background:#edf4ef;color:#28674c}.attempt-card.failed .attempt-state{background:#fbece7;color:#9a4935}.attempt-card.incomplete .attempt-state{background:#f7f0dd;color:#745f28}.attempt-body{border-top:1px solid #e5ebe6;padding:0 15px 15px}.attempt-body h4{font-size:13px;margin:16px 0 6px}.attempt-body pre{margin:6px 0;max-height:360px;font-size:12px;line-height:1.55}.attempt-body .error-text{background:#fbece7;color:#853e2e}.attempt-meta{display:flex;flex-wrap:wrap;gap:5px 14px;font-size:12px;color:#577064;margin:10px 0}.attempt-body p{margin:6px 0;line-height:1.5}.attempt-body figure{margin:12px 0}.attempt-body img{max-height:360px}.trace-heading{margin-bottom:3px}.trace-summary{margin-top:0;color:#577064;font-size:13px}.trace-empty{color:#577064}.attempt-card details{margin:10px 0}.attempt-card details summary{cursor:pointer;font-size:12px;color:#316e52}.attempt-card details pre{max-height:260px}
@media(max-width:760px){main{display:block}#items{max-height:40vh}header{position:static}}
</style>
<header><h1>Artsy · Full SFT data inspector</h1>
<div id="overview"></div>
<div class="controls"><input id="search" placeholder="Search prompt, ID, source">
<select id="automatic"><option value="">All generation outcomes</option><option>accepted</option><option>review</option><option>rejected</option><option>infrastructure_error</option></select>
<select id="decision"><option value="">All manual decisions</option><option>pending</option><option>approved</option><option>rejected</option></select>
<input id="reviewer" placeholder="Reviewer name">
<button id="save">Save decisions</button><button class="primary" id="export">Export approved SFT</button></div>
<div id="status"></div></header>
<main><section id="items"></section><section id="detail" class="panel">Select a candidate to inspect its full trajectory.</section></main>
<script>
const data = __DATA__, reviews = data.review.decisions || {};
const $ = id => document.getElementById(id);
let current = null, changed = false;
window.addEventListener('beforeunload', event => {if(changed){event.preventDefault();event.returnValue='';}});
$('reviewer').value = data.review.reviewer || '';
function node(tag, text, cls) { const x = document.createElement(tag); if(text!==undefined)x.textContent=text; if(cls)x.className=cls; return x; }
function count(kind) { return data.items.filter(x => (reviews[x.id] || {}).decision === kind).length; }
function status(message) {
 const a=count('approved'), r=count('rejected');
 $('overview').textContent = data.items.length + ' unfiltered candidates · ' + a + ' manually approved · ' + r + ' rejected · ' + (data.items.length-a-r) + ' pending';
 $('status').textContent = message || (changed ? 'Unsaved changes' : 'Review file: ' + data.reviewPath);
}
function decision(id, value) {
 reviews[id] = {decision:value,notes:(reviews[id] || {}).notes || ''};
 changed=true; render(); status();
}
function render() {
 $('items').replaceChildren();
 const q=$('search').value.toLowerCase();
 for(const x of data.items) {
  if(q && !(x.prompt+' '+x.id+' '+x.reason+' '+x.family).toLowerCase().includes(q))continue;
  if($('automatic').value && x.status!==$('automatic').value)continue;
  const d=(reviews[x.id] || {}).decision || 'pending';
  if($('decision').value && d!==$('decision').value)continue;
  const c=node('article');if(current===x.id)c.className='selected';c.tabIndex=0;c.setAttribute('role','button');c.setAttribute('aria-label',x.prompt);c.setAttribute('aria-pressed',String(current===x.id));c.onkeydown=e=>{if(e.key==='Enter' || e.key===' '){e.preventDefault();openRecord(x.id);}};
  c.append(node('div',x.prompt),node('div',x.id+' · '+x.family+' · generated: '+x.status+' · manual: '+d,'meta'));
  if(x.score!==null)c.append(node('div','Judge score: '+x.score,'meta'));
  c.onclick=()=>openRecord(x.id);$('items').append(c);
 }
 status();
}
function button(text, action) {
 const b=node('button',text);b.type='button';b.onclick=action;return b;
}
function codeView(value) {
 const text=typeof value==='string'?value:JSON.stringify(value,null,2);
 const wrap=node('div',undefined,'code-view'), bar=node('div',undefined,'controls'), pre=node('pre',text);
 const toggle=button('Wrap lines',()=>{const on=toggle.getAttribute('aria-pressed')!=='true';toggle.setAttribute('aria-pressed',String(on));pre.classList.toggle('no-wrap',!on);});
 toggle.setAttribute('aria-pressed','true');
 const copy=button('Copy',async()=>{try{await navigator.clipboard.writeText(text);copy.textContent='Copied';}catch{copy.textContent='Copy unavailable';}});
 bar.append(toggle,copy);wrap.append(bar,pre);return wrap;
}
function valueView(value) {
 if(value===null || typeof value!=='object')return node('p',String(value),'field-value');
 const list=node('div',undefined,'field-list');
 for(const [key,val] of Object.entries(value)) {
  if(val!==null && typeof val==='object') {
   const d=node('details'), summary=node('summary',key+' · '+Object.keys(val).length+' '+(Array.isArray(val)?'items':'fields'));
   let loaded=false;d.append(summary);d.ontoggle=()=>{if(d.open && !loaded){loaded=true;d.append(valueView(val));}};list.append(d);
  } else {
   const field=node('div',undefined,'record-field');field.append(node('strong',key),typeof val==='string' && val.includes('\\n')?codeView(val):node('span',String(val)));list.append(field);
  }
 }
 return list;
}
function section(title, value) {
 const d=node('details');d.append(node('summary',title),valueView(value));return d;
}
function tabs(entries) {
 const wrap=node('div'), bar=node('div',undefined,'candidate-tabs');bar.setAttribute('role','tablist');bar.setAttribute('aria-label','Candidate details');
 const buttons=[], panels=[];
 entries.forEach(([title,content],i)=>{
  const panel=node('section');panel.id='candidate-panel-'+i;panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby','candidate-tab-'+i);panel.tabIndex=0;panel.append(content);
  const tab=button(title,()=>select(i));tab.id='candidate-tab-'+i;tab.setAttribute('role','tab');tab.setAttribute('aria-controls',panel.id);
  tab.onkeydown=e=>{let next;if(e.key==='ArrowRight')next=(i+1)%entries.length;if(e.key==='ArrowLeft')next=(i+entries.length-1)%entries.length;if(e.key==='Home')next=0;if(e.key==='End')next=entries.length-1;if(next!==undefined){e.preventDefault();select(next);buttons[next].focus();}};
  buttons.push(tab);panels.push(panel);bar.append(tab);
 });
 function select(index){buttons.forEach((b,i)=>{b.setAttribute('aria-selected',String(i===index));b.tabIndex=i===index?0:-1;panels[i].hidden=i!==index;});}
 wrap.append(bar,...panels);select(0);return wrap;
}
function conversationView(example) {
 const wrap=node('div');
 if(!example){wrap.append(node('p','No converted conversation is available for this candidate.'));return wrap;}
 const label=node('label','Show messages '), filter=node('select');filter.setAttribute('aria-label','Filter conversation by role');
 for(const role of ['all',...new Set(example.messages.map(m=>m.role))]){const option=node('option',role);option.value=role;filter.append(option);}label.append(filter);
 const list=node('div');
 function renderMessages(){list.replaceChildren();example.messages.forEach((message,index)=>{
  if(filter.value!=='all' && filter.value!==message.role)return;
  const card=node('details',undefined,'message-card');card.open=message.role!=='system';
  card.append(node('summary',(index+1)+'. '+message.role+(example.trainable_message_indices?.includes(index)?' · supervised action':'')));
  if(typeof message.content==='string')card.append(codeView(message.content));
  else for(const part of message.content || []){if(part.type==='text' && part.text)card.append(codeView(part.text));else if(part.type!=='text')card.append(valueView(part));}
  for(const call of message.tool_calls || []){card.append(node('h4','Tool call: '+(call.function?.name || call.id)),valueView(call.function?.arguments || call));}
  if(message.tool_call_id)card.append(node('p','Response to '+message.tool_call_id,'meta'));
  list.append(card);
 });}
 filter.onchange=renderMessages;wrap.append(label,list,section('Conversation metadata',Object.fromEntries(Object.entries(example).filter(([key])=>key!=='messages'))));renderMessages();return wrap;
}
function imageSection(images) {
 const group=node('div');
 for(const image of images) {const d=node('figure');d.append(node('figcaption',image.label,'meta'));const i=document.createElement('img');i.src=image.url;i.alt=image.label;i.loading='lazy';
 const zoom=button('Enlarge '+image.label,()=>{const dialog=node('dialog',undefined,'image-dialog'), full=document.createElement('img');full.src=image.url;full.alt=image.label;dialog.setAttribute('aria-label',image.label);dialog.append(button('Close preview',()=>dialog.close()),full);dialog.onclose=()=>{dialog.remove();zoom.focus();};document.body.append(dialog);dialog.showModal();});
 d.append(i,zoom);group.append(d);}
 return group;
}
function appendText(parent, heading, value, cls) {
 if(value===undefined || value===null || value==='')return;
 const content=codeView(value);if(cls)content.classList.add(cls);parent.append(node('h4',heading),content);
}
function appendReason(parent, heading, value) {
 if(!value)return;
 const block=node('div');block.append(node('strong',heading+' '),node('span',value));parent.append(block);
}
function attemptTimeline(record, images) {
 const trace=Array.isArray(record.trace)?record.trace:[];
 const wrap=node('section');wrap.append(node('h3','Generation trace','trace-heading'));
 if(!trace.length){wrap.append(node('p','No generation attempts were recorded.','trace-empty'));return wrap;}
 const rendered=trace.filter(step=>step.feedback?.ok).length;
 const failed=trace.filter(step=>step.feedback?.ok===false || step.error).length;
 wrap.append(node('p',trace.length+' attempt'+(trace.length===1?'':'s')+' · '+rendered+' rendered · '+failed+' with errors','trace-summary'));
 const controls=node('div',undefined,'controls'), list=node('ol');list.className='attempt-list';
 controls.append(button('Expand all attempts',()=>list.querySelectorAll('.attempt-card').forEach(c=>c.open=true)),button('Collapse all attempts',()=>list.querySelectorAll('.attempt-card').forEach(c=>c.open=false)));
 const errors=button('Errors only',()=>{const active=errors.getAttribute('aria-pressed')!=='true';errors.setAttribute('aria-pressed',String(active));list.querySelectorAll(':scope > li').forEach(li=>li.hidden=active && !li.querySelector('.failed'));});errors.setAttribute('aria-pressed','false');controls.append(errors);wrap.append(controls);
 trace.forEach((step,index)=>{
  const feedback=step.feedback || null, completion=step.completion || {};
  const state=feedback?.ok===false || step.error?'failed':feedback?.ok===true?'success':'incomplete';
  const label=feedback?.ok===true?(feedback.outcome==='refused'?'Refused':'Rendered'):feedback?.ok===false?'Render failed':step.error?'Generation failed':'No render';
  const item=node('li'), card=node('details');card.className='attempt-card '+state;
  card.open=trace.length===1 || index===trace.length-1 || state==='failed';
  const summary=node('summary');summary.append(node('span',index+1,'attempt-number'),node('span','Attempt '+(index+1),'attempt-title'),node('span',label,'attempt-state'));card.append(summary);
  const body=node('div');body.className='attempt-body';
  const meta=node('div');meta.className='attempt-meta';
  if(completion.model)meta.append(node('span','Model: '+completion.model));
  if(completion.usage){const usage=completion.usage;const parts=[];if(usage.prompt_tokens!=null)parts.push(usage.prompt_tokens+' input tokens');if(usage.completion_tokens!=null)parts.push(usage.completion_tokens+' output tokens');if(parts.length)meta.append(node('span',parts.join(' · ')));}
  if(feedback?.elapsedMs!=null)meta.append(node('span','Render: '+feedback.elapsedMs+' ms'));
  if(feedback?.width && feedback?.height)meta.append(node('span',feedback.width+' × '+feedback.height));
  if(meta.childNodes.length)body.append(meta);
  appendText(body,'Generation error',step.error,'error-text');
  appendText(body,'Render error',feedback?.error || feedback?.stderr,'error-text');
  if(feedback)body.append(node('p','Render outcome: '+(feedback.outcome || (feedback.ok?'ok':'failed'))+(feedback.exitCode!=null?' · exit code '+feedback.exitCode:''),'meta'));
  const attemptImages=images.filter(image=>image.label.startsWith('Attempt '+(index+1)+' '));
  if(attemptImages.length)body.append(imageSection(attemptImages));
  appendText(body,'Executed code',step.executedCode);
  const evaluation=step.evaluation || {};
  if(evaluation.correctness || evaluation.safety){
   const score=evaluation.correctness?.score;
   if(score!=null)body.append(node('p','Correctness score: '+score,'meta'));
   appendReason(body,'Correctness:',evaluation.correctness?.reason);
   appendReason(body,'Safety:',evaluation.safety?.reason);
  }
  appendText(body,'Tool output',feedback?.stdout);
  if(completion.text){const raw=node('details');raw.append(node('summary','Full model response'),codeView(completion.text));body.append(raw);}
  body.append(section('Inspect attempt fields',step));
  card.append(body);item.append(card);list.append(item);
 });
 wrap.append(list);return wrap;
}
async function openRecord(id) {
 current=id;render();$('detail').replaceChildren(node('p','Loading complete record...'));
 try {
  const response=await fetch('/api/record?id='+encodeURIComponent(id));const x=await response.json();
  if(!response.ok)throw Error(x.error);
  if(current!==id)return;
  const d=reviews[id] || {decision:'pending',notes:''}, box=$('detail');box.replaceChildren();
  box.append(node('h2',x.record.spec.prompt));
  box.append(node('p','Candidate '+id+' · '+x.record.status+' · '+(x.record.reason || ''),'meta'));
  const bar=node('div');bar.className='controls';
  for(const choice of ['approved','rejected','pending']) {
   const b=node('button',choice);if(choice==='approved')b.className='primary';b.setAttribute('aria-pressed',String(d.decision===choice));
   b.onclick=()=>{decision(id,choice);openRecord(id);};bar.append(b);
  }
  box.append(bar);
  const notes=node('textarea');notes.placeholder='Manual review notes';notes.setAttribute('aria-label','Manual review notes');notes.value=d.notes || '';
  notes.oninput=()=>{reviews[id]={decision:(reviews[id] || {}).decision || 'pending',notes:notes.value};changed=true;status();};
  box.append(notes);
  const overview=node('div');
  const finalImage=x.images.filter(image=>image.label==='Final rendered image');
  if(finalImage.length)overview.append(node('h3','Final image'),imageSection(finalImage));

  const otherImages=x.images.filter(image=>!image.label.startsWith('Attempt ') && image.label!=='Final rendered image');
  if(otherImages.length)overview.append(node('h3','Other judge renders'),imageSection(otherImages));
  if(x.conversionError)overview.append(node('p','Not SFT-convertible: '+x.conversionError,'bad'));
  else overview.append(node('p','Structurally SFT-convertible after manual approval.','good'));
  if(x.record.refusal)overview.append(node('h3','Refusal'),node('p',x.record.refusal));
  const fields=node('div'), raw=node('details');raw.append(node('summary','Raw JSON'),codeView(x.record));fields.append(valueView(x.record),raw);
  box.append(tabs([['Overview',overview],['Attempts',attemptTimeline(x.record,x.images)],['Conversation',conversationView(x.example)],['Record fields',fields]]));
 } catch(e) {if(current===id)$('detail').replaceChildren(node('p',e.message,'bad'));}
}
async function post(path, body) {
 const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const value=await response.json();if(!response.ok)throw Error(value.error || 'Request failed');return value;
}
async function save() {
 try {
  if(!$('reviewer').value.trim())throw Error('Enter a reviewer name');
  await post('/api/review',{dataset_sha256:data.sha256,reviewer:$('reviewer').value.trim(),decisions:reviews});
  changed=false;status('Review decisions saved');
 } catch(e) {status(e.message);}
}
$('save').onclick=save;
$('export').onclick=async()=>{
 try {
  if(changed)throw Error('Save manual decisions before exporting');
  const result=await post('/api/export',{dataset_sha256:data.sha256});
  status('Exported '+result.count+' approved SFT examples to '+result.path);
 } catch(e) {status(e.message);}
};
for(const id of ['search','automatic','decision'])$(id).oninput=render;
$('reviewer').oninput=()=>{changed=true;status();};
render();
</script></html>"""


def load_candidates(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    records = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL record at line {line_number}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not isinstance(record.get("spec"), dict):
            raise ValueError(f"Malformed candidate at line {line_number}")
        if not isinstance(record["spec"].get("prompt"), str):
            raise ValueError(f"Missing prompt at line {line_number}")
        records.append(record)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Candidate file is empty or contains duplicate IDs")
    return records, hashlib.sha256(raw).hexdigest()


def validate_review(value: dict, sha: str, records: list[dict]) -> None:
    if not isinstance(value, dict) or value.get("dataset_sha256") != sha:
        raise ValueError("Review is for a different candidate dataset revision")
    if not isinstance(value.get("reviewer"), str) or not value["reviewer"].strip():
        raise ValueError("A reviewer name is required")
    decisions = value.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) - {r["id"] for r in records}:
        raise ValueError("Review contains unknown candidate IDs")
    for d in decisions.values():
        if not isinstance(d, dict) or d.get("decision") not in ("pending", "approved", "rejected") or not isinstance(d.get("notes", ""), str):
            raise ValueError("Invalid manual review decision")


def approved_record(record: dict, reviewer: str, decision: dict) -> dict:
    result = copy.deepcopy(record)
    result["automaticStatus"] = result["status"]
    result["automaticReason"] = result.get("reason")
    result["status"] = "accepted"
    result["humanReviewed"] = True
    result["manualReview"] = {"reviewer": reviewer, "notes": decision.get("notes", "")}
    return result


def export_approved(records: list[dict], review: dict, root: Path, output: Path) -> int:
    if output.resolve() == (root / "candidates.jsonl").resolve():
        raise ValueError("Refusing to overwrite the unfiltered candidate file")
    selected = []
    for record in records:
        decision = review["decisions"].get(record["id"], {})
        if decision.get("decision") != "approved":
            continue
        curated = approved_record(record, review["reviewer"], decision)
        try:
            convert_record(curated, root)
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            raise ValueError(f"{record['id']} cannot be exported as SFT: {exc}") from exc
        selected.append(curated)
    if not selected:
        raise ValueError("No approved candidates to export")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in selected), encoding="utf-8")
    os.replace(temp, output)
    return len(selected)


def image_paths(record: dict, root: Path) -> list[tuple[str, Path]]:
    images = []
    artifacts = record.get("artifacts") or {}
    def add(label: str, relative: str | None) -> None:
        if not relative:
            return
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or path.suffix.lower() != ".png":
            return
        if path not in (p for _, p in images):
            images.append((label, path))
    for i, trace in enumerate(record.get("trace") or []):
        feedback = trace.get("feedback") or {}
        if feedback.get("ok") and feedback.get("artifact") and artifacts.get("directory"):
            add(f"Attempt {i + 1} preview", str(Path(artifacts["directory"]) / Path(feedback["artifact"]).name))
        observation = trace.get("observation") or {}
        add(f"Attempt {i + 1} observation", observation.get("image"))
    for i, evaluation in enumerate(record.get("evaluations") or []):
        add(f"Judge render {i + 1}", evaluation.get("artifact"))
    add("Final rendered image", artifacts.get("image"))
    return images


def make_handler(path: Path, port: int = 4173, review_path: Path | None = None,
                 accepted_output: Path | None = None, *, base_path="", page_transform=None):
    path = path.resolve()
    root = path.parent
    records, sha = load_candidates(path)
    by_id = {record["id"]: record for record in records}
    review_path = (review_path or root / "manual-review.json").resolve()
    accepted_output = (accepted_output or root / "accepted.jsonl").resolve()
    if review_path == path or accepted_output == path or review_path == accepted_output:
        raise ValueError("Candidate, review and accepted-output paths must be different")

    def read_review() -> dict:
        if not review_path.exists():
            return {"dataset_sha256": sha, "reviewer": "", "decisions": {}}
        value = json.loads(review_path.read_text(encoding="utf-8"))
        if value.get("dataset_sha256") != sha:
            raise ValueError("Existing manual review is for a different candidate file")
        validate_review(value, sha, records)
        return value

    review = read_review()
    summaries = []
    for record in records:
        score = next((evaluation.get("correctness", {}).get("result", {}).get("score")
                      for evaluation in reversed(record.get("evaluations") or [])
                      if evaluation.get("correctness", {}).get("result")), None)
        summaries.append({
            "id": record["id"], "prompt": record["spec"]["prompt"],
            "family": record["spec"].get("family", ""),
            "status": record.get("status", "unknown"), "reason": record.get("reason", ""),
            "score": score,
        })

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, status: int, value: dict) -> None:
            self.respond(status, json.dumps(value, ensure_ascii=False).encode("utf-8"))

        def do_GET(self) -> None:
            target = urlsplit(self.path)
            if target.path == "/":
                payload = {"items": summaries, "review": review, "sha256": sha, "reviewPath": str(review_path)}
                page = PAGE.replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
                if base_path:
                    page = page.replace("'/api/", "'" + base_path + "/api/")
                if page_transform:
                    page = page_transform(page)
                return self.respond(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if target.path == "/api/record":
                rid = parse_qs(target.query).get("id", [""])[0]
                record = by_id.get(rid)
                if record is None:
                    return self.json_response(404, {"error": "Unknown candidate ID"})
                pairs = image_paths(record, root)
                images = [{"label": label, "url": base_path + "/image/" + rid + "/" + str(i)}
                          for i, (label, _) in enumerate(pairs)]
                try:
                    example = convert_record(approved_record(record, "inspection", {}), root)
                    error = None
                except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
                    example, error = None, str(exc)
                return self.json_response(200, {"record": record, "images": images,
                                                "example": example, "conversionError": error})
            if target.path.startswith("/image/"):
                pieces = target.path.split("/")
                if len(pieces) != 4 or pieces[2] not in by_id or not pieces[3].isdigit():
                    return self.json_response(404, {"error": "Unknown image"})
                pairs = image_paths(by_id[pieces[2]], root)
                index = int(pieces[3])
                if index >= len(pairs):
                    return self.json_response(404, {"error": "Unknown image"})
                return self.respond(200, pairs[index][1].read_bytes(), "image/png")
            return self.json_response(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                return self.json_response(403, {"error": "Invalid origin"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4_000_000:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict) or value.get("dataset_sha256") != sha:
                    raise ValueError("Candidate dataset changed")
                if self.path == "/api/review":
                    validate_review(value, sha, records)
                    temp = review_path.with_name(review_path.name + ".tmp")
                    review_path.parent.mkdir(parents=True, exist_ok=True)
                    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    os.replace(temp, review_path)
                    review.clear()
                    review.update(value)
                    return self.json_response(200, {"saved": True})
                if self.path == "/api/export":
                    latest = read_review()
                    count = export_approved(records, latest, root, accepted_output)
                    return self.json_response(200, {"count": count, "path": str(accepted_output)})
                return self.json_response(404, {"error": "Not found"})
            except (ValueError, TypeError, OSError, KeyError) as exc:
                return self.json_response(400, {"error": str(exc)})

    return Handler

