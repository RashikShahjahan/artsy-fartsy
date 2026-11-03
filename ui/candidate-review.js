'use strict';
const data = JSON.parse(document.getElementById('review-data').textContent), reviews = JSON.parse(JSON.stringify(data.review.decisions || {}));
const $ = id => document.getElementById(id);
let current = null, changed = false, saving = false, revision = 0, acknowledgedRevision = 0;
let detailRequest = 0, loadedDetailId = null;
const cardNodes = new Map();
function markDirty(){revision++;changed=true;status();}
window.addEventListener('beforeunload', event => {if(changed){event.preventDefault();event.returnValue='';}});
$('reviewer').value = data.review.reviewer || '';
function node(tag, text, cls) { const x = document.createElement(tag); if(text!==undefined)x.textContent=text; if(cls)x.className=cls; return x; }
function count(kind) { return data.items.filter(x => (reviews[x.id] || {}).decision === kind).length; }
function status(message) {
 const a=count('approved'), r=count('rejected');
 $('overview').textContent = data.items.length + ' unfiltered candidates · ' + a + ' manually approved · ' + r + ' rejected · ' + (data.items.length-a-r) + ' pending';
 window.ArtsyUI.setStatus($('status'), message || (changed ? 'Unsaved changes' : 'Review file: ' + data.reviewPath));
 $('save').disabled=saving;
 $('export').disabled=saving;
}
function decision(id, value) {
 const previous=reviews[id]||{decision:'pending',notes:''};
 if(previous.decision===value)return;
 reviews[id]={...previous,decision:value};
 markDirty();render();
 if(current===id)for(const element of $('detail').querySelectorAll('[data-review-decision]')){
  element.setAttribute('aria-pressed',String(element.dataset.reviewDecision===value));
 }
}
function render() {
 const q=$('search').value.toLowerCase(), visible=[];
 for(const x of data.items){
  if(q&&!(x.prompt+' '+x.id+' '+x.reason+' '+x.family).toLowerCase().includes(q))continue;
  if($('automatic').value&&x.status!==$('automatic').value)continue;
  const manual=(reviews[x.id]||{}).decision||'pending';
  if($('decision').value&&manual!==$('decision').value)continue;
  let entry=cardNodes.get(x.id);
  if(!entry){
   const card=node('article');card.tabIndex=0;card.setAttribute('role','button');card.setAttribute('aria-label',x.prompt);
   const title=node('div',x.prompt),meta=node('div',undefined,'meta');card.append(title,meta);
   if(x.score!==null)card.append(node('div','Judge score: '+x.score,'meta'));
   card.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openRecord(x.id);}};
   card.onclick=()=>openRecord(x.id);
   entry={card,meta};cardNodes.set(x.id,entry);
  }
  entry.card.classList.toggle('selected',current===x.id);
  entry.card.setAttribute('aria-pressed',String(current===x.id));
  entry.meta.textContent=x.id+' · '+x.family+' · generated: '+x.status+' · manual: '+manual;
  visible.push(entry.card);
 }
 window.ArtsyUI.reconcile($('items'),visible);
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
 const zoom=button('Enlarge '+image.label,()=>window.ArtsyUI.openImage({src:image.url,alt:image.label,returnFocus:zoom}));
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
 if(record.trajectoryFormat==='inspect-bash-v1'){
  wrap.append(node('p','Native rollout recorded; inspect Conversation for actual actions and observations.','trace-summary'));
  const previews=images.filter(image=>image.label.startsWith('Native preview '));
  if(previews.length)wrap.append(imageSection(previews));
  else wrap.append(node('p','No validated native preview images are available.','trace-empty'));
  return wrap;
 }
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
 if(current===id && loadedDetailId===id)return;
 const request=++detailRequest;
 current=id;loadedDetailId=null;render();
 $('detail').replaceChildren(node('p','Loading complete record...'));
 try {
  const response=await fetch(data.basePath+'/api/record?id='+encodeURIComponent(id));const x=await response.json();
  if(!response.ok)throw Error(x.error);
  if(current!==id || request!==detailRequest)return;
  loadedDetailId=id;
  const d=reviews[id] || {decision:'pending',notes:''}, box=$('detail');box.replaceChildren();
  box.append(node('h2',x.record.spec.prompt));
  box.append(node('p','Candidate '+id+' · '+x.record.status+' · '+(x.record.reason || ''),'meta'));
  const bar=node('div');bar.className='controls';
  for(const choice of ['approved','rejected','pending']) {
   const b=node('button',choice);if(choice==='approved')b.className='primary';b.setAttribute('aria-pressed',String(d.decision===choice));
   b.dataset.reviewDecision=choice;b.onclick=()=>decision(id,choice);bar.append(b);
  }
  box.append(bar);
  const notes=node('textarea');notes.placeholder='Manual review notes';notes.setAttribute('aria-label','Manual review notes');notes.value=d.notes || '';
  notes.oninput=()=>{reviews[id]={decision:(reviews[id] || {}).decision || 'pending',notes:notes.value};markDirty();};
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
 } catch(e) {if(current===id && request===detailRequest)$('detail').replaceChildren(node('p',e.message,'bad'));}
}
async function post(path, body) {
 const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const value=await response.json();if(!response.ok)throw Error(value.error || 'Request failed');return value;
}
async function save() {
 if(saving)return;
 try{
  const reviewer=$('reviewer').value.trim();
  if(!reviewer)throw Error('Enter a reviewer name');
  const submittedRevision=revision;
  const body={dataset_sha256:data.sha256,reviewer,decisions:JSON.parse(JSON.stringify(reviews))};
  saving=true;status('Saving review decisions…');
  await post(data.basePath+'/api/review',body);
  acknowledgedRevision=Math.max(acknowledgedRevision,submittedRevision);
  changed=revision>acknowledgedRevision;
  status(changed?'Submitted revision saved; newer edits are still unsaved.':'Review decisions saved');
 }catch(error){status(error.message);}
 finally{saving=false;status();}
}
$('save').onclick=save;
$('export').onclick=async()=>{
 try {
  if(changed || saving)throw Error('Save manual decisions before exporting');
  const result=await post(data.basePath+'/api/export',{dataset_sha256:data.sha256});
  status('Exported '+result.count+' approved SFT examples to '+result.path);
 } catch(e) {status(e.message);}
};
for(const id of ['search','automatic','decision'])$(id).oninput=render;
$('reviewer').oninput=markDirty;
render();
