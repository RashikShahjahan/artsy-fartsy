'use strict';
/* Prompt decisions, draft revisions and keyed cards are independent of transport. */
const data=JSON.parse(document.getElementById('review-data').textContent);
const reviews=JSON.parse(JSON.stringify(data.review.decisions||{}));
const safetyOnly=data.scope==='safety';
const $=id=>document.getElementById(id);
const cards=new Map();
let finalized=(safetyOnly?data.review.safetyApproved:data.review.approved)===true;
let revision=0, acknowledgedRevision=0, changed=false, saving=false;
$('reviewer').value=data.review.reviewer||'';
$('final').textContent=safetyOnly?'Approve safety prompts':'Approve dataset for evaluation';
window.addEventListener('beforeunload',event=>{
  if(changed){event.preventDefault();event.returnValue='';}
});
function markDirty(){
  revision++;
  changed=true;
  finalized=false;
  status();
}
function status(message=''){
  const approved=data.items.filter(x=>reviews[x.id]?.decision==='approved').length;
  const rejected=data.items.filter(x=>reviews[x.id]?.decision==='rejected').length;
  const fallback=changed?'Unsaved changes':finalized?
    (safetyOnly?'Safety review approved':'Dataset approval saved'):
    (data.validated?'Dataset already validated; optional review':safetyOnly?'Safety review pending':'Evaluation remains blocked');
  window.ArtsyUI.setStatus($('status'),
    `${approved} approved · ${rejected} rejected · ${data.items.length-approved-rejected} pending — ${message||fallback}`);
  $('final').disabled=saving||!data.items.length||approved!==data.items.length||!$('reviewer').value.trim();
  $('save').disabled=saving;
}
function node(tag,text){
  const element=document.createElement(tag);
  if(text!==undefined)element.textContent=text;
  return element;
}
function cardFor(item){
  let entry=cards.get(item.id);
  if(entry)return entry;
  const card=node('article');
  card.dataset.recordId=item.id;
  card.append(node('div',`${item.benchmarkSlice} · ${item.difficulty} · ${item.id}`),node('p',item.prompt));
  const meta=node('div',`Skills: ${(item.skills||[]).join(', ')} · Seed: ${item.seed}`);
  meta.className='meta';
  card.append(meta,node('small',`Expected response: ${item.kind==='refusal'?'Refuse the unsafe request':'Produce a drawing'}`));
  if(item.refusal){
    const details=node('details');
    details.append(node('summary','Reference refusal'),node('p',item.refusal));
    card.append(details);
  }
  card.append(node('small',`Source: ${item.benchmarkSource}`));
  const notes=node('textarea');
  notes.placeholder='Review notes or requested changes';
  notes.setAttribute('aria-label','Review notes for '+item.id);
  notes.value=reviews[item.id]?.notes||'';
  notes.oninput=()=>{
    const current=reviews[item.id]||{decision:'pending',notes:''};
    reviews[item.id]={...current,notes:notes.value};
    markDirty();
  };
  card.append(notes);
  const controls=node('div');
  controls.className='controls';
  const buttons=new Map();
  for(const [value,label] of [['approved','Approve'],['rejected','Reject / needs changes'],['pending','Reset']]){
    const button=node('button',label);
    button.type='button';
    if(value==='approved')button.className='approve';
    button.onclick=()=>{
      const current=reviews[item.id]||{decision:'pending',notes:''};
      if(current.decision===value)return;
      reviews[item.id]={...current,decision:value,notes:notes.value};
      markDirty();
      render();
    };
    controls.append(button);
    buttons.set(value,button);
  }
  const summary=node('span');
  controls.append(summary);card.append(controls);
  entry={card,notes,buttons,summary};
  cards.set(item.id,entry);
  return entry;
}
function render(){
  const visible=[];
  const search=$('search').value.toLowerCase();
  for(const item of data.items){
    const decision=reviews[item.id]?.decision||'pending';
    if($('slice').value && item.benchmarkSlice!==$('slice').value)continue;
    if($('filter').value && decision!==$('filter').value)continue;
    if(search && !JSON.stringify(item).toLowerCase().includes(search))continue;
    const entry=cardFor(item);
    entry.card.className=decision;
    entry.summary.textContent=decision;
    for(const [value,button] of entry.buttons)button.setAttribute('aria-pressed',String(value===decision));
    visible.push(entry.card);
  }
  window.ArtsyUI.reconcile($('cards'),visible);
  status();
}
async function save(approved){
  if(saving)return;
  const reviewer=$('reviewer').value.trim();
  if(!reviewer){status('Enter a reviewer name');return;}
  const submittedRevision=revision;
  const body={
    dataset_sha256:data.sha256,reviewer,
    approved:!safetyOnly&&approved,
    safetyApproved:safetyOnly&&approved,
    decisions:JSON.parse(JSON.stringify(reviews))
  };
  saving=true;status('Saving review…');
  let notice='';
  try{
    const response=await fetch(data.basePath+'/review',{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)
    });
    const result=await response.json();
    if(!response.ok)throw Error(result.error||'Could not save review');
    acknowledgedRevision=Math.max(acknowledgedRevision,submittedRevision);
    changed=revision>acknowledgedRevision;
    finalized=approved&&!changed;
    notice=changed?'Saved submitted revision; newer edits are still unsaved.':
      approved?(safetyOnly?'Safety review approved.':'Dataset approval saved. Evaluation can now be launched.'):
      'Review progress saved.';
  }catch(error){notice=error.message;}
  finally{saving=false;status(notice);}
}
$('save').onclick=()=>save(false);
$('final').onclick=()=>save(true);
for(const id of ['search','slice','filter'])$(id).oninput=render;
$('reviewer').oninput=markDirty;
render();
