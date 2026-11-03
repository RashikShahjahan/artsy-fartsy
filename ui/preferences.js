'use strict';
const el=id=>document.getElementById(id);
let name='', packet=null, index=0;
const rankDrafts = new Map();
let rankRevision = 0, rankSaving = false;
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await r.json();if(!r.ok)throw new Error(data.error||'Request failed');return data;}
function status(text){el('status').textContent=text;}
function progress(){const done=packet.cases.filter(c=>c.ranks).length;el('reveal').disabled=done!==packet.cases.length||!done||rankDrafts.size>0||rankSaving;status(`${done} / ${packet.cases.length} unique drawing prompts ranked.`);}
function show(){progress();const c=packet.cases[index];if(!c){status('No matched images available for these evaluations.');el('review').hidden=true;return;}el('prompt').textContent=c.prompt;el('position').textContent=`Comparison ${index+1} of ${packet.cases.length}`;el('previous').disabled=index===0;el('next').disabled=index===packet.cases.length-1;el('images').replaceChildren();
 for(let slot=0;slot<3;slot++){const article=document.createElement('article'),title=document.createElement('h3'),button=document.createElement('button'),img=document.createElement('img'),label=document.createElement('label'),select=document.createElement('select');title.textContent=`Image ${'ABC'[slot]}`;img.alt=`Anonymous image ${'ABC'[slot]}`;img.src=`/api/preferences/image/${c.id}/${slot}?reviewer=${encodeURIComponent(name)}`;button.type='button';button.setAttribute('aria-label',`Enlarge image ${'ABC'[slot]}`);if(c.available[slot]){button.append(img);}else{button.textContent='No valid saved image available';button.disabled=true;}button.onclick=()=>{el('zoom').querySelector('img').src=img.src;el('zoom').showModal();};label.textContent='Rank';select.id=`rank-${slot}`;select.disabled=packet.sealed;for(const [value,text] of [['','Choose…'],['1','1 · best'],['2','2'],['3','3 · worst']]){const o=document.createElement('option');o.value=value;o.textContent=text;select.append(o);}const draft=rankDrafts.get(c.id);select.value=draft?draft.ranks[slot]:(c.ranks?String(c.ranks[slot]):'');select.onchange=()=>{rankDrafts.set(c.id,{revision:++rankRevision,ranks:[0,1,2].map(i=>el(`rank-${i}`).value)});status('Unsaved ranking changes');};label.append(select);article.append(title,button,label);el('images').append(article);}el('save').disabled=packet.sealed||rankSaving;}
el('start').onsubmit=async e=>{e.preventDefault();try{name=el('reviewer').value.trim();packet=await api(`/api/preferences?reviewer=${encodeURIComponent(name)}`);rankDrafts.clear();localStorage.setItem('artsy-preference-reviewer',name);index=Math.max(0,packet.cases.findIndex(c=>!c.ranks));el('review').hidden=false;el('results').hidden=true;show();}catch(e){status(e.message);}};
el('previous').onclick=()=>{if(index>0){index--;show();}};el('next').onclick=()=>{if(index<packet.cases.length-1){index++;show();}};el('unranked').onclick=()=>{const found=packet.cases.findIndex(c=>!c.ranks);if(found>=0){index=found;show();}};
async function saveRanking() {
  if(rankSaving || !packet || packet.sealed) return;
  const reviewedPacket=packet, reviewer=name, selected=packet.cases[index];
  if(!selected)return;
  const caseId=selected.id, selectedIndex=index;
  const draft=rankDrafts.get(caseId);
  const revision=draft?.revision ?? 0;
  const ranks=(draft ? [...draft.ranks] : [0,1,2].map(i=>el(`rank-${i}`).value)).map(Number);
  if(ranks.some(rank=>!Number.isInteger(rank)||rank<1||rank>3)){status('Choose a rank from 1 to 3 for each image.');return;}
  rankSaving=true;el('save').disabled=true;
  try {
    await api('/api/preferences/rank',{reviewer,case:caseId,ranks});
    if(packet!==reviewedPacket || name!==reviewer)return;
    const submittedCase=packet.cases.find(c=>c.id===caseId);
    if(submittedCase)submittedCase.ranks=ranks;
    if(rankDrafts.get(caseId)?.revision===revision)rankDrafts.delete(caseId);
    if(packet.cases[index]?.id===caseId && !rankDrafts.has(caseId)){
      const next=packet.cases.findIndex((c,i)=>i>selectedIndex&&!c.ranks);
      const remaining=packet.cases.findIndex(c=>!c.ranks);
      index=next>=0?next:remaining>=0?remaining:index;
      show();
    } else {
      progress();
      if(rankDrafts.size)status('Saved submitted ranking; newer edits are still unsaved.');
    }
  } catch(error) {status(error.message);}
  finally {rankSaving=false;el('save').disabled=Boolean(packet?.sealed);progress();}
}
el('save').onclick=saveRanking;
el('reveal').onclick=async()=>{try{const result=await api('/api/preferences/reveal',{reviewer:name});packet.sealed=true;show();el('results').hidden=false;el('score-table').replaceChildren();const table=document.createElement('table');const header=table.createTHead().insertRow();for(const text of ['Model','Preference score / 100']){const th=document.createElement('th');th.textContent=text;header.append(th);}const body=table.createTBody();for(const model of result.models.sort((a,b)=>b.preference_score-a.preference_score)){const row=body.insertRow();row.insertCell().textContent=model.model;row.insertCell().textContent=model.preference_score.toFixed(1);}el('score-table').append(table);status(`Review complete · ${result.comparisons} comparisons across ${result.prompts} prompts. Scores saved.`);}catch(e){status(e.message);}};
el('close-zoom').onclick=()=>el('zoom').close();
try{el('reviewer').value=localStorage.getItem('artsy-preference-reviewer')||'';}catch{}
