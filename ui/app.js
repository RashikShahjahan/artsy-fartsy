const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let active = null, expanded = false;
function contentHTML(content) {
  if (typeof content === 'string') return `<pre>${esc(content)}</pre>`;
  if (!Array.isArray(content)) return `<pre>${esc(JSON.stringify(content,null,2))}</pre>`;
  return content.map(part => {
    if(part.type === 'image') {
      const src = part.image;
      if(typeof src === 'string' && /^data:image\/(png|jpeg|webp);base64,/.test(src)) return `<img class="trace-image" src="${esc(src)}" alt="Agent preview">`;
    }
    return `<pre>${esc(part.text ?? JSON.stringify(part,null,2))}</pre>`;
  }).join('');
}
function renderTrace(logs) {
  const entries=[];
  for(const log of logs) for(const sample of log.samples || []) {
    for(const [index,message] of (sample.messages || []).entries()) {
      let body=contentHTML(message.content || '');
      if(message.tool_calls) body+=`<h4>Tool calls</h4><pre>${esc(JSON.stringify(message.tool_calls,null,2))}</pre>`;
      if(message.error) body+=`<pre>${esc(JSON.stringify(message.error,null,2))}</pre>`;
      entries.push(`<details ${expanded?'open':''}><summary><span>${index+1}</span> ${esc(message.role)}${message.function?' · '+esc(message.function):''}</summary>${body}</details>`);
    }
    entries.push(`<details ${expanded?'open':''}><summary>Recorded events</summary><pre>${esc(JSON.stringify(sample.events || [],null,2))}</pre></details>`);
  }
  $('#trace').innerHTML=entries.join('') || '<p class="empty">No agent messages were recorded.</p>';
}
async function poll(id) {
  try {
    const response=await fetch(`/api/runs/${id}`,{cache:'no-store'});
    if(!response.ok) throw new Error('Could not read the run. Reload to retry.');
    const run=await response.json();
    if(active===id && !$('#prompt').value) { $('#prompt').value=run.prompt; $('#seed').value=run.seed; $('#model').value=run.model; }
    $('#run-status').textContent=run.result?.outcome || run.state;
    $('#console').textContent=run.console || 'Waiting for process output…';
    $('#downloads').innerHTML=(run.artifacts || []).map(name=>`<a href="/api/runs/${id}/${name}" download="${name}">${esc(name)}</a>`).join('');
    if(run.artifacts.includes('image.png')) $('#output').innerHTML=`<img src="/api/runs/${id}/image.png" alt="Generated image">`;
    if(['queued','running'].includes(run.state)) {
      $('#trace-note').textContent='Process output updates during generation. The complete recorded agent trace appears when the run ends.';
      setTimeout(()=>poll(id),1500);return;
    }
    $('#generate').disabled=false;active=null;
    $('#trace-note').textContent='Recorded messages and events from this run.';
    if(run.trace) renderTrace(run.trace);
    else $('#trace').innerHTML='<p class="empty">No agent trace was produced. See process output below.</p>';
    if(run.state==='failed') {
      $('#form-status').textContent=run.error || 'Generation failed. See process output for details.';
      $('#console-panel').open=true;
      if(!run.artifacts.includes('image.png')) $('#output').innerHTML='<p>No image produced.</p>';
    } else if(!run.artifacts.includes('image.png')) $('#output').innerHTML=`<p>${esc(run.result?.reason || 'The run ended without an image.')}</p>`;
  } catch(error) {
    $('#form-status').textContent=error.message;
    $('#run-status').textContent='Connection lost';
    // Keep generation disabled: the job may still be running. Reload resumes polling.
  }
}
$('#generation-form').onsubmit=async e=>{
  e.preventDefault();if(active)return;
  const prompt=$('#prompt').value.trim();if(!prompt){$('#form-status').textContent='Enter a prompt.';return;}
  $('#generate').disabled=true;$('#form-status').textContent='';
  try {
    const response=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,seed:Number($('#seed').value),model:$('#model').value.trim()})});
    const data=await response.json();if(!response.ok)throw new Error(data.error || 'Could not start generation.');
    active=data.id;try{sessionStorage.setItem('artsy-active-run',active);}catch{}
    $('#output').innerHTML='<p>Generating…</p>';$('#downloads').innerHTML='';$('#trace').innerHTML='';$('#console').textContent='Starting…';$('#run-status').textContent='Starting';
    poll(active);
  } catch(error) {$('#form-status').textContent=error.message;$('#generate').disabled=false;}
};
$('#expand').onclick=()=>{expanded=!expanded;document.querySelectorAll('#trace details').forEach(d=>d.open=expanded);$('#expand').textContent=expanded?'Collapse all':'Expand all';};
try{const id=sessionStorage.getItem('artsy-active-run');if(id&&/^[a-f0-9]{32}$/.test(id)){active=id;$('#generate').disabled=true;poll(id);}}catch{}
