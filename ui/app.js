const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let active = null, expanded = false;
let drawingOnly = false;
const sessionKey = 'artsy-active-run';
let drawingData = null, selectedDrawing = null, displayedRun = null, freeformDraft = '';
let datasetSelection = null;
function setSource(source, reset=true) {
  if(active && reset)return;
  if(reset && !drawingOnly)freeformDraft=$('#prompt').value;
  if(drawingOnly && selectedDrawing)datasetSelection=selectedDrawing;
  drawingOnly=source==='dataset';
  $('#drawing-library').hidden=!drawingOnly;
  $('#drawing-history').hidden=!drawingOnly;
  $('#prompt').readOnly=drawingOnly;
  $('#prompt').placeholder=drawingOnly?'Choose a drawing prompt above':'Describe what to draw…';
  $('#generate').textContent=drawingOnly?'Run this prompt':'Generate';
  document.querySelectorAll('[data-source]').forEach(button=>button.setAttribute('aria-pressed', String((button.dataset.source==='dataset')===drawingOnly)));
  if(reset) {
    if(drawingOnly && drawingData) {filterDrawings(datasetSelection?.id);showDrawing(selectedDrawing);}
    else showDrawing(null);
    if(!drawingOnly)$('#prompt').value=freeformDraft;
  }
  updateModelAvailability();
}
document.querySelectorAll('[data-source]').forEach(button=>button.onclick=()=>setSource(button.dataset.source));
function runHash(id) {
  const params = new URLSearchParams();
  if (drawingOnly && selectedDrawing) params.set('prompt', selectedDrawing.id);
  params.set('run', id); return params.toString();
}
function renderDrawingRuns() {
  if (!drawingOnly) return;
  const runs = (drawingData?.runs || []).filter(run => run.prompt_id === selectedDrawing?.id);
  $('#drawing-runs').replaceChildren();
  if (!runs.length) {const p=document.createElement('p');p.className='note';p.textContent='No runs yet.';$('#drawing-runs').append(p);}
  for (const run of runs) {
    const button=document.createElement('button');button.type='button';
    button.textContent=`${run.model} · ${run.state} · ${run.id.slice(0,8)}`;
    button.disabled=Boolean(active);
    button.onclick=()=>{active=run.id;displayedRun=null;location.hash=runHash(active);updateModelAvailability();poll(active);};
    $('#drawing-runs').append(button);
  }
}
function showDrawing(spec, reset=true) {
  selectedDrawing=spec;
  $('#prompt').value=spec?.prompt || '';
  $('#drawing-meta').textContent=spec ? `${spec.difficulty || 'Unlabeled'} · ${(spec.skills || []).join(', ')} · Seed ${spec.seed}` : 'No matching drawings.';
  if(spec) $('#drawing-select').value=spec.id;
  if(reset) {
    displayedRun=null;traceSamples=[];expanded=false;$('#expand').textContent='Expand all';$('#trace-toolbar').hidden=true;$('#event-controls').hidden=true;$('#trace-stats').textContent='';
    $('#trace').innerHTML='<p class="empty">No run selected.</p>';$('#downloads').replaceChildren();
    $('#output').innerHTML='<p>Run this prompt to see its drawing.</p>';$('#run-status').textContent='Not started';
    $('#console').textContent='No process output.';$('#form-status').textContent='';
    location.hash=spec ? `prompt=${encodeURIComponent(spec.id)}` : '';
    try {sessionStorage.removeItem(sessionKey);} catch {}
  }
  renderDrawingRuns();updateModelAvailability();
}
function filterDrawings(preferred) {
  const query=$('#drawing-search').value.toLowerCase();
  const items=drawingData.items.filter(row=>JSON.stringify(row).toLowerCase().includes(query));
  $('#drawing-select').replaceChildren(...items.map(row=>new Option(`${row.difficulty || '—'} · ${row.prompt.slice(0,110)}`,row.id)));
  const selected=items.find(row=>row.id===(preferred || selectedDrawing?.id)) || items[0] || null;
  if(drawingOnly) showDrawing(selected, Boolean(selectedDrawing && selectedDrawing.id!==selected?.id));
  else selectedDrawing=selected;
}
async function loadDrawings() {
  const response=await fetch('/api/drawing-prompts',{cache:'no-store'});
  const data=await response.json();if(!response.ok)throw new Error(data.error || 'Could not load drawings.');
  drawingData=data;$('#dataset-label').textContent=`${data.items.length} ordinary drawing prompts · ${data.label}`;
  filterDrawings(new URLSearchParams(location.hash.slice(1)).get('prompt'));
  $('#drawing-search').oninput=()=>filterDrawings();
  $('#drawing-select').onchange=()=>showDrawing(data.items.find(row=>row.id===$('#drawing-select').value));
}
async function refreshDrawingRuns() {
  if(!drawingOnly)return;
  try {const response=await fetch('/api/drawing-prompts',{cache:'no-store'});if(!response.ok)return;
    const data=await response.json();if(data.sha256===drawingData.sha256){drawingData.runs=data.runs;renderDrawingRuns();}
  } catch {}
}
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
let traceSamples = [], traceView = 'actions', eventFilter = '';
let lazyRecords = [];
const jsonHTML = value => `<pre>${esc(JSON.stringify(value, null, 2))}</pre>`;
const statusLabels = {success: 'Succeeded', failed: 'Failed', requested: 'Requested', unknown: 'No result recorded', recorded: 'Recorded'};
function lazyJSON(value, label = 'Raw record') {
  const id = lazyRecords.push(value) - 1;
  return `<details class="raw-record" data-record="${id}"><summary>${esc(label)}</summary><div></div></details>`;
}
function modelHTML(model) {
  if (!model) return '';
  const seconds = (Date.parse(model.completed) - Date.parse(model.timestamp)) / 1000;
  const usage = model.output?.usage;
  const parts = [];
  if (Number.isFinite(seconds) && seconds >= 0) parts.push(`Model response ${seconds.toFixed(1)}s`);
  if (usage?.input_tokens != null) parts.push(`${usage.input_tokens.toLocaleString()} input tokens`);
  if (usage?.output_tokens != null) parts.push(`${usage.output_tokens.toLocaleString()} output tokens`);
  return parts.length ? `<p class="step-meta">${esc(parts.join(' · '))}</p>` : '';
}
function diffHTML(lines) {
  const keep = new Set();
  lines.forEach(([mark], index) => {if (mark !== ' ') for (let i = Math.max(0, index - 3); i <= Math.min(lines.length - 1, index + 3); i++) keep.add(i);});
  if (!keep.size) return '<p class="note">No source changes.</p>';
  let skipped = 0, html = '';
  lines.forEach(([mark, line], index) => {
    if (!keep.has(index)) {skipped++; return;}
    if (skipped) {html += `<span class="diff-same">… ${skipped} unchanged lines …</span>`; skipped = 0;}
    html += `<span class="diff-${mark === '+' ? 'add' : mark === '-' ? 'remove' : 'same'}">${esc(mark + ' ' + line)}</span>`;
  });
  if (skipped) html += `<span class="diff-same">… ${skipped} unchanged lines …</span>`;
  return `<pre class="source-diff">${html}</pre>`;
}
function actionHTML(step, index) {
  let body = ArtsyTrace.text(step.content) || Array.isArray(step.content) ? contentHTML(step.content) : '';
  if (step.call) {
    body += step.source !== null ? `<details><summary>Full command</summary><pre>${esc(step.command)}</pre></details>` : `<h4>Command / arguments</h4><pre>${esc(step.command)}</pre>`;
    if (step.source !== null) {
      if (step.previousSource !== null) {
        const lines = ArtsyTrace.diff(step.previousSource, step.source);
        body += '<h4>Changes from previous successful write</h4>';
        body += lines ? diffHTML(lines)
          : `<p class="note">Large file: compare the full sources below.</p>${lazyJSON(step.previousSource, 'Previous source')}`;
      }
      body += `<details><summary>Full source</summary><pre>${esc(step.source)}</pre></details>`;
    }
    if (step.call.parse_error) body += `<h4>Call error</h4>${jsonHTML(step.call.parse_error)}`;
    if (!step.results.length) body += `<p class="note">${step.kind === 'submit' ? 'Submission requested. Final validation is recorded separately below.' : 'No tool response was recorded for this call.'}</p>`;
    for (const result of step.results) {
      const data = ArtsyTrace.feedback(result);
      if (data) {
        body += `<h4>Render feedback${data.width && data.height ? ` · ${esc(data.width)} × ${esc(data.height)}` : ''}</h4>`;
        if (data.error || data.stderr) body += `<pre class="error-output">${esc(data.stderr || data.error)}</pre>`;
      }
      body += `<h4>Tool output</h4>${contentHTML(result.content)}`;
      if (result.error) body += jsonHTML(result.error);
    }
    for (const observation of step.observations) body += `<h4>Preview returned to the agent</h4>${contentHTML(observation.content)}`;
    body += modelHTML(step.model) + lazyJSON(step.call, 'Recorded tool call');
  }
  if (step.error) body += jsonHTML(step.error);
  const failure = step.results?.map(ArtsyTrace.feedback).find(d => d?.ok === false);
  const brief = failure ? (failure.stderr || failure.error || '').trim().split('\n').at(-1) : '';
  return `<li><details class="action-card ${step.status}" ${expanded || step.status === 'failed' ? 'open' : ''}><summary><span class="step-number">${index + 1}</span><span class="step-title">${esc(step.title)}${brief ? `<small>${esc(brief)}</small>` : ''}</span><span class="step-status">${statusLabels[step.status]}</span></summary><div class="step-body">${body}</div></details></li>`;
}
function validationHTML(sample) {
  // Scorer metadata is specific to this sample; never reuse another sample's result.
  const score = [...(sample.events || [])].reverse().find(e => e.event === 'score' && e.scorer === 'drawing_score');
  const result = score?.score?.metadata;
  const feedback = result?.feedback;
  if (!result && !sample.error) return '';
  const ok = feedback?.ok;
  const label = ok === true ? 'Final render passed' : ok === false ? 'Final render failed' : 'Run outcome';
  return `<div class="validation ${ok === false || sample.error ? 'failed' : ''}"><strong>${label}</strong><span>${esc(result?.outcome || 'Run error')}${feedback?.width && feedback?.height ? ` · ${esc(feedback.width)} × ${esc(feedback.height)}` : ''}</span>${lazyJSON(sample.error || result, 'Validation details')}</div>`;
}
function summaryHTML() {
  const steps = traceSamples.flatMap(s => s.steps).filter(s => s.call);
  const failed = steps.filter(s => s.status === 'failed').length;
  const previews = steps.reduce((n, s) => n + [...s.observations, ...s.results].reduce((count, m) => count + (Array.isArray(m.content) ? m.content.filter(p => p.type === 'image').length : 0), 0), 0);
  let input = 0, output = 0;
  for (const {sample} of traceSamples) for (const usage of Object.values(sample.model_usage || {})) {
    input += usage.input_tokens || 0; output += usage.output_tokens || 0;
  }
  return `<span>${steps.length} actions</span><span>${failed} failed</span><span>${previews} ${previews === 1 ? 'preview' : 'previews'}</span>${input || output ? `<span>${input.toLocaleString()} input / ${output.toLocaleString()} output tokens</span>` : ''}`;
}
function renderTrace(logs) {
  traceSamples = logs.flatMap(log => (log.samples || []).map(ArtsyTrace.normalizeSample));
  $('#trace-toolbar').hidden = false;
  $('#trace-stats').innerHTML = summaryHTML();
  renderTraceView();
}
function renderTraceView() {
  lazyRecords = [];
  $('#event-controls').hidden = traceView !== 'events';
  document.querySelectorAll('[data-trace-view]').forEach(button => {
    const selected = button.dataset.traceView === traceView;
    button.setAttribute('aria-pressed', String(selected));
  });
  const sections = traceSamples.map(({sample, steps, events}, sampleIndex) => {
    const heading = traceSamples.length > 1 ? `<h3 class="sample-heading">Sample ${esc(sample.id ?? sampleIndex + 1)} · epoch ${esc(sample.epoch ?? 1)}</h3>` : '';
    let body;
    if (traceView === 'actions') body = `<ol class="action-list">${steps.map(actionHTML).join('')}</ol>${validationHTML(sample)}`;
    else if (traceView === 'transcript') body = (sample.messages || []).map((message, i) =>
      `<details ${expanded ? 'open' : ''}><summary>${i + 1} · ${esc(message.role)}${message.function ? ' · ' + esc(message.function) : ''}</summary>${contentHTML(message.content || '')}${message.tool_calls?.length ? `<h4>Tool calls</h4>${jsonHTML(message.tool_calls)}` : ''}${message.error ? jsonHTML(message.error) : ''}${lazyJSON(message, 'Complete message record')}</details>`).join('');
    else {
      const matches = events.map((event, i) => ({event, i})).filter(({event}) => !eventFilter || JSON.stringify(event).toLowerCase().includes(eventFilter));
      body = `<p class="note">${matches.length} of ${events.length} events · Search includes raw fields.</p>` + matches.map(({event, i}) => {
        const id = lazyRecords.push(event) - 1;
        return `<details class="event-record" data-record="${id}" ${expanded ? 'open' : ''}><summary><span>${i + 1}</span> ${esc(event.event || 'event')}${event.name || event.scorer || event.action ? ' · ' + esc(event.name || event.scorer || event.action) : ''}<small>${esc(event.timestamp || '')}</small></summary><div>${expanded ? jsonHTML(event) : ''}</div></details>`;
      }).join('');
    }
    return heading + (body || '<p class="empty">No records in this view.</p>');
  });
  $('#trace').innerHTML = sections.join('') || '<p class="empty">No agent trace was recorded.</p>';
  if (expanded) hydrateRecords();
}
function hydrateRecords() {
  document.querySelectorAll('#trace details[data-record][open]').forEach(details => {
    if (!details.lastElementChild.childNodes.length) details.lastElementChild.innerHTML = jsonHTML(lazyRecords[Number(details.dataset.record)]);
  });
}
$('#trace').addEventListener('toggle', hydrateRecords, true);
document.querySelectorAll('[data-trace-view]').forEach(button => button.onclick = () => {
  traceView = button.dataset.traceView; renderTraceView();
});
$('#event-search').oninput = e => {eventFilter = e.target.value.toLowerCase(); renderTraceView();};
async function poll(id) {
  try {
    const response=await fetch(`/api/runs/${id}`,{cache:'no-store'});
    if(!response.ok) throw new Error('Could not read the run. Reload to retry.');
    const run=await response.json();
    if(run.prompt_id && run.dataset_sha256!==drawingData?.sha256) {
      active=null;try{sessionStorage.removeItem(sessionKey);}catch{}
      $('#form-status').textContent='This saved run belongs to another dataset revision. Select a drawing from this dataset.';
      updateModelAvailability();return;
    }
    if(active===id && displayedRun!==id) { setSource(run.prompt_id?'dataset':'freeform',false); displayedRun=id; if(drawingOnly){const spec=drawingData?.items.find(row=>row.id===run.prompt_id);if(spec)showDrawing(spec,false);displayedRun=id;} $('#prompt').value=run.prompt;
      if ([...$('#model').options].some(option => option.value === run.model)) $('#model').value=run.model; }
    $('#run-status').textContent=(run.result?.outcome || run.state) + (drawingOnly ? ` · ${run.model}` : '');
    $('#console').textContent=run.console || 'Waiting for process output…';
    $('#downloads').innerHTML=(run.artifacts || []).map(name=>`<a href="/api/runs/${id}/${name}" download="${name}">${esc(name)}</a>`).join('');
    if(run.artifacts.includes('image.png')) $('#output').innerHTML=`<img src="/api/runs/${id}/image.png" alt="Generated image">`;
    if(['queued','running'].includes(run.state)) {
      $('#trace-note').textContent='Process output updates during generation. The complete recorded agent trace appears when the run ends.';
      setTimeout(()=>poll(id),1500);return;
    }
    active=null;updateModelAvailability();refreshDrawingRuns();
    $('#trace-note').textContent='Commands, results, and previews in order. Expand any action for its full record.';
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
  e.preventDefault();if(active || (drawingOnly && !selectedDrawing))return;
  const prompt=$('#prompt').value.trim();if(!prompt){$('#form-status').textContent='Enter a prompt.';return;}
  $('#generate').disabled=true;$('#form-status').textContent='';
  try {
    const response=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(drawingOnly ? {prompt_id:selectedDrawing.id,dataset_sha256:drawingData.sha256,model:$('#model').value.trim()} : {prompt,seed:42,model:$('#model').value.trim()})});
    const data=await response.json();if(!response.ok)throw new Error(data.error || 'Could not start generation.');
    active=data.id;location.hash=runHash(active);try{sessionStorage.setItem(sessionKey,active);}catch{}
    updateModelAvailability();displayedRun=null;
    traceSamples=[]; expanded=false; $('#expand').textContent='Expand all'; $('#trace-toolbar').hidden=true; $('#trace-stats').innerHTML=''; $('#event-controls').hidden=true;
    $('#output').innerHTML='<p>Generating…</p>';$('#downloads').innerHTML='';$('#trace').innerHTML='';$('#console').textContent='Starting…';$('#run-status').textContent='Starting';
    poll(active);
  } catch(error) {$('#form-status').textContent=error.message;updateModelAvailability();}
};
$('#expand').onclick=()=>{expanded=!expanded;document.querySelectorAll('#trace details').forEach(d=>d.open=expanded);hydrateRecords();$('#expand').textContent=expanded?'Collapse all':'Expand all';};
let bridgeModels = null;
let directModels = new Set();
function updateModelAvailability() {
  const direct = directModels.has($('#model').value);
  const unavailable = !direct && (bridgeModels === null || !bridgeModels.includes($('#model').value));
  $('#model-note').textContent = direct ? 'Tinker API · same Docker drawing harness and PNG previews.' : bridgeModels === null ? 'Native Tinker bridge is offline. Start make serve, then reload.'
    : unavailable ? 'This model is unavailable on the running bridge. Restart make serve, then reload.'
    : 'Native Tinker sampling · same Docker drawing harness and PNG previews.';
  $('#generate').disabled = Boolean(active) || unavailable || (drawingOnly && (!selectedDrawing || !drawingData));
  $('#drawing-select').disabled=Boolean(active);$('#drawing-search').disabled=Boolean(active);
  document.querySelectorAll('[data-source]').forEach(button=>button.disabled=Boolean(active));
}
$('#model').onchange = updateModelAvailability;
async function loadModels() {
  try {
    const response = await fetch('/api/models', {cache:'no-store'});
    if (!response.ok) throw new Error('Could not load models');
    const {models, bridge_models} = await response.json();
    bridgeModels = bridge_models ?? null;
    directModels = new Set(models.filter(model => model.direct).map(model => model.id));
    const selected = $('#model').value;
    $('#model').replaceChildren(...models.map(model => new Option(model.label, model.id)));
    if (models.some(model => model.id === selected)) $('#model').value = selected;

    updateModelAvailability();
  } catch { $('#model-note').textContent = 'Could not load Studio models. Restart the UI server and reload.'; }
}
async function initialize() {
  await loadModels();
  try {
    const params=new URLSearchParams(location.hash.slice(1));
    setSource(params.has('prompt') || location.pathname.startsWith('/drawings')?'dataset':'freeform',false);
    try {await loadDrawings();} catch(error) {$('#dataset-label').textContent=error.message;}
    const id=new URLSearchParams(location.hash.slice(1)).get('run') || sessionStorage.getItem(sessionKey);
    if(id&&/^[a-f0-9]{32}$/.test(id)){active=id;updateModelAvailability();poll(id);}
  } catch(error) {$('#form-status').textContent=error.message;updateModelAvailability();}
}
initialize();
