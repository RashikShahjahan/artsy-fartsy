/* Display recorded agent activity, live and when reopening a saved drawing. */
(function(root) {
  'use strict';
  const labels = {running: 'Working', pending: 'Waiting', completed: 'Done', failed: 'Failed',
    interrupted: 'Stopped', refused: 'Declined', recorded: 'Recorded', requested: 'Requested', unknown: 'No result'};

  function actionDescription(step) {
    const drawingTools = {
      update_drawing: ['Refine your drawing', 'Refining your drawing…', 'Drawing updated'],
      read_drawing: ['Review your drawing', 'Reviewing your drawing…', 'Drawing reviewed'],
      preview_drawing: ['Check the preview', 'Checking the preview…', 'Preview checked'],
      submit_drawing: ['Finish your drawing', 'Finishing your drawing…', 'Drawing submitted for final checks'],
      refuse_drawing: ['Decline this request', 'Declining the request…', 'Request declined'],
    };
    const tense = step.status === 'pending' ? 0 : ['completed', 'success', 'recorded'].includes(step.status) ? 2 : 1;
    if (drawingTools[step.tool]) return drawingTools[step.tool][tense];
    // Historical drawings still have shell commands instead of named tools.
    const command = step.command || '';
    let phrases;
    if (step.kind === 'submit' || step.title === 'Submit drawing' || /COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT/.test(command))
      phrases = ['Finish your drawing', 'Finishing your drawing…', 'Drawing submitted for final checks'];
    else if (step.kind === 'render' || step.title === 'Render preview' || /\brender\.py\b/.test(command))
      phrases = ['Check the preview', 'Checking the preview…', 'Preview checked'];
    else if (/\b(?:sed|perl|patch|apply_patch)\b|replace\(/.test(command) || step.title === 'Update program.py')
      phrases = ['Refine your drawing', 'Refining your drawing…', 'Drawing refined'];
    else if (step.kind === 'write' || step.title === 'Write drawing code' || /(?:cat\s*>|write_text\(|write_bytes\(|open\([^\n]+['"]w)/.test(command))
      phrases = ['Create your drawing', 'Creating your drawing…', 'Drawing created'];
    else if (/\b(?:cat|head|tail|ls|find|grep|rg|wc)\b/.test(command))
      phrases = ['Review your drawing', 'Reviewing your drawing…', 'Drawing reviewed'];
    else if (/\b(?:python\S*|compileall|py_compile)\b/.test(command))
      phrases = ['Check your drawing', 'Checking your drawing…', 'Drawing checked'];
    else phrases = ['Continue drawing', 'Working on your drawing…', 'Drawing updated'];
    return phrases[tense];
  }

  function responseSummary(step) {
    if (step.kind !== 'model' && !(step.kind === 'message' && step.title === 'Agent message')) return step;
    const output = step.output || '';
    const heading = output.match(/^\s*(?:\*\*([^\n]+?)\*\*|#{1,6}\s+([^\n]+?)(?:\s+#+)?)(?:\s*\n|\s*$)/);
    return {...step, title: heading ? (heading[1] || heading[2]).trim()
      : /^Agent (?:turn \d+|message)$/.test(step.title) ? step.status === 'running' ? 'Thinking…' : 'Drawing update' : step.title,
      output: heading ? output.slice(heading[0].length).trimStart() : output};
  }

  function resultSummary(step) {
    if (step.status === 'pending') return 'Ready when you approve this step.';
    if (step.status === 'running') return 'This step is in progress.';
    if (step.status === 'interrupted') return 'This step stopped before it finished.';
    if (step.status === 'requested') return 'The request was sent; completion has not been confirmed.';
    if (step.status === 'unknown') return 'No result was recorded for this step.';
    if (step.status === 'recorded') return 'This action was recorded; success has not been confirmed.';
    if (step.status === 'failed') {
      const output = step.output || '';
      if (/SyntaxError|IndentationError/.test(output)) return 'The drawing code has a syntax error. Artsy may try again.';
      if (/timed out|timeout/i.test(output)) return 'This step took too long to finish. Artsy may try again.';
      if (/Python execution failed|Traceback/.test(output)) return 'The preview couldn’t render because of an error in the drawing code. Artsy may try again.';
      return 'This step couldn’t finish. Artsy may try again.';
    }
    return {
      update_drawing: 'The drawing code was saved and is ready to preview.',
      read_drawing: 'Artsy checked the current drawing code.',
      preview_drawing: 'The preview rendered successfully.',
      submit_drawing: 'The drawing was sent for final checks.',
      refuse_drawing: 'Artsy declined this request.',
    }[step.tool] || 'This step finished successfully.';
  }

  function traceSteps(trace, api) {
    const records = Array.isArray(trace) ? trace : [trace];
    return records.flatMap((record, recordIndex) => (record?.samples || []).flatMap((sample, sampleIndex) =>
      api.normalizeSample(sample).steps.filter(step => step.title !== 'Request / context').map((step, index) => ({
        id: `trace-${recordIndex}-${sampleIndex}-${index}`, title: step.title, kind: step.kind, tool: step.tool,
        status: step.status === 'success' ? 'completed' : step.status,
        command: step.tool ? undefined : step.command, arguments: step.call?.arguments, output: [api.text(step.content), ...(step.results || []).map(result => api.text(result.content))].filter(Boolean).join('\n'),
        previews: [...(step.results || []), ...(step.observations || [])].flatMap(message =>
          Array.isArray(message.content) ? message.content.filter(part => part.type === 'image').map(part => part.image) : []),
      }))));
  }

  function stepsFor(run, traceApi) {
    const terminal = !['queued', 'running'].includes(run.state);
    let steps = [...(run.progress?.steps || [])];
    if (!steps.length && run.trace && traceApi) steps = traceSteps(run.trace, traceApi);
    steps = steps.map(step => terminal && ['running', 'pending'].includes(step.status)
      ? {...step, status: 'interrupted'} : step);
    const ready = run.state === 'completed' && run.result?.outcome === 'submitted' && run.artifacts?.includes('image.png');
    const refused = run.result?.outcome === 'refused';
    if (!steps.length && !terminal) steps = [{id: 'starting', title: 'Getting ready to draw', status: 'running'}];
    if (run.progress?.phase === 'rendering' || terminal) steps.push({
      id: 'outcome', kind: 'result', title: terminal ? ready ? 'Drawing ready' : refused ? 'Request declined' : 'Drawing stopped' : 'Checking the final drawing',
      status: terminal ? ready ? 'completed' : refused ? 'refused' : 'failed' : 'running',
      output: terminal && !ready ? run.result?.refusalReason || run.result?.reason || 'The drawing could not be completed.' : '',
    });
    return steps.map(responseSummary);
  }

  class Activity {
    constructor(panel, list, status, onApprove) {
      this.panel = panel; this.list = list; this.status = status;
      this.id = null; this.rows = new Map();
      this.onApprove = onApprove;
    }
    reset(id = null) {
      this.id = id; this.rows.clear(); this.list.replaceChildren();
      this.list.scrollTop = 0;
      this.run = null;
      this.active = null;
      this.selected = 0;
      this.panel.hidden = !id; this.status.textContent = id ? 'Getting ready…' : '';
    }
    fail(message) {
      if (this.run && ['queued', 'running'].includes(this.run.state))
        this.render({...this.run, state: 'failed', result: {reason: message}});
    }
    render(run) {
      if (this.id !== run.id) this.reset(run.id);
      this.run = run;
      const steps = stepsFor(run, root.ArtsyTrace);
      this.panel.hidden = false;
      const active = ['queued', 'running'].includes(run.state);
      if (this.active !== active) {
        this.rows.clear(); this.list.replaceChildren(); this.list.scrollTop = 0;
        this.active = active;
      }
      // A tool-only model response has no displayable text. Keep the last actual
      // action on screen while the next model call runs, including when several
      // updates arrive between polls. Full history belongs to the saved drawing.
      const visibleSteps = steps.filter(step => step.kind !== 'model' || step.output || step.previews?.length);
      const current = [...visibleSteps].reverse().find(step => ['running', 'pending'].includes(step.status))
        || visibleSteps.at(-1) || steps.at(-1);
      const history = steps.filter(step => step.kind !== 'model' || step.output || step.previews?.length);
      this.selected = Math.max(0, Math.min(this.selected, history.length - 1));
      const displayed = active ? current ? [current] : [] : history.length ? [history[this.selected]] : [];
      const turn = run.progress?.turn;
      this.status.textContent = active ? run.progress?.phase === 'awaiting_approval' ? 'Your approval is needed' : turn ? `Turn ${turn}${run.progress.max_turns ? ` of ${run.progress.max_turns}` : ''}` : 'Getting ready…'
        : `${history.length} ${history.length === 1 ? 'step' : 'steps'} recorded`;
      const visible = new Set(displayed.map(step => step.id));
      for (const [id, row] of this.rows) if (!visible.has(id)) { row.node.remove(); this.rows.delete(id); }
      for (const step of displayed) {
        const signature = JSON.stringify(step);
        let row = this.rows.get(step.id);
        if (!row) {
          const node = document.createElement('li');
          row = {node}; this.rows.set(step.id, row); this.list.append(node);
          if (active) this.list.scrollTop = 0;
        }
        row.node.hidden = !active && step.kind === 'model' && !step.output && !step.previews?.length;
        const renderSignature = JSON.stringify([signature, active, run.execution_mode]);
        if (row.signature === renderSignature) continue;
        row.signature = renderSignature;
        const node = row.node;
        node.className = `activity-step ${step.status}`;
        const heading = document.createElement('div'); heading.className = 'activity-step-heading';
        const previewReady = (step.tool === 'preview_drawing' || step.kind === 'render')
          && ['completed', 'success'].includes(step.status) && step.previews?.length;
        const hasDetails = step.tool || step.command || ['tool', 'write', 'read', 'render', 'submit', 'refuse'].includes(step.kind);
        const title = document.createElement('strong');
        title.textContent = previewReady ? 'Drawing preview' : hasDetails
          ? ['pending', 'running'].includes(step.status) ? actionDescription(step)
            : step.tool || ['failed', 'interrupted', 'unknown', 'requested'].includes(step.status) ? resultSummary(step) : actionDescription(step)
          : step.title;
        const badge = document.createElement('span'); badge.className = 'activity-step-status';
        badge.textContent = labels[step.status] || 'Recorded';
        heading.append(title, badge);
        const children = [heading];
        if (step.output && (step.kind === 'model' || step.kind === 'result' || (step.kind === 'message' && step.title === 'Agent message'))) {
          const message = document.createElement('p'); message.className = 'activity-message'; message.textContent = step.output;
          children.push(message);
        }
        const images = [...new Set(step.previews || [])].filter(url => /^preview-\d+-\d+\.png$/.test(url) || /^data:image\/png;base64,/.test(url));
        for (const [index, url] of images.entries()) {
          const image = document.createElement('img'); image.className = 'activity-preview';
          image.alt = `Preview ${index + 1} from ${step.title.toLowerCase()}`;
          image.loading = 'lazy';
          image.src = url.startsWith('data:') ? url : `/api/runs/${run.id}/${url}`;
          children.push(image);
        }
        if (active && run.execution_mode === 'approval' && step.status === 'pending' && this.onApprove) {
          const approve = document.createElement('button'); approve.type = 'button'; approve.className = 'approve-action';
          approve.textContent = 'Approve action';
          approve.onclick = async () => {
            approve.disabled = true;
            const accepted = await this.onApprove(step.id);
            approve.disabled = !!accepted; approve.textContent = accepted ? 'Approved · continuing…' : 'Approve action';
          };
          children.push(approve);
        }
        node.replaceChildren(...children);
      }
      this.navigation?.remove(); this.navigation = null;
      if (!active && history.length > 1) {
        const navigation = document.createElement('li'); navigation.className = 'activity-navigation';
        navigation.hidden = false;
        const previous = document.createElement('button'), next = document.createElement('button');
        previous.type = next.type = 'button'; previous.textContent = '← Previous'; next.textContent = 'Next →';
        previous.disabled = this.selected === 0; next.disabled = this.selected === history.length - 1;
        previous.onclick = () => { this.selected--; this.render(this.run); };
        next.onclick = () => { this.selected++; this.render(this.run); };
        const count = document.createElement('span'); count.textContent = `${this.selected + 1} of ${history.length}`;
        navigation.append(previous, count, next); this.list.append(navigation); this.navigation = navigation;
      }
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {Activity, stepsFor, traceSteps, actionDescription, resultSummary};
  else root.ArtsyActivity = Activity;
})(globalThis);
