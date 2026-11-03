/* Display recorded agent activity, live and when reopening a saved drawing. */
(function(root) {
  'use strict';
  const labels = {running: 'Working', pending: 'Waiting', completed: 'Done', failed: 'Failed',
    interrupted: 'Stopped', refused: 'Declined', recorded: 'Recorded', requested: 'Requested', unknown: 'No result'};

  function responseSummary(step) {
    if (step.kind !== 'model' && !(step.kind === 'message' && step.title === 'Agent message')) return step;
    const output = step.output || '';
    const heading = output.match(/^\s*(?:\*\*([^\n]+?)\*\*|#{1,6}\s+([^\n]+?)(?:\s+#+)?)(?:\s*\n|\s*$)/);
    return {...step, title: heading ? (heading[1] || heading[2]).trim()
      : /^Agent (?:turn \d+|message)$/.test(step.title) ? step.status === 'running' ? 'Thinking…' : 'Drawing update' : step.title,
      output: heading ? output.slice(heading[0].length).trimStart() : output};
  }

  function traceSteps(trace, api) {
    const records = Array.isArray(trace) ? trace : [trace];
    return records.flatMap((record, recordIndex) => (record?.samples || []).flatMap((sample, sampleIndex) =>
      api.normalizeSample(sample).steps.filter(step => step.title !== 'Request / context').map((step, index) => ({
        id: `trace-${recordIndex}-${sampleIndex}-${index}`, title: step.title, kind: step.kind,
        status: step.status === 'success' ? 'completed' : step.status,
        command: step.command, output: [api.text(step.content), ...(step.results || []).map(result => api.text(result.content))].filter(Boolean).join('\n'),
        previews: [...(step.results || []), ...(step.observations || [])].flatMap(message =>
          Array.isArray(message.content) ? message.content.filter(part => part.type === 'image').map(part => part.image) : []),
      }))));
  }

  function stepsFor(run, traceApi) {
    const terminal = !['queued', 'running'].includes(run.state);
    let steps = run.progress?.steps || [];
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
      this.run = null;
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
      const follow = this.list.scrollHeight - this.list.scrollTop - this.list.clientHeight < 48;
      this.panel.hidden = false;
      const active = ['queued', 'running'].includes(run.state);
      const turn = run.progress?.turn;
      this.status.textContent = active ? run.progress?.phase === 'awaiting_approval' ? 'Your approval is needed' : turn ? `Turn ${turn}${run.progress.max_turns ? ` of ${run.progress.max_turns}` : ''}` : 'Getting ready…'
        : `${steps.length} ${steps.length === 1 ? 'step' : 'steps'} recorded`;
      const visible = new Set(steps.map(step => step.id));
      for (const [id, row] of this.rows) if (!visible.has(id)) { row.node.remove(); this.rows.delete(id); }
      for (const step of steps) {
        const signature = JSON.stringify(step);
        let row = this.rows.get(step.id);
        if (!row) {
          const node = document.createElement('li');
          row = {node}; this.rows.set(step.id, row); this.list.append(node);
        }
        if (row.signature === signature) continue;
        row.signature = signature;
        const newlyFailed = step.status === 'failed' && row.status !== 'failed';
        row.status = step.status;
        const node = row.node;
        node.className = `activity-step ${step.status}`;
        const heading = document.createElement('div'); heading.className = 'activity-step-heading';
        const title = document.createElement('strong'); title.textContent = step.title;
        const badge = document.createElement('span'); badge.className = 'activity-step-status';
        badge.textContent = labels[step.status] || 'Recorded';
        heading.append(title, badge);
        const children = [heading];
        if (step.output && step.kind === 'model') {
          const message = document.createElement('p'); message.className = 'activity-message'; message.textContent = step.output;
          children.push(message);
        }
        const images = (step.previews || []).filter(url => /^preview-\d+-\d+\.png$/.test(url) || /^data:image\/png;base64,/.test(url));
        for (const [index, url] of images.entries()) {
          const image = document.createElement('img'); image.className = 'activity-preview';
          image.alt = `Preview ${index + 1} from ${step.title.toLowerCase()}`;
          image.loading = 'lazy'; image.src = url.startsWith('data:') ? url : `/api/runs/${run.id}/${url}`;
          children.push(image);
        }
        if (step.command || (step.output && step.kind !== 'model')) {
          const details = document.createElement('details');
          details.open = newlyFailed || (row.details ? row.details.open : false);
          const summary = document.createElement('summary'); summary.textContent = step.command ? 'View action details' : 'View result';
          details.append(summary);
          for (const [label, value] of [['Action', step.command], ['Result', step.output]]) if (value) {
            const caption = document.createElement('p'); caption.textContent = label;
            const pre = document.createElement('pre'); pre.textContent = value;
            details.append(caption, pre);
          }
          row.details = details; children.push(details);
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
      if (follow) this.list.scrollTop = this.list.scrollHeight;
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {Activity, stepsFor, traceSteps};
  else root.ArtsyActivity = Activity;
})(globalThis);
