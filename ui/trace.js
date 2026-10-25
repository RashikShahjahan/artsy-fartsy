/* Normalize recorded Inspect data without inferring unrecorded actions. */
(function(root) {
  function text(content) {
    if (typeof content === 'string') return content;
    return Array.isArray(content) ? content.filter(p => p.type === 'text').map(p => p.text || '').join('\n') : '';
  }
  function feedback(message) {
    const value = text(message.content);
    // ArtCanvas returns a JSON feedback line inside the bash tool response.
    for (const line of value.split('\n')) {
      try {
        const data = JSON.parse(line);
        if (data && typeof data === 'object' && typeof data.ok === 'boolean' && 'exitCode' in data) return data;
      } catch {}
    }
    return null;
  }
  function statusFor(call, results, kind) {
    if (call.parse_error) return 'failed';
    if (results.some(m => {
      const data = feedback(m);
      const exit = text(m.content).match(/^Return code:\s*(-?\d+)/m);
      return m.error || data?.ok === false || (data?.exitCode != null && data.exitCode !== 0) || (exit && Number(exit[1]) !== 0);
    })) return 'failed';
    if (kind === 'submit') return 'requested';
    if (!results.length) return 'unknown';
    return results.some(m => feedback(m)?.ok === true || /^Return code:\s*0\s*$/m.test(text(m.content))) ? 'success' : 'recorded';
  }
  function commandFor(call) {
    let args = call.arguments;
    if (typeof args === 'string') { try { args = JSON.parse(args); } catch { return args; } }
    return typeof args?.command === 'string' ? args.command : JSON.stringify(args ?? {}, null, 2);
  }
  function sourceFor(command) {
    // Only extract a literal, complete here-document. Other edits stay as commands.
    const match = command.match(/(?:^|\n)cat\s*>\s*(?:\/workspace\/)?program\.py\s*<<\s*(['"]?)([A-Za-z_][\w]*)\1\s*\n([\s\S]*?)\n\2(?:\n|$)/);
    return match && (match[1] || !/[\$`\\]/.test(match[3])) ? match[3] : null;
  }
  function normalizeSample(sample) {
    const messages = sample.messages || [], events = sample.events || [];
    const results = new Map(), models = new Map();
    messages.filter(m => m.role === 'tool' && m.tool_call_id).forEach(m => {
      const list = results.get(m.tool_call_id) || [];
      list.push(m); results.set(m.tool_call_id, list);
    });
    events.filter(e => e.event === 'model').forEach(e => {
      for (const choice of e.output?.choices || []) for (const call of choice.message?.tool_calls || []) {
        if (call.id) models.set(call.id, e);
      }
    });
    const steps = [], used = new Set();
    let previousSource = null, lastResult = null;
    for (const message of messages) {
      if (message.role === 'assistant' && message.tool_calls?.length) {
        lastResult = null;
        message.tool_calls.forEach((call, index) => {
          const command = commandFor(call), source = sourceFor(command);
          const kind = /^\s*echo\s+['"]?COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT['"]?\s*$/.test(command) ? 'submit'
            : source !== null ? 'write' : /(?:^|\s)\/opt\/artsy\/render\.py(?:\s|$)/.test(command) ? 'render' : 'tool';
          const matched = call.id ? results.get(call.id) || [] : [];
          matched.forEach(m => used.add(m));
          const step = {kind, title: kind === 'write' ? (previousSource === null ? 'Write program.py' : 'Update program.py')
            : kind === 'render' ? 'Render preview' : kind === 'submit' ? 'Submit drawing' : `Run ${call.function || 'tool'}`,
            call, command, source, previousSource, results: matched, model: models.get(call.id),
            content: index === 0 ? message.content : '', observations: [], status: statusFor(call, matched, kind)};
          if (source !== null) previousSource = step.status === 'success' ? source : null;
          steps.push(step);
        });
      } else if (message.role === 'tool') {
        lastResult = steps.find(s => s.results?.includes(message)) || null;
        if (!used.has(message)) steps.push({kind: 'message', title: 'Unmatched tool result', content: message.content, status: message.error ? 'failed' : 'recorded', error: message.error});
      } else if (message.role === 'user' && lastResult && text(message.content).startsWith('[The following images are tool output')) {
        lastResult.observations.push(message);
        lastResult = null;
      } else {
        lastResult = null;
        if (message.role === 'assistant' || message.role === 'user') steps.push({kind: 'message', title: message.role === 'assistant' ? 'Agent message' : 'Request / context', content: message.content, status: 'recorded'});
      }
    }
    return {sample, steps, events};
  }
  function diff(before, after) {
    const a = before.split('\n'), b = after.split('\n');
    // Bound quadratic work for long generated files; full sources remain available.
    if (a.length * b.length > 500000) return null;
    const rows = Array.from({length: a.length + 1}, () => new Uint32Array(b.length + 1));
    for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--)
      rows[i][j] = a[i] === b[j] ? rows[i+1][j+1] + 1 : Math.max(rows[i+1][j], rows[i][j+1]);
    const lines = []; let i = 0, j = 0;
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j]) {lines.push([' ', a[i]]); i++; j++;}
      else if (i < a.length && (j === b.length || rows[i+1][j] >= rows[i][j+1])) lines.push(['-', a[i++]]);
      else lines.push(['+', b[j++]]);
    }
    return lines;
  }
  const api = {text, feedback, normalizeSample, diff};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ArtsyTrace = api;
})(globalThis);
