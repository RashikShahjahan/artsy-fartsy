const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(saved = [], runResponse) {
  const elements = {};
  function node(id = '') {
    return {id, hidden: false, value: '', checked: false, children: [], options: [],
      scrollTop: 0, scrollHeight: 0, clientHeight: 300,
      classList: {toggle() {}, add() {}, remove() {}},
      setAttribute() {}, removeAttribute() {}, toggleAttribute() {}, focus() {},
      querySelector() { return node(); }, append(...items) { this.children.push(...items); items.forEach(item => item.parent = this); },
      remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); },
      replaceChildren(...items) { this.children = items; }};
  }
  for (const [, id] of fs.readFileSync(`${__dirname}/artsy.html`, 'utf8').matchAll(/id="([^"]+)"/g)) elements[id] = node(id);
  const sessions = [], requests = [];
  class Recognition {
    constructor() { sessions.push(this); }
    start() {}
    stop() {}
    abort() {}
  }
  const storage = new Map([['artsy.sketchbook.v1', JSON.stringify(saved)]]);
  const document = {querySelectorAll: () => Object.values(elements), querySelector: () => node(),
    createElement: () => node(), addEventListener() {}, hidden: false};
  const timers = [];
  const context = vm.createContext({
    document, navigator: {language: 'en'},
    window: {SpeechRecognition: Recognition, AudioContext: function() {},
      ArtsySpeech: class { speak() {} cancel() {} unlock() {} setEnabled() {} },
    },
    localStorage: {getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)},
    fetch: (url, options) => {
      requests.push({url, options});
      if (!runResponse) return new Promise(() => {});
      return Promise.resolve({ok: true, json: async () => runResponse(url, options)});
    },
    Option: function(text, value) { return {text, value}; },
    AbortController: class { constructor(){this.signal={};} abort(){this.signal.aborted=true;} },
    setTimeout: callback => timers.push(callback), clearTimeout() {},
  });
  vm.runInContext(fs.readFileSync(`${__dirname}/activity.js`, 'utf8'), context);
  context.window.ArtsyActivity = context.ArtsyActivity;
  vm.runInContext(fs.readFileSync(`${__dirname}/run-client.js`, 'utf8'), context);
  vm.runInContext(fs.readFileSync(`${__dirname}/artsy.js`, 'utf8'), context);
  return {elements, sessions, requests, timers};
}

test('discarded recognition callbacks cannot submit or interrupt a new voice session', () => {
  const {elements: ui, sessions, requests} = setup();
  ui['mic-button'].onclick();
  const old = sessions[0]; old.onstart();
  ui['new-drawing'].onclick();
  ui['mic-button'].onclick();
  const next = sessions[1]; next.onstart();
  next.onresult({results: [Object.assign([{transcript: 'a cat'}], {isFinal: true})]});
  old.onresult({results: [Object.assign([{transcript: 'discard me'}], {isFinal: true})]});
  old.onerror({error: 'not-allowed'}); old.onend();
  assert.equal(ui['voice-title'].textContent, 'a cat');
  assert.equal(requests.length, 0);
  next.onend();
  assert.equal(JSON.parse(requests[0].options.body).prompt, 'a cat');
});

test('opening history uses the displayed fallback version for the next revision', () => {
  const version = 'b'.repeat(32);
  const {elements: ui, requests} = setup([{id: 'a'.repeat(32), title: 'Original', selectedId: 'c'.repeat(32), versions: [{id: version, prompt: 'A cat'}]}]);
  ui.history.children[0].onclick();
  ui.prompt.value = 'Make it blue';
  ui['prompt-form'].onsubmit({preventDefault() {}});
  assert.equal(JSON.parse(requests.at(-1).options.body).parent_id, version);
  assert.equal(ui.prompt.placeholder, 'What would you like to change?');
});

test('live drawing advances from recorded steps to a downloadable validated render', async () => {
  const id = 'b'.repeat(32);
  const step = {id: 'action-1', kind: 'render', tool: 'preview_drawing', status: 'completed',
    title: 'Checking the preview', previews: ['preview-1-0.png']};
  let run = {state: 'running', execution_mode: 'auto', prompt: 'A cat', progress: {phase: 'previewing', steps: [step]}};
  const {elements: ui, timers, requests} = setup([], url => url === '/api/artsy/runs' ? {id} : run);
  ui.prompt.value = run.prompt;
  ui['prompt-form'].onsubmit({preventDefault() {}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ui['drawing-space'].hidden, false);
  assert.equal(ui['agent-activity'].hidden, false);
  assert.equal(ui['drawing-process'].hidden, false);
  assert.equal(ui['activity-steps'].children[0].children[1].src, `/api/runs/${id}/preview-1-0.png`);
  assert.equal(ui.download.hidden, true);
  run = {...run, progress: {phase: 'thinking', steps: [step,
    {id: 'turn-2', kind: 'model', title: 'Agent turn 2', status: 'running'}]}};
  await timers.shift()();
  assert.equal(ui['activity-steps'].children.length, 1);
  assert.equal(ui['activity-steps'].children[0].hidden, false);
  assert.equal(ui['activity-steps'].children[0].children[1].src, `/api/runs/${id}/preview-1-0.png`);
  assert.ok(timers.length, 'Auto mode keeps polling while the next step runs');
  assert.equal(requests.some(request => request.url.endsWith('/approve')), false);
  run = {...run, state: 'completed', result: {outcome: 'submitted'}, artifacts: ['image.png']};
  await timers.shift()();
  assert.equal(ui['drawing-image'].src, `/api/runs/${id}/image.png`);
  assert.equal(ui['drawing-image'].hidden, false);
  assert.equal(ui.download.href, ui['drawing-image'].src);
  assert.equal(ui.download.hidden, false);
  assert.equal(ui['drawing-process'].hidden, false);
});

test('the ten-turn limit shows the latest drawing and continues from it on follow-up', async () => {
  const id = 'b'.repeat(32);
  const run = {state: 'completed', prompt: 'A cat', result: {outcome: 'incomplete', turnLimitReached: true},
    artifacts: ['image.png', 'program.py'], progress: {turn: 10, steps: [
      {id: 'first', kind: 'model', title: 'First step', status: 'completed', output: 'Starting'},
      {id: 'last', kind: 'tool', tool: 'preview_drawing', title: 'Latest preview', status: 'completed', previews: ['preview-9-0.png']},
    ]}};
  const {elements: ui, requests} = setup([], url => url === '/api/artsy/runs' ? {id} : run);
  ui.prompt.value = run.prompt;
  ui['prompt-form'].onsubmit({preventDefault() {}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ui['drawing-image'].src, `/api/runs/${id}/image.png`);
  assert.equal(ui['drawing-image'].hidden, false);
  assert.equal(ui['input-error'].hidden, true);
  assert.equal(ui.prompt.value, '');
  assert.equal(ui.send.disabled, false);
  assert.equal(ui['activity-steps'].children[0].children[1].src, `/api/runs/${id}/preview-9-0.png`);
  ui.prompt.value = 'Add a blue hat';
  ui['prompt-form'].onsubmit({preventDefault() {}});
  const followUp = JSON.parse(requests.filter(request => request.url === '/api/artsy/runs').at(-1).options.body);
  assert.equal(followUp.parent_id, id);
  assert.equal(followUp.prompt, 'Add a blue hat');
});
