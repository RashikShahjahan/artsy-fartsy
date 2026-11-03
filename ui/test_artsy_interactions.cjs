const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(saved = []) {
  const elements = {};
  function node(id = '') {
    return {id, hidden: false, value: '', checked: false, children: [], options: [],
      classList: {toggle() {}, add() {}, remove() {}},
      setAttribute() {}, removeAttribute() {}, toggleAttribute() {}, focus() {},
      querySelector() { return node(); }, append(...items) { this.children.push(...items); },
      replaceChildren() { this.children = []; }};
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
  vm.runInNewContext(fs.readFileSync(`${__dirname}/artsy.js`, 'utf8'), {
    document, navigator: {language: 'en'},
    window: {SpeechRecognition: Recognition, AudioContext: function() {},
      ArtsySpeech: class { speak() {} cancel() {} unlock() {} setEnabled() {} },
      ArtsyActivity: class { reset() {} render() {} fail() {} }},
    localStorage: {getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)},
    fetch: (url, options) => { requests.push({url, options}); return new Promise(() => {}); },
    Option: function(text, value) { return {text, value}; }, setTimeout, clearTimeout,
  });
  return {elements, sessions, requests};
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
