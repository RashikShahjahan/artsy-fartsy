const {test, beforeEach} = require('node:test');
const assert = require('node:assert/strict');
const Speech = require('./speech.js');
let contexts, sources;
beforeEach(() => {
  contexts = []; sources = [];
  global.window = {AudioContext: class {
    constructor() { this.state = 'suspended'; this.destination = {}; contexts.push(this); }
    async resume() { this.state = 'running'; }
    async decodeAudioData(bytes) { return {bytes}; }
    createBufferSource() {
      const source = {started: false, stopped: false, connect() {}, disconnect() {},
        start() { this.started = true; }, stop() { this.stopped = true; }};
      sources.push(source); return source;
    }
  }};
  global.fetch = async () => ({ok: true, arrayBuffer: async () => new ArrayBuffer(8)});
});
test('greeting waits for an audio-unlocking gesture and repeated speech reuses generated audio', async () => {
  let calls = 0;
  global.fetch = async () => { calls++; return {ok:true,arrayBuffer:async()=>new ArrayBuffer(8)}; };
  const speech = new Speech();
  await speech.speak('Hi Artsy');
  assert.equal(sources.length, 0);
  await speech.unlock();
  assert.equal(sources.length, 1);
  assert.equal(sources[0].started, true);
  await speech.speak('Hi Artsy');
  assert.equal(sources[0].stopped, true);
  assert.equal(sources.length, 2);
  assert.equal(calls, 1);
});
test('muting stops playback and discards the pending greeting', async () => {
  const speech = new Speech();
  await speech.speak('Hi');
  speech.setEnabled(false);
  await speech.unlock();
  assert.equal(sources.length, 0);
  speech.setEnabled(true);
  await speech.speak('Hi');
  assert.equal(sources.length, 1);
  speech.setEnabled(false);
  assert.equal(sources[0].stopped, true);
});
test('cancellation aborts requests and late audio cannot interrupt newer speech', async () => {
  const pending = [];
  global.fetch = (url, options) => new Promise(resolve => pending.push({options, resolve}));
  const speech = new Speech();
  await speech.unlock();
  const older = speech.speak('Old');
  const newer = speech.speak('New');
  assert.equal(pending[0].options.signal.aborted, true);
  pending[1].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)});
  await newer;
  pending[0].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)});
  await older;
  assert.equal(sources.length, 1);
  speech.cancel();
  assert.equal(sources[0].stopped, true);
});
