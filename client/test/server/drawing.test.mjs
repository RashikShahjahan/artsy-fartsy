import assert from 'node:assert/strict';
import { test, mock } from 'node:test';
import { createHash } from 'node:crypto';

// Exercise the production workflow steps with network services replaced at their boundary.
const records = new Map(), boxes = new Map(), creations = [], events = [];
const redis = {
  hgetall: async key => records.get(key) || null,
  hset: async (key, value) => records.set(key, { ...records.get(key), ...value }),
  eval: async (script, keys, args) => {
    if (keys[0].startsWith('artsy:active:')) return 1;
    const value = records.get(keys[0]) || {};
    for (let i = 0; i < args.length; i += 2) {
      if (args[i] === 'state' && value.state === 'cancelled') continue;
      try { value[args[i]] = JSON.parse(args[i + 1]); }
      catch { value[args[i]] = args[i + 1]; }
    }
    records.set(keys[0], value);
    events.push('persist');
    return 1;
  },
};
mock.module('@upstash/redis', { namedExports: { Redis: { fromEnv: () => redis } } });
mock.module('@vercel/blob', { namedExports: {
  put: async (pathname, data, options) => { assert.equal(options.access, 'private'); return { pathname }; },
  get: async () => null,
} });
const makeBox = name => {
  const files = new Map();
  const value = {
    name, files, stopped: false,
    writeFiles: async writes => { for (const f of writes) { files.set(f.path, f.content); if (f.path.includes('reply-')) events.push('acknowledge'); } },
    readFileToBuffer: async ({ path }) => files.get(path) || null,
    runCommand: async options => { value.lastCommand = options; return { cmdId: 'command', exitCode: 0 }; },
    getCommand: async () => ({ exitCode: null }),
    stop: async () => { value.stopped = true; },
  };
  boxes.set(name, value);
  return value;
};
mock.module('@vercel/sandbox', { namedExports: { Sandbox: {
  getOrCreate: async options => { creations.push(options); return boxes.get(options.name) || makeBox(options.name); },
  get: async ({ name }) => boxes.get(name),
} } });
const { initialize, advance, finalize, cleanup } = await import('../../workflows/steps.ts');
const { patch } = await import('../../lib/server/store.ts');
const id = 'a'.repeat(32), key = `artsy:job:${id}`;
const data = (box, name, value) => box.files.set(`/vercel/sandbox/job/${name}`, Buffer.from(JSON.stringify(value)));
const sha = value => createHash('sha256').update(value).digest('hex');

test('job runs the Python agent separately from credential-free, network-denied code execution', async () => {
  process.env.ARTSY_SANDBOX_IMAGE = 'test-image';
  process.env.TINKER_API_KEY = 'test-key';
  process.env.ARTSY_TINKER_MODEL_PATH = 'tinker://test';
  records.set(key, { id, state: 'queued', owner: 'owner', createdAt: Date.now(), prompt: 'A fox' });
  await initialize(id);
  assert.equal(creations.length, 2);
  const renderConfig = creations.find(c => c.name.includes('render'));
  assert.equal(renderConfig.networkPolicy, 'deny-all');
  assert.equal(renderConfig.env, undefined);
  const agent = boxes.get(`artsy-agent-${id}`), renderer = boxes.get(`artsy-render-${id}`);
  assert.equal(agent.lastCommand.env.TINKER_API_KEY, 'test-key');
  assert.equal(renderer.lastCommand, undefined);
  assert.equal(records.get(key).state, 'running');

  const program = 'print("drawing")', png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=', 'base64');
  data(agent, 'action.json', { id: 1, command: 'artsy-submit program.py' });
  data(renderer, 'result-1.json', { returncode: 0, content: [{ type: 'text', text: 'Submitted' }], terminal: true,
    submission: { source: program, png: png.toString('base64'), sourceHash: sha(program), imageHash: sha(png) } });
  assert.equal(await advance(id), false);
  assert.equal(renderer.lastCommand.env.TINKER_API_KEY, undefined);
  assert.equal(renderer.lastCommand.env.PYTHONPATH, '/opt/artsy/app:/opt/artsy/app/runtime');
  assert.equal(records.get(key).submitted, true);
  assert.deepEqual(records.get(key).artifacts.sort(), ['image.png', 'program.py']);
  assert.equal(events.at(-1), 'acknowledge');
  assert.equal(events.at(-2), 'persist');
  const reply = JSON.parse(agent.files.get('/vercel/sandbox/job/reply-1.json'));
  assert.equal(reply.submission, undefined);
  data(agent, 'done.json', { ok: true });
  assert.equal(await advance(id), true);
  await finalize(id);
  assert.equal(records.get(key).state, 'completed');
  await cleanup(id);
  assert.ok(agent.stopped && renderer.stopped);
});

test('cancellation wins over concurrent worker status updates and skips VM creation', async () => {
  records.set(key, { id, owner: 'owner', state: 'cancelled', createdAt: Date.now() });
  const count = creations.length;
  await initialize(id);
  await patch(id, { state: 'running' });
  assert.equal(records.get(key).state, 'cancelled');
  assert.equal(await advance(id), true);
  await finalize(id);
  assert.equal(creations.length, count);
  assert.equal(records.get(key).state, 'cancelled');
});

test('corrupt artifacts fail validation before they are recorded', async () => {
  records.set(key, { id, owner: 'owner', state: 'running', createdAt: Date.now(), agent: `artsy-agent-${id}`, renderer: `artsy-render-${id}` });
  const agent = boxes.get(`artsy-agent-${id}`), renderer = boxes.get(`artsy-render-${id}`);
  agent.files.delete('/vercel/sandbox/job/done.json');
  data(agent, 'action.json', { id: 2, command: 'artsy-submit program.py' });
  data(renderer, 'result-2.json', { returncode: 0, content: [], submission: { source: 'tampered', png: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=', sourceHash: 'wrong', imageHash: 'wrong' } });
  await assert.rejects(advance(id), /Invalid submission/);
  assert.equal(records.get(key).submitted, undefined);
});

test('turn exhaustion retains editable source even without a valid image', async () => {
  records.set(key, { id, owner: 'owner', state: 'running', createdAt: Date.now(), agent: `artsy-agent-${id}`, renderer: `artsy-render-${id}` });
  const renderer = boxes.get(`artsy-render-${id}`);
  const source = '# partial drawing';
  data(renderer, 'snapshot.json', { source, sourceHash: sha(source) });
  await finalize(id);
  const value = records.get(key);
  assert.equal(value.state, 'completed');
  assert.equal(value.result.outcome, 'incomplete');
  assert.equal(value.result.turnLimitReached, true);
  assert.deepEqual(value.artifacts, ['program.py']);
});

test('cleanup attempts both VMs even if one stop fails', async () => {
  const agent = boxes.get(`artsy-agent-${id}`), renderer = boxes.get(`artsy-render-${id}`);
  renderer.stopped = false;
  const stop = agent.stop;
  agent.stop = async () => { throw new Error('Temporary failure'); };
  await assert.rejects(cleanup(id), /Temporary failure/);
  assert.equal(renderer.stopped, true);
  agent.stop = stop;
});


test('a prepared snapshot takes precedence over the optional registry image', async () => {
  records.set(key, { id, state: 'queued', owner: 'owner', createdAt: Date.now(), prompt: 'A circle' });
  process.env.ARTSY_SANDBOX_SNAPSHOT = 'snapshot-test';
  const count = creations.length;
  try {
    await initialize(id);
    for (const options of creations.slice(count)) {
      assert.deepEqual(options.source, { type: 'snapshot', snapshotId: 'snapshot-test' });
      assert.equal(options.image, undefined);
    }
  } finally {
    delete process.env.ARTSY_SANDBOX_SNAPSHOT;
  }
});
