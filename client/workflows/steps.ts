import { Sandbox } from '@vercel/sandbox';
import { get, put } from '@vercel/blob';
import { contentHash as hash, pngData } from '../lib/server/artifacts.ts';
import { job, patch, redis, type Job } from '../lib/server/store.ts';
import { box, readJson, sandboxAuth } from '../lib/server/sandbox.ts';
import type { Step } from '../lib/types';

async function save(id: string, name: string, data: Buffer) {
  const blob = await put(`artsy/${id}/${name}`, data, { access: 'private', addRandomSuffix: false, allowOverwrite: true, contentType: name.endsWith('.png') ? 'image/png' : 'application/octet-stream' });
  return blob.pathname;
}
async function source(parent: Job) {
  const path = parent.files?.['program.py'];
  if (!path || parent.state !== 'completed') throw new Error('Parent unavailable');
  const result = await get(path, { access: 'private' });
  if (!result || result.statusCode !== 200) throw new Error('Parent unavailable');
  const data = Buffer.from(await new Response(result.stream).arrayBuffer());
  if (data.length > 65536 || hash(data) !== parent.result?.sourceHash) throw new Error('Parent changed');
  return data;
}

export async function initialize(id: string) {
  'use step';
  console.log('Initializing drawing', id);
  const value = (await job(id))!;
  if (value.state === 'cancelled') return;
  const image = process.env.ARTSY_SANDBOX_IMAGE;
  const snapshotId = process.env.ARTSY_SANDBOX_SNAPSHOT;
  const runtimeSource = snapshotId ? { source: { type: 'snapshot' as const, snapshotId } } : { image };
  if ((!image && !snapshotId) || !process.env.TINKER_API_KEY || !process.env.ARTSY_TINKER_MODEL_PATH) throw new Error('Drawing environment missing');
  const agentName = `artsy-agent-${id}`, rendererName = `artsy-render-${id}`;
  await patch(id, { agent: agentName, renderer: rendererName });
  const renderer = await Sandbox.getOrCreate({ name: rendererName, ...runtimeSource, persistent: false, timeout: 1_200_000, networkPolicy: 'deny-all', ...sandboxAuth() });
  if (value.parent_id) {
    const parent = (await job(value.parent_id))!;
    await renderer.writeFiles([{ path: '/workspace/program.py', content: await source(parent) }]);
  }
  const agent = await Sandbox.getOrCreate({ name: agentName, ...runtimeSource, persistent: false, timeout: 1_200_000, ...sandboxAuth() });
  await agent.writeFiles([{ path: '/vercel/sandbox/job/request.json', content: Buffer.from(JSON.stringify({ prompt: value.prompt, parentPrompt: value.parent_id ? (await job(value.parent_id))?.prompt : undefined })) }]);
  const command = await agent.runCommand({ cmd: '/usr/local/bin/python', args: ['/opt/artsy/app/agent.py'], detached: true, cwd: '/opt/artsy/app', env: { PYTHONPATH: '/opt/artsy/app:/opt/artsy/app/runtime', MSWEA_GLOBAL_CONFIG_DIR: '/tmp/mini-swe-agent', TINKER_API_KEY: process.env.TINKER_API_KEY, ARTSY_TINKER_MODEL_PATH: process.env.ARTSY_TINKER_MODEL_PATH } });
  await patch(id, { commandId: command.cmdId });
  // Cancellation can arrive while the VMs are starting. Never overwrite it.
  if ((await job(id))?.state === 'cancelled') { await cleanup(id); return; }
  await patch(id, { state: 'running' });
}
initialize.maxRetries = 0;

type ActionResult = { returncode: number; content: Array<{ type: string; text?: string; image_url?: { url: string } }>; terminal?: boolean; submission?: { source: string; png: string; sourceHash: string; imageHash: string } };
export async function advance(id: string) {
  'use step';
  const value = (await job(id))!;
  if (value.state === 'cancelled') return true;
  if (Date.now() - value.createdAt > 1_100_000) throw new Error('Drawing timed out');
  const agent = await box(value.agent!);
  const done = await readJson<{ ok: boolean }>(agent, 'done.json');
  if (done) { if (!done.ok) throw new Error('Agent failed'); return true; }
  const action = await readJson<{ id: number; command: string }>(agent, 'action.json');
  if (!action || action.id <= (value.action || 0)) {
    const command = await agent.getCommand(value.commandId!);
    if (command.exitCode !== null) throw new Error('Agent exited without a result');
    return false;
  }
  console.log('Executing drawing action', id, action.id);
  const renderer = await box(value.renderer!);
  await renderer.writeFiles([{ path: `/vercel/sandbox/job/action-${action.id}.json`, content: Buffer.from(JSON.stringify(action)) }]);
  const step: Step = { id: `action-${action.id}`, title: 'Editing drawing', kind: 'tool', tool: 'bash', status: 'running' };
  const steps = [...(value.progress?.steps || []).filter(s => s.id !== step.id), step];
  await patch(id, { progress: { phase: 'drawing', steps } });
  const execution = await renderer.runCommand({ cmd: 'timeout', args: ['--signal=KILL', '100', 'python', '/opt/artsy/app/action.py', String(action.id)], cwd: '/workspace', env: { PYTHONPATH: '/opt/artsy/app:/opt/artsy/app/runtime', MSWEA_GLOBAL_CONFIG_DIR: '/tmp/mini-swe-agent' } });
  if (execution.exitCode !== 0) throw new Error('Drawing action failed');
  const result = (await readJson<ActionResult>(renderer, `result-${action.id}.json`))!;
  const files = { ...value.files };
  const previews: string[] = [];
  for (const part of result.content) {
    if (part.type === 'image_url' && part.image_url?.url.startsWith('data:image/png;base64,')) {
      const name = `preview-${action.id}-` + previews.length + '.png';
      files[name] = await save(id, name, pngData(part.image_url.url.split(',')[1]));
      previews.push(name);
    }
  }
  if (result.submission) {
    const { source, png, sourceHash, imageHash } = result.submission;
    const program = Buffer.from(source), image = pngData(png);
    if (program.length > 65536 || image.length > 8388608 || hash(program) !== sourceHash || hash(image) !== imageHash) throw new Error('Invalid submission');
    files['program.py'] = await save(id, 'program.py', program);
    files['image.png'] = await save(id, 'image.png', image);
    await patch(id, { submitted: true, result: { outcome: 'submitted', sourceHash, imageHash } });
  }
  steps[steps.length - 1] = { ...step, status: result.returncode === 0 ? 'completed' : 'failed', output: result.content.filter(p => p.type === 'text').map(p => p.text).join('\n').slice(0, 16384), previews };
  await patch(id, { files, artifacts: Object.keys(files), action: action.id, progress: { phase: 'thinking', steps } });
  // Persist progress before acknowledging so a retried poll cannot lose an action.
  await agent.writeFiles([{ path: `/vercel/sandbox/job/reply-${action.id}.json`, content: Buffer.from(JSON.stringify({ ...result, submission: undefined })) }]);
  return false;
}
advance.maxRetries = 0;

export async function finalize(id: string) {
  'use step';
  console.log('Finalizing drawing', id);
  const value = (await job(id))!;
  if (value.state === 'cancelled') return;
  const agent = await box(value.agent!);
  const trace = await agent.readFileToBuffer({ path: '/vercel/sandbox/job/trace.json' });
  const files = { ...value.files };
  if (trace) files['trace.json'] = await save(id, 'trace.json', trace);
  let result = value.result;
  if (!value.submitted) {
    const renderer = await box(value.renderer!);
    const command = await renderer.runCommand({ cmd: 'timeout', args: ['--signal=KILL', '75', 'python', '/opt/artsy/app/snapshot.py'], cwd: '/workspace', env: { PYTHONPATH: '/opt/artsy/app:/opt/artsy/app/runtime', MSWEA_GLOBAL_CONFIG_DIR: '/tmp/mini-swe-agent' } });
    if (command.exitCode !== 0) throw new Error('Unable to save partial drawing');
    const partial = await readJson<{ source?: string; sourceHash?: string; png?: string; imageHash?: string }>(renderer, 'snapshot.json');
    if (partial?.source) {
      const program = Buffer.from(partial.source);
      if (program.length > 65536 || hash(program) !== partial.sourceHash) throw new Error('Invalid partial source');
      files['program.py'] = await save(id, 'program.py', program);
      if (partial.png) {
        const image = pngData(partial.png);
        if (image.length > 8388608 || hash(image) !== partial.imageHash) throw new Error('Invalid partial image');
        files['image.png'] = await save(id, 'image.png', image);
      }
      result = { outcome: 'incomplete', turnLimitReached: true, sourceHash: partial.sourceHash, imageHash: partial.imageHash };
    }
  }
  await patch(id, { files, artifacts: Object.keys(files), state: files['program.py'] ? 'completed' : 'failed', result: result || { outcome: 'incomplete', turnLimitReached: true } });
}
export async function fail(id: string) {
  'use step';
  console.error('Drawing failed', id);
  if ((await job(id))?.state !== 'cancelled') await patch(id, { state: 'failed' });
}
export async function cleanup(id: string) {
  'use step';
  console.log('Stopping drawing sandboxes', id);
  const value = await job(id);
  const stopped = await Promise.allSettled([value?.agent, value?.renderer].filter((name): name is string => !!name).map(async name => {
    try { await (await box(name)).stop(); }
    catch (error) { if ((error as { response?: { status?: number } }).response?.status !== 404) throw error; }
  }));
  const failure = stopped.find(result => result.status === 'rejected');
  if (failure?.status === 'rejected') throw failure.reason;
  if (value) await redis().eval(
    "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0",
    [`artsy:active:${value.owner}`], [id]);

}

