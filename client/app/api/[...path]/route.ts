import { randomBytes } from 'node:crypto';
import { get } from '@vercel/blob';
import { start } from 'workflow/api';
import { draw } from '@/workflows/drawing';
import { cleanup } from '@/workflows/steps';
import { trustedOrigin } from '@/lib/proxy';
import { create, history, job, patch, publicJob, redis } from '@/lib/server/store';
import { session } from '@/lib/server/session';
import { isActive, RUN_ID } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 180;
const json = (data: unknown, status = 200) => Response.json(data, { status, headers: { 'Cache-Control': 'no-store' } });
async function body(request: Request) {
  const reader = request.body?.getReader();
  if (!reader) throw new Error('Invalid request');
  const chunks: Uint8Array[] = [];
  let size = 0;
  while (true) {
    const part = await reader.read();
    if (part.done) break;
    size += part.value.length;
    if (size > 20000) { await reader.cancel(); throw new Error('Request is too large'); }
    chunks.push(part.value);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}
async function handle(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const path = (await context.params).path.join('/');
  if (request.method === 'POST' && !trustedOrigin(request.headers.get('origin'), request.headers.get('host'))) return json({ error: 'Untrusted request origin.' }, 403);
  try {
    const owner = await session();
    if (request.method === 'GET' && path === 'runs') return json({ runs: (await history(owner)).map(value => ({ ...publicJob(value), progress: undefined })) });
    if (request.method === 'POST' && path === 'artsy/runs') {
      let input;
      try { input = await body(request); } catch { return json({ error: 'Enter a valid drawing request (up to 20 KB).' }, 400); }
      if (!input || typeof input.prompt !== 'string' || !input.prompt.trim() || input.prompt.length > 4000) return json({ error: 'Enter a prompt of 1–4000 characters.' }, 400);
      if (input.parent_id !== undefined) {
        if (typeof input.parent_id !== 'string' || !RUN_ID.test(input.parent_id)) return json({ error: 'Choose a completed drawing to edit.' }, 400);
        const parent = await job(input.parent_id);
        if (!parent || parent.owner !== owner || parent.state !== 'completed' || !parent.files?.['program.py']) return json({ error: 'That drawing is unavailable for editing.' }, 400);
      }
      const id = randomBytes(16).toString('hex');
      const lock = `artsy:active:${owner}`;
      if (!(await redis().set(lock, id, { nx: true, ex: 1200 }))) return json({ error: 'A drawing is already running.' }, 409);
      try {
        await create({ id, owner, state: 'queued', prompt: input.prompt.trim(), ...(input.parent_id ? { parent_id: input.parent_id } : {}), createdAt: Date.now() });
        const run = await start(draw, [id]);
        await patch(id, { runId: run.runId });
        return json({ id }, 202);
      } catch (error) {
        await patch(id, { state: 'failed' });
        await redis().del(lock);
        throw error;
      }
    }
    const match = /^runs\/([a-f0-9]{32})(?:\/(image\.png|program\.py|trace\.json|preview-\d+-\d+\.png|cancel))?$/.exec(path);
    if (!match) return json({ error: 'Not found.' }, 404);
    const [, id, artifact] = match;
    const value = await job(id);
    if (!value || value.owner !== owner) return json({ error: 'Drawing not found.' }, 404);
    if (request.method === 'POST' && artifact === 'cancel') {
      if (!isActive(value)) return json({ id, state: value.state });
      // Keep the workflow alive so it also cleans up VMs still being created.
      await patch(id, { state: 'cancelled' });
      await cleanup(id);
      return json({ id, state: 'cancelled' });
    }
    if (request.method !== 'GET' || artifact === 'cancel') return json({ error: 'Not found.' }, 404);
    if (!artifact) return json(publicJob(value));
    const file = value.files?.[artifact];
    if (!file) return json({ error: 'Artifact not found.' }, 404);
    const result = await get(file, { access: 'private' });
    if (!result || result.statusCode !== 200) return json({ error: 'Artifact not found.' }, 404);
    return new Response(result.stream, { headers: { 'Content-Type': artifact.endsWith('.png') ? 'image/png' : artifact.endsWith('.json') ? 'application/json' : 'text/plain; charset=utf-8', 'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff' } });
  } catch (error) {
    console.error('Drawing request failed', error);
    return json({ error: 'Drawing is temporarily unavailable. Please try again.' }, 503);
  }
}
export { handle as GET, handle as POST };
