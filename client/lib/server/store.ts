import { Redis } from '@upstash/redis';
import type { Run } from '../types';
export type Job = Run & { owner: string; createdAt: number; runId?: string; agent?: string; renderer?: string; commandId?: string; action?: number; submitted?: boolean; files?: Record<string, string> };
export function redis() {
  if (process.env.KV_REST_API_URL && process.env.KV_REST_API_TOKEN && !process.env.UPSTASH_REDIS_REST_URL) {
    return new Redis({ url: process.env.KV_REST_API_URL, token: process.env.KV_REST_API_TOKEN });
  }
  return Redis.fromEnv();
}
const key = (id: string) => `artsy:job:${id}`;
export async function job(id: string) { return redis().hgetall<Job>(key(id)); }
export async function patch(id: string, value: Partial<Job>) {
  // Cancellation wins even when a worker finishes concurrently.
  const entries = Object.entries(value).filter(([, v]) => v !== undefined);
  await redis().eval(`
    local cancelled = redis.call('HGET', KEYS[1], 'state') == 'cancelled'
    for i = 1, #ARGV, 2 do
      if ARGV[i] ~= 'state' or not cancelled then
        redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1])
      end
    end
    return 1
  `, [key(id)], entries.flatMap(([k, v]) => [k, typeof v === 'string' ? v : JSON.stringify(v)]));
}
export async function create(value: Job) {
  const tx = redis().multi();
  tx.hset(key(value.id), value);
  tx.zadd(`artsy:history:${value.owner}`, { score: value.createdAt, member: value.id });
  await tx.exec();
}
export async function history(owner: string) {
  const ids = await redis().zrange<string[]>(`artsy:history:${owner}`, 0, 99, { rev: true });
  return (await Promise.all(ids.map(job))).filter((j): j is Job => !!j);
}
export function publicJob(value: Job): Run {
  const { id, state, prompt, parent_id, artifacts, progress, result } = value;
  return { id, state, prompt, parent_id, artifacts, progress, result };
}
