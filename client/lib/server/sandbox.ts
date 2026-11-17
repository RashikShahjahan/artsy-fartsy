import { Sandbox } from '@vercel/sandbox';
export function sandboxAuth() {
  if (process.env.VERCEL_TOKEN) return { token: process.env.VERCEL_TOKEN, teamId: process.env.VERCEL_TEAM_ID!, projectId: process.env.VERCEL_PROJECT_ID! };
  return {};
}
export async function box(id: string) { return Sandbox.get({ name: id, ...sandboxAuth() }); }
export async function readJson<T>(sandbox: Sandbox, name: string): Promise<T | null> {
  const data = await sandbox.readFileToBuffer({ path: `/vercel/sandbox/job/${name}` });
  return data ? JSON.parse(data.toString('utf8')) : null;
}
