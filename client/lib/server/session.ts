import { randomBytes } from 'node:crypto';
import { cookies } from 'next/headers';
const COOKIE = 'artsy_session';
export async function session() {
  const jar = await cookies();
  let id = jar.get(COOKIE)?.value;
  if (!id || !/^[a-f0-9]{64}$/.test(id)) {
    id = randomBytes(32).toString('hex');
    jar.set(COOKIE, id, { httpOnly: true, secure: process.env.NODE_ENV === 'production', sameSite: 'lax', path: '/', maxAge: 31536000 });
  }
  return id;
}
