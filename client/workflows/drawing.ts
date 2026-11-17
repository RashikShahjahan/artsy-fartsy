import { sleep } from 'workflow';
import { initialize, advance, finalize, fail, cleanup } from './steps';

export async function draw(id: string) {
  'use workflow';
  console.log('Drawing workflow started', id);
  try {
    await initialize(id);
    // Each wait is durable; no Function stays open waiting for Tinker.
    while (!(await advance(id))) await sleep('3s');
    await finalize(id);
  } catch {
    await fail(id);
  } finally {
    await cleanup(id);
  }
}
