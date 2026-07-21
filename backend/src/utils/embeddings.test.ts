import { describe, expect, test } from 'bun:test';
import { buildVectorUrl, mergeSearchResults } from './embeddings';

describe('Upstash REST URLs', () => {
  test('uses the full endpoint with current operation paths', () => {
    const endpoint = 'https://example-vector.upstash.io';
    expect(buildVectorUrl(endpoint, 'upsert')).toBe(`${endpoint}/upsert`);
    expect(buildVectorUrl(`${endpoint}/`, 'query')).toBe(`${endpoint}/query`);
    expect(buildVectorUrl(endpoint, 'delete')).toBe(`${endpoint}/delete`);
  });
});

describe('vector result merging', () => {
  test('interleaves PostgreSQL and Upstash results without duplicates', () => {
    expect(mergeSearchResults(['pg-1', 'shared', 'pg-2'], ['upstash-1', 'shared'], 4)).toEqual([
      'pg-1',
      'upstash-1',
      'shared',
      'pg-2',
    ]);
  });
});
