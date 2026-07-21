import fetch from 'node-fetch';
import { pool, vectorExtensionAvailable } from './db';

const VECTOR_ENDPOINT = process.env.VECTOR_ENDPOINT;
const VECTOR_TOKEN = process.env.VECTOR_TOKEN;
const VECTOR_READONLY_TOKEN = process.env.VECTOR_READONLY_TOKEN;
const VOYAGE_API_KEY = process.env.VOYAGE_API_KEY;

const upstashWriteAvailable = Boolean(VECTOR_ENDPOINT && VECTOR_TOKEN);
const upstashReadAvailable = Boolean(VECTOR_ENDPOINT && (VECTOR_READONLY_TOKEN || VECTOR_TOKEN));
const embeddingProviderAvailable = Boolean(VOYAGE_API_KEY);
const EXTERNAL_REQUEST_TIMEOUT_MS = 15_000;

let outboxFlush: Promise<void> | undefined;
let outboxFlushRequested = false;

export function buildVectorUrl(endpoint: string, operation: 'upsert' | 'query' | 'delete'): string {
  return new URL(operation, `${endpoint.replace(/\/+$/, '')}/`).toString();
}

function getVectorUrl(operation: 'upsert' | 'query' | 'delete'): string {
  if (!VECTOR_ENDPOINT) {
    throw new Error('VECTOR_ENDPOINT is not configured');
  }
  return buildVectorUrl(VECTOR_ENDPOINT, operation);
}

async function generateEmbedding(text: string): Promise<number[]> {
  if (!VOYAGE_API_KEY) {
    throw new Error('VOYAGE_API_KEY is required for similarity search');
  }

  const response = await fetch('https://api.voyageai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${VOYAGE_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      input: [text],
      model: 'voyage-3-lite',
      output_dimension: 512,
    }),
    signal: AbortSignal.timeout(EXTERNAL_REQUEST_TIMEOUT_MS),
  });

  if (!response.ok) {
    const detail = (await response.text()).slice(0, 1000);
    throw new Error(`Voyage embedding failed (${response.status}): ${detail || response.statusText}`);
  }

  const body = await response.json() as { data?: Array<{ embedding?: number[] }> };
  const embedding = body.data?.[0]?.embedding;
  if (!embedding?.length || !embedding.every(Number.isFinite)) {
    throw new Error('Voyage returned an empty embedding');
  }
  if (embedding.length !== 512) {
    throw new Error(`Voyage returned ${embedding.length} dimensions; expected 512`);
  }
  return embedding;
}

function arrayToVector(values: number[]): string {
  return `[${values.map((value) => value.toFixed(8)).join(',')}]`;
}

async function upstashRequest(
  operation: 'upsert' | 'query' | 'delete',
  token: string,
  method: 'POST' | 'DELETE',
  body: unknown,
): Promise<unknown> {
  const response = await fetch(getVectorUrl(operation), {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(EXTERNAL_REQUEST_TIMEOUT_MS),
  });

  if (!response.ok) {
    const detail = (await response.text()).slice(0, 1000);
    throw new Error(`Upstash ${operation} failed (${response.status}): ${detail || response.statusText}`);
  }

  return response.json();
}

async function storeInUpstash(id: string, embedding: number[]): Promise<void> {
  if (!VECTOR_TOKEN) {
    throw new Error('VECTOR_TOKEN is required to store vectors');
  }
  await upstashRequest('upsert', VECTOR_TOKEN, 'POST', { id, vector: embedding });
}

async function searchInUpstash(embedding: number[], limit: number, artType: 'drawing'): Promise<string[]> {
  const token = VECTOR_READONLY_TOKEN || VECTOR_TOKEN;
  if (!token) return [];

  const response = (await upstashRequest('query', token, 'POST', {
    vector: embedding,
    topK: Math.min(Math.max(limit * 20, 50), 1000),
  })) as { result?: Array<{ id?: string | number }> };
  const ids = (response.result || [])
    .map((result) => String(result.id))
    .filter((id) => /^\d+$/.test(id));
  if (ids.length === 0) return [];

  const dbResult = await pool.query(
    'SELECT id, code FROM documents WHERE id = ANY($1::int[]) AND art_type = $2',
    [ids.map(Number), artType],
  );
  const codeById = new Map(dbResult.rows.map((row) => [String(row.id), String(row.code)]));
  return ids.flatMap((id) => {
    const code = codeById.get(id);
    return code ? [code] : [];
  }).slice(0, limit);
}

export function mergeSearchResults(primary: string[], secondary: string[], limit: number): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();
  const maxLength = Math.max(primary.length, secondary.length);

  for (let index = 0; index < maxLength && merged.length < limit; index += 1) {
    for (const code of [primary[index], secondary[index]]) {
      if (code && !seen.has(code)) {
        seen.add(code);
        merged.push(code);
        if (merged.length === limit) break;
      }
    }
  }

  return merged;
}

async function drainVectorOutbox(limit: number): Promise<number> {
  if (!upstashWriteAvailable) return 0;

  const result = await pool.query(
    `SELECT document_id, embedding
     FROM vector_outbox
     ORDER BY created_at ASC
     LIMIT $1`,
    [limit],
  );

  for (const row of result.rows) {
    const documentId = String(row.document_id);
    const embedding = Array.isArray(row.embedding) ? row.embedding.map(Number) : [];
    if (embedding.length !== 512 || !embedding.every(Number.isFinite)) {
      console.error(`Discarding invalid vector outbox entry for document ${documentId}`);
      await pool.query('DELETE FROM vector_outbox WHERE document_id = $1', [documentId]);
      continue;
    }

    await storeInUpstash(documentId, embedding);
    await pool.query('DELETE FROM vector_outbox WHERE document_id = $1', [documentId]);
  }
  return result.rows.length;
}

export async function flushPendingVectors(limit = 20): Promise<void> {
  if (!upstashWriteAvailable) return;
  outboxFlushRequested = true;
  if (outboxFlush) return outboxFlush;

  outboxFlush = (async () => {
    do {
      outboxFlushRequested = false;
      const processed = await drainVectorOutbox(limit);
      if (processed === limit) outboxFlushRequested = true;
    } while (outboxFlushRequested);
  })();
  try {
    await outboxFlush;
  } finally {
    outboxFlush = undefined;
  }
}

export async function storeDocument(prompt: string, code: string, artType: 'drawing'): Promise<void> {
  const shouldEmbed = embeddingProviderAvailable && (vectorExtensionAvailable || upstashWriteAvailable);
  const embedding = shouldEmbed ? await generateEmbedding(prompt) : undefined;
  const client = await pool.connect();

  try {
    await client.query('BEGIN');
    const result = embedding && vectorExtensionAvailable
      ? await client.query(
          `INSERT INTO documents (prompt, code, art_type, embedding)
           VALUES ($1, $2, $3, $4::vector)
           RETURNING id`,
          [prompt, code, artType, arrayToVector(embedding)],
        )
      : await client.query(
          'INSERT INTO documents (prompt, code, art_type) VALUES ($1, $2, $3) RETURNING id',
          [prompt, code, artType],
        );

    const id = String(result.rows[0].id);
    if (embedding && upstashWriteAvailable) {
      await client.query(
        `INSERT INTO vector_outbox (document_id, embedding)
         VALUES ($1, $2::jsonb)
         ON CONFLICT (document_id) DO UPDATE SET embedding = EXCLUDED.embedding`,
        [id, JSON.stringify(embedding)],
      );
    }

    await client.query('COMMIT');
  } catch (error) {
    await client.query('ROLLBACK').catch(() => undefined);
    throw error;
  } finally {
    client.release();
  }

  void flushPendingVectors().catch((error) => {
    console.error('Failed to flush vector outbox:', error);
  });
}

export async function findSimilarDocuments(
  query: string,
  artType: 'drawing',
  limit = 3,
): Promise<string[]> {
  let embedding: number[] | undefined;
  if (embeddingProviderAvailable && (upstashReadAvailable || vectorExtensionAvailable)) {
    try {
      embedding = await generateEmbedding(query);
    } catch (error) {
      console.error('Embedding generation failed:', error);
    }
  }

  void flushPendingVectors().catch((error) => {
    console.error('Failed to flush vector outbox:', error);
  });

  let postgresResults: string[] = [];
  let upstashResults: string[] = [];

  if (embedding && vectorExtensionAvailable) {
    try {
      const result = await pool.query(
        `SELECT code
         FROM documents
         WHERE art_type = $2 AND embedding IS NOT NULL
         ORDER BY embedding <-> $1::vector ASC
         LIMIT $3`,
        [arrayToVector(embedding), artType, limit * 5],
      );
      postgresResults = result.rows.map((row) => String(row.code));
    } catch (error) {
      console.error('PostgreSQL vector search failed:', error);
    }
  }

  if (embedding && upstashReadAvailable) {
    try {
      upstashResults = await searchInUpstash(embedding, limit * 5, artType);
    } catch (error) {
      console.error('Upstash vector search failed:', error);
    }
  }

  const mergedResults = mergeSearchResults(postgresResults, upstashResults, limit);
  if (mergedResults.length > 0) return mergedResults;

  console.log('Vector search unavailable, returning random drawings');
  const result = await pool.query(
    'SELECT code FROM documents WHERE art_type = $2 ORDER BY RANDOM() LIMIT $1',
    [limit, artType],
  );
  return result.rows.map((row) => String(row.code));
}
