import { Pool, type PoolConfig } from 'pg';

const SSL_QUERY_PARAMETERS = ['ssl', 'sslmode', 'sslcert', 'sslkey', 'sslrootcert'];

export function createPoolConfig(env: NodeJS.ProcessEnv = process.env): PoolConfig {
  if (!env.DATABASE_URL) {
    return {
      user: 'postgres',
      password: 'postgres',
      host: 'localhost',
      port: 15432,
      database: 'codeart',
      ssl: false,
    };
  }

  if (env.DATABASE_SSL && env.DATABASE_SSL !== 'true' && env.DATABASE_SSL !== 'false') {
    throw new Error('DATABASE_SSL must be either "true" or "false"');
  }

  const databaseUrl = new URL(env.DATABASE_URL);
  for (const parameter of SSL_QUERY_PARAMETERS) {
    databaseUrl.searchParams.delete(parameter);
  }

  const useTls = env.DATABASE_SSL !== 'false';
  const ca = env.DATABASE_CA_CERT?.replace(/\\n/g, '\n');

  return {
    connectionString: databaseUrl.toString(),
    ssl: useTls
      ? {
          rejectUnauthorized: true,
          ...(ca ? { ca } : {}),
        }
      : false,
  };
}

export const pool = new Pool(createPoolConfig());

export let vectorExtensionAvailable = false;

export async function initializeDatabase(): Promise<void> {
  const client = await pool.connect();
  try {
    try {
      await client.query('CREATE EXTENSION IF NOT EXISTS vector');
      vectorExtensionAvailable = true;
      console.log('Vector extension is available and enabled');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error('Vector extension is not available:', message);
      console.log('Application will use Upstash Vector or random results instead');
      vectorExtensionAvailable = false;
    }

    await client.query(`
      CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        prompt TEXT NOT NULL,
        code TEXT NOT NULL,
        art_type TEXT NOT NULL
      )
    `);

    if (vectorExtensionAvailable) {
      await client.query('ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding vector(512)');
    }

    await client.query(`
      CREATE TABLE IF NOT EXISTS vector_outbox (
        document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
        embedding JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )
    `);
  } finally {
    client.release();
  }
}
