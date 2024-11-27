// src/db.ts
import { Pool } from "pg";

export const pool = new Pool(
  process.env.DATABASE_URL
    ? {
        connectionString: process.env.DATABASE_URL,
        ssl: {
          rejectUnauthorized: false // Required for Heroku
        }
      }
    : {
        user: "postgres",
        password: "postgres",
        host: "localhost",
        port: 15432,
        database: "codeart",
      }
);


export async function initializeDatabase() {
  const client = await pool.connect();
  try {
    await client.query("CREATE EXTENSION IF NOT EXISTS vector");

    await client.query(`
      CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        prompt TEXT NOT NULL,
        code TEXT NOT NULL,
        art_type TEXT NOT NULL,
        embedding vector(512)
      )
    `);
  } finally {
    client.release();
  }
}

