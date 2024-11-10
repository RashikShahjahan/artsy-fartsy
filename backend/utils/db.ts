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
        port: 5432,
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
        content TEXT NOT NULL,
        embedding vector(1536)
      )
    `);
  } finally {
    client.release();
  }
}

