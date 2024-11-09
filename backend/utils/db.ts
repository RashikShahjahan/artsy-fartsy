// src/db.ts
import { Pool } from "pg";

export const pool = new Pool({
  user: "postgres",
  password: "postgres",
  host: "localhost",
  port: 15432,
  database: "ragdemo",
});


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

