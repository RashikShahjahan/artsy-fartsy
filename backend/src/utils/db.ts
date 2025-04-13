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

// Flag to track if vector extension is available
export let vectorExtensionAvailable = false;

export async function initializeDatabase() {
  const client = await pool.connect();
  try {
    try {
      await client.query("CREATE EXTENSION IF NOT EXISTS vector");
      vectorExtensionAvailable = true;
      console.log("Vector extension is available and enabled");
    } catch (error) {
      console.error("Vector extension is not available:", error.message);
      console.log("Application will run with limited functionality - similarity search will be disabled");
      vectorExtensionAvailable = false;
    }

    // Create tables regardless of vector extension status
    if (vectorExtensionAvailable) {
      await client.query(`
        CREATE TABLE IF NOT EXISTS documents (
          id SERIAL PRIMARY KEY,
          prompt TEXT NOT NULL,
          code TEXT NOT NULL,
          art_type TEXT NOT NULL,
          embedding vector(512)
        )
      `);
    } else {
      // Create table without vector column if extension isn't available
      await client.query(`
        CREATE TABLE IF NOT EXISTS documents (
          id SERIAL PRIMARY KEY,
          prompt TEXT NOT NULL,
          code TEXT NOT NULL,
          art_type TEXT NOT NULL
        )
      `);
    }

    await client.query(`
      CREATE TABLE IF NOT EXISTS errors (
        id SERIAL PRIMARY KEY,
        code TEXT NOT NULL,
        error TEXT NOT NULL
      )
    `);
  } finally {
    client.release();
  }
}

