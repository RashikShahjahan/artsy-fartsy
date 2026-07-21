-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documents table
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  prompt TEXT NOT NULL,
  code TEXT NOT NULL,
  art_type TEXT NOT NULL,
  embedding vector(512)
);

-- Persist vectors that still need to be replicated to Upstash.
CREATE TABLE IF NOT EXISTS vector_outbox (
  document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  embedding JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
