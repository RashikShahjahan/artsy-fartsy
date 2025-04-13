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

-- Create the errors table
CREATE TABLE IF NOT EXISTS errors (
  id SERIAL PRIMARY KEY,
  code TEXT NOT NULL,
  error TEXT NOT NULL
); 