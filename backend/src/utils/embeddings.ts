import { pool } from './db';
import { VoyageAIClient } from "voyageai";

const client = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });

async function generateEmbedding(text: string): Promise<number[]> {
    const response = await client.embed({
        input: text,
        model: "voyage-3",
    });
    
    if (!response.data) {
        throw new Error('Failed to generate embedding: No response data received');
    }
    
    if (!response.data[0]) {
        throw new Error('Failed to generate embedding: Empty response data array');
    }
    
    if (!response.data[0].embedding) {
        throw new Error('Failed to generate embedding: No embedding found in response');
    }
    
    return response.data[0].embedding;
}

function arrayToVector(arr: number[]): string {
  const formattedNumbers = arr.map(num => num.toFixed(8));
  return `[${formattedNumbers.join(',')}]`;
}

export async function storeDocument(prompt: string, code: string): Promise<void> {
  const embedding = await generateEmbedding(prompt);
  const vectorString = arrayToVector(embedding);
  await pool.query(
    'INSERT INTO documents (prompt, code, embedding) VALUES ($1, $2, $3::vector)',
    [prompt, code, vectorString]
  );
}

export async function findSimilarDocuments(query: string, limit: number = 3): Promise<string[]> {
  const queryEmbedding = await generateEmbedding(query);
  const vectorString = arrayToVector(queryEmbedding);
  const result = await pool.query(
    `SELECT prompt, code, embedding <-> $1::vector as distance
     FROM documents
     ORDER BY distance ASC
     LIMIT $2`,
    [vectorString, limit]
  );

  return result.rows.map((row) => row.code);
}