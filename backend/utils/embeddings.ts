import { pool } from './db';
import { VoyageAIClient } from "voyageai";

const client = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });

async function generateEmbedding(text: string): Promise<number[]> {
    const response = await client.embed({
        input: text,

        model: "voyage-code-2",
    });
    return response.data[0].embedding;
}

function arrayToVector(arr: number[]): string {
  const formattedNumbers = arr.map(num => num.toFixed(8));
  return `[${formattedNumbers.join(',')}]`;
}

export async function storeDocument(content: string): Promise<void> {
  const embedding = await generateEmbedding(content);
  const vectorString = arrayToVector(embedding);
  await pool.query(
    'INSERT INTO documents (content, embedding) VALUES ($1, $2::vector)',
    [content, vectorString]
  );
}

export async function findSimilarDocuments(query: string, limit: number = 3): Promise<string[]> {
  const queryEmbedding = await generateEmbedding(query);
  const vectorString = arrayToVector(queryEmbedding);
  const result = await pool.query(
    `SELECT content, embedding <-> $1::vector as distance
     FROM documents
     ORDER BY distance ASC
     LIMIT $2`,
    [vectorString, limit]
  );

  return result.rows.map((row) => row.content + ", distance: " + row.distance);
}