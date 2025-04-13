import { pool, vectorExtensionAvailable } from './db';
import { VoyageAIClient } from "voyageai";

const client = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });

async function generateEmbedding(text: string): Promise<number[]> {
    // Only make API call if vector extension is available
    if (!vectorExtensionAvailable) {
        console.log("Skipping embedding generation since vector extension is not available");
        return [];
    }
    
    const response = await client.embed({
        input: text,
        model: "voyage-3-lite",
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
  if (arr.length === 0) return '';
  const formattedNumbers = arr.map(num => num.toFixed(8));
  return `[${formattedNumbers.join(',')}]`;
}

export async function storeDocument(prompt: string, code: string, artType: string): Promise<void> {
  try {
    if (vectorExtensionAvailable) {
      const embedding = await generateEmbedding(prompt);
      const vectorString = arrayToVector(embedding);
      await pool.query(
        'INSERT INTO documents (prompt, code, art_type, embedding) VALUES ($1, $2, $3, $4::vector)',
        [prompt, code, artType, vectorString]
      );
    } else {
      // Store without the embedding if vector extension is not available
      await pool.query(
        'INSERT INTO documents (prompt, code, art_type) VALUES ($1, $2, $3)',
        [prompt, code, artType]
      );
    }
  } catch (error) {
    console.error("Error storing document:", error);
    // Store document without embedding as fallback
    try {
      await pool.query(
        'INSERT INTO documents (prompt, code, art_type) VALUES ($1, $2, $3)',
        [prompt, code, artType]
      );
    } catch (fallbackError) {
      console.error("Failed fallback document storage:", fallbackError);
      throw fallbackError;
    }
  }
}

export async function findSimilarDocuments(query: string, limit: number = 3): Promise<string[]> {
  if (!vectorExtensionAvailable) {
    console.log("Vector search unavailable, returning random documents instead");
    // Fallback to returning random documents when vector extension isn't available
    const result = await pool.query(
      `SELECT prompt, code FROM documents ORDER BY RANDOM() LIMIT $1`,
      [limit]
    );
    return result.rows.map((row) => row.code);
  }
  
  try {
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
  } catch (error) {
    console.error("Error in vector search, falling back to random selection:", error);
    // Fallback to returning random documents on error
    const result = await pool.query(
      `SELECT prompt, code FROM documents ORDER BY RANDOM() LIMIT $1`,
      [limit]
    );
    return result.rows.map((row) => row.code);
  }
}