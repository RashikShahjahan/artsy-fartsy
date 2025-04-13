import { pool, vectorExtensionAvailable } from './db';
import { VoyageAIClient } from "voyageai";
import fetch from 'node-fetch';

const client = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });
const VECTOR_ENDPOINT = process.env.VECTOR_ENDPOINT;
const VECTOR_TOKEN = process.env.VECTOR_TOKEN;
const VECTOR_READONLY_TOKEN = process.env.VECTOR_READONLY_TOKEN;
const VECTOR_INDEX = 'documents';

// Check if Upstash Vector is available
const upstashVectorAvailable = !!(VECTOR_ENDPOINT && (VECTOR_TOKEN || VECTOR_READONLY_TOKEN));

async function generateEmbedding(text: string): Promise<number[]> {
    // Skip generation if neither vector service is available
    if (!vectorExtensionAvailable && !upstashVectorAvailable) {
        console.log("Skipping embedding generation since no vector service is available");
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

// Store documents in Upstash Vector
async function storeInUpstash(id: string, embedding: number[]): Promise<void> {
  try {
    if (!upstashVectorAvailable) return;
    
    const response = await fetch(`https://${VECTOR_ENDPOINT}/vectorset/${VECTOR_INDEX}/vector`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${VECTOR_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        id: id,
        vector: embedding
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to store vector in Upstash: ${response.statusText}`);
    }
  } catch (error) {
    console.error("Error storing vector in Upstash:", error);
  }
}

// Search for similar documents in Upstash Vector
async function searchInUpstash(embedding: number[], limit: number): Promise<string[]> {
  try {
    if (!upstashVectorAvailable) return [];
    
    const response = await fetch(`https://${VECTOR_ENDPOINT}/vectorset/${VECTOR_INDEX}/search`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${VECTOR_READONLY_TOKEN || VECTOR_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        vector: embedding,
        topK: limit
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to search vectors in Upstash: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    // Get the IDs from the search results
    const ids = result.vectors.map((v: any) => v.id);
    
    // Fetch the actual documents from PostgreSQL
    if (ids.length > 0) {
      const dbResult = await pool.query(
        `SELECT code FROM documents WHERE id = ANY($1)`,
        [ids]
      );
      return dbResult.rows.map((row) => row.code);
    }
    
    return [];
  } catch (error) {
    console.error("Error searching vectors in Upstash:", error);
    return [];
  }
}

export async function storeDocument(prompt: string, code: string, artType: string): Promise<void> {
  try {
    let id: string;
    
    // Store document in PostgreSQL
    const result = await pool.query(
      'INSERT INTO documents (prompt, code, art_type) VALUES ($1, $2, $3) RETURNING id',
      [prompt, code, artType]
    );
    id = result.rows[0].id;
    
    // Store embedding in PostgreSQL if pgvector is available
    if (vectorExtensionAvailable) {
      const embedding = await generateEmbedding(prompt);
      const vectorString = arrayToVector(embedding);
      await pool.query(
        'UPDATE documents SET embedding = $1::vector WHERE id = $2',
        [vectorString, id]
      );
      
      // Also store in Upstash if available
      if (upstashVectorAvailable) {
        await storeInUpstash(id.toString(), embedding);
      }
    }
    // If pgvector not available but Upstash is available
    else if (upstashVectorAvailable) {
      const embedding = await generateEmbedding(prompt);
      await storeInUpstash(id.toString(), embedding);
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
  // Try Upstash Vector first if available
  if (upstashVectorAvailable) {
    try {
      const queryEmbedding = await generateEmbedding(query);
      const results = await searchInUpstash(queryEmbedding, limit);
      if (results.length > 0) {
        return results;
      }
    } catch (error) {
      console.error("Error in Upstash vector search:", error);
    }
  }
  
  // Try pgvector if available
  if (vectorExtensionAvailable) {
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
      console.error("Error in PostgreSQL vector search:", error);
    }
  }
  
  // Fallback to random selection if both vector services fail
  console.log("Vector search unavailable, returning random documents instead");
  const result = await pool.query(
    `SELECT prompt, code FROM documents ORDER BY RANDOM() LIMIT $1`,
    [limit]
  );
  return result.rows.map((row) => row.code);
}