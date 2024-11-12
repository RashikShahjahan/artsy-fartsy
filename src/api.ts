import axios from 'axios';

const BASE_URL = import.meta.env.VITE_NODE_ENV === 'development' 
  ? 'http://localhost:8000'
  : '';

export async function retrieveArtCode(userPrompt: string) {
  const response = await axios.post(`${BASE_URL}/generate_code`, 
    { userPrompt },
    { 
      headers: { 
        'Content-Type': 'application/json'
      }
    }
  );
  return response.data;
}

export async function runDrawingCode(code: string) {
  const response = await axios.post(`${BASE_URL}/run_code`, 
    { code },
    { 
      responseType: 'blob'
    }
  );
  return URL.createObjectURL(response.data);
}

export async function storeCode(code: string) {
  const response = await axios.post(`${BASE_URL}/store_code`, 
    { code },
    { headers: { 'Content-Type': 'application/json' } }
  );
  return response.data;
}

export async function findSimilarDrawing(prompt: string): Promise<string[]> {
  const response = await axios.post(`${BASE_URL}/find_similar`, 
    { prompt },
    { headers: { 'Content-Type': 'application/json' } }
  );
  return response.data.images;
}