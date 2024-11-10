import axios from 'axios';


export async function retrieveArtCode(userPrompt: string) {
  const response = await axios.post(`/generate_code`, 
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
  const response = await axios.post(`/run_code`, 
    { code },
    { 
      responseType: 'blob'
    }
  );
  return URL.createObjectURL(response.data);
}

export async function storeCode(code: string) {
  const response = await axios.post(`/store_code`, 
    { code },
    { headers: { 'Content-Type': 'application/json' } }
  );
  return response.data;
}

export async function findSimilarDrawing(prompt: string): Promise<string[]> {
  const response = await axios.post(`/find_similar`, 
    { prompt },
    { headers: { 'Content-Type': 'application/json' } }
  );
  return response.data.images;
}