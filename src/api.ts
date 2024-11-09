import axios from 'axios';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

export async function retrieveArtCode(userPrompt: string) {
  const response = await axios.post(`${apiBaseUrl}/generate_code`, 
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
  const response = await axios.post(`${apiBaseUrl}/run_code`, 
    { code },
    { 
      responseType: 'blob'
    }
  );
  return URL.createObjectURL(response.data);
}

export async function storeCode(code: string) {
  const response = await axios.post(`${apiBaseUrl}/store_code`, 
    { code },
    { headers: { 'Content-Type': 'application/json' } }
  );
  return response.data;
}