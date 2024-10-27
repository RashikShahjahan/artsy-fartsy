import axios from 'axios';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;


export async function retrieveArtCode(userPrompt: string, token: string) {
  const response = await axios.post(`${apiBaseUrl}/generate_code`, 
    { userPrompt },
    { 
      headers: { Authorization: `Bearer ${token}` },
    }
  );
  return response.data;
}

export async function runDrawingCode(code: string, token: string) {
  const response = await axios.post(`${apiBaseUrl}/run_code`, 
    { code },
    { headers: { Authorization: `Bearer ${token}` } }
  );
  return response.data;
}

export async function modifyDrawing(code: string, userPrompt: string, token: string) {
  const response = await axios.post(`${apiBaseUrl}/modify_art_code`, 
    { code, userPrompt },
    { headers: { Authorization: `Bearer ${token}` } }
  );
  return response.data;
}