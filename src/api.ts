import axios from 'axios';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

export async function submitDrawingCommands(code: string, token: string) {
  const response = await axios.post(`${apiBaseUrl}/interpret`, { code }, {
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });
  
  if (response.status !== 200) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.data;
}

export async function saveDrawingToServer(blob: Blob, token: string) {
  const formData = new FormData();
  formData.append('image', blob, 'drawing.png');

  await axios.post(`${apiBaseUrl}/save_drawing`, formData, {
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'multipart/form-data',
    },
  });
}
