import axios from 'axios';
import { Command } from './types';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

export async function submitDrawingCommands(code: string, token: string) {
  const response = await axios.post(`${apiBaseUrl}/interpret`, { code }, {
    withCredentials: true,
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

export async function saveDrawingToServer(drawCommands: Command[], token: string) {
  await axios.post(`${apiBaseUrl}/save_art`, { drawCommands }, {
    withCredentials: true,
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });
}

export async function fetchArtFromServer(skip: number, token: string) {
  const response = await axios.get(`${apiBaseUrl}/get_art`, {
    params: { skip },
    withCredentials: true,
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  return response.data;
}

export async function fetchArtCountFromServer(token: string) {
  const response = await axios.get(`${apiBaseUrl}/get_art_count`, {
    headers: { Authorization: `Bearer ${token}` },
      });
  return response.data;
}



export async function fetchLikes(artId: string, token: string) {
  const response = await axios.get(`${apiBaseUrl}/get_likes`, { params: { artId }, headers: { Authorization: `Bearer ${token}` } });
  return response.data;
}

export async function toggleLikeRequest(artId: string, token: string) {
  const response = await axios.post(`${apiBaseUrl}/toggle_like`, { artId }, { headers: { Authorization: `Bearer ${token}` } });
  return response.data;
}

export async function isLikedRequest(artId: string, token: string) {
  const response = await axios.get(`${apiBaseUrl}/is_liked`, { params: { artId }, headers: { Authorization: `Bearer ${token}` } });
  return response.data;
}

export async function aiDrawingCommands(userPrompt: string, token: string) {
  const response = await axios.post(`${apiBaseUrl}/generate_code`, 
    { userPrompt },
    { 
      headers: { Authorization: `Bearer ${token}` },
    }
  );
  return response.data;
}
