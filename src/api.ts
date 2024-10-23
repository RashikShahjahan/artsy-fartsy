import axios from 'axios';
import { Command } from './types';

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

export async function saveDrawingToServer(drawCommands: Command[], token: string) {
  await axios.post(`${apiBaseUrl}/save_art`, { drawCommands }, {
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });
}

export async function fetchDrawCommands(artId: number, token: string) {
  const response = await axios.get(`${apiBaseUrl}/get_art/${artId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return response.data;
}
