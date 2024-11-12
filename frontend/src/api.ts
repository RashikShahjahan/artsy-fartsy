import axios from 'axios';
import { 
  GenerateCodeSchema, 
  RunCodeSchema, 
  StoreCodeSchema, 
  FindSimilarSchema,
  GenerateCodeResponseSchema,
  FindSimilarResponseSchema
} from '../../shared/schemas';

const BASE_URL = import.meta.env.PROD ? '' : 'http://localhost:8000';

export async function retrieveArtCode(userPrompt: string) {
  const validatedData = GenerateCodeSchema.parse({ userPrompt });
  const response = await axios.post(`${BASE_URL}/generate_code`, validatedData);
  return GenerateCodeResponseSchema.parse(response.data);
}

export async function runDrawingCode(code: string): Promise<string> {
  const validatedData = RunCodeSchema.parse({ code });
  const response = await axios.post(`${BASE_URL}/run_code`, validatedData, { 
    responseType: 'blob'
  });
  return URL.createObjectURL(response.data);
}

export async function storeCode(code: string): Promise<void> {
  const validatedData = StoreCodeSchema.parse({ code });
  await axios.post(`${BASE_URL}/store_code`, validatedData);
}

export async function findSimilarDrawing(prompt: string): Promise<string[]> {
  const validatedData = FindSimilarSchema.parse({ prompt });
  const response = await axios.post(`${BASE_URL}/find_similar`, validatedData);
  return FindSimilarResponseSchema.parse(response.data).images;
}