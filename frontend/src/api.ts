import axios from 'axios';
import { 
  GenerateCodeSchema, 
  RunCodeSchema, 
  StoreCodeSchema, 
  FindSimilarSchema,
  GenerateCodeResponseSchema,
  FindSimilarResponseSchema,
  EditCodeSchema,
  EditCodeResponseSchema
} from '../../shared/schemas';

const BASE_URL = import.meta.env.PROD ? '' : 'http://localhost:8000';

export async function retrieveArtCode(userPrompt: string, artType: string): Promise<string> {
  const validatedData = GenerateCodeSchema.parse({ userPrompt, artType });
  const response = await axios.post(`${BASE_URL}/generate_code`, validatedData);
  return GenerateCodeResponseSchema.parse(response.data).code;
}

export async function runArtCode(code: string, artType: string, ranByAI: boolean): Promise<string> {
  const validatedData = RunCodeSchema.parse({ code, artType, ranByAI });
  const response = await axios.post(`${BASE_URL}/run_code`, validatedData, { 
    responseType: 'blob'
  });
  return URL.createObjectURL(response.data);
}

export async function storeCode(prompt: string, code: string, artType: string): Promise<boolean> {
  const validatedData = StoreCodeSchema.parse({ prompt, code, artType });
  const response = await axios.post(`${BASE_URL}/store_code`, validatedData);
  return response.status === 200;
}

export async function findSimilarArt(prompt: string, artType: string): Promise<string[]> {
  const validatedData = FindSimilarSchema.parse({ prompt, artType });
  const response = await axios.post(`${BASE_URL}/find_similar`, validatedData);
  return FindSimilarResponseSchema.parse(response.data).images;
}

export async function editArtCode(prompt: string, code: string, artType: string): Promise<string> {
  const validatedData = EditCodeSchema.parse({ prompt, code, artType });
  const response = await axios.post(`${BASE_URL}/edit_code`, validatedData);
  return EditCodeResponseSchema.parse(response.data).code;
}