import { z } from 'zod';

const ArtTypeSchema = z.literal('drawing');

export const GenerateCodeSchema = z.object({
  userPrompt: z.string().min(1, "Prompt is required"),
  artType: ArtTypeSchema
});

export const RunCodeSchema = z.object({
  code: z.string().min(1, "Code is required").max(100_000, "Code is too large"),
  artType: ArtTypeSchema
});

export const StoreCodeSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  code: z.string().min(1, "Code is required").max(100_000, "Code is too large"),
  artType: ArtTypeSchema,
  executionToken: z.string().min(1, "Execution token is required").max(2048, "Execution token is too large")
});

export const FindSimilarSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  artType: ArtTypeSchema
});

export const EditCodeSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  code: z.string().min(1, "Code is required").max(100_000, "Code is too large"),
  artType: ArtTypeSchema
});

// Response schemas
export const GenerateCodeResponseSchema = z.object({
  code: z.string()
});

export const FindSimilarResponseSchema = z.object({
  images: z.array(z.string())
});

export const EditCodeResponseSchema = z.object({
  code: z.string()
});

// Infer types from schemas
export type GenerateCodeRequest = z.infer<typeof GenerateCodeSchema>;
export type RunCodeRequest = z.infer<typeof RunCodeSchema>;
export type StoreCodeRequest = z.infer<typeof StoreCodeSchema>;
export type FindSimilarRequest = z.infer<typeof FindSimilarSchema>;
export type GenerateCodeResponse = z.infer<typeof GenerateCodeResponseSchema>;
export type FindSimilarResponse = z.infer<typeof FindSimilarResponseSchema>; 
export type EditCodeRequest = z.infer<typeof EditCodeSchema>;
export type EditCodeResponse = z.infer<typeof EditCodeResponseSchema>;
