import { z } from 'zod';

export const GenerateCodeSchema = z.object({
  userPrompt: z.string().min(1, "Prompt is required"),
  artType: z.string().min(1, "Art type is required")
});

export const RunCodeSchema = z.object({
  code: z.string().min(1, "Code is required"),
  artType: z.string().min(1, "Art type is required")
});

export const StoreCodeSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  code: z.string().min(1, "Code is required"),
  artType: z.string().min(1, "Art type is required")
});

export const FindSimilarSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  artType: z.string().min(1, "Art type is required")
});

export const EditCodeSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  code: z.string().min(1, "Code is required"),
  artType: z.string().min(1, "Art type is required")
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