import { executeDrawingInSandbox, type ArtExecution } from './sandbox';

export async function executeArtCode(code: string, artType: string, signal?: AbortSignal): Promise<ArtExecution> {
  if (artType !== 'drawing') {
    throw new Error(`Unsupported art type: ${artType}`);
  }

  return executeDrawingInSandbox(code, signal);
}

export type { ArtExecution } from './sandbox';
