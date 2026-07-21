import { OpenRouter } from '@openrouter/sdk';
import { ARTCANVAS_EDIT_GUIDE, ARTCANVAS_GUIDE } from './prompts/drawing';

const MODEL = 'openrouter/free';

let openrouter: OpenRouter | undefined;

function getOpenRouter(): OpenRouter {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY is required');
  }

  openrouter ??= new OpenRouter({
    apiKey,
    httpReferer: process.env.APP_URL,
    appTitle: 'Artsy Fartsy',
    timeoutMs: 60_000,
    retryConfig: { strategy: 'none' },
  });

  return openrouter;
}

function cleanGeneratedCode(code: string): string {
  return code
    .split('\n')
    .filter((line) => !line.trimStart().startsWith('```') && line.trim())
    .join('\n')
    .trim();
}

async function generateArtCode(prompt: string, artType: string): Promise<string> {
  if (artType !== 'drawing') {
    throw new Error(`Unsupported art type: ${artType}`);
  }

  const result = getOpenRouter().callModel({
    model: MODEL,
    maxOutputTokens: 8000,
    input: `${ARTCANVAS_GUIDE} ${prompt} Only respond with code as plain text without code block syntax around it`,
  });

  const code = cleanGeneratedCode(await result.getText());
  if (!code) {
    throw new Error('OpenRouter returned an empty response');
  }

  return code;
}

async function editArtCode(prompt: string, code: string, artType: string): Promise<string> {
  if (artType !== 'drawing') {
    throw new Error(`Unsupported art type: ${artType}`);
  }

  const result = getOpenRouter().callModel({
    model: MODEL,
    maxOutputTokens: 8000,
    input: `${ARTCANVAS_EDIT_GUIDE} ${prompt} Only respond with code as plain text without code block syntax around it ${code}`,
  });

  const editedCode = cleanGeneratedCode(await result.getText());
  if (!editedCode) {
    throw new Error('OpenRouter returned an empty response');
  }

  return editedCode;
}

export { editArtCode, generateArtCode };
