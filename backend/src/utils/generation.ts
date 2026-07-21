import { OpenRouter } from '@openrouter/sdk';
import { ARTCANVAS_EDIT_GUIDE, ARTCANVAS_GUIDE } from './prompts/drawing';
import { executeDrawingInSandbox, SandboxExecutionError } from './sandbox';

const MODEL = 'openrouter/free';
const MAX_CODE_ATTEMPTS = 3;

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

async function requestExecutableCode(instructions: string, input: string): Promise<string> {
  let validationFeedback = '';

  for (let attempt = 0; attempt < MAX_CODE_ATTEMPTS; attempt += 1) {
    const result = getOpenRouter().callModel({
      model: MODEL,
      maxOutputTokens: 8000,
      instructions,
      input: `${input}${validationFeedback}`,
    });
    const code = cleanGeneratedCode(await result.getText());

    if (!code) {
      validationFeedback = '\n\nThe previous response was empty. Return complete executable Python code.';
      continue;
    }

    try {
      const execution = await executeDrawingInSandbox(code);
      await execution.cleanup();
      return code;
    } catch (error) {
      if (!(error instanceof SandboxExecutionError)) throw error;
      validationFeedback = [
        '\n\nThe previous Python code failed validation. Return corrected, complete code.',
        'Validation error:',
        error.message.slice(0, 2000),
      ].join('\n');
    }
  }

  throw new Error('OpenRouter could not generate executable drawing code');
}

async function generateArtCode(prompt: string, artType: string): Promise<string> {
  if (artType !== 'drawing') {
    throw new Error(`Unsupported art type: ${artType}`);
  }

  return requestExecutableCode(ARTCANVAS_GUIDE, `Drawing request:\n${prompt}`);
}

async function editArtCode(prompt: string, code: string, artType: string): Promise<string> {
  if (artType !== 'drawing') {
    throw new Error(`Unsupported art type: ${artType}`);
  }

  return requestExecutableCode(
    ARTCANVAS_EDIT_GUIDE,
    `Requested changes:\n${prompt}\n\nCurrent code:\n${code}`,
  );
}

export { editArtCode, generateArtCode };
