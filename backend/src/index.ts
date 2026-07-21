import cors from 'cors';
import express from 'express';
import { rateLimit } from 'express-rate-limit';
import fs from 'node:fs';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import {
  EditCodeSchema,
  FindSimilarSchema,
  GenerateCodeSchema,
  RunCodeSchema,
  StoreCodeSchema,
} from '../../shared/schemas';
import { executeArtCode, type ArtExecution } from './utils/codeExecution';
import { initializeDatabase } from './utils/db';
import { findSimilarDocuments, flushPendingVectors, storeDocument } from './utils/embeddings';
import { createExecutionToken, InvalidExecutionTokenError, verifyExecutionToken } from './utils/executionToken';
import { editArtCode, generateArtCode } from './utils/generation';
import {
  prepareSandboxStorage,
  SandboxBusyError,
  SandboxExecutionError,
  SandboxUnavailableError,
  verifySandboxRuntime,
} from './utils/sandbox';

const app = express();
const PORT = process.env.PORT || 8000;
const __dirname = path.dirname(fileURLToPath(import.meta.url));

if (process.env.TRUST_PROXY_HOPS) {
  const trustProxyHops = Number(process.env.TRUST_PROXY_HOPS);
  if (!Number.isSafeInteger(trustProxyHops) || trustProxyHops < 1) {
    throw new Error('TRUST_PROXY_HOPS must be a positive integer');
  }
  app.set('trust proxy', trustProxyHops);
}
app.use(express.json({ limit: '128kb' }));
app.use(cors({ exposedHeaders: ['X-Execution-Token'] }));
const apiLimiter = rateLimit({
  windowMs: 60_000,
  limit: 60,
  standardHeaders: 'draft-7',
  legacyHeaders: false,
});
app.use(['/generate_code', '/edit_code', '/run_code', '/store_code', '/find_similar'], apiLimiter);
const executionLimiter = rateLimit({
  windowMs: 60_000,
  limit: 12,
  standardHeaders: 'draft-7',
  legacyHeaders: false,
});

await prepareSandboxStorage();
await verifySandboxRuntime();
await initializeDatabase();
void flushPendingVectors().catch((error) => console.error('Failed to flush vector outbox at startup:', error));
const vectorOutboxInterval = setInterval(() => {
  void flushPendingVectors().catch((error) => console.error('Failed to flush vector outbox:', error));
}, 30_000);
vectorOutboxInterval.unref();

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function errorStatus(error: unknown): number {
  const isZodError = error instanceof Error
    && error.name === 'ZodError'
    && Array.isArray((error as Error & { issues?: unknown }).issues);
  if (isZodError || error instanceof InvalidExecutionTokenError) return 400;
  if (error instanceof SandboxBusyError || error instanceof SandboxUnavailableError) return 503;
  if (error instanceof SandboxExecutionError) return 422;
  return 500;
}

app.post('/generate_code', async (req, res) => {
  try {
    const { userPrompt, artType } = GenerateCodeSchema.parse(req.body);
    const generatedCode = await generateArtCode(userPrompt, artType);
    res.status(200).json({ code: generatedCode });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: errorMessage(error, 'Failed to generate code') });
  }
});

app.post('/edit_code', async (req, res) => {
  try {
    const { prompt, code, artType } = EditCodeSchema.parse(req.body);
    const editedCode = await editArtCode(prompt, code, artType);
    res.status(200).json({ code: editedCode });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: errorMessage(error, 'Failed to edit code') });
  }
});

app.post('/run_code', executionLimiter, async (req, res) => {
  let execution: ArtExecution | undefined;
  const abortController = new AbortController();
  const abortOnClose = () => {
    if (!res.writableEnded) abortController.abort();
  };
  const deadline = setTimeout(() => {
    abortController.abort();
    res.destroy();
  }, 90_000);
  deadline.unref();
  res.once('close', abortOnClose);

  try {
    const { code, artType } = RunCodeSchema.parse(req.body);
    execution = await executeArtCode(code, artType, abortController.signal);

    res.status(200);
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', 'attachment; filename="drawing.png"');
    res.setHeader('X-Execution-Token', createExecutionToken(code, artType));
    await pipeline(fs.createReadStream(execution.outputPath), res);
  } catch (error) {
    console.error('Error running code:', error);
    if (res.destroyed || res.headersSent) {
      res.destroy();
      return;
    }
    res.status(errorStatus(error)).json({
      error: errorMessage(error, 'Failed to run code'),
      type: error instanceof SandboxExecutionError ? 'execution_error' : 'server_error',
    });
  } finally {
    clearTimeout(deadline);
    res.off('close', abortOnClose);
    await execution?.cleanup().catch((error) => console.error('Failed to clean sandbox workspace:', error));
  }
});

app.post('/store_code', async (req, res) => {
  try {
    const { prompt, code, artType, executionToken } = StoreCodeSchema.parse(req.body);
    verifyExecutionToken(executionToken, code, artType);
    await storeDocument(prompt, code, artType);
    res.status(200).json({ success: true });
  } catch (error) {
    res.status(errorStatus(error)).json({ error: errorMessage(error, 'Failed to store code') });
  }
});

app.post('/find_similar', executionLimiter, async (req, res) => {
  const abortController = new AbortController();
  const abortOnClose = () => {
    if (!res.writableEnded) abortController.abort();
  };
  const deadline = setTimeout(() => {
    abortController.abort();
    res.destroy();
  }, 90_000);
  deadline.unref();
  res.once('close', abortOnClose);

  try {
    const { prompt, artType } = FindSimilarSchema.parse(req.body);
    const similarCode = await findSimilarDocuments(prompt, artType);
    const images: string[] = [];

    for (const code of similarCode) {
      let execution: ArtExecution | undefined;
      try {
        execution = await executeArtCode(code, artType, abortController.signal);
        const imageBuffer = await fs.promises.readFile(execution.outputPath);
        images.push(`data:image/png;base64,${imageBuffer.toString('base64')}`);
      } finally {
        await execution?.cleanup();
      }
    }

    res.status(200).json({ images });
  } catch (error) {
    console.error('Error finding similar documents:', error);
    if (!res.destroyed && !res.headersSent) {
      res.status(errorStatus(error)).json({ error: errorMessage(error, 'Failed to find similar drawings') });
    }
  } finally {
    clearTimeout(deadline);
    res.off('close', abortOnClose);
  }
});

app.use(express.static(path.join(__dirname, '..', '..', 'frontend', 'dist')));

app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', '..', 'frontend', 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
