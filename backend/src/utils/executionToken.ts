import { createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto';

const TOKEN_TTL_MS = 60 * 60 * 1000;

function getTokenSecret(): string {
  if (process.env.EXECUTION_TOKEN_SECRET) {
    if (process.env.EXECUTION_TOKEN_SECRET.length < 32) {
      throw new Error('EXECUTION_TOKEN_SECRET must be at least 32 characters');
    }
    return process.env.EXECUTION_TOKEN_SECRET;
  }
  if (process.env.NODE_ENV === 'production') {
    throw new Error('EXECUTION_TOKEN_SECRET is required in production');
  }
  return randomBytes(32).toString('base64url');
}

const tokenSecret = getTokenSecret();

interface ExecutionTokenPayload {
  codeHash: string;
  artType: 'drawing';
  expiresAt: number;
}

export class InvalidExecutionTokenError extends Error {
  constructor() {
    super('Code must be run successfully before it can be saved');
    this.name = 'InvalidExecutionTokenError';
  }
}

function hashCode(code: string): string {
  return createHash('sha256').update(code).digest('base64url');
}

function sign(payload: string, secret: string): string {
  return createHmac('sha256', secret).update(payload).digest('base64url');
}

export function createExecutionToken(
  code: string,
  artType: 'drawing',
  now = Date.now(),
  secret = tokenSecret,
): string {
  const payload: ExecutionTokenPayload = {
    codeHash: hashCode(code),
    artType,
    expiresAt: now + TOKEN_TTL_MS,
  };
  const encodedPayload = Buffer.from(JSON.stringify(payload)).toString('base64url');
  return `${encodedPayload}.${sign(encodedPayload, secret)}`;
}

export function verifyExecutionToken(
  token: string,
  code: string,
  artType: 'drawing',
  now = Date.now(),
  secret = tokenSecret,
): void {
  const [encodedPayload, providedSignature, extra] = token.split('.');
  if (!encodedPayload || !providedSignature || extra) {
    throw new InvalidExecutionTokenError();
  }

  const expectedSignature = new TextEncoder().encode(sign(encodedPayload, secret));
  const signature = new TextEncoder().encode(providedSignature);
  if (signature.length !== expectedSignature.length || !timingSafeEqual(signature, expectedSignature)) {
    throw new InvalidExecutionTokenError();
  }

  try {
    const payload = JSON.parse(Buffer.from(encodedPayload, 'base64url').toString('utf8')) as ExecutionTokenPayload;
    if (payload.expiresAt <= now || payload.artType !== artType || payload.codeHash !== hashCode(code)) {
      throw new InvalidExecutionTokenError();
    }
  } catch (error) {
    if (error instanceof InvalidExecutionTokenError) {
      throw error;
    }
    throw new InvalidExecutionTokenError();
  }
}
