import { describe, expect, test } from 'bun:test';
import { createExecutionToken, verifyExecutionToken } from './executionToken';

describe('execution tokens', () => {
  const secret = 'test-secret';
  const now = 1_000;

  test('accepts the exact code that produced a drawing', () => {
    const token = createExecutionToken('print("drawing")', 'drawing', now, secret);
    expect(() => verifyExecutionToken(token, 'print("drawing")', 'drawing', now + 1, secret)).not.toThrow();
  });

  test('rejects changed code', () => {
    const token = createExecutionToken('print("drawing")', 'drawing', now, secret);
    expect(() => verifyExecutionToken(token, 'print("changed")', 'drawing', now + 1, secret)).toThrow();
  });

  test('rejects tampered and expired tokens', () => {
    const token = createExecutionToken('print("drawing")', 'drawing', now, secret);
    expect(() => verifyExecutionToken(`${token}x`, 'print("drawing")', 'drawing', now + 1, secret)).toThrow();
    expect(() => verifyExecutionToken(token, 'print("drawing")', 'drawing', now + 3_600_001, secret)).toThrow();
  });
});
