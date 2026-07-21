import { describe, expect, test } from 'bun:test';
import { createPoolConfig } from './db';

describe('database TLS configuration', () => {
  test('verifies certificates by default and strips insecure URL overrides', () => {
    const config = createPoolConfig({
      DATABASE_URL: 'postgres://user:pass@example.com:5432/app?sslmode=require',
    });

    expect(config.connectionString).not.toContain('sslmode');
    expect(config.ssl).toEqual({ rejectUnauthorized: true });
  });

  test('supports a private certificate authority', () => {
    const config = createPoolConfig({
      DATABASE_URL: 'postgres://user:pass@example.com/app',
      DATABASE_SSL: 'true',
      DATABASE_CA_CERT: 'line one\\nline two',
    });

    expect(config.ssl).toEqual({ rejectUnauthorized: true, ca: 'line one\nline two' });
  });

  test('disables TLS only when explicitly configured for local development', () => {
    const config = createPoolConfig({
      DATABASE_URL: 'postgres://postgres:postgres@localhost:15432/codeart',
      DATABASE_SSL: 'false',
    });

    expect(config.ssl).toBe(false);
  });
});
