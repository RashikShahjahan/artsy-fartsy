import { afterEach, describe, expect, test } from 'bun:test';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { deflateSync } from 'node:zlib';
import {
  buildSandboxInvocation,
  SandboxExecutionError,
  type SandboxPaths,
  validatePngOutput,
} from './sandbox';

const temporaryPaths: string[] = [];
const pngSignature = Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a);

const testCrcTable = Uint32Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit += 1) {
    value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  return value >>> 0;
});

function testCrc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = testCrcTable[(crc ^ byte) & 0xff]! ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function concatenate(parts: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(parts.reduce((length, part) => length + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const typeBytes = new TextEncoder().encode(type);
  const chunk = new Uint8Array(12 + data.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, data.length);
  chunk.set(typeBytes, 4);
  chunk.set(data, 8);
  view.setUint32(8 + data.length, testCrc32(chunk.subarray(4, 8 + data.length)));
  return chunk;
}

function validPng(width: number, height: number): Uint8Array {
  const header = new Uint8Array(13);
  const headerView = new DataView(header.buffer);
  headerView.setUint32(0, width);
  headerView.setUint32(4, height);
  header.set([8, 2, 0, 0, 0], 8);
  const pixels = new Uint8Array((1 + width * 3) * height);
  return concatenate([
    pngSignature,
    pngChunk('IHDR', header),
    pngChunk('IDAT', Uint8Array.from(deflateSync(pixels))),
    pngChunk('IEND', new Uint8Array()),
  ]);
}

afterEach(async () => {
  await Promise.all(temporaryPaths.splice(0).map((temporaryPath) => (
    fs.rm(temporaryPath, { recursive: true, force: true })
  )));
});

describe('sandbox invocation', () => {
  test('isolates namespaces, clears the environment, and applies resource limits', () => {
    const paths: SandboxPaths = {
      bwrapPath: '/usr/bin/bwrap',
      prlimitPath: '/usr/bin/prlimit',
      sandboxInitPath: '/usr/local/bin/artsy-sandbox-init',
      hostPythonEnvironmentPath: '/app/venv',
      hostArtCanvasPath: '/app/artcanvas.py',
      hostCodePath: '/app/code-00000000-0000-0000-0000-000000000000.py',
      hostOutputDirectoryPath: '/app/output',
      hostOutputFilePath: '/app/output/output.png',
      sandboxCodePath: '/program/code-00000000-0000-0000-0000-000000000000.py',
      useUserNamespace: true,
    };

    const invocation = buildSandboxInvocation(paths);
    expect(invocation.command).toBe('/usr/bin/prlimit');
    expect(invocation.args).toContain('--unshare-net');
    expect(invocation.args).toContain('--unshare-pid');
    expect(invocation.args).toContain('--clearenv');
    expect(invocation.args).toContain('--disable-userns');
    expect(invocation.args).toContain('--as=268435456:268435456');
    expect(invocation.args).toContain('--nproc=64:64');
    expect(invocation.args).toContain('/usr/local/bin/artsy-sandbox-init');
    expect(invocation.args).not.toContain('--share-net');
  });

  test('uses Bubblewrap setuid mode without nested user namespaces', () => {
    const paths: SandboxPaths = {
      bwrapPath: '/usr/bin/bwrap',
      prlimitPath: '/usr/bin/prlimit',
      sandboxInitPath: '/usr/local/bin/artsy-sandbox-init',
      hostPythonEnvironmentPath: '/app/venv',
      hostArtCanvasPath: '/app/artcanvas.py',
      hostCodePath: '/app/code.py',
      hostOutputDirectoryPath: '/app/output',
      hostOutputFilePath: '/app/output/output.png',
      sandboxCodePath: '/program/code.py',
      useUserNamespace: false,
    };

    const invocation = buildSandboxInvocation(paths);
    expect(invocation.args).not.toContain('--unshare-user');
    expect(invocation.args).not.toContain('--disable-userns');
    expect(invocation.args).not.toContain('--size');
    expect(invocation.args).toContain('--unshare-net');
  });
});

describe('sandbox output validation', () => {
  test('accepts a regular PNG file', async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'artsy-sandbox-'));
    temporaryPaths.push(directory);
    const outputPath = path.join(directory, 'output.png');
    await fs.writeFile(outputPath, validPng(2, 1));

    await expect(validatePngOutput(outputPath, 2, 1)).resolves.toBeUndefined();
  });

  test('rejects non-PNG files and symbolic links', async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'artsy-sandbox-'));
    temporaryPaths.push(directory);
    const textPath = path.join(directory, 'not-an-image');
    const linkPath = path.join(directory, 'output.png');
    await fs.writeFile(textPath, 'not a png');
    await fs.symlink(textPath, linkPath);

    await expect(validatePngOutput(textPath)).rejects.toBeInstanceOf(SandboxExecutionError);
    await expect(validatePngOutput(linkPath)).rejects.toBeInstanceOf(SandboxExecutionError);
  });
});
