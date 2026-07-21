import { execFile } from 'node:child_process';
import { constants as fsConstants, promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { inflateSync } from 'node:zlib';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const artLibrariesPath = path.resolve(__dirname, '..', '..', 'art_libraries');
const outputRoot = path.join(artLibrariesPath, 'output');

const SANDBOX_CODE_PATH_PREFIX = '/program';
const SANDBOX_OUTPUT_PATH = '/work';
const SANDBOX_LIBRARY_PATH = '/runtime/lib';
const SANDBOX_PYTHON_PATH = '/runtime/venv/bin/python';
const PNG_SIGNATURE = Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a);
const MAX_OUTPUT_BYTES = 8 * 1024 * 1024;
const DRAWING_WIDTH = 1920;
const DRAWING_HEIGHT = 1200;

const crcTable = Uint32Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit += 1) {
    value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  return value >>> 0;
});

const timeoutMs = readPositiveInteger('SANDBOX_TIMEOUT_MS', 30_000);
const maxConcurrentExecutions = readPositiveInteger('SANDBOX_MAX_CONCURRENT', 2);

let activeExecutions = 0;

export interface SandboxInvocation {
  command: string;
  args: string[];
}

export interface SandboxPaths {
  bwrapPath: string;
  prlimitPath: string;
  sandboxInitPath: string;
  hostPythonEnvironmentPath: string;
  hostArtCanvasPath: string;
  hostCodePath: string;
  hostOutputDirectoryPath: string;
  hostOutputFilePath: string;
  sandboxCodePath: string;
  useUserNamespace: boolean;
}

export interface ArtExecution {
  outputPath: string;
  cleanup: () => Promise<void>;
}

export class SandboxUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SandboxUnavailableError';
  }
}

export class SandboxBusyError extends Error {
  constructor() {
    super('The drawing sandbox is busy; try again shortly');
    this.name = 'SandboxBusyError';
  }
}

export class SandboxExecutionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SandboxExecutionError';
  }
}

function readUint32(bytes: Uint8Array, offset: number): number {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(offset);
}

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc = crcTable[(crc ^ byte) & 0xff]! ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function validatePngData(
  bytes: Uint8Array,
  expectedWidth = DRAWING_WIDTH,
  expectedHeight = DRAWING_HEIGHT,
): void {
  if (bytes.length < PNG_SIGNATURE.length || !PNG_SIGNATURE.every((value, index) => bytes[index] === value)) {
    throw new SandboxExecutionError('Drawing output is not a PNG image');
  }

  let offset = PNG_SIGNATURE.length;
  let sawHeader = false;
  let sawImageData = false;
  let endedImageData = false;
  let sawEnd = false;
  const imageDataChunks: Uint8Array[] = [];

  while (offset < bytes.length) {
    if (offset + 12 > bytes.length) {
      throw new SandboxExecutionError('Drawing output contains a truncated PNG chunk');
    }

    const length = readUint32(bytes, offset);
    const typeOffset = offset + 4;
    const dataOffset = typeOffset + 4;
    const crcOffset = dataOffset + length;
    const nextOffset = crcOffset + 4;
    if (nextOffset > bytes.length) {
      throw new SandboxExecutionError('Drawing output contains an invalid PNG chunk length');
    }

    const typeBytes = bytes.subarray(typeOffset, dataOffset);
    const type = String.fromCharCode(...typeBytes);
    const data = bytes.subarray(dataOffset, crcOffset);
    if (crc32(bytes.subarray(typeOffset, crcOffset)) !== readUint32(bytes, crcOffset)) {
      throw new SandboxExecutionError('Drawing output contains a corrupt PNG chunk');
    }

    if (!sawHeader && type !== 'IHDR') {
      throw new SandboxExecutionError('Drawing output is missing its PNG header');
    }

    if (type === 'IHDR') {
      if (sawHeader || length !== 13) {
        throw new SandboxExecutionError('Drawing output contains an invalid PNG header');
      }
      sawHeader = true;
      const width = readUint32(data, 0);
      const height = readUint32(data, 4);
      if (
        width !== expectedWidth
        || height !== expectedHeight
        || data[8] !== 8
        || data[9] !== 2
        || data[10] !== 0
        || data[11] !== 0
        || data[12] !== 0
      ) {
        throw new SandboxExecutionError('Drawing output has unsupported PNG dimensions or encoding');
      }
    } else if (type === 'IDAT') {
      if (!sawHeader || endedImageData) {
        throw new SandboxExecutionError('Drawing output contains out-of-order PNG image data');
      }
      sawImageData = true;
      imageDataChunks.push(data);
    } else if (type === 'IEND') {
      if (!sawImageData || length !== 0 || nextOffset !== bytes.length) {
        throw new SandboxExecutionError('Drawing output contains an invalid PNG ending');
      }
      sawEnd = true;
    } else {
      if (sawImageData) endedImageData = true;
      const isCriticalChunk = typeBytes[0] !== undefined && (typeBytes[0] & 0x20) === 0;
      if (isCriticalChunk && type !== 'PLTE') {
        throw new SandboxExecutionError(`Drawing output contains unsupported PNG chunk ${type}`);
      }
    }

    offset = nextOffset;
  }

  if (!sawHeader || !sawImageData || !sawEnd) {
    throw new SandboxExecutionError('Drawing output is an incomplete PNG image');
  }

  const compressedLength = imageDataChunks.reduce((total, chunk) => total + chunk.length, 0);
  const compressed = new Uint8Array(compressedLength);
  let compressedOffset = 0;
  for (const chunk of imageDataChunks) {
    compressed.set(chunk, compressedOffset);
    compressedOffset += chunk.length;
  }

  const rowLength = 1 + expectedWidth * 3;
  const expectedDecodedLength = rowLength * expectedHeight;
  let decoded: Uint8Array;
  try {
    decoded = Uint8Array.from(inflateSync(compressed, { maxOutputLength: expectedDecodedLength + 1 }));
  } catch {
    throw new SandboxExecutionError('Drawing output contains invalid compressed PNG data');
  }
  if (decoded.length !== expectedDecodedLength) {
    throw new SandboxExecutionError('Drawing output contains an unexpected amount of pixel data');
  }
  for (let row = 0; row < expectedHeight; row += 1) {
    if (decoded[row * rowLength]! > 4) {
      throw new SandboxExecutionError('Drawing output contains an invalid PNG filter');
    }
  }
}

function readPositiveInteger(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isSafeInteger(value) && value > 0 ? value : fallback;
}

function getSandboxPaths(
  hostCodePath: string,
  hostOutputDirectoryPath: string,
  hostOutputFilePath: string,
): SandboxPaths {
  return {
    bwrapPath: process.env.SANDBOX_BWRAP_PATH || '/usr/bin/bwrap',
    prlimitPath: process.env.SANDBOX_PRLIMIT_PATH || '/usr/bin/prlimit',
    sandboxInitPath: process.env.SANDBOX_INIT_PATH || '/usr/local/bin/artsy-sandbox-init',
    hostPythonEnvironmentPath: path.join(artLibrariesPath, 'venv'),
    hostArtCanvasPath: path.join(artLibrariesPath, 'artcanvas.py'),
    hostCodePath,
    hostOutputDirectoryPath,
    hostOutputFilePath,
    sandboxCodePath: `${SANDBOX_CODE_PATH_PREFIX}/${path.basename(hostCodePath)}`,
    useUserNamespace: process.env.SANDBOX_SETUID !== 'true',
  };
}

export function buildSandboxInvocation(paths: SandboxPaths): SandboxInvocation {
  const bootstrap = [
    'import runpy,sys',
    `sys.path.insert(0,${JSON.stringify(SANDBOX_LIBRARY_PATH)})`,
    `runpy.run_path(${JSON.stringify(paths.sandboxCodePath)},run_name='__main__')`,
  ].join(';');

  const bwrapArgs = [
    ...(paths.useUserNamespace ? ['--unshare-user', '--disable-userns'] : []),
    '--unshare-ipc',
    '--unshare-pid',
    '--unshare-net',
    '--unshare-uts',
    '--unshare-cgroup-try',
    '--die-with-parent',
    '--new-session',
    '--clearenv',
    '--cap-drop',
    'ALL',
    '--ro-bind',
    '/usr',
    '/usr',
    '--ro-bind-try',
    '/bin',
    '/bin',
    '--ro-bind-try',
    '/lib',
    '/lib',
    '--ro-bind-try',
    '/lib64',
    '/lib64',
    '--dir',
    '/etc',
    '--ro-bind-try',
    '/etc/fonts',
    '/etc/fonts',
    '--ro-bind-try',
    '/etc/ld.so.cache',
    '/etc/ld.so.cache',
    '--dir',
    '/var',
    '--dir',
    '/var/cache',
    '--ro-bind-try',
    '/var/cache/fontconfig',
    '/var/cache/fontconfig',
    '--proc',
    '/proc',
    '--dev',
    '/dev',
    '--size',
    '16777216',
    '--perms',
    '0700',
    '--tmpfs',
    '/tmp',
    '--dir',
    '/runtime',
    '--ro-bind',
    paths.hostPythonEnvironmentPath,
    '/runtime/venv',
    '--dir',
    SANDBOX_LIBRARY_PATH,
    '--ro-bind',
    paths.hostArtCanvasPath,
    `${SANDBOX_LIBRARY_PATH}/artcanvas.py`,
    '--dir',
    SANDBOX_CODE_PATH_PREFIX,
    '--ro-bind',
    paths.hostCodePath,
    paths.sandboxCodePath,
    '--ro-bind',
    paths.hostOutputDirectoryPath,
    SANDBOX_OUTPUT_PATH,
    '--remount-ro',
    '/',
    '--remount-ro',
    '/proc',
    '--remount-ro',
    '/dev',
    '--bind',
    paths.hostOutputFilePath,
    `${SANDBOX_OUTPUT_PATH}/output.png`,
    '--chdir',
    SANDBOX_OUTPUT_PATH,
    '--setenv',
    'HOME',
    '/tmp',
    '--setenv',
    'TMPDIR',
    '/tmp',
    '--setenv',
    'PATH',
    '/runtime/venv/bin:/usr/bin:/bin',
    '--setenv',
    'LANG',
    'C.UTF-8',
    '--',
    paths.sandboxInitPath,
    SANDBOX_PYTHON_PATH,
    '-I',
    '-B',
    '-c',
    bootstrap,
  ];

  return {
    command: paths.prlimitPath,
    args: [
      '--cpu=10:10',
      '--as=268435456:268435456',
      `--fsize=${MAX_OUTPUT_BYTES}:${MAX_OUTPUT_BYTES}`,
      '--nproc=4:4',
      '--nofile=64:64',
      '--core=0:0',
      '--',
      paths.bwrapPath,
      ...bwrapArgs,
    ],
  };
}

async function assertSandboxPaths(paths: SandboxPaths): Promise<void> {
  const executables = [
    paths.bwrapPath,
    paths.prlimitPath,
    paths.sandboxInitPath,
    path.join(paths.hostPythonEnvironmentPath, 'bin', 'python'),
  ];
  const readableFiles = [paths.hostArtCanvasPath, paths.hostCodePath];

  try {
    await Promise.all([
      ...executables.map((filePath) => fs.access(filePath, fsConstants.X_OK)),
      ...readableFiles.map((filePath) => fs.access(filePath, fsConstants.R_OK)),
    ]);
  } catch {
    throw new SandboxUnavailableError('The Bubblewrap sandbox runtime is not installed or configured');
  }
}

function runInvocation(invocation: SandboxInvocation, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    execFile(
      invocation.command,
      invocation.args,
      {
        env: {},
        timeout: timeoutMs,
        killSignal: 'SIGKILL',
        maxBuffer: 64 * 1024,
        signal,
      },
      (error, _stdout, stderr) => {
        if (!error) {
          resolve();
          return;
        }

        const detail = stderr
          .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, '?')
          .trim()
          .slice(-4000);
        if (error.name === 'AbortError') {
          reject(new SandboxExecutionError('Drawing execution was cancelled'));
          return;
        }
        if (/creating new namespace failed|no permissions to create new namespace|bwrap:.*operation not permitted|prlimit:.*failed/i.test(detail)) {
          reject(new SandboxUnavailableError('The host does not permit isolated sandbox namespaces'));
          return;
        }
        reject(new SandboxExecutionError(detail || 'Sandboxed code execution failed'));
      },
    );
  });
}

export async function validatePngOutput(
  outputPath: string,
  expectedWidth = DRAWING_WIDTH,
  expectedHeight = DRAWING_HEIGHT,
): Promise<void> {
  const stats = await fs.lstat(outputPath).catch(() => undefined);
  if (!stats?.isFile() || stats.size === 0 || stats.size > MAX_OUTPUT_BYTES) {
    throw new SandboxExecutionError('Drawing did not produce a valid output file');
  }

  const file = await fs.open(outputPath, fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW);
  try {
    validatePngData(Uint8Array.from(await file.readFile()), expectedWidth, expectedHeight);
  } finally {
    await file.close();
  }
}

export async function prepareSandboxStorage(): Promise<void> {
  await fs.mkdir(outputRoot, { recursive: true, mode: 0o700 });
  const entries = await fs.readdir(outputRoot);
  await Promise.all(entries.map((entry) => fs.rm(path.join(outputRoot, entry), { recursive: true, force: true })));
}

export async function verifySandboxRuntime(): Promise<void> {
  const execution = await executeDrawingInSandbox([
    'import os',
    'import socket',
    'try:',
    '    socket.socket()',
    'except PermissionError:',
    '    pass',
    'else:',
    '    raise RuntimeError("sandbox networking is not blocked")',
    'try:',
    '    os.fork()',
    'except PermissionError:',
    '    pass',
    'else:',
    '    raise RuntimeError("sandbox process creation is not blocked")',
    'from artcanvas import ArtCanvas',
    'canvas = ArtCanvas()',
    'canvas.save()',
  ].join('\n'));
  await execution.cleanup();
}

export async function executeDrawingInSandbox(code: string, signal?: AbortSignal): Promise<ArtExecution> {
  if (signal?.aborted) {
    throw new SandboxExecutionError('Drawing execution was cancelled');
  }
  if (activeExecutions >= maxConcurrentExecutions) {
    throw new SandboxBusyError();
  }

  activeExecutions += 1;
  const executionId = randomUUID();
  const workspacePath = path.join(outputRoot, executionId);
  const hostOutputPath = path.join(workspacePath, 'output');
  const codeFilePath = path.join(workspacePath, `code-${randomUUID()}.py`);
  const generatedOutputPath = path.join(hostOutputPath, 'output.png');
  const finalOutputPath = path.join(hostOutputPath, `drawing-${randomUUID()}.png`);

  try {
    await fs.mkdir(hostOutputPath, { recursive: true, mode: 0o700 });
    await fs.writeFile(codeFilePath, code, { mode: 0o600, flag: 'wx' });
    await fs.writeFile(generatedOutputPath, new Uint8Array(), { mode: 0o600, flag: 'wx' });

    const paths = getSandboxPaths(codeFilePath, hostOutputPath, generatedOutputPath);
    await assertSandboxPaths(paths);
    await runInvocation(buildSandboxInvocation(paths), signal);
    await validatePngOutput(generatedOutputPath);
    await fs.rename(generatedOutputPath, finalOutputPath);

    let cleaned = false;
    return {
      outputPath: finalOutputPath,
      cleanup: async () => {
        if (!cleaned) {
          await fs.rm(workspacePath, { recursive: true, force: true });
          cleaned = true;
        }
      },
    };
  } catch (error) {
    await fs.rm(workspacePath, { recursive: true, force: true }).catch((cleanupError) => {
      console.error('Failed to clean sandbox after execution error:', cleanupError);
    });
    throw error;
  } finally {
    activeExecutions -= 1;
  }
}
