import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const execAsync = promisify(exec);

export async function executeArtCode(code: string): Promise<string> {
  const timestamp = Date.now();
  const uniqueId = crypto.randomBytes(4).toString('hex');
  const codeFilePath = path.join(__dirname, '..', 'drawing', `generated_art_script_${timestamp}.py`);
  const defaultOutputPath = path.join(__dirname, '..', 'output', 'output.png');
  const finalOutputPath = path.join(__dirname, '..', 'output', `${timestamp}_${uniqueId}.png`);

  await fs.promises.mkdir(path.dirname(codeFilePath), { recursive: true });
  await fs.promises.mkdir(path.dirname(defaultOutputPath), { recursive: true });

  if (containsMaliciousCode(code)) {
    throw new Error('Potentially malicious code detected');
  }

  await fs.promises.writeFile(codeFilePath, code);

  try {
    const pythonPath = path.join(__dirname, '..', 'venv', 'bin', 'python');
    const { stdout, stderr } = await execAsync(`"${pythonPath}" "${codeFilePath}"`, {
      timeout: 30000,
      maxBuffer: 1024 * 1024, // 1MB output limit
      env: {
        PATH: path.join(__dirname, '..', 'venv', 'bin'),
        PYTHONPATH: path.join(__dirname, '..', 'drawing'),
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1',
        PYTHONUNBUFFERED: '1',
        PYTHONDONTWRITEBYTECODE: '1',
        PYTHONHOME: undefined,
        PYTHONSTARTUP: undefined,
        PYTHONPATH_ORIG: undefined,
        HOME: undefined,
        USER: undefined,
      },
      cwd: path.dirname(finalOutputPath)
    });

    if (!fs.existsSync(defaultOutputPath)) {
      throw new Error('Image was not generated');
    }

    await fs.promises.rename(defaultOutputPath, finalOutputPath);

    return finalOutputPath;
  } finally {
    try {
      if (fs.existsSync(codeFilePath)) {
        await fs.promises.unlink(codeFilePath);
      }
    } catch (error) {
      console.error('Error cleaning up code file:', error);
    }
  }
}

export function containsMaliciousCode(code: string): boolean {
  const dangerousPatterns = [
    /import\s+os/,
    /import\s+sys/,
    /import\s+subprocess/,
    /from\s+os\s+import/,
    /from\s+sys\s+import/,
    /from\s+subprocess\s+import/,
    /open\(/,
    /exec\(/,
    /eval\(/,
    /Popen\(/,
    /environ\[/,
    /process\./,
    /child_process/,
    /spawn\(/,
    /fork\(/,
    /\.system\(/,
    /\.popen\(/,
    /\.call\(/,
  ];

  // Remove line continuations before testing
  const normalizedCode = code.replace(/\\\s*\n/g, '');
  return dangerousPatterns.some(pattern => pattern.test(normalizedCode));
} 