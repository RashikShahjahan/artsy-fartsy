import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { pool } from './db';


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const execAsync = promisify(exec);


interface ArtTypeConfig {
  extension: string;
  outputFile: string;
}

const ART_TYPE_CONFIGS: Record<string, ArtTypeConfig> = {
  drawing: { extension: '.py', outputFile: 'output.png' },
  music: { extension: '.py', outputFile: 'output.mid' }
  // Add more art types as needed
};


export class MaliciousCodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MaliciousCodeError';
  }
}


export async function storeError(code: string, error: string) {
  await pool.query(`INSERT INTO errors (code, error) VALUES ($1, $2)`, [code, error]);
}

export async function executeArtCode(code: string, artType: string, ranByAI: boolean): Promise<string> {
  const timestamp = Date.now();
  const uniqueId = crypto.randomBytes(4).toString('hex');
  const config = ART_TYPE_CONFIGS[artType];
  
  const outputDir = path.join(__dirname, '..','..', 'art_libraries','output', `${timestamp}_${uniqueId}`);
  const codeFilePath = path.join(__dirname, '..', '..', 'art_libraries', `generated_art_script_${timestamp}${config.extension}`);
  const defaultOutputPath = path.join(outputDir, config.outputFile);
  const finalOutputPath = path.join(outputDir, `final${path.extname(config.outputFile)}`);

  await fs.promises.mkdir(path.dirname(codeFilePath), { recursive: true });
  await fs.promises.mkdir(outputDir, { recursive: true });

  if (containsMaliciousCode(code)) {
    throw new MaliciousCodeError('Code contains potentially unsafe operations');
  }

  await fs.promises.writeFile(codeFilePath, code);

  try {
    const pythonPath = path.join(__dirname, '..', '..', 'art_libraries', 'venv', 'bin', 'python');
    await execAsync(`"${pythonPath}" "${codeFilePath}"`, {
      timeout: 30000,
      maxBuffer: 1024 * 1024, // 1MB output limit
      env: {
        PATH: path.join(__dirname, '..', '..', 'art_libraries', 'venv', 'bin'),
        PYTHONPATH: path.join(__dirname, '..', '..', 'art_libraries'),
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
      throw new Error('Art was not generated');
    }

    await fs.promises.rename(defaultOutputPath, finalOutputPath);

    return finalOutputPath;
    
  } catch (error) {
    if (ranByAI) {
      await storeError(code, error.message);
    }
    throw error;
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