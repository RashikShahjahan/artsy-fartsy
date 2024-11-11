import express from 'express';
import cors from 'cors';
import { z } from 'zod';
import { generateArtCode } from './utils/generation';
import { exec } from 'child_process';
import fs from 'fs';
import { promisify } from 'util';
import { findSimilarDocuments, storeDocument } from './utils/embeddings';
import { initializeDatabase } from './utils/db';
import { fileURLToPath } from 'url';
import path from 'path';
import crypto from 'crypto';
import process from 'process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


const execAsync = promisify(exec);
const app = express();

app.use(express.json());
app.use(cors());

const GenerateCodeSchema = z.object({
  userPrompt: z.string()
});

const RunCodeSchema = z.object({
  code: z.string()
});

const StoreCodeSchema = z.object({
  code: z.string()
});

async function executeArtCode(code: string): Promise<string> {
  // Generate random file names to prevent conflicts and predictable paths
  const randomId = crypto.randomBytes(16).toString('hex');
  const codeFilePath = path.join(__dirname, 'drawing', `generated_art_script_${randomId}.py`);
  const outputPath = path.join(__dirname, 'output', `${randomId}.png`);

  await fs.promises.mkdir(path.dirname(codeFilePath), { recursive: true });
  await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });

  if (containsMaliciousCode(code)) {
    throw new Error('Potentially malicious code detected');
  }

  await fs.promises.writeFile(codeFilePath, code);

  try {
    const pythonPath = path.join(__dirname, 'venv', 'bin', 'python');
    const { stdout, stderr } = await execAsync(`"${pythonPath}" "${codeFilePath}"`, {
      timeout: 30000,
      maxBuffer: 1024 * 1024, // 1MB output limit
      env: {
        PATH: path.join(__dirname, 'venv', 'bin'),
        PYTHONPATH: path.join(__dirname, 'drawing'),
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
      cwd: path.dirname(outputPath)
    });

    if (!fs.existsSync(outputPath)) {
      throw new Error('Image was not generated');
    }

    return outputPath;
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

function containsMaliciousCode(code: string): boolean {
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

app.post('/generate_code', async (req, res) => {
  try {
    const { userPrompt } = GenerateCodeSchema.parse(req.body);
    const generatedCode = await generateArtCode(userPrompt);
    res.json({ code: generatedCode });
  } catch (error) {
    res.status(500).json({ error: error instanceof Error ? error.message : 'Invalid request' });
  }
});

app.post('/run_code', async (req, res) => {
  try {
    const { code } = RunCodeSchema.parse(req.body);
    
    const outputPath = await executeArtCode(code);
    
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', `attachment; filename="${outputPath}"`);
    
    const stream = fs.createReadStream(outputPath);
    stream.pipe(res);
    
    stream.on('end', async () => {
      try {
        await fs.promises.unlink(outputPath);
      } catch (error) {
        console.error('Error deleting output file:', error);
      }
    });

  } catch (error) {
    res.status(500).json({ 
      error: error instanceof Error ? error.message : 'Server error'
    });
  }
});

app.post('/store_code', async (req, res) => {
    try {
        await initializeDatabase();
        const { code } = StoreCodeSchema.parse(req.body);
        await storeDocument(code);
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ 
            error: error instanceof Error ? error.message : 'Server error'
        });
    }
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

app.post('/find_similar', async (req, res) => {
    try {
        const { prompt } = req.body;
        const generatedCode = await generateArtCode(prompt);
        const similarCode = await findSimilarDocuments(generatedCode);
        
        const images = [];
        
        for (const code of similarCode) {
            const outputPath = await executeArtCode(code);
            const imageBuffer = await fs.promises.readFile(outputPath);
            const base64Image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
            images.push(base64Image);
            
            await fs.promises.unlink(outputPath).catch(err => 
                console.error('Error deleting file:', err)
            );
        }
        
        res.json({ images });
        
    } catch (error) {
        res.status(500).json({ 
            error: error instanceof Error ? error.message : 'Server error'
        });
    }
});

app.use(express.static(path.join(__dirname, '../dist')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../dist/index.html'));
});