import express from 'express';
import cors from 'cors';
import { z } from 'zod';
import { generateArtCode } from './utils/generation';
import { exec } from 'child_process';
import fs from 'fs';
import { promisify } from 'util';
import { findSimilarDocuments, storeDocument } from './utils/embeddings';
import { initializeDatabase } from './utils/db';
import archiver from 'archiver';



const execAsync = promisify(exec);
const archive = archiver('zip');
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
  const codeFilePath = 'drawing/generated_art_script.py';
  const outputPath = `output.png`;

  await fs.promises.writeFile(codeFilePath, code);

  try {
    await execAsync(`python ${codeFilePath}`, { timeout: 30000 });
    
    if (!fs.existsSync(outputPath)) {
      throw new Error('Image was not generated');
    }

    return outputPath;
  } finally {
    if (fs.existsSync(codeFilePath)) {
      await fs.promises.unlink(codeFilePath);
    }
  }
}

app.post('/generate_code', async (req, res) => {
  try {
    const { userPrompt } = GenerateCodeSchema.parse(req.body);
    const generatedCode = await generateArtCode(userPrompt);
    res.json({ code: generatedCode });
  } catch (error) {
    res.status(400).json({ error: error instanceof Error ? error.message : 'Invalid request' });
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
    await initializeDatabase();
    const { code } = StoreCodeSchema.parse(req.body);
    await storeDocument(code);
    res.json({ success: true });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

app.post('/find_similar', async (req, res) => {
    try {
        const { prompt } = req.body;
        const similarCode = await findSimilarDocuments(prompt);
        
        // Array to store base64 images
        const images = [];
        
        // Process each code example
        for (const code of similarCode) {
            const outputPath = await executeArtCode(code);
            // Read file and convert to base64
            const imageBuffer = await fs.promises.readFile(outputPath);
            const base64Image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
            images.push(base64Image);
            
            // Cleanup image
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
