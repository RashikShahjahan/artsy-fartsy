import express from 'express';
import cors from 'cors';
import { 
  GenerateCodeSchema, 
  RunCodeSchema, 
  StoreCodeSchema, 
  FindSimilarSchema 
} from '../../shared/schemas';
import { generateArtCode } from './utils/generation';
import { findSimilarDocuments, storeDocument } from './utils/embeddings';
import { initializeDatabase } from './utils/db';
import { executeArtCode, MaliciousCodeError } from './utils/codeExecution';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const app = express();

app.use(express.json());
app.use(cors());

app.post('/generate_code', async (req, res) => {
  try {
    const { userPrompt, artType } = GenerateCodeSchema.parse(req.body);
    const generatedCode = await generateArtCode(userPrompt, artType);
    res.status(200).json({ code: generatedCode });
  } catch (error) {
    res.status(400).json({ 
      error: error instanceof Error ? error.message : 'Invalid request'
    });
  }
});

app.post('/run_code', async (req, res) => {
  try {
    const { code, artType } = RunCodeSchema.parse(req.body);
    const outputPath = await executeArtCode(code, artType);
    
    res.status(200);
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
    console.error('Error running code:', error);
    
    if (error instanceof MaliciousCodeError) {
      res.status(400).json({ 
        error: error.message,
        type: 'malicious_code'
      });
    }
    
    res.status(500).json({ 
      error: error instanceof Error ? error.message : 'Server error',
      type: 'server_error'
    });
  }
});

app.post('/store_code', async (req, res) => {
    try {
        await initializeDatabase();
        const { prompt, code, artType } = StoreCodeSchema.parse(req.body);
        await storeDocument(prompt, code, artType);
        res.status(200).json({ success: true });
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
        await initializeDatabase();
        const { prompt } = FindSimilarSchema.parse(req.body);
        const similarCode = await findSimilarDocuments(prompt);
        
        const images: string[] = [];
        
        for (const code of similarCode) {
            const outputPath = await executeArtCode(code, 'drawing');
            const imageBuffer = await fs.promises.readFile(outputPath);
            const base64Image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
            images.push(base64Image);
            
            await fs.promises.unlink(outputPath).catch(err => 
                console.error('Error deleting file:', err)
            );
        }
        
        res.status(200).json({ images });
        
    } catch (error) {
        console.error('Error finding similar documents:', error);
        res.status(500).json({ 
            error: error instanceof Error ? error.message : 'Server error'
        });
    }
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

app.use(express.static(path.join(__dirname, '..','..', 'frontend', 'dist')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', '..', 'frontend', 'dist', 'index.html'));
});
