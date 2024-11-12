import express from 'express';
import cors from 'cors';
import { 
  GenerateCodeSchema, 
  RunCodeSchema, 
  StoreCodeSchema, 
  FindSimilarSchema 
} from '../shared/schemas';
import { generateArtCode } from './utils/generation';
import { findSimilarDocuments, storeDocument } from './utils/embeddings';
import { initializeDatabase } from './utils/db';
import { executeArtCode } from './utils/codeExecution';
import fs from 'fs';
import path from 'path';

const app = express();

app.use(express.json());
app.use(cors());

app.post('/generate_code', async (req, res) => {
  try {
    const { userPrompt } = GenerateCodeSchema.parse(req.body);
    const generatedCode = await generateArtCode(userPrompt);
    res.json({ code: generatedCode });
  } catch (error) {
    res.status(400).json({ 
      error: error instanceof Error ? error.message : 'Invalid request'
    });
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
        const { prompt } = FindSimilarSchema.parse(req.body);
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