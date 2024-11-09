import express from 'express';
import cors from 'cors';
import { z } from 'zod';
import { generateArtCode } from './utils/generation';
import { exec } from 'child_process';
import fs from 'fs';
import { promisify } from 'util';

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
    const codeFilePath = 'drawing/generated_art_script.py';
    const outputPath = 'output.png';

    await fs.promises.writeFile(codeFilePath, code);

    try {
      await execAsync(`python ${codeFilePath}`, { timeout: 30000 });
      
      if (!fs.existsSync(outputPath)) {
        throw new Error('Image was not generated');
      }

      const timestamp = Date.now();
      const filename = `generated_art_${timestamp}.png`;
      
      res.setHeader('Content-Type', 'image/png');
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      fs.createReadStream(outputPath).pipe(res);

    } finally {
      if (fs.existsSync(codeFilePath)) {
        await fs.promises.unlink(codeFilePath);
      }
    }
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