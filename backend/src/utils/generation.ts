import { ARTCANVAS_EDIT_GUIDE, ARTCANVAS_GUIDE } from './prompts/drawing';
import { MUSIC_GUIDE } from './prompts/music';

import OpenAI from "openai";

const client = new OpenAI({
        baseURL: 'https://api.deepseek.com',
        apiKey: process.env.DEEPSEEK_API_KEY
});

const GUIDES = {
    drawing: ARTCANVAS_GUIDE,
    music: MUSIC_GUIDE
}

const EDIT_GUIDES = {
    drawing: ARTCANVAS_EDIT_GUIDE
}




async function generateArtCode(prompt: string, artType: string): Promise<string> {
    if (!GUIDES[artType]) {
        throw new Error(`Unsupported art type: ${artType}`);
    }

    const message = await client.chat.completions.create({
        model: "deepseek-reasoner",
        max_tokens: 8000,
        messages: [
            { role: "user", content: `${GUIDES[artType]} ${prompt} Only respond with code as plain text without code block syntax around it` }
        ],
    });

    // Extract text from the first content block
    const code = message.choices[0].message.content??'';


    // Remove code block markers and any non-code text
    return code
        .split('\n')
        .filter((line: string) => 
            !line.startsWith('```') && // Remove code block markers
            line.trim()                // Remove empty lines
        )
        .join('\n')
        .trim();
}

async function editArtCode(prompt: string, code: string, artType: string): Promise<string> {
    if (!EDIT_GUIDES[artType]) {
        throw new Error(`Unsupported art type: ${artType}`);
    }

    const message = await client.chat.completions.create({
        model: "deepseek-reasoner",
        max_tokens: 8000,
        messages: [
            { role: "user", content: `${EDIT_GUIDES[artType]} ${prompt} Only respond with code as plain text without code block syntax around it ${code}` },
        ],
    });


    // Extract text from the first content block
    const editedCode =  message.choices[0].message.content??'';
    // Remove code block markers and any non-code text
    return editedCode
        .split('\n')
        .filter((line: string) => 
            !line.startsWith('```') && // Remove code block markers
            line.trim()                // Remove empty lines
        )
        .join('\n')
        .trim();
}


export { generateArtCode, editArtCode };