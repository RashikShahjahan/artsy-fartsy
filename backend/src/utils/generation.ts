import { ARTCANVAS_EDIT_GUIDE, ARTCANVAS_GUIDE } from './prompts/drawing';
import { MUSIC_GUIDE } from './prompts/music';
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
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

    const message = await anthropic.messages.create({
        model: "claude-3-7-sonnet-20250219",
        max_tokens: 8000,
        messages: [
            { role: "user", content: `${GUIDES[artType]} ${prompt} Only respond with code as plain text without code block syntax around it` }
        ],
    });

    // Extract text from content blocks
    let code = '';
    for (const block of message.content) {
        if (block.type === 'text') {
            code = block.text;
            break;
        }
    }

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

    const message = await anthropic.messages.create({
        model: "claude-3-7-sonnet",
        max_tokens: 8000,
        messages: [
            { role: "user", content: `${EDIT_GUIDES[artType]} ${prompt} Only respond with code as plain text without code block syntax around it ${code}` },
        ],
    });


    // Extract text from content blocks
    let editedCode = '';
    for (const block of message.content) {
        if (block.type === 'text') {
            editedCode = block.text;
            break;
        }
    }
    
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