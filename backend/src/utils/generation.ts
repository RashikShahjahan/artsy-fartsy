import { Anthropic } from '@anthropic-ai/sdk';
import { ARTCANVAS_GUIDE } from './prompts/drawing';
import { MUSIC_GUIDE } from './prompts/music';

const client = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY // You'll need to add your API key
});

const GUIDES = {
    drawing: ARTCANVAS_GUIDE,
    music: MUSIC_GUIDE
}




async function generateArtCode(prompt: string, artType: string): Promise<string> {
    if (!GUIDES[artType]) {
        throw new Error(`Unsupported art type: ${artType}`);
    }

    const message = await client.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 1024,
        messages: [
            { role: "user", content: GUIDES["music"] },
            { role: "user", content: `${prompt} Only respond with code as plain text without code block syntax around it` }
        ],
    });

    // Extract text from the first content block
    const code = message.content[0].type === 'text' ? message.content[0].text : '';
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



export { generateArtCode};