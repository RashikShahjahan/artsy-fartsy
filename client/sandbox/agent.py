"""Trusted mini agent; filesystem RPC sends shell actions to a separate microVM."""
import asyncio
import base64
import io
import json
import os
from pathlib import Path

from PIL import Image
from drawing.config import IMAGE_SIZE, MAX_RESPONSE_TOKENS, MODEL_CONTEXT_WINDOW, MODEL_ID, MODEL_REVISION, TINKER_SAMPLING_MODEL_ID
from drawing.mini_agent import TOOLS, run_agent
from inference.tinker_backend import TinkerBackend

ROOT = Path('/vercel/sandbox/job')


def write(name, value):
    temporary = ROOT / (name + '.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(ROOT / name)


async def main():
    request = json.loads((ROOT / 'request.json').read_text())
    backend = TinkerBackend(MODEL_ID, MODEL_REVISION, IMAGE_SIZE,
                            os.environ['ARTSY_TINKER_MODEL_PATH'],
                            sampling_model_id=TINKER_SAMPLING_MODEL_ID)
    sequence = 0

    async def complete(messages):
        from inference.server import template_messages, native_message
        import copy
        response, _, generated = await asyncio.to_thread(
            backend.generate, template_messages(copy.deepcopy(messages)), TOOLS,
            MAX_RESPONSE_TOKENS, MODEL_CONTEXT_WINDOW)
        if not isinstance(response, dict):
            raise ValueError('Expected native Tinker response')
        text, calls = native_message(response)
        return {'text': text, 'tool_calls': calls,
                'finishReason': 'length' if generated >= MAX_RESPONSE_TOKENS else None}

    async def execute(command):
        nonlocal sequence
        sequence += 1
        write('action.json', {'id': sequence, 'command': command})
        while True:
            path = ROOT / f'reply-{sequence}.json'
            if path.exists():
                return json.loads(path.read_text())
            await asyncio.sleep(0.5)

    async def progress(turn):
        write('thinking.json', {'turn': turn})

    prompt = request['prompt']
    if request.get('parentPrompt'):
        prompt = 'Edit the existing program.py. Previous request: ' + request['parentPrompt'] + '\nRequested change: ' + prompt
    trace = await run_agent(complete, execute, prompt, 42, max_turns=10, progress=progress)
    write('trace.json', trace)
    write('done.json', {'ok': True})


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception:
        import traceback
        traceback.print_exc()
        write('done.json', {'ok': False})
        raise
