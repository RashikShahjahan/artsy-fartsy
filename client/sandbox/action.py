"""Run one mini-agent action inside the credential-free rendering VM."""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

from drawing.mini_session import DrawingSession, READ_SOURCE
from evaluation.render import render

ROOT = Path('/vercel/sandbox/job')


async def main():
    number = int(sys.argv[1])
    target = ROOT / f'result-{number}.json'
    if target.exists():
        return
    request = json.loads((ROOT / f'action-{number}.json').read_text())

    async def shell(command):
        process = await asyncio.create_subprocess_exec('timeout', '--signal=KILL', '65', 'bash', '-lc', command,
            cwd='/workspace', env={**os.environ, 'ARTSY_SEED': '42'},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await process.communicate()
        if len(output) > 24 * 1024 * 1024:
            raise ValueError('Command output too large')
        return {'returncode': process.returncode, 'output': output.decode(errors='replace')}

    async def source(entrypoint):
        process = await asyncio.create_subprocess_exec('python', '-c', READ_SOURCE, *([entrypoint] if entrypoint else []),
            cwd='/workspace', stdout=asyncio.subprocess.PIPE)
        output, _ = await process.communicate()
        if process.returncode:
            raise ValueError('Unable to read submitted source')
        return json.loads(output)

    session = DrawingSession(shell, source)
    result = await session.execute(request['command'])
    if session.submitted:
        (ROOT / 'program.py').write_text(session.source)
        feedback = await asyncio.to_thread(render, session.source, 42, ROOT / 'final')
        if not feedback['ok']:
            raise ValueError('Final render failed')
        png = (ROOT / 'final/image.png').read_bytes()
        result['submission'] = {'source': session.source, 'png': base64.b64encode(png).decode(),
            'sourceHash': hashlib.sha256(session.source.encode()).hexdigest(), 'imageHash': hashlib.sha256(png).hexdigest()}
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result))
    temporary.replace(target)


if __name__ == '__main__':
    asyncio.run(main())
