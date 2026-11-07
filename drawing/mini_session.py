"""Interpret real shell-command events and snapshot validated submissions."""

import base64
import json
from pathlib import Path
import shlex

from .mini_agent import PREVIEW, SUBMIT
from .mini_protocol import COMPLETE, bundle_source, read_events, source_hash
from .render_contract import validate_png


def session_action(command):
    """Classify historical standalone actions for SFT validation; runtime uses shell events."""
    if command.strip() == SUBMIT:
        return 'submit', None
    try:
        words = shlex.split(command)
    except ValueError:
        return None, None
    if words and words[0] in (PREVIEW, 'artsy-submit') and len(words) <= 2:
        return ('preview' if words[0] == PREVIEW else 'submit'), words[1] if len(words) == 2 else None
    return None, None


class DrawingSession:
    def __init__(self, shell, read_source, preview=None, *, max_calls=None):
        self.shell, self.read_source = shell, read_source
        self.max_calls = max_calls
        self.calls = 0
        self.submitted = False
        self.source = ''
        self.entrypoint = None
        self.bundle = None

    def select(self, bundle):
        self.source = bundle_source(bundle)
        self.bundle, self.entrypoint = bundle, bundle['entrypoint']

    async def execute(self, command):
        self.calls += 1
        if self.submitted or (self.max_calls is not None and self.calls > self.max_calls):
            return self.result(1, 'Episode operation budget exhausted or already submitted.')
        try:
            result = await self.shell(command)
            output = result.get('output')
            if output is None:
                output = '\n'.join(p.get('text', '') for p in result.get('content', []))
            events, text = read_events(output)
            images, previews = [], []
            for event in events:
                if event['kind'] == 'preview':
                    feedback = event.get('feedback', {})
                    previews.append(feedback)
                    if event.get('png'):
                        encoded = event['png']
                        if not isinstance(encoded, str) or len(encoded) > 4 * ((8 * 1024 * 1024 + 2) // 3):
                            raise ValueError('Preview exceeds 8 MiB.')
                        png = base64.b64decode(encoded, validate=True)
                        validate_png(png)
                        images.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}})
                    text += '\n' + json.dumps(feedback)
                elif event['kind'] == 'submit':
                    self.select(event['bundle'])
                    self.submitted = True
            if not self.submitted and result['returncode'] == 0 and COMPLETE in text.splitlines():
                # Native mini completion works inside compound shell scripts too.
                self.select(await self.read_source(None))
                self.submitted = True
            content = [{'type': 'text', 'text': f"Return code: {result['returncode']}\n{text[:16384]}"}, *images]
            return {'returncode': result['returncode'], 'content': content,
                    'previewEvents': previews, 'terminal': self.submitted,
                    'sourceHash': source_hash(self.source), 'entrypoint': self.entrypoint}
        except (FileNotFoundError, ValueError, TimeoutError) as error:
            return self.result(1, str(error))

    def result(self, code, text):
        return {'returncode': code, 'content': [{'type': 'text', 'text': f'Return code: {code}\n{text}'}],
                'sourceHash': source_hash(self.source), 'entrypoint': self.entrypoint}


# Embed the same snapshot implementation used by the installed executables.
READ_SOURCE = Path(__file__).with_name('mini_protocol.py').read_text() + (
    "\nimport sys\nprint(json.dumps(snapshot(sys.argv[1] if len(sys.argv)>1 else None)))\n")
