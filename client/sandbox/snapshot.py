"""Save editable source and the latest valid render when the turn budget expires."""
import base64
import hashlib
import json
from pathlib import Path
from drawing.mini_protocol import snapshot, bundle_source
from evaluation.render import render

root = Path('/vercel/sandbox/job')
try:
    source = bundle_source(snapshot(cwd='/workspace'))
except (OSError, ValueError):
    result = {}
else:
    result = {'source': source, 'sourceHash': hashlib.sha256(source.encode()).hexdigest()}
    feedback = render(source, 42, root / 'paused')
    if feedback['ok']:
        png = (root / 'paused/image.png').read_bytes()
        result.update(png=base64.b64encode(png).decode(), imageHash=hashlib.sha256(png).hexdigest())
(root / 'snapshot.json').write_text(json.dumps(result))
