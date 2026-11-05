#!/usr/bin/env python3
"""Shell executables for optional image feedback and source submission."""

import base64
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

if __package__:
    from drawing.mini_protocol import COMPLETE, bundle_source, emit, remember, snapshot, source_hash
    from drawing.render_contract import validate_png
    from .render import render
else:
    from mini_protocol import COMPLETE, bundle_source, emit, remember, snapshot, source_hash
    from render_contract import validate_png
    from render import render


def read_png(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or not 24 <= info.st_size <= 8 * 1024 * 1024:
            raise ValueError('Expected a regular PNG of at most 8 MiB.')
        png = f.read(8 * 1024 * 1024 + 1)
    validate_png(png)
    return png


def main():
    command = Path(sys.argv[0]).name
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help')):
        print(f'Usage: {command} ' + ('PNG_OR_PYTHON_ENTRYPOINT' if command == 'artsy-preview' else 'PYTHON_ENTRYPOINT'))
        return 0 if len(sys.argv) == 2 else 2
    path = sys.argv[1] if len(sys.argv) == 2 else None
    try:
        if command == 'artsy-submit':
            bundle = snapshot(path)
            remember(bundle)
            emit({'kind': 'submit', 'bundle': bundle})
            print(COMPLETE)
            return 0
        if command != 'artsy-preview':
            raise ValueError('Invoke as artsy-preview or artsy-submit.')
        if path and Path(path).suffix.lower() == '.png':
            png = read_png(path)
            feedback = {'ok': True, 'path': str(Path(path).resolve())}
        else:
            bundle = snapshot(path)
            source = bundle_source(bundle)
            runtime = Path(os.getenv('ARTSY_RUNTIME_DIR', str(Path(__file__).parent / 'runtime')))
            with tempfile.TemporaryDirectory(prefix='artsy-preview-') as directory:
                output = Path(directory)
                feedback = render(source, int(os.getenv('ARTSY_SEED', '0')), output, runtime)
                feedback.update(sourceHash=source_hash(source), entrypoint=bundle['entrypoint'])
                if not feedback['ok']:
                    emit({'kind': 'preview', 'feedback': feedback})
                    return 1
                png = read_png(output / 'image.png')
            remember(bundle)
        emit({'kind': 'preview', 'feedback': feedback, 'png': base64.b64encode(png).decode()})
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'{command}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
