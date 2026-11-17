"""Editing and partial-result recovery through the core drawing harness."""
import base64
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from client.backend import server, worker


class WorkerTests(unittest.TestCase):
    def test_edit_revalidates_source_and_retains_an_incomplete_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent_id = 'a' * 32
            parent = root / parent_id
            output = parent / 'output'
            output.mkdir(parents=True)
            (parent / 'job.json').write_text(json.dumps({'state': 'completed', 'prompt': 'A circle'}))
            (output / 'program.py').write_text('original source')
            (output / 'image.png').write_bytes(b'original image')
            (output / 'result.json').write_text(json.dumps({
                'outcome': 'submitted', 'sourceHash': server.content_hash(output / 'program.py'),
                'imageHash': server.content_hash(output / 'image.png')}))
            child = root / ('b' * 32)
            child.mkdir()
            (child / 'job.json').write_text(json.dumps({'experience': 'artsy', 'parent_id': parent_id}))

            def draw(*, progress):
                self.assertEqual(sys.argv[1], 'Previous request: A circle\n\nRequested change: Make it blue')
                self.assertEqual(sys.argv[-2:], ['--reference-source', str(output / 'program.py')])
                target = child / 'output'
                target.mkdir()
                (target / 'program.py').write_text('edited source')
                (target / 'result.json').write_text(json.dumps({'outcome': 'incomplete', 'turnLimitReached': True}))
                progress({'phase': 'previewing', 'step': {'id': 'preview', 'kind': 'render',
                    'previews': ['data:image/png;base64,' + base64.b64encode(b'latest image').decode()]}})
                return 1

            with patch.object(server, 'JOBS', root), \
                 patch.object(sys, 'argv', ['worker', str(child), 'Make it blue', '42', server.MODEL]), \
                 patch.dict(sys.modules, {'evaluation.draw': SimpleNamespace(main=draw)}):
                self.assertEqual(worker.main(), 0)
            result = json.loads((child / 'output/result.json').read_text())
            self.assertEqual((child / 'output/image.png').read_bytes(), b'latest image')
            self.assertEqual(result['sourceHash'], server.content_hash(child / 'output/program.py'))
            self.assertEqual(result['imageHash'], server.content_hash(child / 'output/image.png'))
            progress = json.loads((child / 'progress.json').read_text())
            self.assertEqual(progress['steps'][0]['previews'], ['preview-0-0.png'])
