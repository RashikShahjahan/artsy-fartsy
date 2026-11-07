"""Preflight stale editing images before consuming SFT teacher tokens."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from drawing.mini_dataset import IMAGE_RUNTIME_FILES, inspect_image
from drawing.renderer import DockerRenderer

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = 'sha256:' + 'a' * 64


def image_files():
    return {container_path: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            for container_path, source in IMAGE_RUNTIME_FILES.items()}


class EditingImagePreflightTests(unittest.TestCase):
    def test_import_draw_and_hash_preflight_uses_pinned_image(self):
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            if command[1:3] == ['image', 'inspect']:
                return subprocess.CompletedProcess(command, 0, IMAGE_ID + '\n', '')
            return subprocess.CompletedProcess(
                command, 0, json.dumps({'ok': True, 'files': image_files()}), '')
        with patch('drawing.mini_dataset.subprocess.run', side_effect=fake_run):
            result = inspect_image()
        self.assertEqual(result['id'], IMAGE_ID)
        self.assertEqual(result['preflight'], 'passed')
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][calls[1].index('--cpus') + 2], IMAGE_ID)
        self.assertIn('--network=none', calls[1])
        self.assertIn('--pull=never', calls[1])
        self.assertIn('from artcanvas import ArtCanvas', calls[1][-2])

    def test_stale_image_rejected_before_model_requests(self):
        def fake_run(command, **kwargs):
            if command[1:3] == ['image', 'inspect']:
                return subprocess.CompletedProcess(command, 0, IMAGE_ID + '\n', '')
            files = image_files()
            files['/opt/artsy/runtime/artcanvas.py'] = 'bad-sha256'
            return subprocess.CompletedProcess(command, 0, json.dumps({'ok': True, 'files': files}), '')
        with patch('drawing.mini_dataset.subprocess.run', side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, 'stale or incompatible: runtime/artcanvas.py'):
                inspect_image()

    def test_missing_artcanvas_rejected_before_model_requests(self):
        def fake_run(command, **kwargs):
            if command[1:3] == ['image', 'inspect']:
                return subprocess.CompletedProcess(command, 0, IMAGE_ID + '\n', '')
            return subprocess.CompletedProcess(command, 1, '', "ModuleNotFoundError: No module named 'artcanvas'")
        with patch('drawing.mini_dataset.subprocess.run', side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, 'failed its ArtCanvas smoke test'):
                inspect_image()


class PinnedRendererTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_pins_independent_final_renderer(self):
        from drawing.dataset import generate_dataset
        from drawing.dataset_specs import validate_specs
        from drawing.mini_dataset import candidate

        collected = []
        async def fake_candidate(options, spec, index, api_doc):
            self.assertEqual(options['agentImage'], IMAGE_ID)
            self.assertIsInstance(options['renderer'], DockerRenderer)
            collected.append(options['renderer'].image)
            return {'id': f"{spec['id']}-c{index}", 'spec': spec, 'candidate': index,
                    'status': 'review', 'trace': []}
        specs = validate_specs([{'family': 'test', 'prompt': 'Draw a circle', 'seed': 42}])
        with tempfile.TemporaryDirectory() as directory:
            with patch('drawing.mini_dataset.inspect_image',
                       return_value={'name': 'artsy-editing-agent:latest', 'id': IMAGE_ID, 'preflight': 'passed'}), \
                 patch('drawing.mini_dataset.candidate', side_effect=fake_candidate):
                await generate_dataset({'output': directory, 'specs': specs, 'harness': 'mini-swe',
                                        'candidates': 1, 'renderer': DockerRenderer(image='different')})
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            self.assertEqual(manifest['harness']['editingImage']['id'], IMAGE_ID)
        self.assertEqual(collected, [IMAGE_ID])

    async def test_mini_rejects_local_final_renderer(self):
        from drawing.dataset import generate_dataset
        from drawing.dataset_specs import validate_specs
        from drawing.renderer import LocalRenderer

        specs = validate_specs([{'family': 'test', 'prompt': 'Draw a circle', 'seed': 42}])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'requires Docker for editing and final rendering'):
                await generate_dataset({'output': directory, 'specs': specs,
                                        'harness': 'mini-swe', 'renderer': LocalRenderer()})
