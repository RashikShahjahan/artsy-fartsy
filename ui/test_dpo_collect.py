import asyncio
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image
from drawing.config import MODEL_ID
from ui import dpo_collect as worker


def request(output):
    return {'specs': [{'family': 'circles', 'prompt': 'A red circle', 'seed': 42}],
            'candidates': 2, 'output': str(output), 'model': MODEL_ID, 'model_path': 'tinker://sft'}


class CollectionTests(unittest.TestCase):
    def test_request_limits(self):
        for candidates in (1, 5, True, '2'):
            with self.assertRaises(ValueError):
                worker.validate_collection_request({**request('/tmp/example'), 'candidates': candidates})
        with self.assertRaises(ValueError):
            worker.validate_collection_request({**request('/tmp/example'), 'model': 'external/model'})

    def test_checkpoint_and_drawing_limits(self):
        for checkpoint in (None, '', 'local/file', 'tinker://'):
            with self.assertRaises(ValueError):
                worker.validate_collection_request({**request('/tmp/test'), 'model_path': checkpoint})
        for specs in ([], [dict(family='circles', prompt='circle', seed=i) for i in range(151)],
                      [dict(family='safety_refusal', prompt='unsafe', seed=0, kind='refusal', refusal='no')]):
            with self.assertRaises(ValueError):
                worker.validate_collection_request({**request('/tmp/test'), 'specs': specs})

    def test_cli_partial_infrastructure_failure(self):
        for summary in ({'infrastructureErrors': 1}, {'complete': False}, {'infrastructureErrors': 0}):
            with patch.object(worker.sys, 'argv', ['worker', 'request.json']), patch.object(worker.Path, 'read_text', return_value='{}'), patch.object(worker, 'collect', new=AsyncMock(return_value=summary)):
                self.assertEqual(worker.main(), int(bool(summary.get('infrastructureErrors')) or summary.get('complete') is False))

    def test_catalog_alias_uses_matching_health_profile(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({'model': MODEL_ID, 'model_path': None,
            'models': [{'id': 'artsy/test-sft', 'model_path': 'tinker://sft'}]}).encode()
        with patch.object(worker, 'urlopen', return_value=response):
            selected = worker.bridge_health('tinker://sft', 'artsy/test-sft')
            self.assertEqual(selected['model_path'], 'tinker://sft')
            self.assertEqual(selected['sampling_model'], 'artsy/test-sft')
            with self.assertRaises(RuntimeError):
                worker.bridge_health('tinker://other', 'artsy/test-sft')
            with self.assertRaises(RuntimeError):
                worker.bridge_health('tinker://sft', 'artsy/missing')

    def test_registered_sft_alias_requires_eligible_local_checkpoint(self):
        alias_request = {**request('/tmp/example'), 'model': 'artsy/test-sft'}
        with patch('inference.model_catalog.model_catalog', return_value=[{'id': 'artsy/test-sft', 'model_path': 'tinker://sft'}]), patch('training.dpo.load_initial_checkpoint', return_value={'sampler_path': 'tinker://sft'}) as load:
            self.assertEqual(worker.validate_collection_request(alias_request)['model'], 'artsy/test-sft')
            load.assert_called_once_with(worker.ROOT / 'checkpoints/test-sft/final.json')
        with patch('inference.model_catalog.model_catalog', return_value=[]):
            with self.assertRaises(ValueError):
                worker.validate_collection_request(alias_request)

    def test_health_checkpoint_must_match(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({'model': MODEL_ID, 'model_path': 'actual'}).encode()
        with patch.object(worker, 'urlopen', return_value=response):
            with self.assertRaises(RuntimeError):
                worker.bridge_health('expected')

    def test_native_collection_preserves_logs_and_fails_fast(self):
        from types import SimpleNamespace
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant
        for successful in (True, False):
            with self.subTest(successful=successful), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'new-run'
                calls = []
                messages = [ChatMessageSystem(content='native system'), ChatMessageUser(content='native task'),
                            ChatMessageAssistant(content='actual policy')]
                sample = SimpleNamespace(messages=messages, error=None if successful else 'native error')
                log = SimpleNamespace(status='success' if successful else 'error', samples=[sample],
                                      model_dump=lambda **kw: {'samples': [{'messages': 'resolved'}]})
                async def evaluate(task, **kwargs):
                    calls.append(task)
                    provider = Mock(api=Mock(generate=AsyncMock(return_value=('native-output', 'api-metadata'))))
                    generated = await task['generation_filter'](provider, [], [], 'auto', Mock())
                    self.assertEqual(generated, 'native-output')
                    provider.api.generate.assert_awaited_once()
                    directory = Path(kwargs['log_dir']).parent
                    (directory / 'logs').mkdir()
                    (directory / 'logs/run.eval').write_bytes(b'Inspect log placeholder')
                    artifact = directory / 'scored' / (task['spec']['id'] + '-1')
                    artifact.mkdir(parents=True)
                    (artifact / 'program.py').write_text('native code\n')
                    Image.new('RGB', (1920, 1200)).save(artifact / 'image.png')
                    (artifact / 'result.json').write_text(json.dumps({'outcome':'submitted', 'submitted':True,
                        'scores': {'render_valid': 1}, 'feedback': {'ok': True}}))
                    return [log]
                def task(specs, artifacts, **kwargs):
                    self.assertEqual(kwargs['temperature'], 0.8)
                    self.assertEqual(kwargs['max_tokens'], 8192)
                    self.assertIsNone(kwargs.get('correctness_judge_model'))
                    return {'spec': specs[0], 'generation_filter': kwargs['generation_filter']}
                with patch.object(worker, '_load_env_file'), patch.object(worker, 'bridge_health', return_value={'model_path':'tinker://sft'}), patch('inspect_ai.eval_async', side_effect=evaluate), patch('inspect_ai.log.read_eval_log', return_value=log), patch('evaluation.harness.drawing_task', side_effect=task), patch('evaluation.harness.model_from_env', return_value=Mock()), patch('training.data.convert_record') as convert:
                    summary = asyncio.run(worker.collect(request(output)))
                self.assertEqual(len(calls), 2 if successful else 1)
                self.assertEqual(convert.call_count, 2 if successful else 0)
                self.assertEqual(summary['complete'], successful)
                self.assertEqual(json.loads((output / 'sampling.json').read_text())['mode'], 'native-inspect-v1')
                rows = [json.loads(line) for line in (output / 'candidates.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['teacher']['model_path'], 'tinker://sft')
                self.assertEqual(rows[0]['nativeTrajectory']['messages'][2]['content'][0]['text'], 'actual policy')
                self.assertTrue((output / rows[0]['artifacts']['trajectory']).exists())
                self.assertEqual(rows[0]['sourceHash'], __import__('hashlib').sha256(b'native code').hexdigest())

    def test_portable_trajectory_keeps_thinking_tools_and_previews(self):
        image = io.BytesIO()
        Image.new('RGB', (1920, 1200)).save(image, 'PNG')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            trajectory, digest = worker.portable_trajectory([
                {'role': 'assistant', 'content': [{'type':'reasoning','reasoning':'actual thinking'}],
                 'tool_calls': [{'id':'call-1','function':'bash','arguments':{'command':'preview'}}]},
                {'role':'tool','tool_call_id':'call-1','content': [{'type':'text','text':'Return code: 0'},
                 {'type':'image','image':'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()}]}], output, output)
            self.assertEqual(trajectory['messages'][0]['content'][0], {'type':'thinking','thinking':'actual thinking'})
            self.assertEqual(trajectory['messages'][0]['tool_calls'][0]['function']['arguments']['command'], 'preview')
            self.assertEqual(trajectory['trainable_message_indices'], [0])
            preview = trajectory['messages'][1]['content'][1]
            self.assertEqual((output / preview['image']).read_bytes(), image.getvalue())
            from training.data import native_trajectory_hash
            self.assertEqual(digest, native_trajectory_hash(trajectory))


if __name__ == '__main__':
    unittest.main()
