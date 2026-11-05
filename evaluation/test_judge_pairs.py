"""Four final images must be graded together with anonymous labels."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from evaluation import judge_pairs, compare_pipeline


class FourModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = {}
        for index, role in enumerate(('base', 'rank32', 'rank64', 'rank128')):
            run = self.root / role
            directory = run / 'artifacts/prompt-1'
            directory.mkdir(parents=True)
            content = f'png-{index}'.encode()
            (directory / 'image.png').write_bytes(content)
            (directory / 'result.json').write_text(json.dumps({
                'prompt': 'a shoe', 'seed': 42, 'scores': {'render_valid': True},
                'imageHash': hashlib.sha256(content).hexdigest(), 'outcome': 'submitted'}))
            (run / 'summary.json').write_text(json.dumps({
                'ids': ['prompt'], 'model': f'private-model-{index}',
                'status': 'success', 'completed_samples': 1, 'epochs': 1}))
            self.runs[role] = run

    def test_one_request_contains_all_four_final_images_and_returns_all_four_scores(self):
        messages = []

        class Judge:
            async def complete(self, request, *args, **kwargs):
                messages.append(request)
                answer = {label: {'adherence_score': 3, 'visual_quality_score': 2, 'comments': 'Observed detail.'}
                          for label in 'ABCD'}
                answer.update(preferred='D', comparison='D best fulfills the request.')
                return {'text': json.dumps(answer), 'finishReason': 'stop', 'model': 'openai/gpt-6-luna'}

            async def aclose(self):
                pass

        args = argparse.Namespace(base=None, sft=None, run=[f'{k}={v}' for k, v in self.runs.items()],
                                  output=self.root / 'comparison', concurrency=1, limit=None)
        with patch.object(judge_pairs, 'chat_client', return_value=Judge()), patch('builtins.print'):
            asyncio.run(judge_pairs.main(args))
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]), 1)
        content = messages[0][0]['content']
        self.assertEqual(sum(part['type'] == 'image_url' for part in content), 4)
        text = '\n'.join(part['text'] for part in content if part['type'] == 'text')
        self.assertNotIn('private-model', text)
        self.assertNotIn('Ignore implementation/code', text)
        self.assertIn('Images A, B, C, D are anonymous', text)
        saved = json.loads((args.output / 'records/prompt.json').read_text())
        self.assertEqual(set(saved['scores']), set(self.runs))
        self.assertEqual(saved['image_count'], 4)
        self.assertEqual(saved['preferred'], saved['label_mapping']['D'])
        summary = json.loads((args.output / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(set(summary['models']), set(self.runs))

    def test_mismatched_seeds_and_changed_final_images_are_rejected(self):
        path = self.runs['rank128'] / 'artifacts/prompt-1/result.json'
        data = json.loads(path.read_text())
        data['seed'] = 43
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'seed mismatch'):
            judge_pairs.inputs(runs=self.runs)
        data['seed'] = 42
        path.write_text(json.dumps(data))
        (path.parent / 'image.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            judge_pairs.inputs(runs=self.runs)

    def test_missing_fourth_image_requires_null_score(self):
        _, pairs = judge_pairs.inputs(runs=self.runs)
        pairs[0]['members']['rank128']['image'] = None
        mapping = dict(zip('ABCD', self.runs))
        answer = {label: {'adherence_score': 3, 'visual_quality_score': 2, 'comments': 'Detail.'} for label in mapping}
        answer.update(preferred='C', comparison='C is best.')
        with self.assertRaises(ValueError):
            judge_pairs.validate(answer, mapping, pairs[0])
        answer['D'].update(adherence_score=None, visual_quality_score=None)
        judge_pairs.validate(answer, mapping, pairs[0])

    def test_pipeline_starts_four_model_judging_after_evals_finish(self):
        args = argparse.Namespace(run=[f'{role}={path}' for role, path in self.runs.items()],
                                  output=self.root / 'pipeline', judge='llm')
        with patch.object(compare_pipeline.subprocess, 'run') as launch:
            launch.return_value.returncode = 0
            compare_pipeline.run(args)
        command = launch.call_args.args[0]
        self.assertEqual(command.count('--run'), 4)
        self.assertIn('evaluation.judge_pairs', command)
        state = json.loads((args.output / 'pipeline.json').read_text())
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(list(state['completed_prompts'].values()), [1, 1, 1, 1])

    def test_truncated_judge_reply_is_rejected(self):
        class Judge:
            async def complete(self, request, *args, **kwargs):
                return {'text': '{}', 'finishReason': 'length'}

            async def aclose(self):
                pass

        args = argparse.Namespace(base=None, sft=None, run=[f'{k}={v}' for k, v in self.runs.items()],
                                  output=self.root / 'truncated', concurrency=1, limit=None, judge='llm')
        with patch.object(judge_pairs, 'chat_client', return_value=Judge()), \
                patch.object(judge_pairs.asyncio, 'sleep', new=AsyncMock()), patch('builtins.print'):
            with self.assertRaises(SystemExit):
                asyncio.run(judge_pairs.main(args))
        self.assertFalse((args.output / 'records/prompt.json').exists())


if __name__ == '__main__':
    unittest.main()
