"""HPSv3 scores must describe trusted final images, not previews or failures."""
import asyncio
import io
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image
from inspect_ai.scorer import Target
from evaluation.harness import drawing_score, drawing_task
from evaluation.evaluate import apply_reporting


class HPSv3EvaluationTests(unittest.TestCase):
    def score(self, root, *, submitted=True, failure=False):
        buffer = io.BytesIO()
        Image.new('RGB', (1920, 1200)).save(buffer, 'PNG')
        png = buffer.getvalue()
        async def read(path, text=True):
            if path == '/workspace/program.py':
                return 'trusted final source'
            if path == '/workspace/refusal.json':
                raise FileNotFoundError(path)
            return png
        box = SimpleNamespace(read_file=AsyncMock(side_effect=read), write_file=AsyncMock(),
                              exec=AsyncMock(return_value=SimpleNamespace(success=True,
                                  stdout=json.dumps({'ok': True}))))
        judge = SimpleNamespace(score=AsyncMock(return_value={
            'model': 'pinned', 'hpsv3_score': -3.5, 'preference': 0.33}))
        if failure:
            judge.score.side_effect = RuntimeError('scorer unavailable')
        state = SimpleNamespace(sample_id='a', epoch=1, model='test', messages=[],
                                metadata={'prompt': 'Draw a fox', 'seed': 1, 'submitted': submitted})
        with patch('evaluation.harness.sandbox', return_value=box), \
             patch('evaluation.harness.HPSv3Client', return_value=judge):
            result = asyncio.run(drawing_score(str(root), hpsv3_url='http://score',
                                               hpsv3_model='pinned')(state, Target('')))
        return result, judge, png

    def test_valid_final_render_keeps_negative_raw_score(self):
        with tempfile.TemporaryDirectory() as folder:
            result, judge, png = self.score(Path(folder))
            judge.score.assert_awaited_once_with('Draw a fox', png)
            self.assertEqual(result.value['hpsv3_score'], -3.5)
            self.assertEqual(result.value['hpsv3_preference'], 0.33)
            self.assertEqual(result.metadata['judgments']['hpsv3']['model'], 'pinned')

    def test_missing_submission_is_not_sent_to_scorer(self):
        with tempfile.TemporaryDirectory() as folder:
            result, judge, _ = self.score(Path(folder), submitted=False)
            judge.score.assert_not_awaited()
            self.assertEqual(result.value['hpsv3_preference'], 0)
            self.assertTrue(math.isnan(result.value['hpsv3_score']))
            self.assertNotIn('hpsv3_score', result.metadata['scores'])

    def test_scorer_failure_is_sample_error_not_zero_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'scorer unavailable'):
                self.score(Path(folder), failure=True)
            self.assertFalse((Path(folder) / 'a-1/result.json').exists())

    def test_partial_service_configuration_rejected(self):
        for options in ({'hpsv3_url': 'http://score'}, {'hpsv3_model': 'pinned'}):
            with self.assertRaises(ValueError):
                drawing_task([], '/tmp/unused-hps-test', **options)

    def test_reporting_distinguishes_raw_image_scores_from_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            specs = [{'id': name} for name in ('a', 'b', 'c')]
            for name, scores in [('a', {'hpsv3_score': -2, 'hpsv3_preference': 0.4}),
                                 ('b', {'hpsv3_preference': 0})]:
                directory = root / (name + '-1'); directory.mkdir()
                (directory / 'result.json').write_text(json.dumps({'scores': scores}))
            summary = apply_reporting({'hpsv3_model': 'pinned'}, specs, root, 1)
            self.assertEqual(summary['hpsv3']['images_scored'], 1)
            self.assertEqual(summary['hpsv3']['raw_score_mean'], -2)
            self.assertEqual(summary['hpsv3']['preference_mean_completed'], 0.2)
            self.assertAlmostEqual(summary['hpsv3']['preference_mean_expected'], 0.4 / 3)


class HPSv3CLITests(unittest.TestCase):
    def test_hps_mode_replaces_configured_llm_judges_in_plan(self):
        from evaluation.evaluate import main
        import os
        import sys
        args = ['evaluate', '--dataset', 'datasets/parti-prompts/evaluation/prompts.jsonl',
                '--hpsv3-url', 'http://score', '--hpsv3-model', 'pinned', '--dry-run']
        output = io.StringIO()
        with patch.object(sys, 'argv', args), patch.dict(os.environ, {
                'ARTSY_CORRECTNESS_JUDGE_MODEL': 'old-judge',
                'ARTSY_SAFETY_JUDGE_MODEL': 'old-safety'}), redirect_stdout(output):
            self.assertEqual(main(), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['mode'], 'hpsv3')
        self.assertEqual(plan['hpsv3_model'], 'pinned')
        self.assertIsNone(plan['correctness_judge_model'])
        self.assertIsNone(plan['safety_judge_model'])
        self.assertIsNone(plan['code_safety_judge_model'])
        self.assertIsNone(plan['adherence_rubric'])


class HPSv3ComparisonTests(unittest.TestCase):
    def test_comparison_excludes_missing_images_from_raw_pairs(self):
        from evaluation.compare_hpsv3 import compare
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = {}
            for name, values in [('rank32', [-2.0, None]), ('rank64', [-3.0, 1.0])]:
                path = root / name; path.mkdir(); runs[name] = path
                (path / 'summary.json').write_text(json.dumps({
                    'status': 'success', 'sample_errors': 0, 'ids': ['a', 'b'],
                    'epochs': 1, 'hpsv3_model': 'pinned'}))
                for sid, raw in zip(['a', 'b'], values):
                    directory = path / 'artifacts' / (sid + '-1'); directory.mkdir(parents=True)
                    scores = {'render_valid': int(raw is not None), 'hpsv3_preference': 0.4 if raw is not None else 0}
                    if raw is not None: scores['hpsv3_score'] = raw
                    (directory / 'result.json').write_text(json.dumps({
                        'scores': scores, 'hpsv3Model': 'pinned', 'prompt': sid,
                        'outcome': 'submitted' if raw is not None else 'incomplete'}))
            result = compare(runs, root / 'comparison')
            self.assertEqual(result['models']['rank32']['raw_mean_valid_images'], -2)
            self.assertEqual(result['models']['rank32']['preference_mean_all_episodes'], 0.2)
            pair = result['matched_image_pairs'][0]
            self.assertEqual(pair['matched_images'], 1)
            self.assertEqual(pair['wins_a'], 1)
            summary_path = runs['rank64'] / 'summary.json'
            summary = json.loads(summary_path.read_text()); summary['hpsv3_model'] = 'different'
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                compare(runs, root / 'comparison')


class InspectAggregationTests(unittest.TestCase):
    def test_raw_reward_metric_excludes_nan_missing_image_sentinel(self):
        from inspect_ai import Task, eval
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import scorer, Score, mean, stderr
        from inspect_ai.solver import solver
        @solver
        def noop():
            async def solve(state, generate): return state
            return solve
        @scorer(metrics={'*': [mean(), stderr()]})
        def mixed():
            async def score(state, target):
                return Score(value={'hpsv3_score': -3.0 if state.sample_id == 'valid' else float('nan'),
                                    'hpsv3_preference': 0.3 if state.sample_id == 'valid' else 0.0})
            return score
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('inspect_ai._util.appdirs.user_data_path', side_effect=lambda name: root / 'data' / name), \
                 patch('inspect_ai._util.appdirs.user_cache_path', side_effect=lambda name: root / 'cache' / name):
                logs = eval(Task(dataset=[Sample(id='valid', input='x'), Sample(id='missing', input='x')],
                                 solver=noop(), scorer=mixed()), model='mockllm/model',
                            log_dir=str(root / 'logs'), display='none')
            self.assertEqual(logs[0].status, 'success')
            metrics = {score.name: score for score in logs[0].results.scores}
            self.assertEqual(metrics['hpsv3_score'].metrics['mean'].value, -3)
            self.assertEqual(metrics['hpsv3_score'].scored_samples, 1)
            self.assertEqual(metrics['hpsv3_preference'].metrics['mean'].value, 0.15)
