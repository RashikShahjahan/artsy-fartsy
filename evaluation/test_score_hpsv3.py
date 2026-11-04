"""Deferred scoring must not load weights until all generation is complete."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image
from evaluation.score_hpsv3 import score_runs
from evaluation.resume_eval import prepare_retry_log


class DeferredScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = {}
        image = io.BytesIO(); Image.new('RGB', (1920, 1200)).save(image, 'PNG')
        for name in ('rank32', 'rank64'):
            path = self.root / name; path.mkdir(); self.runs[name] = path
            (path / 'summary.json').write_text(json.dumps({'status': 'success', 'sample_errors': 0,
                'ids': ['valid', 'missing'], 'epochs': 1, 'hpsv3_model': None}))
            for sid in ('valid', 'missing'):
                directory = path / 'artifacts' / (sid + '-1'); directory.mkdir(parents=True)
                result = {'prompt': sid, 'outcome': 'submitted' if sid == 'valid' else 'incomplete',
                          'scores': {'render_valid': int(sid == 'valid')}}
                if sid == 'valid':
                    (directory / 'image.png').write_bytes(image.getvalue())
                    result['imageHash'] = hashlib.sha256(image.getvalue()).hexdigest()
                (directory / 'result.json').write_text(json.dumps(result))
        self.output = self.root / 'scores'
        self.model = SimpleNamespace(score=Mock(return_value=-2.0))
        self.load = Mock(return_value=self.model)

    def test_incomplete_run_prevents_model_load_and_any_scoring(self):
        p = self.runs['rank64'] / 'summary.json'
        value = json.loads(p.read_text()); value['status'] = 'incomplete'; p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Generation must complete'):
            score_runs(self.runs, self.output, 'pinned', self.load)
        self.load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_batch_loads_once_preserves_generation_and_resumes_without_reload(self):
        before = {p: p.read_bytes() for run in self.runs.values() for p in run.rglob('*') if p.is_file()}
        result = score_runs(self.runs, self.output, 'pinned', self.load)
        self.load.assert_called_once(); self.assertEqual(self.model.score.call_count, 2)
        self.assertEqual(result['models']['rank32']['valid_images'], 1)
        self.assertEqual(result['models']['rank32']['raw_mean_valid_images'], -2)
        self.assertEqual(result['matched_image_pairs'][0]['matched_images'], 1)
        for path, data in before.items(): self.assertEqual(path.read_bytes(), data)
        load_again = Mock(side_effect=AssertionError('must reuse saved scores'))
        score_runs(self.runs, self.output, 'pinned', load_again)
        load_again.assert_not_called()
        self.assertFalse((self.output / '.hpsv3-score.lock').exists())

    def test_tampered_image_fails_before_gpu_load(self):
        (self.runs['rank64'] / 'artifacts/valid-1/image.png').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            score_runs(self.runs, self.output, 'pinned', self.load)
        self.load.assert_not_called()

    def test_existing_matching_inline_scores_do_not_load_model(self):
        for run in self.runs.values():
            p = run / 'artifacts/valid-1/result.json'
            result = json.loads(p.read_text()); result['hpsv3Model'] = 'pinned'
            result['scores'].update(hpsv3_score=-1.0, hpsv3_preference=0.45)
            p.write_text(json.dumps(result))
        score_runs(self.runs, self.output, 'pinned', self.load)
        self.load.assert_not_called()

    def test_new_model_rejects_saved_score_checkpoint(self):
        score_runs(self.runs, self.output, 'pinned', self.load)
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            score_runs(self.runs, self.output, 'different', self.load)

    def test_retry_changes_scoring_only_and_preserves_original_metadata(self):
        score = SimpleNamespace(value={'render_valid': 1, 'hpsv3_score': -2}, metadata={'hpsv3Model': 'pinned'})
        log = SimpleNamespace(samples=[SimpleNamespace(scores={'drawing_score': score})],
            eval=SimpleNamespace(task_file=None, task_args={'hpsv3_url': 'http://score',
                'hpsv3_model': 'pinned', 'max_turns': 25},
                task_args_passed={'hpsv3_url': 'http://score', 'hpsv3_model': 'pinned'},
                scorers=[SimpleNamespace(name='drawing_score', options={'hpsv3_url': 'http://score', 'hpsv3_model': 'pinned'})], model_args={}))
        prepare_retry_log(log, execution_only=True)
        self.assertIsNone(log.eval.task_args['hpsv3_url'])
        self.assertIsNone(log.eval.task_args_passed['hpsv3_url'])
        self.assertIsNone(log.eval.scorers[0].options['hpsv3_url'])
        self.assertEqual(log.eval.task_args['max_turns'], 25)
        self.assertEqual(score.value, {'render_valid': 1})
        self.assertEqual(score.metadata['hpsv3Model'], 'pinned')


class DeferredPipelineTests(unittest.TestCase):
    def test_scoring_process_starts_only_after_last_generation_run_completes(self):
        import os
        from unittest.mock import patch
        from evaluation.compare_pipeline import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, last = root / 'first', root / 'last'
            first.mkdir(); last.mkdir()
            success = json.dumps({'status': 'success', 'completed_samples': 1})
            (first / 'summary.json').write_text(success)
            Path(str(last) + '.pid').write_text(str(os.getpid()))
            args = SimpleNamespace(run=[f'first={first}', f'last={last}'], output=root / 'comparison',
                judge='hpsv3', hpsv3_python='isolated-python', hpsv3_python_path=[], hpsv3_cache=root / 'cache')
            with patch('evaluation.compare_pipeline.subprocess.run', return_value=SimpleNamespace(returncode=0)) as launch:
                def finish_generation(_):
                    launch.assert_not_called()
                    (last / 'summary.json').write_text(success)
                with patch('evaluation.compare_pipeline.time.sleep', side_effect=finish_generation):
                    run(args)
            launch.assert_called_once()
            self.assertIn('evaluation.score_hpsv3', launch.call_args.args[0])
            self.assertEqual(json.loads((args.output / 'pipeline.json').read_text())['status'], 'complete')
