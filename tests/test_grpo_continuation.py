"""Continuation planning is local; no Docker, judge, or Tinker calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from drawing.config import IMAGE_SIZE, MODEL_ID, MODEL_REVISION
from training import grpo
from training.sft import batches


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / 'source'
        self.origin.mkdir()
        self.dataset = self.root / 'prompts.jsonl'
        self.specs = [{'id': str(i), 'prompt': f'Draw circle {i}', 'seed': i} for i in range(5)]
        self.dataset.write_text(''.join(json.dumps(s) + '\n' for s in self.specs))
        defaults = vars(grpo.parser().parse_args([]))
        self.config = {k: defaults[k] for k in grpo.INHERITED_PARAMETERS}
        self.config.update(algorithm=grpo.ALGORITHM, model=MODEL_ID,
                           tokenizer_revision=MODEL_REVISION, renderer=grpo.RENDERER,
                           image_size=IMAGE_SIZE, dataset_sha256=grpo.digest(self.dataset.read_bytes()),
                           prompt_ids=[s['id'] for s in self.specs],
                           source_sha256={p: grpo.digest((grpo.ROOT / p).read_bytes())
                                          for p in grpo.PROVENANCE_FILES},
                           reference_sampler_path='tinker://original-sft-reference',
                           initial_state_path='tinker://original-sft-state',
                           correctness_judge_model='saved-judge', correctness_judge_url='saved-url')
        self.config['source_sha256']['training/grpo.py'] = grpo.LEGACY_GRPO_SHA256
        self.record = dict(name='final', batch=3, state_path='tinker://rl-state',
                           sampler_path='tinker://rl-sampler')
        self.write_source()
        self.args = grpo.parser().parse_args([
            '--continue-from', str(self.origin), '--additional-steps', '4',
            '--dataset', str(self.dataset), '--output-dir', str(self.root / 'continued')])

    def write_source(self):
        (self.origin / 'training-manifest.json').write_text(json.dumps(
            dict(config=self.config, total_steps=3)))
        (self.origin / 'final.json').write_text(json.dumps(dict(
            config=self.config, step=3, state_path=self.record['state_path'],
            sampler_path=self.record['sampler_path'])))
        (self.origin / 'checkpoints.jsonl').write_text(json.dumps(self.record) + '\n')

    def test_global_order_inheritance_and_optimizer_resume_record(self):
        specs, plan, total = grpo.load_continuation(self.args)
        self.assertEqual(total, 7)
        self.assertEqual(self.args.epochs, 3)
        self.assertEqual(plan['reference_sampler_path'], 'tinker://original-sft-reference')
        self.assertEqual(plan['initial_state_path'], 'tinker://original-sft-state')
        self.assertEqual(plan['correctness_judge_model'], 'saved-judge')
        original = list(batches(specs, 2, 1, 42))
        extended = list(batches(specs, 2, self.args.epochs, self.args.seed))
        self.assertEqual(extended[:3], original)
        self.assertNotEqual(extended[3], extended[0])
        self.args.output_dir.mkdir()
        grpo.seed_continuation(self.args.output_dir, plan)
        path = self.args.output_dir / 'checkpoints.jsonl'
        seed = json.loads(path.read_text())
        self.assertEqual(seed['batch'], 3)
        self.assertEqual(seed['state_path'], 'tinker://rl-state')
        from tinker_cookbook import checkpoint_utils
        recovered = checkpoint_utils.get_last_checkpoint(str(self.args.output_dir))
        self.assertEqual(recovered.batch, 3)
        self.assertEqual(recovered.state_path, 'tinker://rl-state')
        path.write_text('interrupted run checkpoint\n')
        grpo.seed_continuation(self.args.output_dir, plan)
        self.assertEqual(path.read_text(), 'interrupted run checkpoint\n')

    def test_dry_run_is_read_only_and_never_starts_services(self):
        before = {p: p.read_bytes() for p in self.origin.iterdir()}
        with patch.object(grpo, '_build_sandbox_image') as docker, patch.object(grpo, 'train') as train:
            self.assertEqual(grpo.main([
                '--continue-from', str(self.origin), '--additional-steps', '4',
                '--dataset', str(self.dataset), '--output-dir', str(self.args.output_dir),
                '--dry-run']), 0)
        docker.assert_not_called()
        train.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())
        self.assertEqual(before, {p: p.read_bytes() for p in self.origin.iterdir()})

    def test_runtime_and_dataset_drift_rejected(self):
        self.config['source_sha256']['runtime/artcanvas.py'] = 'changed'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'dependencies changed'):
            grpo.load_continuation(self.args)
        self.config['source_sha256']['runtime/artcanvas.py'] = grpo.digest(
            (grpo.ROOT / 'runtime/artcanvas.py').read_bytes())
        self.write_source()
        self.dataset.write_text(self.dataset.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'dataset changed'):
            grpo.load_continuation(self.args)

    def test_source_must_be_complete_and_output_distinct(self):
        self.record['name'] = '000003'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'completed'):
            grpo.load_continuation(self.args)
        self.args.output_dir = self.origin
        with self.assertRaisesRegex(ValueError, 'new output'):
            grpo.load_continuation(self.args)

    def test_cli_rejects_parameter_drift_or_unbounded_continuation(self):
        for extras in ([], ['--additional-steps', '0'],
                       ['--additional-steps', '4', '--learning-rate', '0.1']):
            with self.assertRaises(SystemExit):
                grpo.main(['--continue-from', str(self.origin), *extras, '--dry-run'])

    def test_chained_continuation_keeps_epoch_position(self):
        specs, plan, total = grpo.load_continuation(self.args)
        self.config = plan
        self.record['batch'] = total
        self.write_source()
        manifest = self.origin / 'training-manifest.json'
        value = json.loads(manifest.read_text())
        value['total_steps'] = total
        manifest.write_text(json.dumps(value))
        final = self.origin / 'final.json'
        value = json.loads(final.read_text())
        value['step'] = total
        final.write_text(json.dumps(value))
        _, chained, end = grpo.load_continuation(self.args)
        self.assertEqual(chained['continuation']['start_step'], 7)
        self.assertEqual(end, 11)
        first = list(batches(specs, 2, 3, 42))
        next_run = list(batches(specs, 2, self.args.epochs, 42))
        self.assertEqual(first[7:9], next_run[7:9])
        self.assertEqual(chained['reference_sampler_path'], 'tinker://original-sft-reference')

    def test_main_persists_plan_and_resumes_without_replacing_checkpoint(self):
        command = ['--continue-from', str(self.origin), '--additional-steps', '4',
                   '--dataset', str(self.dataset), '--output-dir', str(self.args.output_dir)]
        observed = []

        async def train(args, specs, config, total):
            persisted = json.loads((args.output_dir / 'training-manifest.json').read_text())
            self.assertEqual(persisted['config'], config)
            self.assertEqual(persisted['total_steps'], total)
            observed.append(json.loads((args.output_dir / 'checkpoints.jsonl').read_text()))

        with patch.object(grpo, '_build_sandbox_image') as docker, patch.object(grpo, 'train', train):
            self.assertEqual(grpo.main(command), 0)
            self.assertEqual(observed[-1]['batch'], 3)
            self.assertEqual(observed[-1]['state_path'], 'tinker://rl-state')
            resumed = dict(name='000005', batch=5, state_path='tinker://interrupted-optimizer',
                           sampler_path='tinker://interrupted-sampler')
            checkpoint = self.args.output_dir / 'checkpoints.jsonl'
            checkpoint.write_text(json.dumps(resumed) + '\n')
            self.assertEqual(grpo.main(command), 0)
            self.assertEqual(observed[-1], resumed)
            self.assertEqual(docker.call_count, 2)
            altered = command.copy()
            altered[altered.index('--additional-steps') + 1] = '5'
            with self.assertRaises(SystemExit):
                grpo.main(altered)
            self.assertEqual(docker.call_count, 2)
            self.assertEqual(len(observed), 2)
            self.assertEqual(json.loads(checkpoint.read_text()), resumed)

    def test_nested_output_and_abbreviated_options_rejected(self):
        self.args.output_dir = self.origin / 'nested'
        with self.assertRaisesRegex(ValueError, 'new output'):
            grpo.load_continuation(self.args)
        with self.assertRaises(SystemExit):
            grpo.main(['--continue-from', str(self.origin), '--additional-steps', '4',
                       '--learn', '0.1', '--dry-run'])
