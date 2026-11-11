import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training import grpo
from drawing.config import MODEL_CONTEXT_WINDOW


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / 'source'
        self.origin.mkdir()
        self.dataset = self.root / 'prompts.jsonl'
        self.specs = [{'id': str(i), 'prompt': f'Draw fox {i}', 'seed': i} for i in range(8)]
        self.dataset.write_text(''.join(json.dumps(s) + '\n' for s in self.specs))
        defaults = vars(grpo.parser().parse_args([]))
        self.config = {k: defaults[k] for k in grpo.INHERITED_PARAMETERS}
        old_hashes = {p: grpo.digest((grpo.ROOT / p).read_bytes()) for p in grpo.PROVENANCE_FILES
                      if p != 'training/hosted_context.py'}
        old_hashes['training/grpo.py'] = grpo.LEGACY_CONTEXT_DRIVER_SHA256
        self.config.update(algorithm=grpo.ALGORITHM, reward_mode='rlhf', model='Qwen/Qwen3.5-4B',
            reference_model='Qwen/Qwen3.5-4B:peft:262144', dataset_sha256=grpo.digest(self.dataset.read_bytes()),
            source_sha256=old_hashes, rl_stage_split_version='test-split',
            prompt_ids=[s['id'] for s in self.specs], max_length=MODEL_CONTEXT_WINDOW,
            max_steps=4, epochs=1, batch_size=2, group_size=4)
        self.history = [dict(name=f'{step:06d}', batch=step, state_path=f'tinker://state{step}',
                             sampler_path=f'tinker://sampler{step}') for step in (1, 2)]
        self.write()
        self.args = grpo.parser().parse_args(['--recover-from', str(self.origin), '--dataset', str(self.dataset),
                                             '--output-dir', str(self.root / 'recovered')])
        self.stage = patch.object(grpo, 'stage_specs', return_value=(self.specs, 'test-split'))
        self.stage.start()
        self.addCleanup(self.stage.stop)

    def write(self):
        (self.origin / 'training-manifest.json').write_text(json.dumps({'config': self.config, 'total_steps': 4}))
        (self.origin / 'checkpoints.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in self.history))

    def test_bounded_optimizer_recovery_records_immutable_source(self):
        before = {p.name: p.read_bytes() for p in self.origin.iterdir()}
        specs, plan, total = grpo.load_recovery(self.args)
        self.assertEqual(total, 4)
        self.assertEqual(plan['hosted_model'], 'Qwen/Qwen3.5-4B:peft:262144')
        self.assertEqual(plan['continuation']['start_step'], 2)
        self.assertEqual(plan['continuation']['additional_steps'], 2)
        self.assertEqual(plan['continuation']['source_manifest_sha256'], grpo.digest(before['training-manifest.json']))
        self.assertEqual(plan['continuation']['source_checkpoints_sha256'], grpo.digest(before['checkpoints.jsonl']))
        self.args.output_dir.mkdir()
        grpo.seed_continuation(self.args.output_dir, plan)
        record = json.loads((self.args.output_dir / 'checkpoints.jsonl').read_text())
        self.assertEqual(record['state_path'], 'tinker://state2')
        self.assertEqual(record['batch'], 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.origin.iterdir()})

    def test_rejects_unreviewed_source_drift_and_completed_history(self):
        self.config['source_sha256']['training/cookbook_env.py'] = 'changed'
        self.write()
        with self.assertRaisesRegex(ValueError, 'audited'):
            grpo.load_recovery(self.args)
        self.config['source_sha256']['training/cookbook_env.py'] = grpo.digest((grpo.ROOT / 'training/cookbook_env.py').read_bytes())
        self.history[-1].update(batch=4, name='000004')
        self.write()
        with self.assertRaisesRegex(ValueError, 'history'):
            grpo.load_recovery(self.args)

    def test_cli_dry_run_and_drift_rejection(self):
        base = ['--recover-from', str(self.origin), '--dataset', str(self.dataset),
                '--output-dir', str(self.args.output_dir), '--dry-run']
        with contextlib.redirect_stdout(io.StringIO()), patch.object(grpo, 'train') as train:
            self.assertEqual(grpo.main(base), 0)
        train.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in (['--max-steps', '5'], ['--reward-mode', 'rlhf'], ['--additional-steps', '2']):
                with self.assertRaises(SystemExit):
                    grpo.main(base + extra)
