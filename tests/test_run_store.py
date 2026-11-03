"""Local persistence and ownership tests; no hosted services required."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.run_store import atomic_json, run_lock
from training.grpo import validate_resume


class RunStoreTests(unittest.TestCase):
    def test_lock_excludes_other_process_and_releases_on_death(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ('from pathlib import Path; from training.run_store import run_lock; '
                    'import sys; p=Path(sys.argv[1]); \n'
                    'with run_lock(p):\n print("locked", flush=True); sys.stdin.read()')
            proc = subprocess.Popen([sys.executable, '-c', code, directory], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(proc.stdout.readline().strip(), 'locked')
                with self.assertRaisesRegex(ValueError, 'already owned'):
                    with run_lock(Path(directory)):
                        pass
                proc.kill()
                proc.wait()
                with run_lock(Path(directory)):
                    pass
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdin.close()
                proc.stdout.close()

    def test_failed_replace_preserves_prior_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            atomic_json(path, {'old': 1})
            with patch('training.run_store.os.replace', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    atomic_json(path, {'new': 2})
            self.assertEqual(json.loads(path.read_text()), {'old': 1})
            self.assertEqual([p.name for p in path.parent.iterdir()], ['manifest.json'])

    def test_manifest_only_recovery_and_lineage_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / 'training-manifest.json', {})
            config = {'continuation': {'start_step': 3, 'state_path': 'tinker://source',
                                       'sampler_path': 'tinker://sampler'}}
            validate_resume(root, config, 5)
            (root / 'checkpoints.jsonl').write_text(json.dumps({'name': 'continuation', 'batch': 3,
                'state_path': 'tinker://other', 'sampler_path': 'tinker://sampler'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'source'):
                validate_resume(root, config, 5)

    def test_torn_tail_is_backed_up_and_committed_prefix_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'checkpoints.jsonl'
            committed = json.dumps({'name': 'step', 'batch': 1, 'state_path': 'tinker://one'}) + '\n'
            fragment = b'{"batch": 2, "state_path": "tinker://'
            path.write_bytes(committed.encode() + fragment)
            with run_lock(root):
                validate_resume(root, {}, 3)
            self.assertEqual(path.read_text(), committed)
            backups = list(root.glob('checkpoints.jsonl.torn-*.bak'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), fragment)

    def test_corruption_is_never_masked_as_a_torn_tail(self):
        valid = json.dumps({'batch': 1, 'state_path': 'tinker://one'}).encode() + b'\n'
        cases = [b'{broken}\n' + valid + b'{torn', valid + b'{broken}\n',
                 b'{torn', valid + b'{"batch": 0}\n' + b'{torn',
                 valid + json.dumps({'batch': 2, 'state_path': 'tinker://two'}).encode()]
        for data in cases:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / 'checkpoints.jsonl'
                path.write_bytes(data)
                with run_lock(root), self.assertRaises(ValueError):
                    validate_resume(root, {}, 3)
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(list(root.glob('*.bak')), [])
