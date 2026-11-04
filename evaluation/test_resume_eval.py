"""Resume preserves completed artifacts and retains a summary on failure."""
import hashlib
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import resume_eval
from inspect_ai.util import SandboxEnvironmentSpec


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.directory = self.output / 'artifacts/finished-1'
        self.directory.mkdir(parents=True)
        png = b'saved final image'
        (self.directory / 'image.png').write_bytes(png)
        (self.directory / 'result.json').write_text(json.dumps({
            'scores': {'render_valid': True}, 'imageHash': hashlib.sha256(png).hexdigest()}))
        self.original = {'status': 'incomplete', 'model': 'rank64', 'sample_errors': 1,
            'ids': ['finished', 'failed'], 'epochs': 1, 'max_samples': 10,
            'max_sandboxes': 10, 'max_connections': 10, 'logs': ['original.eval'], 'dataset': 'unused'}
        (self.output / 'summary.json').write_text(json.dumps(self.original))
        self.retry_source = SimpleNamespace(eval=SimpleNamespace(model_args={}))
        patcher = patch.object(resume_eval, 'read_eval_log', return_value=self.retry_source)
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch.object(resume_eval, 'check_native_bridge')
    def test_retry_rebuilds_summary_without_replacing_completed_artifacts(self, bridge):
        saved = (self.directory / 'result.json').read_bytes()
        serializer = SimpleNamespace(model_dump=lambda **kwargs: {})
        log = SimpleNamespace(status='success', samples=[SimpleNamespace(error=None)] * 2,
                              location='retried.eval', results=serializer, stats=serializer)
        with patch.object(resume_eval, 'eval_retry', return_value=[log]) as retry, \
                patch.object(resume_eval, '_recover_log', side_effect=lambda log, _: log), \
                patch.object(resume_eval, 'load_specs', return_value=[]), \
                patch.object(resume_eval, 'apply_reporting'), patch('builtins.print'):
            self.assertEqual(resume_eval.resume(self.output), 0)
        self.assertEqual(retry.call_args.args[0], [self.retry_source])
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'success')
        self.assertEqual(summary['logs'], ['retried.eval'])
        self.assertEqual((self.directory / 'result.json').read_bytes(), saved)
        self.assertEqual(len(list(self.output.glob('summary.before-resume-*.json'))), 1)

    @patch.object(resume_eval, 'check_native_bridge')
    def test_startup_failure_retains_incomplete_summary(self, bridge):
        with patch.object(resume_eval, 'eval_retry', side_effect=RuntimeError('connection failed')), \
                patch('builtins.print'):
            with self.assertRaises(RuntimeError):
                resume_eval.resume(self.output)
        self.assertEqual(json.loads((self.output / 'summary.json').read_text()), self.original)

    def test_serialized_transport_and_frozen_relative_sandbox_are_restored(self):
        sandbox = SandboxEnvironmentSpec('docker', 'compose.yaml')
        log = SimpleNamespace(eval=SimpleNamespace(task_file='evaluation/harness.py',
            sandbox=sandbox, model_args={'http_client': None, 'client_timeout': None}))
        result = resume_eval.prepare_retry_log(log)
        self.assertIsNone(result.eval.task_file)
        self.assertEqual(result.eval.sandbox.config, str(resume_eval.ROOT / 'evaluation/compose.yaml'))
        self.assertEqual(sandbox.config, 'compose.yaml')
        self.assertIsNotNone(result.eval.model_args['http_client'])
        asyncio.run(result.eval.model_args['http_client'].aclose())


if __name__ == '__main__':
    unittest.main()
