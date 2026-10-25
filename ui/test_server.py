"""Job lifecycle checks without model requests or Docker."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import server


class JobTests(unittest.TestCase):
    def test_command_and_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            request = {'prompt': "draw 'a'; $(anything)", 'seed': 42, 'model': 'Qwen/Qwen3.5-4B'}
            with patch.object(server.subprocess, 'run') as run:
                run.return_value.returncode = 0
                server.ACTIVE = 'test'
                server.run_job(directory, request)
            args = run.call_args.args[0]
            self.assertEqual(args[-3:], [request['prompt'], '42', request['model']])
            self.assertFalse(run.call_args.kwargs.get('shell', False))
            self.assertEqual(json.loads((directory / 'job.json').read_text())['state'], 'completed')
            self.assertIsNone(server.ACTIVE)

    def test_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(server.subprocess, 'run', side_effect=OSError('missing Python')):
                server.ACTIVE = 'test'
                server.run_job(directory, {'prompt': 'circle', 'seed': 0, 'model': 'test'})
            status = json.loads((directory / 'job.json').read_text())
            self.assertEqual(status['state'], 'failed')
            self.assertEqual(status['error'], 'missing Python')
            self.assertIsNone(server.ACTIVE)


if __name__ == '__main__':
    unittest.main()
