"""Native Tinker/mini-agent adapter smoke test without hosted inference."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from client.sandbox import agent


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_tool_call_uses_file_rpc_and_records_trajectory(self):
        class Backend:
            def generate(self, messages, tools, budget, context):
                self.messages = messages
                return {'role': 'assistant', 'content': 'Ready', 'tool_calls': [
                    {'id': 'one', 'type': 'function', 'function': {
                        'name': 'bash', 'arguments': {'command': 'artsy-submit program.py'}}}]}, 12, 3
        backend = Backend()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'request.json').write_text(json.dumps({'prompt': 'A fox'}))
            async def relay():
                for _ in range(100):
                    if (root / 'action.json').exists():
                        break
                    await asyncio.sleep(0.01)
                action = json.loads((root / 'action.json').read_text())
                self.assertEqual(action['command'], 'artsy-submit program.py')
                (root / 'reply-1.json').write_text(json.dumps({'returncode': 0,
                    'content': [{'type': 'text', 'text': 'Submitted'}], 'terminal': True,
                    'entrypoint': 'program.py'}))
            with patch.object(agent, 'ROOT', root), patch.object(agent, 'TinkerBackend', return_value=backend), patch.dict('os.environ', {'ARTSY_TINKER_MODEL_PATH': 'tinker://test'}):
                await asyncio.wait_for(asyncio.gather(agent.main(), relay()), timeout=10)
            self.assertTrue(json.loads((root / 'done.json').read_text())['ok'])
            trace = json.loads((root / 'trace.json').read_text())
            self.assertTrue(any(m['role'] == 'tool' for m in trace['messages']))
            self.assertEqual(backend.messages[-1]['role'], 'user')
            self.assertTrue((root / 'thinking.json').exists())


if __name__ == '__main__':
    unittest.main()
