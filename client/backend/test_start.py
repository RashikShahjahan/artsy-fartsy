"""Single-command launch ownership without starting hosted inference."""
import unittest
from unittest.mock import Mock, patch

from client.backend import start


class StartTests(unittest.TestCase):
    def run_launcher(self, args, processes, ready=True):
        with patch.object(start, 'health', return_value={}), \
             patch.object(start, 'bridge_ready', return_value=ready), \
             patch.object(start, 'wait_ready'), \
             patch.object(start.subprocess, 'Popen', side_effect=processes) as spawn, \
             patch.object(start, 'stop') as stop:
            code = start.main(args)
        return code, spawn, stop

    def test_one_command_starts_api_bridge_and_frontend_and_stops_them(self):
        api, bridge = Mock(), Mock()
        frontend = Mock(returncode=0)
        frontend.poll.return_value = 0
        code, spawn, stop = self.run_launcher([], [api, bridge, frontend], ready=False)
        self.assertEqual(code, 0)
        commands = [call.args[0] for call in spawn.call_args_list]
        self.assertIn('client.backend.server', commands[0])
        self.assertIn('inference.server:app', commands[1])
        self.assertIn('dev', commands[2])
        self.assertEqual([call.args[0] for call in stop.call_args_list], [frontend, bridge, api])
        self.assertEqual(spawn.call_args.kwargs['env']['ARTSY_CLIENT_API_URL'], 'http://127.0.0.1:3001')

    def test_reused_bridge_is_not_stopped(self):
        api, frontend = Mock(), Mock(returncode=0)
        frontend.poll.return_value = 0
        code, spawn, stop = self.run_launcher([], [api, frontend])
        self.assertEqual(code, 0)
        self.assertEqual(spawn.call_count, 2)
        self.assertEqual([call.args[0] for call in stop.call_args_list], [frontend, api])

    def test_browsing_mode_does_not_start_inference(self):
        api, frontend = Mock(), Mock(returncode=0)
        frontend.poll.return_value = 0
        code, spawn, _ = self.run_launcher(['--no-inference'], [api, frontend], ready=False)
        self.assertEqual(code, 0)
        self.assertEqual(spawn.call_count, 2)

    def test_child_failure_stops_remaining_owned_services(self):
        api = Mock()
        api.poll.return_value = 1
        frontend = Mock()
        frontend.poll.return_value = None
        code, _, stop = self.run_launcher(['--no-inference'], [api, frontend])
        self.assertEqual(code, 1)
        self.assertEqual(stop.call_count, 2)
