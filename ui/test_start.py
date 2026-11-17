"""UI/bridge lifecycle checks without starting hosted model clients."""
import subprocess
import unittest
from unittest.mock import Mock, patch

from ui import start


class StartTests(unittest.TestCase):
    def test_starts_bridge_before_ui_and_forwards_ui_args(self):
        bridge = Mock()
        bridge.poll.return_value = None
        ui = Mock(returncode=0)
        ui.poll.return_value = 0
        with patch.object(start, 'bridge_ready', side_effect=[False, False, True]), \
             patch.object(start.subprocess, 'Popen', side_effect=[bridge, ui]) as spawn, \
             patch.object(start.time, 'sleep'), patch.object(start, 'stop') as stop:
            self.assertEqual(start.main(['--dataset', 'prompts.jsonl']), 0)
        self.assertIn('inference.server:app', spawn.call_args_list[0].args[0])
        self.assertEqual(spawn.call_args_list[1].args[0][-2:],
                         ['--dataset', 'prompts.jsonl'])
        self.assertEqual([call.args[0] for call in stop.call_args_list], [ui, bridge])

    def test_existing_bridge_is_reused_and_not_stopped(self):
        ui = Mock(returncode=7)
        ui.poll.return_value = 7
        with patch.object(start, 'bridge_ready', return_value=True), \
             patch.object(start.subprocess, 'Popen', return_value=ui) as spawn, \
             patch.object(start, 'stop') as stop:
            self.assertEqual(start.main([]), 7)
        self.assertEqual(spawn.call_count, 1)
        self.assertIn('ui.server', spawn.call_args.args[0])
        stop.assert_called_once_with(ui)

    def test_bridge_failure_prevents_ui_start(self):
        bridge = Mock()
        bridge.poll.return_value = 1
        with patch.object(start, 'bridge_ready', return_value=False), \
             patch.object(start.subprocess, 'Popen', return_value=bridge) as spawn, \
             patch.object(start, 'stop') as stop:
            self.assertEqual(start.main([]), 1)
        self.assertEqual(spawn.call_count, 1)
        stop.assert_called_once_with(bridge)

    def test_interrupt_stops_both_owned_services(self):
        bridge, ui = Mock(), Mock()
        bridge.poll.return_value = ui.poll.return_value = None
        with patch.object(start, 'bridge_ready', side_effect=[False, True]), \
             patch.object(start.subprocess, 'Popen', side_effect=[bridge, ui]), \
             patch.object(start.time, 'sleep', side_effect=KeyboardInterrupt), \
             patch.object(start, 'stop') as stop:
            self.assertEqual(start.main([]), 130)
        self.assertEqual([call.args[0] for call in stop.call_args_list], [ui, bridge])

    def test_unresponsive_process_is_killed(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('fixture', 10), 0]
        start.stop(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
