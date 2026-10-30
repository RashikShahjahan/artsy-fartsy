"""Run launcher checks. No Tinker, Docker, or remote model requests."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ui import launch, server


class LauncherTests(unittest.TestCase):
    def test_eval_command_has_unique_output_and_rejects_injected_flags(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(launch, "ROOT", Path(folder)):
            prompt = Path(folder) / "prompts.jsonl"
            prompt.write_text("{}\n")
            argv, output = launch.command("eval", {
                "dataset": "prompts.jsonl", "model": "Qwen/Qwen3.6-35B3A",
                "limit": 8, "epochs": 2, "execution_only": True,
            }, "a" * 32)
            self.assertEqual(argv[1:4], ["-u", "-m", "evaluation.evaluate"])
            self.assertIn("--execution-only", argv)
            self.assertIn(str(prompt), argv)
            self.assertEqual(argv[-2:], ["--output", str(output)])
            self.assertEqual(output, Path(folder) / ("eval-results/ui-eval-" + "a" * 32))
            for invalid in [
                {"unsupported": "anything"},
                {"model": "--help"},
                {"limit": -1},
                {"execution_only": "true"},
                {"dataset": "../../etc/passwd"},
            ]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    launch.command("eval", invalid, "a" * 32)

    def test_training_command_rules(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(launch, "ROOT", Path(folder)):
            source = Path(folder) / "accepted.jsonl"
            source.write_text("{}\n")
            checkpoint = Path(folder) / "final.json"
            checkpoint.write_text("{}")
            previous = Path(folder) / "previous"
            previous.mkdir()
            with self.assertRaisesRegex(ValueError, "requires an accepted"):
                launch.command("sft", {}, "a" * 32)
            sft, _ = launch.command("sft", {"dataset": "accepted.jsonl", "batch_size": 4,
                                           "dry_run": True}, "b" * 32)
            self.assertIn("--dataset", sft)
            self.assertIn("--dry-run", sft)
            with self.assertRaisesRegex(ValueError, "Additional steps requires"):
                launch.command("rl", {"additional_steps": 5}, "c" * 32)
            with self.assertRaisesRegex(ValueError, "inherits training settings"):
                launch.command("rl", {"continue_from": "previous", "additional_steps": 5,
                                       "batch_size": 4}, "c" * 32)
            continuing, _ = launch.command("rl", {"continue_from": "previous",
                "dataset": "accepted.jsonl", "additional_steps": 5}, "c" * 32)
            self.assertIn("--continue-from", continuing)
            self.assertIn("--additional-steps", continuing)
            regular, _ = launch.command("rl", {"init_checkpoint": "final.json",
                 "group_size": 4, "max_samples": 8}, "d" * 32)
            self.assertIn("--init-checkpoint", regular)
            self.assertIn("--max-samples", regular)
            with self.assertRaisesRegex(ValueError, "Invalid group_size"):
                launch.command("rl", {"group_size": 1}, "e" * 32)

    def test_process_status_logs_and_training_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(launch, "ROOT", root), patch.object(launch, "RUNS", root / "jobs"):
                output = root / "checkpoints" / "test-run"
                # Both paths are distinct; only one optimizer process can run.
                long_cmd = [sys.executable, "-u", "-c", "import time;print('ready',flush=True);time.sleep(30)"]
                with patch.object(launch, "command", return_value=(long_cmd, output)):
                    first = launch.start("rl", {})
                    self.assertEqual(launch.get_run(first["id"])["state"], "running")
                    with self.assertRaisesRegex(RuntimeError, "already active"):
                        launch.start("sft", {})
                    self.assertEqual(launch.cancel(first["id"])["state"], "stopping")
                    for _ in range(100):
                        if launch.get_run(first["id"])["state"] == "cancelled":
                            break
                        time.sleep(0.02)
                    self.assertEqual(launch.get_run(first["id"])["state"], "cancelled")
                quick = [sys.executable, "-u", "-c", "print('completed locally')"]
                with patch.object(launch, "command", return_value=(quick, output)):
                    second = launch.start("eval", {})
                    for _ in range(100):
                        if launch.get_run(second["id"])["state"] == "completed":
                            break
                        time.sleep(0.02)
                    result = launch.get_run(second["id"])
                    self.assertEqual(result["state"], "completed")
                    self.assertIn("completed locally", result["console"])
                    self.assertEqual(result["exit_code"], 0)
                    self.assertEqual(len(launch.list_runs()), 2)
                self.assertFalse(launch.PROCESSES)

    def test_runs_ui_and_api_require_local_origin(self):
        handler = object.__new__(server.Handler)
        handler.headers = {"Host": f"127.0.0.1:{server.PORT}"}
        handler.path = "/runs"
        with patch.object(handler, "send_page") as send_page:
            handler.do_GET()
            page = send_page.call_args.args[0]
            self.assertIn("Runs", page)
            self.assertIn('href="/runs" aria-current="page"', page)
            self.assertIn('src="/runs.js"', page)
        handler.path = "/api/launches"
        handler.command = "POST"
        payload = json.dumps({"kind": "eval", "settings": {"limit": 3}}).encode()
        handler.headers["Content-Length"] = str(len(payload))
        handler.rfile = BytesIO(payload)
        with patch.object(server.launch, "start", return_value={"id": "a" * 32}) as launch_job, \
             patch.object(handler, "send_json") as response:
            handler.do_POST()
            launch_job.assert_called_once_with("eval", {"limit": 3})
            self.assertEqual(response.call_args.args[1], 202)
        handler.headers["Origin"] = "https://attacker.example"
        handler.rfile = BytesIO(payload)
        with patch.object(handler, "send_json") as response:
            handler.do_POST()
            self.assertEqual(response.call_args.args[1], 403)


if __name__ == "__main__":
    unittest.main()
