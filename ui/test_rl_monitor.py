"""Regression tests for the passive RL monitor (no Tinker or Docker needed)."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import rl_monitor


class RLMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "grpo-v4"
        self.run.mkdir()
        (self.run / "training-manifest.json").write_text(json.dumps({
            "config": {"algorithm": "tinker-cookbook-rlhf-v4", "model": "Qwen/test",
                       "group_size": 4, "batch_size": 2},
            "total_steps": 5,
        }))

    def sample(self, step=1, prompt="prompt-1", name="a" * 32, reward=0.75,
               finished=True):
        directory = self.run / "rollouts" / f"step-{step:06d}" / prompt / name
        directory.mkdir(parents=True, exist_ok=True)
        if finished:
            (directory / "result.json").write_text(json.dumps({
                "prompt": "Draw a blue circle",
                "outcome": "submitted", "commands": ["echo ok", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
                "previewAttempts": 1, "scores": {
                    "reward": reward, "adherence": 0.75, "render_valid": 1.0,
                },
                "unsafe": False,
            }))
            (directory / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"test")
            (directory / "program.py").write_text("print('circle')\n")
        return directory

    def test_run_discovery_and_single_run_root(self):
        (self.root / "sft-only").mkdir()
        self.assertEqual([row["id"] for row in rl_monitor.runs(self.root)], ["grpo-v4"])
        self.assertEqual(rl_monitor.runs(self.root)[0]["model"], "Qwen/test")
        self.assertFalse(rl_monitor.runs(self.root)[0]["finished"])
        self.assertEqual([row["id"] for row in rl_monitor.runs(self.run)], ["grpo-v4"])
        (self.run / "final.json").write_text("{}")
        self.assertTrue(rl_monitor.runs(self.root)[0]["finished"])

    def test_reads_completed_episodes_without_touching_active_ones(self):
        finished = self.sample()
        unfinished = self.sample(step=2, name="b" * 32, finished=False)
        before = sorted(str(f) for f in self.root.rglob("*"))
        result = rl_monitor.episodes(self.root, "grpo-v4")
        after = sorted(str(f) for f in self.root.rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual(len(result["episodes"]), 1)
        row = result["episodes"][0]
        self.assertEqual(row["score"], 0.75)
        self.assertTrue(row["renderValid"])
        self.assertTrue(row["hasImage"])
        self.assertEqual(row["bashCalls"], 2)
        self.assertEqual(row["step"], "step-000001")
        self.assertTrue(unfinished.is_dir())
        self.assertTrue(finished.is_dir())

    def test_bounded_to_recent_three_steps(self):
        for step in range(1, 6):
            self.sample(step=step, name=f"{step:032x}")
        data = rl_monitor.episodes(self.root, "grpo-v4")
        self.assertEqual(data["steps"], ["step-000005", "step-000004", "step-000003"])
        self.assertEqual([e["step"] for e in data["episodes"]], data["steps"])

    def test_rejects_non_rl_manifests_and_malformed_results(self):
        (self.run / "training-manifest.json").write_text(json.dumps({"config": {"algorithm": "sft"}}))
        self.assertEqual(rl_monitor.runs(self.root), [])
        with self.assertRaises(ValueError):
            rl_monitor.episodes(self.root, "grpo-v4")
        (self.run / "training-manifest.json").write_text(json.dumps({
            "config": {"algorithm": "tinker-cookbook-rlhf-v4"}}))
        directory = self.sample()
        (directory / "result.json").write_text("{incomplete")
        self.assertEqual(rl_monitor.episodes(self.root, "grpo-v4")["episodes"], [])

    def test_artifact_whitelist_and_path_confinement(self):
        self.sample()
        file = rl_monitor.artifact(self.root, "grpo-v4", "step-000001",
                                   "prompt-1", "a" * 32, "image.png")
        self.assertEqual(file.name, "image.png")
        for parts in [
            ("..", "step-000001", "prompt-1", "a" * 32, "image.png"),
            ("grpo-v4", "../step-000001", "prompt-1", "a" * 32, "image.png"),
            ("grpo-v4", "step-000001", "..", "a" * 32, "image.png"),
            ("grpo-v4", "step-000001", "prompt-1", "a" * 32, "secret"),
        ]:
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                rl_monitor.artifact(self.root, *parts)
        target = self.run / "rollouts" / "step-000001" / "prompt-1" / ("c" * 32)
        target.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            rl_monitor.artifact(self.root, "grpo-v4", "step-000001", "prompt-1",
                                "c" * 32, "image.png")

    def prompt_dataset(self):
        """A full dataset whose manifest selects only two of its three prompts."""
        file = self.root / "rl-input.jsonl"
        rows = [
            {"id": "unused", "prompt": "Draw an unused shape", "benchmarkSlice": "drawing"},
            {"id": "red", "prompt": "Draw a red square", "difficulty": "hard",
             "family": "grids", "skills": ["composition"], "benchmarkSlice": "drawing"},
            {"id": "unsafe", "prompt": "An unsafe request requiring a refusal",
             "kind": "refusal", "benchmarkSlice": "malicious_code", "difficulty": "easy"},
        ]
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        file.write_bytes(raw)
        manifest = json.loads((self.run / "training-manifest.json").read_text())
        manifest["config"].update(dataset_sha256=hashlib.sha256(raw).hexdigest(),
                                  prompt_ids=["unsafe", "red"])
        (self.run / "training-manifest.json").write_text(json.dumps(manifest))
        return file

    def test_all_run_prompts_are_manifest_verified_and_keep_selected_order(self):
        dataset = self.prompt_dataset()
        before = sorted(str(p) for p in self.root.rglob("*"))
        result = rl_monitor.prompts(self.root, "grpo-v4", dataset)
        self.assertEqual(sorted(str(p) for p in self.root.rglob("*")), before)
        self.assertTrue(result["available"])
        self.assertEqual(result["total"], 2)
        self.assertEqual([p["id"] for p in result["prompts"]], ["unsafe", "red"])
        self.assertEqual([p["index"] for p in result["prompts"]], [1, 2])
        self.assertEqual(result["prompts"][0]["category"], "malicious_code")
        self.assertEqual(result["prompts"][1]["skills"], ["composition"])
        self.assertEqual(result["prompts"][1]["prompt"], "Draw a red square")

    def test_auto_discovery_and_missing_or_changed_dataset(self):
        dataset = self.prompt_dataset()
        folder = self.root / "project" / "datasets" / "custom-rl"
        folder.mkdir(parents=True)
        auto = folder / "prompts.jsonl"
        auto.write_bytes(dataset.read_bytes())
        with patch.object(rl_monitor, "PROJECT_ROOT", self.root / "project"):
            self.assertTrue(rl_monitor.prompts(self.root, "grpo-v4")["available"])
            auto.write_text(auto.read_text() + " ")
            result = rl_monitor.prompts(self.root, "grpo-v4")
        self.assertFalse(result["available"])
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["prompts"], [])
        self.assertIn("--rl-dataset", result["message"])
        dataset.write_text(dataset.read_text() + " ")
        self.assertFalse(rl_monitor.prompts(self.root, "grpo-v4", dataset)["available"])

    def test_missing_prompt_manifest_is_not_guessed_from_rollouts(self):
        self.sample()
        data = rl_monitor.prompts(self.root, "grpo-v4")
        self.assertFalse(data["available"])
        self.assertEqual(data["prompts"], [])
        with self.assertRaises(ValueError):
            rl_monitor.prompts(self.root, "../outside")

    def test_http_routes_do_not_start_training(self):
        from ui import server
        self.sample()
        dataset = self.prompt_dataset()
        handler = object.__new__(server.Handler)
        handler.headers = {"Host": f"127.0.0.1:{server.PORT}"}
        with patch.object(server, "RL_ROOT", self.root), \
             patch.object(server, "RL_DATASET", dataset), \
             patch.object(handler, "send_json") as send, \
             patch.object(handler, "send_response"), \
             patch.object(handler, "send_header"), \
             patch.object(handler, "end_headers"):
            handler.path = "/api/rl/runs"
            handler.do_GET()
            self.assertEqual(send.call_args.args[0]["runs"][0]["id"], "grpo-v4")
            handler.path = "/api/rl/runs/grpo-v4"
            handler.do_GET()
            self.assertEqual(len(send.call_args.args[0]["episodes"]), 1)
            handler.path = "/api/rl/runs/grpo-v4/prompts"
            handler.do_GET()
            self.assertEqual([p["id"] for p in send.call_args.args[0]["prompts"]],
                             ["unsafe", "red"])
            handler.path = "/api/rl/runs/other/prompts"
            handler.do_GET()
            self.assertEqual(send.call_args.args[1], 404)
            handler.path = "/api/rl/runs/other"
            handler.do_GET()
            self.assertEqual(send.call_args.args[1], 404)
            handler.path = "/api/rl/artifacts/grpo-v4/step-000001/prompt-1/" + "a" * 32 + "/image.png"
            handler.wfile = BytesIO()
            handler.do_GET()
            self.assertTrue(handler.wfile.getvalue().startswith(b"\x89PNG"))
            handler.path = "/api/rl/artifacts/grpo-v4/step-000001/prompt-1/" + "a" * 32 + "/config.json"
            handler.do_GET()
            self.assertEqual(send.call_args.args[1], 404)


if __name__ == "__main__":
    unittest.main()
