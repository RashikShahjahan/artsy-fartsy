import asyncio
import base64
from contextlib import redirect_stdout
import io
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Target
from inspect_ai.solver import solver
from PIL import Image
import yaml

from evaluation.harness import ROOT, drawing_score, drawing_task, load_specs, model_from_env, parse_judgment
from evaluation.render import render

GOOD = ('from artcanvas import ArtCanvas\n'
        'with ArtCanvas(1920, 1200) as c:\n'
        ' c.set_color(0, 0, 1)\n c.draw_circle(960, 600, 200, True)\n')


class FileSandbox:
    """Run only the known test sources/renderer in a temporary test workspace."""
    def __init__(self, root):
        self.root = root

    def path(self, path):
        return self.root / path.lstrip("/")

    async def read_file(self, file, text=True):
        return self.path(file).read_text() if text else self.path(file).read_bytes()

    async def write_file(self, file, contents):
        path = self.path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents if isinstance(contents, bytes) else contents.encode())

    async def exec(self, cmd, input=None, **kwargs):
        result = subprocess.run([sys.executable, str(self.path(cmd[1])), *cmd[2:-1],
                                 str(self.path(cmd[-1]))], input=input, text=True,
                                capture_output=True, timeout=kwargs.get("timeout"))
        return SimpleNamespace(success=result.returncode == 0, stdout=result.stdout, stderr=result.stderr)


class EvaluationTest(unittest.TestCase):
    def test_dataset_is_fresh_valid_and_does_not_reuse_warmup_prompts(self):
        specs = load_specs(ROOT / "evaluation/prompts.jsonl")
        self.assertEqual(len(specs), 8)
        self.assertTrue(all("requirements" not in row for row in specs))
        training = ROOT / "datasets/sft-warmup-100/train.jsonl"
        if training.exists():
            prompts = {json.loads(line)["spec"]["prompt"] for line in training.read_text().splitlines()}
            self.assertFalse(prompts.intersection(row["prompt"] for row in specs))
        task = drawing_task(specs, "/tmp/artsy-test-artifacts")
        self.assertEqual(task.turn_limit, 16)
        self.assertEqual(task.token_limit, 32768)
        self.assertEqual(task.config.max_tokens, 2048)

    def test_dataset_rejects_training_rows_duplicate_prompts_and_refusal_tasks(self):
        original = load_specs(ROOT / "evaluation/prompts.jsonl")[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            for rows in ([{**original, "split": "train"}], [{**original, "seed": True}],
                         [{**original, "expectedOutcome": "refusal"}],
                         [original, {**original, "id": "different"}]):
                path.write_text("\n".join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):
                    load_specs(path)

    def test_unknown_model_costs_are_disabled_and_mini_config_loads(self):
        config = yaml.safe_load((ROOT / "evaluation/mini.yaml").read_text())
        self.assertEqual(config["agent"]["step_limit"], 16)
        self.assertEqual(config["model"]["cost_tracking"], "ignore_errors")

    def test_model_uses_the_local_chat_endpoint_without_global_key_mutation(self):
        with patch.dict(os.environ, {"ARTSY_MODEL_URL": "http://localhost:1234/v1", "ARTSY_MODEL_API_KEY": "test-only"}):
            model = model_from_env()
        self.assertEqual(model.name, "Qwen/Qwen3.5-0.8B")
        self.assertEqual(model.api.base_url, "http://localhost:1234/v1")
        self.assertFalse(model.model_args["responses_api"])

    def test_judge_requires_complete_finite_prompt_score(self):
        valid = {"correct": True, "harmful": False, "score": 0.9,
                 "correctness_reason": "Prompt matches", "harmfulness_reason": "Benign drawing"}
        self.assertEqual(parse_judgment(json.dumps(valid))["score"], 0.9)
        for update in ({"score": -0.1}, {"score": True}, {"score": float("nan")},
                       {"correct": "yes"}, {"harmful": None}, {"harmfulness_reason": ""}):
            with self.assertRaises(ValueError):
                parse_judgment(json.dumps({**valid, **update}))

    def test_full_evaluation_requires_a_vision_judge_before_creating_outputs(self):
        from evaluation.evaluate import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.dict(os.environ, {"ARTSY_JUDGE_MODEL": ""}), \
                 patch.object(sys, "argv", ["evaluate", "--output", str(output)]), \
                 redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO), \
                 self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_fresh_render_code_errors_dimensions_symlinks_and_timeouts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            valid = render(GOOD, 42, output, ROOT / "runtime")
            self.assertTrue(valid["ok"], valid)
            with Image.open(output / "image.png") as image:
                self.assertEqual(image.size, (1920, 1200))
            cases = ['raise RuntimeError("repair me")',
                     'from artcanvas import ArtCanvas\nArtCanvas(32,32).save()',
                     "import os; os.symlink('/etc/passwd', 'output.png')",
                     'while True: pass']
            for source in cases:
                with self.subTest(source=source):
                    self.assertFalse(render(source, 42, output, ROOT / "runtime", seconds=1)["ok"])
            self.assertFalse(render("x" * (64 * 1024 + 1), 0, output, ROOT / "runtime")["ok"])

    def test_actual_inspect_scoring_exports_final_artifacts_and_aggregates_metrics(self):
        @solver
        def no_model():
            async def solve(state, generate):
                return state
            return solve

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sbox = FileSandbox(root / "sandbox")
            asyncio.run(sbox.write_file("/workspace/program.py", GOOD))
            sample = Sample(id="good", input="Draw a circle", metadata={"prompt": "Draw a circle", "seed": 42})
            with patch("evaluation.harness.sandbox", return_value=sbox), \
                 patch("inspect_ai._util.appdirs.user_data_path", side_effect=lambda name: root / "data" / name), \
                 patch("inspect_ai._util.appdirs.user_cache_path", side_effect=lambda name: root / "cache" / name):
                logs = eval(Task(dataset=[sample], solver=no_model(),
                                 scorer=drawing_score(str(root / "artifacts"))),
                            model="mockllm/model", log_dir=str(root / "logs"), display="none")
            self.assertEqual(logs[0].status, "success", logs[0].error)
            self.assertEqual(logs[0].results.scores[0].metrics["mean"].value, 1)
            result = json.loads((root / "artifacts/good-1/result.json").read_text())
            self.assertEqual(result["outcome"], "submitted")
            self.assertEqual((root / "artifacts/good-1/program.py").read_text(), GOOD)

            from evaluation.evaluate import main
            with patch("evaluation.evaluate.eval", return_value=logs), \
                 patch("evaluation.evaluate.model_from_env", return_value="mockllm/model"), \
                 patch.object(sys, "argv", ["evaluate", "--execution-only", "--limit", "1", "--output", str(root / "run")]), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertEqual(summary["sample_errors"], 0)
            self.assertEqual(summary["completed_samples"], 1)
            self.assertEqual(summary["status"], "success")

    def test_preview_emits_the_actual_png_as_a_multimodal_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "runtime", root / "runtime")
            shutil.copyfile(ROOT / "evaluation/render.py", root / "render.py")
            result = subprocess.run([sys.executable, str(root / "render.py"), "--seed", "42",
                                     "--output", str(root / "preview"), "--preview"],
                                    input=GOOD, text=True, capture_output=True, check=True)
            config = yaml.safe_load((ROOT / "evaluation/mini.yaml").read_text())
            match = re.search(config["model"]["multimodal_regex"], result.stdout)
            self.assertEqual(match.group(1), "image_url")
            self.assertEqual(base64.b64decode(match.group(2).split(",", 1)[1]),
                             (root / "preview/image.png").read_bytes())

    def test_visual_pass_requires_every_requirement_and_safety(self):
        for judgment, expected in (({"correct": True, "harmful": False, "score": 0.84}, 0),
                                   ({"correct": True, "harmful": True, "score": 1}, 0),
                                   ({"correct": False, "harmful": False, "score": 1}, 0),
                                   ({"correct": True, "harmful": False, "score": 0.85}, 1)):
            with self.subTest(judgment=judgment), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sbox = FileSandbox(root / "sandbox")
                asyncio.run(sbox.write_file("/workspace/program.py", GOOD))
                state = SimpleNamespace(sample_id="visual", epoch=1, model="mockllm/model",
                    metadata={"prompt": "Circle", "seed": 42})
                judgment.update(correctness_reason="Rubric checked", harmfulness_reason="Content checked")
                judge = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(completion=json.dumps(judgment))))
                with patch("evaluation.harness.sandbox", return_value=sbox), \
                     patch("evaluation.harness.get_model", return_value=judge):
                    score = asyncio.run(drawing_score(str(root / "artifacts"), "mockllm/judge")(state, Target("")))
                self.assertEqual(score.value["render_valid"], 1)
                self.assertEqual(score.value["visual_pass"], expected)
                self.assertEqual(score.value["image_harmful"], int(judgment["harmful"]))
                judge.generate.assert_awaited_once()

    def test_refusal_is_recorded_but_does_not_pass_a_drawing_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sbox = FileSandbox(root / "sandbox")
            asyncio.run(sbox.write_file("/workspace/refusal.json", '{"reason":"Cannot fulfill"}'))
            state = SimpleNamespace(sample_id="refusal", epoch=1, model="mockllm/model",
                                    metadata={"prompt": "Draw", "seed": 42})
            with patch("evaluation.harness.sandbox", return_value=sbox):
                score = asyncio.run(drawing_score(str(root / "artifacts"))(state, Target("")))
            self.assertEqual(score.value["render_valid"], 0)
            self.assertEqual(score.metadata["outcome"], "refused")

    def test_drawing_cli_reuses_only_matching_verified_submissions(self):
        from evaluation.draw import main
        from evaluation.harness import sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = io.BytesIO()
            Image.new("RGB", (1920, 1200), "white").save(image, "PNG")
            def completed(task, **kwargs):
                artifact = root / "artifacts/drawing-1"
                artifact.mkdir(parents=True)
                (artifact / "program.py").write_text(GOOD)
                (artifact / "image.png").write_bytes(image.getvalue())
                (artifact / "result.json").write_text(json.dumps({"outcome": "submitted",
                    "sourceHash": sha256(GOOD.encode()), "imageHash": sha256(image.getvalue())}))
                return [SimpleNamespace(status="success", samples=[SimpleNamespace(error=None)])]
            with patch("evaluation.draw.eval", side_effect=completed) as run, \
                 patch("evaluation.draw.model_from_env", return_value="mockllm/model"), \
                 patch.object(sys, "argv", ["draw", "Draw a circle", "--episode", str(root)]), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
                self.assertEqual(main(), 0)
                self.assertEqual(run.call_count, 1)
                (root / "program.py").write_text("changed")
                with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                    main()
                self.assertEqual(run.call_count, 1)

    def test_stale_preview_cannot_turn_invalid_final_code_into_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sbox = FileSandbox(root / "sandbox")
            asyncio.run(sbox.write_file("/workspace/program.py", "invalid python!"))
            image = io.BytesIO()
            Image.new("RGB", (1920, 1200)).save(image, "PNG")
            asyncio.run(sbox.write_file("/workspace/output.png", image.getvalue()))
            state = SimpleNamespace(sample_id="bad", epoch=1, model="mockllm/model",
                                    metadata={"prompt": "Draw", "seed": 42})
            with patch("evaluation.harness.sandbox", return_value=sbox):
                score = asyncio.run(drawing_score(str(root / "artifacts"))(state, Target("")))
            self.assertEqual(score.value, {"render_valid": 0})
            self.assertFalse((root / "artifacts/bad-1/image.png").exists())


if __name__ == "__main__":
    unittest.main()
