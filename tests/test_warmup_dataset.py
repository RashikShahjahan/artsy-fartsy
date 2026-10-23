"""Parity and selection checks for the Python warmup pipeline."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from drawing.dataset_prompts import compact_json, sha256
from drawing.repair_prompts import generate_repair_prompts
from drawing.warmup_dataset import is_execution_repair, is_natural_repair, select_warmup_records
import drawing.warmup_dataset as warmup


class RepairCurriculumTests(unittest.TestCase):
    def test_curriculum_matches_typescript_golden_and_is_deterministic(self) -> None:
        specs = generate_repair_prompts()
        self.assertEqual(len(specs), 128)
        self.assertEqual(specs, generate_repair_prompts())
        self.assertNotEqual(specs, generate_repair_prompts(43))
        self.assertEqual(hashlib.sha256(compact_json(specs).encode()).hexdigest(),
                         "ba06561bc926630bf6dd01c3309e7e35980581dd06689100a664c0076110ca10")
        self.assertEqual(len({spec["prompt"] for spec in specs}), 128)
        for spec in specs:
            self.assertEqual(spec["split"], "train")
            self.assertNotIn("requirements", spec)
            self.assertGreaterEqual(len(spec["prompt"].split(". ")), 13)
            self.assertEqual(spec["repairTask"]["complexity"], "advanced")
            self.assertGreaterEqual(len(spec["repairTask"]["skills"]), 3)
            self.assertNotIn("defects", spec["repairTask"])
            self.assertNotRegex(spec["prompt"], r"omit import|undefined local|unsupported keyword")

    def test_pool_bounds_and_safe_integer_validation(self) -> None:
        for seed, count in ((-1, 16), (True, 16), (1, 15), (1, 513)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                generate_repair_prompts(seed, count)


class WarmupSelectionTests(unittest.TestCase):
    @staticmethod
    def record(index: int, repair: bool = False) -> dict:
        code = f"# drawing {index}"
        image = f"png:{index}".encode()
        ident = f"example-{index}"
        trace = []
        if repair:
            trace.append({"attempt": 0, "completion": {"text": '{"code":"broken"}'},
                          "feedback": {"ok": False, "failureKind": "invalid_code"}})
        trace.append({"attempt": int(repair), "completion": {"text": compact_json({"code": code})},
                      "feedback": {"ok": True}})
        return {"version": "artsy-dataset-v1", "id": ident, "status": "review", "reason": "Needs visual review",
                "spec": {"id": ident, "family": "circles", "split": "train", "prompt": f"Drawing {index}",
                         "expectedOutcome": "drawing", "seed": index},
                "code": code, "sourceHash": sha256(code), "imageHash": sha256(image), "pixelHash": f"pixels-{index}",
                "artifacts": {"directory": f"artifacts/{ident}", "source": f"artifacts/{ident}/program.py",
                              "image": f"artifacts/{ident}/image.png", "trajectory": f"artifacts/{ident}/trajectory.jsonl"},
                "trace": trace, "evaluations": [], "teacher": {"model": "scripted", "baseUrl": "http://localhost"},
                "humanReviewed": False}

    def test_keeps_80_one_shots_and_four_repairs_with_duplicate_exceptions(self) -> None:
        records = [self.record(i) for i in range(95)]
        repairs = [self.record(100 + i, True) for i in range(4)]
        for record in repairs[:3]:
            record["status"] = "rejected"
            record["reason"] = "Duplicate source or rendered pixels of another example"
        selected = select_warmup_records(records + repairs)
        self.assertEqual(len(selected), 84)
        self.assertEqual(sum(row["kind"] == "one-shot" for row in selected), 80)
        self.assertEqual([row["record"]["id"] for row in selected if row["kind"] == "repair"],
                         [row["id"] for row in repairs])
        with self.assertRaisesRegex(ValueError, "Need 80"):
            select_warmup_records(records[:79] + repairs)
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            select_warmup_records(records)

    def test_repair_classifiers_exclude_malformed_and_synthetic_repairs(self) -> None:
        row = self.record(200, True)
        self.assertTrue(is_execution_repair(row))
        self.assertTrue(is_natural_repair(row))
        row["trace"][0]["feedback"].pop("failureKind")
        self.assertFalse(is_execution_repair(row))
        row["trace"][0]["feedback"]["failureKind"] = "invalid_code"
        row["generation"] = {"initialDraft": "teacher-seeded-bugs"}
        self.assertFalse(is_execution_repair(row))
        row["generation"] = {"initialDraft": "natural"}
        row["selectionAudit"] = {"eligible": False}
        self.assertFalse(is_natural_repair(row))


class WarmupAssemblyTests(unittest.IsolatedAsyncioTestCase):
    def _source(self, root: Path) -> tuple[Path, list[dict]]:
        source = root / "pilot"
        rows = ([WarmupSelectionTests.record(i) for i in range(80)]
                + [WarmupSelectionTests.record(100 + i, True) for i in range(4)])
        rows[1]["spec"]["split"] = "test"
        for row in rows[80:83]:
            row["status"] = "rejected"
            row["reason"] = "Duplicate source or rendered pixels of another example"
        for row in rows:
            episode = source / row["artifacts"]["directory"]
            episode.mkdir(parents=True, exist_ok=True)
            code = row["code"]
            image = f"png:{row['spec']['seed']}".encode()
            (episode / "program.py").write_text(code, encoding="utf-8")
            (episode / "image.png").write_bytes(image)
            (episode / "trajectory.jsonl").write_text("{}\n", encoding="utf-8")
        (source / "candidates.jsonl").write_text("".join(compact_json(row) + "\n" for row in rows), encoding="utf-8")
        return source, rows

    @staticmethod
    def _generated_record(spec: dict, index: int) -> dict:
        code = f"# repaired drawing {index}"
        image = f"repaired-png:{index}".encode()
        ident = f"generated-{index}"
        return {"version": "artsy-dataset-v1", "id": ident, "candidate": 0, "status": "review",
                "reason": "Needs visual review", "spec": spec, "code": code, "sourceHash": sha256(code),
                "imageHash": sha256(image), "pixelHash": f"generated-pixels-{index}",
                "artifacts": {"directory": f"artifacts/{ident}", "source": f"artifacts/{ident}/program.py",
                    "image": f"artifacts/{ident}/image.png", "trajectory": f"artifacts/{ident}/trajectory.jsonl"},
                "trace": [{"attempt": 0, "completion": {"text": '{"code":"broken"}'},
                           "feedback": {"ok": False, "failureKind": "invalid_code"}},
                          {"attempt": 1, "completion": {"text": compact_json({"code": code})},
                           "feedback": {"ok": True}}],
                "evaluations": [], "teacher": {"model": "scripted", "baseUrl": "http://localhost"},
                "runtimeVersion": "artcanvas-v2", "apiDocHash": "api", "humanReviewed": False,
                "generation": {"initialDraft": "natural"}}

    async def test_assembly_is_portable_and_preserves_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="artsy-warmup-") as tmp:
            root = Path(tmp)
            source, originals = self._source(root)
            output = root / "warmup"
            generated_specs = generate_repair_prompts()[:16]
            generated_records = [self._generated_record(spec, i) for i, spec in enumerate(generated_specs)]
            calls = 0

            async def fake_generate(options: dict) -> dict:
                nonlocal calls
                calls += 1
                candidates = Path(options["output"]) / "candidates.jsonl"
                candidates.parent.mkdir(parents=True, exist_ok=True)
                candidates.write_text("".join(compact_json(row) + "\n" for row in generated_records), encoding="utf-8")
                for row in generated_records:
                    episode = Path(options["output"]) / row["artifacts"]["directory"]
                    episode.mkdir(parents=True, exist_ok=True)
                    (episode / "program.py").write_text(row["code"], encoding="utf-8")
                    (episode / "image.png").write_bytes(f"repaired-png:{row['candidate'] if 'candidate' in row else int(row['id'].split('-')[-1])}".encode())
                    (episode / "trajectory.jsonl").write_text("{}\n", encoding="utf-8")
                return {"review": len(generated_records), "infrastructureErrors": 0}

            options = {"source": str(source), "output": str(output), "seed": 42, "poolSize": 128,
                       "teacher": type("Teacher", (), {"model": "scripted", "baseUrl": "http://localhost"})(),
                       "candidates": 2, "workers": 4}
            original_json = (source / "candidates.jsonl").read_bytes()
            with patch.object(warmup, "generate_dataset", fake_generate):
                summary = await warmup.generate_warmup_dataset(options)
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["total"], 100)
                self.assertEqual(summary["oneShot"], 80)
                self.assertEqual(summary["repairs"], 20)
                self.assertEqual(calls, 1)
                all_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
                self.assertEqual(sum(row["warmup"]["duplicateException"] for row in all_rows), 3)
                self.assertTrue(all(row["warmup"]["split"] == "train" for row in all_rows))
                self.assertTrue(all("requirements" not in row["spec"] for row in all_rows))
                self.assertEqual(next(row for row in all_rows if row["id"] == originals[1]["id"])["warmup"]["sourceSplit"], "test")
                self.assertEqual((source / "candidates.jsonl").read_bytes(), original_json)
                self.assertEqual((await warmup.generate_warmup_dataset(options)), summary)
                self.assertEqual(calls, 2)
                manifest = json.loads((output / "warmup-manifest.json").read_text())
                self.assertIn("no longer held out", manifest["splitPolicy"])

    async def test_incomplete_run_never_exports_train_jsonl(self) -> None:
        with tempfile.TemporaryDirectory(prefix="artsy-warmup-") as tmp:
            root = Path(tmp)
            source, _ = self._source(root)
            output = root / "warmup"

            async def fake_generate(options: dict) -> dict:
                candidates = Path(options["output"]) / "candidates.jsonl"
                candidates.parent.mkdir(parents=True, exist_ok=True)
                candidates.write_text("", encoding="utf-8")
                return {"review": 0, "infrastructureErrors": 0}

            options = {"source": str(source), "output": str(output),
                       "teacher": type("Teacher", (), {"model": "scripted", "baseUrl": "http://localhost"})(),
                       "candidates": 1}
            with patch.object(warmup, "generate_dataset", fake_generate):
                summary = await warmup.generate_warmup_dataset(options)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["total"], 84)
            self.assertEqual(summary["remainingRepairs"], 16)
            self.assertFalse((output / "train.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
