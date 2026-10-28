"""Regression tests for the unfiltered SFT inspection and manual export path."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui.candidate_review import (
    approved_record, export_approved, image_paths, load_candidates, validate_review,
)


class ManualInspectorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "candidates.jsonl"
        self.records = [
            {"id": "a-c0", "spec": {"prompt": "Draw a circle"}, "status": "accepted",
             "reason": "Judge passed", "artifacts": {}},
            {"id": "b-c0", "spec": {"prompt": "Draw a square"}, "status": "review",
             "reason": "Vision evaluation pending", "artifacts": {}},
            {"id": "c-c0", "spec": {"prompt": "Draw a house"}, "status": "rejected",
             "reason": "Score below threshold", "artifacts": {}},
        ]
        self.path.write_text("".join(json.dumps(r) + "\n" for r in self.records), encoding="utf-8")
        self.records, self.sha = load_candidates(self.path)

    def test_inspection_retains_all_automatic_outcomes(self):
        self.assertEqual([r["status"] for r in self.records], ["accepted", "review", "rejected"])
        self.assertEqual(self.sha, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(len({r["id"] for r in self.records}), 3)

    def test_review_validation_and_revision_identity(self):
        review = {"dataset_sha256": self.sha, "reviewer": "Rashik",
                  "decisions": {"a-c0": {"decision": "approved", "notes": "Looks good"},
                                "b-c0": {"decision": "pending", "notes": ""}}}
        validate_review(review, self.sha, self.records)
        with self.assertRaisesRegex(ValueError, "different"):
            validate_review(review, "bad-hash", self.records)
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_review({**review, "decisions": {"unknown": {"decision": "approved"}}},
                            self.sha, self.records)
        with self.assertRaisesRegex(ValueError, "reviewer"):
            validate_review({**review, "reviewer": ""}, self.sha, self.records)

    def test_export_requires_explicit_review_and_valid_sft_conversion(self):
        review = {"dataset_sha256": self.sha, "reviewer": "Rashik",
                  "decisions": {"b-c0": {"decision": "approved", "notes": "Checked preview"},
                                "c-c0": {"decision": "approved", "notes": "Manual override"},
                                "a-c0": {"decision": "rejected", "notes": "Duplicate"}}}
        output = self.root / "accepted.jsonl"
        with patch("ui.candidate_review.convert_record", return_value={"messages": []}) as convert:
            self.assertEqual(export_approved(self.records, review, self.root, output), 2)
        result = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([r["id"] for r in result], ["b-c0", "c-c0"])
        self.assertEqual([r["automaticStatus"] for r in result], ["review", "rejected"])
        self.assertTrue(all(r["status"] == "accepted" and r["humanReviewed"] for r in result))
        self.assertEqual(result[0]["manualReview"], {"reviewer": "Rashik", "notes": "Checked preview"})
        self.assertEqual(convert.call_count, 2)
        self.assertEqual([r["status"] for r in self.records], ["accepted", "review", "rejected"])
        self.assertEqual(len(self.path.read_text().splitlines()), 3)

    def test_invalid_sft_candidate_does_not_partially_replace_export(self):
        review = {"dataset_sha256": self.sha, "reviewer": "Rashik",
                  "decisions": {"a-c0": {"decision": "approved"},
                                "b-c0": {"decision": "approved"}}}
        output = self.root / "accepted.jsonl"
        output.write_text("previous export\n")
        with patch("ui.candidate_review.convert_record", side_effect=[
            {"messages": []}, ValueError("missing final render")
        ]):
            with self.assertRaisesRegex(ValueError, "b-c0.*missing final render"):
                export_approved(self.records, review, self.root, output)
        self.assertEqual(output.read_text(), "previous export\n")
        with self.assertRaisesRegex(ValueError, "unfiltered"):
            export_approved(self.records, review, self.root, self.path)

    def test_only_dataset_local_png_artifacts_are_served(self):
        nested = self.root / "artifacts" / "episode"
        nested.mkdir(parents=True)
        (nested / "preview-1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        record = {
            "artifacts": {"directory": "artifacts/episode", "image": "../outside-inspection.png"},
            "trace": [{"feedback": {"ok": True, "artifact": "/old-machine/preview-1.png"},
                       "observation": {"image": "../../outside-inspection.png"}}],
            "evaluations": [],
        }
        images = image_paths(record, self.root)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0][0], "Attempt 1 preview")
        self.assertEqual(images[0][1], (nested / "preview-1.png").resolve())


if __name__ == "__main__":
    unittest.main()
