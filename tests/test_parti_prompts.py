"""Preference reservation preserves the original held-out prompt assignments."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_parti_prompts import digest, normalize, prepare


class PartiPromptsTest(unittest.TestCase):
    def test_preference_reservation_is_disjoint_reproducible_and_sft_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.tsv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["Prompt", "Category", "Challenge", "Note"])
                for i in range(100):
                    writer.writerow([f"Draw scene {i}", f"category-{i % 4}",
                                     f"challenge-{i % 3}", ""])
            with patch("scripts.prepare_parti_prompts.SOURCE_SHA256", digest(source.read_bytes())):
                first = prepare(source, root / "first", 42)
                second = prepare(source, root / "second", 42)
                smaller = prepare(source, root / "smaller", 42, preference_count=7)
                expanded = prepare(source, root / 'expanded', 42, preference_to_evaluation=True)
                expanded_again = prepare(source, root / 'expanded-again', 42, preference_to_evaluation=True)
                for invalid in (0, -1, 70, 71):
                    with self.subTest(count=invalid), self.assertRaises(ValueError):
                        prepare(source, root / "invalid", 42, preference_count=invalid)

            self.assertEqual({name: entry["records"] for name, entry in first["splits"].items()},
                             {"sft": 56, "evaluation": 10, "rl": 20, "preference": 14})
            self.assertEqual(first, second)
            all_ids, all_prompts = set(), set()
            pools = {}
            for name, entry in first["splits"].items():
                data = (root / "first" / entry["file"]).read_bytes()
                self.assertEqual(data, (root / "second" / entry["file"]).read_bytes())
                self.assertEqual(digest(data), entry["sha256"])
                rows = [json.loads(line) for line in data.splitlines()]
                ids = {row["id"] for row in rows}
                prompts = {normalize(row["prompt"]) for row in rows}
                self.assertFalse(all_ids & ids)
                self.assertFalse(all_prompts & prompts)
                all_ids.update(ids)
                all_prompts.update(prompts)
                pools[name] = ids
                if name in ("evaluation", "rl"):
                    self.assertEqual(entry, smaller["splits"][name])
            smaller_pool = set()
            for name in ("sft", "preference"):
                smaller_pool.update(json.loads(line)["id"] for line in
                                    (root / "smaller" / name / "prompts.jsonl").read_text().splitlines())
            self.assertEqual(pools["sft"] | pools["preference"], smaller_pool)
            self.assertEqual(len(all_ids), 100)
            self.assertEqual(len(all_prompts), 100)
            self.assertEqual(expanded, expanded_again)
            self.assertEqual({name: entry['records'] for name, entry in expanded['splits'].items()},
                             {'sft': 56, 'evaluation': 24, 'rl': 20, 'preference': 0})
            for name in ['sft', 'rl']:
                self.assertEqual(expanded['splits'][name], first['splits'][name])
            self.assertEqual((root / 'expanded/evaluation/prompts.jsonl').read_bytes(),
                             (root / 'first/evaluation/prompts.jsonl').read_bytes()
                             + (root / 'first/preference/prompts.jsonl').read_bytes())
            self.assertEqual((root / 'expanded/preference/prompts.jsonl').read_bytes(), b'')
            for name, original in [('original', 'evaluation'), ('added-from-preference', 'preference')]:
                subset = expanded['evaluation_extension']['subsets'][name]
                data = (root / 'expanded' / subset['file']).read_bytes()
                self.assertEqual(data, (root / 'first' / original / 'prompts.jsonl').read_bytes())
                self.assertEqual(digest(data), subset['sha256'])


if __name__ == "__main__":
    unittest.main()
