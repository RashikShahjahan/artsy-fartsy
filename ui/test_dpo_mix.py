"""Prompt-only mix checks; author calls are in-memory fakes."""
import asyncio
from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from drawing.dataset_specs import DIFFICULTIES
from ui.prepare_dpo_mix import assert_novel, choose_reused, normalized, prepare


def unique_text(seed):
    rng = random.Random(seed)
    return ' '.join(''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(10)) for _ in range(20))


def sources():
    return [{'id': f'sft-{i}', 'family': ['circles', 'lines', 'flowers'][i % 3],
             'difficulty': DIFFICULTIES[i % 4], 'prompt': unique_text(i), 'seed': i,
             'benchmarkSlice': 'drawing'} for i in range(80)]


class FakeAuthor:
    model = 'openai/gpt-6-luna'
    baseUrl = 'https://openrouter.ai/api/v1'

    def __init__(self):
        self.calls = 0
        self.requests = []

    async def complete(self, messages, max_tokens):
        self.calls += 1
        self.requests.append(messages)
        slot = json.loads(messages[-1]['content'])['slots'][0]
        return {'text': json.dumps({'items': [{'index': slot['index'], 'prompt': unique_text(1000 + slot['index'])}]}),
                'model': self.model, 'finishReason': 'stop'}


class MixTests(unittest.TestCase):
    def test_normalized_and_near_duplicates_rejected(self):
        old = 'Draw red circles with white borders along the left side of a charcoal canvas.'
        for duplicate in (old, old.upper().replace(' ', '  '), old.replace('red', 'blue')):
            with self.assertRaises(ValueError):
                assert_novel(duplicate, [old])
        assert_novel('Paint an aerial tropical forest at sunset using watercolor washes.', [old])
        self.assertEqual(normalized('DRAW: Circle!'), 'draw circle')

    def test_reuse_balances_difficulty_preserves_specs_and_excludes_evaluation(self):
        rows = sources()
        fresh = [{'difficulty': DIFFICULTIES[i % 4]} for i in range(50)]
        reused = choose_reused(rows, fresh, [rows[0]])
        self.assertEqual(len(reused), 50)
        self.assertNotIn(rows[0], reused)
        self.assertTrue(all(row in rows for row in reused))
        self.assertEqual(Counter(p['difficulty'] for p in fresh + reused), {d: 25 for d in DIFFICULTIES})
        counts = Counter(p['family'] for p in reused)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 2)

    def test_full_mix_has_50_each_and_resumes_without_author_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'source.jsonl'
            rows = sources()
            path.write_text(''.join(json.dumps(p) + '\n' for p in rows))
            frozen = [{**rows[0], 'id': 'eval-renamed'}]
            client = FakeAuthor()
            manifest = asyncio.run(prepare(path, root / 'out', client, frozen=frozen))
            self.assertEqual(client.calls, 50)
            records = [json.loads(line) for line in (root / 'out' / 'prompts.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 100)
            self.assertEqual(Counter(p['source'] for p in manifest['origins'].values()), {'fresh': 50, 'sft': 50})
            self.assertNotIn(rows[0]['prompt'], [p['prompt'] for p in records])
            self.assertEqual(manifest['difficulty_counts'], {d: 25 for d in DIFFICULTIES})
            for p in records:
                if manifest['origins'][p['id']]['source'] == 'sft':
                    self.assertEqual(p, next(row for row in rows if row['id'] == p['id']))
            self.assertEqual(len(list((root / 'out' / 'authoring').glob('author-response-*.json'))), 50)
            first_request = json.loads(client.requests[0][-1]['content'])
            self.assertGreaterEqual(len(first_request['recentPromptsToAvoid']), 8)
            self.assertLessEqual(len(first_request['recentPromptsToAvoid']), 12)
            resumed = FakeAuthor()
            second = asyncio.run(prepare(path, root / 'out', resumed, frozen=frozen))
            self.assertEqual(resumed.calls, 0)
            self.assertEqual(second, manifest)
            (root / 'out' / 'authoring' / 'record-049.json').unlink()
            partial = FakeAuthor()
            third = asyncio.run(prepare(path, root / 'out', partial, frozen=frozen))
            self.assertEqual(partial.calls, 1)
            self.assertEqual(third, manifest)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                asyncio.run(prepare(path, root / 'out', resumed, seed=45, frozen=frozen))


if __name__ == '__main__':
    unittest.main()
