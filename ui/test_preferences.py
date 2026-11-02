import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from ui.preferences import Review, RUNS


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root/'evaluation/generated-50-2026-10-28/prompts.jsonl'
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({'id': 'test', 'prompt': '<script>Draw</script>', 'seed': 4})+'\n')
        for run in RUNS.values():
            for repeat in range(1, 4):
                folder = self.root/'eval-results'/run/'artifacts'/f'test-{repeat}'
                folder.mkdir(parents=True)
                png = (run+str(repeat)).encode()
                (folder/'image.png').write_bytes(png)
                (folder/'result.json').write_text(json.dumps({'prompt': '<script>Draw</script>',
                    'seed': 4, 'imageHash': hashlib.sha256(png).hexdigest(),
                    'scores': {'render_valid': 1, 'code_malicious': 0, 'image_harmful': 0}}))
        self.review = Review(self.root)

    def test_blind_packet_stable_images_and_persistence(self):
        packet = self.review.packet('Alice')
        text = json.dumps(packet)
        for model in RUNS:
            self.assertNotIn(model, text)
        ident = packet['cases'][0]['id']
        self.assertEqual(self.review.image('Alice', ident, 0), Review(self.root).image('Alice', ident, 0))
        self.review.save('Alice', ident, [1, 2, 3])
        self.assertEqual(Review(self.root).packet('Alice')['cases'][0]['ranks'], [1, 2, 3])
        self.assertIsNone(self.review.packet('Bob')['cases'][0]['ranks'])
        with self.assertRaises(ValueError):
            self.review.image('Alice', '../../etc/passwd', 0)

    def test_pairwise_scores_ties_and_sealing(self):
        self.review.packet('Alice')
        state = self.review.state()
        for ident in self.review.cases:
            order = self.review.order(state, 'Alice', ident)
            self.review.save('Alice', ident, [{'New GRPO': 1, 'Parent SFT': 2, 'Base': 3}[m] for m in order])
        scores = {r['model']: r['preference_score'] for r in self.review.results('Alice')['models']}
        self.assertEqual(scores, {'Base': 0, 'Parent SFT': 50, 'New GRPO': 100})
        with self.assertRaises(ValueError):
            self.review.save('Alice', next(iter(self.review.cases)), [1, 1, 1])
        for ident in self.review.cases:
            self.review.save('Bob', ident, [1, 1, 1])
        self.assertTrue(all(r['preference_score'] == 50 for r in self.review.results('Bob')['models']))

    def test_validation_incomplete_and_missing_images(self):
        ident = next(iter(self.review.cases))
        for ranks in [[0, 1, 2], [2, 2, 3], [True, 2, 3], [1, 2], '123']:
            with self.assertRaises(ValueError):
                self.review.save('Alice', ident, ranks)
        with self.assertRaises(ValueError):
            self.review.results('Alice')
        image = next(iter(self.review.cases[ident]['images'].values()))
        image.write_bytes(b'changed')
        review = Review(self.root)
        self.assertEqual(len(review.cases), 1)
        self.assertEqual(review.excluded, 0)
        self.assertNotEqual(next(iter(review.cases)), ident)
        for run in RUNS.values():
            for image in (self.root/"eval-results"/run).glob("artifacts/*/image.png"):
                image.unlink()
        review = Review(self.root)
        self.assertEqual(len(review.cases), 1)
        self.assertEqual(review.packet("Alice")["cases"][0]["available"], [False]*3)
