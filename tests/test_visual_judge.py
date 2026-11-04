"""Final-image scoring contract, independent of hosted judge services."""

import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from drawing.visual_judge import VISUAL_RUBRIC, parse_visual_judgment, visual_judge_prompt
from evaluation.judge_pairs import RUBRIC
from training.cookbook_env import DockerSession


class VisualJudgeTests(unittest.TestCase):
    def test_eval_and_rl_share_absolute_rubric(self):
        self.assertIn(VISUAL_RUBRIC, RUBRIC)
        self.assertIn(VISUAL_RUBRIC, visual_judge_prompt('Draw a circle'))

    def test_accepts_full_range_and_distinct_components(self):
        for adherence in range(5):
            for quality in range(5):
                answer = dict(adherence_score=adherence, visual_quality_score=quality,
                              comments='Concrete image evidence')
                self.assertEqual(parse_visual_judgment(json.dumps(answer)), answer)

    def test_rejects_invalid_judgments(self):
        valid = dict(adherence_score=4, visual_quality_score=2, comments='Evidence')
        for field in ('adherence_score', 'visual_quality_score'):
            for value in (True, False, -1, 5, 2.5, '3', None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    parse_visual_judgment(json.dumps({**valid, field: value}))
        for value in ('', '  ', None, 1):
            with self.subTest(comments=value), self.assertRaises(ValueError):
                parse_visual_judgment(json.dumps({**valid, 'comments': value}))
        with self.assertRaises(ValueError):
            parse_visual_judgment('{"score":4,"reason":"Old adherence schema"}')


class TruncationTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_valid_json_is_rejected(self):
        client = SimpleNamespace(complete=AsyncMock(return_value={
            'text': '{"adherence_score":4,"visual_quality_score":4,"comments":"Evidence"}',
            'finishReason': 'length'}))
        session = DockerSession({}, Path('/tmp'), {'visual': client})
        with self.assertRaisesRegex(ValueError, 'truncated'):
            await session._judge('visual', 'Grade', parse_visual_judgment, b'png')
