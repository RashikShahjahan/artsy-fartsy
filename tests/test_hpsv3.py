"""Exercise the scoring wire contract, failure behavior, and reward calibration."""
import base64
from io import BytesIO
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

from training.hpsv3 import HPSv3Client, normalized_reward


class CalibrationTests(unittest.TestCase):
    def test_monotonic_bounded_linear_and_clipped(self):
        rewards = [normalized_reward(v) for v in (-1e300, -10, 0, 10, 1e300)]
        self.assertEqual(rewards, sorted(rewards))
        self.assertTrue(all(0.02 <= value <= 1 for value in rewards))
        for raw, expected in ((-11, .02), (-10, .09), (-9, .16), (-5, .44),
                              (0, .79), (2, .93), (3, 1)):
            with self.subTest(raw=raw):
                self.assertAlmostEqual(normalized_reward(raw), expected)
        self.assertEqual(normalized_reward(-1e300), .02)
        self.assertEqual(normalized_reward(1e300), 1)
        self.assertAlmostEqual(normalized_reward(5, lower=0, upper=10, minimum=.1), .55)

    def test_bad_scores_and_calibrations_rejected(self):
        for value in (float('nan'), float('inf'), True, '10', None):
            with self.assertRaises(ValueError):
                normalized_reward(value)
        for lower, upper, minimum in ((math.nan, 3, .02), (-11, math.inf, .02),
                (3, 3, .02), (4, 3, .02), (-11, 3, 0), (-11, 3, 1),
                (-11, 3, math.nan), (False, 3, .02), (-1e308, 1e308, .02)):
            with self.assertRaises(ValueError):
                HPSv3Client('http://scorer', lower=lower, upper=upper, minimum=minimum)


class TimeoutTests(unittest.TestCase):
    def test_queue_aware_default_and_environment_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(HPSv3Client('http://scorer').timeout, 1800)
        with patch.dict(os.environ, {'ARTSY_HPSV3_TIMEOUT_SECONDS': '2400.5'}):
            self.assertEqual(HPSv3Client('http://scorer').timeout, 2400.5)
            self.assertEqual(HPSv3Client('http://scorer', timeout=42).timeout, 42)
        with patch.dict(os.environ, {'ARTSY_HPSV3_TIMEOUT_SECONDS': 'invalid'}):
            self.assertEqual(HPSv3Client('http://scorer', timeout=42).timeout, 42)

    def test_rejects_invalid_timeouts(self):
        for timeout in (0, -1, math.nan, math.inf, True, 'invalid'):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                HPSv3Client('http://scorer', timeout=timeout)
        for timeout in ('0', '-1', 'nan', 'inf', 'invalid'):
            with self.subTest(environment=timeout), patch.dict(os.environ, {'ARTSY_HPSV3_TIMEOUT_SECONDS': timeout}):
                with self.assertRaises(ValueError):
                    HPSv3Client('http://scorer')


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_and_image_sent_and_raw_score_retained(self):
        def handler(request):
            import json
            body = json.loads(request.content)
            self.assertEqual(body, {'prompt': 'Draw a fox', 'image': base64.b64encode(b'png').decode(),
                                    'model': 'pinned-weights'})
            self.assertEqual(request.headers['Authorization'], 'Bearer token')
            return httpx.Response(200, json={'model': 'pinned-weights', 'score': 10.0})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch('training.hpsv3.httpx.AsyncClient', return_value=client):
            judgment = await HPSv3Client('http://scorer/', 'pinned-weights', api_key='token').score('Draw a fox', b'png')
        self.assertEqual(judgment['hpsv3_score'], 10)
        self.assertEqual(judgment['preference'], normalized_reward(10))
        self.assertEqual(judgment['reward_version'], 'artsy-final-image-hpsv3-linear-v2')
        self.assertEqual((judgment['lower'], judgment['upper'], judgment['minimum']), (-11, 3, .02))

    async def test_bad_service_responses_propagate(self):
        for status, body in ((500, {}), (200, {'model': 'wrong', 'score': 1}),
                             (200, {'model': 'weights', 'score': 'NaN'}),
                             (200, {'model': 'weights'})):
            client = httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(status, json=body)))
            with patch('training.hpsv3.httpx.AsyncClient', return_value=client):
                with self.assertRaises((httpx.HTTPError, ValueError, KeyError)):
                    await HPSv3Client('http://scorer', 'weights').score('Draw', b'png')

    async def test_inference_failure_does_not_write_a_zero_reward(self):
        from training.cookbook_env import DockerSession
        image = BytesIO()
        Image.new('RGB', (1920, 1200)).save(image, 'PNG')
        import json
        response = SimpleNamespace(returncode=0, stdout=json.dumps({
            'feedback': {'ok': True}, 'image': base64.b64encode(image.getvalue()).decode()}).encode())
        with tempfile.TemporaryDirectory() as root:
            scorer = SimpleNamespace(score=AsyncMock(side_effect=RuntimeError('GPU failed')))
            session = DockerSession({'prompt': 'Draw', 'seed': 1}, Path(root), {'hpsv3': scorer})
            session.submitted = True
            with patch.object(session, '_read_submission', AsyncMock(return_value=('draw()', None))), \
                 patch('training.cookbook_env._docker', AsyncMock(return_value=response)):
                with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
                    await session.grade([])
            self.assertIsNone(session.result)
            self.assertFalse((Path(root) / 'result.json').exists())
