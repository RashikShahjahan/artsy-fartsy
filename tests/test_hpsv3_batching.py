import asyncio
import base64
import io
from pathlib import Path
import threading
import unittest
from PIL import Image
from training.hpsv3_server import MicroBatcher, ScoreRequest, app, score


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score_batch(self, prompts, images):
        self.calls.append((prompts, images))
        if prompts[0] == 'fail':
            raise ValueError('failed batch')
        return [float(image) for image in images]


class BatchingTests(unittest.TestCase):
    def test_coalesces_in_order_and_bounds_batches(self):
        scorer = FakeScorer()
        batcher = MicroBatcher(scorer, 'mlx', max_batch=4, wait_ms=50)
        try:
            futures = [batcher.submit(str(i), str(i)) for i in range(9)]
            self.assertEqual([future.result(timeout=5) for future in futures], list(range(9)))
            self.assertEqual([len(call[0]) for call in scorer.calls], [4, 4, 1])
            self.assertEqual(scorer.calls[0], ([str(i) for i in range(4)], [str(i) for i in range(4)]))
        finally:
            batcher.close()
        with self.assertRaises(RuntimeError):
            batcher.submit('closed', '0')

    def test_failure_propagates_and_worker_recovers(self):
        batcher = MicroBatcher(FakeScorer(), 'mlx', max_batch=2, wait_ms=50)
        try:
            futures = [batcher.submit('fail', '1'), batcher.submit('other', '2')]
            for future in futures:
                with self.assertRaisesRegex(ValueError, 'failed batch'):
                    future.result(timeout=5)
            self.assertEqual(batcher.submit('ok', '3').result(timeout=5), 3)
        finally:
            batcher.close()


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_request_keeps_image_until_worker_finishes(self):
        started, release = threading.Event(), threading.Event()
        paths = []
        class Scorer:
            def score_batch(self, prompts, images):
                paths.extend(images)
                started.set()
                release.wait(5)
                if not Path(images[0]).exists():
                    raise AssertionError('Image removed during inference')
                return [1.0]
        app.state.model_id = 'test'
        app.state.batcher = MicroBatcher(Scorer(), 'mlx', max_batch=1)
        png = io.BytesIO()
        Image.new('RGB', (1920, 1200)).save(png, format='PNG')
        task = asyncio.create_task(score(ScoreRequest(prompt='p', image=base64.b64encode(png.getvalue()).decode(), model='test'), authorization=None))
        try:
            await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(Path(paths[0]).exists())
        finally:
            release.set()
            await asyncio.to_thread(app.state.batcher.close)
        self.assertFalse(Path(paths[0]).exists())
