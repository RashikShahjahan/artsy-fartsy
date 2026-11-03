"""Kokoro API validation, audio validation and caching without model inference."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from ui import server, speech


def wav():
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000)
        audio.writeframes(b'\0\0' * 240)
    return output.getvalue()


class SpeechTests(unittest.TestCase):
    def tearDown(self):
        speech.CACHE.clear()

    def test_text_is_bounded_and_normalized(self):
        self.assertEqual(speech.validate_text({'text': ' Hi Artsy! '}), 'Hi Artsy!')
        for body in (None, [], {}, {'text': 10}, {'text': ' '}, {'text': 'a' * 501}):
            with self.subTest(body=str(body)[:20]), self.assertRaises(ValueError):
                speech.validate_text(body)

    def test_synthesis_uses_isolated_worker_and_caches_valid_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'; path.touch()
            config = {key: str(path) for key in ('python', 'model', 'voices')}
            config['voice'] = 'af_heart'
            with patch.object(speech, 'settings', return_value=config), \
                 patch.object(speech.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=wav())) as run:
                self.assertEqual(speech.synthesize('Hi'), wav())
                self.assertEqual(speech.synthesize('Hi'), wav())
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0], [str(path), '-m', 'ui.tts_worker'])
                payload = json.loads(run.call_args.kwargs['input'])
                self.assertEqual(payload['voice'], 'af_heart')
                self.assertEqual(payload['text'], 'Hi')
                self.assertEqual(run.call_args.kwargs['timeout'], 60)

    def test_worker_failure_or_invalid_audio_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'; path.touch()
            config = {key: str(path) for key in ('python', 'model', 'voices')}
            config['voice'] = 'af_heart'
            for result in (SimpleNamespace(returncode=1, stdout=wav()),
                           SimpleNamespace(returncode=0, stdout=b'not audio' * 30)):
                with patch.object(speech, 'settings', return_value=config), \
                     patch.object(speech.subprocess, 'run', return_value=result), self.assertRaises(RuntimeError):
                    speech.synthesize('Hi')
                self.assertFalse(speech.CACHE)

    def handler(self, body, origin=None):
        handler = object.__new__(server.Handler)
        raw = json.dumps(body).encode()
        handler.path = '/api/artsy/speech'
        handler.headers = {'Host': f'127.0.0.1:{server.PORT}', 'Content-Length': str(len(raw))}
        if origin: handler.headers['Origin'] = origin
        handler.rfile = io.BytesIO(raw); handler.wfile = io.BytesIO()
        return handler

    def test_endpoint_returns_wav_and_rejects_untrusted_origins(self):
        handler = self.handler({'text': 'Hi'})
        with patch.object(speech, 'synthesize', return_value=wav()) as synthesize, \
             patch.object(handler, 'send_response') as status, \
             patch.object(handler, 'send_header') as header, patch.object(handler, 'end_headers'):
            handler.do_POST()
            status.assert_called_once_with(200)
            self.assertIn(('Content-Type', 'audio/wav'), [call.args for call in header.call_args_list])
            self.assertEqual(handler.wfile.getvalue(), wav())
            synthesize.assert_called_once_with('Hi')
        handler = self.handler({'text': 'Hi'}, 'https://untrusted.example')
        with patch.object(handler, 'send_json') as response, patch.object(speech, 'synthesize') as synthesize:
            handler.do_POST()
            self.assertEqual(response.call_args.args[1], 403)
            synthesize.assert_not_called()

    def test_bad_text_is_400_and_unavailable_runtime_is_503(self):
        handler = self.handler({'text': ''})
        with patch.object(handler, 'send_json') as response:
            handler.do_POST()
            self.assertEqual(response.call_args.args[1], 400)
        handler = self.handler({'text': 'Hi'})
        with patch.object(speech, 'synthesize', side_effect=RuntimeError('Kokoro unavailable')), \
             patch.object(handler, 'send_json') as response:
            handler.do_POST()
            self.assertEqual(response.call_args.args[1], 503)


if __name__ == '__main__':
    unittest.main()
