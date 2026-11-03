"""Local Kokoro speech generation, isolated from the drawing runtime."""
from collections import OrderedDict
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import wave

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 500
MAX_AUDIO = 4 * 1024 * 1024
LOCK = threading.Lock()
CACHE = OrderedDict()


def settings():
    from dotenv import dotenv_values
    values = {**dotenv_values(ROOT / '.env'), **os.environ}
    directory = ROOT / '.cache/kokoro'
    return {
        'python': values.get('ARTSY_TTS_PYTHON') or str(ROOT / '.tts-venv/bin/python'),
        'model': values.get('ARTSY_KOKORO_MODEL') or str(directory / 'kokoro-v1.0.int8.onnx'),
        'voices': values.get('ARTSY_KOKORO_VOICES') or str(directory / 'voices-v1.0.bin'),
        'voice': values.get('ARTSY_KOKORO_VOICE') or 'af_heart',
    }


def validate_text(body):
    text = body.get('text') if isinstance(body, dict) else None
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError(f'Enter speech text of 1–{MAX_TEXT} characters.')
    return text.strip()


def synthesize(text):
    text = validate_text({'text': text})
    config = settings()
    key = (text, *config.values())
    if not all(Path(config[name]).is_file() for name in ('python', 'model', 'voices')):
        raise RuntimeError('Kokoro is not installed. Run make tts-setup, then restart the UI.')
    with LOCK:
        if key in CACHE:
            CACHE.move_to_end(key)
            return CACHE[key]
        try:
            result = subprocess.run(
                [config['python'], '-m', 'ui.tts_worker'], cwd=ROOT,
                input=json.dumps({'text': text, **config}).encode(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError('Kokoro could not generate speech. Please try again.') from error
        data = result.stdout
        if result.returncode or not 44 <= len(data) <= MAX_AUDIO:
            raise RuntimeError('Kokoro could not generate speech. Check the local TTS setup.')
        try:
            with wave.open(io.BytesIO(data)) as audio:
                if audio.getnframes() < 1 or audio.getframerate() != 24000 or audio.getnchannels() != 1:
                    raise wave.Error('Invalid Kokoro audio')
        except (wave.Error, EOFError) as error:
            raise RuntimeError('Kokoro returned invalid audio.') from error
        CACHE[key] = data
        while len(CACHE) > 24 or sum(map(len, CACHE.values())) > 16 * 1024 * 1024:
            CACHE.popitem(last=False)
        return data
