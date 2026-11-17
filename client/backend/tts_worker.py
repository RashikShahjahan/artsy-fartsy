"""Synthesize one utterance in the isolated Kokoro environment; stdout is WAV only."""
import contextlib
import io
import json
import sys
import wave


def main():
    request = json.loads(sys.stdin.buffer.read(8192))
    # Keep library logs out of the binary response.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from kokoro_onnx import Kokoro
        engine = Kokoro(request['model'], request['voices'])
        samples, rate = engine.create(request['text'], voice=request['voice'], speed=0.96, lang='en-us')
    pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(pcm)
    sys.stdout.buffer.write(output.getvalue())


if __name__ == '__main__':
    main()
