"""Download the pinned Kokoro model and voices into the ignored project cache."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

DIRECTORY = Path(__file__).resolve().parents[1] / '.cache/kokoro'
BASE = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'
ASSETS = {
    'kokoro-v1.0.int8.onnx': '6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb',
    'voices-v1.0.bin': 'bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d',
}


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        target = DIRECTORY / name
        if target.is_file() and digest(target) == expected:
            print(f'{name}: ready')
            continue
        temporary = target.with_suffix(target.suffix + '.part')
        try:
            with urlopen(BASE + name, timeout=60) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if digest(temporary) != expected:
                raise RuntimeError(f'Checksum mismatch for {name}; model was not installed.')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'{name}: ready')


if __name__ == '__main__':
    main()
