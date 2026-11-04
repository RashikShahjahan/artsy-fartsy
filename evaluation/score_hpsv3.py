"""Score saved final images after every generation run has completed.

Run in the dedicated MLX environment. The model is loaded lazily, used for the
batch, and released when this process exits; no scoring server is needed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

from evaluation.compare_hpsv3 import compare
from training.hpsv3 import DEFAULT_MODEL, normalized_reward


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def prepare_images(runs: dict[str, Path]) -> list[dict]:
    """Validate the entire batch before any GPU model is loaded."""
    rows = []
    configuration = None
    for name, run in runs.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Run names may contain letters, digits, underscores and hyphens')
        summary = json.loads((run / 'summary.json').read_text())
        if summary['status'] != 'success' or summary.get('sample_errors'):
            raise ValueError(f'Generation must complete without sample errors before scoring: {name}')
        settings = {key: summary.get(key) for key in
                    ('ids', 'epochs', 'max_tokens', 'max_turns', 'token_limit', 'time_limit')}
        if configuration is not None and settings != configuration:
            raise ValueError('Generation settings differ across runs')
        configuration = settings
        for prompt_id in summary['ids']:
            for epoch in range(1, summary['epochs'] + 1):
                directory = run / 'artifacts' / f'{prompt_id}-{epoch}'
                result = json.loads((directory / 'result.json').read_text())
                valid = bool(result['scores']['render_valid'])
                image = directory / 'image.png'
                if valid:
                    png = image.read_bytes()
                    if hashlib.sha256(png).hexdigest() != result['imageHash']:
                        raise ValueError(f'Final image hash mismatch: {image}')
                    from drawing.render_contract import validate_png
                    validate_png(png, {'width': 1920, 'height': 1200})
                rows.append({'run': name, 'id': prompt_id, 'epoch': epoch,
                             'image': image, 'result': result, 'valid': valid})
    if not rows:
        raise ValueError('No generation episodes to score')
    return rows


def score_runs(runs: dict[str, Path], output: Path, model_id: str, load_model) -> dict:
    rows = prepare_images(runs)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.hpsv3-score.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    model = None
    try:
        os.write(fd, str(os.getpid()).encode())
        completed = reused = 0
        for row in rows:
            result = row['result']
            image_hash = result.get('imageHash')
            fingerprint = hashlib.sha256(json.dumps(
                [result['prompt'], image_hash, row['valid'], result.get('expectedRefusal', False), model_id],
                sort_keys=True).encode()).hexdigest()
            destination = output / 'records' / row['run'] / f"{row['id']}-{row['epoch']}.json"
            if destination.exists():
                saved = json.loads(destination.read_text())
                if saved.get('fingerprint') != fingerprint:
                    raise ValueError(f'Score checkpoint inputs changed: {destination}')
                reused += 1
            else:
                scores = {'render_valid': float(row['valid']), 'hpsv3_preference': 0.0}
                source = 'no-valid-image'
                if row['valid'] and not result.get('expectedRefusal'):
                    cached = result['scores'].get('hpsv3_score')
                    if (result.get('hpsv3Model') == model_id and type(cached) in (float, int)
                            and math.isfinite(cached)):
                        raw = cached
                        source = 'existing-inline-score'
                        reused += 1
                    else:
                        if model is None:
                            model = load_model()
                        raw = model.score(result['prompt'], str(row['image']))
                        source = 'batch'
                    if type(raw) not in (float, int) or not math.isfinite(raw):
                        raise ValueError('HPSv3 returned a nonfinite score')
                    scores.update(hpsv3_score=raw, hpsv3_preference=normalized_reward(raw))
                write_json(destination, {'id': row['id'], 'epoch': row['epoch'], 'prompt': result['prompt'],
                                         'imageHash': image_hash, 'hpsv3Model': model_id,
                                         'fingerprint': fingerprint, 'source': source, 'scores': scores})
            completed += 1
            write_json(output / 'scoring-status.json', {'phase': 'scoring', 'completed': completed,
                       'total': len(rows), 'reused': reused, 'model_loaded': model is not None,
                       'model': model_id, 'updated_at': datetime.now(timezone.utc).isoformat()})
        comparison = compare(runs, output, score_records=output / 'records')
        write_json(output / 'scoring-status.json', {'phase': 'complete', 'completed': completed,
                   'total': len(rows), 'reused': reused, 'model': model_id})
        return comparison
    finally:
        del model
        os.close(fd)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=GENERATION_OUTPUT_DIR')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True, help='Prepared MLX HPSv3 cache')
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        name, separator, directory = value.partition('=')
        if not separator or not directory or name in runs:
            parser.error('Use unique NAME=GENERATION_OUTPUT_DIR entries')
        runs[name] = Path(directory)
    # Refuse incomplete generation before touching the large checkpoint.
    prepare_images(runs)
    digest = hashlib.sha256()
    with (args.cache / 'mlx-4bit.safetensors').open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    model_id = DEFAULT_MODEL + '@sha256:' + digest.hexdigest() + ':mlx-4bit-g64-v1'
    def load_model():
        from training.hpsv3_mlx import MLXRewardInferencer
        return MLXRewardInferencer(args.cache)
    print(json.dumps(score_runs(runs, args.output, model_id, load_model), indent=2))


if __name__ == '__main__':
    main()
