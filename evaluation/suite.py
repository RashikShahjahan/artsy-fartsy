"""Frozen evaluation suite selection shared by CLI, benchmark and UI."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from drawing.eval_specs import load_specs

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'evaluation/benchmark-v3'


def suite(folder=SUITE):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    path = folder / manifest['dataset']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Benchmark dataset differs from its frozen manifest')
    specs = load_specs(path)
    for spec in specs:
        spec.setdefault('benchmarkSlice', spec.get('safetyCategory', 'harmful_image')
                        if spec.get('kind') == 'refusal' else 'drawing')
    if dict(Counter(s['benchmarkSlice'] for s in specs)) != manifest['counts']:
        raise ValueError('Benchmark slice counts differ from manifest')
    return manifest, specs


def dataset_path(folder=SUITE):
    manifest, _ = suite(folder)
    return (Path(folder) / manifest['dataset']).resolve()
