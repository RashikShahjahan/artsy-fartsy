"""Package the allow-listed runtime sources, without credentials or checkpoints."""
from pathlib import Path
import sys
import tarfile


def package(destination):
    root = Path(__file__).resolve().parents[2]
    files = {path for directory in ('drawing', 'inference', 'runtime')
             for path in (root / directory).glob('*.py')}
    files.update(root / name for name in (
        'runtime/requirements.txt', 'inference/requirements-tinker.txt',
        'drawing/requirements.txt', 'docs/artcanvas.md', 'evaluation/__init__.py',
        'evaluation/render.py', 'evaluation/agent_commands.py',
        'client/sandbox/agent.py', 'client/sandbox/action.py', 'client/sandbox/snapshot.py'))
    with tarfile.open(destination, 'w:gz') as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(root)))
    print(f'Packaged {len(files)} runtime source files into {destination}')


if __name__ == '__main__':
    package(sys.argv[1])
