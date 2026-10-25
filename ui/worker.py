"""Run the existing drawing CLI and export its complete recorded Inspect trace."""
import json
from pathlib import Path
import sys


def main():
    directory, prompt, seed, model = sys.argv[1:]
    output = Path(directory) / 'output'
    sys.argv = ['evaluation.draw', prompt, '--seed', seed, '--model', model, '--episode', str(output)]
    try:
        from evaluation.draw import main as draw
        return draw()
    finally:
        logs = sorted((output / 'logs').glob('*.eval'))
        if logs:
            try:
                from inspect_ai.log import read_eval_log
                records = [read_eval_log(str(path), resolve_attachments=True).model_dump(mode='json') for path in logs]
                target = output / 'trace.tmp'
                target.write_text(json.dumps(records))
                target.replace(output / 'trace.json')
            except Exception as error:
                print(f'Trace export failed: {error}', file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
