"""Run the existing drawing CLI and export its complete recorded Inspect trace."""
import json
from pathlib import Path
import sys


def main():
    directory, prompt, seed, model = sys.argv[1:]
    output = Path(directory) / 'output'
    sys.argv = ['evaluation.draw', prompt, '--seed', seed, '--model', model, '--episode', str(output)]
    job = json.loads((Path(directory) / 'job.json').read_text())
    progress = None
    if job.get('experience') == 'artsy':
        sys.argv += ['--max-turns', '10']
        if job.get('parent_id'):
            from ui.server import artsy_request, JOBS
            # Revalidate hashes immediately before passing completed source to the model.
            artsy_request({'prompt': prompt, 'parent_id': job['parent_id']})
            parent = JOBS / job['parent_id']
            previous = json.loads((parent / 'job.json').read_text())
            sys.argv[1] = f"Previous request: {previous['prompt']}\n\nRequested change: {prompt}"
            sys.argv += ['--reference-source', str(parent / 'output/program.py')]

        def progress(event):
            temporary = Path(directory) / 'progress.tmp'
            temporary.write_text(json.dumps(event))
            temporary.replace(Path(directory) / 'progress.json')
    try:
        from evaluation.draw import main as draw
        return draw(progress=progress)
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
