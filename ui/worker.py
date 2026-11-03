"""Run the existing drawing CLI and export its complete recorded Inspect trace."""
import asyncio
import base64
import json
import shutil
from pathlib import Path
import sys


def main():
    directory, prompt, seed, model = sys.argv[1:]
    output = Path(directory) / 'output'
    sys.argv = ['evaluation.draw', prompt, '--seed', seed, '--model', model, '--episode', str(output)]
    job = json.loads((Path(directory) / 'job.json').read_text())
    steps = {}

    def progress(event):
        event = dict(event)
        step = event.pop('step', None)
        if step is not None:
            step = dict(step)
            previous = steps.get(step['id'], {})
            # Store PNGs once, rather than sending base64 on every status poll.
            if 'previews' in step:
                previews = []
                output.mkdir(parents=True, exist_ok=True)
                number = list(steps).index(step['id']) if step['id'] in steps else len(steps)
                for index, url in enumerate(step['previews']):
                    name = f'preview-{number}-{index}.png'
                    temporary = output / (name + '.tmp')
                    temporary.write_bytes(base64.b64decode(url.split(',', 1)[1], validate=True))
                    temporary.replace(output / name)
                    previews.append(name)
                step['previews'] = previews
            steps[step['id']] = {**previous, **step}
        temporary = Path(directory) / 'progress.tmp'
        temporary.write_text(json.dumps({**event, 'steps': list(steps.values())}))
        temporary.replace(Path(directory) / 'progress.json')

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

    async def approve(step_id):
        while True:
            try:
                decision = json.loads((Path(directory) / 'approval.json').read_text())
                if decision.get('step_id') == step_id:
                    return
            except (OSError, ValueError):
                pass
            await asyncio.sleep(0.2)

    try:
        from evaluation.draw import main as draw
        status = draw(progress=progress, **({'approval': approve} if job.get('execution_mode') == 'approval' else {}))
        result_file = output / 'result.json'
        if job.get('experience') == 'artsy' and result_file.is_file():
            result = json.loads(result_file.read_text())
            if result.get('outcome') == 'incomplete' and result.get('turnLimitReached'):
                from ui.server import content_hash
                # Keep the latest successful preview and current source for the next prompt.
                preview = next((output / name for step in reversed(list(steps.values()))
                                for name in reversed(step.get('previews', []))
                                if (output / name).is_file()), None)
                if preview:
                    shutil.copyfile(preview, output / 'image.png')
                result.update(sourceHash=content_hash(output / 'program.py'),
                              imageHash=content_hash(output / 'image.png'))
                result_file.write_text(json.dumps(result))
                return 0
        return status
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
