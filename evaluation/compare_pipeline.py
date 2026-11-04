"""Wait for launched evals, then judge all models together without intervention."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluation.judge_pairs import write_json


def scoring_command(args):
    if args.judge == 'hpsv3':
        command = [args.hpsv3_python, '-u']
        if args.hpsv3_python_path:
            code = ('import sys,runpy; sys.path.extend(' + repr(args.hpsv3_python_path)
                    + "); runpy.run_module('evaluation.score_hpsv3', run_name='__main__')")
            command += ['-c', code]
        else:
            command += ['-m', 'evaluation.score_hpsv3']
        command += ['--cache', str(args.hpsv3_cache)]
    else:
        command = [sys.executable, '-u', '-m', 'evaluation.judge_pairs']
    command += ['--output', str(args.output)]
    for value in args.run:
        command += ['--run', value]
    return command


def run(args):
    runs = [Path(value.partition('=')[2]).resolve() for value in args.run]
    args.output.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
             'runs': args.run, 'status': 'waiting_for_evals'}
    deadline = time.monotonic() + 24 * 60 * 60

    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'pipeline.json', state)

    try:
        while True:
            pending, progress = [], {}
            for path in runs:
                summary = path / 'summary.json'
                if summary.is_file():
                    result = json.loads(summary.read_text())
                    if result.get('status') != 'success':
                        raise RuntimeError(f'Eval did not complete successfully: {path.name}')
                    progress[path.name] = result['completed_samples']
                    continue
                pending.append(path)
                progress[path.name] = len(list(path.glob('artifacts/*/result.json')))
                pid_file = Path(str(path) + '.pid')
                if not pid_file.is_file():
                    raise RuntimeError(f'No launched eval or summary: {path.name}')
                try:
                    os.kill(int(pid_file.read_text()), 0)
                except ProcessLookupError:
                    raise RuntimeError(f'Eval exited without a summary: {path.name}') from None
            state['completed_prompts'] = progress
            save()
            if not pending:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Timed out waiting for evals after 24 hours')
            time.sleep(30)
        state['status'] = 'judging'
        command = scoring_command(args)
        state['judge_command'] = command
        save()
        with (args.output / 'judge.console.log').open('ab') as log:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Judge exited with code {result.returncode}; see judge.console.log')
        state['status'] = 'complete'
        save()
    except Exception as error:
        state.update(status='failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judge', choices=('llm', 'hpsv3'), default='llm')
    parser.add_argument('--hpsv3-python', default=sys.executable, help='Python environment with MLX HPSv3 dependencies')
    parser.add_argument('--hpsv3-cache', type=Path)
    parser.add_argument('--hpsv3-python-path', action='append', default=[], help='Optional extra dependency directory')
    args = parser.parse_args()
    if args.judge == 'hpsv3' and args.hpsv3_cache is None:
        parser.error('--hpsv3-cache is required for the deferred HPSv3 phase')
    run(args)
