"""Retry only failed Inspect samples, then rebuild the original eval summary."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from inspect_ai import eval_retry
from inspect_ai.log import read_eval_log
from openai import DefaultAsyncHttpxClient

from evaluation.evaluate import ROOT, _recover_log, apply_reporting, load_specs
from evaluation.harness import check_native_bridge
from evaluation.judge_pairs import write_json


def prepare_retry_log(log, *, execution_only=False):
    # Upgrade early HPSv3 logs to Inspect's per-metric unscored sentinel while
    # preserving completed generations and their real raw rewards.
    for sample in getattr(log, 'samples', None) or []:
        for score in (sample.scores or {}).values():
            if (score.metadata or {}).get('hpsv3Model') and isinstance(score.value, dict):
                score.value.setdefault('hpsv3_score', float('nan'))
    if execution_only:
        keys = ('hpsv3_url', 'hpsv3_model', 'correctness_judge_model',
                'safety_judge_model', 'code_safety_judge_model')
        # Inspect retries prefer task_args_passed and restore recorded scorer
        # options separately, so all three configurations must be migrated.
        configurations = [log.eval.task_args, getattr(log.eval, 'task_args_passed', None)]
        configurations += [scorer.options for scorer in getattr(log.eval, 'scorers', []) or []
                           if scorer.name.rsplit('/', 1)[-1] == 'drawing_score']
        for configuration in configurations:
            if configuration is not None:
                for key in keys:
                    configuration[key] = None
        for sample in getattr(log, 'samples', None) or []:
            for score in (sample.scores or {}).values():
                if isinstance(score.value, dict):
                    score.value = {'render_valid': score.value['render_valid']}
    # Resolve the registered task rather than reloading its package as a file.
    log.eval.task_file = None
    sandbox = getattr(log.eval, 'sandbox', None)
    if sandbox is not None and sandbox.type == 'docker' and sandbox.config == 'compose.yaml':
        log.eval.sandbox = sandbox.model_copy(update={'config': str(ROOT / 'evaluation' / 'compose.yaml')})
    # An injected HTTP client is serialized as null in Inspect logs.
    if 'http_client' in log.eval.model_args and log.eval.model_args['http_client'] is None:
        log.eval.model_args['http_client'] = DefaultAsyncHttpxClient(timeout=None)
    return log


def resume(output, *, execution_only=False):
    load_dotenv(ROOT / '.env')
    summary_path = output / 'summary.json'
    original = json.loads(summary_path.read_text())
    if not original.get('logs'):
        original['logs'] = [str(path) for path in sorted((output / 'logs').glob('*.eval'))]
        if not original['logs']:
            raise ValueError('No persisted Inspect logs available to resume')
    if original['status'] == 'success':
        print('Evaluation is already complete.', flush=True)
        return 0
    if execution_only:
        original.setdefault('previous_scoring', {key: original.get(key) for key in
                             ('mode', 'hpsv3_model', 'hpsv3_url', 'scoring_version')})
        original.update(mode='execution-only', hpsv3_model=None, hpsv3_url=None,
                        correctness_judge_model=None, safety_judge_model=None,
                        code_safety_judge_model=None, scoring_version='artsy-eval-v3-submitted-rubric')
        original.pop('hpsv3', None)
    check_native_bridge(original['model'])
    # Inspect omits credentials from logs and reconstructs this named client
    # on retry. The local bridge accepts the same placeholder as the first run.
    if urlsplit(original.get('base_url', '')).hostname in ('127.0.0.1', 'localhost', '::1'):
        os.environ.setdefault('ARTSY_API_KEY', 'local-no-auth')
    # Keep every completed outcome, including valid render failures/refusals.
    preserved = {}
    for result_path in (output / 'artifacts').glob('*/result.json'):
        result = json.loads(result_path.read_text())
        preserved[result_path] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        if result['scores']['render_valid']:
            image = result_path.parent / 'image.png'
            if hashlib.sha256(image.read_bytes()).hexdigest() != result['imageHash']:
                raise ValueError('A completed image has changed; refusing to resume')
            preserved[image] = result['imageHash']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = output / f'summary.before-resume-{stamp}.json'
    summary_path.rename(backup)
    print(json.dumps({'status': 'resuming', 'model': original['model'],
                      'preserved_outcomes': len(list((output / 'artifacts').glob('*/result.json'))),
                      'failed_samples': original.get('sample_errors'), 'backup': str(backup)}), flush=True)
    try:
        retry_logs = [prepare_retry_log(read_eval_log(path), execution_only=execution_only) for path in original['logs']]
        retry_directory = output / f'retry-{stamp}'
        logs = eval_retry(retry_logs, log_dir=str(retry_directory),
                          max_samples=original['max_samples'],
                          max_sandboxes=original['max_sandboxes'],
                          max_connections=original['max_connections'], display='plain')
        if not logs:
            logs = [read_eval_log(str(path)) for path in sorted(retry_directory.glob('*.eval'))]
        expected = len(original['ids']) * original['epochs']
        logs = [_recover_log(log, expected) for log in logs]
        for path, expected_hash in preserved.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f'Previously completed artifact changed: {path}')
        samples = [sample for log in logs for sample in (log.samples or [])]
        errors = sum(sample.error is not None for sample in samples)
        summary = {**original, 'completed_samples': len(samples), 'sample_errors': errors,
                   'logs': [log.location for log in logs],
                   'results': [log.results.model_dump(mode='json') if log.results else None for log in logs],
                   'usage': [log.stats.model_dump(mode='json') for log in logs],
                   'status': 'success' if logs and all(log.status == 'success' for log in logs)
                       and not errors and len(samples) == expected else 'incomplete',
                   'resumed_from': original['logs']}
        wanted = set(original['ids'])
        specs = [spec for spec in load_specs(Path(original['dataset'])) if spec['id'] in wanted]
        apply_reporting(summary, specs, output / 'artifacts', original['epochs'])
        write_json(summary_path, summary)
        print(json.dumps({'status': summary['status'], 'sample_errors': errors,
                          'summary': str(summary_path)}), flush=True)
        return 0 if summary['status'] == 'success' else 1
    except BaseException:
        if not summary_path.exists():
            persisted = sorted(retry_directory.glob('*.eval')) if 'retry_directory' in locals() else []
            if persisted:
                original.update(status='incomplete', logs=[str(path) for path in persisted])
            write_json(summary_path, original)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execution-only', action='store_true', help='Resume generation without image or LLM judges')
    args = parser.parse_args()
    raise SystemExit(resume(args.output.resolve(), execution_only=args.execution_only))
