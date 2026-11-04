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


def prepare_retry_log(log):
    # Resolve the registered task rather than reloading its package as a file.
    log.eval.task_file = None
    sandbox = getattr(log.eval, 'sandbox', None)
    if sandbox is not None and sandbox.type == 'docker' and sandbox.config == 'compose.yaml':
        log.eval.sandbox = sandbox.model_copy(update={'config': str(ROOT / 'evaluation' / 'compose.yaml')})
    # An injected HTTP client is serialized as null in Inspect logs.
    if 'http_client' in log.eval.model_args and log.eval.model_args['http_client'] is None:
        log.eval.model_args['http_client'] = DefaultAsyncHttpxClient(timeout=None)
    return log


def resume(output):
    load_dotenv(ROOT / '.env')
    summary_path = output / 'summary.json'
    original = json.loads(summary_path.read_text())
    if original['status'] == 'success':
        print('Evaluation is already complete.', flush=True)
        return 0
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
        retry_logs = [prepare_retry_log(read_eval_log(path)) for path in original['logs']]
        logs = eval_retry(retry_logs, log_dir=str(output / f'retry-{stamp}'),
                          max_samples=original['max_samples'],
                          max_sandboxes=original['max_sandboxes'],
                          max_connections=original['max_connections'], display='plain')
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
            write_json(summary_path, original)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(resume(parser.parse_args().output.resolve()))
