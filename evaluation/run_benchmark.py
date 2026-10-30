"""Compare base/SFT/RL checkpoints with equal budgets and shared judges."""
import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import httpx
from dotenv import load_dotenv

from .benchmark import digest, prepare, report, write_json
from .suite import ROOT, SUITE, dataset_path, suite


def selected_models(requested, catalog):
    """Explicit named checkpoints, or base plus all discovered completed adapters."""
    from drawing.config import MODEL_ID
    if requested:
        selected = {}
        for value in requested:
            name, separator, model = value.partition('=')
            if not separator or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) or not model:
                raise ValueError('--model must be NAME=MODEL_ID with a filesystem-safe name')
            if name in selected:
                raise ValueError(f'Duplicate comparison name: {name}')
            selected[name] = model
    else:
        selected = {'base': MODEL_ID}
        for item in catalog:
            if item['id'].startswith('artsy/'):
                name = item['id'].split('/', 1)[1]
                if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) or name in selected:
                    raise ValueError(f'Use explicit --model names for checkpoint {item["id"]}')
                selected[name] = item['id']
    if len(selected) < 2 or len(set(selected.values())) != len(selected):
        raise ValueError('Select at least two distinct models; use --model NAME=MODEL_ID for each')
    folders = [a+'-vs-'+b for a,b in combinations(selected, 2)]
    if len(set(folders)) != len(folders):
        raise ValueError('Comparison names produce ambiguous pair directories; use shorter names')
    available = {item['id']: item for item in catalog}
    for model in selected.values():
        if model not in available:
            raise ValueError(f'Model unavailable: {model}')
        if model == MODEL_ID and available[model].get('model_path'):
            raise ValueError('Base alias resolves to an adapter; start bridge with ARTSY_TINKER_MODEL_PATH empty')
    return selected


def main():
    from .harness import DEFAULT_MAX_TOKENS, DEFAULT_TOKEN_LIMIT, DEFAULT_TIME_LIMIT, DEFAULT_MAX_TURNS, LIMITS
    from .evaluate import optional_token_budget, positive
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', type=Path, default=SUITE)
    parser.add_argument('--model', action='append', help='NAME=MODEL_ID; repeat for base, SFT and RL')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--time-limit', type=optional_token_budget, default=DEFAULT_TIME_LIMIT,
                        help='Episode and model-request deadline in seconds; 0 disables it (default: disabled)')
    parser.add_argument('--max-turns', type=positive, default=DEFAULT_MAX_TURNS)
    parser.add_argument('--max-tokens', type=optional_token_budget, default=DEFAULT_MAX_TOKENS,
                        help='Per-response output cap; 0 uses provider capacity (default: 0, same as datagen)')
    parser.add_argument('--token-limit', type=optional_token_budget, default=DEFAULT_TOKEN_LIMIT,
                        help=f'Shared output-token budget; 0 disables it (default: {DEFAULT_TOKEN_LIMIT})')
    parser.add_argument('--approval', type=Path, help='Required only for a suite without recorded validation')
    parser.add_argument('--dry-run', action='store_true', help='Validate the plan locally without bridge, Docker, model calls or writes')
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error('Concurrency must be positive')
    manifest, items = suite(args.suite)
    from ui.prompt_review import validate_review
    if args.approval is not None or manifest.get('status') != 'validated':
        approval = args.approval or ROOT/'eval-results/dataset-review/approval.json'
        if not approval.is_file():
            raise ValueError('This suite needs a saved dataset approval; pass --approval after prompt review')
        validate_review(json.loads(approval.read_text()), manifest, items, require_approval=True)
    load_dotenv(ROOT/'.env')
    if args.dry_run:
        from inference.model_catalog import model_catalog
        catalog = model_catalog()
    else:
        health = httpx.get('http://127.0.0.1:8080/health', timeout=10).raise_for_status().json()
        if health.get('backend') != 'tinker' or health.get('model_path'):
            raise ValueError('Start the bridge with ARTSY_TINKER_MODEL_PATH empty to evaluate the actual base model')
        catalog = health.get('models', [])
    selected = selected_models(args.model, catalog)
    models = {item['id']: item for item in catalog}
    env = dict(os.environ, ARTSY_TINKER_MODEL_PATH='', ARTSY_MODEL_URL='http://127.0.0.1:8080/v1')
    if env.get('ARTSY_CORRECTNESS_JUDGE_URL') == 'tinker://native':
        for suffix in ('MODEL', 'URL', 'API_KEY'):
            env['ARTSY_CORRECTNESS_JUDGE_'+suffix] = env.get('ARTSY_SAFETY_JUDGE_'+suffix, '')
    if not env.get('ARTSY_CORRECTNESS_JUDGE_MODEL') or not env.get('ARTSY_SAFETY_JUDGE_MODEL'):
        raise ValueError('Configure adherence and safety judges before running')
    plan = {'manifest': manifest, 'selected_models': selected,
            'models': [models[m] for m in selected.values()], 'dataset_sha256': manifest['sha256'],
            'max_samples': args.max_samples, 'max_tokens': args.max_tokens, 'token_limit': args.token_limit or None,
            'time_limit': args.time_limit,
            'max_turns': args.max_turns,
            'environment_limits': {**LIMITS, 'maxCalls': None,
                                   'episodeMs': args.time_limit * 1000 if args.time_limit else None},
            'adherence_judge': env['ARTSY_CORRECTNESS_JUDGE_MODEL'], 'safety_judge': env['ARTSY_SAFETY_JUDGE_MODEL'],
            'code_safety_judge': env.get('ARTSY_CODE_SAFETY_JUDGE_MODEL') or env['ARTSY_SAFETY_JUDGE_MODEL'],
            'source_sha256': {p: digest((ROOT/p).read_bytes()) for p in (
                'evaluation/evaluate.py', 'evaluation/harness.py', 'evaluation/host_agent.py', 'evaluation/mini.yaml',
                'drawing/judges.py', 'evaluation/benchmark.py', 'evaluation/run_benchmark.py', 'evaluation/suite.py')}}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'run-plan.json', plan)
    exits = {}
    for name, model in selected.items():
        command = [sys.executable, '-m', 'evaluation.evaluate', '--dataset', str(dataset_path(args.suite)),
                   '--model', model, '--output', str(args.output/name), '--max-samples', str(args.max_samples),
                   '--max-tokens', str(args.max_tokens or 0), '--token-limit', str(args.token_limit or 0),
                   '--time-limit', str(args.time_limit or 0), '--max-turns', str(args.max_turns),
                   '--max-connections', str(args.max_samples), '--max-sandboxes', str(args.max_samples)]
        print(f'Running {name}: {model}', flush=True)
        with (args.output/f'{name}.log').open('w') as log:
            exits[name] = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f'{name} exit code: {exits[name]}', flush=True)
    write_json(args.output/'exit-codes.json', exits)
    comparisons = []
    for reference, candidate in combinations(selected, 2):
        if not all((args.output/name/'summary.json').is_file() for name in (reference, candidate)):
            continue
        folder = args.output/'comparisons'/f'{reference}-vs-{candidate}'
        prepare(SimpleNamespace(base=args.output/reference, sft=args.output/candidate,
            reference_name=reference, candidate_name=candidate, suite=args.suite, output=folder/'review'))
        report(SimpleNamespace(packet=folder/'review', labels=None, output=folder/'report.json'))
        comparisons.append({'reference': reference, 'candidate': candidate,
                            'report': str((folder/'report.json').relative_to(args.output))})
    write_json(args.output/'summary.json', {'suite': manifest['version'], 'dataset_sha256': manifest['sha256'],
        'models': selected, 'exit_codes': exits, 'comparisons': comparisons,
        'status': 'incomplete' if any(exits.values()) else 'success'})
    return int(any(exits.values()))


if __name__ == '__main__':
    raise SystemExit(main())
