"""Run the frozen benchmark on base and SFT with shared Inspect judges."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import httpx
from dotenv import load_dotenv
from .benchmark import ROOT, SUITE, suite, digest, prepare, report, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-samples',type=int,default=4)
    parser.add_argument('--approval',type=Path,default=ROOT/'eval-results/dataset-review/approval.json')
    args=parser.parse_args()
    if args.max_samples<1:parser.error('--max-samples must be positive')
    manifest,items=suite()
    from .dataset_viewer import validate_review
    if not args.approval.is_file():
        raise ValueError('Review and approve the dataset in evaluation.dataset_viewer before running')
    validate_review(json.loads(args.approval.read_text()),manifest,items,require_approval=True)
    load_dotenv(ROOT/'.env')
    health=httpx.get('http://127.0.0.1:8080/health',timeout=10).raise_for_status().json()
    if health.get('backend')!='tinker' or health.get('model_path'):
        raise ValueError('Start the bridge with ARTSY_TINKER_MODEL_PATH empty to evaluate the actual base model')
    models={m['id']:m for m in health.get('models',[])}
    if any(m not in models for m in manifest['models']):raise ValueError('Base or SFT model unavailable on bridge')
    env=dict(os.environ, ARTSY_TINKER_MODEL_PATH='', ARTSY_MODEL_URL='http://127.0.0.1:8080/v1')
    if env.get('ARTSY_CORRECTNESS_JUDGE_URL')=='tinker://native':
        # Inspect's generic client cannot use native Tinker. Make an independent
        # adherence call through the configured compatible safety endpoint.
        for suffix in ('MODEL','URL','API_KEY'):
            env['ARTSY_CORRECTNESS_JUDGE_'+suffix]=env.get('ARTSY_SAFETY_JUDGE_'+suffix,'')
    if not env.get('ARTSY_CORRECTNESS_JUDGE_MODEL') or not env.get('ARTSY_SAFETY_JUDGE_MODEL'):
        raise ValueError('Configure correctness and safety judges before running')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'run-plan.json',{'manifest':manifest,'models':[models[m] for m in manifest['models']],
        'dataset_sha256':manifest['sha256'], 'max_samples':args.max_samples,
        'adherence_judge':env['ARTSY_CORRECTNESS_JUDGE_MODEL'],'safety_judge':env['ARTSY_SAFETY_JUDGE_MODEL'],
        'code_safety_judge':env.get('ARTSY_CODE_SAFETY_JUDGE_MODEL') or env['ARTSY_SAFETY_JUDGE_MODEL'],
        'source_sha256':{p:digest((ROOT/p).read_bytes()) for p in ('evaluation/harness.py','drawing/judges.py','evaluation/benchmark.py')}})
    exits={}
    for name,model in zip(('base','sft'),manifest['models']):
        command=[sys.executable,'-m','evaluation.evaluate','--dataset',str(SUITE/'prompts.jsonl'),
                 '--model',model,'--output',str(args.output/name),'--max-samples',str(args.max_samples)]
        print(f'Running {name}: {model}',flush=True)
        with (args.output/f'{name}.log').open('w') as log:
            exits[name]=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        print(f'{name} exit code: {exits[name]}',flush=True)
    write_json(args.output/'exit-codes.json',exits)
    if all((args.output/name/'summary.json').is_file() for name in exits):
        prepare(SimpleNamespace(base=args.output/'base',sft=args.output/'sft',output=args.output/'review'))
        report(SimpleNamespace(packet=args.output/'review',labels=None,output=args.output/'report.json'))
    return int(any(exits.values()))

if __name__=='__main__':raise SystemExit(main())
