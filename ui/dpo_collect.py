"""Collect unfiltered DPO candidates: python -m ui.dpo_collect REQUEST_JSON_PATH."""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import sys
from urllib.request import urlopen

from drawing.config import MODEL_ID
from drawing.dataset import _write_json, _write_jsonl_atomic
from drawing.dataset_specs import validate_specs
from drawing.generate_dataset import _load_env_file
from drawing.renderer import MAX_PNG_BYTES, MAX_LOG_BYTES, RUNTIME_VERSION, validate_png

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = 'http://127.0.0.1:8080'
def validate_collection_request(request):
    if not isinstance(request, dict):
        raise ValueError('Collection request must be an object')
    model = request.get('model')
    if not isinstance(model, str):
        raise ValueError('DPO collection requires a compatible native bridge model')
    if type(request.get('resume', False)) is not bool:
        raise ValueError('resume must be a boolean')
    candidates = request.get('candidates', 2)
    if type(candidates) is not int or not 2 <= candidates <= 4:
        raise ValueError('candidates must be an integer from 2 to 4')
    if not isinstance(request.get('output'), str) or not request['output'].strip():
        raise ValueError('output must be a new directory path')
    checkpoint = request.get('model_path')
    if not isinstance(checkpoint, str) or not checkpoint.startswith('tinker://') or not checkpoint[9:].strip():
        raise ValueError('model_path must be a nonempty tinker:// SFT checkpoint')
    if model != MODEL_ID:
        from inference.model_catalog import model_catalog
        from training.dpo import load_initial_checkpoint
        profile = next((item for item in model_catalog(ROOT)
                        if item['id'] == model and item.get('model_path') == checkpoint), None)
        if profile is None or not model.startswith('artsy/'):
            raise ValueError('Model must be a locally registered compatible SFT checkpoint alias')
        initial = load_initial_checkpoint(ROOT / 'checkpoints' / model[6:] / 'final.json')
        if initial['sampler_path'] != checkpoint:
            raise ValueError('SFT checkpoint sampler does not match requested model_path')
    specs = validate_specs(request.get('specs'))
    if not 1 <= len(specs) <= 150 or any(spec.get('kind') == 'refusal' for spec in specs):
        raise ValueError('Collection requires 1–150 drawing specifications; refusal specifications are excluded')
    return {**request, 'specs': specs, 'candidates': candidates,
            'model_path': checkpoint}


def bridge_health(expected, model=MODEL_ID):
    with urlopen(BRIDGE + '/health', timeout=5) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise RuntimeError('Bridge health response is too large')
    health = json.loads(raw)
    if not isinstance(health, dict) or health.get('model') != MODEL_ID:
        raise RuntimeError('Native bridge architecture does not match Qwen collection')
    profile = health if model == MODEL_ID else next((item for item in health.get('models', [])
        if isinstance(item, dict) and item.get('id') == model), None)
    if profile is None or profile.get('model_path') != expected:
        raise RuntimeError('Native bridge model/checkpoint does not match collection request')
    return {**health, 'model_path': profile['model_path'], 'sampling_model': model}


def portable_trajectory(messages, output, directory):
    """Losslessly preserve native policy/tool content and externalize PNG observations."""
    from training.data import TOOLS
    converted = []
    for message in messages:
        raw = message.model_dump(mode='json', exclude_none=True) if hasattr(message, 'model_dump') else message
        row = {'role': raw['role'], 'content': []}
        content = raw.get('content', '')
        for part in ([{'type': 'text', 'text': content}] if isinstance(content, str) else content):
            kind = part.get('type')
            if kind == 'text':
                row['content'].append({'type': 'text', 'text': part['text']})
            elif kind == 'reasoning':
                if part.get('redacted') or part.get('signature'):
                    raise ValueError('Opaque reasoning cannot be a native DPO training target')
                row['content'].append({'type': 'thinking', 'thinking': part['reasoning']})
            elif kind == 'image':
                url = part['image']
                if not isinstance(url, str) or not url.startswith('data:image/png;base64,'):
                    raise ValueError('Inspect preview attachment was not resolved to PNG data')
                encoded = url.split(',', 1)[1]
                if len(encoded) > ((MAX_PNG_BYTES + 2) // 3) * 4:
                    raise ValueError('Preview PNG is oversized')
                png = base64.b64decode(encoded, validate=True)
                validate_png(png, {'width': 1920, 'height': 1200})
                digest = __import__('hashlib').sha256(png).hexdigest()
                image = directory / ('preview-' + digest + '.png')
                image.write_bytes(png)
                row['content'].append({'type': 'image', 'image': image.relative_to(output).as_posix(), 'imageHash': digest})
            else:
                raise ValueError('Unsupported native trajectory content: ' + str(kind))
        if raw.get('tool_calls'):
            row['tool_calls'] = [{'id': call['id'], 'type': 'function', 'function': {
                'name': call['function'], 'arguments': call['arguments']}} for call in raw['tool_calls']]
        if raw.get('tool_call_id'):
            row['tool_call_id'] = raw['tool_call_id']
        converted.append(row)
    trajectory = {'messages': converted, 'tools': TOOLS,
                  'trainable_message_indices': [i for i, row in enumerate(converted) if row['role'] == 'assistant']}
    from training.data import native_trajectory_hash
    digest = native_trajectory_hash(trajectory)
    return trajectory, digest


async def collect(request):
    request = validate_collection_request(request)
    _load_env_file(ROOT / '.env')
    health = await asyncio.to_thread(bridge_health, request['model_path'], request['model'])
    # The collection worker always uses the verified localhost bridge, including
    # when .env configures another endpoint for unrelated evaluation commands.
    import os
    os.environ['ARTSY_MODEL_URL'] = BRIDGE + '/v1'
    os.environ['ARTSY_MODEL_API_KEY'] = 'local-no-auth'
    from inspect_ai import eval_async
    from inspect_ai.log import read_eval_log
    from evaluation.harness import drawing_task, model_from_env
    from drawing.dataset_specs import sha256
    output = Path(request['output']).resolve()
    provenance = {'mode': 'native-inspect-v1', 'model': MODEL_ID, 'baseUrl': BRIDGE + '/v1',
        'model_path': health['model_path'], 'sampling_model': request['model'],
        'renderer': 'evaluation/compose.yaml', 'candidates': request['candidates'],
        'temperature': 0.8, 'max_tokens': 8192, 'dataset_sha256': request.get('dataset_sha256')}
    manifest = {'version': 'artsy-native-inspect-v1', 'specs': request['specs'],
        'candidates': request['candidates'], 'teacher': provenance,
        'apiDocHash': sha256((ROOT / 'docs/artcanvas.md').read_bytes())}
    expected = {f"{spec['id']}-c{index}": (spec, index) for spec in request['specs']
                for index in range(request['candidates'])}
    saved = {}
    if request.get('resume'):
        if json.loads((output / 'sampling.json').read_text()) != provenance or json.loads((output / 'manifest.json').read_text()) != manifest:
            raise ValueError('Resume requires the exact saved sampling, checkpoint, specifications, and API context')
        from training.data import convert_record
        for path in sorted((output / 'records').glob('*.json')):
            row = json.loads(path.read_text())
            rid = row.get('id')
            if (rid not in expected or path.name != rid + '.json' or rid in saved
                    or row.get('spec') != expected[rid][0] or row.get('candidate') != expected[rid][1]
                    or row.get('teacher') != {'model': request['model'], 'baseUrl': BRIDGE + '/v1', 'model_path': health['model_path']}
                    or row.get('apiDocHash') != manifest['apiDocHash']
                    or row.get('trajectoryFormat') != 'inspect-bash-v1'
                    or row.get('status') not in ('review', 'rejected', 'infrastructure_error')):
                raise ValueError('Existing candidate does not match resume request: ' + path.name)
            if row['status'] == 'review':
                convert_record(row, output)
            saved[rid] = row
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'records').mkdir()
        _write_json(output / 'sampling.json', provenance)
        _write_json(output / 'manifest.json', manifest)
        (output / 'prompts.jsonl').write_text(''.join(json.dumps(spec) + '\n' for spec in request['specs']))
    records = [saved[rid] for rid in expected if rid in saved]
    _write_jsonl_atomic(output / 'candidates.jsonl', records)
    def summarize():
        return {'version': 'artsy-native-inspect-v1', 'prompts': len(request['specs']),
            'candidates': len(records), 'review': sum(r['status'] == 'review' for r in records),
            'rejected': sum(r['status'] == 'rejected' for r in records),
            'infrastructureErrors': sum(r['status'] == 'infrastructure_error' for r in records),
            'complete': len(records) == len(expected)}
    summary = summarize()
    _write_json(output / 'summary.json', summary)
    async def guarded_generate(model, messages, tools, tool_choice, config):
        await asyncio.to_thread(bridge_health, request['model_path'], request['model'])
        generated = await model.api.generate(messages, tools, tool_choice, config)
        await asyncio.to_thread(bridge_health, request['model_path'], request['model'])
        # Providers may also return API-call metadata. FilteredAPI requires the
        # ModelOutput itself, otherwise it would issue a second generation.
        result = generated[0] if isinstance(generated, tuple) else generated
        if isinstance(result, Exception):
            raise result
        return result

    for spec in request['specs']:
        for candidate in range(request['candidates']):
            rid = f"{spec['id']}-c{candidate}"
            if rid in saved:
                print(f"{rid}: preserved {saved[rid]['status']}", flush=True)
                continue
            directory = output / 'artifacts' / rid
            directory.mkdir(parents=True)
            row = {'version': 'artsy-native-inspect-v1', 'trajectoryFormat': 'inspect-bash-v1',
                'nativeTrajectoryConfig': {'token_limit': None, 'token_limit_type': 'output'},
                'id': rid, 'spec': spec, 'candidate': candidate, 'status': 'infrastructure_error',
                'reason': 'Native rollout did not complete', 'code': '', 'trace': [], 'evaluations': [],
                'teacher': {'model': request['model'], 'baseUrl': BRIDGE + '/v1', 'model_path': health['model_path']},
                'runtimeVersion': RUNTIME_VERSION, 'apiDocHash': sha256((ROOT / 'docs/artcanvas.md').read_bytes()),
                'humanReviewed': False, 'artifacts': {'directory': directory.relative_to(output).as_posix()}}
            try:
                task = drawing_task([spec], str(directory / 'scored'), temperature=0.8,
                    max_connections=1, max_tokens=8192, generation_filter=guarded_generate)
                logs = await eval_async(task, model=model_from_env(request['model']),
                    max_samples=1, max_tasks=1, epochs=1, log_dir=str(directory / 'logs'),
                    log_images=True, retry_on_error=0)
                log_files = sorted((directory / 'logs').glob('*.eval'))
                if not log_files:
                    raise RuntimeError('Inspect did not save its native .eval log')
                resolved = [read_eval_log(str(path), resolve_attachments=True) for path in log_files]
                _write_json(directory / 'trace.json', [log.model_dump(mode='json') for log in resolved])
                row['artifacts'].update(trajectory=(directory / 'trace.json').relative_to(output).as_posix(),
                    inspectLogs=[path.relative_to(output).as_posix() for path in log_files])
                samples = [sample for log in resolved for sample in (log.samples or [])]
                if len(samples) != 1:
                    raise RuntimeError('Expected exactly one native Inspect sample')
                sample = samples[0]
                native, native_hash = portable_trajectory(sample.messages, output, directory)
                row.update(nativeTrajectory=native, nativeTrajectoryHash=native_hash)
                artifact = directory / 'scored' / f"{spec['id']}-1"
                result_path = artifact / 'result.json'
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                row['nativeResult'] = result
                row['status'] = 'rejected'
                row['reason'] = result.get('error') or 'Native agent did not submit a valid drawing'
                for name, key in (('program.py', 'source'), ('image.png', 'image'), ('result.json', 'result')):
                    path = artifact / name
                    if path.is_file():
                        row['artifacts'][key] = path.relative_to(output).as_posix()
                source = artifact / 'program.py'
                if source.exists():
                    code = source.read_text()
                    if len(code.encode()) > 65536:
                        raise ValueError('Native source exceeds 64 KiB')
                    row.update(code=code, sourceHash=sha256(code.replace('\r\n', '\n').strip()))
                if not logs or any(log.status != 'success' for log in logs) or sample.error is not None:
                    raise RuntimeError('Native Inspect rollout failed: ' + str(sample.error or 'see .eval log'))
                if result.get('outcome') == 'submitted' and result.get('submitted') is True and result.get('scores', {}).get('render_valid') == 1:
                    png = (artifact / 'image.png').read_bytes()
                    validate_png(png, {'width': 1920, 'height': 1200})
                    row.update(status='review', reason='Native submitted drawing; human preference review pending', imageHash=sha256(png),
                        submission={'feedback': result['feedback']})
                    from training.data import convert_record
                    try:
                        convert_record(row, output)
                    except ValueError as exc:
                        row.update(status='rejected', reason='Native trajectory is not trainable: ' + str(exc))
            except Exception as exc:
                row.update(status='infrastructure_error', reason=str(exc)[:MAX_LOG_BYTES])
            saved[rid] = row
            records = [saved[key] for key in expected if key in saved]
            _write_json(output / 'records' / (rid + '.json'), row)
            _write_jsonl_atomic(output / 'candidates.jsonl', records)
            summary = summarize()
            _write_json(output / 'summary.json', summary)
            print(f"{rid}: {row['status']}: {row['reason']}", flush=True)
            if row['status'] == 'infrastructure_error':
                print('Stopping after native infrastructure failure; inspect its preserved logs.', flush=True)
                return summary
    return summary


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m ui.dpo_collect REQUEST_JSON_PATH')
    summary = asyncio.run(collect(json.loads(Path(sys.argv[1]).read_text())))
    print(json.dumps(summary), flush=True)
    return 1 if summary.get('complete') is False or summary.get('infrastructureErrors') else 0


if __name__ == '__main__':
    raise SystemExit(main())
