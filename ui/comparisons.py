"""Read saved paired judgments and serve only their verified final images."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote


def confined(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact is outside the evaluation directory.')
    return path


def directory(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,180}', name):
        raise ValueError('Unknown comparison.')
    return confined(root, Path(root) / name)


def manifest(root, name):
    folder = directory(root, name)
    data = json.loads(confined(root, folder / 'manifest.json').read_text())
    if not isinstance(data.get('pairs'), list) or not data.get('fingerprint'):
        raise ValueError('Not a paired image comparison.')
    return folder, data


def list_comparisons(root):
    found = []
    for path in sorted(Path(root).glob('*/manifest.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            _, data = manifest(root, path.parent.name)
            found.append({'id': path.parent.name, 'models': data['source_models'],
                          'judge': data['model'], 'total': len(data['pairs'])})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for path in sorted(Path(root).glob('*/pipeline.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        if any(item['id'] == path.parent.name for item in found):
            continue
        try:
            data = pending_packet(root, path.parent.name)
            found.insert(0, {'id': data['id'], 'models': list(data['models'].values()),
                             'judge': data['judge'], 'total': data['total']})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for path in sorted(Path(root).glob('*/collection.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = collection_packet(root, path.parent.name)
            found.insert(0, {'id': data['id'], 'label': data['label'],
                             'models': list(data['models'].values()),
                             'judge': data['judge'], 'total': len(data['pairs'])})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def pending_packet(root, name):
    folder = directory(root, name)
    pipeline = json.loads(confined(root, folder / 'pipeline.json').read_text())
    models, progress = {}, {}
    total = 0
    for value in pipeline['runs']:
        role, _, source = value.partition('=')
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,39}', role):
            raise ValueError('Invalid model role.')
        run = confined(root, source)
        summary_path = confined(root, run / 'summary.json')
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            models[role] = summary['model']
            total = max(total, len(summary['ids']))
        else:
            launch = json.loads(confined(root, Path(str(run) + '.launch.json')).read_text())
            command = launch['command']
            models[role] = command[command.index('--model') + 1]
        progress[role] = len(list(run.glob('artifacts/*/result.json')))
    return {'id': name, 'pending': True, 'state': pipeline['status'],
            'error': pipeline.get('error'), 'models': models, 'progress': progress,
            'total': total, 'judge': 'openai/gpt-6-luna', 'pairs': []}


def packet(root, name):
    if confined(root, directory(root, name) / 'collection.json').is_file():
        return collection_packet(root, name)
    if not confined(root, directory(root, name) / 'manifest.json').is_file():
        return pending_packet(root, name)
    folder, data = manifest(root, name)
    pairs = []
    for item in data['pairs']:
        sid = item['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', sid):
            raise ValueError('Invalid prompt ID.')
        record = None
        record_path = confined(root, folder / 'records' / f'{sid}.json')
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if record.get('fingerprint') != data['fingerprint']:
                record = None
        members = {}
        for role in data.get('roles', ['base', 'sft']):
            if not re.fullmatch(r'[a-z][a-z0-9_]{0,39}', role):
                raise ValueError('Invalid model role.')
            member = item['members'][role]
            members[role] = {'outcome': member['outcome'],
                'image_url': f'/api/comparisons/{quote(name)}/images/{quote(sid)}/{role}'
                    if member.get('image') else None,
                'scores': record['scores'][role] if record else None}
        pairs.append({'id': sid, 'prompt': item['prompt'], 'members': members,
                      'preferred': record['preferred'] if record else None,
                      'comparison': record['comparison'] if record else None,
                      'labels': record['label_mapping'] if record else None})
    return {'id': name, 'models': dict(zip(data.get('roles', ['base', 'sft']), data['source_models'])),
            'judge': data['model'], 'pairs': pairs}


def image(root, name, sid, role):
    collection_path = confined(root, directory(root, name) / 'collection.json')
    if collection_path.is_file():
        config = json.loads(collection_path.read_text())
        if not any(p['id'] == sid for p in config['prompts']):
            raise ValueError('Unknown image.')
        if sid in config['original_ids']:
            manifest(root, config['original_comparison'])
            return image(root, config['original_comparison'], sid, role)
        if role not in config['added_runs']:
            raise ValueError('Unknown image.')
        result, path = live_result(root, config['added_runs'][role], sid)
        if path is None:
            raise ValueError('No final image.')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != result['imageHash']:
            raise ValueError('Final image has changed.')
        return content
    _, data = manifest(root, name)
    pair = next((p for p in data['pairs'] if p['id'] == sid), None)
    if pair is None or role not in data.get('roles', ['base', 'sft']):
        raise ValueError('Unknown image.')
    member = pair['members'][role]
    if not member.get('image'):
        raise ValueError('No final image.')
    path = confined(root, member['image'])
    if path.name != 'image.png':
        raise ValueError('Not a final image.')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != member['image_sha256']:
        raise ValueError('Final image has changed.')
    return content


def live_result(root, run, sid):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', sid):
        raise ValueError('Invalid prompt ID.')
    folder = directory(root, run)
    result_path = confined(root, folder / 'artifacts' / f'{sid}-1' / 'result.json')
    try:
        result = json.loads(result_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    path = confined(root, result_path.parent / 'image.png')
    valid = bool(result.get('scores', {}).get('render_valid')) and path.is_file()
    return result, path if valid else None


def collection_packet(root, name):
    """Combine historical judgments with newly generated outputs by prompt ID."""
    config = json.loads(confined(root, directory(root, name) / 'collection.json').read_text())
    source = config['original_comparison']
    manifest(root, source)  # Collections cannot recursively reference collections.
    original = packet(root, source)
    models = original['models']
    if set(models) != set(config['added_runs']):
        raise ValueError('Combined models differ from the original comparison.')
    prompts = config['prompts']
    ids = [p['id'] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate prompt IDs in combined dataset.')
    original_by_id = {p['id']: p for p in original['pairs']}
    if set(config['original_ids']) != set(original_by_id) or not set(original_by_id).issubset(ids):
        raise ValueError('Original prompt IDs differ from the saved comparison.')
    added_judgments, added_inputs = {}, {}
    if config.get('added_comparison'):
        folder = directory(root, config['added_comparison'])
        if (folder / 'manifest.json').is_file():
            _, added_manifest = manifest(root, config['added_comparison'])
            added = packet(root, config['added_comparison'])
            if added['models'] != models or added['judge'] != original['judge']:
                raise ValueError('Added comparison uses different models or judge.')
            added_judgments = {p['id']: p for p in added['pairs']}
            added_inputs = {p['id']: p['members'] for p in added_manifest['pairs']}
    generated = {role: 0 for role in models}
    pairs = []
    for spec in prompts:
        sid = spec['id']
        if sid in original_by_id:
            pair = dict(original_by_id[sid])
            if pair['prompt'] != spec['prompt']:
                raise ValueError('Original prompt text changed.')
            pair['subset'] = 'original'
            for role in models:
                pair['members'][role] = {**pair['members'][role], 'finished': True}
                generated[role] += 1
        else:
            members, hashes = {}, {}
            for role, run in config['added_runs'].items():
                result, path = live_result(root, run, sid)
                if result and (result['prompt'] != spec['prompt'] or result['seed'] != spec['seed']
                        or result['model'].removeprefix('openai-api/artsy/') != models[role]):
                    raise ValueError('Generated output does not match the expected prompt/model.')
                members[role] = {'outcome': result['outcome'] if result else 'pending',
                    'finished': result is not None,
                    'image_url': f'/api/comparisons/{quote(name)}/images/{quote(sid)}/{role}' if path else None,
                    'scores': None}
                hashes[role] = result.get('imageHash') if path else None
                generated[role] += int(result is not None)
            pair = {'id': sid, 'prompt': spec['prompt'], 'subset': 'added', 'members': members,
                    'preferred': None, 'comparison': None, 'labels': None}
            judgment = added_judgments.get(sid)
            if judgment and judgment['prompt'] == spec['prompt'] and all(
                    members[role]['finished'] and
                    added_inputs[sid][role]['image_sha256'] == hashes[role] for role in models):
                for role in models:
                    members[role]['scores'] = judgment['members'][role]['scores']
                pair.update({k: judgment[k] for k in ['preferred', 'comparison', 'labels']})
        pairs.append(pair)
    return {'id': name, 'label': config['label'], 'combined': True, 'models': models,
            'judge': original['judge'], 'pairs': pairs, 'progress': generated,
            'subsets': [{'id': 'original', 'label': 'Original prompts', 'count': len(original_by_id)},
                        {'id': 'added', 'label': 'Added prompts', 'count': len(prompts) - len(original_by_id)}]}
