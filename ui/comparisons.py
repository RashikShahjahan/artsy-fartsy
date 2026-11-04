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
    return found


def packet(root, name):
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
        for role in ('base', 'sft'):
            member = item['members'][role]
            members[role] = {'outcome': member['outcome'],
                'image_url': f'/api/comparisons/{quote(name)}/images/{quote(sid)}/{role}'
                    if member.get('image') else None,
                'scores': record['scores'][role] if record else None}
        pairs.append({'id': sid, 'prompt': item['prompt'], 'members': members,
                      'preferred': record['preferred'] if record else None,
                      'comparison': record['comparison'] if record else None,
                      'labels': record['label_mapping'] if record else None})
    return {'id': name, 'models': dict(zip(('base', 'sft'), data['source_models'])),
            'judge': data['model'], 'pairs': pairs}


def image(root, name, sid, role):
    _, data = manifest(root, name)
    pair = next((p for p in data['pairs'] if p['id'] == sid), None)
    if pair is None or role not in ('base', 'sft'):
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
