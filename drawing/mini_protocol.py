"""Shared, framework-independent workspace snapshots and shell event framing."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

EVENT_PREFIX = '<ARTSY_EVENT>'
EVENT_SUFFIX = '</ARTSY_EVENT>'
COMPLETE = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
SELECTION_FILE = '/tmp/artsy-agent-selection.json'


def source_hash(source):
    return hashlib.sha256(source.encode()).hexdigest() if source else None


def selection_file():
    return Path(os.getenv('ARTSY_SELECTION_FILE', SELECTION_FILE))


def snapshot(entrypoint=None, cwd=None):
    cwd = Path(cwd or Path.cwd()).resolve()
    if not entrypoint:
        try:
            entrypoint = json.loads(selection_file().read_text())['path']
        except (OSError, ValueError, KeyError, TypeError):
            candidates = sorted(p for p in cwd.glob('*.py') if p.is_file())
            if len(candidates) != 1:
                raise ValueError('Choose a Python entrypoint with artsy-submit PATH.')
            entrypoint = candidates[0]
    path = Path(entrypoint)
    path = (path if path.is_absolute() else cwd / path).resolve()
    if path.suffix != '.py' or not path.is_file():
        raise ValueError('Submission requires an existing Python entrypoint.')
    root = cwd if path.is_relative_to(cwd) else path.parent
    files, total = {}, 0
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', '.git', '.venv')
                         and not Path(current, d).is_symlink())
        for name in sorted(names):
            if not name.endswith('.py'):
                continue
            p = Path(current, name)
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as f:
                info = os.fstat(f.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                    raise ValueError('Expected regular Python files of at most 64 KiB.')
                data = f.read(65537)
            total += len(data)
            if total > 65536 or len(files) >= 128:
                raise ValueError('Python workspace is too large.')
            files[p.relative_to(root).as_posix()] = data.decode('utf-8')
    result = {'entrypoint': path.relative_to(root).as_posix(), 'files': files,
              'path': str(path), 'root': str(root)}
    bundle_source(result)  # Validate the final portable program's size too.
    return result


def remember(bundle):
    selection_file().write_text(json.dumps({'path': bundle['path']}))


def bundle_source(bundle):
    files, entrypoint = bundle['files'], bundle['entrypoint']
    if not isinstance(files, dict) or not files or entrypoint not in files:
        raise ValueError('Choose an existing Python entrypoint with artsy-submit PATH.')
    for name, source in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not name.endswith('.py') or not isinstance(source, str):
            raise ValueError('Invalid Python workspace snapshot')
    payload = json.dumps(files, sort_keys=True, ensure_ascii=False)
    code = ("import json as _json, pathlib as _pathlib, runpy as _runpy, sys as _sys\n"
            f"_files = _json.loads({payload!r})\n"
            "for _name, _source in _files.items():\n"
            "    _p = _pathlib.Path(_name)\n"
            "    _p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    _p.write_text(_source, encoding='utf-8')\n"
            f"_entry = {entrypoint!r}\n"
            "_sys.path.insert(0, str(_pathlib.Path(_entry).parent.resolve()))\n"
            "_sys.path.insert(0, str(_pathlib.Path.cwd()))\n"
            "_runpy.run_path(_entry, run_name='__main__')\n")
    if len(code.encode()) > 65536:
        raise ValueError('Submitted Python workspace exceeds the 64 KiB source limit.')
    return code


def emit(event):
    print('\n' + EVENT_PREFIX + json.dumps(event, separators=(',', ':')) + EVENT_SUFFIX, flush=True)


def read_events(output):
    """Extract bounded machine events before truncating ordinary shell feedback."""
    events, text = [], []
    for line in output.splitlines():
        if line.startswith(EVENT_PREFIX) and line.endswith(EVENT_SUFFIX):
            try:
                event = json.loads(line[len(EVENT_PREFIX):-len(EVENT_SUFFIX)])
                if not isinstance(event, dict) or event.get('kind') not in ('preview', 'submit'):
                    raise ValueError('Unknown event')
                events.append(event)
            except (ValueError, TypeError):
                text.append('[Invalid Artsy command event]')
        else:
            text.append(line)
    return events, '\n'.join(text)


def export_workspace(roots=('/workspace', '/tmp', '/home/artsy')):
    """Capture generated files, including binary outputs, for interruption recovery."""
    import base64
    entries, total = {}, 0
    for root_name in roots:
        root = Path(root_name)
        if not root.is_dir():
            continue
        for current, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', '.cache', '.git', '.venv')
                             and not Path(current, d).is_symlink())
            for name in sorted(names):
                path = Path(current, name)
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, 'rb') as file:
                        info = os.fstat(file.fileno())
                        if not stat.S_ISREG(info.st_mode):
                            continue
                        if info.st_size > 8 * 1024 * 1024:
                            raise ValueError(f'Recovery file exceeds 8 MiB: {path}')
                        data = file.read(8 * 1024 * 1024 + 1)
                except (FileNotFoundError, PermissionError):
                    continue
                total += len(data)
                if total > 32 * 1024 * 1024 or len(entries) >= 1024:
                    raise ValueError('Recovery snapshot exceeds its size or file-count limit.')
                entries[str(path)] = {'data': base64.b64encode(data).decode(), 'mode': stat.S_IMODE(info.st_mode),
                                      'sha256': hashlib.sha256(data).hexdigest()}
    return {'version': 1, 'files': entries}


def restore_workspace(checkpoint, roots=('/workspace', '/tmp', '/home/artsy')):
    import base64
    if checkpoint.get('version') != 1 or not isinstance(checkpoint.get('files'), dict):
        raise ValueError('Unsupported workspace checkpoint.')
    allowed = [Path(p).resolve() for p in roots]
    total = 0
    for name, entry in checkpoint['files'].items():
        path = Path(name)
        if not path.is_absolute() or not any(path.resolve().is_relative_to(root) for root in allowed):
            raise ValueError('Recovery path is outside the generated workspace.')
        data = base64.b64decode(entry['data'], validate=True)
        total += len(data)
        if len(data) > 8 * 1024 * 1024 or total > 32 * 1024 * 1024:
            raise ValueError('Recovery snapshot is too large.')
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Recovery artifact hash mismatch.')
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as file:
            file.write(data)
        path.chmod(entry.get('mode', 0o600) & 0o777)
