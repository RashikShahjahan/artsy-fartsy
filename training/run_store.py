"""Crash-safe local publication and exclusive ownership of a training output."""
import fcntl
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager


@contextmanager
def run_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Output directory is already owned by another training process: {output_dir}') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_text(path: Path, text: str):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value):
    atomic_text(path, json.dumps(value, indent=2) + '\n')


def contents(output_dir: Path):
    return [p for p in output_dir.iterdir() if p.name != '.run.lock']


def recover_jsonl_tail(path: Path, validate_records):
    """Validate committed records, then preserve and remove a torn final fragment.

    Caller must hold the output run lock. Complete/interior corruption is fatal.
    """
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records = []
    fragment = None
    committed = data
    for index, line in enumerate(lines):
        terminated = line.endswith(b"\n")
        if not terminated:
            if index != len(lines) - 1:
                raise ValueError("Invalid checkpoint history")
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError):
                fragment = line
                committed = data[:-len(line)]
                break
            raise ValueError("Checkpoint history has an unterminated valid record")
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Invalid complete checkpoint record") from error
        if not isinstance(record, dict):
            raise ValueError("Invalid checkpoint record")
        records.append(record)
    if not records:
        raise ValueError("Empty checkpoint history")
    validate_records(records)
    if fragment is not None:
        fd, backup = tempfile.mkstemp(prefix=path.name + '.torn-', suffix='.bak', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as out:
                out.write(fragment)
                out.flush()
                os.fsync(out.fileno())
            atomic_text(path, committed.decode('utf-8'))
        except BaseException:
            # Preserve any successfully created backup for inspection.
            raise
    return records
