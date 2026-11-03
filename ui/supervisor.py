"""Own UI child processes and terminate their process groups on cancellation."""
import os
import signal
import subprocess
import threading


class Supervisor:
    def __init__(self, grace_seconds=5):
        self.lock = threading.RLock()
        self.processes = {}
        self.reservations = set()
        self.grace_seconds = grace_seconds

    def reserve(self, ident):
        with self.lock:
            if ident in self.reservations or ident in self.processes:
                raise RuntimeError('Process already reserved.')
            self.reservations.add(ident)

    def release(self, ident):
        with self.lock:
            self.reservations.discard(ident)
            self.processes.pop(ident, None)

    def spawn(self, ident, argv, **kwargs):
        with self.lock:
            if ident in self.processes:
                raise RuntimeError('Process already running.')
            process = subprocess.Popen(argv, start_new_session=(os.name == 'posix'), **kwargs)
            self.reservations.discard(ident)
            self.processes[ident] = process
            return process

    def active(self, ident):
        with self.lock:
            process = self.processes.get(ident)
            return ident in self.reservations or process is not None

    def get(self, ident):
        with self.lock:
            return self.processes.get(ident)

    def cancel(self, ident):
        with self.lock:
            process = self.processes.get(ident)
            if process is None or process.poll() is not None:
                raise ValueError('This run is no longer controlled by the current UI server.')
            self._signal(process, False)
            threading.Thread(target=self._escalate, args=(process,), daemon=True).start()

    @staticmethod
    def _signal(process, force):
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    def _escalate(self, process):
        try:
            process.wait(timeout=self.grace_seconds)
        except subprocess.TimeoutExpired:
            self._signal(process, True)


def alive(pid):
    if type(pid) is not int or pid < 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


SUPERVISOR = Supervisor()
