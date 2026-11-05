"""Collect actual mini-SWE-agent editing trajectories for experiment 5."""

import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading

from .agent_contract import SFT_MAX_TURNS
from .dataset_specs import sha256
from .environment import DrawingEnvironment
from .mini_agent import FORMAT, TOOLS, VERSION, run_agent
from .mini_session import DrawingSession, READ_SOURCE
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import LimitsExceeded
from .tool_dataset import trajectory_hash


class Workspace(DockerEnvironment):
    # Completion must be an exact action checked by DrawingSession. Arbitrary
    # program stdout must never terminate an episode.
    def _check_finished(self, output):
        pass

    def cleanup(self):
        if getattr(self, 'container_id', None):
            subprocess.run([self.config.executable, 'rm', '-f', self.container_id],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            self.container_id = None

    def execute(self, action, **kwargs):
        # Bound captured logs on the host, including a command that floods stdout.
        cmd = [self.config.executable, 'exec', '-i', '-w', '/workspace', '-e', f'ARTSY_SEED={getattr(self, "seed", 0)}', self.container_id,
               'timeout', '--signal=KILL', '65', 'bash', '-lc', action['command']]
        limit = kwargs.get('limit', 16 * 1024 * 1024)
        captured = bytearray()
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE if 'stdin' in action else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def drain():
            with process.stdout:
                while data := process.stdout.read(8192):
                    remaining = limit + 1 - len(captured)
                    if remaining > 0:
                        captured.extend(data[:remaining])

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        if 'stdin' in action:
            try:
                process.stdin.write(action['stdin'].encode())
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            code = process.wait(timeout=70)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = 124
        reader.join(timeout=2)
        return {'returncode': code, 'output': bytes(captured[:limit]).decode(errors='replace') +
                ('\n[output truncated]' if len(captured) > limit else '')}



def provider_messages(messages):
    return [{k: v for k, v in m.items() if k != 'extra'} for m in messages]


def inspect_image():
    image = os.getenv('ARTSY_AGENT_IMAGE', 'artsy-editing-agent:latest')
    try:
        result = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                                capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('Start Docker and build the editing image: docker build -f '
                           'evaluation/Dockerfile -t artsy-editing-agent:latest .') from error
    image_id = result.stdout.strip()
    if not image_id.startswith('sha256:'):
        raise RuntimeError('Docker did not return an immutable editing image ID')
    return {'name': image, 'id': image_id}


async def candidate(options, spec, index, api_doc):
    from .dataset import DATASET_VERSION, _default_config, _pixel_hash, _provenance

    root = Path(options['output']).resolve()
    env = DrawingEnvironment(root / 'artifacts', renderer=options.get('renderer'),
                             config=options.get('config') or _default_config())
    episode = await env.reset(spec['prompt'], spec['seed'])
    directory = Path(episode['directory'])
    rel = lambda p: os.path.relpath(p, root)
    record = {'version': DATASET_VERSION, 'id': f"{spec['id']}-c{index}", 'spec': spec,
              'candidate': index, 'trajectoryFormat': FORMAT, 'status': 'rejected',
              'reason': f'No valid submission within {SFT_MAX_TURNS} turns',
              'trace': [], 'evaluations': [], 'teacher': _provenance(options['teacher']),
              'runtimeVersion': env.config['runtimeVersion'], 'apiDocHash': sha256(api_doc),
              'humanReviewed': False, 'artifacts': {'directory': rel(directory),
                  'trajectory': rel(directory / 'mini.traj.json')},
              'generation': {'harness': 'mini-swe-agent', 'harnessVersion': VERSION,
                  'initialDraft': 'natural', 'executionFailures': 0, 'imagePreviews': True,
                  'modelTurns': 0, 'maxTurns': SFT_MAX_TURNS}}
    resumed = options.get('resume')
    if resumed:
        record['trace'] = copy.deepcopy(resumed['record']['trace'])
        record['generation']['modelTurns'] = len(record['trace'])
        record['generation']['continuation'] = resumed['provenance']
    workspace = None
    try:
        workspace = await asyncio.to_thread(Workspace,
            image=options.get('agentImage') or os.getenv('ARTSY_AGENT_IMAGE', 'artsy-editing-agent:latest'), cwd='/workspace',
            run_args=['--rm', '--pull=never', '--network=none', '--cap-drop=ALL',
                      '--security-opt=no-new-privileges', '--pids-limit=128', '--memory=512m', '--cpus=1'],
            timeout=70)
        workspace.seed = spec['seed']
        if resumed:
            checkpoint_path = resumed['record'].get('artifacts', {}).get('workspaceCheckpoint')
            if checkpoint_path:
                checkpoint_file = (root / checkpoint_path).resolve()
                if not checkpoint_file.is_relative_to(root):
                    raise ValueError('Checkpoint is outside the dataset')
                checkpoint = json.loads(checkpoint_file.read_text())
                restore = (Path(__file__).with_name('mini_protocol.py').read_text()
                           + '\nrestore_workspace(json.loads(' + repr(json.dumps(checkpoint)) + '))')
            else:
                restore = ('import json,pathlib; files=json.loads(' + repr(json.dumps(resumed['files']))
                           + '); [(pathlib.Path(p).parent.mkdir(parents=True,exist_ok=True), '
                             'pathlib.Path(p).write_text(s)) for p,s in files.items()]')
            # Stream the payload on stdin: binary checkpoints may exceed argv limits.
            restored = await asyncio.to_thread(workspace.execute,
                {'command': 'python', 'stdin': restore})
            if restored['returncode']:
                raise RuntimeError('Could not restore generated workspace files')

        async def shell(command):
            result = await asyncio.to_thread(workspace.execute, {'command': command})
            return result

        async def read_source(entrypoint=None):
            result = await asyncio.to_thread(workspace.execute,
                {'command': 'python -c ' + shlex.quote(READ_SOURCE) +
                 (' ' + shlex.quote(entrypoint) if entrypoint else '')}, limit=512 * 1024)
            if result['returncode']:
                raise ValueError(result['output'][-2048:])
            return json.loads(result['output'])

        session = DrawingSession(shell, read_source, max_calls=env.config['maxCalls'])

        async def complete(messages):
            try:
                response = await options['teacher'].complete(provider_messages(messages),
                    options.get('maxTokens'), tools=TOOLS)
            except RuntimeError as error:
                if str(error) != 'Chat endpoint returned no text completion':
                    raise
                # An empty provider reply consumes a turn. Let mini emit its
                # normal missing-tool correction instead of losing the episode.
                response = {'text': '', 'finishReason': 'empty', 'providerError': str(error)}
            record['generation']['modelTurns'] += 1
            record['trace'].append({'completion': response})
            charged = await env.charge_tokens(response.get('usage', {}).get('completion_tokens', 0))
            if not charged['ok']:
                raise LimitsExceeded({'role': 'exit', 'content': 'Episode token budget exhausted',
                                     'extra': {'exit_status': 'LimitsExceeded', 'submission': ''}})
            return response

        async def execute(command):
            result = await session.execute(command)
            if record['trace']:
                record['trace'][-1]['previewEvents'] = result.get('previewEvents', [])
                for feedback in result.get('previewEvents', []):
                    if feedback.get('sourceHash'):
                        record['trace'][-1].update(feedback=feedback, executedCodeHash=feedback['sourceHash'])
                    if feedback.get('ok') is False:
                        record['generation']['executionFailures'] += 1
            return result

        async with asyncio.timeout(env.config['episodeMs'] / 1000):
            trajectory = await run_agent(complete, execute, spec['prompt'], spec['seed'],
                max_turns=SFT_MAX_TURNS, output_path=directory / 'mini.traj.json',
                resume=resumed['trajectory'] if resumed else None)
            if session.submitted:
                # Submission snapshots all Python modules, regardless of previews.
                written = await env.write_code(session.source)
                feedback = await env.submit() if written['ok'] else written
                if feedback['ok']:
                    png = Path(feedback['artifact']).read_bytes()
                    record.update(status='review', reason='Submitted drawing; human review pending',
                                  code=env.source, sourceHash=sha256(env.source.encode()),
                                  imageHash=sha256(png), pixelHash=_pixel_hash(png),
                                  submission={'feedback': feedback, 'sourceHash': sha256(session.source.encode()),
                                              'entrypoint': session.entrypoint}, sourceBundle=session.bundle)
                    record['artifacts'].update(source=rel(directory / 'program.py'), image=rel(feedback['artifact']))
                else:
                    record['reason'] = feedback.get('error', 'Final rendering failed')
        # Persist images from real shell commands, including compound commands.
        image_number = 0
        for message in trajectory['messages']:
            if isinstance(message.get('content'), list):
                for part in message['content']:
                    if part.get('type') != 'image_url':
                        continue
                    image_number += 1
                    png = base64.b64decode(part['image_url']['url'].split(',', 1)[1], validate=True)
                    path = directory / f'preview-{image_number}.png'
                    path.write_bytes(png)
                    part.clear()
                    part.update(type='image', image=rel(path), imageHash=sha256(png))
        (directory / 'mini.traj.json').write_text(json.dumps(trajectory, indent=2) + '\n')
        record['miniTrajectory'] = trajectory
        record['miniTrajectoryHash'] = trajectory_hash(trajectory)
    except TimeoutError:
        record['reason'] = 'Episode deadline exceeded'
    except Exception as exc:
        record.update(status='infrastructure_error', reason=str(exc))
    finally:
        if workspace:
            if record['status'] != 'review':
                try:
                    export = Path(__file__).with_name('mini_protocol.py').read_text() + '\nprint(json.dumps(export_workspace()))'
                    checkpoint = await asyncio.to_thread(workspace.execute,
                        {'command': 'python', 'stdin': export}, limit=48 * 1024 * 1024)
                    if checkpoint['returncode']:
                        raise RuntimeError(checkpoint['output'][-2048:])
                    data = json.loads(checkpoint['output'])
                    checkpoint_path = directory / 'workspace-checkpoint.json'
                    checkpoint_path.write_text(json.dumps(data))
                    record['artifacts']['workspaceCheckpoint'] = rel(checkpoint_path)
                except Exception as error:
                    record['checkpointError'] = str(error)
            await asyncio.to_thread(workspace.cleanup)
    return record


def resume_options(record, root):
    """Recover a checkpointed interruption without resetting the turn budget."""
    paths = record.get('artifacts', {})
    if not paths.get('workspaceCheckpoint') or not paths.get('trajectory'):
        return None
    root = Path(root).resolve()
    for name in ('workspaceCheckpoint', 'trajectory'):
        path = (root / paths[name]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or invalid interrupted workspace artifact')
    raw = json.loads((root / paths['trajectory']).read_text())
    calls = raw.get('info', {}).get('model_stats', {}).get('api_calls')
    if type(calls) is not int or calls < 0:
        raise ValueError('Interrupted trajectory has no valid model-call budget')
    messages = [m for m in raw['messages'] if m.get('role') != 'exit']
    messages.append({'role': 'user', 'content': 'The connection was interrupted. Your generated files have been restored; continue the task.'})
    return {'record': record, 'trajectory': {'messages': messages, 'model_calls': calls},
            'provenance': {'reason': record.get('reason'), 'checkpoint': paths['workspaceCheckpoint'],
                           'interruptedTrajectory': paths['trajectory'],
                           'tracePrefixTurns': len(record.get('trace', [])), 'requestsBeforeResume': calls}}
