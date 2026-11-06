"""mini-SWE-agent's loop with asynchronous Artsy model and sandbox adapters."""

import asyncio
import copy
import contextvars
import json
import os
import threading
from types import SimpleNamespace
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import suppress

from .agent_contract import ROOT

# Use a project-local config directory; the adapters use Artsy's credentials,
# and mini must not print a startup banner into CLI JSON output.
os.environ.setdefault('MSWEA_GLOBAL_CONFIG_DIR', str(ROOT / '.cache' / 'mini-swe-agent'))
os.environ.setdefault('MSWEA_SILENT_STARTUP', '1')

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions

VERSION = '2.2.3'
FORMAT = 'mini-swe-bash-v1'
TOOLS = [BASH_TOOL]
SUBMIT = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
PREVIEW = 'artsy-preview'
CONTRACT_VERSION = 8


def system_prompt(seed):
    # DefaultAgent receives templates explicitly; its constructor does not load mini.yaml.
    from importlib.resources import files
    import yaml
    settings = yaml.safe_load(files('minisweagent').joinpath('config/mini.yaml').read_text())
    return settings['agent']['system_template'].strip()


def task_prompt(prompt):
    return f"""Please create this drawing:

{prompt}

Create the drawing using pycairo and save a 1920×1200 PNG. Do not add text to the image unless requested.
Save to the relative path output.png in the current working directory. Final validation reruns your program in a fresh directory.
If your drawing uses randomness, call `random.seed(42)` before the first random draw. Explicitly seed any separate random number generators as well so both runs produce the same image.

## Recommended Workflow

1. Your first tool call must only inspect the installed pycairo API reference at `/opt/artsy/pycairo-api.pyi`. Search for the methods you plan to use and wait for the tool result.
2. Use the returned signatures when you write the drawing code in a later tool call.
3. Run the Python entrypoint you intend to submit with `rm -f output.png && python ENTRYPOINT` and wait for the result. If execution fails, fix the error and rerun the script.
4. After the script succeeds, use a separate tool call, `artsy-preview output.png`, to view the generated image. If you edit the source, run it again and view the new PNG before submitting.
5. Once you have viewed the image from the current source and it meets the request, submit that entrypoint. Submission is mandatory.

## Command Execution Rules

- Use exactly one bash tool call per response, with a single string argument in the form `{{"command": "..."}}`.
- Each command runs in a fresh subshell. Files persist, while directory and environment changes do not.
- Submit with `artsy-submit ENTRYPOINT`. Run the submission command by itself. A successful submission ends the task."""


async def run_agent(complete, execute, prompt, seed, *, max_turns, progress=None, output_path=None, resume=None):
    """Keep mini's parser, loop, limits and serialization; bridge I/O to asyncio.

    complete receives OpenAI-style messages and returns our existing client result.
    execute returns content parts plus returncode, and terminal=True on submission.
    Both callbacks run on the caller's event loop (including Inspect's context).
    """
    loop = asyncio.get_running_loop()
    stopped = threading.Event()
    pending = set()
    lock = threading.Lock()

    def call(coro):
        with lock:
            if stopped.is_set():
                coro.close()
                raise CancelledError()
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            pending.add(future)
        try:
            return future.result()
        finally:
            with lock:
                pending.discard(future)

    class Model:
        def query(self, messages):
            response = call(complete(copy.deepcopy(messages)))
            text = '\n'.join(s for s in (response.get('reasoning'), response.get('text')) if s)
            calls = copy.deepcopy(response.get('tool_calls') or [])
            for tool in calls:
                args = tool['function']['arguments']
                if isinstance(args, dict):
                    tool['function']['arguments'] = json.dumps(args)
            message = {'role': 'assistant', 'content': text, 'tool_calls': calls,
                       'extra': {'cost': 0, 'usage': response.get('usage', {})}}
            try:
                if response.get('finishReason') == 'length':
                    raise ValueError('Reply was truncated; use a shorter edit.')
                if len(calls) != 1:
                    raise ValueError('Use exactly one bash tool call per response.')
                args = json.loads(calls[0]['function']['arguments'])
                if not isinstance(args, dict) or set(args) != {'command'} or not isinstance(args['command'], str):
                    raise ValueError('bash requires one string command argument.')
                parsed = [SimpleNamespace(id=c['id'], function=SimpleNamespace(**c['function'])) for c in calls]
                message['extra']['actions'] = parse_toolcall_actions(parsed, format_error_template='{{error}}')
            except (ValueError, TypeError, FormatError) as error:
                # Preserve the original invalid calls in the recording, while
                # keeping the next API request free of unanswered tool calls.
                message['extra']['invalid_tool_calls'] = message.pop('tool_calls')
                # Malformed replies still consume a model turn.
                raise FormatError(message, {'role': 'user', 'content': str(error) or 'Invalid tool call; use bash with a command string.'}) from error
            return message

        def format_message(self, **kwargs):
            return kwargs

        def format_observation_messages(self, message, outputs, template_vars=None):
            return [{'role': 'tool', 'tool_call_id': action['tool_call_id'],
                     'content': result['content'], 'extra': {k: v for k, v in result.items() if k != 'content'}}
                    for action, result in zip(message['extra']['actions'], outputs)]

        def serialize(self):
            return {'info': {'model': 'artsy-async-adapter'}}

        def get_template_vars(self):
            return {}

    class Environment:
        terminal = False

        def execute(self, action):
            result = call(execute(action['command']))
            self.terminal = result.get('terminal', False)
            self.entrypoint = result.get('entrypoint')
            return result

        def serialize(self):
            return {'info': {'environment': 'artsy-sandbox'}}

        def get_template_vars(self):
            return {}

    class DrawingAgent(DefaultAgent):
        def add_messages(self, *messages):
            if resume and not self.messages:
                return super().add_messages(*copy.deepcopy(resume['messages']))
            return super().add_messages(*messages)

        def query(self):
            if progress:
                call(progress(self.n_calls + 1))
            return super().query()

        def execute_actions(self, message):
            results = super().execute_actions(message)
            if self.env.terminal:
                raise Submitted({'role': 'exit', 'content': 'submitted',
                                 'extra': {'exit_status': 'Submitted', 'submission': self.env.entrypoint}})
            return results

    agent = DrawingAgent(Model(), Environment(), system_template='{{instructions}}',
                         instance_template='{{task}}', step_limit=max_turns, cost_limit=0,
                         output_path=output_path)
    if resume:
        agent.n_calls = resume['model_calls']
    # A dedicated coordinator thread avoids consuming the executor slots used
    # by Docker I/O and rendering when many episodes run concurrently.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='artsy-mini')
    context = contextvars.copy_context()
    def coordinate():
        try:
            return agent.run(task_prompt(prompt), instructions=system_prompt(seed))
        except (CancelledError, asyncio.CancelledError):
            if not stopped.is_set():
                raise
            return None

    task = asyncio.wrap_future(executor.submit(context.run, coordinate))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with lock:
            stopped.set()
            for future in pending:
                future.cancel()
        with suppress(CancelledError, asyncio.CancelledError, Exception):
            await task
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return agent.serialize()
