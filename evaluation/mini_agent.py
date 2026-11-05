"""Run mini-SWE-agent with Inspect model accounting and sandbox tools."""

import json
import hashlib
from time import monotonic

from inspect_ai.agent import agent
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser,
                              ContentImage, ContentText, Model, execute_tools, get_model)
from inspect_ai.tool import ToolCall, ToolResult, tool
from inspect_ai.util import sandbox

from drawing.mini_agent import CONTRACT_VERSION, VERSION, run_agent
from drawing.mini_session import DrawingSession, READ_SOURCE
from .host_agent import FilteredAPI, activity_result


def inspect_message(message):
    content = message.get('content') or ''
    if isinstance(content, list):
        content = [ContentImage(image=p['image_url']['url']) if p['type'] == 'image_url'
                   else ContentText(text=p['text']) for p in content]
    role = message['role']
    if role == 'assistant':
        calls = [ToolCall(id=c['id'], function=c['function']['name'],
                         arguments=json.loads(c['function']['arguments'])
                         if isinstance(c['function']['arguments'], str) else c['function']['arguments'])
                 for c in message.get('tool_calls') or []]
        return ChatMessageAssistant(content=content, tool_calls=calls or None)
    if role == 'tool':
        return ChatMessageTool(content=content, tool_call_id=message['tool_call_id'], function='bash')
    return (ChatMessageSystem if role == 'system' else ChatMessageUser)(content=content)



@agent(name='artsy-mini-swe')
def mini_drawing_agent(config, generation_filter=None, max_turns=25, progress=None, approval=None):
    async def run(state):
        from inspect_ai.solver._task_state import sample_state
        sample = sample_state()
        metadata = sample.metadata if sample is not None else {}
        seed = metadata.get('seed', 42)
        metadata.update(submitted=False, trajectoryContractVersion=CONTRACT_VERSION,
                        harness='mini-swe-agent', harnessVersion=VERSION, maxTurns=max_turns)
        timing = {'model_seconds': 0., 'command_seconds': 0., 'model_calls': 0}
        metadata['harnessTiming'] = timing
        started = monotonic()
        prompt = state.messages[-1].text
        reference = metadata.get('reference_source')
        if reference is not None:
            await sandbox().write_file('/workspace/program.py', reference)
            prompt = ('Edit the existing drawing. Preserve everything unless the request asks for a change.\n\n'
                      + metadata['prompt'] + '\n\nExisting source is available as program.py.')

        async def shell(command):
            result = await sandbox().exec(['timeout', '--signal=KILL', '65', 'bash', '-lc', command],
                cwd='/workspace', env={'ARTSY_SEED': str(seed)}, timeout=67, timeout_retry=False)
            return {'returncode': result.returncode, 'output': result.stdout + '\n' + result.stderr}

        async def read_source(entrypoint=None):
            result = await sandbox().exec(['python', '-c', READ_SOURCE] + ([entrypoint] if entrypoint else []), timeout=10, timeout_retry=False)
            if not result.success:
                raise ValueError((result.stderr or result.stdout)[-2048:])
            return json.loads(result.stdout)

        session = DrawingSession(shell, read_source)
        last_result = {}

        @tool(name='bash', parallel=False)
        def bash():
            async def execute(command: str) -> ToolResult:
                """Execute a bash command.

                Args:
                    command: The bash command to execute.
                """
                nonlocal last_result
                turn = timing['model_calls']
                step = {'id': f'action-{session.calls + 1}', 'kind': 'tool', 'tool': 'bash',
                        'title': 'Editing drawing', 'arguments': {'command': command},
                        'turn': turn, 'status': 'running'}
                if approval:
                    if progress:
                        progress({'phase': 'awaiting_approval', 'step': {**step, 'status': 'pending'},
                                  'turn': turn, 'max_turns': max_turns})
                    await approval(step['id'])
                if progress:
                    progress({'phase': 'drawing', 'step': step, 'turn': turn, 'max_turns': max_turns})
                began = monotonic()
                last_result = await session.execute(command)
                timing['command_seconds'] += monotonic() - began
                content = inspect_message({'role': 'tool', 'tool_call_id': 'result',
                                           'content': last_result['content']}).content
                if progress:
                    progress({'phase': 'drawing', 'step': {**step, **activity_result(content)},
                              'turn': turn, 'max_turns': max_turns})
                return content
            return execute

        tools = [bash()]
        model = get_model()
        if generation_filter:
            model = Model(FilteredAPI(model, generation_filter), config=config)

        async def complete(messages):
            state.messages = [inspect_message(m) for m in messages]
            turn = timing['model_calls'] + 1
            if progress:
                progress({'phase': 'thinking', 'turn': turn, 'max_turns': max_turns})
            began = monotonic()
            try:
                output = await model.generate(state.messages, tools=tools, config=config)
            finally:
                timing['model_seconds'] += monotonic() - began
                timing['model_calls'] += 1
            state.output = output
            state.messages.append(output.message)
            return {'text': output.message.text,
                    'finishReason': 'length' if any(c.stop_reason == 'max_tokens' for c in output.choices) else None,
                    'tool_calls': [{'id': c.id, 'type': 'function', 'function': {'name': c.function,
                                    'arguments': c.arguments}} for c in output.message.tool_calls or []]}

        async def execute(command):
            results, _ = await execute_tools(state.messages, tools)
            state.messages.extend(results)
            if not results or getattr(results[0], 'error', None):
                raise RuntimeError('Inspect could not execute mini bash action')
            return last_result

        try:
            trajectory = await run_agent(complete, execute, prompt, seed, max_turns=max_turns or 0)
            state.messages = [inspect_message(m) for m in trajectory['messages'] if m['role'] != 'exit']
        finally:
            metadata.update(submitted=session.submitted, submittedSourceHash=hashlib.sha256(session.source.encode()).hexdigest() if session.submitted else None,
                            submittedSource=session.source if session.submitted else None,
                            submittedEntrypoint=session.entrypoint,
                            environmentCalls=session.calls,
                            turnLimitReached=bool(max_turns and timing['model_calls'] >= max_turns and not session.submitted))
            timing['agent_seconds'] = monotonic() - started
        return state
    return run
