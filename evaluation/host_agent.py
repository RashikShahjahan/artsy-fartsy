"""Host-side drawing policy with structured tools and sandboxed rendering."""

import base64
import json
from pathlib import Path
import re
from time import monotonic

from inspect_ai.agent import AgentState, agent
from inspect_ai.model import (
    ChatMessageSystem, ChatMessageUser, ContentImage, ContentText, GenerateInput,
    Model, ModelAPI, ModelOutput, execute_tools, get_model, modelapi,
)
from inspect_ai.tool import ToolResult, tool
from inspect_ai.util import sandbox
import yaml

from drawing.config import LIMITS
from drawing.agent_contract import (DRAWING_TOOLS_CONTRACT_VERSION, host_user_prompt,
                                    turn_countdown, turn_limit_instruction)

AGENT_NAME = "artsy-host"
AGENT_VERSION = "3"
MAX_IMAGE_BYTES = 8 * 1024 * 1024


@modelapi(name="artsy-filter")
class FilteredAPI(ModelAPI):
    """Run native rollout filters through Inspect's normal timing/usage path."""

    def __init__(self, model, callback):
        super().__init__(model.name, base_url=model.api.base_url)
        self.model, self.callback = model, callback

    async def generate(self, input, tools, tool_choice, config):
        result = await self.callback(self.model, input, tools, tool_choice, config)
        if isinstance(result, ModelOutput):
            return result
        if isinstance(result, GenerateInput):
            input, tools, tool_choice, config = result
        return await self.model.api.generate(input, tools, tool_choice, config)


def observation(result, image_pattern):
    """Keep preview PNGs as image content instead of truncating base64 as text."""
    images = []
    for match in image_pattern.finditer(result.stdout):
        url = match.group(2)
        if len(url) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 32:
            raise ValueError("Preview image exceeds 8 MiB")
        data = base64.b64decode(url.split(",", 1)[1], validate=True)
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Preview must contain PNG data of at most 8 MiB")
        images.append(ContentImage(image=url))
    text = image_pattern.sub("", result.stdout)
    text = f"Return code: {result.returncode}\n{text}"
    if result.stderr:
        text += "\n" + result.stderr
    return [ContentText(text=text), *images] if images else text


def activity_result(content):
    """Expose recorded tool output and previews, including nested render failures."""
    parts = content if isinstance(content, list) else [ContentText(text=content)]
    text = '\n'.join(part.text for part in parts if isinstance(part, ContentText))
    failed = any(int(code) != 0 for code in re.findall(r'^Return code:\s*(-?\d+)', text, re.M))
    for line in text.splitlines():
        try:
            feedback = json.loads(line)
        except ValueError:
            continue
        if isinstance(feedback, dict):
            failed |= feedback.get('ok') is False or feedback.get('exitCode', 0) != 0
    return {'status': 'failed' if failed else 'completed', 'output': text,
            'previews': [part.image for part in parts if isinstance(part, ContentImage)]}


@agent(name=AGENT_NAME)
def drawing_agent(system_prompt, config, generation_filter=None, max_calls=LIMITS["maxCalls"],
                  max_turns=None, progress=None, approval=None):
    """Expose drawing operations directly, without a model-facing shell."""
    settings = yaml.safe_load(Path(__file__).with_name("mini.yaml").read_text())
    image_pattern = re.compile(settings["model"]["multimodal_regex"])
    timeout = settings["environment"]["timeout"]
    turn_limit_instruction(max_turns)  # Validate before starting the agent.

    async def run(state: AgentState):
        began = monotonic()
        timing = {"model_seconds": 0.0, "command_seconds": 0.0, "model_calls": 0}
        # AgentState shares sample metadata with the scorer and Inspect transcript.
        from inspect_ai.solver._task_state import sample_state
        sample = sample_state()
        if sample is not None:
            sample.metadata["harnessTiming"] = timing
            sample.metadata["submitted"] = False
            sample.metadata["trajectoryContractVersion"] = DRAWING_TOOLS_CONTRACT_VERSION
            sample.metadata["maxTurns"] = max_turns
        submitted = False
        operations = 0
        actions = 0

        def report(phase, step=None, *, turn=None):
            if progress is not None:
                progress({'phase': phase, 'turn': timing['model_calls'] if turn is None else turn,
                          'max_turns': max_turns, **({'step': step} if step else {})})

        async def perform(name, kind, title, operation, *, terminal=False, arguments=None):
            nonlocal submitted, operations, actions
            actions += 1
            step = {'id': f'action-{actions}', 'kind': kind, 'tool': name, 'title': title,
                    'turn': timing['model_calls'], 'status': 'running', 'arguments': arguments or {}}
            report('drawing', step)
            started = monotonic()

            def finish(content):
                report('drawing', {**step, **activity_result(content),
                                   'seconds': monotonic() - started})
                return content

            if submitted:
                return finish("Return code: 1\nThis request has already ended.")
            if max_calls is not None and operations >= max_calls:
                return finish("Return code: 1\nEpisode operation budget exhausted.")
            if approval is not None:
                report('awaiting_approval', {**step, 'status': 'pending'})
                await approval(step['id'])
                report('drawing', step)
                started = monotonic()
            operations += 1
            try:
                content = await operation()
            except TimeoutError:
                return finish(f"Return code: 124\nDrawing operation timed out after {timeout} seconds.")
            except FileNotFoundError:
                return finish("Return code: 1\nNo drawing yet. Call update_drawing with the complete source first.")
            except ValueError as error:
                return finish(f"Return code: 1\n{error}")
            except Exception as error:
                finish(f"Return code: 1\n{error}")
                raise
            finally:
                timing["command_seconds"] += monotonic() - started
            finish(content)
            if terminal and activity_result(content)['status'] == 'completed':
                submitted = True
                report('rendering')
            elif isinstance(content, list):
                report('previewing')
            return content

        async def read_source():
            source = await sandbox().read_file('/workspace/program.py')
            if not source.strip() or len(source.encode('utf-8')) > LIMITS['sourceBytes']:
                raise ValueError('Expected 1–65536 bytes of Python source')
            return source

        @tool(name='update_drawing', parallel=False)
        def update_drawing():
            async def execute(source: str) -> ToolResult:
                """Create or replace the drawing with a complete pycairo Python program.

                Args:
                    source: Complete Python source saving a 1920×1200 output.png, at most 64 KiB.
                """
                async def update():
                    if not source.strip() or len(source.encode('utf-8')) > LIMITS['sourceBytes']:
                        return 'Return code: 1\nExpected 1–65536 bytes of Python source.'
                    await sandbox().write_file('/workspace/program.py', source)
                    return 'Return code: 0\nDrawing updated. Call preview_drawing to check it.'
                return await perform('update_drawing', 'write', 'Refining your drawing', update, arguments={'source': source})
            return execute

        @tool(name='read_drawing', parallel=False)
        def read_drawing():
            async def execute() -> ToolResult:
                """Read the current drawing's Python source."""
                async def read():
                    return 'Return code: 0\n' + await read_source()
                return await perform('read_drawing', 'read', 'Reviewing your drawing', read)
            return execute

        @tool(name='preview_drawing', parallel=False)
        def preview_drawing():
            async def execute() -> ToolResult:
                """Render the current drawing and return execution feedback and its PNG preview."""
                async def preview():
                    source = await read_source()
                    seed = sample.metadata.get('seed', 42) if sample is not None else 42
                    result = await sandbox().exec(
                        ['timeout', '--signal=KILL', str(timeout), 'python', '/opt/artsy/render.py',
                         '--seed', str(seed), '--output', '/workspace/preview', '--preview'],
                        input=source, cwd='/workspace', timeout=timeout + 2, timeout_retry=False,
                    )
                    return observation(result, image_pattern)
                return await perform('preview_drawing', 'render', 'Checking the preview', preview)
            return execute

        @tool(name='submit_drawing', parallel=False)
        def submit_drawing():
            async def execute() -> ToolResult:
                """Finish the drawing and submit its current source for independent final validation."""
                async def submit():
                    await read_source()
                    return 'Return code: 0\nDrawing submitted for final validation.'
                return await perform('submit_drawing', 'submit', 'Finishing your drawing', submit, terminal=True)
            return execute

        @tool(name='refuse_drawing', parallel=False)
        def refuse_drawing():
            async def execute(reason: str) -> ToolResult:
                """Decline the drawing request with a reason and end without submitting artwork.

                Args:
                    reason: Concise reason for declining, at most 2048 UTF-8 bytes.
                """
                async def refuse():
                    if not reason.strip() or len(reason.encode('utf-8')) > 2048:
                        return 'Return code: 1\nExpected a reason of 1–2048 bytes.'
                    # Fixed paths and argv only; model arguments never enter a shell.
                    removed = await sandbox().exec(['rm', '-f', '/workspace/program.py'],
                                                  timeout=timeout, timeout_retry=False)
                    if not removed.success:
                        return observation(removed, image_pattern)
                    await sandbox().write_file('/workspace/refusal.json', json.dumps({'reason': reason}))
                    return 'Return code: 0\nRequest declined.'
                return await perform('refuse_drawing', 'refuse', 'Declining the request', refuse, terminal=True, arguments={'reason': reason})
            return execute

        tools = [update_drawing(), read_drawing(), preview_drawing(), submit_drawing(), refuse_drawing()]
        model = get_model()
        if generation_filter is not None:
            model = Model(FilteredAPI(model, generation_filter), config=config)
        state.messages = [
            ChatMessageSystem(content='You are Artsy, a drawing agent. Use the drawing tools to create, '
                               'inspect, preview, and submit artwork, or refuse a request. '
                               'Every response must contain at least one drawing tool call.'),
            ChatMessageUser(content=host_user_prompt(system_prompt, state.messages[-1].text, max_turns)),
        ]
        try:
            turns = 0
            while ((max_calls is None or turns < max_calls)
                   and (max_turns is None or turns < max_turns)):
                turns += 1
                step = {'id': f'turn-{turns}', 'kind': 'model', 'title': f'Agent turn {turns}',
                        'turn': turns, 'status': 'running'}
                report('thinking', step, turn=turns)
                if max_turns is not None:
                    state.messages.append(ChatMessageUser(content=
                        turn_countdown(turns, max_turns)))
                started = monotonic()
                try:
                    output = await model.generate(state.messages, tools=tools, config=config)
                except Exception as error:
                    report('thinking', {**step, 'status': 'failed', 'output': str(error)}, turn=turns)
                    raise
                finally:
                    timing["model_seconds"] += monotonic() - started
                    timing["model_calls"] += 1
                state.output = output
                state.messages.append(output.message)
                report('thinking', {**step, 'status': 'completed', 'output': output.message.text,
                                    'seconds': monotonic() - started}, turn=turns)
                if not output.message.tool_calls:
                    state.messages.append(ChatMessageUser(content=
                        'Use a drawing tool: update_drawing, read_drawing, preview_drawing, '
                        'submit_drawing, or refuse_drawing.'))
                    continue
                results, _ = await execute_tools(state.messages, tools)
                state.messages.extend(results)
                # Invalid tool names/arguments are rejected before a tool can report them.
                for index, result in enumerate(results):
                    if getattr(result, 'error', None):
                        call = next((call for call in output.message.tool_calls
                                     if call.id == getattr(result, 'tool_call_id', None)), None)
                        report('drawing', {'id': f'tool-error-{turns}-{index}', 'kind': 'tool',
                            'title': 'Action could not run', 'turn': turns, 'status': 'failed',
                            'tool': call.function if call else None,
                            'output': result.text or str(result.error)})
                if submitted or (max_calls is not None and operations >= max_calls):
                    break
        finally:
            timing["agent_seconds"] = monotonic() - began
            if sample is not None:
                sample.metadata["submitted"] = submitted
                sample.metadata["environmentCalls"] = operations
                sample.metadata["turnLimitReached"] = bool(
                    max_turns is not None and timing["model_calls"] >= max_turns and not submitted)
        return state

    return run
