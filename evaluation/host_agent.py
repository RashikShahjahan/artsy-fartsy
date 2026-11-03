"""Host-side drawing policy; only tool commands enter the execution sandbox."""

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

AGENT_NAME = "artsy-host"
AGENT_VERSION = "2"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
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


def action_title(command):
    if command.strip() == SUBMIT:
        return 'Submit drawing'
    if re.search(r'(?:^|\s)/opt/artsy/render\.py(?:\s|$)', command):
        return 'Render preview'
    if re.search(r'(?:^|\n)cat\s*>\s*(?:/workspace/)?program\.py\b', command):
        return 'Write drawing code'
    return 'Run drawing action'


@agent(name=AGENT_NAME)
def drawing_agent(system_prompt, config, generation_filter=None, max_calls=LIMITS["maxCalls"],
                  max_turns=None, progress=None, approval=None):
    """Preserve the Mini SWE bash protocol without its container-side harness."""
    settings = yaml.safe_load(Path(__file__).with_name("mini.yaml").read_text())
    image_pattern = re.compile(settings["model"]["multimodal_regex"])
    timeout = settings["environment"]["timeout"]
    if max_turns is not None:
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be positive or None")
        system_prompt += (f"\n\nYou have {max_turns} model turns to complete this task. "
                          "Each response uses one turn. Submit your final output by the last turn.")

    async def run(state: AgentState):
        began = monotonic()
        timing = {"model_seconds": 0.0, "command_seconds": 0.0, "model_calls": 0}
        # AgentState shares sample metadata with the scorer and Inspect transcript.
        from inspect_ai.solver._task_state import sample_state
        sample = sample_state()
        if sample is not None:
            sample.metadata["harnessTiming"] = timing
            sample.metadata["submitted"] = False
        submitted = False
        operations = 0
        actions = 0

        def report(phase, step=None, *, turn=None):
            if progress is not None:
                progress({'phase': phase, 'turn': timing['model_calls'] if turn is None else turn,
                          'max_turns': max_turns, **({'step': step} if step else {})})

        @tool(name="bash", parallel=False)
        def bash():
            async def execute(command: str) -> ToolResult:
                """Execute a bash command in the drawing workspace.

                Args:
                    command: The bash command to execute.
                """
                nonlocal submitted, operations, actions
                actions += 1
                step = {'id': f'action-{actions}', 'kind': 'tool', 'title': action_title(command),
                        'command': command, 'turn': timing['model_calls'], 'status': 'running'}
                report('drawing', step)
                started = monotonic()

                def finish(content):
                    report('drawing', {**step, **activity_result(content),
                                       'seconds': monotonic() - started})
                    return content

                if submitted:
                    return finish("Return code: 1\nSubmission already requested.")
                if max_calls is not None and operations >= max_calls:
                    return finish("Return code: 1\nEpisode operation budget exhausted.")
                if approval is not None:
                    report('awaiting_approval', {**step, 'status': 'pending'})
                    await approval(step['id'])
                    report('drawing', step)
                    started = monotonic()
                operations += 1
                try:
                    # GNU timeout kills the command process group inside Docker.
                    # No generated shell command is ever executed on the host.
                    result = await sandbox().exec(
                        ["timeout", "--signal=KILL", str(timeout), "bash", "-c", command],
                        cwd="/workspace", timeout=timeout + 2, timeout_retry=False,
                    )
                except TimeoutError:
                    return finish(f"Return code: 124\nCommand timed out after {timeout} seconds.")
                except Exception as error:
                    finish(f"Return code: 1\n{error}")
                    raise
                finally:
                    timing["command_seconds"] += monotonic() - started
                if command.strip() == SUBMIT and result.success:
                    submitted = True
                content = observation(result, image_pattern)
                finish(content)
                if submitted:
                    report('rendering')
                elif isinstance(content, list):
                    report('previewing')
                return content
            return execute

        tools = [bash()]
        model = get_model()
        if generation_filter is not None:
            model = Model(FilteredAPI(model, generation_filter), config=config)
        # Match the previous Mini SWE system/instance framing for trained policies.
        state.messages = [
            ChatMessageSystem(content=settings["agent"]["system_template"]),
            ChatMessageUser(content=f"System instructions:\n{system_prompt}\n\nTask:\n"
                            + state.messages[-1].text),
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
                        f"Turn {turns} of {max_turns}." +
                        (" This is your final turn; submit your final output now."
                         if turns == max_turns else "")))
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
                        'Every response must contain a bash tool call with {"command":"..."}.'))
                    continue
                results, _ = await execute_tools(state.messages, tools)
                state.messages.extend(results)
                # Invalid tool names/arguments are rejected before bash() can report them.
                for index, result in enumerate(results):
                    if getattr(result, 'error', None):
                        call = next((call for call in output.message.tool_calls
                                     if call.id == getattr(result, 'tool_call_id', None)), None)
                        report('drawing', {'id': f'tool-error-{turns}-{index}', 'kind': 'tool',
                            'title': 'Action could not run', 'turn': turns, 'status': 'failed',
                            'command': json.dumps(call.arguments) if call else '',
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
