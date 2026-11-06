"""Validate mini-SWE editing traces and retain cumulative editing supervision."""

import copy
import json
import re

from drawing.agent_contract import SFT_MAX_TURNS
from drawing.documentation import DOCUMENTATION_LOCATION
from drawing.mini_agent import TOOLS, VERSION, system_prompt, task_prompt
from drawing.render_contract import OUTPUT_PATH_GUIDANCE
from drawing.tool_dataset import trajectory_hash


# Historical headers are matched as recorded, including their trailing space.
_LEGACY_SYSTEM_PREFIX = (
    'You are a helpful assistant that can interact with a computer.\n\n'
    'Create the requested drawing using pycairo. Save output.png at 1920×1200. '
    'Do not add text to the image unless requested. Use bash tool calls; files persist, shell state does not. '
    'Submit with artsy-submit ENTRYPOINT, or echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT '
    'when the Python entrypoint is unambiguous. '
)
_LEGACY_SYSTEM_SUFFIX = (
    'Save to the relative path output.png in the current working directory. '
    'Python previews and final validation rerun your program in a fresh directory; '
    'an absolute output path or a PNG copied beforehand will not satisfy this requirement. '
    'Official pycairo API reference (searchable with rg): /opt/artsy/pycairo-api.pyi. '
    'Before writing code, search the installed API reference for the operations you plan to use. '
)
_LEGACY_SUBMISSION = 'Submission is mandatory; without a successful submission, the task is incomplete. '
_LEGACY_SYSTEM_PROMPTS = tuple(
    _LEGACY_SYSTEM_PREFIX + preview + _LEGACY_SYSTEM_SUFFIX
    for preview in (
        'View rendered images with artsy-preview ENTRYPOINT. ',
        _LEGACY_SUBMISSION + 'View rendered images with artsy-preview ENTRYPOINT. ',
        _LEGACY_SUBMISSION + 'Before submission, view rendered images with artsy-preview ENTRYPOINT. ',
    )
)


def _historical_task_prompt(prompt, workflow_version):
    output_guidance = OUTPUT_PATH_GUIDANCE
    if workflow_version == 5:
        workflow = f"""1. Identify the requested subjects, composition, style, and constraints. {DOCUMENTATION_LOCATION.strip()}
2. Before submission, view rendered images with `artsy-preview ENTRYPOINT`.
3. Make focused edits to fix the most important issues you observed, then preview again to verify the changes.
4. Submit promptly once the drawing meets the request. Submission is mandatory."""
        submit = 'Submit with `artsy-submit ENTRYPOINT`, or use `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` when the Python entrypoint is unambiguous.'
    elif workflow_version == 6:
        workflow = """1. Your first tool call must only inspect the installed pycairo API reference at `/opt/artsy/pycairo-api.pyi`. Search for the methods you plan to use and wait for the tool result.
2. Use the returned signatures when you write the drawing code in a later tool call.
3. Run `artsy-preview ENTRYPOINT` on the exact Python entrypoint you intend to submit and wait for the result. If the preview fails, fix the reported error and preview again. If you edit the source after a successful preview, preview the updated source.
4. Once you have seen a successful preview of the current source and it meets the request, submit that entrypoint. Submission is mandatory."""
        submit = 'Submit with `artsy-submit ENTRYPOINT`.'
    else:
        output_guidance = ('Save to the relative path output.png in the current working directory. '
                           'Final validation reruns your program in a fresh directory.')
        workflow = """1. Your first tool call must only inspect the installed pycairo API reference at `/opt/artsy/pycairo-api.pyi`. Search for the methods you plan to use and wait for the tool result.
2. Use the returned signatures when you write the drawing code in a later tool call.
3. Run the Python entrypoint you intend to submit with `rm -f output.png && python ENTRYPOINT` and wait for the result. If execution fails, fix the error and rerun the script.
4. After the script succeeds, use a separate tool call, `artsy-preview output.png`, to view the generated image. If you edit the source, run it again and view the new PNG before submitting.
5. Once you have viewed the image from the current source and it meets the request, submit that entrypoint. Submission is mandatory."""
        submit = 'Submit with `artsy-submit ENTRYPOINT`.'

    return f"""Please create this drawing:

{prompt}

Create the drawing using pycairo and save a 1920×1200 PNG. Do not add text to the image unless requested.
{output_guidance}

## Recommended Workflow

{workflow}

## Command Execution Rules

- Use exactly one bash tool call per response, with a single string argument in the form `{{"command": "..."}}`.
- Each command runs in a fresh subshell. Files persist, while directory and environment changes do not.
- {submit} Run the submission command by itself. A successful submission ends the task."""


def edits_python_source(command: str) -> bool:
    """Recognize definite source writes; a shell command's exit status alone is not enough.

    The bash tool permits arbitrary programs, so this intentionally does not
    infer writes from generic Python invocations or failed diagnostic commands.
    """
    return bool(
        re.search(r"""(?m)(?:^|[;&|])\s*(?:cat|printf|echo|tee)\b[^\n]*>{1,2}\s*['"]?[^\s'";]+\.py\b""", command)
        or re.search(r"\b(?:write_text|write_bytes)\s*\(", command)
        or re.search(r"""\bopen\s*\([^)]*,\s*['"][wax][bt+]*['"]""", command)
        or re.search(r"\b(?:apply_patch|patch)\b", command)
        or re.search(r"\b(?:sed|perl)\s+-[^\s]*i\b", command)
        or re.search(r"\b(?:cp|mv)\s+[^\n;]*\.py\b", command)
    )


def convert_mini_record(record, root, loss_policy):
    from training.data import artifact

    rid = record['id']
    trajectory = record.get('miniTrajectory')
    if not isinstance(trajectory, dict) or trajectory_hash(trajectory) != record.get('miniTrajectoryHash'):
        raise ValueError(f'{rid}: mini trajectory hash mismatch')
    info = trajectory.get('info', {})
    if (trajectory.get('trajectory_format') != 'mini-swe-agent-1.1' or info.get('mini_version') != VERSION
            or info.get('exit_status') != 'Submitted'):
        raise ValueError(f'{rid}: unsupported or unfinished mini trajectory')
    messages = copy.deepcopy(trajectory.get('messages'))
    if not isinstance(messages, list) or len(messages) < 5:
        raise ValueError(f'{rid}: missing mini conversation')
    spec = record['spec']
    expected = [
        [{'role': 'system', 'content': system_prompt(spec['seed'])},
         {'role': 'user', 'content': task_prompt(spec['prompt'])}],
        *([{'role': 'system', 'content': system_prompt(spec['seed'])},
           {'role': 'user', 'content': _historical_task_prompt(spec['prompt'], version)}]
          for version in (7, 6, 5)),
        *([{'role': 'system', 'content': system}, {'role': 'user', 'content': spec['prompt']}]
          for system in _LEGACY_SYSTEM_PROMPTS),
    ]
    if messages[:2] not in expected:
        raise ValueError(f'{rid}: mini prompt mismatch')
    if messages[-1].get('role') != 'exit' or messages[-1].get('extra', {}).get('exit_status') != 'Submitted':
        raise ValueError(f'{rid}: missing terminal mini submission')
    messages.pop()
    turns = sum(m.get('role') == 'assistant' for m in messages)
    if (turns > SFT_MAX_TURNS or turns != record['generation']['modelTurns']
            or turns != len(record['trace'])):
        raise ValueError(f'{rid}: mini model turn count mismatch')
    source = artifact(root, record['artifacts']['source'], record['sourceHash']).read_text()
    if source != record['code'] or not source.strip() or len(source.encode()) > 65536:
        raise ValueError(f'{rid}: mini final source mismatch')
    from drawing.mini_session import bundle_source
    if bundle_source(record.get('sourceBundle') or {}) != source:
        raise ValueError(f'{rid}: mini workspace snapshot mismatch')
    artifact(root, record['artifacts']['image'], record['imageHash'])
    if not record.get('submission', {}).get('feedback', {}).get('ok'):
        raise ValueError(f'{rid}: mini final render failed')
    pending = None
    seen = set()
    selected = []
    error_aware = loss_policy == 'error-aware'
    unverified_edits = []
    masked_edits = set()
    submitted = False
    for index, message in enumerate(messages):
        if submitted:
            raise ValueError(f'{rid}: mini conversation continues after submission')
        role = message.get('role')
        extra = message.pop('extra', {})
        content = message.get('content')
        if isinstance(content, str):
            message['content'] = [{'type': 'text', 'text': content}]
        elif isinstance(content, list):
            for part in content:
                if part.get('type') == 'image' and role == 'tool':
                    part['image'] = str(artifact(root, part['image'], part['imageHash']))
                elif part.get('type') != 'text' or not isinstance(part.get('text'), str):
                    raise ValueError(f'{rid}: invalid mini content')
        else:
            raise ValueError(f'{rid}: invalid mini content')
        if role == 'assistant':
            if pending:
                raise ValueError(f'{rid}: unanswered mini action')
            calls = message.get('tool_calls') or []
            if len(calls) != 1:
                continue  # Recorded malformed reply; context only.
            call = calls[0]
            function = call.get('function', {})
            args = function.get('arguments')
            if isinstance(args, str):
                args = json.loads(args)
                function['arguments'] = args
            if (function.get('name') != 'bash' or not isinstance(args, dict) or set(args) != {'command'}
                    or not isinstance(args['command'], str)):
                continue
            if not call.get('id') or call['id'] in seen:
                raise ValueError(f'{rid}: duplicate mini action ID')
            seen.add(call['id'])
            pending = (call['id'], args['command'].strip(), index)
        elif role == 'tool':
            if not pending or message.get('tool_call_id') != pending[0]:
                raise ValueError(f'{rid}: mini tool result mismatch')
            _, command, assistant_index = pending
            pending = None
            selected.append(assistant_index)
            if error_aware:
                if edits_python_source(command):
                    unverified_edits.append(assistant_index)
                    # A failed shell write may already have modified the workspace.
                    if type(extra.get('returncode')) is int and extra['returncode'] != 0:
                        masked_edits.add(assistant_index)
                for feedback in extra.get('previewEvents') or []:
                    # Only a renderer event tied to an actual source snapshot
                    # diagnoses the program. CLI usage and missing-path errors
                    # must not retroactively condemn an otherwise valid edit.
                    if not isinstance(feedback, dict) or not feedback.get('sourceHash'):
                        continue
                    if feedback.get('ok') is False:
                        masked_edits.update(unverified_edits)
                        unverified_edits.clear()
                    elif feedback.get('ok') is True:
                        unverified_edits.clear()
            images = [p for p in message['content'] if p['type'] == 'image']
            from drawing.mini_session import session_action
            action, entrypoint = session_action(command)
            if extra.get('terminal') or action == 'submit':
                if extra.get('returncode') != 0 and not extra.get('terminal'):
                    continue
                if (extra.get('returncode') != 0 or not extra.get('terminal')
                        or extra.get('sourceHash') != record['sourceHash']
                        or record['submission'].get('sourceHash') != record['sourceHash']
                        or extra.get('entrypoint') != record['submission'].get('entrypoint')):
                    raise ValueError(f'{rid}: mini submission snapshot mismatch')
                submitted = True
            elif images and action != 'preview' and not extra.get('previewEvents'):
                raise ValueError(f'{rid}: image outside trusted mini preview')
        elif role not in ('system', 'user') or pending:
            raise ValueError(f'{rid}: invalid mini conversation order')
    if pending or not submitted:
        raise ValueError(f'{rid}: incomplete mini submission')
    # Keep all recorded actions, feedback, and preview pixels in context.
    # Experiment 6 only removes loss on edits implicated by an observed
    # failed render (or the edit command's own nonzero status). A later repair
    # and the diagnostic preview action remain supervised.
    if error_aware:
        selected = [index for index in selected if index not in masked_edits]
    return {'id': rid, 'messages': messages, 'tools': copy.deepcopy(TOOLS),
            'trainable_message_indices': selected}
