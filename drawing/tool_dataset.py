"""Collect native drawing-tool conversations with a strict SFT turn budget."""

from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path

from drawing.agent_contract import DRAWING_TOOLS, SFT_MAX_TURNS, drawing_tools_prompt
from drawing.dataset_specs import compact_json, sha256
from drawing.environment import DrawingEnvironment

FORMAT = 'drawing-tools-v1'


def trajectory_hash(trajectory):
    return sha256(compact_json(trajectory))


def text_message(role, text, **kwargs):
    return {'role': role, 'content': [{'type': 'text', 'text': text}], **kwargs}


def provider_messages(messages, root):
    """Load recorded local preview pixels only when sending the conversation."""
    result = copy.deepcopy(messages)
    for message in result:
        for part in message['content']:
            if part['type'] == 'image':
                png = (root / part['image']).read_bytes()
                if sha256(png) != part['imageHash']:
                    raise ValueError('Preview artifact hash mismatch')
                part.clear()
                part.update(type='image_url', image_url={'url': 'data:image/png;base64,' + base64.b64encode(png).decode()})
        for call in message.get('tool_calls', []):
            arguments = call['function']['arguments']
            if isinstance(arguments, dict):
                call['function']['arguments'] = compact_json(arguments)
    return result


async def candidate(options, spec, index, api_doc):
    # Import here to avoid a cycle with the dataset orchestration module.
    from drawing.dataset import DATASET_VERSION, _default_config, _pixel_hash, _provenance

    output = Path(options['output']).resolve()
    config = options.get('config') or _default_config()
    env = DrawingEnvironment(output / 'artifacts', renderer=options.get('renderer'), config=config)
    episode = await env.reset(spec['prompt'], spec['seed'])
    episode_dir = Path(episode['directory'])
    rel = lambda path: os.path.relpath(path, output)
    record = {
        'version': DATASET_VERSION, 'id': f"{spec['id']}-c{index}", 'spec': spec, 'candidate': index,
        'trajectoryFormat': FORMAT, 'status': 'rejected',
        'reason': f'No valid submission within {SFT_MAX_TURNS} model turns',
        'artifacts': {'directory': rel(episode_dir), 'trajectory': rel(episode_dir / 'trajectory.jsonl')},
        'trace': [], 'evaluations': [], 'teacher': _provenance(options['teacher']),
        'runtimeVersion': config['runtimeVersion'], 'apiDocHash': sha256(api_doc), 'humanReviewed': False,
        'generation': {'initialDraft': 'natural', 'executionFailures': 0, 'imagePreviews': True,
                       'modelTurns': 0, 'maxTurns': SFT_MAX_TURNS},
    }
    messages = [text_message('system', drawing_tools_prompt() +
                            f'\nComplete this request within {SFT_MAX_TURNS} model turns. '
                            'Submit only after inspecting a successful preview of the current source.'),
                text_message('user', spec['prompt'])]
    last_preview = None
    write_turn = None
    call_ids = set()
    try:
        for turn in range(1, SFT_MAX_TURNS + 1):
            completion = await options['teacher'].complete(
                provider_messages(messages, output), options.get('maxTokens'), tools=DRAWING_TOOLS)
            record['generation']['modelTurns'] = turn
            trace = {'attempt': turn - 1, 'completion': completion, 'operations': []}
            record['trace'].append(trace)
            calls = copy.deepcopy(completion.get('tool_calls') or [])
            assistant = text_message('assistant', completion.get('reasoning') or completion.get('text') or '')
            # Keep both reasoning and spoken content, when present.
            if completion.get('reasoning') and completion.get('text'):
                assistant['content'][0]['text'] += '\n' + completion['text']
            if calls:
                assistant['tool_calls'] = calls
            messages.append(assistant)
            charged = await env.charge_tokens(completion.get('usage', {}).get('completion_tokens', 0))
            fatal = not charged['ok']
            terminal = False
            for call in calls:
                name = call['function']['name']
                arguments = call['function']['arguments']
                feedback = None
                image = None
                error = None
                try:
                    if completion.get('finishReason') == 'length':
                        raise ValueError('Teacher completion was truncated')
                    if fatal:
                        raise ValueError('Episode budget exhausted')
                    if call['id'] in call_ids:
                        raise ValueError('Tool call IDs must be unique')
                    call_ids.add(call['id'])
                    if name not in {t['function']['name'] for t in DRAWING_TOOLS}:
                        raise ValueError('Unknown drawing tool')
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    required = {'source'} if name == 'update_drawing' else {'reason'} if name == 'refuse_drawing' else set()
                    if not isinstance(arguments, dict) or set(arguments) != required or any(
                            not isinstance(v, str) for v in arguments.values()):
                        raise ValueError('Invalid drawing tool arguments')
                    call['function']['arguments'] = arguments
                    if name in ('submit_drawing', 'refuse_drawing') and len(calls) != 1:
                        raise ValueError('Submission or refusal must be the only call in its turn')
                    if name == 'update_drawing':
                        feedback = await env.write_code(arguments['source'])
                        if feedback['ok']:
                            record['code'] = arguments['source']
                            trace['executedCode'] = arguments['source']
                            last_preview, write_turn = None, turn
                    elif name == 'read_drawing':
                        feedback = {'ok': bool(env.source), 'source': env.source}
                    elif name == 'preview_drawing':
                        if write_turn == turn:
                            raise ValueError('Call preview_drawing on a separate model turn after update_drawing')
                        feedback = await env.run_code()
                        last_preview = None
                        if feedback['ok']:
                            png = Path(feedback['artifact']).read_bytes()
                            last_preview = {'image': rel(feedback['artifact']), 'imageHash': sha256(png)}
                            image = {'type': 'image', **last_preview}
                            trace['previewObservation'] = last_preview
                    elif name == 'submit_drawing':
                        if last_preview is None:
                            raise ValueError('Render and inspect a successful preview before submitting')
                        feedback = await env.submit()
                        if feedback['ok']:
                            png = Path(feedback['artifact']).read_bytes()
                            record.update(status='review', reason='Preview inspected by teacher; human review pending',
                                          sourceHash=sha256(record['code'].replace('\r\n', '\n').strip()),
                                          imageHash=sha256(png), pixelHash=_pixel_hash(png))
                            record['artifacts'].update(image=rel(feedback['artifact']), source=rel(episode_dir / 'program.py'))
                            record['submission'] = {'feedback': feedback, 'preview': last_preview}
                            terminal = True
                    elif name == 'refuse_drawing':
                        reason = arguments['reason']
                        if not reason.strip() or len(reason.encode()) > 2048:
                            raise ValueError('Expected a reason of 1–2048 bytes')
                        feedback = await env.refuse(reason)
                        if feedback['ok']:
                            record.update(refusal=reason, status='review', reason='Teacher refusal; human review pending')
                            terminal = True
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)
                    feedback = {'ok': False, 'error': error}
                if feedback.get('failureKind') == 'invalid_code':
                    record['generation']['executionFailures'] += 1
                if feedback.get('failureKind') == 'infrastructure':
                    record.update(status='infrastructure_error', reason=feedback.get('error') or 'Renderer infrastructure failure')
                    fatal = True
                trace['operations'].append({'tool': name, 'feedback': feedback})
                visible = {key: value for key, value in feedback.items() if key not in ('artifact',)}
                result = text_message('tool', 'Return code: ' + ('0' if feedback['ok'] else '1') + '\n' + compact_json(visible),
                                      tool_call_id=call['id'], name=name)
                if image:
                    result['content'].append(image)
                messages.append(result)
                if not feedback.get('ok') and feedback.get('outcome', 'active') != 'active':
                    fatal = True
            if not calls:
                messages.append(text_message('user', 'Every response must contain a drawing tool call. Use the drawing tools.'))
            if terminal or fatal:
                break
    except Exception as exc:
        record.update(status='infrastructure_error', reason=str(exc))
    record['drawingTrajectory'] = {'tools': copy.deepcopy(DRAWING_TOOLS), 'messages': messages}
    record['drawingTrajectoryHash'] = trajectory_hash(record['drawingTrajectory'])
    return record
