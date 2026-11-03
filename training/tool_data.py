"""Validate and preserve recorded drawing-tool SFT conversations."""

import copy
import json

from drawing.agent_contract import DRAWING_TOOLS, SFT_MAX_TURNS, drawing_tools_prompt
from drawing.tool_dataset import text_message, trajectory_hash


def convert_drawing_record(record, root, loss_policy):
    from training.data import artifact, digest

    rid = record['id']
    if record['spec'].get('kind') == 'refusal':
        if (record.get('refusal') != record['spec']['refusal']
                or record.get('refusalSource') != 'prompt_plan' or record.get('trace')
                or record.get('code') or record.get('artifacts')):
            raise ValueError(f'{rid}: invalid authored refusal pair')
        reason = record['refusal']
        if not reason.strip() or len(reason.encode()) > 2048:
            raise ValueError(f'{rid}: invalid refusal reason')
        call = {'id': 'call_refusal', 'type': 'function',
                'function': {'name': 'refuse_drawing', 'arguments': {'reason': reason}}}
        return {'id': rid, 'tools': copy.deepcopy(DRAWING_TOOLS), 'trainable_message_indices': [2],
                'messages': [text_message('system', drawing_tools_prompt()),
                             text_message('user', record['spec']['prompt']),
                             text_message('assistant', '', tool_calls=[call]),
                             text_message('tool', 'Return code: 0\nRequest declined.',
                                          tool_call_id='call_refusal', name='refuse_drawing')]}

    trajectory = record.get('drawingTrajectory')
    if not isinstance(trajectory, dict) or trajectory_hash(trajectory) != record.get('drawingTrajectoryHash'):
        raise ValueError(f'{rid}: drawing trajectory hash mismatch')
    if trajectory.get('tools') != DRAWING_TOOLS:
        raise ValueError(f'{rid}: drawing tool schema mismatch')
    messages = copy.deepcopy(trajectory.get('messages'))
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError(f'{rid}: missing drawing conversation')
    expected_system = drawing_tools_prompt() + (
        f'\nComplete this request within {SFT_MAX_TURNS} model turns. '
        'Submit only after inspecting a successful preview of the current source.')
    if messages[:2] != [text_message('system', expected_system), text_message('user', record['spec']['prompt'])]:
        raise ValueError(f'{rid}: drawing conversation prefix mismatch')
    turns = sum(m.get('role') == 'assistant' for m in messages)
    if turns > SFT_MAX_TURNS:
        raise ValueError(f'{rid}: trajectory exceeds {SFT_MAX_TURNS} assistant turns ({turns})')
    if record.get('generation', {}).get('modelTurns') != turns or len(record.get('trace', [])) != turns:
        raise ValueError(f'{rid}: model turn count mismatch')

    pending = []
    ids = set()
    source = None
    last_write = last_preview = None
    preview_ref = None
    terminal = None
    assistant_indices = []
    for index, message in enumerate(messages[2:], 2):
        role = message.get('role')
        if terminal is not None:
            raise ValueError(f'{rid}: conversation continues after terminal action')
        if not isinstance(message.get('content'), list):
            raise ValueError(f'{rid}: invalid message content')
        for part in message['content']:
            if part.get('type') == 'image':
                if role != 'tool':
                    raise ValueError(f'{rid}: image outside tool observation')
                path = artifact(root, part['image'], part['imageHash'])
                part['image'] = str(path)
            elif part.get('type') != 'text' or not isinstance(part.get('text'), str):
                raise ValueError(f'{rid}: invalid message part')
        if role == 'assistant':
            if pending:
                raise ValueError(f'{rid}: missing tool results')
            assistant_indices.append(index)
            calls = message.get('tool_calls', [])
            for call in calls:
                if not isinstance(call.get('id'), str) or call['id'] in ids or call.get('type') != 'function':
                    raise ValueError(f'{rid}: invalid or duplicate tool call ID')
                ids.add(call['id'])
                pending.append((call, index, len(calls)))
        elif role == 'tool':
            if not pending:
                raise ValueError(f'{rid}: unexpected tool result')
            call, assistant_index, call_count = pending.pop(0)
            function = call['function']
            name, args = function['name'], function['arguments']
            if message.get('tool_call_id') != call['id'] or message.get('name') != name:
                raise ValueError(f'{rid}: mismatched tool result')
            text = message['content'][0]['text']
            try:
                prefix, payload = text.split('\n', 1)
                feedback = json.loads(payload)
                ok = feedback['ok']
                if type(ok) is not bool or prefix != 'Return code: ' + ('0' if ok else '1'):
                    raise ValueError('Invalid return code')
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'{rid}: invalid recorded tool feedback') from exc
            if not ok:
                if any(p['type'] == 'image' for p in message['content']):
                    raise ValueError(f'{rid}: failed operation contains preview')
                continue
            schema = next((t['function'] for t in DRAWING_TOOLS if t['function']['name'] == name), None)
            if (schema is None or not isinstance(args, dict)
                    or set(args) != set(schema['parameters']['required'])
                    or any(not isinstance(v, str) for v in args.values())):
                raise ValueError(f'{rid}: invalid successful tool call')
            if name == 'update_drawing':
                source = args['source']
                if not source.strip() or len(source.encode()) > 65536:
                    raise ValueError(f'{rid}: invalid source')
                last_write, last_preview, preview_ref = assistant_index, None, None
            elif name == 'preview_drawing':
                images = [p for p in message['content'] if p['type'] == 'image']
                if source is None or last_write == assistant_index or len(images) != 1:
                    raise ValueError(f'{rid}: preview requires a prior source turn and one image')
                last_preview = assistant_index
                preview_ref = {'image': str((root / images[0]['image']).resolve()), 'imageHash': images[0]['imageHash']}
            elif name in ('submit_drawing', 'refuse_drawing'):
                if call_count != 1 or pending:
                    raise ValueError(f'{rid}: terminal action must be alone')
                if name == 'submit_drawing':
                    if last_preview is None or last_preview >= assistant_index:
                        raise ValueError(f'{rid}: missing confirmation of the current source preview')
                    submission = record.get('submission') or {}
                    ref = submission.get('preview') or {}
                    if (not submission.get('feedback', {}).get('ok') or not ref.get('image')
                            or {'image': str((root / ref['image']).resolve()), 'imageHash': ref.get('imageHash')} != preview_ref):
                        raise ValueError(f'{rid}: final preview confirmation mismatch')
                    saved = artifact(root, record['artifacts']['source']).read_text()
                    normalized = lambda s: s.replace('\r\n', '\n').strip()
                    if (normalized(saved) != normalized(source) or normalized(record['code']) != normalized(source)
                            or digest(normalized(source).encode()) != record['sourceHash']):
                        raise ValueError(f'{rid}: final source mismatch')
                    artifact(root, record['artifacts']['image'], record['imageHash'])
                else:
                    if (args['reason'] != record.get('refusal') or not args['reason'].strip()
                            or len(args['reason'].encode()) > 2048):
                        raise ValueError(f'{rid}: refusal mismatch')
                terminal = (name, assistant_index)
        elif role != 'user' or pending or message.get('tool_calls'):
            raise ValueError(f'{rid}: invalid conversation order')
    if pending or terminal is None:
        raise ValueError(f'{rid}: trajectory has no completed submission or refusal')
    selected = ([*assistant_indices] if loss_policy == 'assistant-only' else
                [terminal[1]] if terminal[0] == 'refuse_drawing' else
                sorted({last_write, last_preview, terminal[1]}))
    return {'id': rid, 'messages': messages, 'tools': copy.deepcopy(DRAWING_TOOLS),
            'trainable_message_indices': selected}
