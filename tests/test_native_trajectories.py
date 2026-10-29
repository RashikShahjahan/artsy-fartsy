"""Native Inspect drawing trajectories remain trainable without preference learning."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from training.data import (ROOT, TOOLS, SUBMIT, action, message, convert_record,
                           load_examples, native_prefix, native_trajectory_hash, digest)


class NativeTrajectoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        directory = self.root / 'drawing'
        directory.mkdir()
        code = "draw('red')\n"
        (directory / 'program.py').write_text(code)
        for name in ('image.png', 'preview-1.png'):
            Image.new('RGB', (24, 16), 'red').save(directory / name)
        image_hash = digest((directory / 'image.png').read_bytes())
        record = {
            'id': 'drawing', 'status': 'accepted',
            'spec': {'prompt': 'Draw a red circle', 'seed': 1},
            'runtimeVersion': 'artcanvas-v2',
            'apiDocHash': digest((ROOT / 'docs/artcanvas.md').read_bytes()),
            'artifacts': {'directory': 'drawing', 'source': 'drawing/program.py',
                          'image': 'drawing/image.png'},
            'code': code, 'sourceHash': digest(code.strip().encode()),
            'imageHash': image_hash, 'trace': [],
        }
        system, user = native_prefix(record['spec'], {'token_limit': None, 'token_limit_type': 'output'})
        preview = {'type': 'image', 'image': 'drawing/preview-1.png', 'imageHash': image_hash}
        messages = [message('system', system), message('user', user),
            action("cat > /workspace/program.py <<'PY'\n" + code + 'PY', 'write', 'Choose shapes and colors.'),
            message('tool', 'Return code: 0\n', tool_call_id='write'),
            action('python /opt/artsy/render.py --seed 1 --output /workspace/preview --preview < /workspace/program.py', 'preview'),
            message('tool', 'Return code: 0\n{"ok":true}', tool_call_id='preview'),
            action(SUBMIT, 'submit', 'The preview satisfies the request.'),
            message('tool', 'Return code: 0\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', tool_call_id='submit')]
        messages[5]['content'].append(preview)
        record.update(trajectoryFormat='inspect-bash-v1',
                      nativeTrajectory={'messages': messages, 'tools': copy.deepcopy(TOOLS),
                                        'trainable_message_indices': [2, 4, 6]},
                      nativeTrajectoryConfig={'token_limit': None, 'token_limit_type': 'output'},
                      submission={'feedback': {'ok': True}},
                      nativeResult={'outcome': 'submitted', 'submitted': True,
                                    'scores': {'render_valid': 1},
                                    'sourceHash': digest(code.encode()), 'imageHash': image_hash})
        self.rehash(record)
        self.record = record

    @staticmethod
    def rehash(record):
        record['nativeTrajectoryHash'] = native_trajectory_hash(record['nativeTrajectory'])

    def test_native_actions_reasoning_and_images_preserved(self):
        original = copy.deepcopy(self.record)
        example = convert_record(self.record, self.root)
        self.assertEqual(example['trainable_message_indices'], [2, 4, 6])
        for index in (2, 6):
            expected = copy.deepcopy(original['nativeTrajectory']['messages'][index])
            for part in expected['content']:
                if part['type'] == 'thinking':
                    part['type'] = 'text'
                    part['text'] = part.pop('thinking')
            self.assertEqual(example['messages'][index], expected)
        self.assertTrue(example['messages'][5]['content'][1]['image'].startswith(str(self.root.resolve())))
        self.assertEqual(self.record, original)

    def test_mutated_artifact_hash_mask_prefix_or_order_rejected(self):
        mutations = [
            lambda r: r.update(sourceHash='0'*64),
            lambda r: r.update(imageHash='0'*64),
            lambda r: r['nativeTrajectory'].update(trainable_message_indices=[2, 6]),
            lambda r: r['nativeTrajectory']['messages'][5]['content'][1].update(imageHash='0'*64),
            lambda r: r['nativeTrajectory']['messages'][1]['content'][0].update(text='Altered prompt'),
            lambda r: r['nativeTrajectory']['messages'][3].update(tool_call_id='wrong'),
            lambda r: r['nativeTrajectory']['messages'][-1]['content'][0].update(text='Return code: 1\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'),
            lambda r: r['submission']['feedback'].update(ok=False),
            lambda r: r['nativeResult'].update(submitted=False),
            lambda r: r['nativeResult'].update(sourceHash='0'*64),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                r = copy.deepcopy(self.record)
                mutation(r)
                self.rehash(r)
                with self.assertRaises(ValueError):
                    convert_record(r, self.root)
        r = copy.deepcopy(self.record)
        r['nativeTrajectory']['messages'][2]['content'][0]['thinking'] = 'Changed without hash update'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            convert_record(r, self.root)

    def test_genuine_no_tool_assistant_and_harness_correction_preserved(self):
        r = copy.deepcopy(self.record)
        r['nativeTrajectory']['messages'][2:2] = [
            message('assistant', 'I will draw this.'),
            message('user', 'Every response must contain a bash tool call with {"command":"..."}.')]
        r['nativeTrajectory']['trainable_message_indices'] = [2, 4, 6, 8]
        self.rehash(r)
        converted = convert_record(r, self.root)
        self.assertEqual(converted['messages'][2]['content'][0]['text'], 'I will draw this.')
        self.assertEqual(converted['trainable_message_indices'], [2, 4, 6, 8])

    def test_native_trajectory_loads_as_sft_example(self):
        path = self.root / 'accepted.jsonl'
        path.write_text(json.dumps(self.record) + '\n')
        examples, manifest = load_examples(path)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]['id'], 'drawing')
        self.assertEqual(examples[0]['trainable_message_indices'], [2, 4, 6])
        self.assertEqual(manifest['records'], 1)


if __name__ == '__main__':
    unittest.main()
