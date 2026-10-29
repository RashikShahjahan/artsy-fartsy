"""Native recorded Inspect trajectories preserve genuine bash actions and previews."""
import copy
import json
import unittest

from training.data import (TOOLS, SUBMIT, action, message, convert_record,
                           native_prefix, native_trajectory_hash, digest)
from training.dpo import load_preferences
from ui import test_dpo_review as review_fixtures
from ui.dpo_review import make_handler


class NativeTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = review_fixtures.DPOReviewTests('test_export_loads_real_trajectories_and_remains_portable')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        for r in self.fixture.records:
            system, user = native_prefix(r['spec'], {'token_limit': None, 'token_limit_type': 'output'})
            preview = {'type': 'image', 'image': r['id']+'/preview-1.png', 'imageHash': r['imageHash']}
            messages = [message('system', system), message('user', user),
                action("cat > /workspace/program.py <<'PY'\n"+r['code']+'PY', 'write', 'Choose shapes and colors.'),
                message('tool', 'Return code: 0\n', tool_call_id='write'),
                action('python /opt/artsy/render.py --seed 1 --output /workspace/preview --preview < /workspace/program.py', 'preview'),
                message('tool', 'Return code: 0\n{"ok":true}', tool_call_id='preview'),
                action(SUBMIT, 'submit', 'The preview satisfies the request.'),
                message('tool', 'Return code: 0\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', tool_call_id='submit')]
            messages[5]['content'].append(preview)
            r.update(trajectoryFormat='inspect-bash-v1', trace=[],
                     nativeTrajectory={'messages': messages, 'tools': copy.deepcopy(TOOLS),
                                       'trainable_message_indices': [2, 4, 6]},
                     nativeTrajectoryConfig={'token_limit': None, 'token_limit_type': 'output'},
                     submission={'feedback': {'ok': True}},
                     nativeResult={'outcome': 'submitted', 'submitted': True, 'scores': {'render_valid': 1},
                                   'sourceHash': digest(r['code'].encode()), 'imageHash': r['imageHash']})
            self.rehash(r)
        self.fixture.write()
        self.record = self.fixture.records[0]

    @staticmethod
    def rehash(record):
        record['nativeTrajectoryHash'] = native_trajectory_hash(record['nativeTrajectory'])

    def test_native_actions_reasoning_and_images_preserved(self):
        original = copy.deepcopy(self.record)
        example = convert_record(self.record, self.root)
        self.assertEqual(example['trainable_message_indices'], [2, 4, 6])
        self.assertEqual(example['messages'][2], original['nativeTrajectory']['messages'][2])
        self.assertEqual(example['messages'][6], original['nativeTrajectory']['messages'][6])
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
                with self.assertRaises(ValueError): convert_record(r, self.root)
        r = copy.deepcopy(self.record)
        r['nativeTrajectory']['messages'][2]['content'][0]['thinking'] = 'Changed without hash update'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): convert_record(r, self.root)

    def test_genuine_no_tool_assistant_and_harness_correction_preserved(self):
        r = copy.deepcopy(self.record)
        r['nativeTrajectory']['messages'][2:2] = [message('assistant', 'I will draw this.'),
            message('user', 'Every response must contain a bash tool call with {"command":"..."}.')]
        r['nativeTrajectory']['trainable_message_indices'] = [2, 4, 6, 8]
        self.rehash(r)
        converted = convert_record(r, self.root)
        self.assertEqual(converted['messages'][2]['content'][0]['text'], 'I will draw this.')
        self.assertEqual(converted['trainable_message_indices'], [2, 4, 6, 8])

    def test_native_portable_export_loads_real_dpo_trajectories(self):
        f = self.fixture
        f.handler = make_handler(f.path, 4173, f.output)
        body = f.body()
        self.assertEqual(f.request('POST', '/api/review', body)['status'], 200)
        body['revision'] = f.state()['revision']
        result = f.request('POST', '/api/export', body)
        self.assertEqual(result['status'], 200, result)
        import shutil
        for rid in ('a', 'b'): shutil.rmtree(self.root/rid)
        pairs, _ = load_preferences(f.output/'labels.jsonl', f.output/'accepted.jsonl')
        self.assertEqual(len(pairs), 1)
        chosen = pairs[0]['chosen']
        self.assertEqual(chosen['messages'][2]['content'][0]['type'], 'thinking')
        self.assertTrue(chosen['messages'][5]['content'][1]['image'].startswith(str(f.output.resolve())))


if __name__ == '__main__':
    unittest.main()
