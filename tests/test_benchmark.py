import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evaluation.benchmark import suite, rate, classification, load_run, load_labels, prepare, report

class BenchmarkTests(unittest.TestCase):
    def test_default_is_validated_latest_bytes_and_ui_uses_same_dataset(self):
        from evaluation.suite import dataset_path
        from drawing.eval_specs import load_specs
        from ui.server import drawing_dataset
        manifest, rows = suite()
        self.assertEqual(manifest['status'], 'validated')
        self.assertEqual(manifest['counts'], {'drawing':35, 'benign_boundary':5, 'harmful_image':5, 'malicious_code':5})
        self.assertEqual([r['id'] for r in rows], [r['id'] for r in load_specs(dataset_path())])
        with patch('ui.server.DRAWING_DATASET', None):
            self.assertEqual(drawing_dataset()['sha256'], manifest['sha256'])

    def test_runner_compares_three_checkpoints_and_propagates_equal_budgets(self):
        from evaluation.run_benchmark import main
        from drawing.config import MODEL_ID
        from evaluation.suite import dataset_path
        models = {'base':MODEL_ID, 'sft':'artsy/sft-v1', 'rl':'artsy/grpo-v2'}
        catalog = [{'id': model, 'model_path':None if name == 'base' else 'tinker://'+name}
                   for name, model in models.items()]
        _, specs = suite()
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            option = lambda key: command[command.index(key)+1]
            output = Path(option('--output')); output.mkdir()
            (output/'summary.json').write_text(json.dumps({'ids':[s['id'] for s in specs],
                'epochs':1, 'model':option('--model'), 'status':'success', 'sample_errors':0,
                'scoring_version':'v3','max_tokens':int(option('--max-tokens')),
                'token_limit':int(option('--token-limit'))}))
            self.assertEqual(option('--dataset'), str(dataset_path()))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'comparison'
            argv = ['benchmark', '--output', str(output), '--max-tokens','12000','--token-limit','48000', '--time-limit', '600']
            for name, model in models.items(): argv.extend(['--model', name+'='+model])
            with patch('sys.argv', argv), patch('evaluation.run_benchmark.httpx.get') as health, \
                 patch('evaluation.run_benchmark.load_dotenv'), \
                 patch.dict('os.environ', {'ARTSY_CORRECTNESS_JUDGE_MODEL':'mockllm/correct',
                                          'ARTSY_SAFETY_JUDGE_MODEL':'mockllm/safe'}), \
                 patch('evaluation.run_benchmark.subprocess.run', side_effect=run), \
                 contextlib.redirect_stdout(io.StringIO()):
                health.return_value.raise_for_status.return_value.json.return_value = {
                    'backend':'tinker','model_path':None,'models':catalog}
                self.assertEqual(main(), 0)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(summary['comparisons']), 3)
            for command in calls:
                self.assertEqual(command[command.index('--max-tokens')+1], '12000')
                self.assertEqual(command[command.index('--token-limit')+1], '48000')
                self.assertEqual(command[command.index('--time-limit')+1], '600')
            result = json.loads((output/'comparisons/sft-vs-rl/report.json').read_text())
            self.assertEqual(set(result['models']), {'sft','rl'})
            self.assertIsNone(result['visual']['candidate_win_fraction_with_half_ties'])

    def test_invalid_and_unavailable_comparison_models_are_rejected(self):
        from evaluation.run_benchmark import selected_models
        catalog = [{'id':'base'}, {'id':'artsy/sft'}]
        for values in (['../bad=base','sft=artsy/sft'], ['base=base','base=artsy/sft'],
                       ['base=base','sft=missing'], ['base=base','same=base']):
            with self.subTest(values=values), self.assertRaises(ValueError):
                selected_models(values, catalog)

    def test_runner_forwards_datagen_defaults_without_restoring_a_response_cap(self):
        from evaluation.run_benchmark import main
        from drawing.config import MODEL_ID, MODEL_CONTEXT_WINDOW, LIMITS
        catalog = [{'id': MODEL_ID, 'model_path': None},
                   {'id': 'artsy/sft-v1', 'model_path': 'tinker://sft'}]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'comparison'
            with patch('sys.argv', ['benchmark', '--output', str(output)]), \
                 patch('evaluation.run_benchmark.httpx.get') as health, \
                 patch('evaluation.run_benchmark.load_dotenv'), \
                 patch.dict('os.environ', {'ARTSY_CORRECTNESS_JUDGE_MODEL':'mockllm/correct',
                                          'ARTSY_SAFETY_JUDGE_MODEL':'mockllm/safe'}), \
                 patch('evaluation.run_benchmark.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                health.return_value.raise_for_status.return_value.json.return_value = {
                    'backend': 'tinker', 'model_path': None, 'models': catalog}
                self.assertEqual(main(), 0)
            plan = json.loads((output / 'run-plan.json').read_text())
            self.assertIsNone(plan['max_tokens'])
            self.assertEqual(plan['token_limit'], MODEL_CONTEXT_WINDOW)
            self.assertEqual(plan['environment_limits'], LIMITS)
            self.assertEqual(plan['time_limit'], 300)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index('--max-tokens') + 1], '0')
                self.assertEqual(command[command.index('--token-limit') + 1], str(MODEL_CONTEXT_WINDOW))

    def test_frozen_mix_and_difficulty(self):
        manifest, rows=suite()
        self.assertEqual(manifest['counts'],dict(drawing=35,benign_boundary=5,harmful_image=5,malicious_code=5))
        self.assertEqual(len(rows),50)
        self.assertEqual({r['difficulty'] for r in rows}, {'easy','medium','hard','very-hard'})

    def test_frozen_bytes_cannot_be_silently_replaced(self):
        from evaluation.suite import dataset_path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _ = suite()
            manifest['dataset'] = 'prompts.jsonl'
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'prompts.jsonl').write_bytes(dataset_path().read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'frozen manifest'):
                suite(root)

    def test_packet_rejects_different_scoring_contracts(self):
        _, specs = suite()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, version in [('base','legacy'),('sft','new')]:
                run = root/name; run.mkdir()
                (run/'summary.json').write_text(json.dumps({'ids':[s['id'] for s in specs],
                    'epochs':1,'model':name,'status':'success','scoring_version':version}))
            with self.assertRaisesRegex(ValueError, 'scoring_version'):
                prepare(SimpleNamespace(base=root/'base',sft=root/'sft',output=root/'packet'))

    def test_no_events_is_not_certainty_and_missing_is_unknown(self):
        self.assertIsNone(rate([])['rate'])
        self.assertGreater(rate([False]*8)['ci95'][1],.3)
        result=classification([(False,True),(True,False),(None,False)])
        self.assertEqual(result['n'],2)
        self.assertEqual(result['false_negative_rate'],1)
        self.assertEqual(result['false_positive_rate'],1)
        self.assertIsNone(classification([(False,False)])['recall'])

    def test_label_identity_and_types(self):
        packet={'packet_id':'abc','items':[{'id':'x'}]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'labels.jsonl'
            p.write_text(json.dumps({'id':'x','packet_id':'other'}))
            with self.assertRaises(ValueError):load_labels(p,packet)
            row={'id':'x','packet_id':'abc','reviewer':'r','sides':{'A':{'adherence_0_4':True},'B':{}},'visual_winner':None}
            p.write_text(json.dumps(row))
            with self.assertRaises(ValueError):load_labels(p,packet)

    def test_missing_results_and_escaped_review(self):
        manifest,specs=suite()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model,name in zip(('Qwen/Qwen3.5-4B','artsy/sft-v1'),('base','sft')):
                run=root/name;run.mkdir()
                (run/'summary.json').write_text(json.dumps({'ids':[s['id'] for s in specs],'epochs':1,'model':model,'status':'incomplete','sample_errors':1}))
            out=root/'packet'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(SimpleNamespace(base=root/'base',sft=root/'sft',output=out))
                report(SimpleNamespace(packet=out,labels=None,output=root/'report.json'))
            data=json.loads((root/'report.json').read_text())
            self.assertEqual(data['models']['base']['slices']['all']['missing'],len(specs))
            self.assertIsNone(data['models']['base']['slices']['all']['harmful_image_among_judged_images']['rate'])
            self.assertIsNone(data['visual']['sft_win_fraction_with_half_ties'])
            packet=json.loads((out/'packet.json').read_text());packet['items'][0]['prompt']='tampered'
            (out/'packet.json').write_text(json.dumps(packet))
            with self.assertRaises(ValueError):report(SimpleNamespace(packet=out,labels=None,output=root/'report.json'))

if __name__=='__main__':unittest.main()
