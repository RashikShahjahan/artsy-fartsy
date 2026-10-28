import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from evaluation.benchmark import suite, rate, classification, load_run, load_labels, prepare, report

class BenchmarkTests(unittest.TestCase):
    def test_frozen_mix_and_difficulty(self):
        manifest, rows=suite()
        self.assertEqual(manifest['counts'],dict(drawing=56,benign_boundary=8,harmful_image=8,malicious_code=8))
        for level in ('easy','medium','hard','very-hard'):
            self.assertEqual(sum(r['benchmarkSlice']=='drawing' and r['difficulty']==level for r in rows),14)

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
            for model,name in zip(manifest['models'],('base','sft')):
                run=root/name;run.mkdir()
                (run/'summary.json').write_text(json.dumps({'ids':[s['id'] for s in specs],'epochs':1,'model':model,'status':'incomplete','sample_errors':1}))
            out=root/'packet'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(SimpleNamespace(base=root/'base',sft=root/'sft',output=out))
                report(SimpleNamespace(packet=out,labels=None,output=root/'report.json'))
            data=json.loads((root/'report.json').read_text())
            self.assertEqual(data['models']['base']['slices']['all']['missing'],80)
            self.assertIsNone(data['models']['base']['slices']['all']['harmful_image_among_judged_images']['rate'])
            self.assertIsNone(data['visual']['sft_win_fraction_with_half_ties'])
            packet=json.loads((out/'packet.json').read_text());packet['items'][0]['prompt']='tampered'
            (out/'packet.json').write_text(json.dumps(packet))
            with self.assertRaises(ValueError):report(SimpleNamespace(packet=out,labels=None,output=root/'report.json'))

if __name__=='__main__':unittest.main()
