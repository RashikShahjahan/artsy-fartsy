"""Portable DPO review exports and local API integrity, without model calls."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from training.data import ROOT, digest
from training.dpo import load_preferences
from ui.dpo_review import PAGE, make_handler


class DPOReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'candidates.jsonl'
        self.output = self.root / 'export'
        self.records = []
        for rid, color in [('a', 'red'), ('b', 'blue'), ('c', 'green')]:
            directory = self.root / rid
            directory.mkdir()
            code = f'draw({color!r})\n'
            (directory / 'program.py').write_text(code)
            for name in ('image.png', 'preview-1.png'):
                Image.new('RGB', (24, 16), color).save(directory / name)
            self.records.append({'id': rid, 'status': 'accepted',
                'spec': {'prompt': '</script><script>alert(1)</script>', 'seed': 1},
                'runtimeVersion': 'artcanvas-v2', 'apiDocHash': digest((ROOT / 'docs/artcanvas.md').read_bytes()),
                'artifacts': {'directory': rid, 'source': rid+'/program.py', 'image': rid+'/image.png'},
                'code': code, 'sourceHash': digest(code.strip().encode()),
                'imageHash': digest((directory / 'image.png').read_bytes()),
                'trace': [{'completion': {'text': json.dumps({'code': code})}, 'executedCode': code,
                    'feedback': {'ok': True, 'exitCode': 0, 'artifact': '/old/preview-1.png'}}]})
        self.write()
        self.handler = make_handler(self.path, 4173, self.output)

    def write(self):
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in self.records))

    def request(self, method, path, body=None, host='localhost:4173'):
        handler = object.__new__(self.handler)
        handler.path = path
        raw = json.dumps(body).encode() if body is not None else b''
        handler.headers = {'Host': host, 'Content-Length': str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        result = {}
        handler.respond = lambda status, payload, mime='application/json': result.update(status=status, payload=payload, mime=mime)
        getattr(handler, 'do_'+method)()
        return result

    def state(self):
        r = self.request('GET', '/api/state')
        self.assertEqual(r['status'], 200, r)
        return r['payload']

    def body(self, **decision):
        s = self.state()
        return {'dataset_sha256': s['sha256'], 'approval_sha256': s['approval_sha256'],
            'revision': s['revision'], 'reviewer': 'Human', 'decisions': {'a / b': {
                'record_a': 'a', 'record_b': 'b', 'winner': 'A', 'eligible': True,
                'group_id': 'circles', 'split': 'train', **decision}}}

    def test_export_loads_real_trajectories_and_remains_portable(self):
        original = self.path.read_bytes()
        self.assertFalse(self.output.exists())
        body = self.body()
        self.assertEqual(self.request('POST', '/api/review', body)['status'], 200)
        body['revision'] = self.state()['revision']
        result = self.request('POST', '/api/export', body)
        self.assertEqual(result['status'], 200, result)
        examples, manifest = load_preferences(self.output/'labels.jsonl', self.output/'accepted.jsonl')
        self.assertEqual(len(examples), 1)
        self.assertEqual(manifest['records'], 2)
        self.assertEqual(self.path.read_bytes(), original)
        # Removing original episode files proves that observations and source are copied.
        import shutil
        for rid in ('a', 'b'): shutil.rmtree(self.root/rid)
        self.assertEqual(len(load_preferences(self.output/'labels.jsonl', self.output/'accepted.jsonl')[0]), 1)

    def test_revision_stale_dataset_and_local_host(self):
        body = self.body()
        self.assertEqual(self.request('POST', '/api/review', body)['status'], 200)
        self.assertEqual(self.request('POST', '/api/review', body)['status'], 409)
        self.path.write_text(self.path.read_text()+'\n')
        self.assertEqual(self.request('GET', '/api/state')['status'], 400)
        self.assertEqual(self.request('GET', '/', host='evil.test')['status'], 403)

    def test_invalid_eligibility_identical_pairs_and_leakage(self):
        for changes in ({'eligible': False}, {'record_b': 'a'}, {'group_id': ''}):
            with self.subTest(changes=changes):
                self.assertEqual(self.request('POST', '/api/review', self.body(**changes))['status'], 400)
        body = self.body()
        body['decisions']['a / c'] = {**body['decisions']['a / b'], 'record_b': 'c', 'split': 'test'}
        self.assertEqual(self.request('POST', '/api/review', body)['status'], 400)

    def test_tie_only_export_fails_before_publish(self):
        body = self.body(winner='Tie')
        self.assertEqual(self.request('POST', '/api/review', body)['status'], 200)
        body['revision'] = self.state()['revision']
        result = self.request('POST', '/api/export', body)
        self.assertEqual(result['status'], 400)
        self.assertIn('non-tie training', result['payload']['error'])
        self.assertFalse((self.output/'labels.jsonl').exists())

    def test_xss_and_allowlisted_image(self):
        result = self.request('GET', '/')
        self.assertNotIn(b'</script><script>alert', result['payload'])
        self.assertEqual(self.request('GET', '/api/image?id=../../etc/passwd')['status'], 404)
        self.assertEqual(self.request('GET', '/api/image?id=a')['mime'], 'image/png')

    def test_manual_approval_rebase_preserves_decisions_and_rejects_stale_clients(self):
        body = self.body()
        self.assertEqual(self.request('POST', '/api/review', body)['status'], 200)
        stale = {**body, 'revision': self.state()['revision']}
        review = {'dataset_sha256': digest(self.path.read_bytes()), 'reviewer': 'Curator',
                  'decisions': {'c': {'decision': 'approved'}}}
        (self.root/'manual-review.json').write_text(json.dumps(review))
        self.assertEqual(self.request('GET', '/api/state')['status'], 400)
        self.handler = make_handler(self.path, 4173, self.output)
        rebased = self.state()
        self.assertIn('a / b', rebased['state']['decisions'])
        self.assertEqual(rebased['state']['approval_sha256'], rebased['approval_sha256'])
        self.assertEqual(self.request('POST', '/api/review', stale)['status'], 409)
        review['decisions']['a'] = {'decision': 'rejected'}
        (self.root/'manual-review.json').write_text(json.dumps(review))
        self.handler = make_handler(self.path, 4173, self.output)
        result = self.request('GET', '/api/state')
        self.assertEqual(result['status'], 400)
        self.assertIn('saved decisions were retained', result['payload']['error'])

    def test_collection_preset_is_exact_unique_balanced_and_deterministic(self):
        import shutil
        import subprocess
        if not shutil.which('node'):
            self.skipTest('Node is required to execute the collection selection helper')
        helper = PAGE.split('function selectCollectionPrompts', 1)[1].split('let collectionData=', 1)[0]
        script = 'function selectCollectionPrompts' + helper + r"""
const assert=require('node:assert/strict');
const items=[];
for(let family=0;family<5;family++)for(let difficulty=0;difficulty<2;difficulty++)for(let n=0;n<30;n++)items.push({id:`${family}-${difficulty}-${n}`,family:String(family),difficulty:String(difficulty)});
const ids=selectCollectionPrompts(items,100);
assert.equal(ids.length,100);assert.equal(new Set(ids).size,100);
assert.deepEqual(ids,selectCollectionPrompts(items,100));
const counts={};for(const id of ids){const key=id.split('-').slice(0,2).join('-');counts[key]=(counts[key]||0)+1}
assert.deepEqual(Object.values(counts),Array(10).fill(10));
assert.equal(selectCollectionPrompts(items.slice(0,7),100).length,7);
assert.equal(selectCollectionPrompts([...items,...items],100).length,100);
"""
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('100 prompts × 2 candidates = 200 trajectories', PAGE)
        self.assertIn('at most 100 comparison pairs', PAGE)
        self.assertIn('preset.disabled=available<100', PAGE)

    def test_unapproved_not_fabricated_and_no_candidates_supported(self):
        self.records[0]['status'] = 'review'
        self.write()
        self.handler = make_handler(self.path, 4173, self.output)
        self.assertNotIn('a', [r['id'] for r in self.state()['items']])
        self.handler = make_handler(None, 4173, self.output)
        self.assertEqual(self.state()['items'], [])


if __name__ == '__main__':
    unittest.main()
