"""Blind three-model preference review of matched, saved evaluation images."""
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import threading

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'Base': 'base-qwen3.8-27b-luna-judge-2026-10-30',
    'Parent SFT': 'sft-final-preview-qwen3.8-27b-luna-judge-2026-10-30',
    'New GRPO': 'grpo-subset200-final-2026-11-01',
}
LOCK = threading.RLock()


class Review:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.folder = self.root/'eval-results/human-preferences'
        self.cases = {}
        self.excluded = 0
        source = self.root/'evaluation/generated-50-2026-10-28/prompts.jsonl'
        for spec in map(json.loads, source.read_text().splitlines()):
            if spec.get('kind') == 'refusal':
                continue
            images = {}
            for label, run in RUNS.items():
                images[label] = None
                for repeat in range(1, 4):
                    folder = self.root/'eval-results'/run/'artifacts'/f'{spec["id"]}-{repeat}'
                    result = folder/'result.json'
                    if not result.is_file():
                        continue
                    row = json.loads(result.read_text())
                    scores = row.get('scores', {})
                    image = folder/'image.png'
                    if (row.get('prompt') != spec['prompt'] or row.get('seed') != spec['seed']
                            or not scores.get('render_valid') or scores.get('image_harmful')
                            or scores.get('code_malicious') or not image.is_file()
                            or hashlib.sha256(image.read_bytes()).hexdigest() != row.get('imageHash')):
                        continue
                    images[label] = image
                    break
            # One case per prompt; bind votes to the exact selected images.
            fingerprint = ':'.join(hashlib.sha256(p.read_bytes()).hexdigest() if p else 'unavailable'
                                   for p in images.values())
            ident = hashlib.sha256(f'unique-v2:{spec["id"]}:{fingerprint}'.encode()).hexdigest()[:24]
            self.cases[ident] = {'prompt': spec['prompt'], 'prompt_id': spec['id'], 'images': images}

    def state(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder/'rankings.json'
        if path.exists():
            return json.loads(path.read_text())
        return {'version': 1, 'secret': secrets.token_hex(32), 'reviewers': {}}

    def write(self, state):
        path = self.folder/'rankings.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(path)

    def reviewer(self, name):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError('Enter a reviewer name (1–80 characters).')
        return name.strip()

    def order(self, state, name, case):
        return sorted(RUNS, key=lambda model: hmac.new(bytes.fromhex(state['secret']),
                      json.dumps([name, case, model]).encode(), hashlib.sha256).digest())

    def packet(self, name):
        name = self.reviewer(name)
        with LOCK:
            state = self.state()
            self.write(state)
            reviewer = state['reviewers'].get(name, {})
            votes = reviewer.get('votes', {})
            return {'cases': [{'id': ident, 'prompt': case['prompt'], 'ranks': votes.get(ident),
                                         'available': [case['images'][label] is not None for label in self.order(state, name, ident)]}
                              for ident, case in self.cases.items()],
                    'excluded': self.excluded, 'completed': sum(k in self.cases for k in votes),
                    'sealed': reviewer.get('sealed', False) and all(k in votes for k in self.cases)}

    def save(self, name, ident, ranks):
        name = self.reviewer(name)
        if ident not in self.cases:
            raise ValueError('Unknown comparison.')
        if (not isinstance(ranks, list) or len(ranks) != 3 or
                any(type(v) is not int or v not in (1, 2, 3) for v in ranks) or min(ranks) != 1):
            raise ValueError('Rank every image from 1 (best) to 3. Ties allowed; include rank 1.')
        with LOCK:
            state = self.state()
            reviewer = state['reviewers'].setdefault(name, {'votes': {}})
            if reviewer.get('sealed') and all(k in reviewer['votes'] for k in self.cases):
                raise ValueError('Scores revealed: this reviewer’s rankings are locked.')
            reviewer['votes'][ident] = ranks
            self.write(state)
        return {'saved': True}

    def image(self, name, ident, slot):
        name = self.reviewer(name)
        if ident not in self.cases or slot not in (0, 1, 2):
            raise ValueError('Unknown image.')
        with LOCK:
            state = self.state()
            self.write(state)
            label = self.order(state, name, ident)[slot]
            image = self.cases[ident]['images'][label]
            if image is None:
                raise ValueError('No valid saved image available.')
            return image.read_bytes()

    def results(self, name):
        name = self.reviewer(name)
        with LOCK:
            state = self.state()
            reviewer = state['reviewers'].get(name, {})
            votes = reviewer.get('votes', {})
            if not self.cases or not all(k in votes for k in self.cases):
                raise ValueError('Rank every comparison before revealing model scores.')
            reviewer['sealed'] = True
            self.write(state)
            # Average repetitions within each prompt so prompts carry equal weight.
            scores = {label: {} for label in RUNS}
            for ident, case in self.cases.items():
                ranks = votes[ident]
                for i, label in enumerate(self.order(state, name, ident)):
                    points = sum(1 if ranks[i] < ranks[j] else .5 if ranks[i] == ranks[j] else 0
                                 for j in range(3) if j != i)/2
                    scores[label].setdefault(case['prompt_id'], []).append(points)
            result = {'reviewer': name, 'comparisons': len(self.cases),
                      'prompts': len(next(iter(scores.values()))),
                      'models': [{'model': label, 'preference_score': 100*sum(
                          sum(v)/len(v) for v in prompts.values())/len(prompts)}
                          for label, prompts in scores.items()]}
            (self.folder/(hashlib.sha256(name.encode()).hexdigest()+'.scores.json')).write_text(
                json.dumps(result, indent=2)+'\n')
            return result
