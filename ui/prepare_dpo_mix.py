"""Prepare 50 reused SFT + 50 model-authored novel drawing prompts, never drawings.

Run explicitly: python -m ui.prepare_dpo_mix --output datasets/dpo-mixed-v1
Interrupted authoring resumes from verified slot checkpoints in the output folder.
"""
import argparse
import asyncio
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

from drawing.dataset_client import ChatClient, OPENROUTER_MODEL_URL, endpoint_api_key, parse_json_reply
from drawing.dataset_specs import validate_specs, DIFFICULTIES
from drawing.llm_prompts import AUTHOR_SYSTEM, GENERATOR_VERSION, generate_llm_prompts
from evaluation.suite import suite


def digest(value):
    return hashlib.sha256(value).hexdigest()


def normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def assert_novel(prompt, prior):
    current = normalized(prompt)
    if not current:
        raise ValueError('Empty authored prompt')
    words = set(current.split())
    for old in prior:
        other = normalized(old)
        tokens = set(other.split())
        if (current == other or SequenceMatcher(None, current, other, autojunk=False).ratio() >= .85
                or len(words & tokens) / max(1, len(words | tokens)) > .8):
            raise ValueError('Authored prompt duplicates or closely resembles an existing prompt; create a different scene and composition')


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def choose_reused(source, fresh, frozen, count=50):
    blocked = {normalized(p['prompt']) for p in frozen}
    eligible = [p for p in source if p.get('kind') != 'refusal'
                and p.get('benchmarkSlice', 'drawing') == 'drawing'
                and normalized(p['prompt']) not in blocked]
    available = {d: sorted([p for p in eligible if p.get('difficulty') == d], key=lambda p: p['id']) for d in DIFFICULTIES}
    fresh_counts = Counter(p.get('difficulty') for p in fresh)
    target = {d: max(0, 25 - fresh_counts[d]) for d in DIFFICULTIES}
    chosen, families, prompts = [], Counter(), set()
    for _ in range(count):
        pool = [p for d, rows in available.items() for p in rows if normalized(p['prompt']) not in prompts]
        if not pool:
            raise ValueError('Not enough distinct eligible SFT drawing prompts')
        difficulty_counts = Counter(p.get('difficulty') for p in chosen)
        best = min(pool, key=lambda p: (difficulty_counts[p['difficulty']] >= target[p['difficulty']],
            families[p['family']], difficulty_counts[p['difficulty']] - target[p['difficulty']], p['id']))
        chosen.append(best)
        families[best['family']] += 1
        prompts.add(normalized(best['prompt']))
        available[best['difficulty']].remove(best)
    return chosen


class NovelAuthor:
    def __init__(self, client, old, completed, directory):
        self.client, self.old, self.directory = client, old, directory
        self.fresh = [r['spec']['prompt'] for r in completed.values()]
        self.lock = asyncio.Lock()
        self.calls = max([int(p.stem.rsplit('-', 1)[1]) for p in directory.glob('author-response-*.json')] or [0])

    async def complete(self, messages, max_tokens):
        request = json.loads(messages[-1]['content'])
        slot = request['slots'][0]
        ranked = sorted(self.old, key=lambda p: (p.get('contentTheme') != slot.get('contentTheme'), p.get('family') != slot.get('family'), p['id']))
        request['recentPromptsToAvoid'] = list(dict.fromkeys([p['prompt'] for p in ranked[:12]] + request.get('recentPromptsToAvoid', [])))
        sent = [*messages[:-1], {**messages[-1], 'content': json.dumps(request, ensure_ascii=False)}]
        self.calls += 1
        call = self.calls
        response = await self.client.complete(sent, max_tokens)
        # Save provider text/reasoning before novelty rejection, so no paid answer vanishes.
        atomic_json(self.directory / f'author-response-{call:05d}.json', {'request': sent, 'response': response})
        reply = parse_json_reply(response['text'])
        items = reply.get('items') if isinstance(reply, dict) else None
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
            raise ValueError('Author must return one prompt item')
        async with self.lock:
            prompt = items[0].get('prompt')
            if not isinstance(prompt, str):
                raise ValueError('Author prompt must be text')
            assert_novel(prompt, [p['prompt'] for p in self.old] + self.fresh)
            self.fresh.append(prompt)
        return response


async def prepare(source, output, client, *, seed=44, concurrency=8, max_tokens=4096, frozen=None):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    rows = validate_specs([json.loads(line) for line in raw.splitlines() if line.strip()])
    if frozen is None:
        _, frozen = suite()
    output.mkdir(parents=True, exist_ok=True)
    directory = output / 'authoring'
    directory.mkdir(exist_ok=True)
    config = {'source_sha256': digest(raw), 'evaluation_sha256': digest(json.dumps(frozen, sort_keys=True).encode()),
        'author_model': client.model, 'author_url': client.baseUrl, 'generator_version': GENERATOR_VERSION,
        'instructions_sha256': digest(AUTHOR_SYSTEM.encode()), 'implementation_sha256': digest(Path(__file__).read_bytes()),
        'fresh_count': 50, 'reused_count': 50, 'seed': seed, 'concurrency': concurrency, 'max_tokens': max_tokens,
        'difficulty': 'mixed', 'harmful_count': 0, 'benign_boundary_count': 0, 'malicious_code_count': 0}
    config_file = directory / 'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError('Author checkpoint configuration changed; choose a new output directory')
    atomic_json(config_file, config)
    completed = {}
    prior = [p['prompt'] for p in rows + frozen]
    for file in sorted(directory.glob('record-*.json')):
        record = json.loads(file.read_text())
        assert_novel(record['spec']['prompt'], prior)
        prior.append(record['spec']['prompt'])
        if record['index'] in completed:
            raise ValueError('Duplicate author checkpoint slot')
        completed[record['index']] = record
    def save_record(record):
        atomic_json(directory / f"record-{record['index']:03d}.json", record)
    def save_attempt(record):
        with (directory / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    author = NovelAuthor(client, rows + frozen, completed, directory)
    generated = await generate_llm_prompts(author, 50, seed, difficulty='mixed', harmful_count=0,
        benign_boundary_count=0, malicious_code_count=0, batch_size=concurrency, max_tokens=max_tokens,
        completed=completed, on_record=save_record, on_attempt=save_attempt)
    fresh = generated['specs']
    if len(fresh) != 50:
        raise ValueError('Exactly 50 freshly authored prompts required')
    prior = [p['prompt'] for p in rows + frozen]
    for spec in fresh:
        assert_novel(spec['prompt'], prior)
        prior.append(spec['prompt'])
    reused = choose_reused(rows, fresh, frozen)
    used_ids = {p['id'] for p in reused}
    origins = {p['id']: {'source': 'sft', 'source_id': p['id'], 'source_sha256': config['source_sha256']} for p in reused}
    for spec in fresh:
        source_id = spec['id']
        if spec['id'] in used_ids:
            spec['id'] = 'fresh-' + spec['id']
        if spec['id'] in used_ids:
            raise ValueError('Prompt ID collision')
        used_ids.add(spec['id'])
        batch = next(b for b in generated['batches'] if b['indices'] == [fresh.index(spec)])
        origins[spec['id']] = {'source': 'fresh', 'source_id': source_id, 'author_model': batch['completion'].get('model', client.model),
            'author_url': client.baseUrl, 'instructions_sha256': config['instructions_sha256'],
            'completion_sha256': digest(json.dumps(batch['completion'], sort_keys=True).encode()),
            'source_sha256': digest(json.dumps(batch['completion'], sort_keys=True).encode())}
    final = validate_specs(reused + fresh)
    if len(final) != 100 or len(origins) != 100:
        raise ValueError('Mix must contain exactly 100 unique prompts')
    payload = ''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in final)
    manifest = {**config, 'dataset_sha256': digest(payload.encode()), 'origins': origins,
        'difficulty_counts': dict(Counter(p.get('difficulty') for p in final)),
        'family_counts': dict(Counter(p['family'] for p in final)), 'fresh_audit': generated['audit']}
    temporary = output / 'prompts.jsonl.tmp'
    temporary.write_text(payload)
    temporary.replace(output / 'prompts.jsonl')
    atomic_json(output / 'mix-manifest.json', manifest)
    return manifest


def main():
    from drawing.generate_dataset import _load_env_file
    _load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('datasets/sft-v2/prompts.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('datasets/dpo-mixed-v1'))
    parser.add_argument('--fresh-seed', type=int, default=44)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=4096)
    args = parser.parse_args()
    model = os.getenv('ARTSY_PROMPT_MODEL', 'openai/gpt-6-luna')
    url = os.getenv('ARTSY_PROMPT_URL', OPENROUTER_MODEL_URL)
    client = ChatClient(model, url, endpoint_api_key(url, os.getenv('ARTSY_PROMPT_API_KEY')), timeout_ms=180000)
    manifest = asyncio.run(prepare(args.source, args.output, client, seed=args.fresh_seed,
        concurrency=args.concurrency, max_tokens=args.max_tokens))
    print(json.dumps({'prompts': 100, 'fresh': 50, 'reused': 50, 'dataset_sha256': manifest['dataset_sha256'],
                      'difficulty_counts': manifest['difficulty_counts']}, indent=2))


if __name__ == '__main__':
    main()
