"""Compare completed HPSv3 Inspect runs with matched prompts and delivery rates."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean


def compare(runs: dict[str, Path], output: Path, *, score_records: Path | None = None) -> dict:
    summaries = {name: json.loads((path / 'summary.json').read_text()) for name, path in runs.items()}
    first = dict(next(iter(summaries.values())))
    if score_records is not None:
        first['hpsv3_model'] = None
    for summary in summaries.values():
        if summary['status'] != 'success' or summary.get('sample_errors'):
            raise ValueError('Resolve failed Inspect samples before comparison')
        for field in ('ids', 'epochs', 'hpsv3_model', 'max_tokens', 'max_turns', 'token_limit', 'time_limit'):
            if field == 'hpsv3_model' and score_records is not None:
                continue
            if summary.get(field) != first.get(field):
                raise ValueError(f'Comparison settings differ: {field}')
        if not summary.get('hpsv3_model') and score_records is None:
            raise ValueError('Expected HPSv3-scored runs')
    rows, by_model = [], {}
    for name, path in runs.items():
        results = {}
        for prompt_id in first['ids']:
            for epoch in range(1, first['epochs'] + 1):
                result = json.loads((path / 'artifacts' / f'{prompt_id}-{epoch}' / 'result.json').read_text())
                if score_records is not None:
                    saved = json.loads((score_records / name / f'{prompt_id}-{epoch}.json').read_text())
                    if saved['prompt'] != result['prompt'] or saved.get('imageHash') != result.get('imageHash'):
                        raise ValueError('Saved HPSv3 score does not match the generated image')
                    result = {**result, 'scores': saved['scores'], 'hpsv3Model': saved['hpsv3Model']}
                scores = result['scores']
                valid = bool(scores['render_valid'])
                raw = scores.get('hpsv3_score')
                if valid and not result.get('expectedRefusal') and (raw is None or not math.isfinite(raw)):
                    raise ValueError(f'Missing raw HPSv3 score for valid image: {name}/{prompt_id}')
                if score_records is not None and not first.get('hpsv3_model'):
                    first['hpsv3_model'] = result.get('hpsv3Model')
                if result.get('hpsv3Model') != first['hpsv3_model']:
                    raise ValueError('Artifact scorer model differs from run metadata')
                row = {'model': name, 'id': prompt_id, 'epoch': epoch, 'prompt': result['prompt'],
                       'outcome': result['outcome'], 'render_valid': int(valid), 'hpsv3_score': raw,
                       'hpsv3_preference': scores['hpsv3_preference']}
                results[(prompt_id, epoch)] = row
                rows.append(row)
        by_model[name] = results
    aggregate = {}
    for name, results in by_model.items():
        values = list(results.values())
        raw = [row['hpsv3_score'] for row in values if row['hpsv3_score'] is not None]
        aggregate[name] = {'episodes': len(values), 'valid_images': sum(row['render_valid'] for row in values),
                           'raw_mean_valid_images': mean(raw) if raw else None,
                           'preference_mean_all_episodes': mean(row['hpsv3_preference'] for row in values)}
    pairs = []
    for a, b in itertools.combinations(runs, 2):
        matched = [(row['hpsv3_score'], by_model[b][key]['hpsv3_score'])
                   for key, row in by_model[a].items()
                   if row['hpsv3_score'] is not None and by_model[b][key]['hpsv3_score'] is not None]
        differences = [x - y for x, y in matched]
        pairs.append({'a': a, 'b': b, 'matched_images': len(matched),
                      'raw_mean_difference_a_minus_b': mean(differences) if differences else None,
                      'wins_a': sum(d > 0 for d in differences), 'wins_b': sum(d < 0 for d in differences),
                      'ties': sum(d == 0 for d in differences)})
    result = {'hpsv3_model': first['hpsv3_model'], 'prompts': len(first['ids']), 'epochs': first['epochs'],
              'runs': {name: str(path.resolve()) for name, path in runs.items()},
              'models': aggregate, 'matched_image_pairs': pairs,
              'interpretation': 'Descriptive checkpoint comparison. Raw means exclude missing images; normalized means include delivery failures as zero. Pairwise wins compare only matched delivered images. No training-seed replication.'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=OUTPUT_DIR')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        name, separator, path = value.partition('=')
        if not separator or not name or not path or name in runs:
            parser.error('Use unique NAME=OUTPUT_DIR entries')
        runs[name] = Path(path)
    if len(runs) < 2:
        parser.error('Provide at least two runs')
    print(json.dumps(compare(runs, args.output), indent=2))


if __name__ == '__main__':
    main()
