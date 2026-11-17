"""Render the model-development chart pack. Requires matplotlib and numpy.

Default: render the checked-in data snapshot without local training artifacts.
--refresh-data: rebuild that snapshot from local experiment records and logs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/model-development'


def snapshot():
    sources = {}

    def read(path):
        p = ROOT / path
        raw = p.read_bytes()
        sources[path] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    def comparison(name):
        d = json.loads(read(f'eval-results/{name}/comparison.json'))
        return {k: v for k, v in d.items() if k != 'runs'}

    def metrics(name):
        rows = [json.loads(x) for x in read(f'checkpoints/{name}/metrics.jsonl').splitlines()]
        # Resumed runs can contain superseded steps: retain last saved occurrence.
        by_step = {r['step']: r for r in rows}
        return [{k: r[k] for k in ['step', 'loss:sum', 'env/all/reward', 'env/all/render_valid'] if k in r}
                for _, r in sorted(by_step.items())]

    final = comparison('experiment10-hard150-12turns-comparison/hpsv3')
    records = list(csv.DictReader(read('eval-results/experiment10-hard150-12turns-comparison/hpsv3/scores.csv').splitlines()))
    ids = {r['id'] for r in records}
    assert len(records) == 750 and len(ids) == 150
    for model, summary in final['models'].items():
        rows = [r for r in records if r['model'] == model]
        assert len(rows) == 150 and {r['id'] for r in rows} == ids
        assert sum(int(r['render_valid']) for r in rows) == summary['valid_images']
        assert abs(sum(float(r['hpsv3_preference']) for r in rows) / 150 - summary['preference_mean_all_episodes']) < 1e-10
    delivered = {m: {r['id']: float(r['hpsv3_score']) for r in records
                     if r['model'] == m and int(r['render_valid'])} for m in final['models']}
    for pair in final['matched_image_pairs']:
        a, b = delivered[pair['a']], delivered[pair['b']]
        shared = a.keys() & b.keys()
        assert len(shared) == pair['matched_images']
        assert sum(a[i] > b[i] for i in shared) == pair['wins_a']
        assert sum(a[i] < b[i] for i in shared) == pair['wins_b']
    prompts = [json.loads(x) for x in read('datasets/experiment9-genai/evaluation-hard/prompts.jsonl').splitlines()]
    skill = {p['id']: p['primarySkill'] for p in prompts}
    categories = sorted(set(skill.values()))
    category = {m: {c: [sum(int(r['render_valid']) for r in records if r['model'] == m and skill[r['id']] == c),
                         sum(1 for r in records if r['model'] == m and skill[r['id']] == c)]
                    for c in categories} for m in final['models']}
    for row in category.values():
        assert all(n == 50 for _, n in row.values())
    # Original metrics were cleaned up; recover completed updates from console tables.
    log = read('checkpoints/rlhf-experiment10-genai1000-linear-epoch1.console.log')
    original = {}
    for match in re.finditer(r'(?m)^\s*Step (\d+)\s*$', log):
        block = log[match.end():match.end()+11000]
        values = dict(re.findall(r'│\s*(env/all/(?:reward|render_valid))\s*│\s*([-\d.]+)', block))
        if len(values) == 2:
            original[int(match[1])] = {'step': int(match[1]), **{k: float(v) for k, v in values.items()}}
    recovery = metrics('rlhf-experiment10-genai1000-linear-epoch1-kl-budget-recovery')
    original.update({r['step']: r for r in recovery})
    assert sorted(original) == list(range(32)), sorted(original)
    read('EXPERIMENT_LOG.md')
    data = dict(final=final, categories=category,
                intermediate=comparison('rlvr-g4-epoch2-parti-hpsv3'),
                hard25=comparison('experiment9-hard150-hpsv3'),
                ranks3=comparison('experiment3-hpsv3'),
                sft=metrics('parti-sft-artcanvas-full-2026-11-07-qwen3.5-4b-r128-b32'),
                rlvr=metrics('rlvr-experiment9-genai681-qwen3.5-4b-r128-b32-g4'),
                rlhf=list(original.values()),
                # Historical numerical tables transcribed from the experiment log.
                historical={'rank2_adherence': [2.563, 2.628, 2.535],
                            'rank2_visual': [2.501, 2.530, 2.451],
                            'human': [46.6, 47.7, 55.7],
                            'human_ci': [[40.5, 52.8], [42.3, 53.1], [50.5, 60.9]],
                            'harness_valid': [48, 97, 56, 56],
                            'recovery_valid': [[56, 91, 122], [56, 96, 113]]},
                sources=sources)
    # Assert final model metadata rather than inheriting the stale pending log status.
    final_meta = json.loads(read('checkpoints/rlhf-experiment10-genai1000-linear-epoch1-kl-budget-recovery/final.json'))
    assert final_meta['step'] == 32
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data.json').write_text(json.dumps(data, indent=2) + '\n')
    return data


def render(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#dce2e6',
                         'axes.labelcolor': '#425364', 'xtick.color': '#425364',
                         'ytick.color': '#425364', 'text.color': '#182c3c',
                         'figure.facecolor': '#f7f9fb', 'axes.facecolor': '#f7f9fb',
                         'svg.fonttype': 'none'})
    keys = ['base_12turns', 'sft_12turns', 'rlvr_12turns', 'rlhf300_12turns', 'rlhf1000_12turns']
    labels = ['Base', 'SFT', 'RLVR · 681', 'RLHF · 300', 'RLHF · 1,000']
    colors = ['#91a0ae', '#3978af', '#15958b', '#9370b5', '#d87535']
    models = [d['final']['models'][k] for k in keys]
    saved = []

    def start(title, subtitle, n=1, size=(13, 6.8)):
        fig, ax = plt.subplots(1, n, figsize=size)
        fig.suptitle(title, x=.06, y=.96, ha='left', fontsize=22, weight='bold')
        fig.text(.06, .885, subtitle, fontsize=11, color='#526678')
        fig.subplots_adjust(left=.08, right=.95, top=.78, bottom=.22, wspace=.35)
        return fig, ax

    def finish(fig, name, note):
        fig.text(.06, .065, note, fontsize=9, color='#526678', linespacing=1.6)
        for ext in ['png', 'svg']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight', pad_inches=.3)
        saved.append(name)
        plt.close(fig)

    def bars(ax, values, names=labels, palette=colors, max_x=None, fmt=lambda v: f'{v:.1f}'):
        y = np.arange(len(values))
        ax.barh(y, values, color=palette, height=.58)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        if max_x:
            ax.set_xlim(0, max_x)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, alpha=.15)
        ax.tick_params(axis='y', length=0)
        for i, v in enumerate(values):
            ax.text(v + (ax.get_xlim()[1] * .015), i, fmt(v), va='center', fontsize=10)

    fig, axes = start('Final models: reliable delivery, stronger preference scores',
                      'Same 150 hard GenAI-Bench prompts • mini-SWE • 12 turns • one generation per prompt', 2)
    bars(axes[0], [m['valid_images'] / 150 * 100 for m in models], max_x=115, fmt=lambda v: f'{v:.1f}%')
    axes[0].set(title='Valid final images', xlabel='Render rate (%)', xticks=[0, 25, 50, 75, 100])
    for i, m in enumerate(models):
        axes[0].text(3, i, f"{m['valid_images']}/150", va='center', color='white', weight='bold')
    bars(axes[1], [m['preference_mean_all_episodes'] for m in models], max_x=.65, fmt=lambda v: f'{v:.3f}')
    axes[1].set(title='Delivery-adjusted HPSv3', xlabel='Mean normalized reward (failures = 0)')
    finish(fig, '01-final-models', 'Source: completed Experiment 10 comparison, 14 Nov 2026. Normalization: 0.02 + 0.98 × clip((raw + 11)/14, 0, 1).\nQuantized MLX HPSv3 is a proxy scorer, not a calibrated human rating. Descriptive checkpoint comparison; one training seed.')

    fig, ax = start('Final RLHF improves over RLVR; RLHF branches remain close',
                    'HPSv3 preference on prompts where both checkpoints delivered an image')
    pairs = [p for p in d['final']['matched_image_pairs'] if p['a'] == keys[-1]]
    names = ['vs Base', 'vs SFT', 'vs RLVR · 681', 'vs RLHF · 300']
    for i, p in enumerate(pairs):
        n = p['matched_images']; win = p['wins_a'] / n * 100
        ax.barh(i, win, color=colors[-1], height=.6)
        ax.barh(i, 100-win, left=win, color='#dce2e6', height=.6)
        ax.text(min(win/2, 80), i, f"{p['wins_a']} wins", va='center', ha='center', color='white', weight='bold')
        ax.text(102, i, f"{win:.1f}%  ·  {n} pairs  ·  {p['wins_b']} losses", va='center')
    ax.set(yticks=range(4), yticklabels=names, xlim=(0, 143), xticks=[0, 25, 50, 75, 100], xlabel='Final RLHF win share (%)')
    ax.invert_yaxis(); ax.axvline(50, color='#425364', ls='--', lw=1)
    finish(fig, '02-final-pairwise', 'Orange = RLHF trained on 1,000 prompts; gray = comparator wins. No ties in these pairs; failed deliveries are excluded.\n72–71 versus RLHF · 300 is an almost even split. These counts do not establish a general advantage across seeds or human judges.')

    category_keys = sorted(d['categories'][keys[0]])
    display_categories = [c.replace('_', ' ').title() for c in category_keys]
    values = np.array([[d['categories'][m][c][0] / d['categories'][m][c][1] * 100 for c in category_keys] for m in keys])
    fig, ax = start('Execution gains extend across the three hard-prompt groups',
                    'Matched 12-turn evaluation • 50 prompts per category • valid final images')
    ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    ax.set(xticks=range(3), xticklabels=display_categories, yticks=range(5), yticklabels=labels)
    ax.tick_params(length=0)
    for i in range(5):
        for j in range(3):
            count = d['categories'][keys[i]][category_keys[j]][0]
            ax.text(j, i, f'{count}/50  ·  {values[i,j]:.0f}%', ha='center', va='center', color='white' if values[i,j] > 65 else '#182c3c', weight='bold')
    finish(fig, '03-category-reliability', 'Source: per-episode Experiment 10 scores joined to primarySkill in the frozen 150-prompt dataset.\nA valid image measures successful delivery; it does not establish correct counting, spatial relations, or visual adherence.')

    fig, axes = start('Training progress through SFT, RLVR and final RLHF',
                      'Observed training batches • separate objectives and datasets • no held-out learning curve', 3, (15, 7))
    sft = d['sft']; rlvr = d['rlvr']; rlhf = sorted(d['rlhf'], key=lambda r:r['step'])
    axes[0].plot([r['step'] for r in sft], [r['loss:sum'] for r in sft], color=colors[1], lw=1.8)
    axes[0].set(title='SFT · 72 updates', xlabel='Optimizer update', ylabel='Logged training loss')
    axes[1].plot([r['step']+1 for r in rlvr], [100*r['env/all/render_valid'] for r in rlvr], color=colors[2], lw=1.8)
    axes[1].set(title='RLVR · 681 prompts', xlabel='Rollout batch (1-based)', ylabel='Training render rate (%)', ylim=(0,105))
    axes[2].plot([r['step']+1 for r in rlhf], [r['env/all/reward'] for r in rlhf], color=colors[-1], lw=1.8)
    axes[2].axvline(27.5, color='#526678', ls='--', lw=1)
    axes[2].text(.98, .95, 'Recovery\nupdates 28–32', transform=axes[2].transAxes, ha='right', va='top', fontsize=9)
    axes[2].set(title='RLHF · 1,000 prompts', xlabel='Rollout batch (1-based)', ylabel='Training mean image reward', ylim=(0,.6))
    for ax in axes:
        ax.grid(alpha=.16)
    finish(fig, '04-training-progress', 'SFT duplicate update records use the last occurrence. RLHF updates 1–27 were recovered from completed console metric tables;\nupdates 28–32 use recovery metrics.jsonl. Training batches vary in prompt difficulty and size; training scores are not evaluation results.')

    h = d['historical']; ranks = ['Rank 32', 'Rank 64', 'Rank 128']
    fig, axes = start('Early rank studies informed selection, with uncertain quality ordering',
                      'Experiments 1–4 • independent fresh adapters in Experiment 3 • each panel has its own denominator', 3, (15, 7))
    x = np.arange(3)
    axes[0].bar(x-.18, h['rank2_adherence'], width=.36, label='Adherence', color=colors[1])
    axes[0].bar(x+.18, h['rank2_visual'], width=.36, label='Visual quality', color=colors[2])
    axes[0].set(xticks=x, xticklabels=ranks, ylim=(0,4), title='Exp 2 · Luna ratings', ylabel='Mean rating /4 · 355 matched prompts')
    axes[0].legend(frameon=False, fontsize=9)
    human = np.array(h['human']); ci = np.array(h['human_ci'])
    axes[1].errorbar(x, human, yerr=[human-ci[:,0],ci[:,1]-human], fmt='o', color=colors[3], capsize=5, markersize=7)
    axes[1].axhline(50, color='#91a0ae', ls='--', lw=1)
    axes[1].set(xticks=x, xticklabels=ranks, ylim=(0,100), title='Exp 4 · blinded human review', ylabel='Preference /100 · original 163 prompts')
    for i,v in enumerate(human): axes[1].text(i,v+10,f'{v:.1f}',ha='center',fontsize=10)
    rank_models = [d['ranks3']['models'][f'rank{rank}'] for rank in [32, 64, 128]]
    axes[2].bar(x,[m['raw_mean_valid_images'] for m in rank_models],color=[colors[1], colors[2], colors[3]],width=.55)
    axes[2].set(xticks=x, xticklabels=ranks, ylim=(-1,0), title='Exp 3 · expanded SFT', ylabel='Raw HPSv3 · valid images (higher is better)')
    for i,m in enumerate(rank_models): axes[2].text(i,m['raw_mean_valid_images']-.07,f"{m['raw_mean_valid_images']:.3f}\nn={m['valid_images']}",ha='center',va='top',fontsize=9)
    finish(fig, '05-rank-studies', 'Exp 2: 391 historical evaluation prompts; quality differences did not survive Holm correction. Exp 4: one reviewer; pointwise 95% CIs.\nExp 3: 163 held-out prompts; raw means condition on different delivered-image subsets. The 228 added training prompts were no longer held out.')

    fig, axes = start('Harness and supervision experiments exposed submission failures',
                      'Experiments 5–6 • original 163 Parti prompts • mini-SWE harness', 2, (14, 7))
    bars(axes[0], [v/163*100 for v in h['harness_valid']],
         ['Prior mini-SWE', 'Exp 5 · revised SFT', 'Exp 6 · control', 'Exp 6 · error-aware'],
         [colors[0], colors[1], colors[2], colors[3]], max_x=78, fmt=lambda v:f'{v:.1f}%')
    axes[0].set(title='Initial evaluation · fixed 12-turn cap', xlabel='Valid final images (%)')
    for vals, color, label in zip(h['recovery_valid'], [colors[2],colors[3]], ['Control','Error-aware']):
        axes[1].plot(range(3), [v/163*100 for v in vals], marker='o', color=color, label=label, lw=2)
        for j,v in enumerate(vals): axes[1].text(j, v/163*100 + ((-7 if j == 2 else 4) if label=='Error-aware' else (4 if j == 2 else -7)), f'{v}/163',ha='center',fontsize=9)
    axes[1].set(xticks=range(3), xticklabels=['Initial 12 turns', '+16-turn retries', '+25-turn retries'], ylim=(0,100), title='Exp 6 · cumulative recovery', ylabel='Prompts with a valid image (%)')
    axes[1].legend(frameon=False,loc='upper left'); axes[1].grid(axis='y',alpha=.15)
    finish(fig, '06-harness-and-loss-masking', 'Left: separate checkpoint comparisons, not a controlled four-arm ablation. Right: only failures were retried; successful images were retained.\nCumulative recovery totals mix turn caps and attempts. Error-aware supervision did not improve initial render counts (56/163 in both arms).')

    fig, axes = start('Intermediate RLVR checkpoints: strong execution gains, mixed quality',
                      'Experiment 8 and continuation branches • compare within each panel only', 2, (15, 7))
    mid = d['intermediate']['models']
    midkeys = ['sft','rlvr32','epoch1','epoch2']
    bars(axes[0], [mid[k]['valid_images']/163*100 for k in midkeys],
         ['SFT · step 72','RLVR · 32 rollouts/update','RLVR · 128 · epoch 1','RLVR · 128 · epoch 2'],
         [colors[1],colors[2],colors[3],colors[-1]],max_x=118,fmt=lambda v:f'{v:.1f}%')
    axes[0].set(title='Parti · 163 prompts · 12 turns', xlabel='Render rate (%)',xticks=[0,25,50,75,100])
    hard = d['hard25']['models']; hardkeys = ['base','sft','parti_rlvr','rlvr']
    bars(axes[1], [hard[k]['valid_images']/150*100 for k in hardkeys],
         ['Base','SFT','RLVR · 227 prompts','RLVR · 681 prompts'],
         [colors[0],colors[1],colors[3],colors[2]],max_x=118,fmt=lambda v:f'{v:.1f}%')
    axes[1].set(title='Hard GenAI · 150 prompts · 25 turns', xlabel='Render rate (%)',xticks=[0,25,50,75,100])
    finish(fig, '07-intermediate-rlvr', 'Parti raw HPSv3 means (delivered images): SFT −2.657; 32-rollout RLVR −2.344; 128-rollout epoch 1 −2.813; epoch 2 −2.712.\nHard-set 681-vs-227 comparison also changes training sources and update count; it does not isolate dataset size. Do not pool the two benchmarks.')

    fig, ax = start('Development map: experiments, branches and the final model lineage',
                    'Qwen3.5-4B • experiment sequence is not a single continuous training run', size=(15,9))
    ax.set_axis_off(); ax.set_xlim(-.3, 12); ax.set_ylim(0, 10)
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.13',facecolor='white',edgecolor=color,linewidth=1.8))
        ax.text(x+.12,y+h-.28,title,weight='bold',fontsize=11,va='top',color=color)
        ax.text(x+.12,y+h-.85,body,fontsize=9.5,va='top',linespacing=1.6)
    box(0,6.8,3.45,2.4,'Experiments 1–4 · rank studies','Ranks 32 / 64 / 128\nOriginal and expanded SFT datasets\nLuna, HPSv3 and blinded human review',colors[1])
    box(4.1,6.8,3.45,2.4,'Experiments 5–6 · harness / loss','mini-SWE and revised teacher traces\nControl vs error-aware supervision\nSeparate fresh SFT experiments',colors[3])
    box(8.2,6.8,3.45,2.4,'Experiment 7 · prompt-only plan','Revised task prompt\nEvaluation pending in experiment log\nNo result claimed in this chart pack',colors[0])
    box(0,2.95,3.15,2.3,'Selected SFT · step 72','2,282 trajectories · rank 128\nFresh starting point for RL branches\nNot a continuation of rank studies',colors[1])
    box(4.1,3.8,3.35,1.9,'Exp 8 · Parti-only RLVR','227 prompts · 908 rollouts / epoch\n32 vs 128 rollouts/update; epoch 2',colors[3])
    box(4.1,.8,3.35,1.9,'Exp 9 · expanded RLVR','681 prompts · 2,724 rollouts\n22 updates · starts from SFT',colors[2])
    box(8.3,3.8,3.35,1.9,'RLHF · 300-prompt branch','300 prompts · 1,200 rollouts\n10 updates · fresh RLHF optimizer',colors[3])
    box(8.3,.8,3.35,1.9,'Exp 10 · final RLHF','1,000 prompts · 4,000 rollouts\n32 updates · recovered after 27',colors[-1])
    for xy,xytext in [((4,4.7),(3.3,4.3)),((4,1.7),(3.3,3.5)),((8.15,4.5),(7.6,2.5)),((8.15,1.7),(7.6,1.7))]:
        ax.annotate('',xy=xy,xytext=xytext,arrowprops={'arrowstyle':'->','color':'#526678','lw':1.6})
    finish(fig, '08-development-map', 'Arrows show checkpoint initialization only. The 300- and 1,000-prompt RLHF runs independently start from Experiment 9 RLVR.\nEarlier experiment boxes summarize research decisions; no checkpoint-inheritance arrows are asserted for them.')
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true')
    args = parser.parse_args()
    data = snapshot() if args.refresh_data else json.loads((OUT / 'data.json').read_text())
    for name in render(data):
        print(name)
