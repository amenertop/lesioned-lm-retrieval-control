"""Describe gate and hard-group agreement across the three completed screening seeds. No selection."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import models, runinfo

SETTINGS = ('model', 'revision', 'mode', 'steps', 'sparsity', 'batch_size', 'dtype', 'moe',
            'checkpointing', 'items_sha256', 'norms_sha256', 'manifest_sha256',
            'means_sha256', 'pool_sha256', 'scales')


def summarize(directories):
    values, orders, sources = {}, {}, {}
    settings = None
    for directory in directories:
        meta = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
        seed = meta['seed']
        current = {key: meta[key] for key in SETTINGS}
        if (not meta['complete'] or meta.get('tiny') or meta['selection_split'] != 'screening'
                or meta['revision'] != models.MODELS[meta['model']][1] or seed in values
                or meta['steps'] != 200 or (settings is not None and current != settings)):
            raise ValueError('stability requires complete distinct seeds with the same screening settings')
        gates = pd.read_csv(directory / 'gates.csv')
        log = pd.read_csv(directory / 'log.csv')
        if (runinfo.sha256(directory / 'gates.csv') != meta['gates_sha256']
                or gates.component.duplicated().any() or gates.empty
                or not np.isfinite(gates.retained).all() or not gates.retained.between(0, 1).all()
                or len(log) != meta['steps'] or list(log.step) != list(range(meta['steps']))):
            raise ValueError('invalid gates or incomplete optimizer trajectory')
        settings = current
        values[seed] = gates.set_index('component').retained.sort_index()
        orders[seed] = gates.sort_values(['retained', 'component']).component.tolist()
        sources[str(seed)] = dict(directory=directory.as_posix(), source=meta,
            files={name: runinfo.sha256(directory / name) for name in ('meta.json', 'gates.csv', 'log.csv')})
    if set(values) != {0, 1, 2} or any(not values[0].index.equals(value.index) for value in values.values()):
        raise ValueError('exactly seeds 0, 1, 2 with the same component pool are required')
    frame = pd.DataFrame(values)

    def metric(value, unit, n, n_unit):
        return dict(estimate=float(value), unit=unit, n=int(n), n_unit=n_unit,
                    note='同一模型和找组材料的优化起点描述，不将组件当作独立实验重复。')

    pairs = {}
    for a, b in combinations((0, 1, 2), 2):
        result = {method: metric(frame[a].corr(frame[b], method=method), 'correlation', len(frame), '组件')
                  for method in ('pearson', 'spearman')}
        result['hard_groups'] = {}
        for size in (4, 8, 16, 32, 64):
            first, second = set(orders[a][:size]), set(orders[b][:size])
            overlap, union = len(first & second), len(first | second)
            result['hard_groups'][str(size)] = dict(
                overlap=metric(overlap, 'components', size, '每个起点选中组件'),
                jaccard=metric(overlap / union, 'intersection / union', union, '并集组件'))
        pairs[f'{a}:{b}'] = result
    all_three = {}
    for size in (4, 8, 16, 32, 64):
        sets = [set(orders[seed][:size]) for seed in (0, 1, 2)]
        overlap, union = len(set.intersection(*sets)), len(set.union(*sets))
        all_three[str(size)] = dict(overlap=metric(overlap, 'components', size, '每个起点选中组件'),
            jaccard=metric(overlap / union, 'intersection / union', union, '并集组件'))
    result = dict(meta=dict(status='screening_seed_description_not_functional_replication', settings=settings,
        sources=sources, analysis_commit=runinfo.commit(), analysis_sha256=runinfo.sha256(Path(__file__)),
        note='Jaccard is shared members divided by the union. All planned hard-group sizes and seed pairs retained. No bootstrap over components.'),
        pairs=pairs, all_three=all_three)
    return result, frame.rename_axis('component').reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='append', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('preserve existing analysis, use a new output directory')
    result, frame = summarize(args.run)
    args.out.mkdir(parents=True)
    (args.out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                                         encoding='utf-8', newline='\n')
    frame.to_csv(args.out / 'aligned-gates.csv', index=False, lineterminator='\n')
    print(result['meta']['settings']['mode'], 'three-seed component agreement saved')


if __name__ == '__main__':
    main()
