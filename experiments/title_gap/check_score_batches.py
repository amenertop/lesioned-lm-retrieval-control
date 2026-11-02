"""One development-only comparison of larger continuous-score batches. Generation is unchanged."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/e1_screen'))
import common
from lesion import fine_semantics as fs, gate_objective, m3_battery, m3_scoring, neutral_text, runinfo, title_gap_states as ts


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def vectors(candidates, labels):
    result = {f'candidate/{name}': rows.set_index(['item_id', 'role']).logprob.sort_index()
              for name, rows in candidates.groupby('condition')}
    result.update({f'label/{name}': rows.set_index(['item_id', 'variant']).margin.sort_index()
                   for name, rows in labels.groupby('fmt')})
    labels = labels.copy()
    labels['claim'] = np.where(labels.fmt == 'property', labels.variant // 2, 0)
    decisions = {name: rows.groupby(['item_id', 'claim']).margin.mean().sort_index()
                 for name, rows in labels.groupby('fmt')}
    return result, decisions


def compare(old, new):
    if any(not old[0].index.equals(series.index) for series in old + new):
        raise ValueError('batch variants contain different observations')
    before, after = np.stack(old), np.stack(new)
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('nonfinite scores')
    difference = np.abs(after.mean(0) - before.mean(0))
    noise = [np.abs(before[a] - before[b]) for a, b in combinations(range(3), 2)]
    result = {}
    for name, function in [('mean', np.mean), ('p99', lambda x: np.quantile(x, .99))]:
        observed, reference = float(function(difference)), max(float(function(x)) for x in noise)
        result[name] = dict(difference=observed, original_batch_noise=reference,
                            tolerance=1.5 * reference + 1e-5, passed=observed <= 1.5 * reference + 1e-5)
    result['passed'] = all(value['passed'] for value in result.values())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    common.add_args(ap)
    ap.add_argument('--items', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('preserve earlier checks, choose a new output directory')
    if args.tiny and not args.out.resolve().is_relative_to((ROOT / 'outputs/title_gap_tiny').resolve()):
        ap.error('tiny checks require outputs/title_gap_tiny')
    if not args.tiny and (args.dtype != 'float16' or args.moe != 'sorted' or args.model != 'qwen3-30b-a3b-instruct'):
        ap.error('the single real comparison uses the pinned main model, float16 and sorted')
    commit = common.guard(args)
    items = fs.load_items(args.items, 'dev', rehearsal=args.tiny)
    batches = ((4, 3, 2), (8, 7, 5)) if args.tiny else ((16, 11, 7), (64, 47, 31))
    if args.tiny:
        items = items.head(6)
    naming = fs.naming_prompts(items) + fs.naming_prompts(items, copy=True) + fs.word_answer_prompts(items)
    labels = fs.label_prompts(items)
    model, tok, chat, means, _ = common.load(args)
    if args.tiny:
        states = [dict(name=name, gates=gates) for name, gates in [('intact', {}),
            ('expert', {'expert:0:0': 0.}), ('head', {'head:0:0': 0.}),
            ('broad', {f'expert:0:{i}': .1 for i in range(model.config.num_experts)})]]
    else:
        source = json.loads((ROOT / 'experiments/title_gap/configs/cue-content-dev.json').read_text(encoding='utf-8'))
        states = source['states']
    ts.validate_states(states, model.config)
    args.out.mkdir(parents=True)
    payload = dict(meta=dict(commit=commit, tiny=args.tiny, model=args.model, dtype=args.dtype, moe=args.moe,
        items_sha256=runinfo.sha256(args.items), means_sha256=runinfo.sha256(args.means),
        code_sha256=runinfo.sha256(Path(__file__)), batches=batches, states=states,
        scope='development continuous scores only, generation and language-loss batches unchanged'), timings={}, checks={})
    path = args.out / 'result.json'
    raw, decisions = {}, {}
    with torch.no_grad():
        gate_objective.lm_loss(model, [tok.encode(neutral_text.load()[0], add_special_tokens=False)[:128]], tok.eos_token_id or 0)
    try:
        for state in states:
            name = state['name']
            raw[name], decisions[name], payload['timings'][name] = {}, {}, {}
            with ts.apply_state(model, means, state):
                for batch in (*batches[0], *batches[1]):
                    sync()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    start = time.monotonic()
                    candidates = m3_scoring.score_naming_prompts(model, tok, naming, chat, batch)
                    choices = m3_battery.label_scores(model, tok, labels, chat, batch)
                    sync()
                    payload['timings'][name][str(batch)] = dict(seconds=time.monotonic()-start,
                        peak_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None)
                    directory = args.out / name / str(batch)
                    directory.mkdir(parents=True)
                    candidates.to_parquet(directory / 'candidates.parquet', index=False)
                    choices.to_parquet(directory / 'labels.parquet', index=False)
                    raw[name][batch], decisions[name][batch] = vectors(candidates, choices)
                    print(name, batch, round(payload['timings'][name][str(batch)]['seconds'], 2), flush=True)
            payload['checks'][name] = {key: compare([raw[name][b][key] for b in batches[0]],
                                                   [raw[name][b][key] for b in batches[1]]) for key in raw[name][batches[0][0]]}
            for key in decisions[name][batches[0][0]]:
                old = np.stack([decisions[name][b][key] for b in batches[0]])
                new = np.stack([decisions[name][b][key] for b in batches[1]])
                noise = max(int(((old[a] > 0) != (old[b] > 0)).sum()) for a, b in combinations(range(3), 2))
                changed = int(((old.mean(0) > 0) != (new.mean(0) > 0)).sum())
                payload['checks'][name]['decision/' + key] = dict(changed=changed, original_batch_noise=noise, passed=changed <= noise)
        for name in raw:
            if name == 'intact':
                continue
            payload['checks'][name + ':intact'] = {key: compare(
                [raw[name][b][key] - raw['intact'][b][key] for b in batches[0]],
                [raw[name][b][key] - raw['intact'][b][key] for b in batches[1]]) for key in raw[name][batches[0][0]]}
        old_time, new_time = [sum(value[str(b)]['seconds'] for value in payload['timings'].values() for b in group) for group in batches]
        peaks = [row['peak_gib'] for state in payload['timings'].values() for row in state.values() if row['peak_gib'] is not None]
        payload.update(status='complete', speedup=old_time/new_time, peak_gib=max(peaks) if peaks else None,
            numerical_pass=all(row['passed'] for state in payload['checks'].values() for row in state.values()))
        payload['adopt'] = bool(not args.tiny and payload['numerical_pass'] and payload['speedup'] >= 1.25 and payload['peak_gib'] <= 90)
    except torch.OutOfMemoryError as error:
        payload.update(status='out_of_memory', adopt=False, error=str(error))
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
