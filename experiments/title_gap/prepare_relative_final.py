"""Assemble the bounded relative-objective comparison after development, using existing state generators."""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import fine_protocol as fp, models, runinfo, title_gap_scoring as scoring, title_gap_states as ts
from lesion.means import Means


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--learned-plan', type=Path, required=True)
    ap.add_argument('--original-plan', type=Path, required=True)
    ap.add_argument('--ranking', type=Path, required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--training-meta', type=Path, action='append', required=True)
    ap.add_argument('--means', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('preserve existing preparation and choose a new output directory')
    original_path = ROOT / 'docs/analysis-plan/selections/title-gap-fine.json'
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    original = json.loads(original_path.read_text(encoding='utf-8'))
    plans = [json.loads(p.read_text(encoding='utf-8')) for p in (args.learned_plan, args.original_plan)]
    for chosen, plan in zip((selection, original), plans):
        require(scoring._fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'}) == plan['fingerprint'],
                'development plan fingerprint differs')
        require(chosen['source_fingerprint'] == plan['fingerprint'] and chosen['provenance'] == plan['provenance'],
                'selection and development provenance differ')
        require(plan['provenance']['split'] == 'dev' and plan['provenance']['battery'] == 'fine'
                and plan['provenance'].get('tiny') is False, 'selection requires real fine development runs')
    provenance = plans[0]['provenance']
    for key in ('model', 'revision', 'tiny', 'items_sha256', 'means_sha256', 'dtype', 'moe'):
        require(provenance[key] == plans[1]['provenance'][key], f'development setting differs: {key}')
    require(provenance['means_sha256'] == runinfo.sha256(args.means), 'means differ from development')
    model, revision = provenance['model'], provenance['revision']
    require((model, revision) == ('qwen3-30b-a3b-instruct', models.MODELS['qwen3-30b-a3b-instruct'][1]),
            'comparison requires the pinned main model')
    config = AutoConfig.from_pretrained(models.local_dir(model), local_files_only=True)
    means = Means.load(args.means)
    require(means.meta['model_name'] == model and means.meta['revision'] == revision, 'means model differs')
    training = fp.training_evidence(plans, args.training_meta)
    bindings = {}
    for key, chosen, plan_path, ranking, receipt in [
            ('relative', selection, args.learned_plan, args.ranking, args.receipt),
            ('original', original, args.original_plan, ROOT / 'docs/analysis-plan/selections/title-gap-fine-ranking.csv',
             ROOT / 'reports/explore/2026-10-19-title-gap-execution/learned-dev-receipt.json')]:
        bindings[key] = dict(plan_sha256=runinfo.sha256(plan_path),
            ranking_path=ranking.resolve().relative_to(ROOT).as_posix(),
            receipt_path=receipt.resolve().relative_to(ROOT).as_posix())
        fp.check_development_binding(ROOT, chosen, bindings[key])
    by_readout = {}
    for readout, plan in zip(('accepted_vs_coordinate', 'target'), plans):
        candidates = [s for s in plan['states'] if s.get('training', {}).get('mode') == 'hub']
        require({(s['training']['seed'], s['kind'], len(s['gates']) if s['kind'] == 'learned_hard' else None)
                for s in candidates} == {(seed, kind, size) for seed in (0, 1, 2)
                for kind, size in [('learned', None), *[('learned_hard', n) for n in (4, 8, 16, 32, 64)]]},
                'development candidates differ from the fixed grid')
        require(all(s['training'].get('naming_readout', 'target') == readout for s in candidates),
                'development candidates have a different objective')
        by_readout[readout] = {s['name']: s for s in candidates}
    core = {'intact': plans[0]['states'][0]}
    selected = {role: value['state'] for role, value in selection['selected'].items()}
    original_selected = {role: original['selected'][role]['state'] for role in selected}
    comparisons = []
    for role, name in selected.items():
        old = original_selected[role]
        for readout, state in [('accepted_vs_coordinate', name), ('target', old)]:
            if state is not None:
                core[state] = by_readout[readout][state]
        if name is not None and old is not None:
            comparisons.append(f'{name}:{old}')
    pairs, weights = {}, {}
    for seed in (0, 1, 2):
        pair = {}
        for readout, candidates in by_readout.items():
            match = [s for s in candidates.values() if s['training']['seed'] == seed
                     and s['kind'] == 'learned_hard' and len(s['gates']) == 64]
            require(len(match) == 1, 'equal-size comparison is missing or duplicated')
            state = match[0]
            core[state['name']] = state
            pair[readout] = state['name']
            weights[state['name']] = 1 / 3 if readout == 'accepted_vs_coordinate' else -1 / 3
        pairs[str(seed)] = pair
        comparisons.append(f"{pair['accepted_vs_coordinate']}:{pair['target']}")
    targets = sorted(name for name in selected.values() if name is not None)
    blocks = {'relative-hub-final-core.json': list(core.values())}
    states = dict(core)
    for name in targets:
        random = ts.matched_states(core[name], config, means, n=200, seed=1910)
        blocks[f'relative-hub-final-random-{name}.json'] = [core['intact'], core[name], *random]
        for state in random:
            require(state['name'] not in states, 'matched state name is duplicated')
            states[state['name']] = state
    design = dict(scope='relative_hub_mechanism', n_families=120, n_boot=10000, seed=1910,
        naming_readouts=['target', 'accepted_vs_coordinate'], random_metric_directions=fp.METRIC_DIRECTIONS,
        core_states=list(core), random_counts=dict.fromkeys(targets, 200), original_selected=original_selected,
        original_selection_path=original_path.relative_to(ROOT).as_posix(), equal_size_pairs=pairs,
        comparisons=list(dict.fromkeys(comparisons)), linear_contrasts={'equal_size_objective': weights},
        semd_pairs=[], semd_random_pairs=[], training_evidence=training, development_bindings=bindings)
    fp.check_relative_design(design, states, selection, original)
    for block in blocks.values():
        ts.validate_states(block, config)
    args.out.mkdir(parents=True)
    for filename, block in blocks.items():
        payload = dict(model=model, revision=revision, tiny=False, states=block, seed=1910,
            means_sha256=provenance['means_sha256'], selection_sha256=runinfo.sha256(args.selection),
            role='relative_mechanism_final_pending_tag')
        (args.out / filename).write_text(json.dumps(payload, indent=1)+'\n', encoding='utf-8', newline='\n')
    design['preparation'] = dict(code_commit=runinfo.commit(),
        sources_sha256={p.resolve().relative_to(ROOT).as_posix(): runinfo.sha256(p) for p in
            (Path(__file__), args.selection, args.learned_plan, args.original_plan, original_path,
             args.ranking, args.receipt, *args.training_meta)})
    (args.out / 'design.json').write_text(json.dumps(design, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(f'{len(core)} core states and {len(targets)*200} matched random states prepared')


if __name__ == '__main__':
    main()
