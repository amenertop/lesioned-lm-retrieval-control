"""Assemble approved fine-final comparisons after the complete development selection. No model loading."""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoConfig
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import fine_protocol as fp, models, runinfo, title_gap_states as ts
from lesion.means import Means


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--learned-plan', type=Path, required=True)
    ap.add_argument('--means', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('preserve existing preparation and choose a new output directory')
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    learned_plan = json.loads(args.learned_plan.read_text(encoding='utf-8'))
    assert selection['source_fingerprint'] == learned_plan['fingerprint']
    assert learned_plan['provenance']['split'] == 'dev'
    assert selection['provenance'] == learned_plan['provenance']
    assert learned_plan['provenance']['means_sha256'] == runinfo.sha256(args.means)
    model = learned_plan['provenance']['model']
    revision = learned_plan['provenance']['revision']
    assert (model, revision) == ('qwen3-30b-a3b-instruct', models.MODELS['qwen3-30b-a3b-instruct'][1])
    config = AutoConfig.from_pretrained(models.local_dir(model), local_files_only=True)
    means = Means.load(args.means)
    assert means.meta['model_name'] == model and means.meta['revision'] == revision
    configs = Path('experiments/title_gap/configs')
    core = {}
    inputs = {}


    def add(state):
        name = state['name']
        if name in core:
            assert core[name] == state, name
        core[name] = state


    def references(filename, names=None):
        path = configs / filename
        source = json.loads(path.read_text(encoding='utf-8'))
        ts.check_source(source, model, revision)
        inputs[path.as_posix()] = runinfo.sha256(path)
        for state in source['states']:
            if names is None or state['name'] in names:
                add(state)


    references('fine-dev.json', {'intact', 'WR8_a0', 'WR30_a0', 'SC4_a0',
        'broad_middle_f1_a0.9_s0', 'broad_middle_f1_a0.75_s0', 'broad_middle_f1_a0.5_s0',
        'broad_early_f1_a0.5_s0', 'broad_late_f1_a0.75_s0'})
    references('fine-extended-dev.json', {'broad_middle_f1_a0.25_s0', 'broad_middle_f1_a0.1_s0'})
    references('homonyms-final-core.json', {'PC8_all_a0'})
    references('coarse-fine-dev.json')
    references('clinical-final.json')
    learned = {state['name']: state for state in learned_plan['states']}
    origins = {(state['training']['mode'], state['training']['seed']) for state in learned.values()
               if 'training' in state}
    assert origins == {(mode, seed) for mode in ('hub', 'comprehension') for seed in (0, 1, 2)}
    selected = {value['state'] for value in selection['selected'].values() if value['state'] is not None}
    assert set(selection['selected']) == {f'{mode}_{kind}' for mode in ('hub', 'comprehension') for kind in ('learned', 'learned_hard')}
    for name in sorted(selected):
        add(learned[name])
    base_names = list(core)
    minimality, variant_comparisons = {}, []
    for role, chosen in selection['selected'].items():
        name = chosen['state']
        if not role.endswith('_hard') or name is None:
            continue
        state = core[name]
        soft = learned[name.rsplit('_hard_', 1)[0] + '_soft']
        order = sorted(soft['gates'], key=lambda member: (soft['gates'][member], member))
        size = len(state['gates'])
        assert set(order[:size]) == set(state['gates'])
        extra = order[size:size + 2]
        assert len(extra) == 2
        variants = ts.variants(state, extra)
        for variant in variants:
            add(variant)
            variant_comparisons.append(f"{variant['name']}:{name}")
        minimality[name] = dict(minus_one=[s['name'] for s in variants if s['kind'] == 'minus_one'],
                               plus_one=[s['name'] for s in variants if s['kind'] == 'plus_one'])
    comp = selection['selected']['comprehension_learned_hard']['state']
    joint = None
    linear = {}
    extra_pairs = []
    if comp is not None:
        assert not set(core['WR8_a0']['gates']) & set(core[comp]['gates'])
        joint = 'WR8_a0_' + comp
        add(dict(name=joint, kind='factorial', gates={**core['WR8_a0']['gates'], **core[comp]['gates']}))
        linear['comprehension_by_retrieval'] = {joint: 1, 'intact': 1, 'WR8_a0': -1, comp: -1}
        extra_pairs = [f'{joint}:WR8_a0', f'{joint}:{comp}']
    comparisons = [f'{name}:SC4_a0' for name in base_names if name not in ('intact', 'SC4_a0')]
    comparisons += variant_comparisons + extra_pairs
    semd_pairs = [f'{name}:SC4_a0' for name in ['WR8_a0', 'WR30_a0', *sorted(selected)]]
    random_targets = sorted(selected | {'WR8_a0', 'SC4_a0'})
    blocks = {'fine-final-core.json': list(core.values())}
    all_states = dict(core)
    for name in random_targets:
        random = ts.matched_states(core[name], config, means, n=200, seed=1910)
        blocks[f'fine-final-random-{name}.json'] = [core['intact'], core[name], *random]
        for state in random:
            assert state['name'] not in all_states
            all_states[state['name']] = state
    for states in blocks.values():
        ts.validate_states(states, config)
    design = dict(n_families=120, n_boot=10000, seed=1910,
        random_metric_directions=fp.METRIC_DIRECTIONS, random_counts=dict.fromkeys(random_targets, 200),
        minimality=minimality, comparisons=comparisons, semd_pairs=semd_pairs,
        semd_random_pairs=['WR8_a0:SC4_a0'], comprehension_joint=joint,
        linear_contrasts=linear, core_states=list(core))
    fp.check_design(design, all_states, selection)
    args.out.mkdir(parents=True)
    for filename, states in blocks.items():
        payload = dict(model=model, revision=revision, states=states, seed=1910,
                       means_sha256=runinfo.sha256(args.means), role='fixed_final_comparisons_pending_tag',
                       selection_sha256=runinfo.sha256(args.selection), source_sha256=runinfo.sha256(args.learned_plan))
        (args.out / filename).write_text(json.dumps(payload, indent=1)+'\n', encoding='utf-8', newline='\n')
    design['preparation'] = dict(script_sha256=runinfo.sha256(Path(__file__)), selection_sha256=runinfo.sha256(args.selection),
        learned_plan_sha256=runinfo.sha256(args.learned_plan), reference_files=inputs, code_commit=runinfo.commit())
    (args.out / 'design.json').write_text(json.dumps(design, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(f'{len(core)} core states, {200*len(random_targets)} random states, {len(blocks)} frozen file candidates')


if __name__ == "__main__":
    main()
