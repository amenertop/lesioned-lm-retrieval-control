"""Checks for the concrete fine-battery freeze, received outputs and blind judgements."""
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import fine_semantics as fs, runinfo, title_gap_scoring as scoring

RANDOM_KINDS = {'matched_random', 'matched_gate_shuffle'}
METRIC_DIRECTIONS = dict(naming=1, definition=1, synonym=1,
    naming_frequency_slope=-1, definition_frequency_slope=-1, synonym_frequency_slope=-1,
    naming_definition_partial_r=1, naming_synonym_partial_r=1)
SOURCE_FILES = (
    'requirements.txt',
    'docs/plans/title-gap-selection.md', 'docs/plans/title-gap-fine-final.md',
    'docs/analysis-plan/title-gap-fine.md', 'data/materials/title_gap/fine_final/manifest.json',
    'docs/plans/title-gap-throughput.md', 'experiments/title_gap/check_score_batches.py',
    'reports/explore/2026-10-19-title-gap-execution/score-batches.json',
    'data/neutral_text/wikitext103_val_seed0.jsonl', 'docs/paper-prep/component-members.csv',
    'experiments/e1_screen/common.py', 'experiments/title_gap/run_battery.py',
    'experiments/title_gap/analyze_fine_final.py', 'experiments/title_gap/prepare_fine_final.py',
    'src/lesion/fine_protocol.py',
    'src/lesion/fine_confirmation.py', 'src/lesion/fine_semantics.py',
    'src/lesion/title_gap_analysis.py', 'src/lesion/title_gap_scoring.py',
    'src/lesion/title_gap_states.py', 'src/lesion/semantic_diversity.py',
    'src/lesion/m3_battery.py', 'src/lesion/m3_scoring.py', 'src/lesion/regional_generation.py',
    'src/lesion/scoring.py', 'src/lesion/hooks.py', 'src/lesion/gates.py',
    'src/lesion/gate_objective.py', 'src/lesion/weight_masks.py', 'src/lesion/fast_moe.py',
    'src/lesion/models.py', 'src/lesion/means.py', 'src/lesion/neutral_text.py', 'src/lesion/runinfo.py',
)


def check_design(protocol, states, selection):
    """The fixed scientific comparisons cannot be removed by editing an optional config list."""
    if (protocol['n_families'] != 120 or protocol['n_boot'] != 10000 or protocol['seed'] != 1910
            or protocol['random_metric_directions'] != METRIC_DIRECTIONS):
        raise ValueError('fine counts, bootstrap seed or main readouts differ from the plan')
    chosen = selection['selected']
    roles = {f'{mode}_{kind}' for mode in ('hub', 'comprehension') for kind in ('learned', 'learned_hard')}
    if set(chosen) != roles or selection['provenance']['split'] != 'dev':
        raise ValueError('all four candidate roles require a development selection')
    selected = {v['state'] for v in chosen.values() if v['state'] is not None}
    if any(v['state'] is None and v['n_eligible'] != 0 for v in chosen.values()):
        raise ValueError('an eligible development candidate cannot be omitted')
    required = {'intact', 'WR8_a0', 'WR30_a0', 'SC4_a0', 'PC8_all_a0'} | selected
    if not required <= set(states):
        raise ValueError('required reference or selected fine states are missing')
    if protocol['random_counts'] != dict.fromkeys(selected | {'WR8_a0', 'SC4_a0'}, 200):
        raise ValueError('every selected target and the WR8/SC4 pair require 200 random states')
    random = [state for state in states.values() if state.get('kind') in RANDOM_KINDS]
    if {state['target'] for state in random} != set(protocol['random_counts']):
        raise ValueError('random states do not cover the fixed targets')
    for target in protocol['random_counts']:
        group = [state for state in random if state['target'] == target]
        if len(group) != 200 or {state['random_index'] for state in group} != set(range(200)):
            raise ValueError('every target requires all 200 fixed random indices')
    required_pairs = {f'{name}:SC4_a0' for name in selected | {'WR8_a0', 'WR30_a0'}}
    hard = {v['state'] for key, v in chosen.items() if key.endswith('_hard') and v['state'] is not None}
    if set(protocol['minimality']) != hard:
        raise ValueError('each selected hard group needs its minus-one and plus-one comparisons')
    for target, variants in protocol['minimality'].items():
        base = states[target]['gates']
        minus, plus = variants['minus_one'], variants['plus_one']
        if (len(minus) != len(base) or len(set(minus)) != len(base) or len(plus) != 2 or len(set(plus)) != 2
                or len({states[name].get('added') for name in plus}) != 2
                or {states[name].get('removed') for name in minus} != set(base)):
            raise ValueError('incomplete minus-one or plus-one states')
        for name in minus + plus:
            state = states[name]
            expected = dict(base)
            if name in minus:
                expected.pop(state['removed'])
            else:
                if state['added'] in base:
                    raise ValueError('plus-one component already belongs to the target')
                expected[state['added']] = 0.0
            if state['gates'] != expected or state.get('region') != states[target].get('region'):
                raise ValueError('minimality state has changed the other interventions')
            required_pairs.add(f'{name}:{target}')
    if not required_pairs <= set(protocol['comparisons']):
        raise ValueError('required direct comparisons are missing')
    for pair in protocol['comparisons'] + protocol['semd_pairs']:
        if len(pair.split(':')) != 2 or not set(pair.split(':')) <= set(states):
            raise ValueError('comparison references an unknown state')
    if 'WR8_a0:SC4_a0' not in protocol['semd_pairs'] or protocol['semd_random_pairs'] != ['WR8_a0:SC4_a0']:
        raise ValueError('the original paired SemD comparison and its random reference are required')
    comprehension = chosen['comprehension_learned_hard']['state']
    expected_contrasts = {}
    if comprehension is not None:
        joint = protocol['comprehension_joint']
        wr, comp = states['WR8_a0'], states[comprehension]
        if (set(wr['gates']) & set(comp['gates']) or states[joint]['gates'] != {**wr['gates'], **comp['gates']}
                or any(states[name].get('region') not in (None, 'all') or states[name].get('weight_mask')
                       for name in ('intact', 'WR8_a0', comprehension, joint)) or states['intact']['gates']):
            raise ValueError('factorial states must contain the actual disjoint intervention union')
        expected_contrasts['comprehension_by_retrieval'] = dict.fromkeys([comprehension, 'WR8_a0'], -1)
        expected_contrasts['comprehension_by_retrieval'].update({joint: 1, 'intact': 1})
    if protocol['linear_contrasts'] != expected_contrasts:
        raise ValueError('the paired factorial contrast differs from the actual design')


def training_evidence(plans, meta_paths):
    """Check recorded training inputs and hard groups against the saved continuous gates."""
    metadata = {runinfo.sha256(path): (Path(path), json.loads(Path(path).read_text(encoding='utf-8')))
                for path in meta_paths}
    records, settings, pool, functions = {}, None, None, None
    keys = ('model', 'revision', 'tiny', 'mode', 'steps', 'sparsity', 'batch_size', 'checkpointing',
            'items_sha256', 'manifest_sha256', 'means_sha256', 'norms_sha256', 'pool_sha256', 'dtype', 'moe')
    for plan in plans:
        candidates = [s for s in plan['states'] if s.get('training', {}).get('mode') == 'hub']
        for soft in [s for s in candidates if s['kind'] == 'learned']:
            recorded = soft['training']
            path, meta = metadata[recorded['meta_sha256']]
            readout, seed = meta.get('naming_readout', 'target'), meta['seed']
            identity = f'{readout}:{seed}'
            if (identity in records or meta.get('tiny') is not False or not meta.get('complete')
                    or meta['selection_split'] != 'screening' or meta['mode'] != 'hub'
                    or meta['steps'] != 200 or meta['sparsity'] != 0.0003):
                raise ValueError('training identity or fixed search settings differ')
            current = {key: meta[key] for key in keys}
            if settings is not None and current != settings:
                raise ValueError('training inputs or hyperparameters differ between objectives')
            if any(current[key] != plan['provenance'][key]
                   for key in ('model', 'revision', 'tiny', 'items_sha256', 'means_sha256', 'dtype', 'moe')):
                raise ValueError('training and development measurements differ')
            settings = current
            gates_path = path.with_name('gates.csv')
            if runinfo.sha256(gates_path) != meta['gates_sha256']:
                raise ValueError('training gates differ from their metadata')
            table = pd.read_csv(gates_path)
            gates = dict(zip(table.component, table.retained))
            if table.component.duplicated().any() or gates != soft['gates'] or (pool is not None and set(gates) != pool):
                raise ValueError('continuous gates or candidate pools differ from training')
            pool = set(gates)
            order = table.sort_values(['retained', 'component']).component.tolist()
            group = [s for s in candidates if s['training']['seed'] == seed]
            if {s['kind'] for s in group} != {'learned', 'learned_hard'} or len(group) != 6:
                raise ValueError('each search start needs its complete soft and hard grid')
            if sorted(len(s['gates']) for s in group if s['kind'] == 'learned_hard') != [4, 8, 16, 32, 64]:
                raise ValueError('hard group sizes differ from the fixed grid')
            for state in group:
                for key in ('seed', 'mode', 'steps', 'sparsity', 'scales', 'gates_sha256'):
                    if state['training'][key] != meta[key]:
                        raise ValueError(f'training metadata mismatch: {key}')
                if state['training']['meta_sha256'] != recorded['meta_sha256']:
                    raise ValueError('hard and soft states have different training sources')
                if state['kind'] == 'learned_hard' and state['gates'] != dict.fromkeys(order[:len(state['gates'])], 0.):
                    raise ValueError('hard group is not the most suppressed continuous members')
            source = runinfo.git_bytes('show', f"{meta['commit']}:src/lesion/fine_objective.py")
            tree = ast.parse(source)
            definitions = {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest()
                           for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('train', 'hub_terms')}
            if set(definitions) != {'train', 'hub_terms'} or (functions is not None and functions != definitions):
                raise ValueError('training algorithm, default weights or learning rate changed')
            functions = definitions
            entry = runinfo.git_bytes('show', f"{meta['commit']}:experiments/title_gap/train_hub.py")
            calls = [node for node in ast.walk(ast.parse(entry)) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == 'fine_objective' and node.func.attr == 'train']
            if len(calls) != 1 or len(calls[0].args) != 10 or {k.arg for k in calls[0].keywords} != {'checkpointing'}:
                raise ValueError('training entry overrides the checked objective defaults')
            records[identity] = dict(seed=seed, naming_readout=readout, meta_sha256=recorded['meta_sha256'],
                gates_sha256=meta['gates_sha256'], soft_state=soft['name'],
                soft_gates_fingerprint=scoring._fingerprint(gates), top_members=order[:64],
                execution_commit=meta['commit'], resident_means=meta.get('resident_means', False),
                objective_sha256=hashlib.sha256(source).hexdigest(), entry_sha256=hashlib.sha256(entry).hexdigest())
    if set(records) != {f'{readout}:{seed}' for readout in ('target', 'accepted_vs_coordinate') for seed in (0, 1, 2)}:
        raise ValueError('both objectives need all three complete training sources')
    if set(metadata) != {r['meta_sha256'] for r in records.values()}:
        raise ValueError('unexpected training metadata source')
    return dict(settings=settings, unchanged_functions=functions, records=records,
                resource_note='Earlier seed zero used host means. The recorded paired score/gradient check is reused.')


def check_development_binding(root, selection, binding):
    """Bind the ranking and raw plan through the committed receipt, without copying raw runs into Git."""
    root = Path(root)
    receipt = json.loads((root / binding['receipt_path']).read_text(encoding='utf-8'))
    if receipt['runs'][selection['source_fingerprint']]['plan.json'] != binding['plan_sha256']:
        raise ValueError('development plan differs from its received bytes')
    ranking_path = root / binding['ranking_path']
    if selection.get('naming_readout') == 'accepted_vs_coordinate':
        for name, digest in [('ranking_sha256', runinfo.sha256(ranking_path)),
                             ('development_plan_sha256', binding['plan_sha256']),
                             ('receipt_sha256', runinfo.sha256(root / binding['receipt_path']))]:
            if selection.get(name) != digest:
                raise ValueError(f'development selection has a different {name}')
    ranking = pd.read_csv(ranking_path)
    if set(ranking['mode'] + '_' + ranking['kind']) != set(selection['selected']):
        raise ValueError('development ranking has missing or unexpected candidate roles')
    for (mode, kind), rows in ranking.groupby(['mode', 'kind'], sort=False):
        eligible = rows[rows.eligible]
        expected = dict(state=eligible.iloc[0].state if len(eligible) else None,
                        complete_profile=bool(eligible.iloc[0].complete_profile) if len(eligible) else False,
                        n_eligible=len(eligible), n_candidates=len(rows))
        if selection['selected'][mode + '_' + kind] != expected:
            raise ValueError('selected candidate is not the first eligible ranked candidate')


def check_relative_design(protocol, states, selection, original_selection):
    """The bounded objective comparison has its own roles, random groups and equal-size pairs."""
    if (protocol['scope'] != 'relative_hub_mechanism' or protocol['n_families'] != 120
            or protocol['n_boot'] != 10000 or protocol['seed'] != 1910
            or protocol['random_metric_directions'] != METRIC_DIRECTIONS
            or protocol['naming_readouts'] != ['target', 'accepted_vs_coordinate']):
        raise ValueError('relative hub counts or readouts differ from the plan')
    chosen = selection['selected']
    if (set(chosen) != {'hub_learned', 'hub_learned_hard'}
            or selection['provenance']['split'] != 'dev'
            or selection.get('naming_readout') != 'accepted_vs_coordinate'
            or selection['provenance'].get('tiny') is not False
            or original_selection['provenance'].get('tiny') is not False):
        raise ValueError('relative hub needs its own two-role development selection')
    if any(v['state'] is None and v['n_eligible'] != 0 for v in chosen.values()):
        raise ValueError('an eligible relative candidate cannot be omitted')
    selected = {v['state'] for v in chosen.values() if v['state'] is not None}
    original = {role: original_selection['selected'][role]['state'] for role in chosen}
    if protocol['original_selected'] != original:
        raise ValueError('original candidates differ from their completed development selection')
    for role, value in chosen.items():
        if value['state'] is None:
            continue
        state = states[value['state']]
        kind = 'learned_hard' if role.endswith('_hard') else 'learned'
        if (state['kind'] != kind or state['training']['mode'] != 'hub'
                or state['training'].get('naming_readout') != 'accepted_vs_coordinate'):
            raise ValueError('selected relative state has a different training objective or kind')
    if protocol['random_counts'] != dict.fromkeys(selected, 200):
        raise ValueError('each relative candidate requires 200 matched groups')
    core = {'intact'} | selected
    pairs = set()
    for role, value in chosen.items():
        original = protocol['original_selected'][role]
        if original is not None:
            core.add(original)
        if value['state'] is not None and original is not None:
            pairs.add(f"{value['state']}:{original}")
    weights = {}
    if set(protocol['equal_size_pairs']) != {'0', '1', '2'}:
        raise ValueError('all three paired search starts are required')
    for seed, pair in protocol['equal_size_pairs'].items():
        if set(pair) != {'target', 'accepted_vs_coordinate'}:
            raise ValueError('an equal-size pair needs both objectives')
        for readout, name in pair.items():
            state = states[name]
            training = state['training']
            if (state['kind'] != 'learned_hard' or len(state['gates']) != 64
                    or set(state['gates'].values()) != {0.0} or training['mode'] != 'hub'
                    or training['seed'] != int(seed) or training.get('naming_readout', 'target') != readout):
                raise ValueError('equal-size comparison differs from the fixed search start or objective')
            core.add(name)
            weights[name] = 1 / 3 if readout == 'accepted_vs_coordinate' else -1 / 3
        pairs.add(f"{pair['accepted_vs_coordinate']}:{pair['target']}")
    if (set(protocol['core_states']) != core or set(protocol['comparisons']) != pairs
            or protocol['linear_contrasts'] != {'equal_size_objective': weights}
            or protocol['semd_pairs'] or protocol['semd_random_pairs']):
        raise ValueError('relative core comparisons differ from the fixed mechanism plan')
    random = [s for s in states.values() if s.get('kind') in RANDOM_KINDS]
    if {s['target'] for s in random} != selected:
        raise ValueError('relative matched groups have different targets')
    for target in selected:
        group = [s for s in random if s['target'] == target]
        if len(group) != 200 or {s['random_index'] for s in group} != set(range(200)):
            raise ValueError('a relative candidate is missing matched groups')
    if set(states) != core | {s['name'] for s in random}:
        raise ValueError('relative final states contain an unplanned comparison')
    records = protocol['training_evidence']['records']
    for name in core - {'intact'}:
        state, gates = states[name], states[name]['gates']
        training = state['training']
        record = records[f"{training.get('naming_readout', 'target')}:{training['seed']}"]
        if training['meta_sha256'] != record['meta_sha256']:
            raise ValueError('final state has a different training source')
        expected = (dict.fromkeys(record['top_members'][:len(gates)], 0.) if state['kind'] == 'learned_hard' else None)
        if (expected is not None and gates != expected) or (expected is None and scoring._fingerprint(gates) != record['soft_gates_fingerprint']):
            raise ValueError('final state differs from the verified continuous or top-ranked gates')


def check_protocol(plan, root, items_path, norms_path):
    root = Path(root)
    provenance = plan['provenance']
    tag = provenance.get('freeze_tag')
    filename = provenance.get('protocol_path', 'docs/analysis-plan/title-gap-fine.json')
    if filename not in ('docs/analysis-plan/title-gap-fine.json', 'docs/analysis-plan/title-gap-relative-hub.json'):
        raise ValueError('unknown final fine protocol')
    path = root / filename
    if (provenance.get('split') != 'test' or provenance.get('battery') != 'fine'
            or provenance.get('tiny') or not tag or not tag.startswith('analysis-frozen')):
        raise ValueError('final fine analysis requires a real frozen test run')
    digest = runinfo.sha256(path)
    if digest != provenance.get('protocol_sha256') or digest != runinfo.sha256_at(tag, path.relative_to(root)):
        raise ValueError('fine protocol differs from the scored and tagged version')
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if (protocol['status'] != 'frozen' or runinfo.sha256(items_path) != protocol['items_sha256']
            or runinfo.sha256(norms_path) != protocol['norms_sha256']
            or provenance['items_sha256'] != protocol['items_sha256']):
        raise ValueError('fine materials or norms differ from the freeze')
    if any(provenance[key] != protocol[key] for key in ('model', 'revision', 'dtype', 'moe', 'means_sha256')):
        raise ValueError('fine model, precision or means differ from the freeze')
    required = set(SOURCE_FILES) | {protocol['selection_path']}
    if filename.endswith('relative-hub.json'):
        required |= {'docs/plans/title-gap-relative-hub.md', 'docs/analysis-plan/title-gap-relative-hub.md',
                     'experiments/title_gap/prepare_relative_final.py', 'experiments/title_gap/select_hub.py',
                     'experiments/title_gap/train_hub.py', 'experiments/title_gap/make_states.py',
                     'src/lesion/fine_objective.py',
                     'reports/explore/2026-10-19-title-gap-execution/search-execution-versions.json',
                     protocol['original_selection_path']}
        required |= {binding[key] for binding in protocol['development_bindings'].values()
                     for key in ('receipt_path', 'ranking_path')}
    if not required <= set(protocol['sources_sha256']):
        raise ValueError('the fine freeze is missing required code or material sources')
    for name, digest in {**protocol['sources_sha256'], **protocol['states_sha256']}.items():
        if runinfo.sha256(root / name) != digest or runinfo.sha256_at(tag, name) != digest:
            raise ValueError(f'fine source changed: {name}')
    sources = plan.get('source_runs', [dict(provenance=provenance)])
    if sorted(r['provenance']['states_sha256'] for r in sources) != sorted(protocol['states_sha256'].values()):
        raise ValueError('final fine analysis requires every frozen run exactly once')
    if (list(plan['batches']) != [16, 11, 7] or protocol['batches'] != [16, 11, 7]
            or plan.get('naming_batches', plan['batches']) != protocol.get('naming_batches', protocol['batches'])
            or protocol.get('naming_batches', protocol['batches']) not in ([16, 11, 7], [64, 47, 31])
            or plan['chat'] is not True or plan['max_new_tokens'] != 8
            or scoring._fingerprint(plan['neutral_sequences']) != protocol['neutral_sequences_sha256']
            or len(plan['neutral_sequences']) != 40 or any(len(seq) != 128 for seq in plan['neutral_sequences'])):
        raise ValueError('fine scoring settings or neutral tokens differ from the freeze')
    items = pd.read_csv(items_path, keep_default_na=False).query("split == 'test' and review == 'approved'")
    fs.validate(items)
    if (len(items) != 120 or items.family.nunique() != 120
            or items.frequency_bin.value_counts().to_dict() != {'low': 40, 'middle': 40, 'high': 40}):
        raise ValueError('fine test concepts must retain the three fixed frequency quotas')
    prompts = fs.naming_prompts(items) + fs.naming_prompts(items, copy=True) + fs.word_answer_prompts(items)
    for key, expected in [('naming', prompts), ('labels', fs.label_prompts(items))]:
        if scoring._fingerprint(plan[key]) != scoring._fingerprint([vars(p) for p in expected]):
            raise ValueError('scored prompts differ from the frozen fine materials')
    states = {}
    for name in protocol['states_sha256']:
        for state in json.loads((root / name).read_text(encoding='utf-8'))['states']:
            if state['name'] in states and state != states[state['name']]:
                raise ValueError('frozen files contain conflicting interventions')
            states[state['name']] = state
    if len(plan['states']) != len(states) or {s['name']: s for s in plan['states']} != states:
        raise ValueError('scored interventions differ from the frozen state definitions')
    if {name for name, state in states.items() if state.get('kind') not in RANDOM_KINDS} != set(protocol['core_states']):
        raise ValueError('fine core states differ from the freeze')
    selection = json.loads((root / protocol['selection_path']).read_text(encoding='utf-8'))
    if filename.endswith('relative-hub.json'):
        original = json.loads((root / protocol['original_selection_path']).read_text(encoding='utf-8'))
        check_relative_design(protocol, states, selection, original)
        for key, chosen in [('relative', selection), ('original', original)]:
            check_development_binding(root, chosen, protocol['development_bindings'][key])
    else:
        check_design(protocol, states, selection)
    return protocol


def output_receipt(directories):
    """Hash completed runs at receipt. Persist this record before blind review and analysis."""
    runs = {}
    for directory in map(Path, directories):
        plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
        paths = ['plan.json', 'complete.json', 'language.csv']
        for state in plan['states']:
            prefix = state['name']
            marker = json.loads((directory / prefix / 'complete.json').read_text(encoding='utf-8'))
            if set(marker['files']) != {'candidates.parquet', 'labels.parquet', 'generated.parquet'}:
                raise ValueError('a state receipt needs all three scored tables')
            for name, digest in marker['files'].items():
                if runinfo.sha256(directory / prefix / name) != digest:
                    raise ValueError('scored table differs from its completion marker')
            paths += [f'{prefix}/complete.json', *(f'{prefix}/{name}' for name in marker['files'])]
        if plan['fingerprint'] in runs:
            raise ValueError('duplicate source run')
        runs[plan['fingerprint']] = {name: runinfo.sha256(directory / name) for name in paths}
    return dict(status='received_outputs', boundary='Bytes received from completed runs, before response review and analysis.', runs=runs)


def checked_review(queue, queue_path, review_path=None, meta_path=None):
    if queue.empty and review_path is None and meta_path is None:
        return None, None
    if review_path is None or meta_path is None:
        raise ValueError('every unknown final response needs a blind judgement or explicit abstention')
    review = pd.read_csv(review_path, keep_default_na=False)
    allowed = {'correct', 'coordinate', 'superordinate', 'associate', 'phonological', 'omission',
               'unrelated', 'format_invalid', 'incorrect', 'unclassified'}
    if (not {'response_id', 'category'} <= set(review) or review.response_id.duplicated().any()
            or set(review.category) - allowed or set(review.response_id) != set(queue.response_id)):
        raise ValueError('review must cover exactly the initially unknown responses')
    meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
    if (meta.get('queue_sha256') != runinfo.sha256(queue_path) or meta.get('review_sha256') != runinfo.sha256(review_path)
            or not meta.get('reviewer_model') or not meta.get('trace_path')
            or meta.get('review_independence') not in ('same-family', 'cross-family', 'human')
            or meta.get('acceptance_status') not in ('provisional', 'accepted')
            or (meta['review_independence'] == 'same-family' and meta['acceptance_status'] != 'provisional')):
        raise ValueError('blind review identity, acceptance status or hashes do not match')
    trace = Path(meta['trace_path'])
    if not trace.is_file() or meta.get('trace_sha256') != runinfo.sha256(trace):
        raise ValueError('blind review trace is missing or changed')
    joined = queue.merge(review[['response_id', 'category']], on='response_id', validate='one_to_one')
    joined['normalized_answer'] = joined.answer.astype(str).str.lower().str.strip().str.strip('\"\'.,! ')
    if (joined.groupby(['target', 'synset', 'condition', 'normalized_answer']).category.nunique() > 1).any():
        raise ValueError('the same normalized response has conflicting blind judgements')
    return review[review.category != 'unclassified'], meta
