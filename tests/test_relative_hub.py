"""The objective comparison preserves raw outcomes and rejects missing or mismatched final controls."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from lesion import fine_confirmation as fc, fine_protocol as fp, title_gap_analysis as ta


def test_competition_readout_can_worsen_while_both_correct_words_improve():
    items = pd.DataFrame(dict(item_id=[f'i{i}' for i in range(10)], family=[f'f{i}' for i in range(10)],
                              frequency=np.arange(10), semd=np.sin(np.arange(10))))
    candidates, labels = [], []
    for state in ('intact', 'lesion'):
        for i, item in enumerate(items.item_id):
            for role, base, increase in [('target', -4., 1.), ('synonym', -5., 1.), ('coordinate', -6., 3.)]:
                candidates.append(dict(state=state, item_id=item, condition='naming', role=role, word=role,
                    n_tokens=1 + i % 3, logprob=base - .1 * i + (increase if state == 'lesion' else 0)))
            for task in ('definition', 'synonym'):
                labels.append(dict(state=state, item_id=item, fmt=task,
                                    margin=4 + .1 * i - ((2 + .05 * i) if state == 'lesion' else 0)))
    tables = dict(candidates=pd.DataFrame(candidates), labels=pd.DataFrame(labels))
    original = {name: frame.copy() for name, frame in tables.items()}
    protocol = dict(core_states=['intact', 'lesion'], n_boot=0, comparisons=[],
        random_metric_directions=fp.METRIC_DIRECTIONS, random_counts={},
        linear_contrasts={'difference': dict(intact=-1, lesion=1)})
    plan = dict(states=[dict(name='intact'), dict(name='lesion')])
    result, _ = fc.relative_summary(tables, plan, items, protocol)
    raw, _ = ta.fine_profiles(tables, items, n_boot=0)
    assert raw['lesion']['naming']['estimate'] == pytest.approx(-1)
    assert result['results']['lesion']['naming']['estimate'] == pytest.approx(2)
    assert result['linear_contrasts']['difference']['naming']['estimate'] == pytest.approx(2)
    assert result['results']['lesion']['definition'] == raw['lesion']['definition']
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], original[name])


@pytest.fixture
def design():
    states = {'intact': dict(name='intact', kind='intact', gates={})}
    pairs, weights = {}, {}
    for seed in (0, 1, 2):
        pair = {}
        for prefix, readout, sign in [('old', 'target', -1), ('new', 'accepted_vs_coordinate', 1)]:
            name = f'{prefix}{seed}'
            states[name] = dict(name=name, kind='learned_hard', gates={f'expert:0:{i}': 0. for i in range(64)},
                training=dict(seed=seed, mode='hub', naming_readout=readout))
            pair[readout] = name
            weights[name] = sign / 3
        pairs[str(seed)] = pair
    for name, readout in [('old_soft', 'target'), ('new_soft', 'accepted_vs_coordinate')]:
        states[name] = dict(name=name, kind='learned', gates={'expert:0:0': .5},
                            training=dict(seed=0, mode='hub', naming_readout=readout))
    selected = dict(hub_learned=dict(state='new_soft', n_eligible=3), hub_learned_hard=dict(state='new0', n_eligible=15))
    original = dict(selected=dict(hub_learned=dict(state='old_soft'), hub_learned_hard=dict(state='old0')),
                    provenance=dict(split='dev', tiny=False))
    selection = dict(selected=selected, provenance=dict(split='dev', tiny=False), naming_readout='accepted_vs_coordinate')
    protocol = dict(scope='relative_hub_mechanism', n_families=120, n_boot=10000, seed=1910,
        random_metric_directions=fp.METRIC_DIRECTIONS, naming_readouts=['target', 'accepted_vs_coordinate'],
        random_counts=dict(new_soft=200, new0=200), original_selected=dict(hub_learned='old_soft', hub_learned_hard='old0'),
        core_states=list(states), comparisons=['new_soft:old_soft', 'new0:old0', 'new1:old1', 'new2:old2'],
        equal_size_pairs=pairs, linear_contrasts={'equal_size_objective': weights}, semd_pairs=[], semd_random_pairs=[])
    records = {}
    for state in states.values():
        if 'training' not in state:
            continue
        training = state['training']
        identity = f"{training['naming_readout']}:{training['seed']}"
        training['meta_sha256'] = identity
        records[identity] = dict(meta_sha256=identity, top_members=[f'expert:0:{i}' for i in range(64)],
                                soft_gates_fingerprint=fp.scoring._fingerprint({'expert:0:0': .5}))
    protocol['training_evidence'] = dict(records=records)
    for target in protocol['random_counts']:
        for i in range(200):
            name = f'{target}_random_{i}'
            states[name] = dict(name=name, kind='matched_random', target=target, random_index=i)
    return protocol, states, selection, original


def test_relative_design_requires_all_random_controls_and_the_original_selected_reference(design):
    protocol, states, selection, original = design
    fp.check_relative_design(protocol, states, selection, original)
    missing = deepcopy(states)
    missing.pop('new0_random_199')
    with pytest.raises(ValueError, match='missing matched groups'):
        fp.check_relative_design(protocol, missing, selection, original)
    changed = deepcopy(protocol)
    changed['original_selected']['hub_learned'] = 'old1'
    with pytest.raises(ValueError, match='original candidates differ'):
        fp.check_relative_design(changed, states, selection, original)


def test_equal_size_comparison_rejects_a_changed_objective_or_member_count(design):
    protocol, states, selection, original = design
    changed = deepcopy(states)
    changed['old1']['training']['naming_readout'] = 'accepted_vs_coordinate'
    with pytest.raises(ValueError, match='equal-size comparison'):
        fp.check_relative_design(protocol, changed, selection, original)
    changed = deepcopy(states)
    changed['new2']['gates'].pop('expert:0:63')
    with pytest.raises(ValueError, match='equal-size comparison'):
        fp.check_relative_design(protocol, changed, selection, original)


def test_relative_comparison_rejects_unrelated_same_size_groups_and_tiny_sources(design):
    protocol, states, selection, original = design
    changed = deepcopy(states)
    changed['new2']['gates']['expert:0:100'] = changed['new2']['gates'].pop('expert:0:63')
    with pytest.raises(ValueError, match='verified continuous or top-ranked gates'):
        fp.check_relative_design(protocol, changed, selection, original)
    selection['provenance']['tiny'] = True
    with pytest.raises(ValueError, match='development selection'):
        fp.check_relative_design(protocol, states, selection, original)
