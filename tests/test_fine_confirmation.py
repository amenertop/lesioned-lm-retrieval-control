"""Final-only provenance, paired contrasts and conservative handling of undefined readouts."""
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lesion import fine_confirmation as fc, fine_protocol as fp, fine_semantics as fs, runinfo, semantic_diversity as sd, title_gap_scoring


def test_random_rank_counts_ties_and_retains_undefined_groups():
    states = [dict(name=f'r{i}', kind='matched_gate_shuffle', target='chosen', random_index=i) for i in range(3)]
    profiles = {name: {'loss': dict(estimate=value), 'slope': dict(estimate=-value)}
                for name, value in [('chosen', 2.), ('r0', 1.), ('r1', 2.), ('r2', 3.)]}
    result = fc.random_ranks(profiles, states, {'loss': 1, 'slope': -1}, {'chosen': 3})
    assert result['chosen']['loss']['empirical_rank'] == .75
    assert result['chosen']['slope']['exceedances'] == 2
    profiles['r1']['loss']['estimate'] = np.nan
    assert fc.random_ranks(profiles, states, {'loss': 1}, {'chosen': 3})['chosen']['loss']['empirical_rank'] is None
    with pytest.raises(ValueError, match='incomplete'):
        fc.random_ranks(profiles, states[:-1], {'loss': 1}, {'chosen': 3})


def test_factorial_contrast_pairs_concepts_and_has_zero_under_additivity():
    frame = pd.DataFrame([dict(item_id=f'i{i}', family=f'f{i}', state=state, naming=scale * i)
                          for i in range(1, 9) for state, scale in
                          [('intact', 0), ('retrieval', 1), ('comprehension', 2), ('joint', 3)]])
    weights = dict(intact=1, retrieval=-1, comprehension=-1, joint=1)
    assert fc.linear_contrast(frame, weights, ['naming'], 50)['naming']['ci_high'] == 0
    frame.loc[frame.state == 'joint', 'naming'] += 2
    result = fc.linear_contrast(frame.sample(frac=1, random_state=5), weights, ['naming'], 50)
    assert result['naming']['estimate'] == result['naming']['ci_low'] == 2
    with pytest.raises(ValueError, match='missing factorial'):
        fc.linear_contrast(frame[frame.state != 'joint'], weights, ['naming'], 0)


def test_frequency_contrasts_use_paired_outcomes_and_profile_covariates():
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(rng.normal(size=(30, 6)),
        columns=['frequency', 'n_tokens', 'semd', 'intact_naming', 'intact_definition', 'intact_synonym'])
    frame['item_id'] = frame['family'] = [f'i{i}' for i in range(len(frame))]
    parts = []
    for state, coefficient in [('a', .3), ('b', 1.)]:
        part = frame.assign(state=state)
        for task in ('naming', 'definition', 'synonym'):
            part[task] = coefficient * part.frequency + .5 * part.n_tokens + .2 * part.semd + .4 * part['intact_' + task]
        parts.append(part)
    result = fc.paired_contrast(pd.concat(parts).sample(frac=1, random_state=9), 'a', 'b', 30)
    for task in ('naming', 'definition', 'synonym'):
        slope = result[task + '_frequency_slope_difference']
        assert slope['estimate'] == pytest.approx(-.7)
        assert slope['ci_low'] == pytest.approx(-.7)
        assert slope['ci_high'] == pytest.approx(-.7)


def test_clinical_transfer_rejects_implicit_or_old_final_materials():
    root = Path(__file__).resolve().parents[1]
    digest = runpy.run_path(root / 'experiments/title_gap/compare_clinical.py')['final_material_digest']
    for path in (None, root / 'data/materials/title_gap/fine/items.csv'):
        with pytest.raises(ValueError, match='explicit approved'):
            digest(path)
    approved = root / 'data/materials/title_gap/fine_final/items.csv'
    assert digest(approved) == runinfo.sha256(approved)


def test_final_design_requires_actual_joint_intervention():
    states = {name: dict(name=name, gates={f'expert:0:{i}':0})
              for i, name in enumerate(['WR8_a0', 'WR30_a0', 'SC4_a0', 'PC8_all_a0', 'comp'])}
    states.update(intact=dict(name='intact', gates={}), minus=dict(name='minus', removed='expert:0:4', gates={}))
    for i in (5, 6):
        states[f'plus{i}'] = dict(name=f'plus{i}', added=f'expert:0:{i}', gates={**states['comp']['gates'], f'expert:0:{i}':0})
    states['joint'] = dict(name='joint', gates={**states['WR8_a0']['gates'], **states['comp']['gates']})
    for target in ('WR8_a0', 'SC4_a0', 'comp'):
        for i in range(200):
            name = f'{target}_random_{i}'
            states[name] = dict(name=name, target=target, kind='matched_random', random_index=i, gates={})
    selected = {f'{mode}_{kind}':dict(state=None, n_eligible=0)
                for mode in ('hub', 'comprehension') for kind in ('learned', 'learned_hard')}
    selected['comprehension_learned_hard'] = dict(state='comp', n_eligible=1)
    selection = dict(provenance=dict(split='dev'), selected=selected)
    protocol = dict(n_families=120, n_boot=10000, seed=1910, random_metric_directions=fp.METRIC_DIRECTIONS,
        random_counts=dict.fromkeys(['WR8_a0', 'SC4_a0', 'comp'], 200),
        minimality={'comp':dict(minus_one=['minus'], plus_one=['plus5','plus6'])},
        comparisons=['WR8_a0:SC4_a0','WR30_a0:SC4_a0','comp:SC4_a0','minus:comp','plus5:comp','plus6:comp'],
        semd_pairs=['WR8_a0:SC4_a0'], semd_random_pairs=['WR8_a0:SC4_a0'], comprehension_joint='joint',
        linear_contrasts={'comprehension_by_retrieval':dict(comp=-1, WR8_a0=-1, joint=1, intact=1)})
    fp.check_design(protocol, states, selection)
    states['joint']['gates']['expert:0:7'] = 0
    with pytest.raises(ValueError, match='actual disjoint'):
        fp.check_design(protocol, states, selection)


def test_review_requires_unknown_coverage_and_consistent_response_identity(tmp_path):
    queue = pd.DataFrame([dict(response_id='a', target='tree', synset='tree.n.01', condition='naming', answer='forest'),
                          dict(response_id='b', target='tree', synset='tree.n.01', condition='naming', answer='FOREST.')])
    queue_path, review_path, meta_path = [tmp_path / name for name in ('queue.csv', 'review.csv', 'meta.json')]
    queue.to_csv(queue_path, index=False)
    trace = tmp_path / 'review-trace.md'
    trace.write_text('Unit fixture only, not a real judgement.')

    def save(labels):
        pd.DataFrame(labels).to_csv(review_path, index=False)
        meta_path.write_text(json.dumps(dict(queue_sha256=runinfo.sha256(queue_path),
            review_sha256=runinfo.sha256(review_path), reviewer_model='test',
            review_independence='same-family', acceptance_status='provisional', trace_path=str(trace),
            trace_sha256=runinfo.sha256(trace))), encoding='utf-8')

    save([dict(response_id='a', category='coordinate'), dict(response_id='b', category='correct')])
    with pytest.raises(ValueError, match='conflicting'):
        fp.checked_review(queue, queue_path, review_path, meta_path)
    save([dict(response_id='a', category='unclassified'), dict(response_id='b', category='unclassified')])
    review, _ = fp.checked_review(queue, queue_path, review_path, meta_path)
    assert review.empty
    meta = json.loads(meta_path.read_text())
    meta['acceptance_status'] = 'rejected'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='acceptance status'):
        fp.checked_review(queue, queue_path, review_path, meta_path)
    save([dict(response_id='a', category='correct')])
    with pytest.raises(ValueError, match='exactly'):
        fp.checked_review(queue, queue_path, review_path, meta_path)


def test_final_protocol_checks_union_and_actual_interventions(tmp_path, monkeypatch):
    directory = tmp_path / 'docs/analysis-plan'
    directory.mkdir(parents=True)
    items, norms, state_file = [tmp_path / name for name in ('items.csv', 'norms.xlsx', 'states.json')]
    from pathlib import Path
    items.write_bytes(Path('data/materials/title_gap/fine_final/items.csv').read_bytes())
    norms.write_text('fixture', encoding='utf-8')
    states = [dict(name='intact', gates={})] + [dict(name=name, gates={f'expert:0:{i}': 0})
        for i, name in enumerate(('WR8_a0', 'WR30_a0', 'SC4_a0', 'PC8_all_a0'))]
    states += [dict(name=f'{target}_random_{i}', kind='matched_random', target=target, random_index=i, gates={})
               for target in ('WR8_a0', 'SC4_a0') for i in range(200)]
    state_file.write_text(json.dumps(dict(states=states)), encoding='utf-8')
    identity = dict(model='test', revision='fixed', dtype='float16', moe='sorted', means_sha256='fixed-means')
    for name in fp.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    selection_path = tmp_path / 'selection.json'
    selection_path.write_text(json.dumps(dict(provenance=dict(split='dev'), selected={
        f'{mode}_{kind}': dict(state=None, n_eligible=0) for mode in ('hub', 'comprehension') for kind in ('learned', 'learned_hard')})))
    neutral = [[0] * 128 for _ in range(40)]
    protocol = dict(status='frozen', items_sha256=runinfo.sha256(items), norms_sha256=runinfo.sha256(norms),
        states_sha256={'states.json': runinfo.sha256(state_file)},
        sources_sha256={name: runinfo.sha256(tmp_path / name) for name in (*fp.SOURCE_FILES, 'selection.json')},
        selection_path='selection.json', n_families=120, n_boot=10000, seed=1910,
        batches=[16, 11, 7], neutral_sequences_sha256=title_gap_scoring._fingerprint(neutral),
        core_states=[s['name'] for s in states if s.get('kind') != 'matched_random'], random_metric_directions=fp.METRIC_DIRECTIONS,
        random_counts={'WR8_a0':200, 'SC4_a0':200}, minimality={}, linear_contrasts={},
        comparisons=['WR8_a0:SC4_a0', 'WR30_a0:SC4_a0'], semd_pairs=['WR8_a0:SC4_a0'],
        semd_random_pairs=['WR8_a0:SC4_a0'], **identity)
    protocol_path = directory / 'title-gap-fine.json'
    protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
    monkeypatch.setattr(runinfo, 'sha256_at', lambda tag, path: runinfo.sha256(tmp_path / path))
    provenance = dict(split='test', battery='fine', tiny=False, freeze_tag='analysis-frozen-test',
                      protocol_sha256=runinfo.sha256(protocol_path), items_sha256=runinfo.sha256(items),
                      states_sha256=runinfo.sha256(state_file), **identity)
    concepts = pd.read_csv(items, keep_default_na=False).query("split == 'test' and review == 'approved'")
    prompts = fs.naming_prompts(concepts) + fs.naming_prompts(concepts, copy=True) + fs.word_answer_prompts(concepts)
    plan = dict(provenance=provenance, states=states, batches=[16, 11, 7], chat=True, max_new_tokens=8,
                naming=[vars(p) for p in prompts], labels=[vars(p) for p in fs.label_prompts(concepts)], neutral_sequences=neutral)
    assert fp.check_protocol(plan, tmp_path, items, norms) == protocol
    protocol['naming_batches'] = [64, 47, 31]
    protocol_path.write_text(json.dumps(protocol), encoding='utf-8')
    provenance['protocol_sha256'] = runinfo.sha256(protocol_path)
    with pytest.raises(ValueError, match='scoring settings'):
        fp.check_protocol(plan, tmp_path, items, norms)
    plan['naming_batches'] = [64, 47, 31]
    assert fp.check_protocol(plan, tmp_path, items, norms) == protocol
    plan['source_runs'] = [dict(provenance=provenance), dict(provenance=provenance)]
    with pytest.raises(ValueError, match='exactly once'):
        fp.check_protocol(plan, tmp_path, items, norms)
    del plan['source_runs']
    plan['max_new_tokens'] = 9
    with pytest.raises(ValueError, match='scoring settings'):
        fp.check_protocol(plan, tmp_path, items, norms)
    plan['max_new_tokens'] = 8
    plan['states'] = [states[0], dict(name='lesion', gates={'expert:0:1': 1})]
    with pytest.raises(ValueError, match='interventions'):
        fp.check_protocol(plan, tmp_path, items, norms)
    plan['states'] = states
    selection = json.loads(selection_path.read_text())
    bad = dict(protocol, random_counts={'WR8_a0': 199, 'SC4_a0': 200})
    with pytest.raises(ValueError, match='200 random'):
        fp.check_design(bad, {s['name']:s for s in states}, selection)
    bad = dict(protocol, comparisons=[])
    with pytest.raises(ValueError, match='direct comparisons'):
        fp.check_design(bad, {s['name']:s for s in states}, selection)
    protocol['sources_sha256'] = {}
    protocol_path.write_text(json.dumps(protocol))
    provenance['protocol_sha256'] = runinfo.sha256(protocol_path)
    with pytest.raises(ValueError, match='required code'):
        fp.check_protocol(plan, tmp_path, items, norms)


def test_answer_intervals_keep_pairing_unknowns_and_error_denominators():
    rows = pd.DataFrame([dict(state=state, family=f'f{i}', condition='naming', category=category)
        for state in ('a', 'b') for i, category in enumerate(['correct', 'coordinate', 'superordinate', 'unclassified'])])
    missing = fc.response_intervals(rows, ['a:b'], 50)
    assert missing['states']['a']['naming_generated_accuracy']['estimate'] is None
    rows.loc[rows.category == 'unclassified', 'category'] = 'correct'
    result = fc.response_intervals(rows, ['a:b'], 1000)
    assert result['states']['a']['naming_generated_accuracy']['estimate'] == .5
    assert result['states']['a']['coordinate_share_of_semantic_errors']['estimate'] == .5
    assert result['contrasts']['a:b']['naming_generated_accuracy']['ci_low'] == 0
    assert result['contrasts']['a:b']['naming_generated_accuracy']['ci_high'] == 0
    rows['category'] = 'correct'
    assert fc.response_intervals(rows, [], 50)['states']['a']['coordinate_share_of_semantic_errors']['estimate'] is None


def test_received_output_hashes_bind_markers_and_all_tables(tmp_path):
    state = tmp_path / 'intact'
    state.mkdir()
    (tmp_path / 'plan.json').write_text(json.dumps(dict(fingerprint='fixture', states=[dict(name='intact')])))
    for name in ('complete.json', 'language.csv'):
        (tmp_path / name).write_text('fixture')
    names = ('candidates.parquet', 'labels.parquet', 'generated.parquet')
    for name in names:
        (state / name).write_text('fixture')
    marker = dict(files={name: runinfo.sha256(state / name) for name in names}, language=dict(loss=1))
    (state / 'complete.json').write_text(json.dumps(marker))
    receipt = fp.output_receipt([tmp_path])
    marker['language']['loss'] = 2
    (state / 'complete.json').write_text(json.dumps(marker))
    assert fp.output_receipt([tmp_path]) != receipt
    del marker['files']['labels.parquet']
    (state / 'complete.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='all three'):
        fp.output_receipt([tmp_path])


def test_ledger_exports_paired_contrasts_ranks_and_semd(monkeypatch):
    monkeypatch.setattr(fc.ta, 'fine_ledger_summary', lambda *args: dict(states={'a':{}}, meta={}))
    estimate = dict(estimate=.2, ci_low=.1, ci_high=.3, n_families=120)
    rank = dict(empirical_rank=1/201, n_random=200, exceedances=0, target=.2, direction=1, status='estimated')
    response = dict(estimate=.2, ci_low=.1, ci_high=.3, n=120, n_unit='概念', unit='proportion', valid_boot=10000)
    result = dict(response_intervals=dict(states={'a':{'naming_generated_accuracy':response}}, contrasts={'a:b':{}}),
        candidate_specificity={}, properties={},
        general_impairment={}, review_metadata={}, contrasts={'a:b':{'definition_damage_difference':estimate,
                                                                  'naming_frequency_slope_difference':estimate}},
        linear_contrasts={'joint':{'naming':estimate}}, matched_random={'a':{'naming':rank}},
        semd_matched_random={'a:b':{'tasks':{'naming':rank}}},
        semd={'a:b':{'naming':dict(n=120, coefficients=[dict(state='SC_minus_WR',predictor='semd',**estimate)],
                                  frequency_without_semd={}, frequency_shift={})}})
    report = fc.ledger_summary(result, None, 40)
    assert report['contrasts']['a:b']['definition_damage_difference']['unit'] == 'logit'
    assert report['linear_contrasts']['joint']['naming']['n'] == 120
    assert report['matched_random']['a']['naming']['rank']['n'] == 200
    assert report['semd']['a:b']['naming']['coefficients']['SC_minus_WR/semd']['estimate'] == .2
    assert report['states']['a']['naming_generated_accuracy']['ci_low'] == .1
    assert report['contrasts']['a:b']['naming_frequency_slope_difference']['unit'] == 'nat / log10 BNC frequency'
    ledger = runpy.run_path(Path(__file__).resolve().parents[1] / 'docs/paper-prep/build_ledger.py')
    ledger['SOURCES']['fixture'] = ('fixture.json', 'fixture', 'fixture')
    rows = ledger['new_rows']('fixture', report, 'fixture')
    assert len(rows) == 9
    assert not any(any(key in row['指标路径'] for key in ('valid_boot', 'n_families', 'direction', 'target')) for row in rows)
    counts = [row for row in rows if 'exceedances' in row['指标路径']]
    assert len(counts) == 2 and all(row['单位'] == 'random groups' for row in counts)


def test_random_semd_uses_original_units_without_synthetic_confidence_intervals():
    rng = np.random.default_rng(9)
    covariates = rng.normal(size=(60, 4))
    frame = pd.DataFrame(covariates, columns=['frequency', 'n_tokens', 'intact', 'semd'])
    frame['family'] = frame['item_id'] = [f'i{i}' for i in range(len(frame))]
    frame['h_wr'] = .2 * frame.frequency
    frame['h_sc'] = .2 * frame.frequency + .7 * frame.semd
    result = sd.paired_regression(frame, n_boot=0)
    interaction = next(c for c in result['coefficients'] if c['state'] == 'SC_minus_WR' and c['predictor'] == 'semd')
    assert interaction['estimate'] == pytest.approx(.7 * frame.semd.to_numpy().std())
    assert interaction['ci_low'] is None and result['status'] == 'point_only'
    assert result['valid_boot'] == 0
    boot = sd.paired_regression(frame, n_boot=30)
    assert result['coefficients'][0]['estimate'] == boot['coefficients'][0]['estimate']
