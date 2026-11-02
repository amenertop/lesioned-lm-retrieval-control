"""Frozen fine-battery summaries, matched-group ranks and response-review provenance.

The development statistics remain in title_gap_analysis. Random states need their point estimates for
the empirical comparison, not thousands of separate confidence intervals. All states keep raw answers.
"""
import numpy as np
import pandas as pd

from . import runinfo, semantic_diversity as sd, title_gap_analysis as ta

RANDOM_KINDS = {'matched_random', 'matched_gate_shuffle'}


def review_queue(tables, items):
    _, rows = ta.fine_responses(tables, items, n_boot=0)
    columns = ['response_id', 'target', 'synset', 'definition', 'condition', 'answer']
    queue = rows.loc[rows.category == 'unclassified', columns]
    return queue.sort_values(['response_id', 'answer']).drop_duplicates('response_id')


def random_ranks(profiles, states, directions, expected_groups):
    """Inclusive ties, fixed directions and all groups. Undefined statistics remain unavailable."""
    controls = [s for s in states if s.get('kind') in RANDOM_KINDS]
    if {s['target'] for s in controls} != set(expected_groups):
        raise ValueError('random targets differ from the frozen comparison')
    result = {}
    for target, expected in expected_groups.items():
        selected = [s for s in controls if s['target'] == target]
        names = [s['name'] for s in selected]
        if (len(names) != expected or len(set(names)) != expected
                or {s['random_index'] for s in selected} != set(range(expected))):
            raise ValueError('incomplete or duplicated matched groups')
        result[target] = {}
        for metric, sign in directions.items():
            if sign not in (-1, 1):
                raise ValueError('rank directions must be fixed as +1 or -1')
            value = profiles[target][metric]['estimate']
            reference = np.array([profiles[name][metric]['estimate'] for name in names], dtype=float)
            valid = value is not None and np.isfinite(value) and np.isfinite(reference).all()
            count = int(np.sum(sign * reference >= sign * value)) if valid else None
            result[target][metric] = dict(target=value, direction=sign, n_random=len(names),
                exceedances=count, empirical_rank=(1 + count) / (1 + len(names)) if valid else None,
                status='estimated' if valid else 'undefined_statistic',
                controls=[dict(state=name, estimate=float(v)) for name, v in zip(names, reference)])
    return result


def linear_contrast(frame, weights, tasks, n_boot):
    """A fixed paired factorial or intervention contrast on the original damage scale."""
    if not weights or not np.isclose(sum(weights.values()), 0):
        raise ValueError('paired contrast weights must sum to zero')
    rows = frame[frame.state.isin(weights)]
    results = {}
    for task in tasks:
        wide = rows.pivot(index=['item_id', 'family'], columns='state', values=task)
        if set(wide) != set(weights):
            raise ValueError('missing factorial state')
        wide = wide.dropna()
        if wide.empty:
            raise ValueError('empty paired contrast')
        paired = sum(wide[name] * value for name, value in weights.items()).rename('contrast').reset_index()
        results[task] = ta.bootstrap_family_mean(paired, 'contrast', n_boot)
    return results


def paired_contrast(frame, first, second, n_boot):
    """Add direct frequency-gradient differences using the same covariates as each profile."""
    result = ta.fine_contrast(frame, first, second, n_boot)
    left = frame[frame.state == first].set_index('item_id')
    right = frame[frame.state == second].set_index('item_id').loc[left.index]
    for task in ('naming', 'definition', 'synonym'):
        paired = left.copy()
        paired['difference'] = left[task] - right[task]
        paired = paired.dropna(subset=['difference', f'intact_{task}'])
        covariates = ['frequency', 'n_tokens', f'intact_{task}']
        if 'semd' in paired and paired.semd.notna().all():
            covariates.append('semd')

        def slope(rows, covariates=covariates):
            x = np.column_stack([np.ones(len(rows)), rows[covariates].to_numpy(float)])
            return np.linalg.lstsq(x, rows.difference, rcond=None)[0][1]

        result[task + '_frequency_slope_difference'] = ta.bootstrap(paired, slope, n_boot)
    return result


def semd_random_ranks(frame, states, pairs, observed):
    """Pair the pre-generated random indices once, then compare the signed SemD interaction."""
    groups = {}
    for state in states:
        if state.get('kind') in RANDOM_KINDS:
            groups.setdefault(state['target'], {})[state['random_index']] = state['name']
    result = {}
    for pair in pairs:
        wr, sc = pair.split(':')
        if wr not in groups or sc not in groups or set(groups[wr]) != set(groups[sc]):
            raise ValueError('SemD comparisons require paired frozen random indices on both sides')
        points = []
        for index in sorted(groups[wr]):
            frames = sd.fine_frames(frame, groups[wr][index], groups[sc][index])
            for task, data in frames.items():
                fit = sd.paired_regression(data, n_boot=0)
                coefficient = next((c['estimate'] for c in fit.get('coefficients', [])
                                    if c['state'] == 'SC_minus_WR' and c['predictor'] == 'semd'), None)
                points.append(dict(random_index=index, task=task, estimate=coefficient))
        summaries = {}
        for task, fit in observed[pair].items():
            target = next((c['estimate'] for c in fit.get('coefficients', [])
                           if c['state'] == 'SC_minus_WR' and c['predictor'] == 'semd'), None)
            values = np.array([p['estimate'] for p in points if p['task'] == task], dtype=float)
            valid = target is not None and np.isfinite(target) and np.isfinite(values).all()
            count = int((values >= target).sum()) if valid else None
            summaries[task] = dict(target=target, n_random=len(values), exceedances=count,
                empirical_rank=(1 + count) / (1 + len(values)) if valid else None,
                status='estimated' if valid else 'undefined_statistic', direction='SC_minus_WR larger')
        result[pair] = dict(tasks=summaries, rows=points)
    return result


def response_intervals(responses, comparisons, n_boot, seed=1910):
    """Resample the same concepts across states. Unknown answers and zero error denominators stay missing."""
    families = sorted(responses.family.unique())
    n = len(families)
    draws = np.random.default_rng(seed).integers(n, size=(n_boot, n))
    points, boots = {}, {}
    for state, rows in responses.groupby('state'):
        points[state], boots[state] = {}, {}
        for condition, group in rows.groupby('condition'):
            group = group.set_index('family').reindex(families)
            if len(group) != n or group.category.isna().any():
                raise ValueError('response intervals require every paired concept')
            key = condition + '_generated_accuracy'
            unknown = group.category.eq('unclassified').any()
            values = group.category.eq('correct').to_numpy(float)
            points[state][key] = None if unknown else float(values.mean())
            boots[state][key] = np.full(n_boot, np.nan) if unknown else values[draws].mean(1)
            if condition == 'naming':
                semantic = group.category.isin(['coordinate', 'superordinate', 'associate']).to_numpy()
                denominator = semantic[draws].sum(1)
                for category in ('coordinate', 'superordinate', 'associate'):
                    key = category + '_share_of_semantic_errors'
                    values = group.category.eq(category).to_numpy()
                    points[state][key] = float(values.sum() / semantic.sum()) if semantic.any() and not unknown else None
                    boots[state][key] = np.full(n_boot, np.nan)
                    if not unknown:
                        np.divide(values[draws].sum(1), denominator, out=boots[state][key], where=denominator > 0)

    def summary(point, boot):
        valid = np.isfinite(boot)
        low, high = np.quantile(boot[valid], [.025, .975]) if point is not None and valid.sum() >= .9 * n_boot and n_boot else (None, None)
        return dict(estimate=point, ci_low=low, ci_high=high, n=n, n_unit='概念', unit='proportion',
                    status='estimated' if low is not None else 'undefined_or_unstable', valid_boot=int(valid.sum()))

    states = {state: {key: summary(value, boots[state][key]) for key, value in metrics.items()} for state, metrics in points.items()}
    contrasts = {}
    for pair in comparisons:
        first, second = pair.split(':')
        contrasts[pair] = {}
        for key, value in points[first].items():
            other = points[second][key]
            point = value - other if value is not None and other is not None else None
            contrasts[pair][key] = summary(point, boots[first][key] - boots[second][key])
    return dict(states=states, contrasts=contrasts)


def general_impairment(answers, language):
    """Apply the fixed language, copying and format limits without selecting a new final state."""
    losses = language.set_index('state').loss
    format_rate = lambda answer: 1 - answer['categories'].get('format_invalid', {}).get('naming', 0) / sum(
        counts.get('naming', 0) for counts in answer['categories'].values())
    out = {}
    for state, answer in answers.items():
        lm = float(losses[state] / losses['intact'] - 1)
        copy = answer['generated_accuracy']['copy']
        copy_delta = copy - answers['intact']['generated_accuracy']['copy'] if copy is not None else None
        format_delta = format_rate(answer) - format_rate(answers['intact'])
        out[state] = dict(lm_change=lm, copy_change=copy_delta, format_change=format_delta,
                          language_pass=lm <= .1, copy_pass=copy_delta is not None and copy_delta >= -.05,
                          format_pass=format_delta >= -.05)
        out[state]['all_pass'] = all(out[state][key] for key in ('language_pass', 'copy_pass', 'format_pass'))
    return out


def label_intervals(labels, items, comparisons, n_boot):
    rows = ta.fine_label_accuracy(labels).merge(items[['item_id', 'family']], on='item_id', validate='many_to_one')
    states, contrasts = {}, {}

    def summarize(group, column):
        value = ta.bootstrap_family_mean(group, column, n_boot)
        value.update(n=value.pop('n_families'), n_unit='概念', unit='proportion')
        return value

    for (state, task), group in rows.groupby(['state', 'fmt']):
        states.setdefault(state, {})[task + '_accuracy'] = summarize(group, 'correct_response')
    for pair in comparisons:
        first, second = pair.split(':')
        contrasts[pair] = {}
        for task, group in rows.groupby('fmt'):
            wide = group.pivot(index=['item_id', 'family'], columns='state', values='correct_response')
            paired = (wide[first] - wide[second]).rename('difference').reset_index()
            contrasts[pair][task + '_accuracy'] = summarize(paired, 'difference')
    return states, contrasts


def relative_summary(tables, plan, items, protocol):
    """Apply the same continuous endpoints to candidate competition, preserving raw scoring and answers."""
    relative = ta.fine_readout_tables(tables, items, 'accepted_vs_coordinate')
    profiles, frame = ta.fine_profiles(relative, items, n_boot=0)
    core = {key: rows[rows.state.isin(protocol['core_states'])] for key, rows in relative.items()}
    core_profiles, core_frame = ta.fine_profiles(core, items, protocol['n_boot'])
    profiles.update(core_profiles)
    contrasts = {pair: paired_contrast(core_frame, *pair.split(':'), protocol['n_boot'])
                 for pair in protocol['comparisons']}
    ranks = random_ranks(profiles, plan['states'], protocol['random_metric_directions'], protocol['random_counts'])
    linear = {name: linear_contrast(core_frame, weights, ['naming', 'definition', 'synonym'], protocol['n_boot'])
              for name, weights in protocol['linear_contrasts'].items()}
    return dict(naming_readout='accepted_vs_coordinate', results=profiles,
                contrasts=contrasts, matched_random=ranks, linear_contrasts=linear), frame


def summarize(tables, plan, items, protocol, review):
    core_names = set(protocol['core_states'])
    core = {key: rows[rows.state.isin(core_names)] for key, rows in tables.items()}
    n_boot = protocol['n_boot']
    profiles, frame = ta.fine_profiles(tables, items, n_boot=0)
    core_profiles, core_frame = ta.fine_profiles(core, items, n_boot)
    profiles.update(core_profiles)
    answers, responses = ta.fine_responses(tables, items, n_boot=0, review=review)
    core_ids = set(responses.loc[responses.state.isin(core_names), 'response_id'])
    core_review = review[review.response_id.isin(core_ids)] if review is not None else None
    core_answers, _ = ta.fine_responses(core, items, n_boot, core_review)
    answers.update(core_answers)
    intervals = response_intervals(responses[responses.state.isin(core_names)], protocol['comparisons'], n_boot, protocol['seed'])
    label_states, label_contrasts = label_intervals(core['labels'], items, protocol['comparisons'], n_boot)
    for state, metrics in label_states.items():
        intervals['states'][state].update(metrics)
    for pair, metrics in label_contrasts.items():
        intervals['contrasts'][pair].update(metrics)
    comparisons = {pair: paired_contrast(core_frame, *pair.split(':'), n_boot)
                   for pair in protocol['comparisons']}
    ranks = random_ranks(profiles, plan['states'], protocol['random_metric_directions'], protocol['random_counts'])
    factorial = {name: linear_contrast(core_frame, weights, ['naming', 'definition', 'synonym'], n_boot)
                 for name, weights in protocol.get('linear_contrasts', {}).items()}
    specificity, specificity_rows = ta.fine_candidate_specificity(core['candidates'], items, n_boot)
    properties, property_rows = ta.property_response_diagnostics(core['labels'], items, n_boot)
    semd = {}
    for pair in protocol['semd_pairs']:
        wr, sc = pair.split(':')
        frames = sd.fine_frames(frame, wr, sc)
        semd[pair] = {task: sd.paired_regression(data, n_boot=n_boot) for task, data in frames.items()}
    payload = dict(status='internal_holdout_confirmation', provenance=plan['provenance'],
        source_fingerprint=plan['fingerprint'], source_runs=plan.get('source_runs'), n_boot=n_boot,
        results=profiles, contrasts=comparisons, actual_answers=answers, matched_random=ranks,
        linear_contrasts=factorial, candidate_specificity=specificity, properties=properties, semd=semd,
        response_intervals=intervals, general_impairment=general_impairment(core_answers, tables['language']),
        semd_matched_random=semd_random_ranks(frame, plan['states'], protocol['semd_random_pairs'], semd),
        language=tables['language'].to_dict('records'), analysis_commit=runinfo.commit())
    frames = dict(items=frame, responses=responses, candidate_specificity=specificity_rows,
                  property_pairs=property_rows, language=tables['language'])
    if protocol.get('scope') == 'relative_hub_mechanism':
        payload['candidate_competition'], frames['relative_items'] = relative_summary(tables, plan, items, protocol)
        payload['interventions'] = {state['name']: dict(n_gates=len(state['gates']),
            off_mass=sum(1 - value for value in state['gates'].values()),
            kind=state.get('kind', 'intact'), training=state.get('training'),
            seed=state.get('training', {}).get('seed'),
            naming_readout=state.get('training', {}).get('naming_readout', 'target') if 'training' in state else None,
            roles=([f'original_{role}' for role, name in protocol['original_selected'].items() if name == state['name']]
                   + ([f"relative_hub_{state['kind']}"] if state['name'] in protocol['random_counts'] else [])
                   + [f'equal_size_{seed}_{readout}' for seed, pair in protocol['equal_size_pairs'].items()
                      for readout, name in pair.items() if name == state['name']]))
            for state in plan['states'] if state['name'] in core_names}
    return payload, frames


def ledger_summary(result, frame, n_neutral_texts):
    """Expose every planned comparison to the writing ledger, keeping units and missing values."""
    report = ta.fine_ledger_summary(result, frame, n_neutral_texts)

    def statistic(value):
        saved = {k: v for k, v in value.items()
                 if k in ('estimate', 'ci_low', 'ci_high', 'n', 'n_unit', 'unit', 'status')}
        if 'valid_boot' in value:
            saved['meta'] = dict(valid_boot=value['valid_boot'])
        return saved

    for state, metrics in result['response_intervals']['states'].items():
        report['states'][state].update({name: statistic(value) for name, value in metrics.items()})
    report['response_contrasts'] = {pair: {name: statistic(value) for name, value in metrics.items()}
                                    for pair, metrics in result['response_intervals']['contrasts'].items()}
    report['candidate_specificity'] = result['candidate_specificity']
    report['properties'] = result['properties']
    report['meta']['general_impairment'] = result['general_impairment']
    report['meta']['review_metadata'] = result['review_metadata']
    def contrasts(source):
        sections = {}
        for section in ('contrasts', 'linear_contrasts'):
            sections[section] = {}
            for pair, metrics in source[section].items():
                sections[section][pair] = {}
                for name, value in metrics.items():
                    unit = 'correlation difference' if 'consistency' in name else 'nat' if name.startswith('naming') else 'logit'
                    if 'frequency_slope' in name:
                        unit += ' / log10 BNC frequency'
                    sections[section][pair][name] = {**statistic(value), 'unit': unit, 'n': value['n_families'], 'n_unit': '概念'}
        return sections

    report.update(contrasts(result))

    def ranks(metrics):
        return {name: dict(rank=dict(estimate=value['empirical_rank'], n=value['n_random'], n_unit='随机损伤组',
                                    unit='inclusive empirical rank', status=value['status']),
                          exceedances=dict(estimate=value['exceedances'], n=value['n_random'], n_unit='随机损伤组',
                                           unit='random groups', status=value['status']),
                          meta=dict(target=value['target'], direction=value['direction']))
                for name, value in metrics.items()}

    report['matched_random'] = {state: ranks(metrics) for state, metrics in result['matched_random'].items()}
    report['semd_matched_random'] = {pair: ranks(metrics['tasks']) for pair, metrics in result['semd_matched_random'].items()}
    if 'candidate_competition' in result:
        relative = result['candidate_competition']
        relative_states = ta.fine_ledger_summary({**result, 'results': relative['results']}, frame, n_neutral_texts)['states']
        report['candidate_competition'] = dict(meta=dict(naming_readout=relative['naming_readout']),
            states=relative_states, **contrasts(relative),
            matched_random={state: ranks(metrics) for state, metrics in relative['matched_random'].items()})
        report['interventions'] = {name: dict(
            n_gates=dict(estimate=row['n_gates'], unit='components'),
            off_mass=dict(estimate=row['off_mass'], unit='equivalent fully suppressed components'),
            meta={key: row[key] for key in ('kind', 'training', 'seed', 'naming_readout', 'roles')})
            for name, row in result['interventions'].items()}
    report['semd'] = {}
    for pair, tasks in result['semd'].items():
        report['semd'][pair] = {}
        for task, fit in tasks.items():
            unit = 'nat / predictor SD' if task == 'naming' else 'logit / predictor SD'
            report['semd'][pair][task] = dict(n=fit['n'], n_unit='概念', unit=unit,
                coefficients={f"{row['state']}/{row['predictor']}": {k: row[k] for k in ('estimate', 'ci_low', 'ci_high')}
                              for row in fit['coefficients']},
                frequency_without_semd=fit['frequency_without_semd'], frequency_shift=fit['frequency_shift'])
    return report


def finite_json(value):
    """Undefined correlations are JSON null, never zero or a nonstandard NaN token."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value
