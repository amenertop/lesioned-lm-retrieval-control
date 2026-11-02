"""Five clinical marker families in their original units, with explicit missing coverage."""
import numpy as np
import pandas as pd

from . import damage, fine_confirmation as fc, title_gap_analysis as ta

MARKERS = {
    'frequency': ('naming_frequency', 'definition_frequency', 'synonym_frequency'),
    'consistency': ('naming_definition', 'naming_synonym'),
    'control': ('demand_difference',),
    'cueing': ('low_context_gain', 'high_context_gain'),
    'errors': ('associative_fraction',),
}


def direction_interval(estimate, expected, lower=None, upper=None):
    """Direction of an identified contrast or its tied-response classification bounds."""
    if expected not in (-1, 1):
        raise ValueError('expected direction must be -1 or 1')
    if estimate is not None and np.isfinite(estimate):
        lower = upper = float(estimate)
    if lower is None or upper is None or not np.isfinite([lower, upper]).all():
        return None, None
    if lower > upper:
        raise ValueError('reversed classification bounds')
    a, b = sorted((expected * lower, expected * upper))
    sign = lambda value: 0.5 if abs(value) <= 1e-12 else 1.0 if value > 0 else 0.0
    return sign(a), sign(b)


def scorecard(rows):
    """Equal family weights. Missing measurements never become zero scores."""
    by_marker = {row['marker']: row for row in rows}
    expected = {marker for markers in MARKERS.values() for marker in markers}
    if len(by_marker) != len(rows) or set(by_marker) != expected:
        raise ValueError('the scorecard requires every prespecified marker exactly once')
    families = {}
    for family, markers in MARKERS.items():
        bounds = []
        for name in markers:
            row = by_marker[name]
            if row['family'] != family:
                raise ValueError('marker assigned to a different clinical family')
            bound = row.get('classification_bounds', {})
            bounds.append(direction_interval(row.get('estimate'), row['expected_direction'],
                                             bound.get('lower'), bound.get('upper')))
        valid = [a is not None for a, _ in bounds]
        families[family] = dict(n_markers=len(markers), n_measured=sum(valid),
            lower=float(np.mean([a for a, _ in bounds])) if all(valid) else None,
            upper=float(np.mean([b for _, b in bounds])) if all(valid) else None)
    complete = [row for row in families.values() if row['lower'] is not None]
    return dict(families=families, n_families=5, n_covered_families=len(complete),
        lower=float(np.mean([row['lower'] for row in complete])) if len(complete) == 5 else None,
        upper=float(np.mean([row['upper'] for row in complete])) if len(complete) == 5 else None,
        interpretation='Equal-family direction coverage, not an effect size or sampling interval.')


def _row(family, marker, expected, unit, result=None):
    return dict(family=family, marker=marker, expected_direction=expected, unit=unit,
                **(result or dict(estimate=None, ci_low=None, ci_high=None, status='not_measured')))


def fine_markers(frame, sd, sa, n_boot):
    result = fc.paired_contrast(frame, sd, sa, n_boot) if frame is not None else {}
    rows = [_row('frequency', task + '_frequency', -1, unit + ' / log BNC frequency',
                 result.get(task + '_frequency_slope_difference'))
            for task, unit in (('naming', 'nat'), ('definition', 'logit'), ('synonym', 'logit'))]
    rows += [_row('consistency', 'naming_' + task, 1, 'correlation difference',
                  result.get(task + '_consistency_difference')) for task in ('definition', 'synonym')]
    return rows


def control_marker(frame, n_boot, seed):
    """Input is a validated paired M1 item frame with h_sd and h_sa columns."""
    result = None
    if frame is not None:
        paired = frame.copy()
        paired['h'] = paired.h_sa - paired.h_sd
        if not set(damage.MATCHING) <= set(paired.task):
            raise ValueError('both control tasks are required')
        result = damage.bootstrap(paired, lambda cols: dict(contrast=damage.effects(cols)['control_effect']),
                                  n=n_boot, seed=seed)['contrast']
        result.update(n_families=int(paired.concept_group.nunique()))
    return _row('control', 'demand_difference', 1, 'logit', result)


def cue_markers(rows, sd, sa, n_boot, seed):
    """Only subordinate meanings enter cue gains, with one label per unresolved answer ID."""
    results = []
    for context in ('low', 'high'):
        result = None
        if rows is not None:
            group = rows[(rows.context == context) & (rows.dominance == 'subordinate') &
                         rows.state.isin([sd, sa]) & rows.cue.isin(['none', 'correct'])].copy()
            wide = group.pivot(index='family', columns=['state', 'cue'], values='semantic_correct')
            expected = {(s, c) for s in (sd, sa) for c in ('none', 'correct')}
            if set(wide.columns) != expected or len(group) != len(wide) * 4 or wide.empty:
                raise ValueError('cue contrast requires paired states and both cue conditions')
            weights = (np.where(group.state == sa, 1., -1.) *
                       np.where(group.cue == 'correct', 1., -1.) / len(wide))
            bounds = ta.binary_response_bounds(group, weights, n_boot, seed)
            known = not group.semantic_correct.isna().any()
            envelope = bounds.get('sampling_envelope', {})
            result = dict(estimate=bounds['lower'] if known else None,
                ci_low=envelope.get('lower') if known else None,
                ci_high=envelope.get('upper') if known else None,
                n_families=len(wide), classification_bounds=bounds,
                status='fully_classified' if known else 'bounded_unresolved_responses',
                baseline={state: ta.binary_response_bounds(g, np.ones(len(g)) / len(g))
                          for state, g in group[group.cue == 'none'].groupby('state')},
                baseline_matching='not imposed by the direction table')
        results.append(_row('cueing', context + '_context_gain', 1, 'proportion difference', result))
    return results


def error_marker(rows, sd, sa, n_boot):
    """Associative errors as a share of actual single-word semantic naming errors."""
    result = None
    if rows is not None:
        group = rows[(rows.letters == 0) & rows.state.isin([sd, sa])]
        wide = group.pivot(index='item_id', columns='state', values='category')
        if set(wide.columns) != {sd, sa} or wide.empty or wide.isna().any().any():
            raise ValueError('error contrast requires the same naming concepts in both states')
        if not wide.isin(['unclassified']).any().any():
            paired = pd.DataFrame(dict(family=wide.index,
                sd_error=wide[sd].isin(['coordinate', 'superordinate', 'associate']).to_numpy(),
                sa_error=wide[sa].isin(['coordinate', 'superordinate', 'associate']).to_numpy(),
                sd_associate=wide[sd].eq('associate').to_numpy(),
                sa_associate=wide[sa].eq('associate').to_numpy()))

            def difference(sample):
                a, b = sample.sd_error.sum(), sample.sa_error.sum()
                return sample.sa_associate.sum() / b - sample.sd_associate.sum() / a if a and b else np.nan

            result = ta.bootstrap(paired, difference, n_boot)
            result['denominators'] = dict(sd=int(paired.sd_error.sum()), sa=int(paired.sa_error.sum()))
        else:
            result = dict(estimate=None, ci_low=None, ci_high=None, status='needs_blind_response_review')
    return _row('errors', 'associative_fraction', 1, 'proportion difference', result)


def compare(fine, control, homonyms, clinical, sd, sa, n_boot=1000, seed=1910):
    """Call existing paired statistics, then summarize direction without mixing metric units."""
    if sd == sa or n_boot < 0 or seed != 1910:
        raise ValueError('distinct states, a nonnegative count and the fixed seed 1910 are required')
    rows = fine_markers(fine, sd, sa, n_boot)
    rows.append(control_marker(control, n_boot, seed))
    rows.extend(cue_markers(homonyms, sd, sa, n_boot, seed))
    rows.append(error_marker(clinical, sd, sa, n_boot))
    return fc.finite_json(dict(sd_state=sd, sa_state=sa, n_boot=n_boot, seed=seed,
                               markers=rows, directions=scorecard(rows)))


def ledger_report(result):
    """Keep sampling intervals separate from response bounds and direction counts."""
    report = dict(meta=dict(analysis=result['meta'], sd_state=result['sd_state'], sa_state=result['sa_state'],
        n_boot=result['n_boot'], seed=result['seed'], marker_details=result['markers']), contrasts={}, directions={})
    for row in result['markers']:
        value = {key: row[key] for key in ('estimate', 'ci_low', 'ci_high', 'unit', 'status') if key in row}
        value.update(n=row.get('n_families'), n_unit='概念家族',
                     note='方向与原尺度组差分别报告，分类界限不是抽样区间。')
        bounds = row.get('classification_bounds')
        if bounds:
            value['classification_bounds'] = {key: bounds[key] for key in ('lower', 'upper')}
            value['sampling_envelope'] = {key: bounds['sampling_envelope'][key] for key in ('lower', 'upper')
                                          if key in bounds.get('sampling_envelope', {})}
        if row.get('baseline'):
            value['note'] += '无提示成绩与可恢复空间未作匹配。'
            value['baseline'] = {state: dict(lower=bound['lower'], upper=bound['upper'], unit='proportion',
                recovery_room=dict(lower=1 - bound['upper'], upper=1 - bound['lower'], unit='proportion'),
                note='固定样本的未决回答分类界限，不是抽样区间。')
                for state, bound in row['baseline'].items()}
        if row.get('denominators'):
            den = row['denominators']
            value['note'] += (f"比例分母为 {result['sd_state']} 的 {den['sd']} 个、"
                              f"{result['sa_state']} 的 {den['sa']} 个单词语义错误。n 是配对概念数。")
        report['contrasts'][row['marker']] = value
    score = scorecard(result['markers'])
    report['meta']['direction_coverage'] = score
    for family, value in score['families'].items():
        report['directions'][family] = dict(lower=value['lower'], upper=value['upper'],
            n=value['n_markers'], n_unit='预定读数', unit='equal-weight direction score',
            note='平局计半分，未覆盖保持缺失。这不是效应量、拟合百分比或置信区间。')
    report['directions']['overall'] = dict(lower=score['lower'], upper=score['upper'], n=score['n_families'],
        n_unit='标记族', unit='equal-weight direction score', note=score['interpretation'])
    return report
