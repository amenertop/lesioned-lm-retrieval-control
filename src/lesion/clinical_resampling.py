"""Paired item and available-patient uncertainty for the measured clinical comparison.

Published frequency and cue means stay fixed. Their unavailable patient distributions are not invented.
Figure transcription bounds are a separate analysis, not sampling confidence intervals.
"""
import numpy as np
import pandas as pd

from . import clinical_alignment as ca, title_gap_analysis as ta


def balanced_errors(predictions):
    """Equal group and family weights, retaining missing held-out errors in each aggregate."""
    complete_mean = lambda x: x.mean() if x.notna().all() else np.nan
    family = predictions.groupby(['model', 'family', 'group']).error.agg(complete_mean)
    family = family.groupby(['model', 'family']).agg(complete_mean).rename('error').reset_index()
    overall = family.groupby('model').error.agg(complete_mean).reset_index().assign(family='overall')
    result = pd.concat([family, overall], ignore_index=True)
    if {'single', 'dual'} <= set(result.model):
        paired = result.pivot(index='family', columns='model', values='error')
        result = pd.concat([result, (paired.dual-paired.single).rename('error').reset_index()
                            .assign(model='dual_minus_single')], ignore_index=True)
    return result


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan), where=denominator > 0)


def marker_samples(clinical, labels, items, patients, individual, n_boot=1000, seed=1910):
    """Yield paired marker draws. Draw zero uses the original sample once, for a point-estimate check."""
    if n_boot < 0 or clinical.category.eq('unclassified').any():
        raise ValueError('resampling requires nonnegative draws and fully reviewed clinical responses')
    ca.cue_profiles(clinical, n_boot=0)  # Validate the complete progressive sequences.
    states = sorted(clinical.state.unique())
    concepts = sorted(clinical.item_id.unique())
    if any(set(part.item_id) != set(concepts) for _, part in clinical.groupby('state')):
        raise ValueError('clinical concepts must remain paired across states')
    correct = clinical.assign(correct=clinical.category.eq('correct'))
    none = correct[correct.letters == 0].pivot(index='state', columns='item_id', values='correct').reindex(index=states, columns=concepts)
    first_two = correct[correct.letters <= 2].groupby(['state', 'item_id']).correct.any().unstack().reindex(index=states, columns=concepts)
    final = correct.groupby(['state', 'item_id']).correct.any().unstack().reindex(index=states, columns=concepts)
    wrong = 1-none.to_numpy(float)
    rescued = [wrong*endpoint.to_numpy(float) for endpoint in (first_two, final)]
    error_kinds = ['coordinate', 'superordinate', 'associate']
    categories = clinical[clinical.letters == 0].pivot(index='state', columns='item_id', values='category').reindex(index=states, columns=concepts)
    errors = np.asarray([categories.eq(kind).to_numpy(float) for kind in error_kinds])

    accuracy = ta.fine_label_accuracy(labels).query("fmt == 'synonym'")
    frequency_items = items.set_index('item_id').reindex(sorted(accuracy.item_id.unique()))
    matrix = accuracy.pivot(index='state', columns='item_id', values='correct_response').reindex(index=states, columns=frequency_items.index)
    if 'intact' not in matrix.index or matrix.loc['intact'].isna().any():
        raise ValueError('frequency resampling requires the shared intact baseline')
    if any(row.notna().any() and row.isna().any() for _, row in matrix.iterrows()):
        raise ValueError('partially measured frequency states cannot be resampled')
    bins = frequency_items.frequency_bin.to_numpy()
    if not {'low', 'high'} <= set(bins) or frequency_items.frequency_bin.isna().any():
        raise ValueError('frequency bins must be defined before resampling')

    source = individual[individual.task.eq('naming_error_analysis') &
                        individual.measure.isin(['coordinate', 'superordinate', 'associative'])].copy()
    if set(source.denominator) != {'single_word_semantic_errors'}:
        raise ValueError('patient resampling requires the original semantic-error proportions')
    source['marker'] = 'error_' + source.measure.replace({'associative': 'associate'})
    patient_arrays = {}
    for group in sorted(patients.group.unique()):
        data = source[source.group == group].pivot(index='patient_id', columns='marker', values='value').apply(pd.to_numeric, errors='coerce')
        columns = ['error_'+kind for kind in error_kinds]
        values = data.reindex(columns=columns).to_numpy(float)
        target = patients[patients.group == group].set_index('marker').value.reindex(columns).to_numpy(float)
        if not len(values) or not np.allclose(np.nanmean(values, axis=0), target, rtol=0, atol=1e-12):
            raise ValueError('patient rows do not reproduce the published error profile')
        patient_arrays[group] = values

    rng = np.random.default_rng(seed)
    for draw in range(n_boot+1):
        clinical_weights = np.ones(len(concepts)) if draw == 0 else rng.multinomial(len(concepts), np.full(len(concepts), 1/len(concepts)))
        frequency_weights = np.ones(len(bins))
        if draw:
            for name in sorted(set(bins)):
                size = int((bins == name).sum())
                frequency_weights[bins == name] = rng.multinomial(size, np.full(size, 1/size))
        high, low = [matrix.to_numpy(float)[:, bins == name] @ frequency_weights[bins == name] / frequency_weights[bins == name].sum()
                     for name in ('high', 'low')]
        healthy = float(matrix.loc['intact'].to_numpy(float) @ frequency_weights / frequency_weights.sum())
        values = {'frequency_gap': _ratio(high-low, healthy-.5)}
        for endpoint, numerator in zip(('two', 'final'), rescued):
            values['rescue_'+endpoint] = _ratio(numerator @ clinical_weights, wrong @ clinical_weights)
        counts = errors @ clinical_weights
        for kind, numerator in zip(error_kinds, counts):
            values['error_'+kind] = _ratio(numerator, counts.sum(axis=0))
        model = pd.DataFrame(values, index=pd.Index(states, name='state')).reset_index().melt(id_vars='state', var_name='marker', value_name='value')
        patient = patients[['group', 'marker', 'family', 'value']].copy()
        for group, data in patient_arrays.items():
            weights = np.ones(len(data)) if draw == 0 else rng.multinomial(len(data), np.full(len(data), 1/len(data)))
            mean = _ratio(weights @ np.nan_to_num(data), weights @ np.isfinite(data))
            for kind, value in zip(error_kinds, mean):
                patient.loc[(patient.group == group) & (patient.marker == 'error_'+kind), 'value'] = value
        yield draw, patient, model


def bootstrap_comparison(clinical, labels, items, patients, individual, curves, n_boot=1000, seed=1910, selection=None):
    """Refit development on each draw, or keep the development selection fixed on final items."""
    scores, choices = [], []
    markers = sorted(patients.marker.unique())
    keys = curves.merge(pd.DataFrame({'marker': markers}), how='cross')
    expected = ca.measured_marker_profiles(clinical, labels, items).set_index(['state', 'marker']).value.sort_index()
    point = None
    for draw, patient, model in marker_samples(clinical, labels, items, patients, individual, n_boot, seed):
        if draw == 0:
            actual = model.set_index(['state', 'marker']).value.reindex(expected.index)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)
        if not np.isfinite(patient.value).all():
            continue
        profiles = keys.merge(model, on=['state', 'marker'], how='left', validate='many_to_one')
        fit = (ca.compare_measured_profiles(patient, profiles) if selection is None else
               ca.predict_frozen_profiles(patient, profiles, selection))
        scores.append(balanced_errors(fit).assign(draw=draw))
        if draw == 0:
            point = fit
        else:
            choices.append(fit[['model', 'group', 'family', 'curve', 'state', 'status']].drop_duplicates().assign(draw=draw))
    draws = pd.concat(scores, ignore_index=True)
    summaries = []
    for (model, family), group in draws.groupby(['model', 'family'], sort=True):
        sample = group[(group.draw > 0) & group.error.notna()].error.to_numpy(float)
        value = group.loc[group.draw == 0, 'error'].iloc[0]
        ci = (np.quantile(sample, [.025, .975]).tolist() if pd.notna(value) and len(sample) >= .9*n_boot and n_boot
              else [None, None])
        summaries.append(dict(model=model, family=family, estimate=float(value) if pd.notna(value) else None,
            ci_low=ci[0], ci_high=ci[1], valid_draws=len(sample), requested_draws=n_boot,
            status=('missing_point_estimate' if pd.isna(value) else 'point_only' if not n_boot else
                    'estimated' if ci[0] is not None else 'insufficient_valid_draws'),
            negative_fraction=float((sample < 0).mean()) if len(sample) and model == 'dual_minus_single' else None))
    choices = pd.concat(choices, ignore_index=True) if choices else pd.DataFrame()
    counts = choices.groupby(['model', 'group', 'family', 'curve', 'state', 'status'], dropna=False).size().rename('draws').reset_index() if len(choices) else choices
    metadata = dict(n_boot=n_boot, seed=seed, scope='conditional on published frequency and cue group means',
        model_resampling='paired concepts across states, all cues within concept, frequency strata fixed',
        patient_resampling='paired error proportions within each available individual, separately within SD and SA',
        unavailable='patient distributions and cross-study covariance for frequency and cue means',
        selection='development refit within each draw' if selection is None else 'development states fixed on final items',
        figure_precision='handled separately, not included in these sampling intervals',
        clinical_concepts=int(clinical.item_id.nunique()), frequency_concepts=int(labels[labels.fmt == 'synonym'].item_id.nunique()))
    return dict(meta=metadata, summary=summaries), draws, counts, point


def ledger_report(answers, comparison, markers):
    """Export the measured comparison without treating resamples or figure corners as new patients."""
    for key in ('source_fingerprint', 'review_sha256'):
        if answers['meta'][key] != comparison['meta'][key]:
            raise ValueError('cue report and clinical comparison have different sources')
    report = dict(meta=dict(status=comparison['meta']['status'], cue_analysis=answers['meta'],
        comparison=comparison['meta'], sampling=comparison['sampling'],
        equal_weighting='mean squared error within group and marker family, then equal groups and families',
        figure_precision='coherent figure boundary scenarios, not a confidence interval',
        raw_predictions=comparison['predictions']), states={}, prediction={})
    for state, answer in answers['states'].items():
        if answer['status'] != 'fully_classified':
            raise ValueError('clinical report requires complete blind response judgements')
        saved = {}
        for metric in ('accuracy_none', 'accuracy_two', 'accuracy_final', 'rescue_two', 'rescue_final',
                       'independent_two', 'independent_final', 'initially_correct_lost_at_two'):
            saved[metric] = {k: v for k, v in answer[metric].items() if k != 'numerator'}
            saved[metric]['n_unit'] = ('无提示答错概念' if metric.startswith('rescue') else
                                      '无提示答对概念' if metric.startswith('initially') else '概念')
        errors = answer['naming_error_profile']
        for kind, value in errors['single_word_semantic_proportions'].items():
            saved['error_'+kind] = dict(estimate=value, n=errors['semantic_errors'], n_unit='单词语义错误', unit='proportion')
        frequency = markers[(markers.state == state) & (markers.marker == 'frequency_gap')].iloc[0]
        saved['frequency_gap'] = dict(estimate=float(frequency.value) if pd.notna(frequency.value) else None,
            n=int(frequency.n), n_unit='高低频概念', unit='chance-calibrated accuracy difference')
        report['states'][state] = saved
    fit = pd.DataFrame(comparison['predictions'])
    uncertainty = {(r['model'], r['family']): r for r in comparison['sampling']['summary']}
    for row in balanced_errors(fit).itertuples():
        ci = uncertainty[row.model, row.family]
        value = dict(estimate=float(row.error) if pd.notna(row.error) else None,
            ci_low=ci['ci_low'], ci_high=ci['ci_high'],
            n=int(fit.family.nunique()) if row.family == 'overall' else int(fit.group.nunique()),
            n_unit='留出标记族' if row.family == 'overall' else '病人组', unit='mean squared calibrated error')
        if row.model == 'dual_minus_single':
            if row.family == 'overall':
                report['prediction']['dual_minus_single'] = value
        else:
            report['prediction'].setdefault(row.model, {})[row.family] = value
    sensitivity = pd.DataFrame(comparison['figure_sensitivity'])
    differences = pd.Series({name: balanced_errors(part).query("model == 'dual_minus_single' and family == 'overall'").error.iloc[0]
                             for name, part in sensitivity.groupby('scenario')})
    report['figure_precision'] = {key: dict(estimate=float(value) if pd.notna(value) else None,
        n=len(differences), n_unit='读图边界场景', unit='mean squared calibrated error')
        for key, value in [('dual_minus_single_min', differences.min(skipna=False)),
                           ('dual_minus_single_max', differences.max(skipna=False))]}
    return report
