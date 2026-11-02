"""Development-only precision planning. Noise projections are not confirmatory effects or confidence intervals."""
import numpy as np
from scipy.stats import norm


def mean_precision(values, planned_n, n_sim=10000, seed=1910):
    """Resample centred, variance-corrected family differences at the proposed sample size."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all() or planned_n < 4:
        raise ValueError('precision planning needs complete independent family differences')
    noise = (values - values.mean()) * np.sqrt(len(values) / (len(values) - 1))
    ix = np.random.default_rng(seed).integers(len(values), size=(n_sim, planned_n))
    errors = noise[ix].mean(1)
    lo, hi = np.quantile(errors, [.025, .975])
    return dict(estimate=float((hi - lo) / 2), error_low=float(lo), error_high=float(hi),
                n=planned_n, n_unit='planned families', development_families=len(values),
                status='empirical_variance_zero_not_precision_guarantee' if np.ptp(values) == 0 else 'development_noise_projection',
                interpretation='projected 95% sampling-error half width, zero-centred development noise')


def homonym_precision(scores, responses, planned_n=40, n_sim=10000, seed=1910):
    """Keep two meanings and all cues in each family together. Semantic ambiguity is not filled in."""
    tables = [(name, rows, 'score') for name, rows in scores.groupby('readout')]
    tables.append(('exact_designated_target', responses, 'exact_correct'))
    result = {}
    for readout, rows, endpoint in tables:
        if set(rows.split) != {'dev'}:
            raise ValueError('precision planning must use development data only')
        for context, group in rows.groupby('context'):
            group = group[group.state.isin(['PC8_all_a0', 'intact'])]
            wide = group.pivot(index='family', columns=['state', 'dominance', 'cue'], values=endpoint)
            expected = {(s, d, c) for s in ('PC8_all_a0', 'intact') for d in ('dominant', 'subordinate')
                        for c in ('none', 'correct', 'misleading')}
            if set(wide.columns) != expected or wide.isna().any().any():
                raise ValueError('planning needs paired states, meanings and cues')
            a, b = wide['PC8_all_a0'].astype(float), wide.intact.astype(float)
            differences = {
                'subordinate_extra': (a.dominant['none'] - a.subordinate['none']) - (b.dominant['none'] - b.subordinate['none']),
                'cue_gain': (a.subordinate.correct - a.subordinate['none']) - (b.subordinate.correct - b.subordinate['none']),
                'misleading_cost': (a.subordinate['none'] - a.subordinate.misleading) - (b.subordinate['none'] - b.subordinate.misleading),
            }
            unit = {'candidate_log_odds': 'nat', 'choice_logit': 'logit', 'exact_designated_target': 'proportion'}[readout]
            result[f'{readout}/{context}'] = {
                name: dict(mean_precision(values, planned_n, n_sim, seed), unit=unit)
                for name, values in differences.items()}
    return result


def correlation_sensitivity(planned_n, covariates=5, correlations=(.1, .2, .3, .4, .5)):
    """Fisher-z approximation for one two-sided partial-correlation test, not power for the full SD pattern."""
    if planned_n <= covariates + 3:
        raise ValueError('too few concepts for the specified covariates')
    scale = np.sqrt(planned_n - covariates - 3)
    cutoff = norm.ppf(.975)
    return {f'r_{r:g}': dict(estimate=float(norm.sf(cutoff - np.arctanh(r)*scale)
                                          + norm.cdf(-cutoff - np.arctanh(r)*scale)),
                            unit='approximate power at two-sided alpha .05', n=planned_n, n_unit='planned concepts')
            for r in correlations}
