"""Experiment 3 statistics (docs/plans/experiment3-protocol.md v0.5 §5 to §7, §10): H3a error types in free
naming, H3b cue and miscue, each compared with the general degradation model (lesion.degradation).

Inputs are free-naming score tables as written by lesion.m3_scoring.score_naming_prompts, with a `set` column
(intact, K, C, ...): one row per item, condition and candidate word, `logprob` summed over the word's tokens.

- H3a (§6): condition "none", the five candidates ERROR_ROLES. Observed distribution: softmax over the five
  log-probabilities. S = p(associate) - p(coordinate) - p(superordinate); dS = S_lesion - S_intact;
  D = dS_C - dS_K, predicted > 0.
- H3b (§7): conditions none, cue, miscue, candidates target and associate. Delta = logP(target) - logP(associate);
  R = Delta_cue - Delta_miscue; dR = R_lesion - R_intact; D = dR_C - dR_K, predicted > 0.
- Degradation (§5): per lesion and task, (T, lam) fitted on calibration concepts (H3a: part 2b; H3b: part 3) from
  the intact and lesioned scores of those concepts; the predicted statistic on the test concepts comes from the
  intact test scores. Excess = D_obs - D_pred. Intervals: calibration and test concepts resampled independently,
  the fit repeated on every resample (§5.4).
"""
import numpy as np
import pandas as pd

from . import degradation as dg
from .m3_scoring import ERROR_ROLES

H3B_CONDITIONS = ("none", "cue", "miscue")
H3B_ROLES = ("target", "associate")


def _logprobs(scores, condition, roles):
    t = scores[scores.condition == condition].pivot_table(index="item_id", columns="role", values="logprob", sort=False)
    return t[list(roles)]


def candidate_zipf(items, zipf, roles=ERROR_ROLES):
    """Zipf of every candidate word (items: final_items rows indexed by item_id); raises on a word without one."""
    missing = sorted({items.at[i, r] for i in items.index for r in roles if items.at[i, r] not in zipf})
    if missing:
        raise ValueError(f"no SUBTLEX-US frequency for {missing}")
    return pd.DataFrame({r: [zipf[items.at[i, r]] for i in items.index] for r in roles}, index=items.index)


# --- H3a --------------------------------------------------------------------------------------------------------

def s_stat(logp):
    """S per item from log-probabilities over ERROR_ROLES (columns in that order), after a softmax over them."""
    p = np.exp(dg.log_softmax(np.asarray(logp, dtype=float)))
    r = {role: k for k, role in enumerate(ERROR_ROLES)}
    return p[:, r["associate"]] - p[:, r["coordinate"]] - p[:, r["superordinate"]]


def h3a_tables(scores, items_ids):
    """{set: log-probabilities (items x ERROR_ROLES)} for condition none, rows in items_ids order."""
    return {s: _logprobs(g, "none", ERROR_ROLES).loc[list(items_ids)].to_numpy() for s, g in scores.groupby("set")}


def h3a_predicted_grid(z, prior):
    """Predicted S for every test item at every grid point: (n_items, len(T_GRID) * len(LAMBDA_GRID))."""
    out = np.empty((len(z), len(dg.T_GRID), len(dg.LAMBDA_GRID)))
    for i, T in enumerate(dg.T_GRID):
        for j, lam in enumerate(dg.LAMBDA_GRID):
            out[:, i, j] = s_stat(dg.log_predict(z, prior, T, lam))
    return out.reshape(len(z), -1)


# --- H3b --------------------------------------------------------------------------------------------------------

def h3b_tables(scores, items_ids):
    """{set: array (items x 3 conditions x 2 candidates)} of log-probabilities, conditions H3B_CONDITIONS."""
    out = {}
    for s, g in scores.groupby("set"):
        out[s] = np.stack([_logprobs(g, c, H3B_ROLES).loc[list(items_ids)].to_numpy() for c in H3B_CONDITIONS], axis=1)
    return out


def r_stat(logp):
    """R per item from (items x conditions x [target, associate]) log-probabilities (restricted log odds)."""
    d = logp[..., 0] - logp[..., 1]
    return d[:, 1] - d[:, 2]


def h3b_item_ce_grid(lesioned, intact, prior):
    """Per-concept cross-entropy on the grid, summed over the three conditions: (n_items, nT, nLam)."""
    n = len(intact)
    p_obs = np.exp(dg.log_softmax(lesioned.reshape(n * 3, 2)))
    ce = dg.item_cross_entropy_grid(p_obs, intact.reshape(n * 3, 2), np.repeat(prior, 3, axis=0))
    return ce.reshape(n, 3, *ce.shape[1:]).sum(axis=1)


def h3b_predicted_grid(z, prior):
    """Predicted R for every test item at every grid point; z (items x 3 x 2), prior (items x 2)."""
    n = len(z)
    out = np.empty((n, len(dg.T_GRID), len(dg.LAMBDA_GRID)))
    flat_prior = np.repeat(prior, 3, axis=0)
    for i, T in enumerate(dg.T_GRID):
        for j, lam in enumerate(dg.LAMBDA_GRID):
            out[:, i, j] = r_stat(dg.log_predict(z.reshape(n * 3, 2), flat_prior, T, lam).reshape(n, 3, 2))
    return out.reshape(n, -1)


# --- one candidate's share against the degradation model (S5, analysis plan §7; user decision Q21) ----------------

def candidate_excess_test(z_intact, logp_lesion, prior, ce_cal, role, n_boot=10_000, seed=0):
    """Change of one candidate's share (softmax over ERROR_ROLES) under a lesion, observed minus predicted by the
    degradation model fitted on calibration concepts (ce_cal: per-concept grid of that lesion). Calibration and
    test concepts resampled independently, refit on every resample. Returns observed, predicted, excess with
    percentile intervals and the one-sided p for excess < 0."""
    j = ERROR_ROLES.index(role)
    p_int = np.exp(dg.log_softmax(np.asarray(z_intact, dtype=float)))[:, j]
    obs = np.exp(dg.log_softmax(np.asarray(logp_lesion, dtype=float)))[:, j] - p_int
    grid = np.empty((len(p_int), len(dg.T_GRID) * len(dg.LAMBDA_GRID)))
    k = 0
    for T in dg.T_GRID:
        for lam in dg.LAMBDA_GRID:
            grid[:, k] = np.exp(dg.log_predict(z_intact, prior, T, lam))[:, j] - p_int
            k += 1
    rng = np.random.default_rng(seed)
    ct, cc = resample_counts(len(p_int), n_boot, rng), resample_counts(len(ce_cal), n_boot, rng)
    _, _, (i0, j0) = dg.best(ce_cal.sum(axis=0))
    g0 = i0 * len(dg.LAMBDA_GRID) + j0
    idx = dg.bootstrap_fits(ce_cal, cc)
    gb = idx[:, 0] * len(dg.LAMBDA_GRID) + idx[:, 1]
    b_obs = ct @ obs / len(obs)
    b_pred = np.einsum("bi,ib->b", ct, grid[:, gb]) / len(obs)
    b_exc = b_obs - b_pred

    def ci(x):
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    exc = float(obs.mean() - grid[:, g0].mean())
    return dict(role=role, observed=float(obs.mean()), observed_ci=ci(b_obs), predicted=float(grid[:, g0].mean()),
                predicted_ci=ci(b_pred), excess=exc, excess_ci=ci(b_exc),
                p_one_sided=float(((b_exc >= 0).sum() + 1) / (n_boot + 1)), n=int(len(obs)))


# --- evidence weighting (protocol §5.3a, user decision Q19 2026-10-08) ----------------------------------------------

def evidence_weights(d_intact, d_lesion):
    """Least-squares weights on the log-odds scale (Delta = logP target - logP associate), per lesion:

        Delta_hat_base   = a * Di_base
        Delta_hat_cue    = a * Di_base + b_cue    * (Di_cue    - Di_base)
        Delta_hat_miscue = a * Di_base + b_miscue * (Di_miscue - Di_base)

    d_intact, d_lesion: arrays (n_items, 3) with columns base, cue, miscue (base = no hint; with
    name_hinted_demos as base the demo shift is removed). The fixed-cue null of §5.3a is b_cue = b_miscue = 1.
    Returns (a, b_cue, b_miscue)."""
    di, dl = np.asarray(d_intact, dtype=float), np.asarray(d_lesion, dtype=float)
    n = len(di)
    X = np.zeros((3 * n, 3))
    X[:, 0] = np.tile(di[:, 0], 3)
    X[n:2 * n, 1] = di[:, 1] - di[:, 0]
    X[2 * n:, 2] = di[:, 2] - di[:, 0]
    y = np.concatenate([dl[:, 0], dl[:, 1], dl[:, 2]])
    return tuple(float(v) for v in np.linalg.lstsq(X, y, rcond=None)[0])


def evidence_weights_bootstrap(d_intact, d_lesion, n_boot=10_000, seed=0):
    """Point weights and percentile intervals over resampled concepts."""
    rng = np.random.default_rng(seed)
    di, dl = np.asarray(d_intact, dtype=float), np.asarray(d_lesion, dtype=float)
    point = evidence_weights(di, dl)
    boots = np.array([evidence_weights(di[i], dl[i]) for i in rng.integers(0, len(di), (n_boot, len(di)))])
    names = ("a", "b_cue", "b_miscue")
    return {k: dict(estimate=point[j], ci=[float(np.quantile(boots[:, j], 0.025)), float(np.quantile(boots[:, j], 0.975))])
            for j, k in enumerate(names)}


# --- comparison with the degradation model ------------------------------------------------------------------------

def resample_counts(n, n_boot, rng):
    """How often each of n units is drawn in each of n_boot resamples (n_boot x n)."""
    draws = rng.integers(0, n, size=(n_boot, n))
    return np.stack([np.bincount(d, minlength=n) for d in draws])


def excess_test(obs, pred_grid, ce_cal, n_boot=10_000, seed=0, mde=None):
    """D_obs, D_pred and excess = D_obs - D_pred with percentile intervals (§5.4).

    obs: {"intact", "C", "K"} -> per-test-item statistic (S or R); pred_grid: {"C", "K"} -> (n_test, n_grid)
    predicted statistic; ce_cal: {"C", "K"} -> per-calibration-concept cross-entropy grid (n_cal, nT, nLam).
    Calibration and test concepts are resampled independently; C and K share each resample (paired)."""
    rng = np.random.default_rng(seed)
    n_test, n_cal = len(obs["intact"]), len(next(iter(ce_cal.values())))
    ct, cc = resample_counts(n_test, n_boot, rng), resample_counts(n_cal, n_boot, rng)
    nlam = len(dg.LAMBDA_GRID)
    point, boots, fits = {}, {}, {}
    for L in ("C", "K"):
        ce = ce_cal[L]
        T, lam, (i, j) = dg.best(ce.sum(axis=0))
        g = i * nlam + j
        idx = dg.bootstrap_fits(ce, cc)
        fits[L] = dict(T=T, lam=lam, **dg.ridge_check(idx))
        d_obs = obs[L] - obs["intact"]
        d_pred_items = pred_grid[L] - obs["intact"][:, None]
        point[L] = (float(d_obs.mean()), float(d_pred_items[:, g].mean()))
        gb = idx[:, 0] * nlam + idx[:, 1]
        boots[L] = ((ct @ d_obs) / n_test, np.einsum("bi,ib->b", ct, d_pred_items[:, gb]) / n_test)
    d_obs = point["C"][0] - point["K"][0]
    d_pred = point["C"][1] - point["K"][1]
    b_obs = boots["C"][0] - boots["K"][0]
    b_pred = boots["C"][1] - boots["K"][1]
    b_exc = b_obs - b_pred

    def ci(x):
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    out = dict(n_test=n_test, n_calibration=n_cal, n_boot=n_boot, seed=seed, fits=fits,
               change={L: dict(observed=point[L][0], predicted=point[L][1], observed_ci=ci(boots[L][0]),
                               predicted_ci=ci(boots[L][1])) for L in ("C", "K")},
               D_obs=d_obs, D_obs_ci=ci(b_obs), D_pred=d_pred, D_pred_ci=ci(b_pred),
               excess=d_obs - d_pred, excess_ci=ci(b_exc),
               p_one_sided=float(((b_exc <= 0).sum() + 1) / (n_boot + 1)))
    out["supported"] = bool(out["D_obs_ci"][0] > 0 and out["excess_ci"][0] > 0)
    if mde is not None:
        out["small"] = bool(out["supported"] and out["excess"] < mde)
    return out
