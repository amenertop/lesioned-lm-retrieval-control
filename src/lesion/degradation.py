"""General degradation model (docs/plans/experiment3-protocol.md §5): the alternative account in which a lesion
causes no specific deficit and only weakens the model overall, by flattening its output distribution and pulling
it towards frequent words.

For one item with intact scores z over a fixed candidate space and a word-frequency prior pi over the same
candidates, the predicted distribution after the lesion is

    p_hat = (1 - lam) * softmax(z / T) + lam * pi,     T >= 1, 0 <= lam <= 1.

(T, lam) are fitted per lesion and task on calibration concepts by grid search (60 log-spaced T in [1, 10], lam in
steps of 0.01), minimising the cross-entropy between the lesioned model's distribution over the candidates and
p_hat, summed over items. Uncertainty: concepts are resampled and the fit is repeated (§5.4); per-item
cross-entropies on the grid are computed once, so a resample's fit is a weighted sum.

Arrays: z, prior and p_obs have shape (n_items, n_candidates); candidates in a fixed order per task.
"""
import numpy as np

T_GRID = np.exp(np.linspace(0.0, np.log(10.0), 60))
LAMBDA_GRID = np.round(np.arange(0, 101) / 100, 2)


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def frequency_prior(zipf):
    """Prior over the candidates from SUBTLEX Zipf values (10^Zipf, normalised per item)."""
    w = 10.0 ** np.asarray(zipf, dtype=float)
    return w / w.sum(axis=-1, keepdims=True)


def log_predict(z, prior, T, lam):
    """log p_hat for every item (n_items x n_candidates), in log space so that lam = 0 or 1 stays finite where it
    should."""
    with np.errstate(divide="ignore"):
        a = np.log1p(-lam) + log_softmax(np.asarray(z, dtype=float) / T) if lam < 1 else np.full(np.shape(z), -np.inf)
        b = np.log(lam) + np.log(prior) if lam > 0 else np.full(np.shape(z), -np.inf)
    return np.logaddexp(a, b)


def item_cross_entropy_grid(p_obs, z, prior, t_grid=T_GRID, lam_grid=LAMBDA_GRID):
    """Per-item cross-entropy -sum_o p_obs log p_hat on the whole grid: (n_items, len(t_grid), len(lam_grid))."""
    p_obs, z, prior = (np.asarray(a, dtype=float) for a in (p_obs, z, prior))
    out = np.empty((len(z), len(t_grid), len(lam_grid)))
    with np.errstate(divide="ignore"):
        log_prior = np.log(prior)
        for i, T in enumerate(t_grid):
            ls = log_softmax(z / T)
            for j, lam in enumerate(lam_grid):
                a = np.log1p(-lam) + ls if lam < 1 else np.full(z.shape, -np.inf)
                b = np.log(lam) + log_prior if lam > 0 else np.full(z.shape, -np.inf)
                lp = np.logaddexp(a, b)
                out[:, i, j] = -np.where(p_obs > 0, p_obs * lp, 0.0).sum(axis=1)
    return out


def best(ce_sum, t_grid=T_GRID, lam_grid=LAMBDA_GRID):
    """(T, lam, index pair) at the minimum of a summed cross-entropy grid; ties go to the smaller T, then lam."""
    i, j = np.unravel_index(np.argmin(ce_sum), ce_sum.shape)
    return float(t_grid[i]), float(lam_grid[j]), (int(i), int(j))


def fit(p_obs, z, prior, t_grid=T_GRID, lam_grid=LAMBDA_GRID):
    """Point fit with the separability report of §5.3: the near-optimal region (summed cross-entropy within 1% of
    the minimum) and the profile of each parameter."""
    ce = item_cross_entropy_grid(p_obs, z, prior, t_grid, lam_grid).sum(axis=0)
    T, lam, (i, j) = best(ce, t_grid, lam_grid)
    near = ce <= ce.min() * 1.01
    return dict(T=T, lam=lam, cross_entropy=float(ce.min()),
                near_optimal=dict(n_points=int(near.sum()), T_range=[float(t_grid[near.any(axis=1)].min()),
                                                                     float(t_grid[near.any(axis=1)].max())],
                                  lam_range=[float(lam_grid[near.any(axis=0)].min()), float(lam_grid[near.any(axis=0)].max())]),
                profile_T=ce.min(axis=1).tolist(), profile_lam=ce.min(axis=0).tolist())


def bootstrap_fits(ce_items, counts, t_grid=T_GRID, lam_grid=LAMBDA_GRID):
    """Grid indices of the fit for each resample: counts (n_boot x n_items) are how often each calibration item
    was drawn. Returns (n_boot, 2) indices into (t_grid, lam_grid)."""
    flat = ce_items.reshape(len(ce_items), -1)
    sums = np.asarray(counts, dtype=float) @ flat
    k = np.argmin(sums, axis=1)
    return np.column_stack(np.unravel_index(k, (len(t_grid), len(lam_grid))))


def ridge_check(idx, t_grid=T_GRID, lam_grid=LAMBDA_GRID):
    """Correlation of T and lam across resamples; |r| > 0.9 means the two are not separable and the pair is used
    only as a prediction tool (§5.3)."""
    T, lam = t_grid[idx[:, 0]], lam_grid[idx[:, 1]]
    r = float(np.corrcoef(T, lam)[0, 1]) if T.std() > 0 and lam.std() > 0 else float("nan")
    return dict(correlation=r, ridge=bool(abs(r) > 0.9) if np.isfinite(r) else False,
                T_interval=[float(np.quantile(T, 0.025)), float(np.quantile(T, 0.975))],
                lam_interval=[float(np.quantile(lam, 0.025)), float(np.quantile(lam, 0.975))])
