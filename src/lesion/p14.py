"""P14, learned gates that grow minimal-sufficient component sets (docs/plans/p14-learned-gates.md; agent1's module):
the data splits, the candidate pool, the training objective's pieces and the rule that picks a set from a run.

Revised after agent2's plan review (reports/explore/2026-10-15-p14-prep/1-agent2-plan-review.md):
- splits: the knowledge attribution used the screening and dev items, so dev is not a clean selection set. The gates
  are trained on the screening items; the set size is chosen on one half of the 645 unused eligible words ("select")
  and the minimality tests run on the other half ("confirm"); the sealed test items come last. M1: screening trains,
  dev selects, test reports.
- objective: a selectivity coefficient (knowledge: the rare-word coefficient of the item damage with the answer's
  token count as covariate; control: the strong-minus-weak distractor difference within colour and size) is
  maximised, the general damage measures are constraints with fixed tolerances (squared hinge in tolerance units),
  and off_mass = sum(1 - g) is the sparsity term.
- the set is not "g < 0.5": the components are ordered by g, real mean ablation of the first k is measured on the
  select split for k on a grid, and the smallest feasible k within delta of the best feasible selectivity is taken.
"""
import numpy as np
import pandas as pd
import torch

from .components import Component

SPLIT_SEED = 1415
GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)
# Tolerances, fixed before training, about twice what matched random sets of six components did in session 13
# (100 sets: common naming damage 95th percentile 0.09 nats; 20 sets: language-model loss change max 0.6%; M1 semantic
# and string damage medians 0.35 and 0.38).
TOLERANCES = {
    "knowledge": {"common_naming": 0.2, "lm_change": 0.012, "m1_semantic": 0.5, "m1_string": 0.5},
    "control": {"naming": 0.2, "lm_change": 0.012, "m1_string": 0.5},
    # an SD-like hub loses rare concepts in comprehension too (controlled semantic cognition), so M1 semantic damage is
    # part of its selectivity, not a constraint
    "hub": {"common_naming": 0.2, "lm_change": 0.012, "m1_string": 0.5},
}
MATCHING = ("colour", "size")


# --- data splits -------------------------------------------------------------------------------------------------------

def unused_words(items, demos, eligible):
    """The eligible words never used as an item or a demo, ids u000..., in the order of eligible.csv (as
    run_k8b_split.py's length check)."""
    used = set(items.target) | set(demos.target)
    pool = eligible[~eligible.target.isin(used)].reset_index(drop=True).copy()
    pool["item_id"] = [f"u{i:03d}" for i in range(len(pool))]
    return pool


def split_unused(words, seed=SPLIT_SEED):
    """{"select": ..., "confirm": ...}: the words sorted by Zipf and taken in consecutive pairs, one of each pair drawn
    at random for each half, so that both halves span the same frequencies (an odd last word goes to select)."""
    rng = np.random.default_rng(seed)
    order = words.sort_values(["zipf", "item_id"]).item_id.tolist()
    select, confirm = [], []
    for i in range(0, len(order), 2):
        pair = order[i:i + 2]
        if len(pair) == 1:
            select += pair
            continue
        first = int(rng.integers(2))
        select.append(pair[first])
        confirm.append(pair[1 - first])
    by_id = words.set_index("item_id")
    return {name: by_id.loc[ids].reset_index() for name, ids in (("select", sorted(select)), ("confirm", sorted(confirm)))}


# --- candidate pool ----------------------------------------------------------------------------------------------------

def build_pool(attribution, column, top, screened=(), exclude=(), n_sentinels=0, seed=SPLIT_SEED):
    """(members, sources): the `top` components by attribution[column] (excluded ones skipped), every screened
    component, and n_sentinels drawn at random from the rest, spread over kinds and layers as the rest is (they
    estimate what the attribution cut misses). sources maps each member to "attribution", "screened" and/or
    "sentinel"."""
    exclude = set(exclude)
    ranked = attribution[~attribution.component.isin(exclude)].sort_values(column, ascending=False)
    sources = {}
    for c in ranked.component.head(top):
        sources.setdefault(c, []).append("attribution")
    for c in screened:
        if c not in exclude:
            sources.setdefault(c, []).append("screened")
    rest = ranked[~ranked.component.isin(sources)]
    if n_sentinels:
        rng = np.random.default_rng(seed)
        strata = rest.groupby(["kind", "layer"]).component.apply(list)
        weights = strata.apply(len).to_numpy(dtype=float)
        picks = set()
        want = min(n_sentinels, len(rest))
        for _ in range(100 * want + 1000):          # bounded: duplicates are redrawn, never forever
            if len(picks) >= want:
                break
            s = strata.iloc[int(rng.choice(len(strata), p=weights / weights.sum()))]
            picks.add(s[int(rng.integers(len(s)))])
        for c in sorted(picks):
            sources[c] = ["sentinel"]
    members = sorted(sources, key=lambda k: Component.parse(k))
    return members, sources


# --- objective pieces (torch, differentiable in the damage) ------------------------------------------------------------

MAX_CONDITION = 1e6     # above this the token-count covariate is dropped (it is then nearly the rare flag itself)


def rare_coefficient(h, rare, n_tokens):
    """OLS coefficient of the rare flag in h ~ 1 + rare + standardised token count (both sets of items needed).
    In a small batch the token count can be (nearly) collinear with the rare flag; then the covariate is dropped and
    the coefficient is the plain rare-minus-common difference of means (agent2's code review, 2026-10-15)."""
    r = rare.to(device=h.device, dtype=h.dtype)
    t = n_tokens.to(device=h.device, dtype=h.dtype)
    if not (bool(r.any()) and bool((r == 0).any())):
        raise ValueError("rare_coefficient needs rare and common items")
    cols = [torch.ones_like(r), r]
    if t.numel() > 3 and float(t.std()) > 0:          # a constant token count adds nothing (and would be singular)
        cols.append((t - t.mean()) / t.std())
    X = torch.stack(cols, dim=1)
    xtx = X.T @ X
    if X.shape[1] == 3:
        cond = float(torch.linalg.cond(xtx.detach().double()))
        if not np.isfinite(cond) or cond > MAX_CONDITION:
            X = X[:, :2]
            xtx = X.T @ X
    return torch.linalg.solve(xtx, X.T @ h)[1]         # normal equations: differentiable in h


def control_difference(h, task, high):
    """Within colour and size separately, mean h on the strong-distractor items minus the weak ones, then averaged
    over the tasks that have both kinds in the batch (as lesion.damage's control_effect). task: list of names; high:
    bool tensor."""
    parts = []
    for name in MATCHING:
        on = torch.tensor([t == name for t in task], device=h.device)
        hi, lo = on & high, on & ~high
        if hi.any() and lo.any():
            parts.append(h[hi].mean() - h[lo].mean())
    if not parts:
        raise ValueError("the batch needs strong- and weak-distractor items of one matching task")
    return torch.stack(parts).mean()


def violation(value, tolerance):
    """Squared hinge in tolerance units: 0 within the tolerance, ((value / tolerance) - 1)^2 beyond it."""
    return torch.relu(value / tolerance - 1) ** 2


def off_mass(g):
    """Sum of (1 - g): how far the gates are closed in total (the sparsity term; not a count of kept components)."""
    return (1 - g).sum()


# --- choosing a set from a run's curve ---------------------------------------------------------------------------------

def feasible(row, side):
    tol = TOLERANCES[side]
    return all(row[k] <= v for k, v in tol.items())


def choose_size(curve, side, delta=None):
    """(k, delta): the smallest k whose real-ablation selectivity on the select split is within delta of the best
    feasible one, among the feasible rows (curve: one row per k with "selectivity", "selectivity_se" and the
    tolerance columns); delta defaults to the standard error at the best feasible row. (None, None) if no row is
    feasible."""
    ok = curve[[feasible(r, side) and np.isfinite(r["selectivity"]) for r in curve.to_dict("records")]]
    if ok.empty:
        return None, None
    top = ok.loc[ok.selectivity.idxmax()]
    delta = float(top.selectivity_se) if delta is None else delta
    return int(ok[ok.selectivity >= top.selectivity - delta].k.min()), delta


def rare_coefficient_np(h, rare, n_tokens):
    """rare_coefficient on numpy arrays (for the curves and the confirmation); nan unless both rare and common items
    are present (a resample can miss one)."""
    rare = np.array(rare, dtype=bool)
    if rare.all() or not rare.any():
        return float("nan")
    return float(rare_coefficient(torch.tensor(np.array(h, dtype=float)), torch.tensor(rare),
                                  torch.tensor(np.array(n_tokens, dtype=float))))


def bootstrap_se(values_fn, n_items, n=1000, seed=SPLIT_SEED):
    """Standard error of values_fn(index array) over item resamples."""
    rng = np.random.default_rng(seed)
    draws = [values_fn(rng.integers(n_items, size=n_items)) for _ in range(n)]
    return float(np.nanstd(draws))


def matched_order(scores):
    """Components ordered by their final g, most closed first (ties by key): the order the curve adds them in."""
    return [k for k, _ in sorted(scores.items(), key=lambda kv: (kv[1], Component.parse(kv[0])))]


def stability(groups):
    """Across runs: each component's selection frequency, and the mean pairwise Jaccard index of the sets."""
    keys = sorted({c for g in groups for c in g}, key=Component.parse)
    freq = {c: sum(c in g for g in groups) / len(groups) for c in keys}
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in pairs if set(a) | set(b)]
    return dict(frequency=freq, mean_jaccard=float(np.mean(jac)) if jac else None, n_runs=len(groups))


def spearman(a, b):
    """Spearman rank correlation of two equal-length sequences."""
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))
