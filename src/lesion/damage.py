"""Damage h and the control and knowledge effects (spec.md, 统计分析与数据划分; ticket pilot 11).

Pure pandas/numpy: works on label-score tables from lesion.scoring, knows nothing about models or hooks.

h per item = margin on the intact model - margin with the lesion, where the margin is the logit of the
correct option minus the wrong one, averaged over the two option orders. Larger h = more harm.
"""
import numpy as np
import pandas as pd

MATCHING = ("colour", "size")
SEMANTIC = ("colour", "size", "synonym")
EFFECTS = ("control_effect", "knowledge_effect", "control_slope", "knowledge_slope", "semantic_damage", "string_damage")
N_BOOT, SEED = 10_000, 0


def item_margin(label_scores):
    """Mean over the two option orders of logit_correct - logit_wrong, indexed by item_id."""
    s = label_scores.assign(margin=label_scores.logit_correct - label_scores.logit_wrong)
    g = s.groupby("item_id")["margin"]
    if not (g.size() == 2).all():
        raise ValueError("every item needs both option orders")
    return g.mean()


def item_damage(intact_scores, lesioned_scores):
    """h per item (Series indexed by item_id); both tables must cover the same items."""
    a, b = item_margin(intact_scores), item_margin(lesioned_scores)
    if set(a.index) != set(b.index):
        raise ValueError("intact and lesioned scores cover different items")
    return (a - b.reindex(a.index)).rename("h")


def frequency_median(all_items):
    """Median min_freq of all M1 synonym items (2.585, splitting them 80/80); fixed, independent of the split."""
    return float(pd.to_numeric(all_items.loc[all_items.task == "synonym", "min_freq"]).median())


def item_frame(h, items, freq_median):
    """One row per item with its damage columns and the item properties the effects need.

    `h` is a Series (one damage column "h") or a DataFrame (one column per lesion, e.g. h_control, h_knowledge).
    """
    h = h.to_frame() if isinstance(h, pd.Series) else h
    meta = items.set_index("item_id")[["task", "concept_group", "high_demand", "tds", "min_freq", "synonym_type"]]
    t = h.join(meta, how="left")
    if t.task.isna().any():
        raise ValueError("damage given for items not in the item table")
    t["tds"] = pd.to_numeric(t.tds, errors="coerce")
    t["min_freq"] = pd.to_numeric(t.min_freq, errors="coerce")
    t["high"] = np.where(t.task.isin(MATCHING), (t.high_demand == "1").astype(int), -1)
    t["low_freq"] = (t.task == "synonym") & (t.min_freq < freq_median)
    return t.drop(columns="high_demand").reset_index()


def _paired_margins(intact_scores, lesioned_scores, items):
    """Per item: margin intact and lesioned, task and high_demand."""
    a, b = item_margin(intact_scores), item_margin(lesioned_scores)
    if set(a.index) != set(b.index):
        raise ValueError("intact and lesioned scores cover different items")
    t = pd.DataFrame({"intact": a, "lesioned": b.reindex(a.index)}).join(
        items.set_index("item_id")[["task", "high_demand"]], how="left")
    if t.task.isna().any():
        raise ValueError("scores given for items not in the item table")
    return t


def demand_breakdown(intact_scores, lesioned_scores, items):
    """Descriptive table, one row per task and control demand (colour and size split into high and low demand,
    synonym and string whole): n, the mean margin intact and lesioned, mean h, and the share of items answered
    correctly (margin averaged over the two option orders above 0) intact and lesioned."""
    t = _paired_margins(intact_scores, lesioned_scores, items)
    t["demand"] = np.where(t.task.isin(MATCHING), np.where(t.high_demand == "1", "high", "low"), "all")
    t["h"] = t.intact - t.lesioned
    g = t.groupby(["task", "demand"])
    return pd.DataFrame(dict(n=g.size(), intact_margin=g.intact.mean(), lesioned_margin=g.lesioned.mean(), h=g.h.mean(),
                             intact_correct=g.intact.apply(lambda x: float((x > 0).mean())),
                             lesioned_correct=g.lesioned.apply(lambda x: float((x > 0).mean())))).reset_index()


def margin_scaling(intact_scores, lesioned_scores, items):
    """Descriptive, one row per task and one for all items: the least-squares line lesioned margin = slope * intact
    margin + intercept over items, and the correlation. A lesion that shrinks every answer's margin by the same factor,
    whatever the task, gives the same slope below 1 on every task with an intercept near 0; h then grows with the
    intact margin, so the tasks the model is surest of look most damaged."""
    t = _paired_margins(intact_scores, lesioned_scores, items)

    def fit(g):
        slope, intercept = np.polyfit(g.intact, g.lesioned, 1)
        return dict(n=len(g), slope=slope, intercept=intercept, r=np.corrcoef(g.intact, g.lesioned)[0, 1])

    rows = [dict(task=task, **fit(g)) for task, g in t.groupby("task")]
    return pd.DataFrame(rows + [dict(task="all", **fit(t))])


def _slope(y, x, covariate):
    """OLS slope of y on standardised x with an intercept and a 0/1 covariate; nan if not estimable."""
    if len(y) < 4 or np.nanstd(x) == 0:
        return np.nan
    z = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones(len(y)), z, covariate]) if np.ptp(covariate) > 0 else np.column_stack([np.ones(len(y)), z])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def effects(cols, h="h"):
    """The six effects from column arrays (a dict of numpy arrays, one entry per item_frame column).

    control_effect: within colour and size separately, mean h on high-demand minus low-demand items, then averaged.
    knowledge_effect: synonym items, mean h below the frequency median minus above it.
    control_slope / knowledge_slope: the same directions as regressions on standardised TDS / min_freq
      (sign flipped so that positive means more harm where control demand is high / words are rare).
    semantic_damage / string_damage: mean h on the semantic tasks / on string matching.
    """
    y, task, high = cols[h], cols["task"], cols["high"]
    per_task = []
    for tk in MATCHING:
        hi, lo = y[(task == tk) & (high == 1)], y[(task == tk) & (high == 0)]
        if len(hi) and len(lo):
            per_task.append(hi.mean() - lo.mean())
    syn = task == "synonym"
    rare, common = y[syn & cols["low_freq"]], y[syn & ~cols["low_freq"]]
    fm = np.isin(task, MATCHING)
    sem, string = np.isin(task, SEMANTIC), task == "string"
    return dict(
        control_effect=float(np.mean(per_task)) if per_task else np.nan,
        knowledge_effect=float(rare.mean() - common.mean()) if len(rare) and len(common) else np.nan,
        control_slope=-_slope(y[fm], cols["tds"][fm], (task[fm] == "size").astype(float)),
        knowledge_slope=-_slope(y[syn], cols["min_freq"][syn], (cols["synonym_type"][syn] == "similar").astype(float)),
        semantic_damage=float(y[sem].mean()) if sem.any() else np.nan,
        string_damage=float(y[string].mean()) if string.any() else np.nan,
    )


def columns(frame):
    return {c: frame[c].to_numpy() for c in frame.columns}


def bootstrap(frame, stat, n=N_BOOT, seed=SEED, group="concept_group"):
    """Point estimate and 95% percentile interval of every value stat() returns, resampling concept groups.

    stat takes a dict of column arrays and returns a dict of floats. Resamples where a value cannot be
    computed (e.g. no high-demand item drawn) are left out of that value's interval; n_valid says how many remain.
    """
    cols = columns(frame)
    uniq, inv = np.unique(cols[group], return_inverse=True)
    rows_of = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    point = stat(cols)
    samples = {k: np.empty(n) for k in point}
    rng = np.random.default_rng(seed)
    for b in range(n):
        idx = np.concatenate([rows_of[i] for i in rng.integers(len(uniq), size=len(uniq))])
        s = stat({c: v[idx] for c, v in cols.items()})
        for k in point:
            samples[k][b] = s[k]
    out = {}
    for k, v in point.items():
        ok = samples[k][np.isfinite(samples[k])]
        lo, hi = np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan)
        out[k] = dict(estimate=float(v), ci_low=float(lo), ci_high=float(hi), n_valid=int(len(ok)))
    return out
