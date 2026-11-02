"""Offline analyses of P14's final tests (agent1's module; exploratory, added after the data were seen; agent2's code
review 2026-10-17, reports/explore/2026-10-17-p14-code-review.md).

- M1: margins averaged over the two option orders first (lesion.damage.item_margin, the pilot protocol's item-level
  convention), then joined to the item table with every item's task, control demand and concept group required.
  margin_scaling: lesioned on intact margin per task. demand_at_equal_intact: the coefficient of the strong-distractor
  flag in lesioned ~ 1 + intact + high + size + intact:size over the colour and size items, a conditional association
  under a linear scaling model, with a cluster bootstrap over concept groups.
- Naming: the P14 objective's own statistic, the coefficient of the rare flag in h ~ 1 + rare + standardised token
  count (lesion.p14.rare_coefficient_np), with an item bootstrap; also for random sets, for an empirical comparison.
"""
import numpy as np
import pandas as pd

from . import damage, naming, p14


def minimality_fields(side, group):
    """The hub's confirmation tested naming only, not its joint search objective."""
    fields = dict(necessary=group["necessary"], n_redundant=len(group["redundant"]),
                  saturated_share=group["saturated_share"], pairs=group["pairs"])
    if side == "hub":
        return dict(minimality_scope="naming_only", **{f"naming_only_{k}": v for k, v in fields.items()})
    return dict(minimality_scope=side, **fields)


def control_minimality(table):
    """Signs and the registered necessity criterion are different summaries."""
    singles = table[(table.objective == "control") & (table.state == "minus_one")]
    pairs = table[(table.objective == "control") & (table.state == "minus_two")]
    return dict(singles=len(singles), singles_point_estimate_down=int((singles.change < 0).sum()),
                singles_point_estimate_up=int((singles.change > 0).sum()),
                singles_point_estimate_zero=int((singles.change == 0).sum()),
                singles_ci_excludes_zero=int((singles.ci_high < 0).sum()),
                singles_necessary=int(singles.necessary.astype(str).eq("True").sum()), pairs_tested=len(pairs),
                pairs_ci_excludes_zero=int((pairs.ci_high < 0).sum()),
                pairs_necessary=int(pairs.necessary.astype(str).eq("True").sum()))

MATCHED = ("colour", "size")


def item_table(items, required=("task", "concept_group")):
    """The M1 item table indexed by item_id as str (task, high_demand, concept_group), refusing a missing task or
    concept group. high_demand exists only for the colour and size items; order_averaged checks it there."""
    t = items.astype({"item_id": str}).set_index("item_id")[["task", "high_demand", "concept_group"]]
    if t.index.has_duplicates:
        raise ValueError("duplicate item_id in the item table")
    req = t[list(required)]
    if req.isna().any().any() or (req.astype(str).apply(lambda c: c.str.strip() == "")).any().any():
        raise ValueError("the item table has missing labels")
    return t


def order_averaged(scores, items, sets):
    """One row per item: each set's margin averaged over the two option orders, with task, high (bool) and concept
    group. Every set must cover the same items, each in both orders, and every item must be in the item table."""
    scores = scores.astype({"item_id": str})
    cols = {}
    for s in sets:
        part = scores[scores.set == s]
        if part.empty:
            raise ValueError(f"no scores for set {s}")
        cols[s] = damage.item_margin(part)              # raises unless every item has both orders
    index = cols[sets[0]].index
    for s in sets[1:]:
        if set(cols[s].index) != set(index):
            raise ValueError(f"set {s} covers other items than {sets[0]}")
    w = pd.DataFrame({s: cols[s].reindex(index) for s in sets})
    missing = sorted(set(w.index) - set(items.index))
    if missing:
        raise ValueError(f"items without labels: {missing[:5]}")
    w = w.join(items, how="left")
    hd = w.high_demand.astype(str)
    if not hd[w.task.isin(MATCHED)].isin(["0", "1"]).all():
        raise ValueError("colour and size items need high_demand 0 or 1")
    w["high"] = hd == "1"
    return w


def margin_scaling(w, set_name):
    out = {}
    for task, g in w.groupby("task"):
        slope, intercept = np.polyfit(g["intact"], g[set_name], 1)
        out[task] = dict(n=int(len(g)), slope=float(slope), intercept=float(intercept),
                         r=float(np.corrcoef(g["intact"], g[set_name])[0, 1]))
    return out


def demand_breakdown(w, set_name):
    out = {}
    for task in MATCHED:
        for high, label in ((True, "strong"), (False, "weak")):
            g = w[(w.task == task) & (w.high == high)]
            out[f"{task}_{label}"] = dict(n_items=int(len(g)), damage=float((g["intact"] - g[set_name]).mean()),
                                          accuracy_intact=float((g["intact"] > 0).mean()),
                                          accuracy_lesioned=float((g[set_name] > 0).mean()))
    return out


def _demand_coef(g, set_name):
    size = (g.task == "size").to_numpy(float)
    x = np.column_stack([np.ones(len(g)), g["intact"], g["high"].astype(float), size, g["intact"].to_numpy() * size])
    return float(np.linalg.lstsq(x, g[set_name].to_numpy(float), rcond=None)[0][2])


def demand_at_equal_intact(w, set_name, n_boot=2000, seed=0, group="concept_group"):
    """Coefficient of the strong-distractor flag at equal intact margin (negative: strong items fall further), with a
    cluster bootstrap over `group`."""
    g = w[w.task.isin(MATCHED)].reset_index()
    est = _demand_coef(g, set_name)
    rng = np.random.default_rng(seed)
    units = g[group].unique()
    rows = {u: g.index[g[group] == u].to_numpy() for u in units}
    draws = [_demand_coef(g.loc[np.concatenate([rows[u] for u in rng.choice(units, len(units))])], set_name)
             for _ in range(n_boot)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return dict(estimate=est, ci_low=float(lo), ci_high=float(hi), n_items=int(len(g)), n_groups=int(len(units)),
                n_boot=n_boot, seed=seed, resampled=group,
                model="lesioned ~ 1 + intact + high + size + intact:size, colour and size items, margins averaged "
                      "over the two option orders")


def naming_frame(scores, test_items, median, intact="intact", lesioned="main", scoring="averaged"):
    """h, rare flag and the answer's token count per test item, in the test table's order."""
    s = scores[scores.scoring == scoring]
    a, b = s[s.set == intact], s[s.set == lesioned]
    if a.empty or b.empty:
        raise ValueError(f"no {scoring} naming scores for {intact} or {lesioned}")
    h = naming.item_damage(a, b)
    ids = test_items.item_id
    if set(h.index) != set(ids):
        raise ValueError("naming scores and test items cover different items")
    if test_items.zipf.isna().any():
        raise ValueError("test items without zipf")
    ntok = b.set_index("item_id").n_tokens.reindex(ids)
    return pd.DataFrame(dict(h=h.reindex(ids).to_numpy(), rare=(test_items.zipf < median).to_numpy(),
                             n_tokens=ntok.to_numpy(float)), index=ids)


def rare_coefficient(frame):
    return p14.rare_coefficient_np(frame.h.to_numpy(), frame.rare.to_numpy(), frame.n_tokens.to_numpy())


def rare_coefficient_ci(frame, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(frame)
    draws = [rare_coefficient(frame.iloc[rng.integers(n, size=n)]) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return dict(estimate=rare_coefficient(frame), ci_low=float(lo), ci_high=float(hi), n_items=n, n_boot=n_boot,
                seed=seed, resampled="item",
                statistic="coefficient of rare in h ~ 1 + rare + standardised token count (P14's objective)")


def random_comparison(scores, test_items, median, target="main", prefix="random_"):
    """The rare coefficient of the target and of every random set, all on the single-composition scores the random
    sets have (as lesion's final test compares them)."""
    sets = sorted(s for s in scores.set.unique() if s.startswith(prefix))
    single = lambda s: rare_coefficient(naming_frame(scores, test_items, median, lesioned=s, scoring="single"))
    target_value = single(target)
    values = np.array([single(s) for s in sets])
    return dict(target_single=target_value, n=int(len(values)), median=float(np.median(values)),
                max=float(values.max()), at_least_target=int((values >= target_value).sum()),
                empirical_p=float((1 + (values >= target_value).sum()) / (1 + len(values))))
