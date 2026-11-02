"""Exploratory re-screens of 2026-10-13 (docs/plans/gpu-session11.md): the knowledge-store candidates and selection,
the M1 statistics they need, and named component sets given on a command line. Pure pandas/numpy, like damage."""
import numpy as np

from . import damage
from .components import Component

STORE_SIZE, STORE_MIN = 8, 4


def zscore(x):
    return (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else x * 0


def store_candidates(naming_attr, m1_attr, top):
    """Union of the top `top` components by naming knowledge effect, by M1 semantic damage, and by the smaller of the
    two standardised damages (naming damage, semantic damage), from the attribution tables of the naming screen
    (knowledge_effect, naming_damage) and the M1 screen (semantic_damage, string_damage)."""
    t = naming_attr[["component", "knowledge_effect", "naming_damage"]].merge(
        m1_attr[["component", "semantic_damage", "string_damage"]], on="component")
    t["both"] = np.minimum(zscore(t.naming_damage), zscore(t.semantic_damage))
    keys = []
    for col in ("knowledge_effect", "semantic_damage", "both"):
        keys += [k for k in t.nlargest(top, col).component if k not in keys]
    return t.set_index("component").loc[keys].reset_index()


def select_store(summary, size=STORE_SIZE, min_size=STORE_MIN):
    """The store set from the real-ablation screen: positive naming knowledge effect, naming damage and M1 semantic
    damage, not domain-general (string damage above semantic damage), ranked by the smaller standardised naming /
    semantic damage; `size` of them ("short" when fewer qualify, none under `min_size`)."""
    s = summary.dropna(subset=["knowledge_effect", "naming_damage", "m1_semantic_damage", "m1_string_damage"])
    ok = s[(s.knowledge_effect > 0) & (s.naming_damage > 0) & (s.m1_semantic_damage > 0)
           & ~(s.m1_string_damage > s.m1_semantic_damage)].copy()
    ok["rank_score"] = np.minimum(zscore(ok.naming_damage), zscore(ok.m1_semantic_damage))
    chosen = ok.nlargest(size, "rank_score").component.tolist()
    status = "ok" if len(chosen) == size else f"short ({len(chosen)} of {size})" if len(chosen) >= min_size else "not_found"
    return dict(store=chosen if len(chosen) >= min_size else [], status=status, n_eligible=int(len(ok)),
                rule="positive naming knowledge effect, naming damage and M1 semantic damage; not domain-general "
                     "(string damage above semantic damage); ranked by the smaller standardised naming / semantic damage")


def m1_effects_with_synonym(cols):
    """lesion.damage.effects on M1 plus the damage on the synonym items alone, every name prefixed m1_ (so the
    names do not clash with the naming test's)."""
    syn = cols["task"] == "synonym"
    e = {**damage.effects(cols), "synonym_damage": float(cols["h"][syn].mean()) if syn.any() else np.nan}
    return {f"m1_{k}": v for k, v in e.items()}


def parse_named_sets(specs):
    """["NAME=key,key", ...] -> {NAME: [Component, ...]}; names unique and not "intact"."""
    sets = {}
    for spec in specs:
        name, _, keys = spec.partition("=")
        if not name or not keys or name == "intact" or name in sets:
            raise ValueError(f"bad set {spec!r}: NAME=key,key,... with a new name other than intact")
        sets[name] = [Component.parse(k) for k in keys.split(",")]
    return sets


# --- second knowledge set K8b (GPU session 12, item 5; docs/plans/gpu-session12.md) -----------------------------------

K8B_SIZE, K8B_TOP = 8, 40


def k8b_candidates(naming_attr, k8, top=K8B_TOP):
    """The top `top` components of the naming attribution table by knowledge effect, without K8's (K8 itself was
    chosen from the top 32 of the same table)."""
    ranked = naming_attr.sort_values("knowledge_effect", ascending=False).component.head(top)
    return [k for k in ranked if k not in set(k8)]


def null_singles(candidates, pool, n, seed=0, exclude=()):
    """n random single components, each drawn like a candidate picked at random: same kind and layer, never a
    candidate or excluded, without repeats. candidates and pool: Component lists."""
    rng = np.random.default_rng(seed)
    banned = set(candidates) | set(exclude)
    by_place = {}
    for c in pool:
        if c not in banned:
            by_place.setdefault((c.kind, c.layer), []).append(c)
    chosen, tries = [], 0
    while len(chosen) < n:
        tries += 1
        if tries > 100 * n:
            raise ValueError(f"only {len(chosen)} of {n} null components could be drawn")
        target = candidates[rng.integers(len(candidates))]
        free = [c for c in by_place.get((target.kind, target.layer), []) if c not in chosen]
        if free:
            chosen.append(free[rng.integers(len(free))])
    return chosen


def select_k8b(summary, null, size=K8B_SIZE, min_size=4, q=0.99):
    """K8b: K8's rule (positive real knowledge effect and naming damage, the largest knowledge effects) among the
    candidates whose language-model loss change is not above the random single components' 99th percentile and whose
    M1 margin-scaling slope lies between their 1st and 99th percentiles. summary: one row per candidate with
    knowledge_effect, naming_damage, lm_loss_change, m1_slope; null: the same two columns for the random singles."""
    lm_cut = float(np.quantile(null.lm_loss_change, q))
    lo, hi = float(np.quantile(null.m1_slope, 1 - q)), float(np.quantile(null.m1_slope, q))
    s = summary.copy()
    s["excluded_lm_loss"] = s.lm_loss_change > lm_cut
    s["excluded_scaling"] = (s.m1_slope < lo) | (s.m1_slope > hi)
    ok = s[(s.knowledge_effect > 0) & (s.naming_damage > 0) & ~s.excluded_lm_loss & ~s.excluded_scaling]
    chosen = ok.nlargest(size, "knowledge_effect").component.tolist()
    status = "ok" if len(chosen) == size else f"short ({len(chosen)} of {size})" if len(chosen) >= min_size else "not_found"
    return dict(k8b=chosen if len(chosen) >= min_size else [], status=status, n_eligible=int(len(ok)),
                lm_loss_cut=lm_cut, slope_range=[lo, hi],
                excluded_lm_loss=s.loc[s.excluded_lm_loss, "component"].tolist(),
                excluded_scaling=s.loc[s.excluded_scaling, "component"].tolist(),
                quantile=q,
                rule="K8's rule (positive real knowledge effect and naming damage; the largest knowledge effects) among "
                     "candidates whose LM loss change is not above the random single components' 99th percentile and "
                     "whose M1 margin-scaling slope lies between their 1st and 99th percentiles")


def super_expert_flags(expert_mean, factor=20.0):
    """{(layer, expert): ratio} for experts whose mean-output norm is more than `factor` times their layer's
    median (a label, not an exclusion). expert_mean: (layers, experts, hidden) tensor."""
    norms = expert_mean.float().norm(dim=-1).numpy()
    med = np.median(norms, axis=1, keepdims=True)
    ratio = norms / np.where(med > 0, med, np.nan)
    return {(int(l), int(e)): float(ratio[l, e]) for l, e in zip(*np.nonzero(ratio > factor))}


ANCHOR = 30           # P13: K30, the knowledge screen's positive components, as in the session-9 dose probe


def breadth_sets(screen, attribution, exclude=(), sizes=(64, 128, 256), anchor=ANCHOR):
    """P13 (docs/plans/p13-knowledge-breadth.md): K8, K30 and nested larger knowledge sets.

    screen: the knowledge screen (component, knowledge_effect; the components that were really ablated); K30 is its
    components with a positive effect, largest first, at most `anchor` of them, and K8 their first 8. Larger sets add
    attribution candidates (component, knowledge_effect, naming_damage) that are not in K30 or `exclude` and have both
    attributed effects positive, ranked by the smaller of the two z-scores (ties by name), so S64 contains K30 and
    S128 contains S64. Returns ({name: [component keys]}, ranked candidates as a DataFrame with their score)."""
    import pandas as pd

    k = screen[screen.knowledge_effect > 0].sort_values(["knowledge_effect", "component"], ascending=[False, True])
    k30 = k.component.tolist()[:anchor]
    cand = attribution[(attribution.knowledge_effect > 0) & (attribution.naming_damage > 0)
                       & ~attribution.component.isin(set(k30) | set(exclude))].copy()
    cand["score"] = np.minimum(zscore(cand.knowledge_effect.to_numpy()), zscore(cand.naming_damage.to_numpy()))
    cand = cand.sort_values(["score", "component"], ascending=[False, True]).reset_index(drop=True)
    sets = {"K8": k30[:8], f"K{len(k30)}": k30}
    for n in sizes:
        if n <= len(k30) or n - len(k30) > len(cand):
            raise ValueError(f"cannot build a set of {n} from {len(k30)} anchors and {len(cand)} candidates")
        sets[f"S{n}"] = k30 + cand.component.tolist()[:n - len(k30)]
    return sets, pd.DataFrame(cand)
