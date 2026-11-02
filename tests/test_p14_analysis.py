"""lesion.p14_analysis: order averaging and label checks, the scaling model (a pure stretch leaves no strong-weak
coefficient, a planted one is recovered), the cluster bootstrap, and the naming statistic (agent2's code review
2026-10-17)."""
import numpy as np
import pandas as pd
import pytest

from lesion import p14_analysis as pa


def m1_scores(lesion, n=40, seed=0):
    """Two option orders per item; lesion(intact margin, high) gives the lesioned margin."""
    rng = np.random.default_rng(seed)
    rows, items = [], []
    for i in range(n):
        task = "colour" if i % 2 else "size"
        high = i % 4 < 2
        intact = rng.normal(6 if not high else 2, 2)
        items.append(dict(item_id=str(i), task=task, high_demand="1" if high else "0", concept_group=f"g{i // 2}"))
        for order, shift in ((0, 0.5), (1, -0.5)):              # the orders differ; their mean is the item margin
            for s, m in (("intact", intact), ("control", lesion(intact, high))):
                rows.append(dict(set=s, item_id=str(i), task=task, order=order, logit_correct=10 + m + shift,
                                 logit_wrong=10.0))
    return pd.DataFrame(rows), pd.DataFrame(items)


def test_order_averaging_and_label_checks():
    scores, items = m1_scores(lambda m, h: m)
    table = pa.item_table(items)
    w = pa.order_averaged(scores, table, ["intact", "control"])
    assert len(w) == 40 and np.allclose(w["intact"], w["control"])
    with pytest.raises(ValueError, match="both option orders"):
        pa.order_averaged(scores[~((scores.item_id == "3") & (scores.order == 1))], table, ["intact", "control"])
    with pytest.raises(ValueError, match="without labels"):
        pa.order_averaged(scores, table.drop(index="5"), ["intact", "control"])
    bad = items.copy()
    bad.loc[2, "concept_group"] = None
    with pytest.raises(ValueError, match="missing labels"):
        pa.item_table(bad)
    bad = items.copy()
    bad.loc[2, "high_demand"] = None                       # allowed in the table, refused for a colour or size item
    with pytest.raises(ValueError, match="high_demand"):
        pa.order_averaged(scores, pa.item_table(bad), ["intact", "control"])
    bad = items.copy()
    bad.loc[2, "high_demand"] = "2"
    with pytest.raises(ValueError, match="high_demand"):
        pa.order_averaged(scores, pa.item_table(bad), ["intact", "control"])
    with pytest.raises(ValueError, match="other items"):
        pa.order_averaged(scores[~((scores.set == "control") & (scores.item_id == "7"))], table, ["intact", "control"])


def test_a_pure_stretch_leaves_no_strong_weak_coefficient_and_a_planted_one_is_recovered():
    for lesion, expected in ((lambda m, h: 2 * m, 0.0), (lambda m, h: 2 * m - 3 * h, -3.0)):
        scores, items = m1_scores(lesion)
        w = pa.order_averaged(scores, pa.item_table(items), ["intact", "control"])
        res = pa.demand_at_equal_intact(w, "control", n_boot=200, seed=1)
        assert abs(res["estimate"] - expected) < 1e-8
        assert res["ci_low"] <= res["estimate"] + 1e-8 and res["ci_high"] >= res["estimate"] - 1e-8
        assert res["n_groups"] == 20 and res["resampled"] == "concept_group"
        if expected == 0.0:
            assert all(abs(v["slope"] - 2) < 1e-8 for v in pa.margin_scaling(w, "control").values())
    scores, items = m1_scores(lambda m, h: m - 3 * h)
    w = pa.order_averaged(scores, pa.item_table(items), ["intact", "control"])
    b = pa.demand_breakdown(w, "control")
    assert b["colour_strong"]["damage"] == pytest.approx(3.0) and b["colour_weak"]["damage"] == pytest.approx(0.0)


def naming_scores(effect, n=80, seed=0):
    rng = np.random.default_rng(seed)
    items = pd.DataFrame(dict(item_id=[f"k{i:03d}" for i in range(n)], zipf=rng.uniform(1, 5, n)))
    ntok = rng.integers(1, 5, n)
    rows = []
    for (iid, z), t in zip(items[["item_id", "zipf"]].itertuples(index=False), ntok):
        base = -2.0 - 0.3 * t
        harm = 0.5 + 0.4 * t + (effect if z < 3 else 0.0)        # token count matters, rarity adds `effect`
        for s, lp in (("intact", base), ("main", base - harm), ("random_000", base - 0.5), ("random_001", base - 0.6)):
            for scoring in ("averaged", "single"):
                rows.append(dict(set=s, item_id=iid, logprob=lp, n_tokens=int(t), scoring=scoring))
    return pd.DataFrame(rows), items


def test_the_naming_statistic_recovers_a_planted_rare_effect_net_of_token_count():
    scores, items = naming_scores(effect=1.5)
    frame = pa.naming_frame(scores, items, median=3.0)
    assert len(frame) == 80
    assert pa.rare_coefficient(frame) == pytest.approx(1.5, abs=1e-6)
    ci = pa.rare_coefficient_ci(frame, n_boot=200, seed=0)
    assert ci["ci_low"] <= 1.5 + 1e-6 <= ci["ci_high"] + 2e-6
    cmp = pa.random_comparison(scores, items, median=3.0)
    assert cmp["n"] == 2 and cmp["at_least_target"] == 0 and cmp["target_single"] == pytest.approx(1.5, abs=1e-6)
    with pytest.raises(ValueError, match="different items"):
        pa.naming_frame(scores, items.iloc[:-1], median=3.0)
    with pytest.raises(ValueError, match="no single"):
        pa.naming_frame(scores[scores.scoring == "averaged"], items, median=3.0, scoring="single")
