"""Damage h and the control/knowledge effects (lesion.damage), on M1 items with fake scores."""
import numpy as np
import pandas as pd
import pytest

import fakes
from lesion import damage, materials, scoring


@pytest.fixture(scope="module")
def items():
    return materials.load_split("screening")


@pytest.fixture(scope="module")
def freq_median():
    return damage.frequency_median(scoring.load_materials()[0])


def test_frequency_median_splits_synonyms_evenly(freq_median):
    all_items = scoring.load_materials()[0]
    f = pd.to_numeric(all_items[all_items.task == "synonym"].min_freq)
    assert (f < freq_median).sum() == (f > freq_median).sum() == 80


def test_damage_sign_and_order_check(items):
    intact = fakes.label_scores(items, lambda r: 2.0)
    lesioned = fakes.label_scores(items, lambda r: 0.5)
    h = damage.item_damage(intact, lesioned)
    assert np.allclose(h, 1.5)
    with pytest.raises(ValueError):
        damage.item_damage(intact, lesioned[lesioned.order == 0])


def test_effects_recover_planted_damage(items, freq_median):
    intact = fakes.label_scores(items, lambda r: 2.0)
    rare = lambda r: r["task"] == "synonym" and float(r["min_freq"]) < freq_median
    control = fakes.label_scores(items, lambda r: 2.0 - (1.0 if r["high_demand"] == "1" else 0.0))
    knowledge = fakes.label_scores(items, lambda r: 2.0 - (0.6 if rare(r) else 0.0))
    ec = damage.effects(damage.columns(damage.item_frame(damage.item_damage(intact, control), items, freq_median)))
    ek = damage.effects(damage.columns(damage.item_frame(damage.item_damage(intact, knowledge), items, freq_median)))
    assert ec["control_effect"] == pytest.approx(1.0) and ec["knowledge_effect"] == pytest.approx(0.0)
    assert ek["knowledge_effect"] == pytest.approx(0.6) and ek["control_effect"] == pytest.approx(0.0)
    assert ec["control_slope"] > 0 and ek["knowledge_slope"] > 0
    assert ec["string_damage"] == pytest.approx(0.0)


def test_bootstrap_interval_and_missing_cells(items, freq_median):
    rng = np.random.default_rng(1)
    noise = {i: rng.normal() for i in items.item_id}
    intact = fakes.label_scores(items, lambda r: 2.0)
    lesioned = fakes.label_scores(items, lambda r: 2.0 - noise[r["item_id"]])
    frame = damage.item_frame(damage.item_damage(intact, lesioned), items, freq_median)
    ci = damage.bootstrap(frame, damage.effects, n=300)
    for v in ci.values():
        assert v["ci_low"] <= v["estimate"] <= v["ci_high"] and v["n_valid"] > 0
    # Only two concept groups: some resamples miss a condition, they are dropped instead of making the interval nan.
    tiny = frame[frame.concept_group.isin(frame[frame.task == "colour"].concept_group.unique()[:2])]
    out = damage.bootstrap(tiny, damage.effects, n=200)
    assert out["knowledge_effect"]["n_valid"] == 0 and np.isnan(out["knowledge_effect"]["ci_low"])


def test_demand_breakdown_splits_matching_tasks_by_demand(items):
    intact = fakes.label_scores(items, lambda r: 2.0)
    lesioned = fakes.label_scores(items, lambda r: -1.0 if r["high_demand"] == "1" else 3.0)
    t = damage.demand_breakdown(intact, lesioned, items).set_index(["task", "demand"])
    assert set(t.index) == {("colour", "high"), ("colour", "low"), ("size", "high"), ("size", "low"),
                            ("synonym", "all"), ("string", "all")}
    assert t.n.sum() == items.item_id.nunique()
    assert t.loc[("colour", "high"), "h"] == pytest.approx(3.0) and t.loc[("size", "low"), "h"] == pytest.approx(-1.0)
    assert t.loc[("colour", "high"), "lesioned_correct"] == 0.0 and t.loc[("colour", "high"), "intact_correct"] == 1.0
    with pytest.raises(ValueError):
        damage.demand_breakdown(intact, lesioned[lesioned.item_id != lesioned.item_id.iloc[0]], items)


def test_margin_scaling_finds_a_shrink_by_a_common_factor(items):
    size = {i: 1.0 + k % 7 for k, i in enumerate(items.item_id)}
    intact = fakes.label_scores(items, lambda r: size[r["item_id"]])
    lesioned = fakes.label_scores(items, lambda r: 0.7 * size[r["item_id"]])
    t = damage.margin_scaling(intact, lesioned, items).set_index("task")
    assert set(t.index) == set(items.task) | {"all"}
    assert t.n["all"] == items.item_id.nunique()
    assert t.slope.to_numpy() == pytest.approx(0.7) and t.intercept.to_numpy() == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        damage.margin_scaling(intact, lesioned[lesioned.item_id != lesioned.item_id.iloc[0]], items)
