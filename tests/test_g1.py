"""Checks the G1 analysis (ticket pilot 05) on simulated scores with a known TDS effect (pilot protocol §8 step 2)."""
import numpy as np
import pandas as pd
import pytest

from lesion import g1, scoring


@pytest.fixture(scope="module")
def items():
    return scoring.load_materials()[0]


def simulate(items, slope, intercept=2.5, noise=0.4, bias=0.3, string_consistent=True, seed=1):
    """Fake label scores: logit(correct) - logit(wrong) = intercept + slope*TDS + noise, plus a position bias for A."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in items.itertuples():
        tds = float(r.tds) if r.tds not in ("", "nan") else 0.0
        base_diff = intercept + (slope * tds if r.task in g1.MATCHING else 0.0) + rng.normal(0, noise)
        if r.task == "string" and not string_consistent:
            base_diff = rng.normal(0, 0.5)
        for order in (0, 1):
            option_a = "option1" if order == 0 else "option2"
            correct_label = "A" if option_a == r.correct_option else "B"
            diff = base_diff + (bias if correct_label == "A" else -bias)
            lc, lw = diff, 0.0
            la, lb = (lc, lw) if correct_label == "A" else (lw, lc)
            choice = "A" if la > lb else "B"
            chosen = option_a if choice == "A" else ("option2" if option_a == "option1" else "option1")
            rows.append(dict(item_id=r.item_id, task=r.task, order=order, correct_label=correct_label,
                             logit_A=la, logit_B=lb, logit_correct=lc, logit_wrong=lw, choice=choice,
                             chosen_option=chosen, option_A=option_a))
    return pd.DataFrame(rows)


def test_gap_matches_protocol(items):
    assert abs(g1.tds_gap(items) - 0.615) < 0.001


def test_recovers_known_effect(items):
    slope = 0.5 / 0.615          # true E = 0.5
    tab = g1.item_table(simulate(items, slope), items)
    m = g1.matching_measures(tab, g1.tds_gap(items), n=500)
    assert abs(m["tds_effect"]["estimate"] - 0.5) < 0.15
    assert m["tds_effect"]["ci_low"] < m["tds_effect"]["estimate"] < m["tds_effect"]["ci_high"]


def test_decisions(items):
    gap = g1.tds_gap(items)
    strong = g1.matching_measures(g1.item_table(simulate(items, 1.2 / 0.615, noise=0.3), items), gap, n=500)
    assert g1.g1_decision(strong) == "pass"
    null = g1.matching_measures(g1.item_table(simulate(items, 0.0, noise=0.3), items), gap, n=500)
    assert g1.g1_decision(null) in ("exclude", "insufficient")
    weak_model = g1.matching_measures(g1.item_table(simulate(items, 1.2 / 0.615, intercept=-0.5), items), gap, n=500)
    assert g1.g1_decision(weak_model) == "measurement_fail"


def test_version_choice_uses_string_items_only(items):
    good = g1.item_table(simulate(items, 0.0), items)
    bad_format = g1.item_table(simulate(items, 2.0, string_consistent=False), items)
    chosen, ok = g1.select_version({"olmoe-0924-base": g1.string_measures(bad_format),
                                    "olmoe-0924-instruct": g1.string_measures(good)})
    assert chosen == "olmoe-0924-instruct" and ok


def test_version_tie_goes_to_base():
    same = dict(accuracy=0.95, order_consistency=0.97)
    chosen, ok = g1.select_version({"olmoe-0924-instruct": same, "olmoe-0924-base": dict(same)})
    assert chosen == "olmoe-0924-base" and ok


def test_full_analysis_runs(items):
    res = g1.analyze({"olmoe-0924-base": simulate(items, 1.0), "olmoe-0924-instruct": simulate(items, 1.0, seed=2)},
                     items, n=200)
    assert res["g1_decision"] in ("pass", "insufficient", "exclude", "measurement_fail")
    v = res["versions"][res["chosen_version"]]
    assert set(v["human_agreement"]) == {"colour", "size", "synonym", "string"}
    assert "estimate" in v["synonym_frequency"]
