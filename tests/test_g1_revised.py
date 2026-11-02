"""Revised G1 (ADR 0003) on simulated scores: order-averaged high-demand accuracy above chance."""
import pytest

from lesion import g1, g1_revised, scoring
from test_g1 import simulate


@pytest.fixture(scope="module")
def items():
    return scoring.load_materials()[0]


def test_strong_position_bias_fails_frozen_rule_but_passes_revised(items):
    # Content signal plus a large preference for label A: per-prompt answers follow position, averages do not.
    scores = simulate(items, 1.2 / 0.615, intercept=1.6, noise=0.3, bias=2.5)
    res = g1_revised.analyze({"m": scores}, items, n=300)
    v = res["versions"]["m"]
    assert v["frozen_rule_decision"] == "measurement_fail"
    assert v["order_averaged_high_demand_accuracy"]["ci_low"] > 0.5
    assert res["g1_decision"] == "pass" and res["rules"] == g1_revised.RULES


def test_model_drawn_to_distractors_still_fails(items):
    # High-demand items answered wrongly even after averaging the two orders.
    scores = simulate(items, 1.2 / 0.615, intercept=-0.8, noise=0.3, bias=0.0)
    assert g1_revised.analyze({"m": scores}, items, n=300)["g1_decision"] == "measurement_fail"


def test_decision_rule():
    ok, weak = dict(ci_low=0.55), dict(ci_low=0.49)
    assert g1_revised.decision(dict(ci_low=0.3, ci_high=0.9), ok) == "pass"
    assert g1_revised.decision(dict(ci_low=0.1, ci_high=0.9), ok) == "insufficient"
    assert g1_revised.decision(dict(ci_low=0.0, ci_high=0.15), ok) == "exclude"
    assert g1_revised.decision(dict(ci_low=0.3, ci_high=0.9), weak) == "measurement_fail"
