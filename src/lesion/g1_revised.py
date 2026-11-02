"""Revised G1 for the switch to Qwen3-30B-A3B (user decision 2026-09-28; docs/adr/0003-qwen-and-revised-g1.md).

Both OLMoE versions failed the frozen G1. The revision was made after seeing those results and the
exploratory runs of larger models, and is reported as such. Changes relative to lesion.g1:

- High-demand accuracy is scored on order-averaged margins: an item counts as correct when the
  correct-minus-wrong logit margin, averaged over the two option orders, is positive. Every later
  analysis (damage h) already averages the two orders, so a constant preference for label A or B
  cancels. The criterion is that its cluster-bootstrap interval lies above chance (lower bound > 0.50);
  the 0.70 point requirement is dropped.
- Order consistency is reported but no longer gates.
- The TDS effect criterion (interval lower bound above 0.20) and every report-only result are unchanged.

lesion.g1 stays frozen for the OLMoE branches (tags g1-frozen, g1-backup-frozen).
"""
from . import g1

ACC_CI_MIN = 0.50
RULES = "revised 2026-09-28 (ADR 0003)"


def order_averaged_high_demand_accuracy(tab, n=g1.N_BOOT):
    fm = tab[tab.task.isin(g1.MATCHING)]
    hd = fm[fm.high_demand == "1"]
    return g1.cluster_bootstrap(hd, lambda t: float((t.d > 0).mean()), n)


def decision(tds_effect, oa_accuracy):
    if not oa_accuracy["ci_low"] > ACC_CI_MIN:
        return "measurement_fail"
    if tds_effect["ci_low"] > g1.THRESHOLD:
        return "pass"
    if tds_effect["ci_high"] < g1.THRESHOLD:
        return "exclude"
    return "insufficient"


def analyze(label_scores_by_version, items, cloze_by_version=None, n=g1.N_BOOT):
    """lesion.g1.analyze plus the revised criterion; g1_decision follows the revised rule."""
    res = g1.analyze(label_scores_by_version, items, cloze_by_version, n)
    for v, s in label_scores_by_version.items():
        tab = g1.item_table(s, items)
        oa = order_averaged_high_demand_accuracy(tab, n)
        vv = res["versions"][v]
        vv["frozen_rule_decision"] = vv["decision_if_chosen"]
        vv["order_averaged_high_demand_accuracy"] = oa
        vv["decision_if_chosen"] = decision(vv["matching"]["tds_effect"], oa)
    res["g1_decision"] = res["versions"][res["chosen_version"]]["decision_if_chosen"]
    res["rules"] = RULES
    return res
