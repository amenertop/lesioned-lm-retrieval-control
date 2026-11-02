"""Exploratory re-screens of 2026-10-13 (docs/plans/gpu-session11.md): the knowledge-store selection (lesion.rescreen),
the semantic-only control selection (lesion.screen) and the letter-cued naming (lesion.m3_cued)."""
import numpy as np
import pandas as pd
import pytest
import torch

from lesion import m3_cued, rescreen, screen
from lesion.components import Component


def _summary(rows):
    cols = ["component", "knowledge_effect", "naming_damage", "m1_semantic_damage", "m1_string_damage"]
    return pd.DataFrame(rows, columns=cols)


def test_store_needs_every_damage_and_not_domain_general():
    s = _summary([
        ("a", 0.5, 0.5, 0.5, 0.0),      # eligible
        ("b", 0.5, 0.5, 0.5, 0.9),      # string damage above semantic damage: domain-general
        ("c", -0.1, 0.5, 0.5, 0.0),     # no frequency grading
        ("d", 0.5, -0.1, 0.5, 0.0),     # no naming damage
        ("e", 0.5, 0.5, -0.1, 0.0),     # no comprehension damage
        ("f", 0.2, 0.3, 0.2, 0.1),      # eligible
        ("g", 0.1, 0.9, 0.9, -0.2),     # eligible, largest damages
        ("h", 0.3, 0.2, 0.6, 0.0),      # eligible
    ])
    r = rescreen.select_store(s, size=4, min_size=2)
    assert set(r["store"]) == {"a", "f", "g", "h"} and r["status"] == "ok" and r["n_eligible"] == 4
    assert r["store"][0] == "g"                                  # ranked by the smaller standardised damage
    short = rescreen.select_store(s, size=6, min_size=2)
    assert short["status"] == "short (4 of 6)" and len(short["store"]) == 4
    assert rescreen.select_store(s, size=6, min_size=5)["store"] == []


def test_store_candidates_union_without_duplicates():
    naming = pd.DataFrame(dict(component=list("abcdef"), knowledge_effect=[6, 5, 4, 3, 2, 1], naming_damage=[1, 2, 3, 4, 5, 6]))
    m1 = pd.DataFrame(dict(component=list("abcdef"), semantic_damage=[1, 1, 1, 9, 8, 1], string_damage=[0] * 6))
    keys = rescreen.store_candidates(naming, m1, top=2).component.tolist()
    assert keys[:2] == ["a", "b"] and {"d", "e"} <= set(keys) and len(keys) == len(set(keys))


def test_semantic_control_rule():
    s = pd.DataFrame(dict(component=list("abcdefgh"),
                          control_effect=[2.0, 1.8, 1.5, 1.2, 1.1, 0.1, 0.0, -0.1],
                          knowledge_effect=[0.0, 0.1, 0.0, 0.1, 0.0, 0.9, 1.0, 0.8],
                          string_damage=[-0.1, 0.4, -0.3, 0.0, -0.2, 0.0, 0.0, 0.0]))
    r = screen.select_semantic_control(s, k=3)
    assert r["control"] == ["a", "c", "d"] and r["status"] == "ok"     # b hurts strings; f-h point to knowledge
    assert screen.select_semantic_control(s, k=5)["control"] == [] and "not_found" in screen.select_semantic_control(s, k=5)["status"]


def test_cued_prompts_cover_every_hint_length():
    items = pd.DataFrame(dict(item_id=["m1", "m2"], target=["zebra", "ant"], definition=["a striped horse", "a small insect"]))
    p = m3_cued.prompts(items, chat=True)
    assert [x.condition for x in p] == ["cue0", "cue1", "cue2", "cue3", "cue4", "cue0", "cue1", "cue2"]
    assert 'starts with "zebr"' in p[4].text and "Hint" not in p[0].text.split("\n\n")[-1]
    assert p[4].text.split("\n\n")[-1].endswith("Answer:")


def test_cued_naming_accuracy_benefits_and_letters_needed():
    rows = []
    # zebra (5 letters): right only from 3 letters on; ant (3 letters): right without a hint
    for j in range(5):
        rows.append(dict(state="intact", item_id="m1", letters=j, n_letters=5, kind="target" if j >= 3 else "other"))
    for j in range(3):
        rows.append(dict(state="intact", item_id="m2", letters=j, n_letters=3, kind="target"))
    r = m3_cued.summarize(pd.DataFrame(rows), n_boot=200)["intact"]
    assert r["accuracy_none"]["estimate"] == 0.5 and r["accuracy_two"]["estimate"] == 0.5 and r["accuracy_final"]["estimate"] == 1.0
    assert r["benefit_final"]["estimate"] == 0.5 and r["benefit_two"]["estimate"] == 0.0
    assert r["share_of_letters_needed"] == dict(estimate=0.6, n=1)       # 3 of zebra's 5 letters; ant needed none


def test_named_sets_parse_and_refuse_bad_names():
    sets = rescreen.parse_named_sets(["S8=expert:1:2,head:0:3", "C4s=head:9:15"])
    assert [c.key for c in sets["S8"]] == ["expert:1:2", "head:0:3"] and list(sets) == ["S8", "C4s"]
    for bad in (["intact=head:0:1"], ["A=head:0:1", "A=head:0:2"], ["A="], ["=head:0:1"]):
        with pytest.raises(ValueError):
            rescreen.parse_named_sets(bad)


def test_k8b_candidates_drop_k8_and_keep_rank_order():
    attr = pd.DataFrame(dict(component=["a", "b", "c", "d", "e"], knowledge_effect=[5, 4, 3, 2, 1]))
    assert rescreen.k8b_candidates(attr, ["b", "d"], top=4) == ["a", "c"]


def test_null_singles_match_kind_and_layer_without_candidates():
    pool = [Component("head", l, h) for l in range(3) for h in range(6)] + [Component("expert", 1, e) for e in range(4)]
    cands = [Component("head", 2, 0), Component("expert", 1, 3)]
    null = rescreen.null_singles(cands, pool, 6, seed=1, exclude=[Component("head", 2, 1)])
    assert len(set(null)) == 6
    assert all((c.kind, c.layer) in {("head", 2), ("expert", 1)} for c in null)
    assert not set(null) & {Component("head", 2, 0), Component("expert", 1, 3), Component("head", 2, 1)}
    with pytest.raises(ValueError):
        rescreen.null_singles(cands, pool, 50, seed=1)


def test_select_k8b_applies_the_null_bounds_then_k8s_rule():
    summary = pd.DataFrame(dict(component=list("abcdefghij"),
                                knowledge_effect=[9, 8, 7, 6, 5, 4, 3, 2, 1, -1],
                                naming_damage=[1] * 10,
                                lm_loss_change=[0.5, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                                m1_slope=[1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    null = pd.DataFrame(dict(lm_loss_change=np.linspace(0, 0.004, 101), m1_slope=np.linspace(0.95, 1.05, 101)))
    sel = rescreen.select_k8b(summary, null, size=6)
    assert sel["lm_loss_cut"] == pytest.approx(0.00396) and sel["slope_range"] == pytest.approx([0.951, 1.049])
    assert sel["k8b"] == list("cdefgh") and sel["status"] == "ok"
    assert sel["excluded_lm_loss"] == ["a"] and sel["excluded_scaling"] == ["b"]
    assert rescreen.select_k8b(summary, null, size=8)["status"] == "short (7 of 8)"
    # the percentile, not the extreme: one outlying random component does not move the bound
    outlier = null.copy()
    outlier.loc[0, "lm_loss_change"] = 0.9
    assert rescreen.select_k8b(summary, outlier, size=6)["excluded_lm_loss"] == ["a"]


def test_super_expert_flags_report_outliers_only():
    mean = torch.ones(2, 5, 4) * 0.1
    mean[1, 3] = 5.0
    flags = rescreen.super_expert_flags(mean, factor=20)
    assert list(flags) == [(1, 3)] and flags[(1, 3)] == pytest.approx(50.0)


def test_breadth_sets_are_nested_and_keep_the_anchor():
    import pandas as pd

    from lesion import rescreen

    screen = pd.DataFrame(dict(component=[f"e{i}" for i in range(6)], knowledge_effect=[.5, .4, -.1, .3, .2, .1]))
    attr = pd.DataFrame(dict(component=[f"a{i}" for i in range(10)] + ["e0", "bad", "x"],
                             knowledge_effect=[1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 2, 5, -1],
                             naming_damage=[.1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 2, 5, 3]))
    sets, ranked = rescreen.breadth_sets(screen, attr, exclude=["bad"], sizes=(7, 9), anchor=4)
    assert sets["K8"] == ["e0", "e1", "e3", "e4"] == sets["K4"]            # positive only, largest first
    assert sets["S7"][:4] == sets["K4"] and sets["S9"][:7] == sets["S7"]   # nested
    assert not {"bad", "x", "e0"} & set(ranked.component)                  # excluded, a negative effect, an anchor
    assert ranked.component.iloc[0] == "a1"          # the smaller z-score ranks: a0 has a tiny naming damage
    with pytest.raises(ValueError):
        rescreen.breadth_sets(screen, attr, sizes=(40,), anchor=4)
