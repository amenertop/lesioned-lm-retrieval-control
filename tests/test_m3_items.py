"""M3 item construction (lesion.m3_items): curation checks, definition rules, assignment, and the menus file."""
import numpy as np
import pandas as pd
import pytest

from lesion import knowledge_items as ki, m3_items as m3

MENUS = m3.OUT / "menus.csv"


def menu_row(**kw):
    row = dict(order=0, target="camel", zipf=3.5, zipf_bin=1, domain="organism", category="animal",
               wordnet_gloss="cud-chewing mammal", synset="camel.n.01", synonyms="",
               coordinate_menu="llama (0.3); horse (0.9)", associate_menu="desert (0.21); hump (0.1)",
               superordinate_menu="animal; mammal")
    row.update(kw)
    return row


def curation_row(**kw):
    row = dict(target="camel", decision="keep", coordinate="llama", superordinate="animal", associate="desert",
               definition="a large hoofed mammal with one or two humps that carries loads across dry lands",
               note="", avoid_unrelated="")
    row.update(kw)
    return row


def test_definition_rules():
    ok = "a large hoofed mammal with one or two humps that carries loads across dry lands"
    assert m3.definition_problems(ok, ["camel", "llama", "animal", "desert"]) == []
    assert any("words" in p for p in m3.definition_problems("a humped beast", ["camel"]))
    assert m3.definition_problems(ok + " of the desert", ["desert"]) == ["definition contains 'desert'"]
    assert m3.definition_problems("", ["camel"]) == ["no definition"]


def test_check_curation_flags_each_rule():
    menu = pd.DataFrame([menu_row()])
    assert m3.check_curation(pd.DataFrame([curation_row()]), menu) == []
    cases = {"not in its menu": curation_row(coordinate="zebra"),
             "options repeat": curation_row(coordinate="llama", superordinate="llama"),
             "needs a reason": curation_row(decision="drop", note=""),
             "definition contains": curation_row(definition="a large hoofed animal with humps that carries loads across dry lands"),
             "not in the menus": curation_row(target="zebra")}
    for message, row in cases.items():
        problems = m3.check_curation(pd.DataFrame([row]), menu)
        assert any(message in p for p in problems), (message, problems)
    assert m3.check_curation(pd.DataFrame([curation_row(decision="drop", note="rare sense")]), menu) == []
    twice = pd.DataFrame([curation_row(), curation_row()])
    assert any("twice" in p for p in m3.check_curation(twice, menu))


def test_assign_takes_calibration_b_by_direction_then_fills_bins_in_order():
    kept = pd.DataFrame(dict(order=range(12), zipf_bin=[0] * 5 + [1] * 7, target=[f"w{i}" for i in range(12)],
                             free_naming_ok=[True] * 12, target_more_common=[True, False] * 6))
    items, short = m3.assign(kept, n_test=4, n_cal_a=2, n_b=2)
    assert short == {}
    b = items[items.role == "calibration_b"]
    assert sorted(b.order) == [0, 1, 2, 3] and b.target_more_common.sum() == 2      # first two of each direction
    assert items.role.value_counts().to_dict() == {"test": 4, "calibration_b": 4, "calibration_a": 2, "reserve": 2}
    for _, g in items[items.role != "calibration_b"].groupby("zipf_bin"):         # test, then a, then reserves
        order = {"test": 0, "calibration_a": 1, "reserve": 2}
        assert g.sort_values("order").role.map(order).is_monotonic_increasing
    _, short = m3.assign(kept.assign(target_more_common=False), n_test=4, n_cal_a=2, n_b=2)
    assert short == {"calibration_b, target more common": 2}


def test_quotas_sum_and_follow_sizes():
    q = m3.quotas(pd.Series({0: 150, 1: 245, 2: 81}), 160)
    assert q.sum() == 160 and q[1] > q[0] > q[2]


def test_free_naming_needs_equal_tokens_and_different_onsets():
    class Tok:
        def encode(self, w, add_special_tokens=False):
            return list(range(len(w) // 4 + 1))

    items = pd.DataFrame(dict(target=["camel", "kettle", "sock", "pea", "binoculars"],
                              associate=["desert", "ketchup", "shoe", "pod", "eyes"]))
    # tokens here are len // 4 + 1: binoculars 3, eyes 2 (gap 1 allowed); kettle/ketchup share "ke"
    assert m3.free_naming_ok(items, Tok()).tolist() == [True, False, True, True, True]
    items.loc[4, "associate"] = "a"
    assert not m3.free_naming_ok(items, Tok())[4]


def test_rhymes_and_overlaps():
    assert m3.rhymes("footstool", "toadstool") and not m3.rhymes("camel", "desert")
    assert m3.overlaps("bee", ["bumblebee"]) and not m3.overlaps("bee", ["wasp"])


@pytest.mark.skipif(not ki.NLTK_DIR.exists(), reason="WordNet not downloaded")
def test_labels_resolve_and_categories_prefer_earlier_entries():
    wn = ki.wordnet()
    table = m3.label_table(wn)
    assert len(table) == len(m3.LABELS)
    assert m3.labels_above(wn.synset("wagon.n.01"), table)[0][2] == "vehicle"     # not "container"
    assert m3.category_key(wn.synset("sparrow.n.01"), table) == "bird.n.01"


@pytest.mark.skipif(not MENUS.exists(), reason="menus not built")
def test_menus_file():
    t = pd.read_csv(MENUS, keep_default_na=False, na_values=[""])
    assert t.order.is_unique and t.target.is_unique and t.synset.is_unique
    assert not set(t.target) & m3.excluded_words()
    assert not set(t.target) & m3.LABEL_WORDS
    for r in t.itertuples():
        for col in ("coordinate_menu", "associate_menu", "superordinate_menu"):
            assert r.target not in m3.menu_words(getattr(r, col))
    assert (np.diff(t.zipf_bin.to_numpy()) >= 0).all()
