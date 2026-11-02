"""Regression checks for battery orchestration and cohort isolation, without a GPU."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lesion import m3_battery as mb, m3_items

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e3_errors"))
rb = importlib.import_module("run_battery")
ab = importlib.import_module("analyze_battery")


def test_label_winner_need_not_exceed_half_probability(monkeypatch):
    labels = pd.DataFrame([dict(item_id="x", fmt="options_letter", variant=0, correct="A",
                                margin=1 - np.log(5), top_correct=True,
                                **{f"logit_{c}": float(c == "A") for c in "ABCDEF"})])
    cand = pd.DataFrame([dict(item_id="x", logprob=-1.)])
    monkeypatch.setattr(mb, "scored_prompts", lambda *_: [])
    monkeypatch.setattr(mb, "label_items", lambda *_: [])
    monkeypatch.setattr(mb, "candidate_scores", lambda *_: cand.copy())
    monkeypatch.setattr(mb, "label_scores", lambda *_: labels.copy())
    res = rb.run_state(None, None, True, None, None, None, None, None, "test", greedy=False)
    assert res["labels"].top_correct.tolist() == [True]
    assert res["labels"].margin.iloc[0] < 0


@pytest.mark.parametrize("suffix", ["", ".partial"])
def test_existing_results_are_preserved_before_loading(tmp_path, suffix):
    out = tmp_path / "battery"
    existing = out.with_name(out.name + suffix)
    existing.mkdir()
    (existing / "done.parquet").write_bytes(b"paid run checkpoint")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        rb.check_destination(out)
    assert (existing / "done.parquet").read_bytes() == b"paid run checkpoint"


def test_reference_foils_do_not_use_evaluation_concepts():
    all_items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    pool = all_items[all_items.role.isin(mb.POOL_ROLES)]
    items = pool[pool.role.isin(("calibration_a", "calibration_b"))]
    prompts = rb.task_mean_prompts(items, pool, True)
    roles = pool.set_index("item_id").role
    assert all(roles[p.item_id] == "calibration_a" for p in prompts)
    foils = [p.foil_id for p in prompts if isinstance(p, mb.LabelItem) and p.foil_id]
    assert foils and all(roles[i] in ("calibration_a", "reserve") for i in foils)


def test_foil_metadata_marks_category_fallback():
    pool = pd.DataFrame([
        dict(item_id="a", target="one", definition="first description", category="only", domain="shared", order=0),
        dict(item_id="b", target="two", definition="second description", category="pair", domain="shared", order=1),
        dict(item_id="c", target="three", definition="third description", category="pair", domain="shared", order=2)])
    items = mb.word_to_definition_items(pool, pool)
    assert [(p.item_id, p.foil_match) for p in items[::2]] == [("a", "domain"), ("b", "category"), ("c", "category")]


def test_taskmean_evaluation_excludes_reference_items(tmp_path, monkeypatch):
    ids = ["a1", "a2", "b1", "b2", "b3"]
    intact = pd.DataFrame({
        "name_target_logprob": [-1., -2., -3., -4., -5.], "name_p_target": [.9] * 5, "name_S": [-.1] * 5,
        "options_word_target_logprob": [-.1] * 5, "copy_target_logprob": [-.1] * 5,
        "options_letter_margin": [5.] * 5, "word_to_definition_margin": [4.] * 5,
        "options_letter_all_correct": [1.] * 5, "word_to_definition_all_correct": [1.] * 5,
        "delta_none": [5.] * 5, "delta_none_hinted_demos": [5.] * 5,
        "delta_cue": [7.] * 5, "delta_miscue": [1.] * 5}, index=ids)
    lesioned = intact.copy()
    lesioned.loc[["a1", "a2"], "name_target_logprob"] -= 100
    lesioned.loc[["b1", "b2", "b3"], "name_target_logprob"] -= 1
    labels = pd.DataFrame(dict(state=["intact"] * 5, fmt=["word_to_definition"] * 5,
                               foil_match=["category"] * 5, item_id=ids))
    pd.DataFrame({"placeholder": [0]}).to_parquet(tmp_path / "candidates.parquet")
    labels.to_parquet(tmp_path / "labels.parquet")
    pd.concat([pd.DataFrame(dict(item_id=ids, state=s, answer=["word"] * 5,
                                 kind=["other", "other", "target", "target", "target"]))
               for s in ("intact", "K8", "K8_taskmean")], ignore_index=True).to_parquet(tmp_path / "greedy.parquet")
    meta = dict(states=["intact", "K8", "K8_taskmean"], positional=False,
                item_roles={i: "calibration_a" if i.startswith("a") else "calibration_b" for i in ids})
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(ab, "per_concept", lambda *_: {"intact": intact, "K8": intact.copy(), "K8_taskmean": lesioned})
    monkeypatch.setattr(ab, "N_BOOT", 40)
    ab.main(["--dir", str(tmp_path), "--out", str(tmp_path / "report.json")])
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    state = report["states"]["K8_taskmean"]
    assert state["n_concepts"] == 3 and state["cohort"] == "calibration_b"
    assert state["change"]["name_target_logprob"]["mean"] == -1.
    assert state["change"]["name_target_logprob"]["n"] == 3
    comp = state["mean_reference_comparison"]["name_target_logprob"]
    assert comp["task_minus_wiki"]["mean"] == -1. and comp["wiki_change"]["mean"] == 0.
    assert report["greedy_n"]["K8_taskmean"] == 3
    assert report["greedy"]["K8_taskmean"] == {"target": 1.}
    assert report["greedy_reference_comparison"]["K8_taskmean"]["intact"] == {"target": 1.}
