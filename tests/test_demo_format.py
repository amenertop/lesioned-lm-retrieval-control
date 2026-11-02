"""Demonstration-format check (docs/plans/demo-format-check.md): the local prompt builders against the frozen ones,
the run script's scoring on a tiny random model and the analysis with its reading rules."""
import functools
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lesion import fine_semantics as fs, m3_items, m3_scoring as ms, models, runinfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))
import analyze_demo_format as adf  # noqa: E402
import demo_format_prompts as dfp  # noqa: E402
import run_demo_format as rdf  # noqa: E402

QWEN = "qwen3-30b-a3b-instruct"
BATTERY = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
FAMILY = pd.read_csv(ROOT / "data/materials/title_gap/fine_final/items.csv", keep_default_na=False)
# SHA-256 of json.dumps([vars(p) for p in prompts]) of the frozen builders on all rows of both item files, 2026-10-28
FROZEN_PROMPTS_SHA256 = {
    "battery_default_chat": "764aee9d81ee08ce1141d5ed753a8f1ba087f8236bf089aedfef16419b94aac8",
    "battery_default_plain": "c743d5ad1c19cc6f543256efe71bf1d4fe7ce6fe8335fc32fd5ebf62df40eff7",
    "battery_none_error_roles": "7bdd5ec603e07d5e50387270deb0623055faf09020e9a44d5b650a40cb5df14a",
    "family_naming": "7bc39f59077b25d6e1f62890dca49b3c20aa96d9eebc86ec8de0c04ef71e610e",
}


def digest(prompts):
    return hashlib.sha256(json.dumps([vars(p) for p in prompts], ensure_ascii=False).encode()).hexdigest()


def same(a, b):
    return [vars(p) for p in a] == [vars(p) for p in b]


def test_frozen_prompt_modules_keep_the_hashes_the_protocols_record():
    pinned = json.loads((ROOT / "docs/analysis-plan/title-gap-fine.json").read_text(encoding="utf-8"))["sources_sha256"]
    for path in ("src/lesion/m3_scoring.py", "src/lesion/fine_semantics.py"):
        assert runinfo.sha256(ROOT / path) == pinned[path]


def test_local_builders_reproduce_the_frozen_builders_exactly():
    frozen = {
        "battery_default_chat": ms.build_naming_prompts(BATTERY),
        "battery_default_plain": ms.build_naming_prompts(BATTERY, chat=False),
        "battery_none_error_roles": ms.build_naming_prompts(BATTERY, ("none",), True, roles=ms.ERROR_ROLES),
        "family_naming": fs.naming_prompts(FAMILY),
    }
    assert {k: digest(v) for k, v in frozen.items()} == FROZEN_PROMPTS_SHA256
    local = {
        "battery_default_chat": dfp.battery_prompts(BATTERY),
        "battery_default_plain": dfp.battery_prompts(BATTERY, chat=False),
        "battery_none_error_roles": dfp.battery_prompts(BATTERY, ("none",), True, ms.ERROR_ROLES, demos=True),
        "family_naming": dfp.family_prompts(FAMILY, demos=False),
    }
    for key in frozen:
        assert same(local[key], frozen[key]), key
    dfp.check_against_frozen(BATTERY[BATTERY.role == "test"], FAMILY)


@pytest.mark.parametrize("chat", [True, False])
def test_battery_without_demos_is_the_frozen_prompt_minus_the_demo_text(chat):
    items = BATTERY.head(12)
    for cond in ms.CONDITIONS:
        text = dfp.demo_text(cond)
        for a, b in zip(ms.build_naming_prompts(items, (cond,), chat), dfp.battery_prompts(items, (cond,), chat, demos=False)):
            assert a.text.count(text) == 1 and text not in b.text
            assert a.text.replace(text, "", 1) == b.text
            assert "Answer: clock" not in b.text and b.text.startswith(ms.NAMING_INSTRUCTION)
            assert (a.item_id, a.condition, a.candidates) == (b.item_id, b.condition, b.candidates)


def test_families_with_demos_insert_the_same_demo_text_and_nothing_else():
    items = FAMILY[(FAMILY.split == "test") & (FAMILY.review == "approved")].head(12)
    text = dfp.demo_text()
    assert text in ms.build_naming_prompts(BATTERY.head(1), ("none",))[0].text     # the battery's own demo text
    for a, b in zip(fs.naming_prompts(items), dfp.family_prompts(items, demos=True)):
        assert b.text.count(text) == 1 and b.text.replace(text, "", 1) == a.text
        assert b.text.startswith(dfp.FAMILY_INSTRUCTION) and a.text.startswith(dfp.FAMILY_INSTRUCTION)
        assert b.text.endswith(a.text.split("\n", 1)[1])          # the description and answer cue come last
        assert (b.item_id, b.candidates, b.condition) == (a.item_id, a.candidates, "naming_demos")


def test_run_script_cells_sets_and_items():
    battery, family = rdf.battery_items(), rdf.family_items()
    assert (len(battery), len(family)) == (rdf.N_BATTERY, rdf.N_FAMILY)
    sets = rdf.component_sets(QWEN)
    assert sets["SC4"] == ("head:9:15", "head:27:9", "head:25:24", "head:24:5") and len(sets["WR8"]) == 8
    cells = rdf.build_cells(rdf.NEW_CELLS + rdf.EXISTING_CELLS, battery, family)
    assert {k: len(v) for k, v in cells.items()} == {("battery", "no_demos"): 160, ("family", "demos"): 120,
                                                     ("battery", "demos"): 160, ("family", "no_demos"): 120}
    assert all(len(p.candidates) == 5 for p in cells[("battery", "no_demos")])
    assert all(set(p.candidates) == {"target", "synonym", "coordinate"} for p in cells[("family", "demos")])
    # the existing cells are the frozen builders' prompts, the new ones differ only by the demo text
    assert [p.text for p in cells[("battery", "demos")]] == \
        [p.text for p in ms.build_naming_prompts(battery, ("none",), True, roles=ms.ERROR_ROLES)]
    assert [p.text for p in cells[("family", "no_demos")]] == [p.text for p in fs.naming_prompts(family)]
    assert [p.text.replace(dfp.demo_text(), "", 1) for p in cells[("battery", "demos")]] == \
        [p.text for p in cells[("battery", "no_demos")]]
    assert [p.text.replace(dfp.demo_text(), "", 1) for p in cells[("family", "demos")]] == \
        [p.text for p in cells[("family", "no_demos")]]


def _tiny():
    if not models.local_dir(QWEN).exists():
        pytest.skip("tokenizer files not downloaded")
    try:
        model, tok = models.load_tiny(QWEN)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"transformers cannot build the tiny model: {e}")
    import torch

    torch.manual_seed(3)
    cfg = model.config
    means = SimpleNamespace(expert_mean=torch.randn(cfg.num_hidden_layers, cfg.num_experts, cfg.hidden_size) * 0.5,
                            head_mean=torch.randn(cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim) * 0.5)
    return model, tok, means


def test_tiny_run_scores_every_cell_and_the_analysis_reads_it():
    from lesion import hooks, m3_battery as mb
    from lesion.components import Component

    model, tok, means = _tiny()
    ablate = functools.partial(hooks.ablate, model, means=means, alpha=0.0, mode="mean")
    battery, family = rdf.battery_items(3), rdf.family_items(3)
    cells = rdf.build_cells(rdf.NEW_CELLS + rdf.EXISTING_CELLS, battery, family)
    plan = [("intact", None)] + [(k, [Component.parse(c) for c in v]) for k, v in rdf.TINY_SETS.items()]
    tables = [rdf.score_state(model, tok, True, cells, (2, 1), name, comps, ablate)[0] for name, comps in plan]
    scores = pd.concat(tables, ignore_index=True)
    assert len(scores) == 3 * 2 * (3 * 5 + 3 * 3) * 2       # states x layouts x candidates of the four cells
    assert np.isfinite(scores.logprob).all()
    # each cell is scored on its own: the new cells do not depend on whether the existing ones are run
    alone = rdf.score_state(model, tok, True, rdf.build_cells(rdf.NEW_CELLS, battery, family), (2, 1), "intact")[0]
    new = np.array([cell in rdf.NEW_CELLS for cell in zip(scores.item_set, scores.format)])
    both = scores[(scores.state == "intact").to_numpy() & new]
    assert np.array_equal(alone.logprob.to_numpy(), both.logprob.to_numpy())
    # same computation as the original scorer of each cell
    direct = mb.candidate_scores(model, tok, cells[("battery", "demos")], True, 2)
    mine = scores[(scores.state == "intact") & (scores.item_set == "battery") & (scores.format == "demos")
                  & (scores.layout == 2)]
    assert np.allclose(direct.logprob.to_numpy(), mine.logprob.to_numpy())
    lesioned = scores[scores.state == "SC4"].logprob.to_numpy()
    assert not np.allclose(lesioned, scores[scores.state == "intact"].logprob.to_numpy())

    ledger = {i: dict(value=v, ci_low=None, ci_high=None, n=None, source="", pointer="")
              for i, v in zip(adf.EXISTING.values(), (0.0289, 0.84, 0.78, 0.90))}
    meta = dict(model=QWEN, tiny=True, sets={k: list(v) for k, v in rdf.TINY_SETS.items()}, batches=[2, 1])
    report = adf.analyse(scores, meta, ledger, n_boot=200, family_items=family)
    assert set(report["naming_loss"]) == {"battery", "family"}
    assert set(report["naming_loss"]["battery"]) == {"demos", "no_demos"}
    assert set(report["format_effect_within_run"]["family"]) == {"WR8", "SC4", "intact_target_logprob"}
    assert report["reading"]["rule"] in ("format", "concept", "between") and "reading_within_run" in report
    sc4 = report["naming_loss"]["battery"]["no_demos"]["SC4"]
    assert sc4["ci_low"] <= sc4["estimate"] <= sc4["ci_high"] and sc4["n"] == 3


def stat(estimate, lo, hi):
    return dict(estimate=estimate, ci_low=lo, ci_high=hi)


def test_reading_rules_follow_the_plan():
    family, battery = 0.78, 0.03
    assert adf.reading(stat(0.70, 0.50, 0.90), stat(0.05, -0.04, 0.15), family, battery)["rule"] == "format"
    assert adf.reading(stat(0.04, -0.03, 0.12), stat(0.80, 0.62, 0.97), family, battery)["rule"] == "concept"
    # battery rises but its interval misses 0.78; family unchanged; mixed changes are reported, not classified
    r = adf.reading(stat(0.40, 0.25, 0.55), stat(0.79, 0.60, 0.95), family, battery)
    assert r["rule"] == "between" and abs(r["change_from_existing"]["battery"]["estimate"] - 0.37) < 1e-12
    # the format rule needs the battery's lower limit above 0.20 even when its interval reaches 0.78
    assert adf.reading(stat(0.50, 0.15, 0.85), stat(0.05, -0.04, 0.15), family, battery)["rule"] == "between"
    # near zero means an upper limit below 0.20
    assert adf.reading(stat(0.04, -0.03, 0.21), stat(0.80, 0.62, 0.97), family, battery)["rule"] == "between"


def test_archived_outputs_reproduce_the_four_ledger_values():
    """0.0289, 0.8388, 0.782 and 0.904 from the stored original runs; the family SC4 interval exactly (outputs needed)."""
    if not (adf.ORIGINAL_FAMILY / "plan.json").exists() or not (adf.ORIGINAL_BATTERY / "meta.json").exists():
        pytest.skip("original outputs not present")
    original, notes, _ = adf.original_scores(adf.ORIGINAL_BATTERY, adf.ORIGINAL_FAMILY)
    assert not notes
    ledger = adf.ledger_rows()
    for item_set in ("battery", "family"):
        ids = adf.concept_order(item_set, original[(item_set, adf.OLD[item_set], "intact")].index)
        loss = adf.losses(original, item_set, adf.OLD[item_set], ids)
        for lesion in adf.LESIONS:
            s = adf.Resampler().stat(item_set, loss[lesion])
            row = ledger[adf.EXISTING[(item_set, lesion)]]
            assert abs(s["estimate"] - row["value"]) < 5e-5        # experiment 3's report rounds to 4 decimals
            if item_set == "family":
                assert (s["ci_low"], s["ci_high"]) == pytest.approx((row["ci_low"], row["ci_high"]), abs=1e-12)
    assert original["_family_prompts"] == {p.item_id: p.text for p in fs.naming_prompts(rdf.family_items())}
