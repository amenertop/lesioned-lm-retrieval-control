"""Screening loop, selection and joint comparison (lesion.screen) with a fake scorer; no hooks needed."""
from contextlib import contextmanager

import pytest
import torch

import fakes
from lesion import damage, materials, models, scoring, screen
from lesion.components import Component

COMPONENTS = [Component("expert", l, e) for l in range(2) for e in range(8)]
CONTROL, KNOWLEDGE = Component("expert", 0, 1), Component("expert", 1, 2)


@pytest.fixture(scope="module")
def setup():
    items = materials.load_split("screening")
    fm = damage.frequency_median(scoring.load_materials()[0])
    rare = lambda r: r["task"] == "synonym" and float(r["min_freq"]) < fm

    def margin(r, lesioned):
        m = 2.0 + 0.01 * (int(r["item_id"]) % 7)
        for c in lesioned:
            m -= 0.05 * (c.index + 1) / 8                     # every component hurts a little, uniformly
            if c == CONTROL and r["high_demand"] == "1":
                m -= 1.0
            if c == KNOWLEDGE and rare(r):
                m -= 0.8
        return m

    calls = []

    def score(components=()):
        calls.append(tuple(components))
        return fakes.label_scores(items, lambda r: margin(r, components))

    return items, fm, score, calls


def test_screen_finds_planted_components_and_resumes(setup, tmp_path):
    items, fm, score, calls = setup
    screen.run_screen(score, COMPONENTS[:5], tmp_path, flush_every=2, log=lambda *_: None)
    calls.clear()
    shards = screen.run_screen(score, COMPONENTS, tmp_path, flush_every=4, log=lambda *_: None)
    assert len(calls) == len(COMPONENTS) - 5                      # resumed, earlier components not rerun
    assert shards.groupby("component").size().eq(2 * len(items)).all()
    summary = screen.summarize(score(), shards, items, fm)
    assert len(summary) == len(COMPONENTS)
    chosen = screen.select(summary, k=2)
    assert chosen["control"][0] == CONTROL.key and chosen["knowledge"][0] == KNOWLEDGE.key
    assert not set(chosen["control"]) & set(chosen["knowledge"])


def test_compare_sets_and_verdict(setup):
    items, fm, score, _ = setup
    lesioned = {"control": score([CONTROL]), "knowledge": score([KNOWLEDGE])}
    for i in range(3):
        lesioned[f"random_control_{i:02d}"] = score([Component("expert", 0, 4 + i)])
        lesioned[f"random_knowledge_{i:02d}"] = score([Component("expert", 1, 4 + i)])
    points, ci, rank = screen.compare_sets(score(), lesioned, items, fm, n_boot=200)
    assert ci["delta_C"]["estimate"] == pytest.approx(1.0) and ci["delta_F"]["estimate"] == pytest.approx(0.8)
    assert rank["control"] == dict(n_random=3, share_at_least_target=0.0)
    v = screen.mvp_verdict("pass", ci)
    assert v["outcome"] == "signal" and v["above_threshold"] == {"control": True, "knowledge": True}
    assert v["g2_preserved_below"] == {"control": True, "knowledge": True}
    assert screen.mvp_verdict("measurement_fail", ci)["outcome"] == "g1_fail"


@pytest.mark.skipif(not models.local_dir("olmoe-0924-base").exists(), reason="tokenizer files not downloaded")
def test_make_scorer_uses_ablate_only_when_lesioning():
    model, tok = models.load_tiny("olmoe-0924-base")
    items, demos, templates = scoring.load_materials()
    prompts = scoring.build_label_prompts(items.head(3), demos, templates)
    entered = []

    @contextmanager
    def fake_ablate(components):
        entered.append(components)
        handle = model.lm_head.register_forward_hook(lambda m, i, out: out + 1.0 * (out.argmax(-1, keepdim=True) == torch.arange(out.shape[-1])))
        try:
            yield
        finally:
            handle.remove()

    score = screen.make_scorer(model, tok, prompts, chat=False, ablate=fake_ablate, batch_size=4)
    a, b, c = score(), score([CONTROL]), score()
    assert entered == [[CONTROL]]
    assert a[["logit_A", "logit_B"]].equals(c[["logit_A", "logit_B"]])     # hook removed on exit


def test_select_reports_short_or_flat_directions():
    import pandas as pd

    split = [dict(component=f"expert:0:{i}", control_effect=10.0 + i, knowledge_effect=0.0) for i in range(5)] +             [dict(component=f"expert:1:{i}", control_effect=0.0, knowledge_effect=10.0 + i) for i in range(5)]
    chosen = screen.select(pd.DataFrame(split), k=4)
    assert chosen["status"] == {"control": "ok", "knowledge": "ok"}
    assert chosen["control"][0] == "expert:0:4" and chosen["knowledge"][0] == "expert:1:4"
    short = screen.select(pd.DataFrame(split[:3] + split[5:]), k=4)      # only 3 components lean to control
    assert short["status"]["control"].startswith("not_found") and short["control"] == []
    assert short["status"]["knowledge"] == "ok"
    flat = pd.DataFrame([dict(component=f"expert:0:{i}", control_effect=1.0, knowledge_effect=float(i)) for i in range(6)])
    with pytest.raises(ValueError, match="cannot standardise"):
        screen.select(flat)


def test_domain_general_exclusion_and_supplement():
    import pandas as pd

    rows = [dict(component=f"expert:0:{i}", control_effect=10.0 + i, knowledge_effect=0.0, string_damage=0.0) for i in range(5)]
    rows += [dict(component=f"expert:1:{i}", control_effect=0.0, knowledge_effect=10.0 + i, string_damage=0.0) for i in range(5)]
    rows[4]["string_damage"] = 50.0                  # the strongest control component also wrecks string matching
    chosen = screen.select_with_supplement(pd.DataFrame(rows), k=4)
    assert "expert:0:4" not in chosen["control"] and chosen["control"] == ["expert:0:3", "expert:0:2", "expert:0:1", "expert:0:0"]
    assert chosen["supplementary"]["control_unfiltered"][0] == "expert:0:4"
    assert chosen["supplementary"]["excluded_domain_general"] == ["expert:0:4"]
    assert chosen["knowledge"][0] == "expert:1:4"    # the knowledge direction is untouched by the exclusion
