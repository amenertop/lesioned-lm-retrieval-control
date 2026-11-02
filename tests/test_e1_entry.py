"""End-to-end check of the MVP entry scripts on a tiny random OLMoE.

The real ablation code (lesion.hooks, lesion.means) belongs to agent2; here common.load is replaced by a
stand-in that zeroes expert outputs, so this test only covers the entry scripts' plumbing and file flow.
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from lesion import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e1_screen"))
pytestmark = pytest.mark.skipif(not models.local_dir("olmoe-0924-base").exists(), reason="tokenizer files not downloaded")

MODEL = "olmoe-0924-base"


class FakeMeans:
    meta = dict(model_name=MODEL)

    def routing_frequency(self):
        return np.tile(np.linspace(0.05, 0.4, 32), (2, 1))


@pytest.fixture
def entry(tmp_path, monkeypatch):
    import common

    model, tok = models.load_tiny(MODEL, num_experts=32)   # enough same-layer stand-ins for the random sets

    @contextmanager
    def zero_experts(components):
        handles = [model.model.layers[c.layer].mlp.experts[c.index].register_forward_hook(lambda m, i, o: o * 0)
                   for c in components]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    means_file = tmp_path / "means.pt"
    means_file.write_bytes(b"stand-in")
    monkeypatch.setattr(common, "load", lambda args: (model, tok, False, FakeMeans(), zero_experts))
    monkeypatch.setattr(common, "guard", lambda args: "0" * 40)
    monkeypatch.setattr(common, "out_dir", lambda args, stage: tmp_path / "out" / stage)
    return tmp_path, means_file


def test_screen_joint_report(entry, monkeypatch):
    import functools

    import report
    import run_joint
    import run_screen
    from lesion import screen

    # Tiny random model: pick 2 per direction, and mark the strongest control component as domain-general so
    # both the main (excluding) and the supplementary (unfiltered) control sets are exercised (ADR 0005).
    monkeypatch.setattr(screen, "select_with_supplement", functools.partial(screen.select_with_supplement, k=2))
    monkeypatch.setattr(screen, "domain_general", lambda s: s.control_effect == s.control_effect.max())
    tmp_path, means_file = entry
    base = ["--model", MODEL, "--means", str(means_file), "--tiny", "--batch-size", "8", "--n-boot", "50"]
    monkeypatch.setattr(sys, "argv", ["run_screen.py", *base, "--limit", "12", "--flush-every", "4"])
    run_screen.main()
    screen_dir = tmp_path / "out/screen"
    assert len(json.loads((screen_dir / "meta.json").read_text(encoding="utf-8"))) > 5
    selection = json.loads((screen_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["status"]["control"] == "ok" and len(selection["control"]) == 2
    unfiltered = selection["supplementary"]["control_unfiltered"]
    assert len(unfiltered) == 2 and unfiltered != selection["control"]
    assert selection["supplementary"]["excluded_domain_general"][0] in unfiltered

    monkeypatch.setattr(sys, "argv", ["run_joint.py", *base, "--n-random", "2"])
    run_joint.main()
    monkeypatch.setattr(sys, "argv", ["run_joint.py", *base, "--n-random", "3"])   # same folder, other settings
    with pytest.raises(SystemExit):
        run_joint.main()
    sets = json.loads((tmp_path / "out/joint/sets.json").read_text(encoding="utf-8"))
    assert {k for k in sets if k.startswith("random_")} and "control" in sets and "supp_control" in sets
    results = json.loads((tmp_path / "out/joint/results.json").read_text(encoding="utf-8"))
    supp = results["supplementary_unfiltered_control"]
    assert "control.control_effect" in supp["intervals"] and supp["random_rank"]["control"]["n_random"] == 2

    g1 = dict(chosen_version=MODEL, g1_decision="pass")
    text, verdict = report.build(MODEL, g1, screen_dir, tmp_path / "out/joint", tiny=True)
    assert verdict["outcome"] in {"signal", "no_signal"} and "MVP 报告" in text


def test_explore_dose_and_attention(tmp_path, monkeypatch):
    """Dose-response, zero ablation and head attention on a tiny Qwen with the real hooks and means."""
    import functools

    import common
    import run_explore
    import run_screen
    from lesion import hooks, means as means_mod, neutral_text, screen

    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3, model_name=name)
    means_file = tmp_path / "means.pt"
    m.save(means_file)
    monkeypatch.setattr(common, "load", lambda args: (model, tok, True, m,
                                                      functools.partial(hooks.ablate, model, means=m, alpha=0.0, mode="mean")))
    monkeypatch.setattr(common, "guard", lambda args: "0" * 40)
    monkeypatch.setattr(common, "out_dir", lambda args, stage: tmp_path / "out" / stage)
    monkeypatch.setattr(screen, "select_with_supplement", functools.partial(screen.select_with_supplement, k=2))
    base = ["--model", name, "--means", str(means_file), "--tiny", "--batch-size", "8", "--n-boot", "50"]
    monkeypatch.setattr(sys, "argv", ["run_screen.py", *base, "--kind", "head", "--flush-every", "4"])
    run_screen.main()
    monkeypatch.setattr(sys, "argv", ["run_explore.py", *base, "--n-random-heads", "2"])
    run_explore.main()
    dose = json.loads((tmp_path / "out/explore/dose.json").read_text(encoding="utf-8"))
    assert [(d["alpha"], d["mode"]) for d in dose] == [(0.75, "mean"), (0.5, "mean"), (0.25, "mean"), (0.0, "mean"), (0.0, "zero")]
    att = json.loads((tmp_path / "out/explore/attention_summary.json").read_text(encoding="utf-8"))
    assert {r["group"] for r in att["by_group"]} == {"control", "random"}


def test_control_test_script(tmp_path, monkeypatch):
    """The held-out control test end to end on a tiny Qwen; the dev split stands in for the locked test split."""
    import functools

    import common
    import run_screen
    from lesion import hooks, materials, means as means_mod, neutral_text, runinfo, screen

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e2_confirm"))
    import run_control_test

    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3, model_name=name)
    means_file = tmp_path / "means.pt"
    m.save(means_file)
    monkeypatch.setattr(common, "load", lambda args: (model, tok, True, m,
                                                      functools.partial(hooks.ablate, model, means=m, alpha=0.0, mode="mean")))
    monkeypatch.setattr(common, "guard", lambda args: "0" * 40)
    monkeypatch.setattr(common, "out_dir", lambda args, stage: tmp_path / "out" / stage)
    monkeypatch.setattr(screen, "select_with_supplement", functools.partial(screen.select_with_supplement, k=2))
    base = ["--model", name, "--means", str(means_file), "--tiny", "--batch-size", "8", "--n-boot", "50"]
    monkeypatch.setattr(sys, "argv", ["run_screen.py", *base, "--kind", "head", "--flush-every", "4"])
    run_screen.main()

    real_load_split = materials.load_split
    monkeypatch.setattr(runinfo, "check_analysis_freeze", lambda tag: "0" * 40)
    monkeypatch.setattr(materials, "load_split", lambda split, freeze_tag=None, **kw: real_load_split("dev"))
    monkeypatch.setattr(sys, "argv", ["run_control_test.py", *base, "--freeze-tag", "analysis-frozen-x",
                                      "--n-random", "3", "--out", str(tmp_path / "confirm")])
    run_control_test.main()
    res = json.loads((tmp_path / "confirm/results.json").read_text(encoding="utf-8"))
    assert res["primary"]["verdict"] in {"supported", "falsified", "insufficient"}
    assert res["random"]["n"] == 3 and "association_slope" in res["intervals"]

    # exploratory follow-up: scores on a non-held-out split, then the association analysis on them
    monkeypatch.setattr(materials, "load_split", real_load_split)
    import explore_association
    import score_sets

    monkeypatch.setattr(sys, "argv", ["score_sets.py", *base, "--split", "dev", "--out", str(tmp_path / "sdev")])
    score_sets.main()
    monkeypatch.setattr(sys, "argv", ["explore_association.py", "--dir", str(tmp_path / "sdev"), "--n-boot", "20"])
    explore_association.main()
    assoc = json.loads((tmp_path / "sdev/association.json").read_text(encoding="utf-8"))
    assert list(assoc) == ["control"] and "target_coef_margin_fixed" in assoc["control"]["intervals"]


def test_knowledge_screen_script(tmp_path, monkeypatch):
    """Naming-test attribution, verification and dev joint lesion end to end on a tiny Qwen."""
    import functools

    import common
    import run_screen
    from lesion import hooks, means as means_mod, neutral_text, screen

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e3_knowledge"))
    import run_knowledge

    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name, num_attention_heads=8, head_dim=4)   # enough same-layer heads for random sets
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3, model_name=name)
    means_file = tmp_path / "means.pt"
    m.save(means_file)
    monkeypatch.setattr(common, "load", lambda args: (model, tok, True, m,
                                                      functools.partial(hooks.ablate, model, means=m, alpha=0.0, mode="mean")))
    monkeypatch.setattr(common, "guard", lambda args: "0" * 40)
    monkeypatch.setattr(common, "out_dir", lambda args, stage: tmp_path / "out" / stage)
    monkeypatch.setattr(screen, "select_with_supplement", functools.partial(screen.select_with_supplement, k=2))
    base = ["--model", name, "--means", str(means_file), "--tiny", "--batch-size", "8", "--n-boot", "50"]
    monkeypatch.setattr(sys, "argv", ["run_screen.py", *base, "--kind", "head", "--flush-every", "4"])
    run_screen.main()

    monkeypatch.setattr(run_knowledge, "K", 2)
    monkeypatch.setattr(run_knowledge, "acceptable", lambda rows: len(rows) == 2)    # random model: effects have any sign
    monkeypatch.setattr(sys, "argv", ["run_knowledge.py", *base, "--top", "3", "--n-random", "2", "--out", str(tmp_path / "k")])
    run_knowledge.main()
    joint = json.loads((tmp_path / "k/joint.json").read_text(encoding="utf-8"))
    assert joint["selection"]["status"]["knowledge"] == "ok" and len(joint["selection"]["knowledge"]) == 2
    assert "knowledge_effect" in joint["knowledge_set_on_naming"] and joint["random_knowledge_effect"]["n"] == 2
    assert "control_effect" in joint["knowledge_set_on_m1"] and "preview_delta_F" in joint

    import run_expand8
    monkeypatch.setattr(run_expand8, "SIZES", (2, 3))
    monkeypatch.setattr(sys, "argv", ["run_expand8.py", *base, "--n-random", "2", "--knowledge-dir", str(tmp_path / "k"),
                                      "--out", str(tmp_path / "e8")])
    run_expand8.main()
    res = json.loads((tmp_path / "e8/results.json").read_text(encoding="utf-8"))
    for size in ("k2", "k3"):
        assert {"delta_F", "knowledge.knowledge_effect"} <= set(res["sets"][size]["naming"])
        assert "delta_C" in res["sets"][size]["m1"] and res["sets"][size]["random_knowledge_effect"]["n"] == 2

    # held-out dissociation test; the dev splits stand in for the locked test splits
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e2_confirm"))
    import run_dissociation_test as rdt
    from lesion import knowledge_items as ki, materials, runinfo

    monkeypatch.setattr(rdt, "MAIN", 3)
    monkeypatch.setattr(rdt, "REPORTED", 2)
    monkeypatch.setattr(runinfo, "check_analysis_freeze", lambda tag: "0" * 40)
    real_split, real_items = materials.load_split, ki.load_items
    monkeypatch.setattr(materials, "load_split", lambda split, freeze_tag=None, **kw: real_split("dev"))
    monkeypatch.setattr(ki, "load_items", lambda split=None, freeze_tag=None, **kw: real_items("dev"))
    monkeypatch.setattr(sys, "argv", ["run_dissociation_test.py", *base, "--freeze-tag", "analysis-frozen-x", "--n-random", "2",
                                      "--knowledge-dir", str(tmp_path / "k"), "--out", str(tmp_path / "d")])
    rdt.main()
    res = json.loads((tmp_path / "d/results.json").read_text(encoding="utf-8"))
    assert res["h2"]["decision"] in {"supported", "excluded", "insufficient"} and res["random"]["n"] == 2
    assert len(res["knowledge_sets"]["main"]) == 3 and res["knowledge_sets"]["main"][:2] == res["knowledge_sets"]["reported"]
    for name in ("main", "reported"):
        assert {"delta_F", "control.common_damage"} <= set(res["sets"][name]["naming"])
        assert {"delta_C", "control.low_demand_damage"} <= set(res["sets"][name]["m1"])


def test_dissociation_verdict():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e2_confirm"))
    import run_dissociation_test as rdt

    ci = lambda lo, hi: dict(ci_low=lo, ci_high=hi)
    assert rdt.verdict(ci(0.3, 1.0)) == dict(direction="supported", size="meaningful")
    assert rdt.verdict(ci(0.05, 1.0)) == dict(direction="supported", size="uncertain")
    assert rdt.verdict(ci(0.02, 0.15)) == dict(direction="supported", size="small")
    assert rdt.verdict(ci(-0.1, 0.15))["direction"] == "excluded"
    assert rdt.verdict(ci(-0.1, 0.5))["direction"] == "insufficient"
    s, e, i = (dict(direction=d) for d in ("supported", "excluded", "insufficient"))
    assert rdt.h2(s, s) == "supported" and rdt.h2(e, e) == "excluded" and rdt.h2(s, e) == "insufficient"


def test_attention_profile_script(tmp_path, monkeypatch):
    import functools

    import common
    import run_attention_profile
    import run_screen
    from lesion import hooks, means as means_mod, neutral_text, screen

    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3, model_name=name)
    means_file = tmp_path / "means.pt"
    m.save(means_file)
    monkeypatch.setattr(common, "load", lambda args: (model, tok, True, m,
                                                      functools.partial(hooks.ablate, model, means=m, alpha=0.0, mode="mean")))
    monkeypatch.setattr(common, "guard", lambda args: "0" * 40)
    monkeypatch.setattr(common, "out_dir", lambda args, stage: tmp_path / "out" / stage)
    monkeypatch.setattr(screen, "select_with_supplement", functools.partial(screen.select_with_supplement, k=2))
    base = ["--model", name, "--means", str(means_file), "--tiny", "--batch-size", "8", "--n-boot", "50"]
    monkeypatch.setattr(sys, "argv", ["run_screen.py", *base, "--kind", "head", "--flush-every", "4"])
    run_screen.main()
    monkeypatch.setattr(sys, "argv", ["run_attention_profile.py", *base, "--n-random-heads", "2"])
    run_attention_profile.main()
    summary = json.loads((tmp_path / "out/attention_profile/summary.json").read_text(encoding="utf-8"))
    assert {r["group"] for r in summary["by_group"]} == {"control", "random"} and summary["implementation"] == "eager"
