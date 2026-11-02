"""Experiment 3 analysis helpers (experiments/e3_errors/analyze_experiment3.py) and the confirmatory runner's plan."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import analyze_experiment3 as ax  # noqa: E402
import run_experiment3 as rx  # noqa: E402


def test_holm():
    adj = ax.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)


def test_summary_sides_and_tiers():
    boots = np.array([0.1, 0.2, 0.3, 0.4])
    s = ax.summary([0.25], boots, 0.0)
    assert s["p_one_sided"] == 1 / 5 and s["estimate"] == 0.25
    assert ax.summary([0.25], boots, 0.35, side="less")["p_one_sided"] == 2 / 5
    assert ax.tier(0.3, 0.9, 0.2).endswith("达到最小有意义效应") and ax.tier(0.05, 0.15, 0.2).endswith("很小")
    assert ax.tier(-0.1, 0.5, 0.2) == "证据不足"


def test_six_choice_roles_follow_the_prompt_rotations():
    import pandas as pd

    from lesion import m3_scoring as ms

    items = pd.DataFrame([dict(item_id="m001", target="camel", definition="d", coordinate="llama", superordinate="animal",
                               associate="desert", unrelated="violin")])
    prompts = ms.build_choice_prompts(items)
    rows = []
    for p in prompts:                          # give the target's letter a high logit in every rotation
        rows.append(dict(state="intact", item_id="m001", fmt="options_letter", variant=p.rotation,
                         **{f"logit_{l}": (5.0 if p.letter_to_role[l] == "target" else 0.0) for l in ms.LETTERS}))
    pr = ax.six_choice_role_probabilities(pd.DataFrame(rows), ["m001"])
    assert pr.loc[("intact", "m001"), "p_target"] > 0.9


def test_registered_s7_without_intrusion_scores_stops(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(ax, "S7_PREREGISTERED", True)
    battery = ROOT / "outputs/e3_errors/qwen3-30b-a3b-instruct/battery"
    dla = ROOT / "outputs/e3_errors/qwen3-30b-a3b-instruct/dla/direct.parquet"
    if not battery.exists() or not dla.exists():
        pytest.skip("calibration battery outputs not available")
    with pytest.raises(SystemExit, match="family must not shrink"):
        ax.main(["--dir", str(battery), "--dla", str(dla), "--dry-run-role", "calibration", "--n-boot", "50",
                 "--out", str(tmp_path / "x.json")])


def test_fixed_foils_are_checked(monkeypatch, tmp_path):
    import pandas as pd
    import pytest

    from lesion import m3_items

    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role == "test"].reset_index(drop=True)
    pool = final[final.role.isin(rx.POOL_ROLES)].reset_index(drop=True)
    rx.check_foils(items, pool)                                   # the stored map matches the rule
    bad = pd.read_csv(rx.FOILS, keep_default_na=False)
    bad.loc[0, "foil_id"] = bad.loc[1, "foil_id"] if bad.loc[1, "foil_id"] != bad.loc[0, "foil_id"] else "m999"
    bad.to_csv(tmp_path / "foils.csv", index=False)
    monkeypatch.setattr(rx, "FOILS", tmp_path / "foils.csv")
    with pytest.raises(SystemExit, match="foils differ"):
        rx.check_foils(items, pool)
    assert len(rx.K30) == 30 and len(rx.C16) == 16 and rx.MEANS_SHA256.startswith("98b3865a")


def test_confirmatory_plan_and_constants():
    names = [s[0] for s in rx.plan({k: [] for k in ("K8", "C4", "K30", "C16")}, None)]
    assert names == ["intact", "K8", "C4", "C4_answer", "K30", "C16"]
    assert rx.plan({k: [] for k in ("K8", "C4", "K30", "C16")}, None)[3][3] == "answer"
    assert ax.H3A_EXCLUDED == ("m000", "m043", "m081", "m124") and ax.NONINFERIORITY == 0.73
