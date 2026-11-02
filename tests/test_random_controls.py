"""Random-set specificity control for the mechanism battery (experiments/e3_errors/run_random_controls.py)."""
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import analyze_battery  # noqa: E402
import run_random_controls as rc  # noqa: E402
from lesion import m3_battery as mb, m3_items, models  # noqa: E402

QWEN = "qwen3-30b-a3b-instruct"


def test_states_metrics_and_comparison_on_a_tiny_model():
    if not models.local_dir(QWEN).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(QWEN)
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    pool = final[final.role.isin(mb.POOL_ROLES)].reset_index(drop=True)
    items = pool[pool.role.isin(("calibration_a", "calibration_b")) & pool.free_naming_ok.astype(bool)].head(3)
    cands, labs = [], []
    for name in ("intact", "K8", "randomK_00", "randomK_01"):
        c, l = rc.score_state(model, tok, True, items, pool, None if name == "intact" else ["x"], lambda comps: nullcontext(), name)
        cands.append(c)
        labs.append(l)
    pc = analyze_battery.per_concept(pd.concat(cands, ignore_index=True), pd.concat(labs, ignore_index=True))
    m = rc.state_metrics(pc)
    assert set(m) == {"K8", "randomK_00", "randomK_01"} and set(m["K8"]) == set(rc.METRICS)
    assert abs(m["K8"]["naming_loss"]) < 1e-6          # the fake lesion changes nothing (same composition)
    cmp = rc.compare(m, "K8", "randomK_")
    assert cmp["naming_loss"]["n_random"] == 2


def test_intrusion_prompts_measures_and_summary():
    if not models.local_dir(QWEN).exists():
        pytest.skip("tokenizer files not downloaded")
    import run_intrusion as ri

    model, tok = models.load_tiny(QWEN)
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role == "calibration_b"].head(3).reset_index(drop=True)
    prompts = mb.intrusion_prompts(items)
    assert len(prompts) == 9 and {p.condition for p in prompts} == set(mb.INTRUSION_CONDITIONS)
    row = items.iloc[0]
    by = {p.condition: p for p in prompts if p.item_id == row.item_id}
    assert f"Ignore this word: {row.associate}\nDescription: {row.definition}" in by["distract_assoc"].text
    assert f"Ignore this word: {row.unrelated}\nDescription:" in by["distract_unrel"].text
    assert by["distract_none"].text.count("Ignore this word:") == 3          # only the demos
    cands = [mb.candidate_scores(model, tok, prompts, True, 4).assign(state=s) for s in ("intact", "intact1", "C4", "randomC_00")]
    m = mb.intrusion_measures(pd.concat(cands, ignore_index=True))
    assert set(m.columns) == {"pull_assoc", "pull_unrel", "target_logprob_assoc_shown"}
    s = ri.summarize(pd.concat(cands, ignore_index=True), None, items)
    assert abs(s["states"]["C4"]["pull_assoc_change"]["mean"]) < 1e-6 and s["C4_vs_random"]["n_random"] == 1
    sp = mb.scored_prompts(items, intrusion=True)
    assert {"distract_assoc", "distract_unrel", "distract_none"} <= {p.condition for p in sp}


def test_top_correct_needs_the_highest_label():
    lab = pd.DataFrame([dict(correct="A", logit_A=1.0, logit_B=0.0, logit_C=0.0, logit_D=0.0, logit_E=0.0, logit_F=0.0),
                        dict(correct="B", logit_A=1.0, logit_B=0.5, logit_C=None, logit_D=None, logit_E=None, logit_F=None)])
    assert rc.top_correct(lab) == [True, False]
