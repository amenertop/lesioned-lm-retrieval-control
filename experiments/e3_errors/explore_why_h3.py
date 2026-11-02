"""Exploratory (2026-10-08): why H3a and H3b fail on the calibration concepts, from existing runs (no GPU).

    python experiments/e3_errors/explore_why_h3.py

1. H3a, candidate by candidate: under K8 (and K30), how much each wrong candidate gains in free naming, observed
   against the general degradation model fitted on the same lesion (calibration_b; per-candidate excess).
2. C and association in selection: per concept, the change of each option's gap to the target in six-choice
   (probe_dose choice scores, 6 rotations) under C4, C8, C16, and whether the associate's gain grows with the
   SWOW strength from target to associate.
3. H3b decomposed: cue benefit B and miscue harm H (hinted-demo baseline, battery) per lesion, B against the
   concept's naming loss and H against SWOW strength and associate frequency.
Writes reports/explore/qwen_m3_why_h3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import analyze_battery  # noqa: E402
from lesion import degradation as dg, e3_analysis as ea, knowledge_items as ki, m3_items, m3_scoring as ms  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
N_BOOT, SEED = 5000, 0


def ci(x, rng):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return dict(mean=round(float(x.mean()), 3), ci=[round(float(np.quantile(b, q)), 3) for q in (0.025, 0.975)], n=int(len(x)))


def rho(x, y, rng):
    x, y = pd.Series(np.asarray(x, dtype=float)), pd.Series(np.asarray(y, dtype=float))
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    f = lambda a, b: float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])
    bs = [f(x[i], y[i]) for i in rng.integers(0, len(x), (N_BOOT, len(x)))]
    return dict(rho=round(f(x, y), 3), ci=[round(float(np.nanquantile(bs, q)), 3) for q in (0.025, 0.975)], n=int(len(x)))


def main():
    rng = np.random.default_rng(SEED)
    probe = ROOT / "outputs/e3_errors" / MODEL
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False).set_index("item_id")
    zipf = ki.load_zipf()
    fits = json.loads((ROOT / "reports/explore/qwen_m3_degradation_fits.json").read_text(encoding="utf-8"))
    out = dict(note="exploratory, calibration concepts", h3a_by_candidate={}, selection_and_association={}, h3b_decomposed={})

    # 1. H3a candidate by candidate: observed vs degradation-predicted change of each candidate's share
    errors = pd.read_parquet(probe / "probe_errors/naming.parquet")
    cal_b = items.index[items.role == "calibration_b"].tolist()
    a = ea.h3a_tables(errors, cal_b)
    prior = dg.frequency_prior(ea.candidate_zipf(items.loc[cal_b], zipf).to_numpy())
    p_int = np.exp(dg.log_softmax(a["intact"]))
    for L in ("K8", "K16", "K30"):
        T, lam = fits["h3a"][L]["T"], fits["h3a"][L]["lam"]
        p_obs = np.exp(dg.log_softmax(a[L]))
        p_pred = np.exp(dg.log_predict(a["intact"], prior, T, lam))
        out["h3a_by_candidate"][L] = {r: dict(observed=ci(p_obs[:, j] - p_int[:, j], rng), predicted=ci(p_pred[:, j] - p_int[:, j], rng),
                                              excess=ci((p_obs[:, j] - p_int[:, j]) - (p_pred[:, j] - p_int[:, j]), rng))
                                      for j, r in enumerate(ms.ERROR_ROLES)}

    # 2. C and association in six-choice: option gaps to the target, change per role
    choice = pd.read_parquet(probe / "probe_dose/choice.parquet")
    for r in ms.ROLES[1:]:
        choice[f"gap_{r}"] = choice[f"logit_{r}"] - choice.logit_target
    g = choice.groupby(["set", "item_id"])[[f"gap_{r}" for r in ms.ROLES[1:]]].mean()
    swow = m3_items.load_swow()
    strength = {(c, r): st for c, r, st in swow.itertuples(index=False)}
    ids = g.loc["intact"].index
    st = pd.Series([strength.get((items.at[i, "target"], items.at[i, "associate"]), 0.0) for i in ids], index=ids)
    for L in ("C4", "C8", "C16", "K8"):
        ch = g.loc[L] - g.loc["intact"]
        others = ch[["gap_coordinate", "gap_superordinate", "gap_unrelated"]].mean(axis=1)
        out["selection_and_association"][L] = dict(
            gain_by_role={r: ci(ch[f"gap_{r}"], rng) for r in ms.ROLES[1:]},
            associate_gain_minus_other_words=ci(ch.gap_associate - others, rng),
            corr_associate_extra_gain_vs_swow_strength=rho(ch.gap_associate - others, st.loc[ch.index], rng))

    # 3. H3b decomposed (battery, hinted-demo baseline)
    bat = ROOT / "outputs/e3_errors" / MODEL / "battery"
    pc = analyze_battery.per_concept(pd.read_parquet(bat / "candidates.parquet"), pd.read_parquet(bat / "labels.parquet"))
    I = pc["intact"]
    cols = ["delta_none_hinted_demos", "delta_cue", "delta_miscue"]
    for L in ("K8", "K30", "C4", "C4_answer", "C16"):
        T_ = pc[L].reindex(I.index)
        ok = I[cols].notna().all(axis=1) & T_[cols].notna().all(axis=1)
        Ii, Tt = I.loc[ok], T_.loc[ok]
        B = (Tt.delta_cue - Tt.delta_none_hinted_demos) - (Ii.delta_cue - Ii.delta_none_hinted_demos)
        H = -((Tt.delta_miscue - Tt.delta_none_hinted_demos) - (Ii.delta_miscue - Ii.delta_none_hinted_demos))
        loss = (Ii.name_target_logprob - Tt.name_target_logprob)
        ids_ok = B.index
        s = pd.Series([strength.get((items.at[i, "target"], items.at[i, "associate"]), 0.0) for i in ids_ok], index=ids_ok)
        az = items.loc[ids_ok, "associate_zipf"].astype(float)
        out["h3b_decomposed"][L] = dict(B=ci(B, rng), H=ci(H, rng),
                                        corr_B_vs_naming_loss=rho(B, loss, rng), corr_H_vs_naming_loss=rho(H, loss, rng),
                                        corr_H_vs_swow_strength=rho(H, s, rng), corr_H_vs_associate_zipf=rho(H, az, rng),
                                        corr_H_vs_intact_miscue_lead=rho(H, Ii.delta_miscue, rng))

    path = ROOT / "reports/explore/qwen_m3_why_h3.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
