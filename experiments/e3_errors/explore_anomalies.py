"""Exploratory (2026-10-08): why do the lesions depart from the clinical prior? Offline, from the GPU session 6
calibration runs (80 calibration concepts; no test concept, no GPU):

    python experiments/e3_errors/explore_anomalies.py

1. Recognition versus production: the knowledge lesion leaves six-choice definition naming intact (even more
   confident) while free naming of the same concepts loses the target. Per concept: naming loss h (drop of the
   target's log-probability without a hint) against the change of the six-choice target margin.
2. Cue dependence: per concept, how cue and miscue effects (Delta_cue - Delta_none, Delta_miscue - Delta_none)
   change under each lesion, and whether the rise in R follows the concept's own naming loss (an additive account
   predicts no rise: hint evidence would simply add to weaker definition evidence).
3. Control lesion and association: whether the miscue change under the control heads follows the SWOW strength
   from target to associate and the associate's frequency.
4. Where the components sit (layer of 48).
Writes reports/explore/qwen_m3_anomalies.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import m3_items, m3_scoring as ms  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
N_BOOT, SEED = 5000, 0


def spearman(x, y):
    x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(x, y)[0, 1])


def boot_corr(x, y, rng):
    x, y = np.asarray(x), np.asarray(y)
    r = [spearman(x[i], y[i]) for i in rng.integers(0, len(x), (N_BOOT, len(x)))]
    return dict(rho=round(spearman(x, y), 3), ci=[round(float(np.nanquantile(r, q)), 3) for q in (0.025, 0.975)], n=len(x))


def mean_ci(x, rng):
    x = np.asarray(x, dtype=float)
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return dict(mean=round(float(x.mean()), 3), ci=[round(float(np.quantile(b, q)), 3) for q in (0.025, 0.975)], n=len(x))


def choice_margin(choice):
    """Per set and item: target logit minus log-sum-exp of the five other labels, averaged over rotations."""
    others = choice[[f"logit_{r}" for r in ms.ROLES[1:]]].to_numpy()
    m = others.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(others - m).sum(axis=1, keepdims=True)))[:, 0]
    return choice.assign(margin=choice.logit_target - lse).groupby(["set", "item_id"]).margin.mean()


def main():
    rng = np.random.default_rng(SEED)
    probe = ROOT / "outputs/e3_errors" / MODEL
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False).set_index("item_id")
    errors = pd.read_parquet(probe / "probe_errors/naming.parquet")
    dose = pd.read_parquet(probe / "probe_dose/naming.parquet")
    choice = pd.read_parquet(probe / "probe_dose/choice.parquet")
    meta = json.loads((probe / "probe_dose/meta.json").read_text(encoding="utf-8"))
    sets = [s for s in ("C4", "C8", "C16", "K8", "K16", "K30")]
    out = dict(note="exploratory, calibration concepts only (GPU session 6 runs)", components={}, recognition_vs_production={},
               cue_dependence={}, control_and_association={})

    # 4. where the components sit
    for s in sets:
        layers = [int(k.split(":")[1]) for k in meta["sets"][s]]
        out["components"][s] = dict(layers=sorted(layers), kinds=sorted({k.split(":")[0] for k in meta["sets"][s]}))

    # 1. recognition versus production
    tl = {s: ms.naming_errors(g).set_index("item_id").target_logprob for s, g in errors.groupby("set")}
    margin = choice_margin(choice)
    for s in sets:
        h = tl["intact"] - tl[s]
        dm = (margin.loc[s] - margin.loc["intact"]).loc[h.index]
        big = h > 2.0
        out["recognition_vs_production"][s] = dict(
            naming_loss=mean_ci(h, rng), six_choice_margin_change=mean_ci(dm, rng),
            corr_naming_loss_vs_margin_change=boot_corr(h, dm, rng),
            concepts_losing_over_2_nats_in_naming=int(big.sum()),
            of_which_still_top_in_six_choice=int((margin.loc[s].loc[h.index[big]] > 0).sum()))

    # 2. cue dependence
    deltas = {s: ms.naming_deltas(g).pivot(index="item_id", columns="condition", values="delta") for s, g in dose.groupby("set")}
    lp = {s: g.pivot_table(index=["item_id", "condition"], columns="role", values="logprob") for s, g in dose.groupby("set")}
    I = deltas["intact"]
    for s in sets:
        d = deltas[s]
        loss = I["none"] - d["none"]                       # how much the target lost its lead without a hint
        cue_ch = (d.cue - d["none"]) - (I.cue - I["none"])
        mis_ch = (d.miscue - d["none"]) - (I.miscue - I["none"])
        dR = (d.cue - d.miscue) - (I.cue - I.miscue)
        t = lp[s]["target"].unstack()
        ti = lp["intact"]["target"].unstack()
        a = lp[s]["associate"].unstack()
        ai = lp["intact"]["associate"].unstack()
        out["cue_dependence"][s] = dict(
            no_hint_lead_lost=mean_ci(loss, rng), cue_effect_change=mean_ci(cue_ch, rng),
            miscue_effect_change=mean_ci(mis_ch, rng), R_change=mean_ci(dR, rng),
            corr_lead_lost_vs_R_change=boot_corr(loss, dR, rng),
            target_logprob_change={c: mean_ci(t[c] - ti[c], rng) for c in ms.CONDITIONS},
            associate_logprob_change={c: mean_ci(a[c] - ai[c], rng) for c in ms.CONDITIONS},
            share_miscue_now_prefers_associate=round(float((d.miscue < 0).mean()), 3),
            share_miscue_prefers_associate_intact=round(float((I.miscue < 0).mean()), 3))

    # 3. control lesion and association strength
    swow = m3_items.load_swow()
    strength = {(c, r): st for c, r, st in swow.itertuples(index=False)}
    ids = deltas["intact"].index
    st = pd.Series([strength.get((items.at[i, "target"], items.at[i, "associate"]), 0.0) for i in ids], index=ids)
    az = items.loc[ids, "associate_zipf"].astype(float)
    for s in ("C4", "C8", "C16", "K8"):
        d = deltas[s]
        mis_ch = (d.miscue - d["none"]) - (I.miscue - I["none"])
        out["control_and_association"][s] = dict(
            corr_miscue_change_vs_swow_strength=boot_corr(mis_ch, st, rng),
            corr_miscue_change_vs_associate_zipf=boot_corr(mis_ch, az, rng),
            corr_miscue_change_vs_intact_miscue_delta=boot_corr(mis_ch, I.miscue, rng))

    path = ROOT / "reports/explore/qwen_m3_anomalies.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
