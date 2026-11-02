"""Fit the general degradation model on the calibration concepts before the freeze (experiment3-protocol.md v0.5
§5.3), from the GPU session 6 calibration runs (no model, no GPU):

    python experiments/e3_errors/fit_degradation.py

- H3a: part 2b, free naming without a hint, five candidates (outputs/e3_errors/<model>/probe_errors/naming.parquet);
- H3b: part 3, the same 40 concepts under none, cue and miscue, target and associate (probe_dose/naming.parquet).
For every lesion set (main: K8, C4; dose curve: K16, K30, C8, C16): (T, lam) at the grid minimum, the near-optimal
region, the profiles, and 10,000 concept resamples for the separability check; also the part 2a severity h_bar
(drop of the target's log-probability without a hint, §4.2). Writes reports/explore/qwen_m3_degradation_fits.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import degradation as dg, e3_analysis as ea, knowledge_items as ki, m3_items, runinfo  # noqa: E402
from lesion import m3_scoring as ms  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
SETS = ("C4", "K8", "C8", "C16", "K16", "K30")
N_BOOT, SEED = 10_000, 0


def separability(ce, rng):
    idx = dg.bootstrap_fits(ce, ea.resample_counts(len(ce), N_BOOT, rng))
    return dg.ridge_check(idx)


def main():
    probe = ROOT / "outputs/e3_errors" / MODEL
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False).set_index("item_id")
    cal_b = items.index[items.role == "calibration_b"].tolist()
    cal_a = items.index[items.role == "calibration_a"].tolist()
    zipf = ki.load_zipf()
    errors = pd.read_parquet(probe / "probe_errors/naming.parquet")
    hints = pd.read_parquet(probe / "probe_dose/naming.parquet")
    rng = np.random.default_rng(SEED)
    out = dict(model=MODEL, n_boot=N_BOOT, seed=SEED, grid=dict(T=[1, 10, len(dg.T_GRID), "log-spaced"], lam=[0, 1, 0.01]),
               inputs={str(p.relative_to(ROOT)): runinfo.sha256(p) for p in
                       (probe / "probe_errors/naming.parquet", probe / "probe_dose/naming.parquet",
                        probe / "probe_errors/meta.json", probe / "probe_dose/meta.json", m3_items.OUT / "final_items.csv")},
               n_part_2b=len(cal_b), n_part_2a=len(cal_a), h3a={}, h3b={}, severity_part_2a={})

    # H3a: five candidates, part 2b
    a = ea.h3a_tables(errors, cal_b)
    prior_a = dg.frequency_prior(ea.candidate_zipf(items.loc[cal_b], zipf).to_numpy())
    s_intact = ea.s_stat(a["intact"])
    for L in SETS:
        p_obs = np.exp(dg.log_softmax(a[L]))
        f = dg.fit(p_obs, a["intact"], prior_a)
        ce = dg.item_cross_entropy_grid(p_obs, a["intact"], prior_a)
        pred = ea.s_stat(dg.log_predict(a["intact"], prior_a, f["T"], f["lam"]))
        out["h3a"][L] = dict(T=f["T"], lam=f["lam"], cross_entropy=f["cross_entropy"], near_optimal=f["near_optimal"],
                             separability=separability(ce, rng),
                             dS_observed=float((ea.s_stat(a[L]) - s_intact).mean()),
                             dS_predicted=float((pred - s_intact).mean()),
                             profile_T=f["profile_T"], profile_lam=f["profile_lam"])

    # H3b: target and associate under three conditions, part 3 (= the part 2b concepts)
    b = ea.h3b_tables(hints, cal_b)
    prior_b = dg.frequency_prior(ea.candidate_zipf(items.loc[cal_b], zipf, ea.H3B_ROLES).to_numpy())
    r_intact = ea.r_stat(b["intact"])
    for L in SETS:
        ce = ea.h3b_item_ce_grid(b[L], b["intact"], prior_b)
        T, lam, _ = dg.best(ce.sum(axis=0))
        n = len(cal_b)
        pred = ea.r_stat(dg.log_predict(b["intact"].reshape(n * 3, 2), np.repeat(prior_b, 3, axis=0), T, lam).reshape(n, 3, 2))
        total = ce.sum(axis=0)
        near = total <= total.min() * 1.01
        out["h3b"][L] = dict(T=T, lam=lam, cross_entropy=float(total.min()), near_optimal_points=int(near.sum()),
                             separability=separability(ce, rng),
                             dR_observed=float((ea.r_stat(b[L]) - r_intact).mean()),
                             dR_predicted=float((pred - r_intact).mean()))

    # severity on part 2a (§4.2): drop of the target's log-probability without a hint
    t = {s: ms.naming_errors(g).set_index("item_id").target_logprob for s, g in errors.groupby("set")}
    for L in SETS:
        h = (t["intact"] - t[L]).loc[cal_a].to_numpy()
        boots = ea.resample_counts(len(h), N_BOOT, rng) @ h / len(h)
        out["severity_part_2a"][L] = dict(h_bar=float(h.mean()), ci=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))])

    path = ROOT / "reports/explore/qwen_m3_degradation_fits.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for task, key in (("h3a", "dS"), ("h3b", "dR")):
        for L in SETS:
            r = out[task][L]
            print(f"{task} {L:4s} T={r['T']:.2f} lam={r['lam']:.2f} sep r={r['separability']['correlation']:.2f} "
                  f"ridge={r['separability']['ridge']} {key} obs {r[key + '_observed']:.3f} pred {r[key + '_predicted']:.3f}")
    for L in SETS:
        print("severity", L, out["severity_part_2a"][L])


if __name__ == "__main__":
    main()
