"""Analysis of experiment 3 on the test concepts (docs/analysis-plan/experiment3.md §4 to §7; no GPU):

    python experiments/e3_errors/analyze_experiment3.py [--dir outputs/e3_errors/<model>/confirm]

Primary family (Holm): H3a (error types in free naming) and H3b (cue and miscue), each D_obs with the excess over
the general degradation model fitted on calibration_b (GPU session 6 runs, probe_errors and probe_dose).
Secondary family (Holm): S1 cross-format double dissociation, S2 K's naming loss with comprehension spared,
S3 C4's miscue asymmetry at the answer region, S4 K8's direct push of the specific target word.
Report-only quantities as listed in §6 and §7. Writes reports/confirm/experiment3/results.json (or --out).

--dry-run-role calibration runs the same analysis on a battery folder of calibration concepts (code check only;
the degradation fit is then on the same concepts and means nothing).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))

import analyze_battery  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import degradation as dg, e3_analysis as ea, e3_sources, knowledge_items as ki, m3_battery as mb, m3_items, m3_scoring as ms  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
H3A_EXCLUDED = ("m000", "m043", "m081", "m124")       # Q15: intact target not top or below half of five candidates
MDE_H3A, MDE_H3B, MDE_S = 0.05, 0.20, 0.20
NONINFERIORITY = 0.73                                  # S2: 0.2 x SD of the intact word-to-definition margin (calibration)
S7_PREREGISTERED = False   # fixed at the freeze: the calibration intrusion run did not support S7 (C4 -0.24 [-0.36, -0.12]), so report only
N_BOOT, SEED = 10_000, 0


def holm(pvals):
    """Holm-adjusted p-values for a dict name -> p."""
    names = sorted(pvals, key=pvals.get)
    m, out, running = len(names), {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        out[n] = running
    return out


def boot_mean(x, counts):
    x = np.asarray(x, dtype=float)
    return counts @ x / len(x)


def summary(x, boots, threshold=0.0, side="greater"):
    """Mean, 95% interval and one-sided p for 'mean > threshold' (side greater) or 'mean < threshold' (less)."""
    p = ((boots <= threshold).sum() + 1) / (len(boots) + 1) if side == "greater" else ((boots >= threshold).sum() + 1) / (len(boots) + 1)
    return dict(estimate=round(float(np.mean(x)), 4), ci=[round(float(np.quantile(boots, q)), 4) for q in (0.025, 0.975)],
                p_one_sided=float(p), n=int(len(x)))


def tier(lo, hi, mde):
    if lo > mde:
        return "支持，达到最小有意义效应"
    if lo > 0 and hi < mde:
        return "支持，存在但很小"
    if lo > 0:
        return "支持，大小未定"
    return "证据不足"


def calibration_fits(items):
    """Per-concept cross-entropy grids on calibration_b for H3a (probe_errors) and H3b (probe_dose), K8 and C4."""
    probe = ROOT / "outputs/e3_errors" / MODEL
    cal_b = items.index[items.role == "calibration_b"].tolist()
    zipf = ki.load_zipf()
    errors = pd.read_parquet(probe / "probe_errors/naming.parquet")
    hints = pd.read_parquet(probe / "probe_dose/naming.parquet")
    a = ea.h3a_tables(errors, cal_b)
    prior_a = dg.frequency_prior(ea.candidate_zipf(items.loc[cal_b], zipf).to_numpy())
    b = ea.h3b_tables(hints, cal_b)
    prior_b = dg.frequency_prior(ea.candidate_zipf(items.loc[cal_b], zipf, ea.H3B_ROLES).to_numpy())
    ce_a = {L: dg.item_cross_entropy_grid(np.exp(dg.log_softmax(a[src])), a["intact"], prior_a)
            for L, src in (("K", "K8"), ("C", "C4"), ("K30", "K30"))}
    ce_b = {L: ea.h3b_item_ce_grid(b[src], b["intact"], prior_b) for L, src in (("K", "K8"), ("C", "C4"))}
    return ce_a, ce_b, zipf


def six_choice_role_probabilities(labels, ids):
    """Per state and concept: probability of each option role in six-choice, softmax over the six letters within a
    rotation, averaged over rotations (rotation r shows roles base_order[6-r:] + base_order[:6-r] at A..F)."""
    lab = labels[(labels.fmt == "options_letter") & labels.item_id.isin(ids)]
    rows = []
    for rec in lab.to_dict("records"):
        order = ms.base_order(rec["item_id"])
        r = int(rec["variant"])
        roles = order[len(order) - r:] + order[:len(order) - r]
        lg = np.array([rec[f"logit_{l}"] for l in ms.LETTERS], dtype=float)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        rows.append(dict(state=rec["state"], item_id=rec["item_id"], **{f"p_{role}": float(v) for role, v in zip(roles, p)}))
    return pd.DataFrame(rows).groupby(["state", "item_id"]).mean()


def report_sensitivity(rep, items, test_ids, h3a_ids, h3b_ids, pos, cand, labels, I, T, b, counts, cnt_b, spec, d, args):
    """Sensitivity and supplementary results promised in analysis plan §6 and §7 (report only)."""
    # H3a on the log scale and per token (ms.naming_errors), D_log = dS_log_C - dS_log_K, on the H3a concepts
    ne = {s: ms.naming_errors(g.assign(condition="none")).set_index("item_id").reindex(h3a_ids)
          for s, g in cand[cand.condition == "name_free"].groupby("state")}
    c_h = ea.resample_counts(len(h3a_ids), args.n_boot, np.random.default_rng(SEED + 1))
    for col in ("S_log", "S_log_pt"):
        ch = {s: (ne[s][col] - ne["intact"][col]).to_numpy() for s in ne if s != "intact"}
        rep[f"H3a_{col}"] = {s: summary(v, boot_mean(v, c_h)) for s, v in ch.items()}
        dd = ch["C4"] - ch["K8"]
        rep[f"H3a_{col}"]["D_C4_minus_K8"] = summary(dd, boot_mean(dd, c_h))
    # H3b: fixed-cue null (predicts dR = 0, so the observed dR is its residual), dR per state incl. the dose sets
    R = {s: ea.r_stat(v) for s, v in b.items()}
    rep["H3b_dR"] = {s: summary(R[s] - R["intact"], boot_mean(R[s] - R["intact"], cnt_b)) for s in R if s != "intact"}
    # A_C - A_K paired on the same concepts (hinted-demo baseline)
    cols = ["delta_none_hinted_demos", "delta_cue", "delta_miscue"]
    Ii = I[cols].to_numpy()[pos]

    def asym(state):
        Ts = T[state][cols].to_numpy()[pos]
        return ((Ts[:, 1] - Ts[:, 0]) - (Ii[:, 1] - Ii[:, 0])) + ((Ts[:, 2] - Ts[:, 0]) - (Ii[:, 2] - Ii[:, 0]))

    ac = asym("C4") - asym("K8")
    rep["A_C4_minus_A_K8"] = summary(ac, boot_mean(ac, cnt_b))
    # evidence weights fitted on calibration_b (plain no-hint baseline), applied to the test concepts (plan §6)
    probe = ROOT / "outputs/e3_errors" / MODEL / "probe_dose/naming.parquet"
    if probe.exists():
        cal_b = items.index[items.role == "calibration_b"].tolist()
        cal = ea.h3b_tables(pd.read_parquet(probe), cal_b)
        dcal = {s: v[..., 0] - v[..., 1] for s, v in cal.items()}
        dtest = {s: v[..., 0] - v[..., 1] for s, v in b.items()}
        rep["evidence_weights_calibration_to_test"] = {}
        for s in ("K8", "C4"):
            a_, bc, bm = ea.evidence_weights(dcal["intact"], dcal[s])
            di = dtest["intact"]
            pred = np.column_stack([a_ * di[:, 0], a_ * di[:, 0] + bc * (di[:, 1] - di[:, 0]), a_ * di[:, 0] + bm * (di[:, 2] - di[:, 0])])
            resid_R = (dtest[s][:, 1] - dtest[s][:, 2]) - (pred[:, 1] - pred[:, 2])
            rep["evidence_weights_calibration_to_test"][s] = dict(
                weights=dict(a=round(a_, 4), b_cue=round(bc, 4), b_miscue=round(bm, 4)),
                predicted_dR=round(float(((pred[:, 1] - pred[:, 2]) - (di[:, 1] - di[:, 2])).mean()), 4),
                observed_dR=round(float(((dtest[s][:, 1] - dtest[s][:, 2]) - (di[:, 1] - di[:, 2])).mean()), 4),
                residual_R=summary(resid_R, boot_mean(resid_R, cnt_b)),
                residual_by_condition={c: round(float((dtest[s][:, j] - pred[:, j]).mean()), 4) for j, c in enumerate(("none", "cue", "miscue"))})
    # six-choice: change of every option role's probability
    pr = six_choice_role_probabilities(labels, test_ids)
    rep["six_choice_role_probability_change"] = {
        s: {c: round(float((pr.loc[s][c] - pr.loc["intact"][c]).mean()), 5) for c in pr.columns}
        for s in pr.index.get_level_values(0).unique() if s != "intact"}
    # S2 on the strict same-category foils
    strict = labels[(labels.state == "intact") & (labels.fmt == "word_to_definition") & (labels.get("foil_match") == "category")]
    if len(strict):
        ids = [i for i in test_ids if i in set(strict.item_id)]
        loss = (I.word_to_definition_margin - T["K8"].word_to_definition_margin).reindex(ids).to_numpy()
        rep["S2_comprehension_loss_same_category"] = summary(loss, boot_mean(loss, ea.resample_counts(len(ids), args.n_boot, np.random.default_rng(SEED + 2))))
    # direct against total effect on the same scale (first-token logits, own target minus the other targets)
    tot = d / "dla_total.parquet"
    if tot.exists():
        t = pd.read_parquet(tot)
        ti, tk = t[t.state == "intact"].set_index("item_id"), t[t.state == "K8"].set_index("item_id")
        total = ((ti.target_logit - ti.other_targets_logit) - (tk.target_logit - tk.other_targets_logit)).reindex(test_ids).to_numpy()
        rep["S4_total_specific_K8"] = summary(total, boot_mean(total, counts))
        rep["S4_corr_direct_vs_total"] = round(float(np.corrcoef(spec, total)[0, 1]), 3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs/e3_errors" / MODEL / "confirm"))
    ap.add_argument("--dla", default=None, help="direct.parquet when DLA sits outside --dir (dry run)")
    ap.add_argument("--out", default=str(ROOT / "reports/confirm/experiment3/results.json"))
    ap.add_argument("--dry-run-role", default=None, choices=[None, "calibration"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    e3_sources.check_output(args.out, args.dry_run_role, args.dla)
    provenance = None
    if not args.dry_run_role:
        from run_experiment3 import K30, C16
        control = json.loads((ROOT / "reports/confirm/control/results.json").read_text(encoding="utf-8"))["control"]
        provenance = e3_sources.check_source(d, m3_items.OUT / "final_items.csv",
                                             dict(K8=rdt.K8, C4=control, K30=K30, C16=C16))
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False).set_index("item_id")
    cand = pd.read_parquet(d / "candidates.parquet")
    labels = pd.read_parquet(d / "labels.parquet")
    rng = np.random.default_rng(SEED)
    if args.dry_run_role:
        test_ids = items.index[items.role.isin(("calibration_a", "calibration_b"))].tolist()
        h3a_ids, h3b_ids = test_ids, [i for i in test_ids if items.at[i, "free_naming_ok"] in (True, "True")]
    else:
        test_ids = items.index[items.role == "test"].tolist()
        h3a_ids = [i for i in test_ids if i not in H3A_EXCLUDED]
        h3b_ids = items.index[(items.role == "test") & items.free_naming.astype(str).eq("True")].tolist()
    out = dict(dry_run=bool(args.dry_run_role), n=dict(test=len(test_ids), h3a=len(h3a_ids), h3b=len(h3b_ids)), primary={}, secondary={},
               report={}, provenance=provenance)

    # --- primary ------------------------------------------------------------------------------------------------
    ce_a, ce_b, zipf = calibration_fits(items)
    by_set = cand.rename(columns={"state": "set"})          # lesion.e3_analysis groups by "set"
    a = ea.h3a_tables(by_set[by_set.condition == "name_free"].assign(condition="none"), h3a_ids)
    prior = dg.frequency_prior(ea.candidate_zipf(items.loc[h3a_ids], zipf).to_numpy())
    obs = {"intact": ea.s_stat(a["intact"]), "K": ea.s_stat(a["K8"]), "C": ea.s_stat(a["C4"])}
    grid = ea.h3a_predicted_grid(a["intact"], prior)
    out["primary"]["H3a"] = ea.excess_test(obs, {"C": grid, "K": grid}, ce_a, args.n_boot, SEED, MDE_H3A)
    hint = by_set[by_set.condition.isin(("name_free", "cue", "miscue"))].replace({"condition": {"name_free": "none"}})
    b = ea.h3b_tables(hint, h3b_ids)
    prior_b = dg.frequency_prior(ea.candidate_zipf(items.loc[h3b_ids], zipf, ea.H3B_ROLES).to_numpy())
    obs_b = {"intact": ea.r_stat(b["intact"]), "K": ea.r_stat(b["K8"]), "C": ea.r_stat(b["C4"])}
    grid_b = ea.h3b_predicted_grid(b["intact"], prior_b)
    out["primary"]["H3b"] = ea.excess_test(obs_b, {"C": grid_b, "K": grid_b}, ce_b, args.n_boot, SEED, MDE_H3B)
    adj = holm({k: v["p_one_sided"] for k, v in out["primary"].items()})
    for k, v in out["primary"].items():
        v["p_holm"] = adj[k]
        v["verdict"] = ("支持" + ("，存在但很小" if v.get("small") else "")) if v["supported"] and adj[k] < 0.05 else "证据不足"

    # --- secondary ----------------------------------------------------------------------------------------------
    pc = analyze_battery.per_concept(cand, labels)
    I = pc["intact"].reindex(test_ids)
    T = {s: pc[s].reindex(test_ids) for s in pc}
    counts = ea.resample_counts(len(test_ids), args.n_boot, rng)
    naming_loss = {s: (I.name_target_logprob - T[s].name_target_logprob).to_numpy() for s in T}
    letter_loss = {s: (I.options_letter_margin - T[s].options_letter_margin).to_numpy() for s in T}
    w2d_loss = {s: (I.word_to_definition_margin - T[s].word_to_definition_margin).to_numpy() for s in T}
    s1a = naming_loss["K8"] - naming_loss["C4"]
    s1b = letter_loss["C4"] - letter_loss["K8"]
    S1a, S1b = summary(s1a, boot_mean(s1a, counts)), summary(s1b, boot_mean(s1b, counts))
    out["secondary"]["S1"] = dict(S1a_naming=S1a, S1b_six_choice=S1b, p_one_sided=max(S1a["p_one_sided"], S1b["p_one_sided"]),
                                  tiers=[tier(*S1a["ci"], MDE_S), tier(*S1b["ci"], MDE_S)],
                                  conditions_met=bool(S1a["ci"][0] > 0 and S1b["ci"][0] > 0))
    nk, ck = naming_loss["K8"], w2d_loss["K8"]
    bn, bc = boot_mean(nk, counts), boot_mean(ck, counts)
    S2n, S2c = summary(nk, bn, MDE_S), summary(ck, bc, NONINFERIORITY, side="less")
    out["secondary"]["S2"] = dict(naming_loss_K8=S2n, comprehension_loss_K8=S2c, noninferiority_bound=NONINFERIORITY,
                                  p_one_sided=max(S2n["p_one_sided"], S2c["p_one_sided"]),
                                  conditions_met=bool(S2n["ci"][0] > MDE_S and S2c["ci"][1] < NONINFERIORITY))
    pos = [test_ids.index(i) for i in h3b_ids]
    cols = ["delta_none_hinted_demos", "delta_cue", "delta_miscue"]
    Ii, Ta = I[cols].to_numpy()[pos], T["C4_answer"][cols].to_numpy()[pos]
    B = (Ta[:, 1] - Ta[:, 0]) - (Ii[:, 1] - Ii[:, 0])
    H = -((Ta[:, 2] - Ta[:, 0]) - (Ii[:, 2] - Ii[:, 0]))
    A = B - H
    cnt_b = ea.resample_counts(len(pos), args.n_boot, rng)
    S3 = summary(A, boot_mean(A, cnt_b), 0.0, side="less")
    # A < 0 alone is only a relative asymmetry (B = -1, H = 0 would pass); user decision Q22 (2026-10-08): S3 also
    # needs the miscue harm itself to rise, H > 0. Both must hold, so p is the larger of the two (intersection-union).
    S3H = summary(H, boot_mean(H, cnt_b))
    out["secondary"]["S3"] = dict(A_C4_answer=S3, B=summary(B, boot_mean(B, cnt_b)), H=S3H,
                                  p_one_sided=max(S3["p_one_sided"], S3H["p_one_sided"]),
                                  conditions_met=bool(S3["ci"][1] < 0 and S3H["ci"][0] > 0))
    dla_path = Path(args.dla) if args.dla else d / "dla.parquet"
    dla = pd.read_parquet(dla_path)
    dla = dla[dla.component.isin(rdt.K8)]                 # K8 only (a DLA file may hold other sets too)
    spec = dla.groupby("item_id").specific.sum().reindex(test_ids).to_numpy()
    S4 = summary(spec, boot_mean(spec, counts))
    out["secondary"]["S4"] = dict(direct_specific_K8=S4, p_one_sided=S4["p_one_sided"], conditions_met=bool(S4["ci"][0] > 0))
    # S5: knowledge-lesion errors stay in the category: the associate's share rises less than general weakening
    # predicts (K30, the larger knowledge lesion; K8 reported). All 160 test concepts: the H3a exclusion belongs to
    # H3a only (agent2 2026-10-08).
    a5 = ea.h3a_tables(by_set[by_set.condition == "name_free"].assign(condition="none"), test_ids)
    prior5 = dg.frequency_prior(ea.candidate_zipf(items.loc[test_ids], zipf).to_numpy())
    s5 = ea.candidate_excess_test(a5["intact"], a5["K30"], prior5, ce_a["K30"], "associate", args.n_boot, SEED)
    out["secondary"]["S5"] = dict(associate_excess_K30=s5, p_one_sided=s5["p_one_sided"], conditions_met=bool(s5["excess_ci"][1] < 0))
    out["report"]["S5_K8"] = ea.candidate_excess_test(a5["intact"], a5["K8"], prior5, ce_a["K"], "associate", args.n_boot, SEED)
    for r in ("coordinate", "superordinate"):
        out["report"][f"S5_K30_{r}"] = ea.candidate_excess_test(a5["intact"], a5["K30"], prior5, ce_a["K30"], r, args.n_boot, SEED)

    # S6: under K8, hint reliance (R change) grows with the concept's own naming loss (target only)
    loss_b = naming_loss["K8"][pos]
    dR = ea.r_stat(b["K8"]) - ea.r_stat(b["intact"])
    rho = lambda x, y: float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])
    draws = rng.integers(0, len(pos), (args.n_boot, len(pos)))
    rb = np.array([rho(loss_b[i], dR[i]) for i in draws])
    out["secondary"]["S6"] = dict(spearman=round(rho(loss_b, dR), 4), ci=[float(np.nanquantile(rb, q)) for q in (0.025, 0.975)],
                                  p_one_sided=float(((rb <= 0).sum() + 1) / (args.n_boot + 1)), n=len(pos),
                                  conditions_met=bool(np.nanquantile(rb, 0.025) > 0))

    # S7: a shown associate pulls the answer more after the control lesion (context-intrusion test, Q21)
    has_intrusion = cand.condition.isin(["distract_assoc", "distract_unrel"]).any()
    if S7_PREREGISTERED and not has_intrusion:
        raise SystemExit("S7 is in the secondary family but the run has no intrusion scores; the family must not shrink")
    if has_intrusion:
        m = mb.intrusion_measures(cand)
        base = m.loc["intact"].reindex(test_ids)
        dc = (m.loc["C4"].reindex(test_ids) - base)
        S7 = summary(dc.pull_assoc, boot_mean(dc.pull_assoc.to_numpy(), counts))
        entry = dict(pull_assoc_change_C4=S7, p_one_sided=S7["p_one_sided"], conditions_met=bool(S7["ci"][0] > 0))
        (out["secondary"] if S7_PREREGISTERED else out["report"])["S7"] = entry
        for s in [x for x in ("K8", "C4_answer", "K30", "C16") if x in m.index.get_level_values(0)]:
            dd = m.loc[s].reindex(test_ids) - base
            out["report"][f"intrusion_{s}"] = dict(pull_assoc_change=round(float(dd.pull_assoc.mean()), 4),
                                                   pull_unrel_change=round(float(dd.pull_unrel.mean()), 4))
        out["report"]["intrusion_C4_associate_specific"] = round(float((dc.pull_assoc - dc.pull_unrel).mean()), 4)

    adj = holm({k: v["p_one_sided"] for k, v in out["secondary"].items()})
    for k, v in out["secondary"].items():
        v["p_holm"] = adj[k]
        v["verdict"] = "支持" if v["conditions_met"] and adj[k] < 0.05 else "证据不足"

    # --- report only --------------------------------------------------------------------------------------------
    rep = out["report"]
    for s in [x for x in ("K8", "C4", "C4_answer", "K30", "C16") if x in T]:
        rep[s] = dict(naming_loss=round(float(naming_loss[s].mean()), 4), six_choice_loss=round(float(letter_loss[s].mean()), 4),
                      comprehension_loss=round(float(w2d_loss[s].mean()), 4),
                      word_shown_loss=round(float((I.options_word_target_logprob - T[s].options_word_target_logprob).mean()), 4),
                      copy_loss=round(float((I.copy_target_logprob - T[s].copy_target_logprob).mean()), 4),
                      dS=round(float((T[s].name_S - I.name_S).reindex(h3a_ids).mean()), 4))
        Ts = T[s][cols].to_numpy()[pos]
        Bs = (Ts[:, 1] - Ts[:, 0]) - (Ii[:, 1] - Ii[:, 0])
        Hs = -((Ts[:, 2] - Ts[:, 0]) - (Ii[:, 2] - Ii[:, 0]))
        rep[s].update(B=round(float(Bs.mean()), 4), H=round(float(Hs.mean()), 4), A=round(float((Bs - Hs).mean()), 4),
                      evidence_weights_test_refit_descriptive=ea.evidence_weights_bootstrap(Ii, Ts, n_boot=2000, seed=SEED))
        rep[s]["corr_naming_loss_vs_comprehension_loss"] = round(float(pd.Series(naming_loss[s]).corr(pd.Series(w2d_loss[s]), method="spearman")), 3)
    rep["corr_direct_specific_vs_naming_loss_K8"] = round(float(pd.Series(spec).corr(pd.Series(naming_loss["K8"]), method="spearman")), 3)
    if (d / "greedy.parquet").exists():
        g = pd.read_parquet(d / "greedy.parquet")
        rep["greedy_target_rate"] = {s: round(float((x.kind == "target").mean()), 3) for s, x in g.groupby("state")}
    report_sensitivity(rep, items, test_ids, h3a_ids, h3b_ids, pos, cand, labels, I, T, b, counts, cnt_b, spec, d, args)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for fam in ("primary", "secondary"):
        for k, v in out[fam].items():
            print(f"{k}: {v['verdict']} (p_holm {v['p_holm']:.4f})")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
