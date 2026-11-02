"""Analysis of the optogenetics-inspired run (docs/plans/optogenetics-inspired-experiments.md §9.4; exploratory,
calibration concepts; no GPU):

    python experiments/e3_errors/analyze_optogenetics.py [--dir outputs/e3_errors/<model>/optogenetics] [--out ...]

Per concept, the loss of a state is intact minus lesioned on each readout: naming (target log-probability, no
hint), naming with the word shown, word copying (log-probabilities), six-choice and word-to-definition margins
(logits), and for the hint conditions the cue benefit change B, the miscue harm change H and the cue-dependence
change dR (restricted log-odds of target over associate, nat; baseline: no hint with hinted demonstrations, as in
experiment 3). Every mean comes with a 95% percentile interval from 10,000 bootstrap resamples of concepts, seed 0.
- O1: loss by set and alpha; shape cosine of the five-format loss profile against alpha 0; for alpha above 1 the
  naming change of rare and common words (median split on Zipf frequency); language-model loss per state;
- O5: interaction = loss(K8+C4) - loss(K8) - loss(C4);
- O4: 2 x 2 x 2 contrasts of C4 over the probe, middle and answer segments (main effects, two- and three-way
  interactions) and the context-segment check; K8 by segment;
- O2: rescue curve of K8 at the answer positions (alpha 0, 0.25, 0.5, 0.75, 1), B along it, and the across-concept
  Spearman correlation of B at alpha 0 with the naming rescued at alpha 0.5;
- O3: first the patch fidelity of the self-implants (|error| at most 0.05 nat, median at most 0.01, fixed before the
  run; a set that fails is not interpreted), then the implant effects on the donor's and the recipient's first token
  against the intact model, by set and donor (a context-transfer test: the donor's removed vectors written into
  the recipient's context, not a pure exchange of word meaning).
- O4 interactions get a verdict against an equivalence margin of 0.20: negligible (interval inside), non-additive
  (interval excludes 0) or undetermined.
Writes reports/explore/qwen_optogenetics.json (or --out) and prints a summary.
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

import analyze_battery  # noqa: E402
from lesion import m3_items  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
N_BOOT, SEED = 10_000, 0
FORMATS = {"naming": ("name_target_logprob", 1), "word_shown": ("options_word_target_logprob", 1),
           "copy": ("copy_target_logprob", 1), "six_choice": ("options_letter_margin", 1),
           "comprehension": ("word_to_definition_margin", 1)}
HINTS = ("B", "H", "dR")
EQUIVALENCE_MARGIN = 0.20      # O4: an interaction counts as negligible only if its interval lies inside +-0.20 (the SESOI of S1, S2)
FIDELITY_MAX, FIDELITY_MEDIAN = 0.05, 0.01   # O3: self-implant |log-probability error| (nat) allowed, fixed before the run
LM_LOSS_FLAG = 0.10            # O1: states whose neutral-text loss rises more than 10% over intact are not interpreted


def boot(x, n_boot=N_BOOT, seed=SEED):
    """Mean and 95% percentile interval over concepts (NaNs dropped)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return dict(mean=None, ci=[None, None], n=0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return dict(mean=round(float(x.mean()), 4), ci=[round(float(q), 4) for q in np.quantile(means, (0.025, 0.975))], n=len(x))


def losses(I, T):
    """Per-concept losses of state T against intact I (index item_id)."""
    out = pd.DataFrame(index=I.index)
    for name, (col, _) in FORMATS.items():
        out[name] = I[col] - T[col]
    base_t, base_i = T["delta_none_hinted_demos"], I["delta_none_hinted_demos"]
    out["B"] = (T["delta_cue"] - base_t) - (I["delta_cue"] - base_i)
    out["H"] = -((T["delta_miscue"] - base_t) - (I["delta_miscue"] - base_i))
    out["dR"] = (T["delta_cue"] - T["delta_miscue"]) - (I["delta_cue"] - I["delta_miscue"])
    return out


def summarise(L):
    return {m: boot(L[m]) for m in L.columns}


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return None if d == 0 else round(float(a @ b / d), 4)


def additivity(s):
    """Verdict on an interaction contrast: negligible (interval inside +-margin), non-additive (interval excludes
    0), or undetermined."""
    lo, hi = s["ci"]
    if lo is None:
        return "undetermined"
    if -EQUIVALENCE_MARGIN <= lo and hi <= EQUIVALENCE_MARGIN:
        return "negligible"
    return "non-additive" if lo > 0 or hi < 0 else "undetermined"


def spearman(x, y):
    s = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    return None if len(s) < 3 else round(float(s.rank().corr(method="pearson").iloc[0, 1]), 4)


def analyse(d, items):
    if not (d / "candidates.parquet").exists():          # an implant-only run (O3 version 2)
        o3 = implant_results(d)
        return {} if o3 is None else dict(O3=o3)
    cand, lab = pd.read_parquet(d / "candidates.parquet"), pd.read_parquet(d / "labels.parquet")
    info = cand.groupby("state")[["set", "alpha", "region"]].first()
    pc = analyze_battery.per_concept(cand.drop(columns=["set", "alpha", "region"]), lab.drop(columns=["set", "alpha", "region"]))
    ids = list(pc["intact"].index)
    I = pc["intact"].reindex(ids)
    L = {s: losses(I, pc[s].reindex(ids)) for s in pc if s != "intact"}
    out = dict(n_concepts=len(ids), states=sorted(L))

    # O1: dose and gain
    o1 = {}
    for k in ("K8", "C4", "randomK_00", "randomC_00"):
        names = {0.0: k} if k in L else {}
        names.update({float(info.at[s, "alpha"]): s for s in L if s.startswith(f"{k}@")})
        curve = {str(a): summarise(L[s]) for a, s in sorted(names.items())}
        prof0 = [curve["0.0"][f]["mean"] for f in FORMATS] if "0.0" in curve else None
        shape = {a: cosine([c[f]["mean"] for f in FORMATS], prof0) for a, c in curve.items() if prof0 and float(a) < 1}
        o1[k] = dict(curve=curve, shape_cosine_vs_alpha0=shape)
    zipf = items.set_index("item_id").zipf.reindex(ids).astype(float)
    rare = zipf <= zipf.median()
    gains = {}
    for s in L:
        if "@" in s and float(info.at[s, "alpha"]) > 1:
            gains[s] = dict(naming_rare=boot(L[s].naming[rare]), naming_common=boot(L[s].naming[~rare]))
    o1["gain_naming_by_frequency"] = gains
    f = d / "lm_loss.parquet"
    if f.exists():
        lm = pd.read_parquet(f).set_index("state").lm_loss
        base = float(lm["intact"])
        o1["lm_loss"] = {s: dict(lm_loss=round(float(v), 4), relative=round(float(v / base - 1), 4),
                                 flag_over_10pct=bool(v / base - 1 > LM_LOSS_FLAG)) for s, v in lm.items()}
    out["O1"] = o1

    # O5: interaction of the double lesion
    if all(s in L for s in ("K8+C4", "K8", "C4")):
        inter = L["K8+C4"] - L["K8"] - L["C4"]
        out["O5"] = dict(interaction=summarise(inter), double=summarise(L["K8+C4"]))

    # O4: segment contrasts
    seg = lambda k, g: L.get(f"{k}_seg_{g}")
    zero = L[next(iter(L))] * 0
    get = lambda g: zero if g == "" else seg("C4", g)
    names = ["probe", "middle", "answer", "probe+middle", "probe+answer", "middle+answer", "probe+middle+answer"]
    if all(seg("C4", g) is not None for g in names):
        p, m, a = get("probe"), get("middle"), get("answer")
        pm, pa, ma, pma = get("probe+middle"), get("probe+answer"), get("middle+answer"), get("probe+middle+answer")
        contrasts = {"probe_alone": p, "middle_alone": m, "answer_alone": a,
                     "interaction_probe_middle": pm - p - m, "interaction_probe_answer": pa - p - a,
                     "interaction_middle_answer": ma - m - a,
                     "interaction_three_way": pma - pm - pa - ma + p + m + a,
                     "sum_of_singles_minus_all_three": p + m + a - pma}
        o4 = {"C4": {k: summarise(v) for k, v in contrasts.items()}, "equivalence_margin": EQUIVALENCE_MARGIN}
        o4["C4_additivity"] = {k: {m: additivity(s) for m, s in o4["C4"][k].items()}
                               for k in o4["C4"] if k.startswith("interaction") or k.startswith("sum_of")}
        if seg("C4", "context") is not None and "C4" in L:
            o4["C4"]["context_alone"] = summarise(seg("C4", "context"))
            o4["C4"]["whole_minus_context_minus_three"] = summarise(L["C4"] - seg("C4", "context") - pma)
        o4["K8"] = {g: summarise(seg("K8", g)) for g in ("context", "probe", "middle", "answer") if seg("K8", g) is not None}
        out["O4"] = o4

    # O2: rescue curve at the answer positions
    curve = {"0.0": seg("K8", "answer")}
    curve.update({str(float(info.at[s, "alpha"])): L[s] for s in L if s.startswith("K8_answer@")})
    if curve["0.0"] is not None:
        o2 = dict(curve={a: dict(naming=boot(v.naming), B=boot(v.B), dR=boot(v.dR)) for a, v in sorted(curve.items())})
        if "0.5" in curve:
            o2["spearman_B_at_0_vs_naming_rescued_at_0.5"] = spearman(curve["0.0"].B, curve["0.0"].naming - curve["0.5"].naming)
        out["O2"] = o2

    o3 = implant_results(d)
    if o3 is not None:
        out["O3"] = o3
    for key in ("O2", "O4", "O5"):
        if key in out:
            out[key]["hint_population"] = dict(scope="report_only_available_concepts",
                                               n=int(L[next(iter(L))].B.notna().sum()),
                                               note="Not the preselected free-naming subset used by R1 to R6.")
    return out


def implant_results(d):
    """O3: patch fidelity of the self-implants, then the implant effects (None without implant.parquet)."""
    f = d / "implant.parquet"
    if f.exists():
        imp = pd.read_parquet(f)
        # pre-specified (protocol 9.3): a donor whose name starts with the recipient's first token cannot be told apart
        # by the first-token readout, so such near and far pairs are excluded; self rows share it by definition
        excluded = {f"{s}/{c}": dict(n_pairs=int(len(g)), n_excluded=int(g.same_first_token.sum()))
                    for (s, c), g in imp[imp.condition.isin(["near", "far"])].groupby(["set", "condition"])}
        imp = imp[~imp.same_first_token | imp.condition.isin(["self", "lesion"])]
        ref = imp[imp.set == "intact"].set_index(["condition", "recipient", "donor"])
        fidelity = {}
        for s in sorted(set(imp.set) - {"intact"}):
            g = imp[(imp.set == s) & (imp.condition == "self")]
            err = (g.lp_recipient.to_numpy() - ref.loc["self"].reindex(pd.MultiIndex.from_frame(g[["recipient", "donor"]])).lp_recipient.to_numpy())
            err = np.abs(err)
            fidelity[s] = dict(max=round(float(err.max()), 5), median=round(float(np.median(err)), 5),
                               p95=round(float(np.quantile(err, 0.95)), 5),
                               passed=bool(err.max() <= FIDELITY_MAX and np.median(err) <= FIDELITY_MEDIAN))
        o3 = {"patch_fidelity": fidelity, "excluded_same_first_token": excluded}
        for (s, c), g in imp[imp.set != "intact"].groupby(["set", "condition"]):
            key = "self" if c == "lesion" else c
            r = ref.loc[key].reindex(pd.MultiIndex.from_frame(g[["recipient", "donor"]]))
            o3[f"{s}/{c}"] = dict(interpretable=fidelity[s]["passed"],
                                  donor_gain=boot(g.lp_donor.to_numpy() - r.lp_donor.to_numpy()),
                                  recipient_change=boot(g.lp_recipient.to_numpy() - r.lp_recipient.to_numpy()),
                                  donor_top1_rate=round(float((g.rank_donor == 0).mean()), 4), n=len(g))
        return o3
    return None


# --- confirmatory rules on the sealed test concepts (docs/analysis-plan/optogenetics.md; P7, 2026-10-12) -----------

RULE_MARGIN = EQUIVALENCE_MARGIN   # 0.20: R2's marker-exclusion margin and R6's additivity margin (the SESOI of S1, S2)
DOSE_ALPHAS = (0.25, 0.5, 0.75)
RULE_STATES = dict(R1=("K8", "K8@0.75", "C4", "C4@0.75"),
                   R2=tuple(f"{k}@{a}" for k in ("K8", "C4") for a in DOSE_ALPHAS),
                   R3=("K8@2.0",), R4=("K8_seg_answer", "K8_answer@0.75"),
                   R5=("C4_seg_middle", "C4_seg_answer"), R6=("K8+C4", "K8", "C4"))
# the states whose neutral-text loss decides whether a rule is interpreted. A lesion of one segment has no
# counterpart on neutral text, so R4 and R5 use the whole-sequence lesion of the same set and alpha as a proxy.
# A whole-sequence lesion is not a proven upper bound: K8 and K8@0.75 for R4, C4 for R5 (user
# decision 2026-10-13 on agent3's N4). A missing loss is never read as "below 10%".
LM_STATES = dict(RULE_STATES, R4=("K8", "K8@0.75"), R5=("C4",))


def rule_summary(x, boots, threshold=0.0, side="greater"):
    """Mean, 95% percentile interval and one-sided p as analyze_experiment3.summary, but unrounded: the rules
    compare the interval with 0 and with the margin, and rounding first would move those thresholds (agent3
    2026-10-13, N7). Rounding is left to the display."""
    p = ((boots <= threshold).sum() + 1) / (len(boots) + 1) if side == "greater" else ((boots >= threshold).sum() + 1) / (len(boots) + 1)
    return dict(estimate=float(np.mean(x)), ci=[float(np.quantile(boots, q)) for q in (0.025, 0.975)],
                p_one_sided=float(p), n=int(len(x)))


def load_losses(d, ids):
    """Per-concept losses (losses()) of every state against intact, on the concepts ids; and the run's metadata
    per state (set, alpha, region)."""
    cand, lab = pd.read_parquet(d / "candidates.parquet"), pd.read_parquet(d / "labels.parquet")
    info = cand.groupby("state")[["set", "alpha", "region"]].first()
    pc = analyze_battery.per_concept(cand.drop(columns=["set", "alpha", "region"]), lab.drop(columns=["set", "alpha", "region"]))
    I = pc["intact"].reindex(ids)
    return {s: losses(I, pc[s].reindex(ids)) for s in pc if s != "intact"}, pc, info


def confirm(d, items, ids, n_boot=N_BOOT, seed=SEED):
    """R1 to R6 of the analysis plan on the test run in d: per-concept paired contrasts, bootstrap over concepts
    (10,000 resamples, seed 0; formats on the 160 test concepts, hint measures B, H, dR on the 100 free-naming
    concepts as in S3 and S6, each with its own resample counts drawn in that order), one-sided p-values, the larger
    p of a rule's parts (intersection-union), Holm over R1 to R6. A rule whose states (LM_STATES) raise the
    neutral-text loss by more than 10% is not interpreted. ids: the test concepts in item-file order
    (m3_optogenetics_plan.confirm_ids), never the order of a run's record."""
    test = items.set_index("item_id").loc[ids].reset_index()
    if not (test.role == "test").all():
        raise SystemExit("the confirmatory rules read test concepts only")
    hint_ids = list(test.item_id[test.free_naming.astype(str).str.lower() == "true"])
    L, pc, info = load_losses(d, ids)
    lm = pd.read_parquet(d / "lm_loss.parquet").set_index("state").lm_loss
    lm_rise = {s: float(lm[s] / lm["intact"] - 1) for s in lm.index}
    out = rules(L, ids, hint_ids, lm_rise, n_boot, seed)
    rep = reported(L, pc, info, test, ids)
    rep["O5_six_choice_p_correct_interaction"] = p_correct_interaction(pd.read_parquet(d / "labels.parquet"), ids)
    return dict(rules=out, n_concepts=len(ids), n_hint_concepts=len(hint_ids), reported=rep)


def rules(L, ids, hint_ids, lm_rise, n_boot=N_BOOT, seed=SEED):
    """R1 to R6 from per-concept losses (L: {state: DataFrame indexed by concept, columns of losses()}), with the
    hint measures on hint_ids; lm_rise: {state: relative rise of its language-model loss over intact}. Conditions
    are read on unrounded intervals. Separate from file reading so the rules can be tested on synthetic tables."""
    import analyze_experiment3 as ae3
    from lesion import e3_analysis as ea

    pos = [ids.index(i) for i in hint_ids]
    rng = np.random.default_rng(seed)
    counts, counts_hint = ea.resample_counts(len(ids), n_boot, rng), ea.resample_counts(len(pos), n_boot, rng)

    def stat(x, threshold=0.0, side="greater", hint=False):
        x = np.asarray(x, dtype=float)
        x = x[pos] if hint else x
        if np.isnan(x).any():
            raise ValueError("missing values in a rule statistic")
        return rule_summary(x, ae3.boot_mean(x, counts_hint if hint else counts), threshold, side)

    col = lambda s, m: L[s][m].reindex(ids).to_numpy()
    out = {}
    a, b = stat(col("K8", "naming") - col("K8@0.75", "naming")), stat(col("C4", "six_choice") - col("C4@0.75", "six_choice"))
    out["R1"] = dict(K8_naming_alpha0_minus_0_75=a, C4_six_choice_alpha0_minus_0_75=b,
                       p_one_sided=max(a["p_one_sided"], b["p_one_sided"]), conditions_met=bool(a["ci"][0] > 0 and b["ci"][0] > 0))
    parts, ok = {}, True
    for k, own, other in (("K8", "naming", "six_choice"), ("C4", "six_choice", "naming")):
        for al in DOSE_ALPHAS:
            s = f"{k}@{al}"
            o, x = stat(col(s, own)), stat(col(s, other), RULE_MARGIN, side="less")
            parts[f"{s}/{own}_loss_above_0"], parts[f"{s}/{other}_loss_below_margin"] = o, x
            ok = ok and o["ci"][0] > 0 and x["ci"][1] < RULE_MARGIN
    out["R2"] = dict(parts=parts, margin=RULE_MARGIN, p_one_sided=max(v["p_one_sided"] for v in parts.values()),
                       conditions_met=bool(ok))
    g = stat(col("K8@2.0", "dR"), 0.0, side="less", hint=True)
    out["R3"] = dict(K8_gain2_dR=g, p_one_sided=g["p_one_sided"], conditions_met=bool(g["ci"][1] < 0))
    n4 = stat(col("K8_seg_answer", "naming") - col("K8_answer@0.75", "naming"))
    b4 = stat(col("K8_seg_answer", "B") - col("K8_answer@0.75", "B"), hint=True)
    out["R4"] = dict(naming_alpha0_minus_0_75=n4, B_alpha0_minus_0_75=b4, p_one_sided=max(n4["p_one_sided"], b4["p_one_sided"]),
                       conditions_met=bool(n4["ci"][0] > 0 and b4["ci"][0] > 0))
    hm = stat(col("C4_seg_middle", "H"), 0.0, side="less", hint=True)
    ha = stat(col("C4_seg_answer", "H"), hint=True)
    out["R5"] = dict(H_middle_only=hm, H_answer_only=ha, p_one_sided=max(hm["p_one_sided"], ha["p_one_sided"]),
                       conditions_met=bool(hm["ci"][1] < 0 and ha["ci"][0] > 0))
    inter = lambda m: col("K8+C4", m) - col("K8", m) - col("C4", m)
    six = stat(inter("six_choice"))
    lo, hi = stat(inter("naming"), -RULE_MARGIN), stat(inter("naming"), RULE_MARGIN, side="less")
    out["R6"] = dict(six_choice_interaction=six, naming_interaction=hi | dict(p_one_sided_lower=lo["p_one_sided"]),
                       margin=RULE_MARGIN, p_one_sided=max(six["p_one_sided"], lo["p_one_sided"], hi["p_one_sided"]),
                       conditions_met=bool(six["ci"][0] > 0 and -RULE_MARGIN < hi["ci"][0] and hi["ci"][1] < RULE_MARGIN))

    adj = ae3.holm({k: v["p_one_sided"] for k, v in out.items()})
    for k, v in out.items():
        v["p_holm"] = adj[k]
        missing = [s for s in LM_STATES[k] if s not in lm_rise or not np.isfinite(lm_rise[s])]
        flagged = [s for s in LM_STATES[k] if s not in missing and lm_rise[s] > LM_LOSS_FLAG]
        v["lm_loss_states"], v["lm_loss_flagged_states"], v["lm_loss_missing_states"] = list(LM_STATES[k]), flagged, missing
        v["lm_loss_check"] = "whole_sequence_proxy" if k in ("R4", "R5") else "matched_state"
        v["verdict"] = ("无法检查（语言损失缺测）" if missing else "不解释（语言损失超过 10%）" if flagged
                        else "支持" if v["conditions_met"] and adj[k] < 0.05 else "证据不足")
    return out


def p_correct_interaction(labels, ids):
    """O5 on a bounded scale (agent2 2026-10-12): the six-choice probability of the correct letter (softmax over the
    six letter logits, mean over the six orders); loss = intact minus state; interaction as for the margins."""
    lab = labels[labels.fmt == "options_letter"]
    letters = list("ABCDEF")
    z = lab[[f"logit_{x}" for x in letters]].to_numpy(dtype=float)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    pr = z / z.sum(axis=1, keepdims=True)
    lab = lab.assign(p_correct=pr[np.arange(len(lab)), lab.correct.map(letters.index).to_numpy()])
    pc = lab.groupby(["state", "item_id"]).p_correct.mean().unstack(0).reindex(ids)
    if not all(s in pc for s in ("intact", "K8+C4", "K8", "C4")):
        return None
    loss = {s: pc["intact"] - pc[s] for s in ("K8+C4", "K8", "C4")}
    return dict(intact=boot(pc["intact"]), losses={s: boot(v) for s, v in loss.items()},
                interaction=boot(loss["K8+C4"] - loss["K8"] - loss["C4"]))


def reported(L, pc, info, test, ids):
    """Only-reported results (analysis plan §7): random single and double lesions with their interactions, the K8
    dose-by-frequency gradient (P8c), and the double lesion's interaction on a bounded scale (six-choice all-correct
    rate over the six orders)."""
    out = {}
    six = lambda s: L[s].six_choice
    rnd = {}
    for s in sorted(L):
        if s.startswith("K8+randomC_"):
            r = s.split("+")[1]
            rnd[s] = dict(double=boot(six(s)), single=boot(six(r)), interaction=boot(six(s) - six("K8") - six(r)))
        elif s.startswith("randomK_") and s.endswith("+C4"):
            r = s.split("+")[0]
            rnd[s] = dict(double=boot(six(s)), single=boot(six(r)), interaction=boot(six(s) - six(r) - six("C4")))
    out["random_double_six_choice"] = rnd
    zipf = test.set_index("item_id").zipf.reindex(ids).astype(float)
    rare = (zipf <= zipf.median()).to_numpy()
    grad = {}
    for s in ("K8@0.75", "K8@0.5", "K8@0.25", "K8"):
        if s in L:
            x = L[s].naming.to_numpy()
            grad[s] = dict(rare=boot(x[rare]), common=boot(x[~rare]), spearman_zipf=spearman(zipf.to_numpy(), x))
    out["K8_dose_by_frequency"] = grad
    if all(s in pc for s in ("intact", "K8+C4", "K8", "C4")):
        acc = {s: pc[s].reindex(ids).options_letter_all_correct for s in ("intact", "K8+C4", "K8", "C4")}
        loss = {s: acc["intact"] - acc[s] for s in ("K8+C4", "K8", "C4")}
        out["O5_six_choice_all_correct_interaction"] = boot(loss["K8+C4"] - loss["K8"] - loss["C4"])
    return out


def check_confirm_run(d, meta, smoke=False):
    """The confirmatory analysis reads only a complete test-role run of the frozen plan; the checks, against the
    plan and the files rather than the run's record alone, are lesion.m3_optogenetics_plan.confirm_problems
    (agent2 reviews 2026-10-12, -13)."""
    from lesion import m3_optogenetics_plan as op

    problems = op.confirm_problems(Path(d), meta, smoke=smoke)
    if problems:
        raise SystemExit("refusing the confirmatory analysis: " + "; ".join(problems))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--confirm", action="store_true", help="the frozen rules R1 to R6 on the test run (analysis plan)")
    ap.add_argument("--smoke", action="store_true", help="allow a tiny test-role run (pipeline check; output marked "
                                                         "as such; needs --out, never the confirmatory report)")
    args = ap.parse_args(argv)
    # only the confirmatory analysis of a real run writes into the report folder, however the path is spelt
    # (agent3 2026-10-13, N6, F7, G4)
    official = (ROOT / "reports/confirm/optogenetics").resolve()
    inside = args.out is not None and (Path(args.out).resolve() == official or official in Path(args.out).resolve().parents)
    if args.smoke and not args.confirm:
        raise SystemExit("--smoke goes only with --confirm")
    if args.smoke and not args.out:
        raise SystemExit("--smoke writes only where --out says")
    if inside and args.smoke:
        raise SystemExit(f"--smoke never writes into {official}, the confirmatory report's folder")
    if inside and not args.confirm:
        raise SystemExit(f"only the confirmatory analysis writes into {official}")
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    if args.confirm:
        from lesion import m3_optogenetics_plan as op, runinfo, state_store

        d = Path(args.dir or ROOT / "outputs/e3_errors" / MODEL / "optogenetics_confirm")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        check_confirm_run(d, meta, smoke=args.smoke)
        ids = op.confirm_ids(bool(meta.get("tiny")))            # item-file order, not the record's (agent3 N5)
        res = dict(note=("SMOKE TEST on a tiny random model, not a result" if meta.get("tiny") else
                         "confirmatory, sealed test concepts (docs/analysis-plan/optogenetics.md)"), source=str(d),
                   run_commit=meta.get("git_commit"), freeze_tag=meta.get("freeze_tag"), tiny=meta.get("tiny"),
                   analysis_commit=runinfo.commit(), source_tables_sha256=state_store.tables_sha256(d),
                   source_meta_sha256=runinfo.sha256(d / "meta.json"), **confirm(d, items, ids))
        res["exploratory"] = analyse(d, items)
        out = Path(args.out or ROOT / "reports/confirm/optogenetics/result.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        state_store.write_json(out, res)                      # atomic: a stop never leaves half a report
        for k, v in res["rules"].items():
            print(f"{k}: {v['verdict']} (p_holm {v['p_holm']:.4f})")
        print(f"written to {out}")
        return
    d = Path(args.dir or ROOT / "outputs/e3_errors" / MODEL / "optogenetics")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    res = dict(note="exploratory, calibration concepts (docs/plans/optogenetics-inspired-experiments.md §9.4)",
               source=str(d), run_commit=meta.get("git_commit"), implant_self_check=meta.get("implant_self_max_abs_diff"),
               **analyse(d, items))
    out = Path(args.out or ROOT / "reports/explore/qwen_optogenetics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    fmt = lambda s: f"{s['mean']} [{s['ci'][0]}, {s['ci'][1]}]" if s and s["mean"] is not None else "-"
    for k in ("K8", "C4"):
        for a, c in res.get("O1", {}).get(k, {}).get("curve", {}).items():
            print(f"O1 {k} alpha {a}: naming {fmt(c['naming'])}; six-choice {fmt(c['six_choice'])}; comprehension {fmt(c['comprehension'])}")
    if "O5" in res:
        print("O5 interaction:", {m: fmt(v) for m, v in res["O5"]["interaction"].items()})
    if "O4" in res:
        print("O4 C4:", {k: fmt(v["six_choice"]) for k, v in res["O4"]["C4"].items()})
    if "O3" in res:
        print("O3 patch fidelity:", res["O3"]["patch_fidelity"])
        print("O3 excluded (same first token):", res["O3"]["excluded_same_first_token"])
        print("O3:", {k: (fmt(v["donor_gain"]), v["donor_top1_rate"]) for k, v in res["O3"].items()
                      if k not in ("patch_fidelity", "excluded_same_first_token")})
    print(f"written to {out}")


if __name__ == "__main__":
    main()
