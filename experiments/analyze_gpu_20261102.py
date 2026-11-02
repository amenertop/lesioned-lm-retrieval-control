"""Analysis of the GPU session of 2026-11-02 (exploratory; no model, CPU only).

    python experiments/analyze_gpu_20261102.py [--parts generated,heads,demo,phi]

generated  Greedy answers on the 240 held-out naming concepts (main model: intact, WR8, SC4; Phi: intact, PR6, PC4),
           graded case-insensitively (first word, trailing punctuation removed, a plural or singular form counts).
           Accuracy loss per concept (intact minus lesioned), rare (Zipf below the screen median 2.9999) minus common.
heads      The answer-position lesion of SC4 (main model) or PC4 (Phi) against matched random four-head sets on the
           hinted naming prompts: cue benefit change B, miscue harm change H and H - B per state, as
           analyze_optogenetics.losses; the observed set's rank among the random sets.
demo       The demonstration-format check (run_demo_format.py --also-existing --capitalized): naming loss of WR8 and
           SC4 in the four cells, with the family cells also scored case-insensitively, log(P(word) + P(Word)); the
           reading rules of docs/plans/demo-format-check.md.
phi        Phi on the clinical battery: the eight reading rules of docs/plans/phi-battery.md, on the concepts the
           intact Phi passes by the main model's validity rules and on all 160; the output-bias check.

Intervals: 95% percentile intervals over 10,000 concept resamples (seed 0). Writes
reports/explore/2026-11-02-gpu-session/<part>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))
from lesion import knowledge_items as ki, m3_items, m3_scoring as ms  # noqa: E402

OUT = ROOT / "reports/explore/2026-11-02-gpu-session"
QWEN = "qwen3-30b-a3b-instruct"
PHI = "phi35-moe-instruct"
PHI_DIR = ROOT / "outputs/second_model" / PHI / "battery"
ZIPF_MEDIAN = 2.9999000000000002        # knowledge screen of both models (params.zipf_median)
N_BOOT, SEED = 10_000, 0
NONINFERIORITY = 0.73                   # H2b margin (logits)
EQUIVALENCE = 0.20                      # interaction bounds and negligible copy loss


# --- helpers ----------------------------------------------------------------------------------------------------------

def boot(x, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if not len(x):
        return dict(mean=float("nan"), ci=[float("nan"), float("nan")], n=0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), ci=[float(q) for q in np.quantile(means, (0.025, 0.975))], n=int(len(x)))


def boot_groups(a, b, n_boot=N_BOOT, seed=SEED):
    """mean(a) - mean(b) for two independent groups of concepts, each resampled within itself."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1) - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    return dict(mean=float(a.mean() - b.mean()), ci=[float(q) for q in np.quantile(d, (0.025, 0.975))],
                n=[int(len(a)), int(len(b))])


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(value, indent=1, default=float), encoding="utf-8")
    print(f"written {OUT / (name + '.json')}")


def first_word(answer):
    words = str(answer).strip().lower().split()
    return words[0].strip(".,;:!?\"'*") if words else ""


def forms(x):
    x = x.lower()
    return {x, x + "s", x + "es", x[:-1] if x.endswith("s") else x, x[:-2] if x.endswith("es") else x}


# --- generated answers ------------------------------------------------------------------------------------------------

def generated(table, lesions):
    items = pd.read_csv(ki.OUT / "items.csv")
    test = items[items.split == "test"].set_index("item_id")
    table = table.copy()
    table["correct"] = [first_word(a) in forms(t) for a, t in zip(table.answer, table.target)]
    wide = table.pivot(index="item_id", columns="state", values="correct").astype(float)
    rare = test.loc[wide.index, "zipf"] < ZIPF_MEDIAN
    out = dict(n_concepts=int(len(wide)), n_rare=int(rare.sum()), zipf_median=ZIPF_MEDIAN,
               accuracy={s: dict(all=float(wide[s].mean()), rare=float(wide.loc[rare, s].mean()),
                                 common=float(wide.loc[~rare, s].mean())) for s in wide},
               frequency_effect={}, examples={})
    for s in lesions:
        loss = wide["intact"] - wide[s]
        out["frequency_effect"][s] = boot_groups(loss[rare], loss[~rare])
        changed = table[(table.state == s)].set_index("item_id").loc[wide.index[(wide["intact"] == 1) & (wide[s] == 0)]]
        out["examples"][s] = changed[["target", "answer"]].head(12).to_dict("records")
    a, b = lesions
    la, lb = wide["intact"] - wide[a], wide["intact"] - wide[b]
    diff = (la - lb)
    out[f"frequency_effect_{a}_minus_{b}"] = boot_groups(diff[rare], diff[~rare])
    return out


# --- hinted naming: cue benefit and miscue harm ----------------------------------------------------------------------

def hint_deltas(cand):
    """Per state and concept: target minus associate log-probability in each hinted condition."""
    g = cand.pivot_table(index=["state", "item_id", "condition"], columns="role", values="logprob")
    d = (g["target"] - g["associate"]).rename("delta").reset_index()
    return d.pivot_table(index=["state", "item_id"], columns="condition", values="delta")


def hint_losses(d, state):
    I, T = d.loc["intact"], d.loc[state]
    B = (T.cue - T.none_hinted_demos) - (I.cue - I.none_hinted_demos)
    H = -((T.miscue - T.none_hinted_demos) - (I.miscue - I.none_hinted_demos))
    return pd.DataFrame(dict(B=B, H=H, H_minus_B=H - B))


def hint_concepts():
    """The 100 test concepts of the confirmatory hint measures (analyze_optogenetics: free_naming is true)."""
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    test = final[final.role == "test"]
    return set(test.item_id[test.free_naming.astype(str).str.lower() == "true"])


def heads(cand, observed, random_prefix):
    keep = hint_concepts()
    cand = cand[cand.item_id.isin(keep)]
    d = hint_deltas(cand)
    states = sorted(set(d.index.get_level_values(0)) - {"intact"})
    rnd = [s for s in states if s.startswith(random_prefix)]
    obs = hint_losses(d, observed)
    means = {s: hint_losses(d, s).mean().to_dict() for s in states}
    out = dict(observed=observed, n_random=len(rnd), n_concepts=int(len(obs)),
               observed_stats={k: boot(obs[k]) for k in obs}, random={})
    for k in ("B", "H", "H_minus_B"):
        values = np.array([means[s][k] for s in rnd])
        o = means[observed][k]
        out["random"][k] = dict(observed=o, random_median=float(np.median(values)), random_max=float(values.max()),
                                random_min=float(values.min()), n_random_at_least_observed=int((values >= o).sum()),
                                p_upper=float((1 + (values >= o).sum()) / (len(values) + 1)))
    return out


# --- demonstration format --------------------------------------------------------------------------------------------

def demo(scores):
    s = scores.groupby(["item_set", "format", "state", "item_id", "condition", "role"], as_index=False).logprob.mean()
    out = {}
    for fmt in ("demos", "no_demos"):
        lower = s[(s.item_set == "battery") & (s.format == fmt) & (s.role == "target")]
        if len(lower):
            w = lower.pivot(index="item_id", columns="state", values="logprob")
            out[f"battery/{fmt}"] = {st: boot(w.intact - w[st]) for st in ("WR8", "SC4")}
        fam = s[(s.item_set == "family") & (s.format == fmt) & (s.role == "target")]
        cap = s[(s.item_set == "family_cap") & (s.format == fmt) & (s.role == "target_cap")]
        if len(fam):
            w = fam.pivot(index="item_id", columns="state", values="logprob")
            out[f"family/{fmt}"] = {st: boot(w.intact - w[st]) for st in ("WR8", "SC4")}
            if len(cap):
                c = cap.pivot(index="item_id", columns="state", values="logprob").loc[w.index]
                ci = np.logaddexp(w, c)
                share = np.exp(c - ci)
                out[f"family/{fmt}/case_insensitive"] = {st: boot(ci.intact - ci[st]) for st in ("WR8", "SC4")}
                out[f"family/{fmt}/capitalized_share"] = {st: float(share[st].mean()) for st in share}
    rules = {}
    for label, fam_key in (("lowercase", "family/demos"), ("case_insensitive", "family/demos/case_insensitive")):
        if "battery/no_demos" in out and fam_key in out:
            b, f = out["battery/no_demos"]["SC4"], out[fam_key]["SC4"]
            fmt_rule = (b["ci"][0] > 0.20 and b["ci"][0] <= 0.78 <= b["ci"][1]) and f["ci"][1] < 0.20
            concept_rule = b["ci"][1] < 0.20 and not (f["ci"][1] < 0.20)
            rules[label] = dict(battery_no_demos_SC4=b, family_demos_SC4=f,
                                verdict="format" if fmt_rule else "concepts" if concept_rule else "in between")
    out["reading_rules"] = rules
    return out


# --- Phi battery -------------------------------------------------------------------------------------------------------

def phi_tables():
    cand, labels, greedy, lm = [], [], [], {}
    for d in sorted((PHI_DIR / "battery").iterdir()):
        if not (d / "complete.json").exists():
            continue
        cand.append(pd.read_parquet(d / "candidates.parquet"))
        labels.append(pd.read_parquet(d / "labels.parquet"))
        if (d / "greedy.parquet").exists():
            greedy.append(pd.read_parquet(d / "greedy.parquet"))
        lm[d.name] = json.loads((d / "complete.json").read_text(encoding="utf-8"))["lm_loss"]
    return pd.concat(cand), pd.concat(labels), (pd.concat(greedy) if greedy else None), lm


def p_target(labels):
    """Six-option choice: mean probability of the correct letter over the six rotations, per state and concept."""
    t = labels[labels.fmt == "options_letter"].copy()
    lg = t[[f"logit_{l}" for l in ms.LETTERS]].to_numpy(float)
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    t["p"] = p[np.arange(len(t)), [ms.LETTERS.index(c) for c in t.correct]]
    return t.groupby(["state", "item_id"]).p.mean()


def phi():
    import analyze_battery

    cand, labels, greedy, lm = phi_tables()
    pc = analyze_battery.per_concept(cand, labels)
    pt = p_target(labels)
    I = pc["intact"]
    valid = (pt.loc["intact"].reindex(I.index) >= ms.MIN_P_TARGET) & (I.delta_none > 0)
    fmts = dict(naming="name_target_logprob", word_shown="options_word_target_logprob", copy="copy_target_logprob",
                six_choice="options_letter_margin", comprehension="word_to_definition_margin")

    def losses(state, idx):
        T = pc[state]
        L = pd.DataFrame({k: I[c] - T[c] for k, c in fmts.items()}).loc[idx]
        if "delta_cue" in T:
            base_t, base_i = T.delta_none_hinted_demos, I.delta_none_hinted_demos
            L["B"] = ((T.delta_cue - base_t) - (I.delta_cue - base_i)).reindex(idx)
            L["H"] = (-((T.delta_miscue - base_t) - (I.delta_miscue - base_i))).reindex(idx)
            L["H_minus_B"] = L.H - L.B
        return L

    report = dict(n_concepts=int(len(I)), n_valid=int(valid.sum()),
                  intact=dict(six_choice_p_target=float(pt.loc["intact"].mean()),
                              naming_target_beats_associate=float((I.delta_none > 0).mean()),
                              choice_ok=float((pt.loc["intact"].reindex(I.index) >= ms.MIN_P_TARGET).mean())),
                  lm_loss_relative={s: (v / lm["intact"] - 1 if v is not None else None) for s, v in lm.items()},
                  subsets={})
    for subset, idx in (("valid", I.index[valid.to_numpy()]), ("all", I.index)):
        L = {s: losses(s, idx) for s in pc if s != "intact"}
        r = {s: {k: boot(v) for k, v in L[s].items()} for s in L}
        rules = {}
        if {"PR6", "PC4"} <= set(L):
            n = boot(L["PR6"].naming - L["PC4"].naming)
            c = boot(L["PC4"].six_choice - L["PR6"].six_choice)
            rules["R1_double_dissociation"] = dict(naming_PR6_minus_PC4=n, six_choice_PC4_minus_PR6=c,
                                                   consistent=n["ci"][0] > 0 and c["ci"][0] > 0)
            pn, pm = r["PR6"]["naming"], r["PR6"]["comprehension"]
            rules["R2_matching_spared"] = dict(naming=pn, comprehension=pm, margin=NONINFERIORITY,
                                               consistent=pn["ci"][0] > 0 and pm["ci"][1] < NONINFERIORITY)
        if "PR6_answer" in L:
            an, cp = r["PR6_answer"]["naming"], r["PR6_answer"]["copy"]
            rules["R3_answer_positions"] = dict(naming=an, copy=cp, consistent=an["ci"][0] > 0 and
                                                (cp["ci"][0] <= 0 <= cp["ci"][1] or abs(cp["mean"]) < EQUIVALENCE))
        if "PC4_answer" in L and "H_minus_B" in L["PC4_answer"]:
            hb = r["PC4_answer"]["H_minus_B"]
            rules["R4_miscue"] = dict(H=r["PC4_answer"]["H"], B=r["PC4_answer"]["B"], H_minus_B=hb, consistent=hb["ci"][0] > 0)
        if {"PR6@0.75", "PC4@0.75"} <= set(L):
            gn = boot(L["PR6"].naming - L["PR6@0.75"].naming)
            gc = boot(L["PC4"].six_choice - L["PC4@0.75"].six_choice)
            rules["R5_graded"] = dict(PR6_naming_full_minus_075=gn, PC4_six_choice_full_minus_075=gc,
                                      consistent=gn["ci"][0] > 0 and gc["ci"][0] > 0)
        if "PR6+PC4" in L:
            inter_n = boot(L["PR6+PC4"].naming - L["PR6"].naming - L["PC4"].naming)
            inter_c = boot(L["PR6+PC4"].six_choice - L["PR6"].six_choice - L["PC4"].six_choice)
            rules["R6_joint"] = dict(naming_interaction=inter_n, six_choice_interaction=inter_c,
                                     naming_additive=-EQUIVALENCE <= inter_n["ci"][0] and inter_n["ci"][1] <= EQUIVALENCE,
                                     six_choice_interacts=inter_c["ci"][0] > 0 or inter_c["ci"][1] < 0)
        report["subsets"][subset] = dict(n=int(len(idx)), losses=r, rules=rules)
    if greedy is not None:
        final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False).set_index("item_id")
        g = greedy[greedy.item_id.isin(final.index)]
        report["R8_error_types"] = {s: grp.kind.value_counts(normalize=True).round(4).to_dict() for s, grp in g.groupby("state")}
    gen = [pd.read_parquet(p) for p in sorted((PHI_DIR / "generated").glob("*/generated.parquet"))]
    if gen:
        report["R7_generated"] = generated(pd.concat(gen, ignore_index=True), ("PR6", "PC4"))
    rnd = [pd.read_parquet(p) for p in sorted((PHI_DIR / "random").glob("*/candidates.parquet"))]
    if rnd:
        report["R4_random_heads"] = heads(pd.concat(rnd, ignore_index=True), "PC4_answer", "randomPC4_")
    bias = sorted((PHI_DIR / "bias").glob("*/naming.parquet"))
    if bias:
        b = pd.concat([pd.read_parquet(p) for p in bias], ignore_index=True)
        items = pd.read_csv(ki.OUT / "items.csv")
        zipf = items[items.split == "test"].set_index("item_id").zipf
        out = {}
        for biased, grp in b.groupby("bias"):
            w = grp.pivot(index="item_id", columns="state", values="logprob")
            rare = zipf.loc[w.index] < ZIPF_MEDIAN
            out["with_bias" if biased else "without_bias"] = {
                s: dict(naming_loss=float((w.intact - w[s]).mean()),
                        frequency_effect=float((w.intact - w[s])[rare].mean() - (w.intact - w[s])[~rare].mean()))
                for s in ("PR6", "PC4")}
        report["output_bias_check"] = out
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="generated,heads,demo,phi")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    if "generated" in parts:
        path = ROOT / "outputs/e3_knowledge" / QWEN / "generated_frequency/generated.parquet"
        if path.exists():
            save("generated-frequency", generated(pd.read_parquet(path), ("WR8", "SC4")))
    if "heads" in parts:
        d = ROOT / "outputs/e3_errors" / QWEN / "random_answer_heads"
        files = sorted(d.glob("*.parquet")) if d.exists() else []
        if files:
            save("random-answer-heads", heads(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True),
                                              "SC4_answer", "random_SC4_a0_"))
    if "demo" in parts:
        path = ROOT / "outputs/e3_errors" / QWEN / "demo_format/scores.parquet"
        if path.exists():
            save("demo-format-case", demo(pd.read_parquet(path)))
    if "phi" in parts and (PHI_DIR / "battery").exists():
        save("phi-battery", phi())


if __name__ == "__main__":
    main()
