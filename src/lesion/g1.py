"""G1 analysis for Experiment 0 (docs/plans/pilot-protocol.md §5-7, ticket pilot 05).

Input: per-prompt label scores (lesion.scoring.score_label_prompts) for each model version, plus the
formatted M1 items. Output: version choice, the three G1 criteria with cluster-bootstrap intervals,
the G1 decision, and the report-only results.
"""
import numpy as np
import pandas as pd
from scipy import stats

THRESHOLD = 0.20          # logit, minimum meaningful TDS effect
ACC_MIN, ACC_CI_MIN = 0.70, 0.50
CONSISTENCY_MIN = 0.90
STRING_ACC_MIN, STRING_CONSISTENCY_MIN = 0.80, 0.90
TIE = 0.02
N_BOOT, SEED = 10_000, 0
MATCHING = ("colour", "size")


# --- per-item table ---------------------------------------------------------------------------

def item_table(label_scores, items):
    """One row per item: mean logit difference over the two orders, accuracy, and order consistency."""
    s = label_scores.copy()
    s["diff"] = s.logit_correct - s.logit_wrong
    s["correct"] = (s.choice == s.correct_label).astype(float)
    # logit of the label that points to option1 minus the one that points to option2 (for synonym: similar - different)
    s["opt1_minus_opt2"] = np.where(s.option_A == "option1", s.logit_A - s.logit_B, s.logit_B - s.logit_A)
    g = s.groupby("item_id")
    tab = pd.DataFrame({
        "d": g["diff"].mean(),
        "acc": g["correct"].mean(),
        "consistent": g["chosen_option"].nunique().eq(1).astype(float),
        "opt1_minus_opt2": g["opt1_minus_opt2"].mean(),
        "n_orders": g.size(),
    }).reset_index()
    assert (tab.n_orders == 2).all(), "every item needs both option orders"
    meta = items.set_index("item_id")[["task", "concept_group", "tds", "min_freq", "high_demand", "young_acc",
                                       "synonym_type", "spelling_rule"]]
    tab = tab.join(meta, on="item_id")
    for col in ("tds", "min_freq", "young_acc"):
        tab[col] = pd.to_numeric(tab[col], errors="coerce")
    return tab


def tds_gap(items):
    """Mean TDS of the low-demand half minus the high-demand half, colour and size pooled (0.615 for M1)."""
    fm = items[items.task.isin(MATCHING)]
    tds = pd.to_numeric(fm.tds)
    return tds[fm.high_demand == "0"].mean() - tds[fm.high_demand == "1"].mean()


# --- statistics -------------------------------------------------------------------------------

def cluster_bootstrap(tab, stat, n=N_BOOT, seed=SEED, group="concept_group"):
    """Point estimate and 95% percentile interval, resampling concept groups with replacement."""
    rng = np.random.default_rng(seed)
    groups = tab[group].to_numpy()
    uniq = np.unique(groups)
    rows_of = [np.flatnonzero(groups == u) for u in uniq]
    point = stat(tab)
    samples = np.empty(n)
    for b in range(n):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        samples[b] = stat(tab.iloc[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return dict(estimate=float(point), ci_low=float(lo), ci_high=float(hi))


def tds_effect_stat(gap):
    """E = slope of d on TDS (task as covariate) x gap."""
    def stat(t):
        X = np.column_stack([np.ones(len(t)), t.tds.to_numpy(), (t.task == "size").to_numpy(float)])
        beta = np.linalg.lstsq(X, t.d.to_numpy(), rcond=None)[0]
        return beta[1] * gap
    return stat


def matching_measures(tab, gap, n=N_BOOT):
    fm = tab[tab.task.isin(MATCHING)]
    hd = fm[fm.high_demand == "1"]
    return dict(
        tds_effect=cluster_bootstrap(fm, tds_effect_stat(gap), n),
        high_demand_accuracy=cluster_bootstrap(hd, lambda t: t.acc.mean(), n),
        order_consistency=float(fm.consistent.mean()),
        accuracy_by_task={task: float(fm[fm.task == task].acc.mean()) for task in MATCHING},
    )


def g1_decision(m):
    e, acc = m["tds_effect"], m["high_demand_accuracy"]
    measurement_ok = (acc["estimate"] >= ACC_MIN and acc["ci_low"] > ACC_CI_MIN
                      and m["order_consistency"] >= CONSISTENCY_MIN)
    if not measurement_ok:
        return "measurement_fail"
    if e["ci_low"] > THRESHOLD:
        return "pass"
    if e["ci_high"] < THRESHOLD:
        return "exclude"
    return "insufficient"


# --- version choice (string items only; pilot protocol §5.4) ------------------------------------

def string_measures(tab):
    st = tab[tab.task == "string"]
    return dict(accuracy=float(st.acc.mean()), order_consistency=float(st.consistent.mean()))


def select_version(string_by_version, preferred="olmoe-0924-base"):
    """Choose the version that best understands the two-choice format, using string items only,
    so the choice is independent of the attribute items that G1 is judged on."""
    ok = {v: m for v, m in string_by_version.items()
          if m["accuracy"] >= STRING_ACC_MIN and m["order_consistency"] >= STRING_CONSISTENCY_MIN}
    pool = ok or string_by_version
    ranked = sorted(pool, key=lambda v: pool[v]["order_consistency"], reverse=True)
    best = ranked[0]
    if len(ranked) > 1:
        first, second = ranked[0], ranked[1]
        a, b = pool[first], pool[second]
        if abs(a["order_consistency"] - b["order_consistency"]) < TIE:          # consistency tied
            if abs(a["accuracy"] - b["accuracy"]) >= TIE:                        # higher string accuracy wins
                best = first if a["accuracy"] > b["accuracy"] else second
            elif preferred in (first, second):                                   # still tied: base
                best = preferred
    return best, bool(ok)


# --- report-only results (pilot protocol §7) ------------------------------------------------------

def synonym_frequency_effect(tab, n=N_BOOT):
    """Bias-corrected synonym score regressed on standardised min frequency, with item type as covariate."""
    syn = tab[tab.task == "synonym"].copy()
    bias = syn.opt1_minus_opt2.mean()
    sign = np.where(syn.synonym_type == "similar", 1.0, -1.0)
    syn["d_corrected"] = sign * (syn.opt1_minus_opt2 - bias)

    def stat(t):
        z = (t.min_freq - t.min_freq.mean()) / t.min_freq.std()
        X = np.column_stack([np.ones(len(t)), z.to_numpy(), (t.synonym_type == "similar").to_numpy(float)])
        return np.linalg.lstsq(X, t.d_corrected.to_numpy(), rcond=None)[0][1]

    out = cluster_bootstrap(syn, stat, n)
    out["similar_bias"] = float(bias)
    out["accuracy"] = float(syn.acc.mean())
    return out


def human_agreement(tab):
    res = {}
    for task in (*MATCHING, "synonym", "string"):
        t = tab[(tab.task == task) & tab.young_acc.notna()]
        rho, p = stats.spearmanr(t.d, t.young_acc)
        res[task] = dict(spearman=float(rho), p=float(p), n=int(len(t)))
    return res


def cloze_tds_effect(cloze_scores, items, gap, n=N_BOOT):
    wide = cloze_scores.pivot_table(index="item_id", columns="option", values="logprob").reset_index()
    meta = items.set_index("item_id")[["task", "concept_group", "tds", "correct_option"]]
    wide = wide.join(meta, on="item_id")
    wide["tds"] = pd.to_numeric(wide.tds)
    wide["d"] = np.where(wide.correct_option == "option1", wide.option1 - wide.option2, wide.option2 - wide.option1)
    return cluster_bootstrap(wide, tds_effect_stat(gap), n)


# --- whole analysis ---------------------------------------------------------------------------------

def analyze(label_scores_by_version, items, cloze_by_version=None, n=N_BOOT):
    gap = tds_gap(items)
    tabs = {v: item_table(s, items) for v, s in label_scores_by_version.items()}
    string_by_version = {v: string_measures(t) for v, t in tabs.items()}
    chosen, format_ok = select_version(string_by_version)
    per_version = {}
    for v, t in tabs.items():
        m = matching_measures(t, gap, n)
        per_version[v] = dict(
            string=string_by_version[v], matching=m, decision_if_chosen=g1_decision(m),
            synonym_frequency=synonym_frequency_effect(t, n), human_agreement=human_agreement(t),
            size_dimension_kept=m["accuracy_by_task"]["size"] >= ACC_MIN,
            cloze_tds_effect=cloze_tds_effect(cloze_by_version[v], items, gap, n) if cloze_by_version and v in cloze_by_version else None,
        )
    return dict(tds_gap=float(gap), chosen_version=chosen, string_format_check_passed=format_ok,
                g1_decision=per_version[chosen]["decision_if_chosen"], versions=per_version)
