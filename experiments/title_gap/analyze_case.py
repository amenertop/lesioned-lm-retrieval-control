"""Case-insensitive family naming scores (exploratory correction, 2026-11-02; experiments/title_gap/rescore_case.py).

The family naming prompts do not ask for lowercase, and the intact model capitalizes most of its family answers, so a
lesion that changes the case changes the lowercase score of the target without any change in naming. rescore_case.py
scored every candidate again with its first letter capitalized. This script

1. checks that the rescoring run reproduces the stored lowercase scores (states named in its --check),
2. replaces the naming score of each target with log(P(word) + P(Word)) (logaddexp of the two scores),
3. recomputes every family statistic the paper cites from the naming score with the frozen library of the runs
   (outputs/title_gap_execution_20261019/fine_final_source/src): per-state naming loss, naming-definition and
   naming-synonym partial correlations, frequency slopes, the WR8-SC4 contrasts and the semantic-diversity slopes,
   once with the stored lowercase scores and once with the case-insensitive scores, side by side.

    python experiments/title_gap/analyze_case.py --rescore outputs/case_rescore --out reports/explore/2026-11-02-case-rescore

Point estimates for every state; 95% intervals (10,000 family resamples, seed 1910, as the frozen protocol) for the
states in CI_STATES. Writes case-rescore.json (ledger-readable) and states.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/title_gap_execution_20261019"
SNAPSHOT = BASE / "fine_final_source"
sys.path.insert(0, str(SNAPSHOT / "src"))
from lesion import semantic_diversity as sd, title_gap_analysis as ta  # noqa: E402

RUNS = {"fine_final_core": BASE / "fine_final_core", "relative_hub_final_core": BASE / "relative_hub_final_core"}
ITEMS = SNAPSHOT / "data/materials/title_gap/fine_final/items.csv"
NORMS = ROOT / "outputs/title_gap_sources/SemD.xlsx"
N_BOOT, SEED = 10_000, 1910
WEIGHTS = ("weight_q_L4_s0.5_seed1910", "weight_q_L4_s0.5_seed1911", "weight_q_L4_s1_seed1910", "weight_v_L4_s1_seed1910",
           "weight_gate_L28_s1_seed1910", "weight_up_L28_s0.5_seed1910", "weight_up_L28_s0.5_seed1911",
           "weight_gate_L44_s1_seed1910", "weight_q_L28_s1_seed1910")
CI_STATES = ("WR8_a0", "SC4_a0", "WR30_a0", "WR8_a0_SC4_a0", "WR8_a0_SC4_a0.5", *WEIGHTS,
             "WR8_a0_comprehension_s0_learned_hard_64", "comprehension_s0_learned_hard_64", "comprehension_s2_learned_soft",
             "hub_s2_learned_hard_4", "hub_s2_learned_soft", "broad_early_f1_a0.5_s0", "broad_middle_f1_a0.9_s0",
             "broad_middle_f1_a0.75_s0", "broad_middle_f1_a0.5_s0", "broad_late_f1_a0.75_s0", "broad_middle_f1_a0.25_s0",
             "broad_middle_f1_a0.1_s0", "PC8_all_a0", "hub_s0_learned_hard_64", "hub_s1_learned_hard_64", "hub_s2_learned_hard_64",
             "relative_hub_s0_learned_soft", "relative_hub_s1_learned_hard_64", "relative_hub_s0_learned_hard_64",
             "relative_hub_s2_learned_hard_64")
CONTRASTS = (("WR8_a0", "SC4_a0"), *((w, "SC4_a0") for w in WEIGHTS))


def fast_bootstrap(frame, statistic, n_boot=2000, seed=1910):
    """ta.bootstrap for frames with one row per family: the same random draws and the same resampled rows in the
    same order (groups sorted by family), without concatenating 120 one-row frames per draw."""
    if frame.family.duplicated().any():
        return _slow_bootstrap(frame, statistic, n_boot, seed)
    ordered = frame.sort_values("family", kind="stable").reset_index(drop=True)
    n = len(ordered)
    if n_boot == 0 or n < 4:
        return _slow_bootstrap(frame, statistic, n_boot, seed)
    rng = np.random.default_rng(seed)
    values = [statistic(ordered.iloc[rng.integers(n, size=n)].reset_index(drop=True)) for _ in range(n_boot)]
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) < 0.9 * n_boot:
        return dict(estimate=float(statistic(frame)), ci_low=None, ci_high=None, n_families=n, status="unstable")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return dict(estimate=float(statistic(frame)), ci_low=float(lo), ci_high=float(hi), n_families=n, status="estimated")


_slow_bootstrap = ta.bootstrap
ta.bootstrap = fast_bootstrap
METRICS = ("naming", "definition", "synonym", "naming_definition_partial_r", "naming_synonym_partial_r",
           "naming_frequency_slope")


def load_items():
    items = pd.read_csv(ITEMS, keep_default_na=False).query("split == 'test' and review == 'approved'")
    _, items = sd.coverage(items, sd.load_norms(NORMS))
    items["frequency"] = items.log_bnc_freq
    if len(items) != 120:
        raise SystemExit("final concept count changed")
    return items


def read(run, rescore):
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    names = [s["name"] for s in plan["states"]]
    done = [n for n in names if (rescore / n / "complete.json").exists()]
    if len(done) != len(names):
        raise SystemExit(f"{rescore}: {len(names) - len(done)} states not rescored")
    tables = {k: pd.concat([pd.read_parquet(run / n / f"{k}.parquet").assign(state=n) for n in names], ignore_index=True)
              for k in ("candidates", "labels", "generated")}
    cap = pd.concat([pd.read_parquet(rescore / n / "candidates_cap.parquet").assign(state=n) for n in names], ignore_index=True)
    checks = {}
    for n in names:
        path = rescore / n / "candidates_lower_check.parquet"
        if path.exists():
            new = pd.read_parquet(path)
            old = pd.read_parquet(run / n / "candidates.parquet")
            key = ["item_id", "condition", "role"]
            m = old.merge(new, on=key, suffixes=("_old", "_new"))
            if len(m) != len(old):
                raise SystemExit(f"{n}: reproduction rows differ")
            diff = (m.logprob_old - m.logprob_new).abs()
            checks[n] = dict(rows=len(m), max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()))
    return tables, cap, checks, names


def case_insensitive(candidates, cap):
    """The candidates table with each lowercase score replaced by logaddexp(lowercase, capitalized)."""
    c = cap.assign(role=cap.role.str.removesuffix("_cap"))[["state", "item_id", "condition", "role", "logprob"]]
    m = candidates.merge(c.rename(columns={"logprob": "logprob_cap"}), on=["state", "item_id", "condition", "role"],
                         how="left", validate="one_to_one")
    has = m.logprob_cap.notna()
    out = candidates.copy()
    out.loc[has.to_numpy(), "logprob"] = np.logaddexp(m.loc[has, "logprob"], m.loc[has, "logprob_cap"]).to_numpy()
    out["case_share_cap"] = np.where(has, np.exp(m.logprob_cap - np.logaddexp(m.logprob, m.logprob_cap)), np.nan)
    return out


def flatten(profile):
    row = {}
    for metric in METRICS:
        v = profile.get(metric)
        if v is not None:
            row[metric] = v["estimate"]
            if v.get("ci_low") is not None:
                row[metric + "_ci"] = [v["ci_low"], v["ci_high"]]
    return row


def _state_ci(args):
    tables, items, state = args
    sub = {k: v[v.state.isin(["intact", state])] for k, v in tables.items()}
    ci, _ = ta.fine_profiles(sub, items, n_boot=N_BOOT)
    return state, ci[state]


def _contrast(args):
    tables, items, a, b = args
    sub = {k: v[v.state.isin(["intact", a, b])] for k, v in tables.items()}
    _, frame = ta.fine_profiles(sub, items, n_boot=0)
    r = ta.fine_contrast(frame, a, b, N_BOOT)
    out = {k: dict(estimate=v["estimate"], ci=[v["ci_low"], v["ci_high"]]) for k, v in r.items()}
    if a == "WR8_a0":
        frames = sd.fine_frames(frame, a, b)
        out["semd"] = {task: sd.paired_regression(data, n_boot=N_BOOT) for task, data in frames.items()}
    return f"{a}:{b}", out


def analyse(tables, items, names, with_ci=True, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    point, _ = ta.fine_profiles(tables, items, n_boot=0)
    out = {s: flatten(point[s]) for s in point}
    contrasts = {}
    if not with_ci:
        return out, contrasts
    keep = [n for n in CI_STATES if n in names]
    pairs = [(a, b) for a, b in CONTRASTS if a in names and b in names]
    with ProcessPoolExecutor(workers) as ex:
        for state, summary in ex.map(_state_ci, [(tables, items, s) for s in keep]):
            out[state] = flatten(summary)
        for key, value in ex.map(_contrast, [(tables, items, a, b) for a, b in pairs]):
            contrasts[key] = value
    return out, contrasts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescore", type=Path, default=ROOT / "outputs/case_rescore")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/explore/2026-11-02-case-rescore")
    ap.add_argument("--lower-ci", action="store_true", help="also recompute the lowercase intervals (the ledger has them)")
    args = ap.parse_args()
    items = load_items()
    result = dict(n_boot=N_BOOT, seed=SEED, ci_states=list(CI_STATES), runs={})
    rows = []
    for run_name, run in RUNS.items():
        tables, cap, checks, names = read(run, args.rescore / run_name)
        ci_tables = dict(tables, candidates=case_insensitive(tables["candidates"], cap))
        naming = ci_tables["candidates"].query("condition == 'naming' and role == 'target'")
        intact = naming[naming.state == "intact"]
        lower, lower_c = analyse(tables, items, names, with_ci=args.lower_ci)
        insens, insens_c = analyse(ci_tables, items, names)
        share = naming.groupby("state").case_share_cap.mean()
        result["runs"][run_name] = dict(
            reproduction=checks,
            intact_capitalized_share=float(intact.case_share_cap.mean()),
            capitalized_share_by_state={s: float(v) for s, v in share.items()},
            states={s: dict(lowercase=lower[s], case_insensitive=insens[s]) for s in lower},
            contrasts=dict(lowercase=lower_c, case_insensitive=insens_c))
        for s in lower:
            rows.append(dict(run=run_name, state=s, capitalized_share=float(share.get(s, np.nan)),
                             **{f"lower_{k}": v for k, v in lower[s].items() if not k.endswith("_ci")},
                             **{f"ci_{k}": v for k, v in insens[s].items() if not k.endswith("_ci")}))
        print(run_name, "reproduction", json.dumps(checks), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "case-rescore.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out / "states.csv", index=False)
    print("written", args.out)


if __name__ == "__main__":
    main()
