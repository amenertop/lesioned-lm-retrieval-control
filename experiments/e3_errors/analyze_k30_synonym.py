"""P13 follow-up (exploratory, after agent2's recheck 2026-10-17): is K30's drop in the M1 synonym margins a general
shrinking of all margins, and is it frequency-graded as semantic degradation would predict?

    python experiments/e3_errors/analyze_k30_synonym.py [--out reports/explore/k30_synonym.json]

From session 14's P13 scores (outputs/e3_errors/qwen3-30b-a3b-instruct/breadth/m1.parquet), margins averaged over the
two option orders (lesion.damage.item_margin), for K30 and its five random sets:
- per task: the ratio of the mean lesioned margin to the mean intact margin, and the least-squares slope;
- the synonym damage expected if the synonym margins shrank by the same ratio as the other tasks' pooled margins;
- the synonym rare-minus-common damage (rare: the pair's min_freq below the M1 frequency median, as lesion.damage
  defines it), the frequency signature semantic dementia shows in synonym judgement (Hoffman et al. 2011);
with a cluster bootstrap over concept groups for K30.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import damage, runinfo, scoring  # noqa: E402

SCORES = ROOT / "outputs/e3_errors/qwen3-30b-a3b-instruct/breadth/m1.parquet"
N_BOOT, SEED = 2000, 0


def table(scores, items, state):
    base = damage.item_margin(scores[scores.state == "intact"])
    les = damage.item_margin(scores[scores.state == state])
    if set(base.index) != set(les.index):
        raise ValueError(f"{state} covers other items than intact")
    t = pd.DataFrame(dict(intact=base, lesioned=les.reindex(base.index))).join(items, how="left")
    if t.task.isna().any() or t.concept_group.isna().any():
        raise ValueError("items without labels")
    return t


def stats(t, fm):
    out = {}
    for task, g in t.groupby("task"):
        out[task] = dict(n=int(len(g)), ratio=float(g.lesioned.mean() / g.intact.mean()),
                         slope=float(np.polyfit(g.intact, g.lesioned, 1)[0]), damage=float((g.intact - g.lesioned).mean()))
    syn, other = t[t.task == "synonym"], t[t.task != "synonym"]
    ratio_other = other.lesioned.sum() / other.intact.sum()
    h = syn.intact - syn.lesioned
    rare = syn.min_freq < fm
    out["synonym_expected_from_other_tasks"] = float(((1 - ratio_other) * syn.intact).mean())
    out["synonym_rare_minus_common"] = float(h[rare].mean() - h[~rare].mean())
    out["n_synonym_rare"], out["n_synonym_common"] = int(rare.sum()), int((~rare).sum())
    return out


def bootstrap(t, fm):
    rng = np.random.default_rng(SEED)
    groups = t.concept_group.unique()
    rows = {g: t.index[t.concept_group == g] for g in groups}
    keys = ("synonym_ratio", "synonym_excess", "synonym_rare_minus_common")
    draws = {k: [] for k in keys}
    for _ in range(N_BOOT):
        b = t.loc[np.concatenate([rows[g] for g in rng.choice(groups, len(groups))])]
        s = stats(b, fm)
        if "synonym" not in s:
            continue
        draws["synonym_ratio"].append(s["synonym"]["ratio"])
        draws["synonym_excess"].append(s["synonym"]["damage"] - s["synonym_expected_from_other_tasks"])
        draws["synonym_rare_minus_common"].append(s["synonym_rare_minus_common"])
    return {k: dict(ci_low=float(np.nanpercentile(v, 2.5)), ci_high=float(np.nanpercentile(v, 97.5))) for k, v in draws.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports/explore/k30_synonym.json"))
    args = ap.parse_args()
    scores = pd.read_parquet(SCORES).astype({"item_id": str})
    all_items = scoring.load_materials()[0].astype({"item_id": str})
    fm = damage.frequency_median(all_items)
    items = all_items.set_index("item_id")[["task", "concept_group"]].join(
        pd.to_numeric(all_items.set_index("item_id").min_freq, errors="coerce"))
    states = ["K30"] + sorted(s for s in scores.state.unique() if s.startswith("K30_random"))
    res = {s: stats(table(scores, items, s), fm) for s in states}
    k30 = res["K30"]
    out = dict(stage="k30_synonym_check", evidence="exploratory, after the data were seen", analysis_commit=runinfo.commit(),
               inputs={SCORES.relative_to(ROOT).as_posix(): runinfo.sha256(SCORES),
                       (scoring.M1 / "items.csv").relative_to(ROOT).as_posix(): runinfo.sha256(scoring.M1 / "items.csv")},
               frequency_median=fm, n_boot=N_BOOT, seed=SEED, resampled="concept_group", states=res,
               K30_bootstrap=bootstrap(table(scores, items, "K30"), fm),
               K30_synonym_excess=k30["synonym"]["damage"] - k30["synonym_expected_from_other_tasks"],
               random_synonym_rare_minus_common=[res[s]["synonym_rare_minus_common"] for s in states[1:]])
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(K30=k30, bootstrap=out["K30_bootstrap"]), indent=1))


if __name__ == "__main__":
    main()
