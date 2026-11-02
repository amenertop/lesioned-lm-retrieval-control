"""H3c, cross-task item consistency on the 160 test concepts of experiment 3 (docs/plans/p8-clinical-alignment.md (b);
rule written and committed before this was first computed, 2026-10-12; the data were already unsealed, so the result
is exploratory). No GPU:

    python experiments/e3_errors/analyze_consistency.py [--dir outputs/e3_errors/<model>/confirm] [--out ...]

Per concept and lesion (K8, C4), the damage h (intact minus lesioned) on three semantic formats: naming (target
log-probability, no hint), word-to-definition and six-choice (log-odds log(p / (1 - p)) of the correct option, p from
the softmax over all option letters, averaged over the option orders). Consistency = mean of the three pairwise
Spearman correlations of h across concepts; D = consistency(K8) - consistency(C4), with a 95% percentile interval
from 10,000 bootstrap resamples of concepts (seed 0). Original prediction (SD-like): D > 0.
"""
import argparse
import json
import sys
from itertools import combinations
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
FORMATS = ("naming", "word_to_definition", "options_letter")
LESIONS = ("K8", "C4")


def correct_log_odds(labels):
    """Per state and concept, the log-odds of the correct option for each multiple-choice format, averaged over the
    option orders (columns: state, item_id, fmt, log_odds)."""
    cols = [c for c in labels.columns if c.startswith("logit_")]
    letters = [c[-1] for c in cols]
    z = labels[cols].to_numpy(dtype=float)
    present = ~np.isnan(z)                               # word-to-definition has two options, six-choice six
    z = np.where(present, z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    pc = p[np.arange(len(labels)), labels.correct.map(letters.index).to_numpy()]
    pc = np.clip(pc, 1e-12, 1 - 1e-12)
    out = labels[["state", "item_id", "fmt"]].assign(log_odds=np.log(pc) - np.log1p(-pc))
    return out.groupby(["state", "item_id", "fmt"]).log_odds.mean().reset_index()


def damage(cand, labels, ids, lesions=LESIONS):
    """{lesion: DataFrame (concept x format) of h}."""
    pc = analyze_battery.per_concept(cand, labels)
    lo = correct_log_odds(labels).pivot_table(index=["state", "item_id"], columns="fmt", values="log_odds")
    out = {}
    for s in lesions:
        h = pd.DataFrame(index=ids)
        h["naming"] = (pc["intact"].name_target_logprob - pc[s].name_target_logprob).reindex(ids)
        for f in FORMATS[1:]:
            h[f] = (lo.loc["intact"][f] - lo.loc[s][f]).reindex(ids)
        if h.isna().any().any():
            raise SystemExit(f"missing damage values for {s}")
        out[s] = h
    return out


def consistency(h):
    """Mean of the pairwise Spearman correlations across concepts (h: concept x format)."""
    r = h.rank().corr(method="pearson")
    return float(np.mean([r.loc[a, b] for a, b in combinations(FORMATS, 2)])), \
        {f"{a}~{b}": float(r.loc[a, b]) for a, b in combinations(FORMATS, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs/e3_errors" / MODEL / "confirm"))
    ap.add_argument("--out", default=str(ROOT / "reports/explore/qwen_h3c_consistency.json"))
    args = ap.parse_args(argv)
    d = Path(args.dir)
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    ids = list(items.item_id[items.role == "test"])
    h = damage(pd.read_parquet(d / "candidates.parquet"), pd.read_parquet(d / "labels.parquet"), ids)
    point = {s: consistency(h[s]) for s in LESIONS}
    rng = np.random.default_rng(SEED)
    boots = {s: [] for s in LESIONS}
    pairs = {s: {k: [] for k in point[s][1]} for s in LESIONS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(ids), len(ids))
        for s in LESIONS:
            m, pr = consistency(h[s].iloc[idx].reset_index(drop=True))
            boots[s].append(m)
            for k, v in pr.items():
                pairs[s][k].append(v)
    ci = lambda x: [round(float(q), 4) for q in np.nanquantile(np.asarray(x), (0.025, 0.975))]
    diff = np.asarray(boots["K8"]) - np.asarray(boots["C4"])
    D = point["K8"][0] - point["C4"][0]
    lo, hi = ci(diff)
    verdict = "与 SD 式预测一致" if lo > 0 else ("与预测相反" if hi < 0 else "不确定")
    res = dict(note="exploratory: rule fixed before computing (docs/plans/p8-clinical-alignment.md (b)); data unsealed in experiment 3",
               source=str(d), n_concepts=len(ids), formats=list(FORMATS),
               consistency={s: dict(mean=round(point[s][0], 4), ci=ci(boots[s]),
                                    pairs={k: dict(r=round(v, 4), ci=ci(pairs[s][k])) for k, v in point[s][1].items()})
                            for s in LESIONS},
               D_K8_minus_C4=dict(estimate=round(D, 4), ci=[lo, hi], verdict=verdict))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
