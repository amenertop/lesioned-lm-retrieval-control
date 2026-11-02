"""Exploratory (after unsealing): after the control-head lesion, do answers follow association strength?

Reads the held-out M1 scores the dissociation test already produced (no model, no GPU):

    python experiments/e2_confirm/explore_association.py [--dir outputs/e2_confirm/<model>/dissociation]

For each lesion set, on the colour and size items, damage h (intact minus lesioned margin) is regressed on the
standardised word2vec similarity of the correct option to the probe (w2v_target_probe) and of the distractor
to the probe (w2v_distractor_probe), task as covariate; a second fit also holds the intact margin fixed, since
a lesion that only widens every margin (lesioned = k * intact, k > 1) would by itself make confident items
improve and wrong ones worsen. Intervals: concept groups resampled. Writes association.json next to the scores.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import damage, materials, scoring  # noqa: E402


def frame(scores, lesion, items, assoc):
    intact = damage.item_margin(scores[scores.set == "intact"].drop(columns="set")).rename("m_intact")
    lesioned = damage.item_margin(scores[scores.set == lesion].drop(columns="set")).rename("m_lesioned")
    f = pd.concat([intact, lesioned], axis=1).rename_axis("item_id").reset_index()
    f = f.merge(items[["item_id", "task", "concept_group", "high_demand", "tds"]], on="item_id").merge(assoc, on="item_id")
    f = f[f.task.isin(damage.MATCHING)].reset_index(drop=True)
    f["h"] = f.m_intact - f.m_lesioned
    f["tds"] = pd.to_numeric(f.tds, errors="coerce")
    return f


def stats(f):
    z = lambda x: ((x - x.mean()) / x.std()).to_numpy()
    size = (f.task == "size").to_numpy(float)
    y = f.h.to_numpy()
    a = np.linalg.lstsq(np.column_stack([np.ones(len(f)), z(f.w2v_target_probe), z(f.w2v_distractor_probe), size]), y, rcond=None)[0]
    b = np.linalg.lstsq(np.column_stack([np.ones(len(f)), z(f.m_intact), z(f.w2v_target_probe), z(f.w2v_distractor_probe), size]),
                        y, rcond=None)[0]
    k = np.linalg.lstsq(np.column_stack([np.ones(len(f)), f.m_intact]), f.m_lesioned, rcond=None)[0][1]
    easy = f.tds >= f.tds.quantile(2 / 3)
    return dict(target_coef=a[1], distractor_coef=a[2], target_coef_margin_fixed=b[2], distractor_coef_margin_fixed=b[3],
                margin_coef=b[1], lesioned_per_intact_margin=k, mean_h_easiest_third=float(f.h[easy].mean()))


def bootstrap(f, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    groups = f.concept_group.to_numpy()
    rows = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = list(rows)
    point = stats(f)
    draws = [stats(f.iloc[np.concatenate([rows[keys[i]] for i in rng.integers(len(keys), size=len(keys))])].reset_index(drop=True))
             for _ in range(n)]
    out = {}
    for k, v in point.items():
        lo, hi = np.nanpercentile([d[k] for d in draws], [2.5, 97.5])
        out[k] = dict(estimate=float(v), ci_low=float(lo), ci_high=float(hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs/e2_confirm/qwen3-30b-a3b-instruct/dissociation"))
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    d = Path(args.dir)
    scores = pd.read_parquet(d / "m1_scores.parquet")
    items = scoring.load_materials()[0].astype({"item_id": str})
    assoc = pd.read_csv(materials.ASSOCIATIONS, dtype={"item_id": str})
    results = {}
    for lesion in [s for s in ("control", "main") if s in set(scores.set)]:
        f = frame(scores, lesion, items, assoc)
        high = f.high_demand.astype(str) == "1"
        results[lesion] = dict(n_items=len(f), share_improved_low_demand=float(f.h[~high].lt(0).mean()),
                               share_improved_high_demand=float(f.h[high].lt(0).mean()),
                               intervals=bootstrap(f, args.n_boot))
        print(lesion, json.dumps({k: [round(v["estimate"], 2), round(v["ci_low"], 2), round(v["ci_high"], 2)]
                                  for k, v in results[lesion]["intervals"].items()}))
    (d / "association.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
