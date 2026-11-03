"""Exploratory check (2026-11-03, user decision after the second naive-reader round): does the WR8 frequency effect
survive when the intact model's log probability of the correct name is held constant?

    C:\\Users\\wanghe\\.conda\\envs\\lesion\\python.exe -X utf8 reports/explore/intact-covariate-20261103/intact_covariate.py

Reads the stored per-concept naming scores of the confirmatory dissociation run (no model is run), as
reports/explore/token-covariate-20261028/token_covariate.py does, and writes intact-covariate.json next to this script.
A reader asked whether rare names lose more only because the intact model is already less sure of them. Steps:
1. Rebuild the confirmatory frequency effect (1.14 nats, ledger knowledge-f3ce21647c2f) and stop if it differs.
2. Regress the per-concept WR8 loss on a rare indicator (Zipf below 3.00) and the intact log probability of the
   correct name; then on the rare indicator, the intact log probability and the token count. Report the rare
   coefficient and the covariate coefficients with 95% percentile intervals from 10,000 bootstrap resamples of concepts
   (seed 0). The covariate set was fixed before the regression was run and is reported whatever it shows.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCORES = ROOT / "outputs/server/e2_confirm/qwen3-30b-a3b-instruct/dissociation/naming_scores.parquet"
ITEMS = ROOT / "data/materials/m3_knowledge/items.csv"
LEDGER = ROOT / "docs/paper-prep/results_ledger.csv"
OUT = Path(__file__).with_name("intact-covariate.json")
LEDGER_ID = "knowledge-f3ce21647c2f"
N_BOOT, SEED = 10_000, 0


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(y, X, names):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(SEED)
    boots = np.empty((N_BOOT, X.shape[1]))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        boots[b] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    unit = {"rare": "log probability（nat）", "intact_logprob": "nat 损失 / nat 完好对数概率", "n_tokens": "nat / token"}
    return {n: {"estimate": float(beta[k]), "ci_low": float(lo[k]), "ci_high": float(hi[k]), "n": int(len(y)), "unit": unit[n],
                "n_unit": "概念"} for k, n in enumerate(names) if n != "intercept"}


def main():
    scores = pd.read_parquet(SCORES)
    items = pd.read_csv(ITEMS)[["item_id", "zipf"]]
    avg = scores[scores.scoring == "averaged"]
    intact = avg[avg.set == "intact"].set_index("item_id")
    wr8 = avg[avg.set == "main"].set_index("item_id")
    ids = intact.index.intersection(wr8.index)
    df = pd.DataFrame({"loss": intact.loc[ids, "logprob"] - wr8.loc[ids, "logprob"],
                       "intact_logprob": intact.loc[ids, "logprob"],
                       "n_tokens": intact.loc[ids, "n_tokens"]}).join(items.set_index("item_id"))
    df["rare"] = (df.zipf < 3.0).astype(float)
    effect = df.loc[df.rare == 1, "loss"].mean() - df.loc[df.rare == 0, "loss"].mean()
    target = float(pd.read_csv(LEDGER, encoding="utf-8-sig").set_index("编号").loc[LEDGER_ID, "数值"])
    if abs(effect - target) > 1e-6:
        raise SystemExit(f"rebuilt frequency effect {effect} does not reproduce the ledger value {target}")
    y = df.loss.to_numpy()
    one = np.ones(len(df))
    m1 = fit(y, np.column_stack([one, df.rare, df.intact_logprob]), ["intercept", "rare", "intact_logprob"])
    m2 = fit(y, np.column_stack([one, df.rare, df.intact_logprob, df.n_tokens.astype(float)]),
             ["intercept", "rare", "intact_logprob", "n_tokens"])
    out = {
        "meta": {"description": "WR8 frequency effect with the intact log probability of the correct name as a covariate, "
                                "exploratory, no model run, per-concept scores of the confirmatory dissociation test",
                 "inputs": {str(SCORES.relative_to(ROOT)): sha256(SCORES), str(ITEMS.relative_to(ROOT)): sha256(ITEMS)},
                 "ledger_check": {"id": LEDGER_ID, "ledger_value": target, "rebuilt_value": float(effect)},
                 "n_boot": N_BOOT, "seed": SEED, "rare_rule": "Zipf below 3.00"},
        "intact_logprob_mean": {"rare": float(df.loc[df.rare == 1, "intact_logprob"].mean()),
                                "common": float(df.loc[df.rare == 0, "intact_logprob"].mean())},
        "adjusted_for_intact": m1,
        "adjusted_for_intact_and_tokens": m2,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
