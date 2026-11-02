"""Does attribution patching pick the same experts as exhaustive single-expert ablation? (ADR 0004)

Only meaningful where exhaustive ablation is reliable (OLMoE: batch-composition noise about 0.06 logit).

    python experiments/e1_screen/compare_attribution.py --root outputs/e1_screen --models m1 m2 ... --out reports/mvp

Reads <root>/<model>/screen/{screen.csv, selection.json} (exhaustive, real ablations) and
<root>/<model>/attribution/attribution.csv (first-order estimates) and writes <out>/attribution_vs_exhaustive.{csv,md}.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

DIRECTIONS = {"control": "control_effect", "knowledge": "knowledge_effect"}


def compare(model, base, top=32):
    real = pd.read_csv(base / "screen/screen.csv").set_index("component")
    est = pd.read_csv(base / "attribution/attribution.csv").set_index("component")
    selection = json.loads((base / "screen/selection.json").read_text(encoding="utf-8"))
    both = real.index.intersection(est.index)
    rows = []
    for d, col in DIRECTIONS.items():
        r, e = real.loc[both, col], est.loc[both, col]
        rho = stats.spearmanr(r, e).statistic
        short = set(e.nlargest(top).index)
        chosen = selection.get(d, [])
        rows.append(dict(model=model, direction=d, n_experts=len(both), spearman=rho,
                         pearson=stats.pearsonr(r, e).statistic,
                         top32_overlap=len(short & set(r.nlargest(top).index)),
                         chosen_in_shortlist=sum(c in short for c in chosen), n_chosen=len(chosen)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [r for m in args.models for r in compare(m, Path(args.root) / m)]
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "attribution_vs_exhaustive.csv", index=False, float_format="%.4f")
    lines = ["| 模型版本 | 方向 | 专家数 | 名次相关（Spearman） | 数值相关（Pearson） | 两种方法前 32 名的重合数 | 逐个消融选中的 4 个中落在梯度前 32 名的个数 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['direction']} | {r['n_experts']} | {r['spearman']:.2f} | {r['pearson']:.2f} | "
                     f"{r['top32_overlap']} / 32 | {r['chosen_in_shortlist']} / {r['n_chosen']} |")
    (out / "attribution_vs_exhaustive.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
