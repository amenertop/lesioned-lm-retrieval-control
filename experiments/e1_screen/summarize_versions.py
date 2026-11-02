"""Put the MVP results of several model versions side by side (exploratory replication check).

    python experiments/e1_screen/summarize_versions.py --root outputs/e1_screen --models m1 m2 ... --out reports/mvp

Reads <root>/<model>/{screen/selection.json, joint/results.json} and writes <out>/versions.csv and
<out>/versions.md: per version, the chosen components, the joint-lesion effects on the dev split with their
cluster-bootstrap intervals, and where the target set falls among its matched random sets.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import screen  # noqa: E402


def version_rows(model, base):
    selection = json.loads((base / "screen/selection.json").read_text(encoding="utf-8"))
    res = json.loads((base / "joint/results.json").read_text(encoding="utf-8"))
    iv, pts, rank = res["intervals"], res["points"], res["random_rank"]
    rows = []
    for d, primary in screen.PRIMARY.items():
        if f"{d}.{primary}" not in iv:
            rows.append(dict(model=model, direction=d, status=selection["status"][d]))
            continue
        rand = [v[primary] for n, v in pts.items() if n.startswith(f"random_{d}_")]
        row = dict(model=model, direction=d, status=selection["status"][d], components=" ".join(selection[d]))
        for e in ("control_effect", "knowledge_effect", "control_slope", "semantic_damage", "string_damage"):
            for k in ("estimate", "ci_low", "ci_high"):
                row[f"{e}_{k}"] = iv[f"{d}.{e}"][k]
        row.update(random_median=pd.Series(rand).median(), random_max=max(rand),
                   random_share_at_least=rank[d]["share_at_least_target"], n_random=rank[d]["n_random"])
        rows.append(row)
    for delta in ("delta_C", "delta_F"):
        if delta in iv:
            rows[0][delta] = iv[delta]["estimate"]
            rows[0][f"{delta}_ci"] = f"[{iv[delta]['ci_low']:.2f}, {iv[delta]['ci_high']:.2f}]"
    return rows


def fmt(r, e):
    return f"{r[f'{e}_estimate']:.2f} [{r[f'{e}_ci_low']:.2f}, {r[f'{e}_ci_high']:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs/e1_screen"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "reports/mvp"))
    args = ap.parse_args()
    rows = [r for m in args.models for r in version_rows(m, Path(args.root) / m)]
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "versions.csv", index=False, float_format="%.4f")
    lines = ["| 模型版本 | 专家组 | 控制效应 | 知识效应 | 控制斜率 | 语义题平均损伤 | 字符串题平均损伤 | 随机组中位数（最大） | 随机组达到的比例 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if "control_effect_estimate" not in r or pd.isna(r.get("control_effect_estimate")):
            lines.append(f"| {r['model']} | {r['direction']} | {r['status']} | | | | | | |")
            continue
        primary = screen.PRIMARY[r["direction"]]
        lines.append(f"| {r['model']} | {r['direction']} | {fmt(r, 'control_effect')} | {fmt(r, 'knowledge_effect')} | "
                     f"{fmt(r, 'control_slope')} | {fmt(r, 'semantic_damage')} | {fmt(r, 'string_damage')} | "
                     f"{r['random_median']:.2f}（{r['random_max']:.2f}） | {r['random_share_at_least']:.0%}（{primary}） |")
    (out / "versions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
