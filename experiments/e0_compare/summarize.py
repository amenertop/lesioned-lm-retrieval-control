"""Compare Experiment 0 across models (exploratory; not part of any G1 decision).

For every run folder given, reads label_main.csv (frozen scoring output) and reports, per model version:
the G1 quantities (TDS effect E, high-demand accuracy, order consistency, string format check) as point
estimates, plus the share of items whose correct-minus-wrong logit margin, averaged over the two option
orders, is positive ("order-averaged accuracy"). Averaging cancels a constant additive preference for label
A or B; it does not remove order-by-item interactions or other format failures, so it is a descriptive
diagnostic without intervals, not proof of why a model fails.

    python experiments/e0_compare/summarize.py --runs outputs/e0 outputs/e0_backup outputs/e0_explore/<m> ... \
        --out reports/e0_compare
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import g1, scoring  # noqa: E402


def version_row(version, label_scores, items, gap):
    tab = g1.item_table(label_scores, items)
    fm = tab[tab.task.isin(g1.MATCHING)]
    hd, ld = fm[fm.high_demand == "1"], fm[fm.high_demand == "0"]
    string = tab[tab.task == "string"]
    syn = tab[tab.task == "synonym"]
    e = g1.tds_effect_stat(gap)(fm)
    return dict(
        version=version,
        tds_effect_E=e,
        high_demand_acc=hd.acc.mean(), low_demand_acc=ld.acc.mean(),
        order_consistency=fm.consistent.mean(),
        string_acc=string.acc.mean(), string_consistency=string.consistent.mean(),
        synonym_acc=syn.acc.mean(),
        order_avg_high_demand_acc=(hd.d > 0).mean(), order_avg_low_demand_acc=(ld.d > 0).mean(),
        order_avg_string_acc=(string.d > 0).mean(), order_avg_synonym_acc=(syn.d > 0).mean(),
        share_choose_A=(label_scores.choice == "A").mean(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="folders holding <version>/label_main.csv")
    ap.add_argument("--out", default=str(ROOT / "reports/e0_compare"))
    args = ap.parse_args()

    items, _, _ = scoring.load_materials()
    gap = g1.tds_gap(items)
    rows = []
    for run in args.runs:
        for f in sorted(Path(run).glob("*/label_main.csv")):
            if f.parent.name == "smoke":
                continue
            meta = json.loads((f.parent / "meta.json").read_text(encoding="utf-8"))
            row = version_row(f.parent.name, pd.read_csv(f, dtype={"item_id": str}), items, gap)
            row.update(run=run, exploratory=meta.get("exploratory"), freeze_tag=meta.get("freeze_tag"),
                       git_commit=meta.get("git_commit"))
            rows.append(row)
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "e0_compare.csv", index=False, float_format="%.4f")
    print(table.drop(columns=["run", "git_commit"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
