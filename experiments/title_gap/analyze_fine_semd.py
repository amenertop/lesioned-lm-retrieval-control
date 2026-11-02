"""Paired SemD regressions on completed new-concept measurements, without additional model inference."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import runinfo, semantic_diversity as sd, title_gap_analysis as ta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--norms", type=Path, required=True)
    ap.add_argument("--wr", default="WR8_a0")
    ap.add_argument("--sc", default="SC4_a0")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, help="public statistical summary without per-word norm values")
    args = ap.parse_args()
    tables, plan = ta.read_run(args.run)
    if plan["provenance"]["battery"] != "fine" or runinfo.sha256(args.items) != plan["provenance"]["items_sha256"]:
        ap.error("the run and fine materials must match")
    items = pd.read_csv(args.items, keep_default_na=False)
    items = items[(items.split == plan["provenance"]["split"]) & (items.review != "excluded")]
    coverage, items = sd.coverage(items, sd.load_norms(args.norms))
    items["frequency"] = items.log_bnc_freq
    _, frame = ta.fine_profiles(tables, items, n_boot=0)
    frames = sd.fine_frames(frame, args.wr, args.sc)
    results = {task: sd.paired_regression(rows, n_boot=args.n_boot) for task, rows in frames.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    for task, rows in frames.items():
        rows.to_csv(args.out / f"{task}-items.csv", index=False)
    result = dict(status="exploratory" if plan["provenance"]["split"] != "test" else "held_out",
        source_fingerprint=plan["fingerprint"], provenance=plan["provenance"], coverage=coverage,
        retrieval=args.wr, control=args.sc, n_boot=args.n_boot, results=results,
        analysis_commit=runinfo.commit(), norms_sha256=runinfo.sha256(args.norms),
        sources={str(p.relative_to(ROOT)): runinfo.sha256(p) for p in
                 (Path(__file__), ROOT / "src/lesion/semantic_diversity.py", ROOT / "src/lesion/title_gap_analysis.py")})
    (args.out / "result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        summary = sd.public_summary({**result, "coverage": {"fine": result["coverage"]}})
        args.report.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
