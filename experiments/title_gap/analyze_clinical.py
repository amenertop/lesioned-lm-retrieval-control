"""Classify actual M3 answers and align progressive cue endpoints with published group measurements."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import clinical_alignment as ca, runinfo, title_gap_analysis as ta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, action='append', required=True)
    ap.add_argument("--review", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    tables, plan = ta.read_runs(args.run)
    path = ROOT / "data/materials/m3/final_items.csv"
    if plan["provenance"]["battery"] != "clinical_m3" or plan["provenance"]["items_sha256"] != runinfo.sha256(path):
        ap.error("clinical run and M3 materials must match")
    rows = ca.classify_cued(tables["generated"], pd.read_csv(path, keep_default_na=False),
                           pd.read_csv(args.review) if args.review else None)
    result = ca.cue_profiles(rows, args.n_boot)
    for state, group in rows[rows.letters == 0].groupby("state"):
        result[state]["naming_error_profile"] = ta.fine_error_profile(group)
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / "responses.csv", index=False)
    rows[rows.category == "unclassified"][["response_id", "target", "synset", "definition", "answer"]].drop_duplicates("response_id").to_csv(args.out / "blind-response-review.csv", index=False)
    payload = dict(meta=dict(provenance=plan["provenance"], source_fingerprint=plan["fingerprint"],
        source_runs=plan.get('source_runs', [dict(path=str(args.run[0]), fingerprint=plan['fingerprint'])]),
        n_boot=args.n_boot, analysis_commit=runinfo.commit(),
        review_sha256=runinfo.sha256(args.review) if args.review else None,
        analysis_sha256=runinfo.sha256(ROOT / "src/lesion/clinical_alignment.py")), states=result)
    (args.out / "result.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
