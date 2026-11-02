"""Diagnose truth discrimination and response bias using already scored property questions."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import runinfo, title_gap_analysis as ta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--items', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    tables, plan = ta.read_run(args.run)
    if plan['provenance']['battery'] != 'fine' or runinfo.sha256(args.items) != plan['provenance']['items_sha256']:
        ap.error('property diagnostics require the original fine materials')
    if args.n_boot < 0:
        ap.error('n-boot must be nonnegative')
    result, rows = ta.property_response_diagnostics(tables['labels'], pd.read_csv(args.items), args.n_boot)
    result['meta'] = dict(status='posthoc_property_response_diagnostic', provenance=plan['provenance'],
        source_fingerprint=plan['fingerprint'], n_boot=args.n_boot, seed=1910, analysis_commit=runinfo.commit(),
        analysis_sources={str(p.relative_to(ROOT)): runinfo.sha256(p) for p in
                          (Path(__file__).resolve(), ROOT / 'src/lesion/title_gap_analysis.py', args.items.resolve())},
        definition='Order-averaged forced-choice scores. Original threshold unchanged. AUC is ranking, not generated accuracy.')
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / 'property-pairs.csv', index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
