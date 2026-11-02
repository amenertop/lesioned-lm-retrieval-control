"""Export verified clinical summaries to the numeric writing ledger's report format."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import clinical_resampling, runinfo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--answers', type=Path, required=True)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    paths = [args.answers / 'result.json', args.comparison / 'result.json', args.comparison / 'model-markers.csv']
    report = clinical_resampling.ledger_report(json.loads(paths[0].read_text(encoding='utf-8')),
        json.loads(paths[1].read_text(encoding='utf-8')), pd.read_csv(paths[2]))
    report['meta']['source_files'] = {p.as_posix(): runinfo.sha256(p) for p in paths}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Saved verified clinical comparison')


if __name__ == '__main__':
    main()
