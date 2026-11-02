"""Verify and summarize all frozen fine-battery runs. No model loading or new state selection."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import fine_confirmation as fc, fine_protocol as fp, runinfo, semantic_diversity as sd, title_gap_analysis as ta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, action='append', required=True)
    ap.add_argument('--items', type=Path, required=True)
    ap.add_argument('--norms', type=Path, required=True)
    ap.add_argument('--review', type=Path)
    ap.add_argument('--review-meta', type=Path)
    ap.add_argument('--prepare-review', action='store_true')
    ap.add_argument('--receipt', type=Path, required=True, help='committed hashes of completed outputs at receipt')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding='utf-8'))
    relative = args.receipt.resolve().relative_to(ROOT).as_posix()
    if (receipt != fp.output_receipt(args.run)
            or runinfo.sha256(args.receipt) != runinfo.sha256_at('HEAD', relative)):
        ap.error('received outputs differ from the committed receipt')
    tables, plan = ta.read_runs(args.run)
    protocol = fp.check_protocol(plan, ROOT, args.items, args.norms)
    items = pd.read_csv(args.items, keep_default_na=False).query("split == 'test' and review == 'approved'")
    if items.family.nunique() != protocol['n_families'] or len(items) != protocol['n_families']:
        ap.error('final concept count differs from the frozen plan')
    coverage, items = sd.coverage(items, sd.load_norms(args.norms))
    items['frequency'] = items.log_bnc_freq
    args.out.mkdir(parents=True, exist_ok=True)
    queue = fc.review_queue(tables, items)
    queue_path = args.out / 'blind-response-review.csv'
    queue.to_csv(queue_path, index=False, lineterminator='\n')
    if args.prepare_review:
        print(f'{len(queue)} unknown responses written for blind review')
        return
    review, review_meta = fp.checked_review(queue, queue_path, args.review, args.review_meta)
    payload, frames = fc.summarize(tables, plan, items, protocol, review)
    payload.update(coverage=coverage, review_metadata=review_meta,
        analysis_sources=protocol['sources_sha256'],
        auxiliary_sources={str(p): runinfo.sha256(p) for p in (args.norms, args.review, args.review_meta, args.receipt) if p})
    for name, table in frames.items():
        table.to_csv(args.out / f'{name}.csv', index=False)
    payload = fc.finite_json(payload)
    (args.out / 'result.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n',
                                         encoding='utf-8', newline='\n')
    if args.report:
        report = fc.ledger_summary(payload, frames['items'], len(plan['neutral_sequences']))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(fc.finite_json(report), indent=2, allow_nan=False)+'\n',
                               encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
