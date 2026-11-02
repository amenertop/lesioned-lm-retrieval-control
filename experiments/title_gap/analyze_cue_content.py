"""Analyze the development cue-content diagnostic without changing its initial error sets."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import cue_content, runinfo, title_gap_analysis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--review', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=1910)
    args = ap.parse_args()
    tables, plan = title_gap_analysis.read_run(args.run)
    source = ROOT / 'data/materials/m3/final_items.csv'
    provenance = plan['provenance']
    if (provenance['battery'] != 'cue_content' or provenance['split'] != 'dev'
            or provenance['items_sha256'] != runinfo.sha256(source)):
        ap.error('cue-content analysis requires the reviewed development materials')
    items = pd.read_csv(source, keep_default_na=False)
    items = items[items.item_id.isin(provenance['included_items'])]
    if [vars(p) for p in cue_content.prompts(items, plan['chat'])] != plan['naming']:
        ap.error('reconstructed cue-content prompts differ from the scored inputs')
    rows, queue = cue_content.classify(tables['generated'], items,
        pd.read_csv(args.review) if args.review else None, plan['chat'])
    result = cue_content.summarize(rows, tables['candidates'], args.n_boot, args.seed)
    result['meta'].update(status='postdevelopment_cue_content_diagnostic', provenance=provenance,
        source_fingerprint=plan['fingerprint'], analysis_commit=runinfo.commit(),
        review_sha256=runinfo.sha256(args.review) if args.review else None,
        sources={p.relative_to(ROOT).as_posix(): runinfo.sha256(p) for p in
                 [Path(__file__), ROOT / 'src/lesion/cue_content.py', ROOT / 'src/lesion/clinical_alignment.py']})
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / 'responses.csv', index=False)
    queue.to_csv(args.out / 'blind-response-review.csv', index=False)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    (args.out / 'result.json').write_text(text, encoding='utf-8')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding='utf-8')
    print(f'{len(rows)} answers, {len(queue)} unique responses need blind review')


if __name__ == '__main__':
    main()
