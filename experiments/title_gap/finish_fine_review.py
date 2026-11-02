"""Apply development answer reviews without repeating unchanged continuous-score bootstraps."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import fine_confirmation as fc, runinfo, title_gap_analysis as ta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--items', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--review-meta', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    tables, plan = ta.read_run(args.run)
    source = args.analysis / 'result.json'
    payload = json.loads(source.read_text(encoding='utf-8'))
    analysis_sources = {name.replace('\\', '/'): digest for name, digest in payload['analysis_sources'].items()}
    if (plan['provenance']['split'] != 'dev' or plan['provenance']['battery'] != 'fine'
            or payload['source_fingerprint'] != plan['fingerprint']
            or plan['provenance']['items_sha256'] != runinfo.sha256(args.items)
            or analysis_sources['src/lesion/title_gap_analysis.py'] != runinfo.sha256(ROOT / 'src/lesion/title_gap_analysis.py')
            or args.out.resolve() == args.analysis.resolve()):
        ap.error('review requires matching development sources and a separate output directory')
    items = pd.read_csv(args.items, keep_default_na=False).query("split == 'dev'")
    queue = fc.review_queue(tables, items)
    review = pd.read_csv(args.review, keep_default_na=False)
    if review.response_id.duplicated().any() or set(review.response_id) != set(queue.response_id):
        ap.error('review must cover exactly the initially unknown responses')
    metadata = json.loads(args.review_meta.read_text(encoding='utf-8'))
    if metadata['review_sha256'] != runinfo.sha256(args.review):
        ap.error('review metadata does not match the judgement table')
    joined = queue.merge(review[['response_id', 'category']], on='response_id', validate='one_to_one')
    joined['normalized_answer'] = joined.answer.str.lower().str.strip().str.strip('\"\'.,! ')
    if (joined.groupby(['target', 'synset', 'condition', 'normalized_answer']).category.nunique() > 1).any():
        ap.error('identical responses have inconsistent categories')
    answers, rows = ta.fine_responses(tables, items, 0, review[review.category != 'unclassified'])
    labels = ta.fine_label_accuracy(tables['labels'])
    for state, answer in answers.items():
        for key, value in payload['actual_answers'][state].items():
            if key.endswith(('_binary_r', '_response_r')) and not key.startswith('naming_'):
                answer[key] = value
        if answer['naming_accuracy_status'] != 'fully_classified':
            continue
        named = rows.query("state == @state and condition == 'naming'")[["item_id", "family", "correct_response"]]
        wide = labels[labels.state == state].pivot(index='item_id', columns='fmt', values='correct_response')
        paired = named.rename(columns={'correct_response': 'naming'}).merge(wide, on='item_id', validate='one_to_one')
        for task in ('definition', 'synonym', 'property'):
            if task not in paired:
                continue
            part = paired.dropna(subset=['naming', task])
            metric = 'response_r' if task == 'property' else 'binary_r'
            answer[f'naming_{task}_{metric}'] = ta.bootstrap(part, lambda t, task=task:
                float(np.corrcoef(t.naming.astype(float), t[task].astype(float))[0, 1])
                if t.naming.nunique() > 1 and t[task].nunique() > 1 else np.nan, payload['n_boot'])
    payload['actual_answers'] = answers
    payload['review_metadata'] = metadata
    payload['reviewed_from'] = dict(path=source.as_posix(), sha256=runinfo.sha256(source))
    payload['review_analysis_sources'] = {p.as_posix(): runinfo.sha256(p) for p in
        (Path(__file__).relative_to(ROOT), args.review, args.review_meta)}
    payload['review_analysis_commit'] = runinfo.commit()
    args.out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out / 'responses.csv', index=False)
    (args.out / 'result.json').write_text(json.dumps(fc.finite_json(payload), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    report = ta.fine_ledger_summary(payload, pd.read_csv(args.analysis / 'items.csv'), len(plan['neutral_sequences']))
    report['meta'].update({key: payload[key] for key in
        ('review_metadata', 'reviewed_from', 'review_analysis_sources', 'review_analysis_commit')})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(fc.finite_json(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
