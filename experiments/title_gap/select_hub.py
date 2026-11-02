"""Select soft and hard candidates on development concepts by the prewritten title-gap rule."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import fine_protocol as fp, runinfo, semantic_diversity as sd, title_gap_analysis as ta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--items', type=Path, required=True)
    ap.add_argument('--norms', type=Path, required=True)
    ap.add_argument('--receipt', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    tables, plan = ta.read_run(args.run)
    if (plan['provenance']['split'] != 'dev' or plan['provenance']['battery'] != 'fine'
            or plan['provenance'].get('tiny') is not False):
        ap.error('candidate selection requires real fine development runs')
    if runinfo.sha256(args.items) != plan['provenance']['items_sha256']:
        ap.error('items differ from the source run')
    items = pd.read_csv(args.items, keep_default_na=False).query("split == 'dev'")
    items = sd.match_words(items, sd.load_norms(args.norms))
    items['frequency'] = items.log_bnc_freq
    readouts = {s['training'].get('naming_readout', 'target') for s in plan['states'] if 'training' in s}
    if len(readouts) != 1:
        ap.error('selection requires one consistently recorded training naming objective')
    readout = readouts.pop()
    if readout != 'target' and args.receipt is None:
        ap.error('relative selection requires a committed output receipt')
    if args.receipt is not None:
        receipt_path = args.receipt.resolve().relative_to(ROOT).as_posix()
        if args.receipt.read_bytes() != runinfo.git_bytes('show', f'HEAD:{receipt_path}'):
            ap.error('output receipt must be committed before selection')
        if json.loads(args.receipt.read_text(encoding='utf-8')) != fp.output_receipt([args.run]):
            ap.error('development outputs differ from the committed receipt')
    profiles, frame = ta.fine_profiles(ta.fine_readout_tables(tables, items, readout), items, n_boot=0)
    answers, _ = ta.fine_responses(tables, items, n_boot=0)
    ranking = ta.learned_development_scores(frame, plan['states'], profiles, answers, tables['language'])
    selected = {}
    for (mode, kind), rows in ranking.groupby(['mode', 'kind']):
        eligible = rows[rows.eligible]
        selected[mode + '_' + kind] = dict(state=eligible.iloc[0].state if len(eligible) else None,
            complete_profile=bool(eligible.iloc[0].complete_profile) if len(eligible) else False,
            n_eligible=len(eligible), n_candidates=len(rows))
    payload = dict(selected=selected, provenance=plan['provenance'], source_fingerprint=plan['fingerprint'],
        naming_readout=readout,
        selection_commit=runinfo.commit(), analysis_sha256=runinfo.sha256(ROOT / 'src/lesion/title_gap_analysis.py'),
        norms_sha256=runinfo.sha256(args.norms), rule_sha256=runinfo.sha256(ROOT / 'docs/plans/title-gap-selection.md'))
    if readout != 'target':
        payload['objective_plan_sha256'] = runinfo.sha256(ROOT / 'docs/plans/title-gap-relative-hub.md')
        if any(s['training'].get('objective_plan_sha256') != payload['objective_plan_sha256']
               for s in plan['states'] if 'training' in s):
            ap.error('relative selection plan differs from training')
    args.out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out / 'ranking.csv', index=False, lineterminator='\n')
    payload['ranking_sha256'] = runinfo.sha256(args.out / 'ranking.csv')
    payload['development_plan_sha256'] = runinfo.sha256(args.run / 'plan.json')
    payload['receipt_sha256'] = runinfo.sha256(args.receipt) if args.receipt else None
    (args.out / 'selection.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
