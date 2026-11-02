"""Compare three naming readouts on complete development outputs without changing the chosen groups."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import fine_confirmation as fc, runinfo, semantic_diversity as sd, title_gap_analysis as ta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--items', type=Path, required=True)
    ap.add_argument('--norms', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    tables, plan = ta.read_run(args.run)
    if plan['provenance']['split'] != 'dev' or plan['provenance']['battery'] != 'fine':
        ap.error('readout sensitivity uses only complete fine development outputs')
    if runinfo.sha256(args.items) != plan['provenance']['items_sha256']:
        ap.error('items differ from the source run')
    items = pd.read_csv(args.items, keep_default_na=False).query("split == 'dev'")
    items = sd.match_words(items, sd.load_norms(args.norms))
    items['frequency'] = items.log_bnc_freq
    answers, _ = ta.fine_responses(tables, items, n_boot=0)
    _, relative = ta.fine_candidate_specificity(tables['candidates'], items, n_boot=0)
    rankings, summaries = [], {}
    for readout in ('target', 'target_vs_coordinate', 'accepted_vs_coordinate'):
        current = ta.fine_readout_tables(tables, items, readout)
        profiles, frame = ta.fine_profiles(current, items, n_boot=0)
        ranking = ta.learned_development_scores(frame, plan['states'], profiles, answers, tables['language'])
        ranking['within_role_rank'] = ranking.groupby(['mode', 'kind']).cumcount() + 1
        ranking['readout'] = readout
        rankings.append(ranking)
        summaries[readout] = profiles
    ranking = pd.concat(rankings, ignore_index=True)
    comparisons = []
    for (mode, kind), group in ranking.groupby(['mode', 'kind']):
        original = group[group.readout == 'target'].set_index('state')
        for readout in ('target_vs_coordinate', 'accepted_vs_coordinate'):
            changed = group[group.readout == readout].set_index('state').reindex(original.index)
            comparisons.append(dict(mode=mode, kind=kind, readout=readout,
                rank_correlation=float(original.within_role_rank.corr(changed.within_role_rank)),
                original_first=original.index[0], relative_first=changed.within_role_rank.idxmin(),
                n_candidates=len(original)))
    args.out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out / 'ranking.csv', index=False, lineterminator='\n')
    relative.to_csv(args.out / 'candidate-items.csv', index=False, lineterminator='\n')
    payload = dict(status='posthoc_development_readout_sensitivity', n_boot=0,
        note='Diagnostic rankings retain the original screening scales and objective weights. They do not replace the original selection or validate a new search objective.',
        source_fingerprint=plan['fingerprint'], provenance=plan['provenance'],
        analysis_commit=runinfo.commit(),
        sources={str(p.resolve().relative_to(ROOT)): runinfo.sha256(p) for p in
                 (Path(__file__), ROOT / 'src/lesion/title_gap_analysis.py', args.items, args.norms)},
        profiles=summaries, rank_comparisons=comparisons)
    (args.out / 'result.json').write_text(json.dumps(fc.finite_json(payload), indent=2, allow_nan=False)+'\n',
                                         encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
