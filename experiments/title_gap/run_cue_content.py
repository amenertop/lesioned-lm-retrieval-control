"""Run the reviewed development-only cue-content diagnostic using the existing state scorer."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/e1_screen'))
import common
from lesion import cue_content, models, runinfo, title_gap_scoring, title_gap_states


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    common.add_args(ap)
    ap.add_argument('--states', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.ablation_mode != 'mean':
        ap.error('the diagnostic uses the same mean lesions as the clinical comparison')
    if args.tiny and not args.out.resolve().is_relative_to((ROOT / 'outputs/title_gap_tiny').resolve()):
        ap.error('tiny runs must use outputs/title_gap_tiny')
    commit = common.guard(args)
    path = ROOT / 'data/materials/m3/final_items.csv'
    source = json.loads(args.states.read_text(encoding='utf-8'))
    title_gap_states.check_source(source, args.model, models.MODELS[args.model][1], args.tiny)
    review = json.loads(args.review.read_text(encoding='utf-8'))
    if review['status'] != 'approved' or any(runinfo.sha256(ROOT / p) != digest for p, digest in review['sources'].items()):
        ap.error('the cue content materials must have an approved review matching current sources')
    table = pd.read_csv(path, keep_default_na=False)
    items = table[(table.role == 'calibration_a') & table.free_naming_ok]
    if args.tiny:
        items = items.head(2)
    model, tok, chat, means, _ = common.load(args)
    provenance = dict(commit=commit, model=args.model, revision=models.MODELS[args.model][1], tiny=args.tiny,
        dtype=args.dtype, moe=args.moe, split='dev', battery='cue_content', items_sha256=runinfo.sha256(path),
        means_sha256=runinfo.sha256(args.means), states_sha256=runinfo.sha256(args.states),
        review_sha256=runinfo.sha256(args.review), included_items=items.item_id.tolist(),
        purpose='post-development semantic-content diagnostic, not a new confirmatory battery')
    title_gap_scoring.run_states(model, tok, means, cue_content.prompts(items, chat), [], source['states'],
        args.out, provenance, chat, batches=(4, 3, 2) if args.tiny else (16, 11, 7))


if __name__ == '__main__':
    main()
