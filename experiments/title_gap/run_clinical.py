"""Run the existing M3 progressive letter-cue task under explicit new lesion states."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
import common
from lesion import m3_battery, m3_cued, models, neutral_text, runinfo, title_gap_scoring, title_gap_states


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    common.add_args(ap)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--freeze-tag")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.ablation_mode != "mean":
        ap.error("the clinical comparison uses mean lesions")
    if args.tiny and not args.out.resolve().is_relative_to((ROOT / "outputs/title_gap_tiny").resolve()):
        ap.error("tiny runs must use outputs/title_gap_tiny")
    commit = common.guard(args)
    if args.split == "test":
        runinfo.check_analysis_freeze(args.freeze_tag)
    source = json.loads(args.states.read_text(encoding="utf-8"))
    title_gap_states.check_source(source, args.model, models.MODELS[args.model][1], args.tiny)
    path = ROOT / "data/materials/m3/final_items.csv"
    table = pd.read_csv(path, keep_default_na=False)
    role = "calibration_a" if args.split == "dev" else "test"
    items = table[(table.role == role) & table.free_naming_ok].copy()
    if args.tiny:
        items = items.head(2)
    pool = table[table.role.isin(m3_battery.POOL_ROLES)].copy()
    pool = pd.concat([pool, items]).drop_duplicates("item_id")
    model, tok, chat, means, _ = common.load(args)
    naming = m3_cued.prompts(items, chat)
    labels = m3_battery.word_to_definition_items(items, pool, chat)
    seqs = [tok.encode(t, add_special_tokens=False)[:128] for t in neutral_text.load()[:2 if args.tiny else 40]]
    provenance = dict(commit=commit, model=args.model, revision=models.MODELS[args.model][1],
        tiny=args.tiny, dtype=args.dtype, moe=args.moe, split=args.split, freeze_tag=args.freeze_tag,
        battery="clinical_m3", items_sha256=runinfo.sha256(path), means_sha256=runinfo.sha256(args.means),
        states_sha256=runinfo.sha256(args.states), included_items=items.item_id.tolist(),
        selection="pre-existing free_naming_ok flag, not selected on new model answers",
        source_role=role, prior_use="M3 materials used in earlier experiments, not newly unseen concepts")
    title_gap_scoring.run_states(model, tok, means, naming, labels, source["states"], args.out, provenance, chat,
        batches=(4, 3, 2) if args.tiny else (16, 11, 7), neutral_sequences=seqs)


if __name__ == "__main__":
    main()
