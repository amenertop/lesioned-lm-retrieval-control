"""Score old reviewed development/test items under explicit coarse or component lesion states."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
import common
from lesion import (knowledge_items as ki, legacy_battery, m3_battery, materials, models, naming,
                    neutral_text, runinfo, scoring, title_gap_states)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    common.add_args(ap)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--freeze-tag")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.ablation_mode != "mean":
        ap.error("the component comparison uses mean lesions")
    commit = common.guard(args)
    source = json.loads(args.states.read_text(encoding="utf-8"))
    title_gap_states.check_source(source, args.model, models.MODELS[args.model][1], args.tiny)
    word_items = ki.load_items(args.split, args.freeze_tag)
    control_items = materials.load_split(args.split, freeze_tag=args.freeze_tag)
    pool = pd.read_csv(ROOT / "data/materials/m3/final_items.csv")
    # Development uses experiment 3's foil pool. Test targets are added before fixing test foils.
    comprehension_items = pool[pool.role == ("test" if args.split == "test" else "calibration_a")]
    foil_pool = pd.concat([pool[pool.role.isin(m3_battery.POOL_ROLES)], comprehension_items])
    foil_pool = foil_pool.drop_duplicates("item_id").reset_index(drop=True)
    if args.tiny:
        word_items, control_items, comprehension_items = [t.head(2) for t in
                                                         (word_items, control_items, comprehension_items)]
    model, tok, chat, means, _ = common.load(args)
    words = naming.build_prompts(word_items, pd.read_csv(ki.OUT / "demos.csv"), chat)
    _, demos, templates = scoring.load_materials()
    control = scoring.build_label_prompts(control_items, demos, templates, variant="main", chat=chat)
    comprehension = m3_battery.word_to_definition_items(comprehension_items, foil_pool, chat)
    seqs = [tok.encode(t, add_special_tokens=False)[:128] for t in neutral_text.load()[:2 if args.tiny else 40]]
    files = [args.states, Path(args.means), ki.OUT / "items.csv", ki.OUT / "demos.csv",
             ROOT / "data/materials/m3/final_items.csv", *[scoring.M1 / f for f in common.MATERIAL_FILES]]
    provenance = dict(commit=commit, model=args.model, revision=models.MODELS[args.model][1], tiny=args.tiny,
                      dtype=args.dtype, moe=args.moe, scorer=args.scorer, split=args.split, freeze_tag=args.freeze_tag,
                      source_sha256={str(p): runinfo.sha256(p) for p in files},
                      analysis_plan_sha256=runinfo.analysis_plan_sha256(),
                      environment={k: v for k, v in runinfo.environment().items() if k != "host"})
    legacy_battery.run(model, tok, means, words, control, comprehension, source["states"], args.out, provenance,
                       common.SCORERS[args.scorer], chat, batches=(2, 1) if args.tiny else (16, 11, 7),
                       neutral_sequences=seqs)


if __name__ == "__main__":
    main()
