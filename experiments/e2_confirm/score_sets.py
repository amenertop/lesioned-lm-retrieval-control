"""M1 label scores of the fixed lesion sets on a split that is not held out (exploratory follow-ups).

    python experiments/e2_confirm/score_sets.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3 --split dev

Sets: the control heads (M1 screen selection) and the main knowledge set (K8, as frozen in
run_dissociation_test.py; left out on the tiny test model). Writes outputs/e2_confirm/<model>/scores_<split>/
{m1_scores.parquet, meta.json} in the dissociation test's format, so explore_association.py --dir reads it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))

import common  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import materials, runinfo, scoring, screen  # noqa: E402
from lesion.components import Component  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--split", required=True, choices=["screening", "dev"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    commit = common.guard(args)
    items = materials.load_split(args.split)
    _, demos, templates = scoring.load_materials()
    m1_dir = common.out_dir(args, "screen")
    m1_meta = json.loads((m1_dir / "meta.json").read_text(encoding="utf-8"))
    sets = {"control": [Component.parse(k) for k in common.load_selection(m1_dir, m1_meta)["control"]]}
    if not args.tiny:
        sets["main"] = [Component.parse(k) for k in rdt.K8]

    model, tok, chat, means, ablate = common.load(args)
    prompts = scoring.build_label_prompts(items, demos, templates, variant="main", chat=chat)
    out = Path(args.out) if args.out else ROOT / "outputs/e2_confirm" / args.model / f"scores_{args.split}"
    fields = common.run_fields(args, commit, "score_sets", args.split, len(prompts),
                               sets={k: [c.key for c in v] for k, v in sets.items()})
    common.check_same_run(m1_meta, fields, m1_dir, keys=[k for k in common.SAME_RUN if k != "analysis_plan_sha256"],
                          same_code=False)
    common.check_resume(out, fields)
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()
    score = screen.make_scorer(model, tok, prompts, chat, ablate, args.batch_size, common.SCORERS[args.scorer])
    scores = {"intact": score(), **{k: score(v) for k, v in sets.items()}}
    pd.concat(scores, names=["set"]).reset_index(level=0).to_parquet(out / "m1_scores.parquet", index=False)
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)
    print(f"{args.split}: {len(prompts)} prompts, sets {list(scores)} -> {out}")


if __name__ == "__main__":
    main()
