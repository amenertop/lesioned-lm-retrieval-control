"""MVP step 1: lesion each expert alone on the M1 screening split (docs/plans/mvp.md; ticket pilot 11).

Run on the GPU server from the repo root, after experiments/prep_means/run.py:
    python experiments/e1_screen/run_screen.py --model <G1 version> --means outputs/means/<file>.pt
    python experiments/e1_screen/run_screen.py --model olmoe-0924-base --means <tiny means> --tiny   # local check

Writes outputs/e1_screen/<model>/screen/{intact.csv, shards/, screen.csv, selection.json, meta.json}.
A stopped run resumes where it left off when started again with the same arguments and commit.
"""
import argparse
import json
import time
from pathlib import Path

import common
from lesion import components, damage, materials, runinfo, scoring, screen


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--kind", default="expert", choices=["expert", "head", "expert+head"],
                    help="component kinds to screen (ADR 0005 adds attention heads)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N components (throughput check)")
    ap.add_argument("--flush-every", type=int, default=16)
    ap.add_argument("--candidates", default=None, help="attribution.csv from run_attribution.py: screen only its top experts")
    ap.add_argument("--top", type=int, default=32, help="with --candidates: top experts per direction by estimated effect")
    args = ap.parse_args()

    commit = common.guard(args)
    all_items, demos, templates = scoring.load_materials()
    items = materials.load_split("screening")
    model, tok, chat, _, ablate = common.load(args)
    prompts = scoring.build_label_prompts(items, demos, templates, variant="main", chat=chat)
    out = common.out_dir(args, "screen")
    params = dict(kind=args.kind)
    if args.candidates:
        params.update(candidates_sha256=runinfo.sha256(args.candidates), top=args.top)
    fields = common.run_fields(args, commit, "screen", "screening", len(prompts), **params)
    if args.candidates:
        # the shortlist must come from the same model, precision, means, materials and code, on the screening split
        cand_meta = json.loads((Path(args.candidates).parent / "meta.json").read_text(encoding="utf-8"))
        if cand_meta.get("status") != "done" or cand_meta.get("split") != "screening":
            raise SystemExit(f"{args.candidates} is unfinished or not from the screening split")
        common.check_same_run(cand_meta, fields, args.candidates, keys=[k for k in common.SAME_RUN if k != "scorer"])
    common.check_resume(out, fields)
    runinfo.write_meta(out, status="running", **fields)

    t0 = time.time()
    score = screen.make_scorer(model, tok, prompts, chat, ablate, args.batch_size, common.SCORERS[args.scorer])
    intact = score()
    intact.to_csv(out / "intact.csv", index=False)
    comps = [c for kind in args.kind.split("+") for c in components.all_components(model.config, kind)]
    if args.candidates:
        comps = common.shortlist(args.candidates, comps, args.top)
    comps = comps[:args.limit]
    shards = screen.run_screen(score, comps, out / "shards", args.flush_every)

    freq_median = damage.frequency_median(all_items)
    summary = screen.summarize(intact, shards, items, freq_median)
    summary.to_csv(out / "screen.csv", index=False)
    chosen = screen.select_with_supplement(summary)
    (out / "selection.json").write_text(json.dumps(dict(
        chosen, k=screen.K, freq_median=freq_median,
        rule=("each component goes to the direction with the larger z-scored effect; top k raw effects per direction; "
              "main control set excludes components whose string damage exceeds their control effect (ADR 0005); "
              "supplementary.control_unfiltered keeps them")),
        indent=2), encoding="utf-8")
    runinfo.write_meta(out, status="done", n_components=len(summary), seconds=round(time.time() - t0, 1),
                       selection_sha256=runinfo.sha256(out / "selection.json"),
                       screen_csv_sha256=runinfo.sha256(out / "screen.csv"),
                       screened_components_sha256=common.component_set_sha256(summary.component), **fields)
    print(f"{len(summary)} components -> {out}\ncontrol: {chosen['control']}\nknowledge: {chosen['knowledge']}")


if __name__ == "__main__":
    main()
