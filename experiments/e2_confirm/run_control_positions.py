"""GPU session 13, item 2 (docs/plans/gpu-session13.md), exploratory, one model load: where the control heads act.

    python experiments/e2_confirm/run_control_positions.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3 --freeze-tag analysis-frozen-session13

On the held-out M1 items (both option orders), C4 is lesioned everywhere and in each of seven disjoint parts of the
prompt (lesion.probes.position_parts: context, probe, options, labels, start, end_template, answer), and n_random head
sets matched to C4 (same kind and layer, seed fixed, the same sets for every part) are lesioned in each part. One scoring
function for every state (lesion.probes.label_scores_in_positions: one batch composition, no prefix cache). The random
distribution for the whole-prompt lesion is the session 10 control test's (200 sets), not rerun here.

Writes outputs/e2_confirm/<model>/control_positions/{sets.json, parts.json, scores.parquet, meta.json}.

Session 14b (--split-labels --parts label_a,label_b,label_answer, written to control_positions_labels): the labels
part split into "A:" (before the first option word), "B:" (between the two) and "Answer:" (after the second),
lesion.probes.split_labels; only the parts given are run, with the same random head sets (same seed).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))

import common  # noqa: E402
from lesion import baselines, components, materials, probes, runinfo, scoring  # noqa: E402
from lesion.components import Component  # noqa: E402

SEED = 31


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True)
    ap.add_argument("--n-random", type=int, default=50, help="random head sets matched to C4, lesioned in each part")
    ap.add_argument("--split-labels", action="store_true", help="split the labels part into label_a, label_b, label_answer")
    ap.add_argument("--parts", default=None, help="comma-separated parts to run (default: all of them)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    suf = "_tiny" if args.tiny else ""
    name = "control_positions_labels" if args.split_labels else "control_positions"
    out = Path(args.out) if args.out else ROOT / f"outputs/e2_confirm{suf}" / args.model / name
    all_parts = (tuple(p for p in probes.PARTS if p != "labels") + probes.LABEL_PARTS) if args.split_labels else probes.PARTS
    run_parts = tuple(args.parts.split(",")) if args.parts else all_parts
    if not set(run_parts) <= set(all_parts):
        raise SystemExit(f"unknown parts {sorted(set(run_parts) - set(all_parts))}; choose from {all_parts}")
    m1_dir = common.out_dir(args, "screen")
    m1_meta = json.loads((m1_dir / "meta.json").read_text(encoding="utf-8"))
    c4 = [Component.parse(k) for k in common.load_selection(m1_dir, m1_meta)["control"]]

    model, tok, chat, means, _ = common.load(args)
    kinds = sorted({c.kind for c in c4})           # the real C4 is four heads; the tiny screen's set has experts too
    pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
    freq = means.routing_frequency() if "expert" in kinds else None
    routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if freq is not None else None
    randoms = [[c.key for c in s] for s in baselines.matched_random_sets(c4, pool, args.n_random, seed=SEED, exclude=c4,
                                                                         routing_freq=routing)]
    m1_test = materials.load_split("test", freeze_tag=args.freeze_tag)
    _, m1_demos, templates = scoring.load_materials()
    prompts = scoring.build_label_prompts(m1_test, m1_demos, templates, variant="main", chat=chat)
    parts = [probes.position_parts(tok, p, chat)[1] for p in prompts]
    if args.split_labels:
        parts = [probes.split_labels(q) for q in parts]

    out.mkdir(parents=True, exist_ok=True)
    (out / "sets.json").write_text(json.dumps(dict(C4=[c.key for c in c4], **{f"random_{i:03d}": s for i, s in enumerate(randoms)}),
                                              indent=2), encoding="utf-8")
    (out / "parts.json").write_text(json.dumps([dict(item_id=str(p.item_id), order=p.order, **{k: len(v) for k, v in q.items()})
                                                for p, q in zip(prompts, parts)], indent=0), encoding="utf-8")
    fields = dict(stage="control_positions", model=args.model, tiny=args.tiny, freeze_tag=args.freeze_tag, run_commit=commit,
                  means_path=str(args.means), means_sha256=runinfo.sha256(args.means), c4=[c.key for c in c4],
                  n_random=args.n_random, seed=SEED, parts=list(run_parts), split_labels=args.split_labels, n_prompts=len(prompts),
                  m1_score_method="probes.label_scores_in_positions, one batch composition, no prefix cache (not --scorer)",
                  m1_selection_sha256=runinfo.sha256(m1_dir / "selection.json"))
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()

    def score(keys, part):
        lesion = ([Component.parse(k) if isinstance(k, str) else k for k in keys], means) if keys else None
        positions = None if part == "all" else [q[part] for q in parts]
        return probes.label_scores_in_positions(model, tok, prompts, chat, positions, args.batch_size, lesion)

    rows = [score([], "all").assign(state="intact", set="intact", part="none")]
    for part in ("all", *run_parts):
        rows.append(score(c4, part).assign(state=f"C4_{part}", set="C4", part=part))
    print(f"C4 done ({time.time() - t0:.0f} s)", flush=True)
    for part in run_parts:
        for i, s in enumerate(randoms):
            rows.append(score(s, part).assign(state=f"random_{i:03d}_{part}", set=f"random_{i:03d}", part=part))
        print(f"random sets: {part} ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(out / "scores.parquet", index=False)
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
