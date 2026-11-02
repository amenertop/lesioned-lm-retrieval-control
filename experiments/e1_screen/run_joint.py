"""MVP step 2: joint lesion of the top 4 experts per direction on the M1 dev split, against matched random sets.

Run after run_screen.py, with the same --model and --means:
    python experiments/e1_screen/run_joint.py --model <G1 version> --means outputs/means/<file>.pt

Writes outputs/e1_screen/<model>/joint/{sets.json, intact.csv, lesioned.parquet, results.json, meta.json}.
"""
import argparse
import json
import time

import pandas as pd

import common
from lesion import baselines, components, damage, materials, runinfo, scoring, screen
from lesion.components import Component


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--max-pct-gap", type=float, default=0.25, help="routing-frequency percentile tolerance within a layer")
    args = ap.parse_args()

    commit = common.guard(args)
    screen_dir = common.out_dir(args, "screen")
    screen_meta = json.loads((screen_dir / "meta.json").read_text(encoding="utf-8"))
    selection = common.load_selection(screen_dir, screen_meta)

    all_items, demos, templates = scoring.load_materials()
    items = materials.load_split("dev")
    model, tok, chat, means, ablate = common.load(args)
    prompts = scoring.build_label_prompts(items, demos, templates, variant="main", chat=chat)
    out = common.out_dir(args, "joint")
    fields = common.run_fields(args, commit, "joint", "dev", len(prompts),
                               n_random=args.n_random, max_pct_gap=args.max_pct_gap, n_boot=args.n_boot)
    common.check_same_run(screen_meta, fields, screen_dir)   # selection came from the same model, means, materials and commit
    common.check_resume(out, fields)                          # never mix with an older joint run in the same folder
    runinfo.write_meta(out, status="running", screen_selection_sha256=screen_meta["selection_sha256"], **fields)

    targets = {d: [Component.parse(k) for k in selection[d]] for d in screen.PRIMARY if selection["status"][d] == "ok"}
    if not targets:
        raise SystemExit(f"no direction produced {screen.K} components at screening: {selection['status']}")
    supp = [Component.parse(k) for k in selection.get("supplementary", {}).get("control_unfiltered", [])]
    if supp == targets.get("control"):
        supp = []                                 # the exclusion removed nothing: no separate supplementary set
    kinds = sorted({c.kind for s in [*targets.values(), supp] for c in s})
    candidates = [c for kind in kinds for c in components.all_components(model.config, kind)]
    freq = means.routing_frequency()
    routing = {c: float(freq[c.layer][c.index]) for c in candidates if c.kind == "expert"}
    taken = [c for s in [*targets.values(), supp] for c in s]
    sets = dict(targets)
    draws = list(targets.items()) + ([("supp_control", supp)] if supp else [])
    for seed, (d, t) in enumerate(draws):
        prefix = "supp_random_control" if d == "supp_control" else f"random_{d}"
        for i, s in enumerate(baselines.matched_random_sets(t, candidates, args.n_random, seed=seed, exclude=taken,
                                                            routing_freq=routing, max_pct_gap=args.max_pct_gap)):
            sets[f"{prefix}_{i:02d}"] = s
    if supp:
        sets["supp_control"] = supp
    (out / "sets.json").write_text(json.dumps({k: [c.key for c in v] for k, v in sets.items()}, indent=2), encoding="utf-8")

    t0 = time.time()
    score = screen.make_scorer(model, tok, prompts, chat, ablate, args.batch_size, common.SCORERS[args.scorer])
    intact = score()
    intact.to_csv(out / "intact.csv", index=False)
    lesioned = {name: score(comps) for name, comps in sets.items()}
    pd.concat(lesioned, names=["set"]).reset_index(level=0).to_parquet(out / "lesioned.parquet", index=False)

    main_sets = {n: v for n, v in lesioned.items() if not n.startswith("supp_")}
    points, intervals, random_rank = screen.compare_sets(intact, main_sets, items, selection["freq_median"], args.n_boot)
    result = dict(points=points, intervals=intervals, random_rank=random_rank, n_boot=args.n_boot, n_random=args.n_random,
                  n_items=len(items), n_concept_groups=items.concept_group.nunique())
    if supp:
        # the same comparison with the control set chosen without the domain-general exclusion (ADR 0005)
        alt = {"control": lesioned["supp_control"], **({"knowledge": lesioned["knowledge"]} if "knowledge" in lesioned else {})}
        alt.update({n.replace("supp_random_control", "random_control"): v for n, v in lesioned.items()
                    if n.startswith("supp_random_control")})
        sp, si, sr = screen.compare_sets(intact, alt, items, selection["freq_median"], args.n_boot)
        result["supplementary_unfiltered_control"] = dict(points=sp, intervals=si, random_rank=sr,
                                                          components=[c.key for c in supp])
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    runinfo.write_meta(out, status="done", screen_selection_sha256=screen_meta["selection_sha256"], n_sets=len(sets), seconds=round(time.time() - t0, 1),
                       random_matching="layer (+ routing-frequency percentile for experts, on neutral text; MVP simplification)", **fields)
    for d in targets:
        e = intervals[f"{d}.{screen.PRIMARY[d]}"]
        print(f"{d}: {screen.PRIMARY[d]} = {e['estimate']:.3f} [{e['ci_low']:.3f}, {e['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
