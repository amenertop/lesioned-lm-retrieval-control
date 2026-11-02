"""Pre-specified one-time expansion of the knowledge set to 8 components (spec.md G2), and paired intervals for
the two between-class differences on the dev splits (user decision 2026-09-28; agent2: a difference needs its
own interval, not two point estimates).

    python experiments/e3_knowledge/run_expand8.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 --scorer prefix_avg3

Uses outputs/e3_knowledge/<model>/screen (verified candidates) and the M1 control set. Writes
outputs/e3_knowledge/<model>/expand8/{results.json, meta.json}. Dev splits only; no held-out items are read.
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))

import common  # noqa: E402
from lesion import baselines, coder_sources, components, damage, knowledge_items as ki, materials, naming, runinfo, scoring, screen  # noqa: E402
from lesion.components import Component  # noqa: E402

BATCHES = (16, 11, 7)
SIZES = (4, 8)


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--knowledge-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    commit = common.guard(args)
    kdir = Path(args.knowledge_dir) if args.knowledge_dir else ROOT / "outputs/e3_knowledge" / args.model / "screen"
    kmeta, _, verified = coder_sources.knowledge_screen(kdir, require_lock=args.model == coder_sources.MODEL, minimum=max(SIZES))
    median = kmeta["params"]["zipf_median"]
    m1_screen = common.out_dir(args, "screen")
    m1_selection = common.load_selection(m1_screen, json.loads((m1_screen / "meta.json").read_text(encoding="utf-8")))
    control = [Component.parse(k) for k in m1_selection["control"]]

    dev = ki.load_items("dev")
    demos = pd.read_csv(ki.OUT / "demos.csv")
    m1_dev = materials.load_split("dev")
    m1_all, m1_demos, templates = scoring.load_materials()
    fm = damage.frequency_median(m1_all)
    model, tok, chat, means, ablate = common.load(args)
    out = Path(args.out) if args.out else ROOT / ("outputs/e3_knowledge_tiny" if args.tiny else "outputs/e3_knowledge") / args.model / "expand8"
    fields = common.run_fields(args, commit, "knowledge_expand8", "dev", len(dev) + len(m1_dev), sizes=list(SIZES),
                               n_random=args.n_random, knowledge_selection_sha256=kmeta["selection_sha256"],
                               knowledge_screen_sha256=kmeta.get("screen_csv_sha256"))
    common.check_resume(out, fields)
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()

    p_dev = naming.build_prompts(dev, demos, chat)

    def naming_scores(comps=()):
        with ablate(list(comps)) if comps else nullcontext():
            return naming.averaged_logprobs(model, tok, p_dev, chat, BATCHES)

    label_score = screen.make_scorer(model, tok, scoring.build_label_prompts(m1_dev, m1_demos, templates, variant="main", chat=chat),
                                     chat, ablate, args.batch_size, common.SCORERS[args.scorer])
    intact_n, intact_m1 = naming_scores(), label_score()
    lesioned_n = {"control": naming_scores(control)}
    lesioned_m1 = {"control": label_score(control)}
    results = dict(control=[c.key for c in control], sets={})
    for size in SIZES:
        keys = verified.head(size).component.tolist()
        comps = [Component.parse(k) for k in keys]
        lesioned_n[f"k{size}"], lesioned_m1[f"k{size}"] = naming_scores(comps), label_score(comps)
        # paired frames: both sets' damage on the same items
        hn = pd.concat({"knowledge": naming.item_damage(intact_n, lesioned_n[f"k{size}"]),
                        "control": naming.item_damage(intact_n, lesioned_n["control"])}, axis=1)
        fn = naming.item_frame(hn, dev, median)
        naming_ci = damage.bootstrap(fn, lambda c: {**{f"knowledge.{k}": v for k, v in naming.effects(c, "knowledge").items()},
                                                    **{f"control.{k}": v for k, v in naming.effects(c, "control").items()},
                                                    "delta_F": naming.effects(c, "knowledge")["knowledge_effect"]
                                                    - naming.effects(c, "control")["knowledge_effect"]}, args.n_boot)
        hm = pd.concat({"knowledge": damage.item_damage(intact_m1, lesioned_m1[f"k{size}"]),
                        "control": damage.item_damage(intact_m1, lesioned_m1["control"])}, axis=1)
        fm1 = damage.item_frame(hm, m1_dev, fm)
        m1_ci = damage.bootstrap(fm1, lambda c: {**{f"knowledge.{k}": v for k, v in damage.effects(c, "knowledge").items()},
                                                 **{f"control.{k}": v for k, v in damage.effects(c, "control").items()},
                                                 "delta_C": damage.effects(c, "control")["control_effect"]
                                                 - damage.effects(c, "knowledge")["control_effect"]}, args.n_boot)
        kinds = sorted({c.kind for c in comps})
        pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
        freq = means.routing_frequency() if "expert" in kinds else None
        routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if freq is not None else None
        rand = []
        for s in baselines.matched_random_sets(comps, pool, args.n_random, seed=5 + size, exclude=comps + control, routing_freq=routing):
            f = naming.item_frame(naming.item_damage(intact_n, naming_scores(s)), dev, median)
            rand.append(naming.effects({c: f[c].to_numpy() for c in f.columns})["knowledge_effect"])
        target = naming_ci["knowledge.knowledge_effect"]["estimate"]
        results["sets"][f"k{size}"] = dict(components=keys, naming=naming_ci, m1=m1_ci,
                                          random_knowledge_effect=dict(n=len(rand), median=float(np.median(rand)),
                                                                       max=float(np.max(rand)),
                                                                       at_least_target=int(np.sum(np.array(rand) >= target))))
        e, dF, dC = naming_ci["knowledge.knowledge_effect"], naming_ci["delta_F"], m1_ci["delta_C"]
        print(f"k{size}: knowledge effect {e['estimate']:.2f} [{e['ci_low']:.2f}, {e['ci_high']:.2f}]; "
              f"delta_F {dF['estimate']:.2f} [{dF['ci_low']:.2f}, {dF['ci_high']:.2f}]; "
              f"delta_C {dC['estimate']:.2f} [{dC['ci_low']:.2f}, {dC['ci_high']:.2f}]; random max {max(rand):.2f}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1),
                       results_sha256=runinfo.sha256(out / "results.json"), **fields)


if __name__ == "__main__":
    main()
