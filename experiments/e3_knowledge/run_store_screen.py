"""Exploratory re-screen for a "knowledge store" (user decision 2026-10-13, docs/plans/gpu-session11.md): components
whose ablation hurts both naming (more for rare words) and comprehension, as semantic dementia does. K8 was chosen
on naming alone and spares comprehension; this screen asks the whole profile. In one process:
1. candidates from the two attribution tables that already cover every expert and head: the naming screen's
   (knowledge effect, naming damage) and the M1 screen's (semantic damage, string damage): the top `--top` by naming
   knowledge effect, by M1 semantic damage, and by the smaller of the two standardised damages;
2. real mean ablation of each candidate on the naming screening items and the M1 screening items;
3. the store set: candidates with a positive naming knowledge effect, naming damage and M1 semantic damage that are
   not domain-general (string damage above semantic damage), ranked by the smaller of their standardised naming and
   semantic damages; the top 8 (fewer if fewer qualify, status "short"; under 4, "not_found");
4. dev items: the store set against matched random sets, on naming and on M1.

    python experiments/e3_knowledge/run_store_screen.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3

Writes outputs/e3_knowledge/<model>/store_screen/{candidates.csv, screen.csv, selection.json, joint.json, meta.json}.
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
from lesion import (baselines, components, damage, knowledge_items as ki, materials, naming, rescreen, runinfo,  # noqa: E402
                    scoring, screen)
from lesion.components import Component  # noqa: E402

BATCHES = (16, 11, 7)      # naming scores averaged over these batch compositions, as in the knowledge screen


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    commit = common.guard(args)
    kroot = ROOT / ("outputs/e3_knowledge_tiny" if args.tiny else "outputs/e3_knowledge") / args.model
    attr_dirs = {"naming": kroot / "screen", "m1": common.out_dir(args, "attribution")}
    for d in attr_dirs.values():
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("status") != "done":
            raise SystemExit(f"{d} is unfinished")
    naming_attr = pd.read_csv(attr_dirs["naming"] / "attribution.csv")
    m1_attr = pd.read_csv(attr_dirs["m1"] / "attribution.csv")
    m1_sel = common.load_selection(common.out_dir(args, "screen"), json.loads((common.out_dir(args, "screen") / "meta.json").read_text(encoding="utf-8")))
    knowledge_sel = json.loads((attr_dirs["naming"] / "selection.json").read_text(encoding="utf-8"))

    all_meta = pd.read_csv(ki.OUT / "items.csv")[["item_id", "zipf"]]
    median = naming.zipf_median(all_meta)
    n_scr, n_dev = ki.load_items("screening"), ki.load_items("dev")
    demos = pd.read_csv(ki.OUT / "demos.csv")
    all_m1, m1_demos, templates = scoring.load_materials()
    fm = damage.frequency_median(all_m1)
    m_scr, m_dev = materials.load_split("screening"), materials.load_split("dev")

    model, tok, chat, means, ablate = common.load(args)
    out = Path(args.out) if args.out else kroot / "store_screen"
    fields = common.run_fields(args, commit, "store_screen", "screening+dev", len(n_scr) + len(n_dev) + len(m_scr) + len(m_dev),
                               top=args.top, n_random=args.n_random, batches=list(BATCHES), zipf_median=median,
                               items_sha256=runinfo.sha256(ki.OUT / "items.csv"),
                               attribution_sha256={k: runinfo.sha256(d / "attribution.csv") for k, d in attr_dirs.items()})
    common.check_resume(out, fields)
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()

    def naming_scorer(items):
        prompts = naming.build_prompts(items, demos, chat)

        def score(comps=()):
            with ablate(list(comps)) if comps else nullcontext():
                return naming.averaged_logprobs(model, tok, prompts, chat, BATCHES)
        return score

    def label_scorer(items):
        prompts = scoring.build_label_prompts(items, m1_demos, templates, variant="main", chat=chat)
        return screen.make_scorer(model, tok, prompts, chat, ablate, args.batch_size, common.SCORERS[args.scorer])

    def naming_effects(intact, lesioned, items, boot=False):
        frame = naming.item_frame(naming.item_damage(intact, lesioned), items, median)
        return damage.bootstrap(frame, naming.effects, args.n_boot) if boot else naming.effects(damage.columns(frame))

    def m1_effects(intact, lesioned, items, boot=False):
        frame = damage.item_frame(damage.item_damage(intact, lesioned), items, fm)
        return (damage.bootstrap(frame, rescreen.m1_effects_with_synonym, args.n_boot) if boot
                else rescreen.m1_effects_with_synonym(damage.columns(frame)))

    # 1. candidates from the attribution tables
    cand = rescreen.store_candidates(naming_attr, m1_attr, args.top)
    cand.to_csv(out / "candidates.csv", index=False)
    print(f"{len(cand)} candidates", flush=True)

    # 2. real ablation of each candidate on the screening items
    sn, sm = naming_scorer(n_scr), label_scorer(m_scr)
    intact_n, intact_m = sn(), sm()
    rows = []
    for i, key in enumerate(cand.component, 1):
        c = [Component.parse(key)]
        rows.append(dict(component=key, **naming_effects(intact_n, sn(c), n_scr), **m1_effects(intact_m, sm(c), m_scr)))
        if i % 8 == 0:
            pd.DataFrame(rows).to_csv(out / "screen.csv", index=False)
            print(f"{i}/{len(cand)} candidates, {time.time() - t0:.0f} s", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "screen.csv", index=False)
    selection = rescreen.select_store(summary)
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("store set:", selection["store"], "status:", selection["status"], flush=True)

    # 3. dev items: the store set against matched random sets
    joint = dict(selection=selection)
    if selection["store"]:
        store = [Component.parse(k) for k in selection["store"]]
        exclude = store + [Component.parse(k) for k in m1_sel["control"]] + [Component.parse(k) for k in knowledge_sel["knowledge"]]
        kinds = sorted({c.kind for c in store})
        pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
        freq = means.routing_frequency() if "expert" in kinds else None
        routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if freq is not None else None
        randoms = (baselines.matched_random_sets(store, pool, args.n_random, seed=7, exclude=exclude, routing_freq=routing)
                   if args.n_random else [])
        dn, dm = naming_scorer(n_dev), label_scorer(m_dev)
        in_dev, im_dev = dn(), dm()
        joint["store_on_naming"] = naming_effects(in_dev, dn(store), n_dev, boot=True)
        joint["store_on_m1"] = m1_effects(im_dev, dm(store), m_dev, boot=True)
        rand = [dict(knowledge_effect=naming_effects(in_dev, dn(s), n_dev)["knowledge_effect"],
                     m1_semantic_damage=m1_effects(im_dev, dm(s), m_dev)["m1_semantic_damage"]) for s in randoms]
        joint["random"] = {k: dict(n=len(rand), median=float(np.median([r[k] for r in rand])) if rand else None,
                                   max=float(np.max([r[k] for r in rand])) if rand else None)
                           for k in ("knowledge_effect", "m1_semantic_damage")}
        k, s = joint["store_on_naming"]["knowledge_effect"], joint["store_on_m1"]["m1_semantic_damage"]
        print(f"dev: knowledge effect {k['estimate']:.3f} [{k['ci_low']:.3f}, {k['ci_high']:.3f}], semantic damage "
              f"{s['estimate']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]; random max {joint['random']['knowledge_effect']['max']}, "
              f"{joint['random']['m1_semantic_damage']['max']}", flush=True)
    (out / "joint.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1),
                       selection_sha256=runinfo.sha256(out / "selection.json"), **fields)


if __name__ == "__main__":
    main()
