"""GPU session 12, item 5 (docs/plans/gpu-session12.md), exploratory re-screen for a second knowledge set K8b, not
overlapping K8. In one process:
1. candidates: the top 40 of the naming attribution table by knowledge effect without K8's 8 (K8 came from the top
   32 of the same table);
2. each candidate ablated alone: naming screening items (real knowledge effect and naming damage, scores averaged
   over three batch compositions as in the knowledge screen), M1 screening items (semantic and string damage, and the
   margin-scaling slope over all M1 items, lesion.damage.margin_scaling), language-model loss on 40 neutral passages
   of 128 tokens; experts also get their routing frequency, mean-output norm and a candidate-super-expert label
   (mean-output norm over 20 times the layer's median; a label only);
3. 100 random single components drawn like the candidates (same kind and layer): M1 slope and LM loss, the null
   whose 99th percentile (LM loss) and 1st to 99th percentiles (slope) set the two exclusions (rescreen.select_k8b);
4. K8b by K8's rule among the candidates not excluded;
5. dev items: K8b against matched random sets on naming and M1, with set-level LM loss and slope.

    python experiments/e3_knowledge/run_rescreen_k8b.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3

Writes outputs/e3_knowledge/<model>/k8b/{candidates.csv, screen.csv, null.csv, selection.json, joint.json, meta.json}.
The held-out test is run_dissociation_test.py --knowledge-set <K8b> --control-set <C4> (experiments/gpu_session12.sh).
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
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))

import common  # noqa: E402
import run_dissociation_test  # noqa: E402  (the frozen K8)
from lesion import (baselines, components, damage, knowledge_items as ki, materials, naming, neutral_text, probes,  # noqa: E402
                    rescreen, runinfo, scoring, screen)
from lesion.components import Component  # noqa: E402

BATCHES = (16, 11, 7)      # naming scores averaged over these batch compositions, as in the knowledge screen
N_TEXTS, N_TOKENS = 40, 128


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--top", type=int, default=rescreen.K8B_TOP)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--force-set", default=None, help="tiny rehearsals only: comma-separated set used as K8b, to run "
                                                       "the dev stage when the random tiny model selects nothing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.force_set and not args.tiny:
        raise SystemExit("--force-set is for tiny rehearsals only")

    commit = common.guard(args)
    kroot = ROOT / ("outputs/e3_knowledge_tiny" if args.tiny else "outputs/e3_knowledge") / args.model
    kdir = kroot / "screen"
    if json.loads((kdir / "meta.json").read_text(encoding="utf-8")).get("status") != "done":
        raise SystemExit(f"{kdir} is unfinished")
    naming_attr = pd.read_csv(kdir / "attribution.csv")
    # K8 as the dissociation test defines it: the top 8 verified candidates of the knowledge screen (its selection.json
    # holds their top 4); for the main model it must be the frozen K8
    k8 = pd.read_csv(kdir / "screen.csv").sort_values("knowledge_effect", ascending=False).component.head(8).tolist()
    top4 = json.loads((kdir / "selection.json").read_text(encoding="utf-8"))["knowledge"]
    if k8[:len(top4)] != top4 or (args.model == run_dissociation_test.K8_MODEL and not args.tiny
                                  and tuple(k8) != run_dissociation_test.K8):
        raise SystemExit(f"the knowledge screen gives {k8}, not the frozen K8 (selection {top4})")
    m1_dir = common.out_dir(args, "screen")
    c4 = common.load_selection(m1_dir, json.loads((m1_dir / "meta.json").read_text(encoding="utf-8")))["control"]

    all_meta = pd.read_csv(ki.OUT / "items.csv")[["item_id", "zipf"]]
    median = naming.zipf_median(all_meta)
    n_scr, n_dev = ki.load_items("screening"), ki.load_items("dev")
    demos = pd.read_csv(ki.OUT / "demos.csv")
    all_m1, m1_demos, templates = scoring.load_materials()
    all_m1 = all_m1.astype({"item_id": str})
    fm = damage.frequency_median(all_m1)
    m_scr, m_dev = materials.load_split("screening"), materials.load_split("dev")

    model, tok, chat, means, ablate = common.load(args)
    pad = tok.eos_token_id or 0
    texts = neutral_text.load()[:N_TEXTS]
    lm_seqs = [tok(t, add_special_tokens=False).input_ids[:N_TOKENS] for t in texts]
    out = Path(args.out) if args.out else kroot / "k8b"
    fields = common.run_fields(args, commit, "k8b", "screening+dev", len(n_scr) + len(n_dev) + len(m_scr) + len(m_dev),
                               top=args.top, n_null=args.n_null, n_random=args.n_random, batches=list(BATCHES),
                               zipf_median=median, k8=k8, c4=c4, items_sha256=runinfo.sha256(ki.OUT / "items.csv"),
                               attribution_sha256=runinfo.sha256(kdir / "attribution.csv"))
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()
    log = lambda m: print(f"{m} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731

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

    def m1_profile(intact, lesioned, items):
        i, l = intact.astype({"item_id": str}), lesioned.astype({"item_id": str})
        frame = damage.item_frame(damage.item_damage(i, l), all_m1, fm)
        e = rescreen.m1_effects_with_synonym(damage.columns(frame))
        slope = damage.margin_scaling(i, l, all_m1).set_index("task").slope
        return dict(m1_semantic_damage=e["m1_semantic_damage"], m1_string_damage=e["m1_string_damage"],
                    m1_control_effect=e["m1_control_effect"], m1_slope=float(slope["all"]))

    lm_intact = probes.lm_loss_region(model, lm_seqs, [], means, pad_id=pad)

    def lm_change(comps):
        return probes.lm_loss_region(model, lm_seqs, comps, means, pad_id=pad) / lm_intact - 1

    flags = rescreen.super_expert_flags(means.expert_mean) if means.expert_mean is not None else {}
    freq = means.routing_frequency() if means.expert_count is not None else None

    def expert_info(c):
        if c.kind != "expert":
            return {}
        norms = means.expert_mean[c.layer].float().norm(dim=-1).numpy()
        return dict(routing_frequency=float(freq[c.layer][c.index]) if freq is not None else np.nan,
                    mean_output_norm=float(norms[c.index]), norm_over_layer_median=float(norms[c.index] / np.median(norms)),
                    candidate_super_expert=(c.layer, c.index) in flags)

    # 1. candidates
    cand = rescreen.k8b_candidates(naming_attr, k8, args.top)
    attr = naming_attr.set_index("component")
    pd.DataFrame(dict(component=cand, attribution_knowledge_effect=[attr.loc[k, "knowledge_effect"] for k in cand])
                 ).to_csv(out / "candidates.csv", index=False)
    log(f"{len(cand)} candidates")

    # 2. each candidate alone on the screening items
    sn, sm = naming_scorer(n_scr), label_scorer(m_scr)
    intact_n, intact_m = sn(), sm()
    rows = []
    for i, key in enumerate(cand, 1):
        c = [Component.parse(key)]
        rows.append(dict(component=key, **naming_effects(intact_n, sn(c), n_scr), **m1_profile(intact_m, sm(c), m_scr),
                         lm_loss_change=lm_change(c), **expert_info(c[0])))
        if i % 8 == 0:
            pd.DataFrame(rows).to_csv(out / "screen.csv", index=False)
            log(f"{i}/{len(cand)} candidates")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "screen.csv", index=False)

    # 3. random single components: the null for the two exclusions
    pool = [c for kind in ("expert", "head") for c in components.all_components(model.config, kind)]
    exclude = [Component.parse(k) for k in k8 + c4]
    nulls = rescreen.null_singles([Component.parse(k) for k in cand], pool, args.n_null, seed=13, exclude=exclude)
    null = pd.DataFrame([dict(component=c.key, **m1_profile(intact_m, sm([c]), m_scr), lm_loss_change=lm_change([c]),
                              **expert_info(c)) for c in nulls])
    null.to_csv(out / "null.csv", index=False)
    log(f"{len(nulls)} random single components")

    # 4. K8b
    selection = rescreen.select_k8b(summary, null)
    selection.update(k8=k8, c4=c4, lm_intact=lm_intact,
                     candidate_super_experts=[k for k in cand if Component.parse(k).kind == "expert"
                                              and (Component.parse(k).layer, Component.parse(k).index) in flags])
    if args.force_set:
        selection.update(k8b=args.force_set.split(","), status="forced (tiny rehearsal)")
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    log(f"K8b: {selection['k8b']} status: {selection['status']}")

    # 5. dev items: K8b against matched random sets
    joint = dict(selection=selection)
    if selection["k8b"]:
        k8b = [Component.parse(k) for k in selection["k8b"]]
        kinds = sorted({c.kind for c in k8b})
        set_pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
        routing = ({c: float(freq[c.layer][c.index]) for c in set_pool if c.kind == "expert"}
                   if freq is not None and "expert" in kinds else None)
        randoms = (baselines.matched_random_sets(k8b, set_pool, args.n_random, seed=7, exclude=k8b + exclude,
                                                 routing_freq=routing) if args.n_random else [])
        dn, dm = naming_scorer(n_dev), label_scorer(m_dev)
        in_dev, im_dev = dn(), dm()
        joint["k8b_on_naming"] = naming_effects(in_dev, dn(k8b), n_dev, boot=True)
        joint["k8b_on_m1"] = m1_profile(im_dev, dm(k8b), m_dev)
        joint["k8b_lm_loss_change"] = lm_change(k8b)
        rand = []
        for s in randoms:
            rand.append(dict(set=[c.key for c in s], knowledge_effect=naming_effects(in_dev, dn(s), n_dev)["knowledge_effect"],
                             lm_loss_change=lm_change(s), **m1_profile(im_dev, dm(s), m_dev)))
        joint["random"] = rand
        if rand:
            r = pd.DataFrame(rand)
            joint["specific"] = dict(
                lm_loss_within_random=bool(joint["k8b_lm_loss_change"] <= r.lm_loss_change.max()),
                slope_within_random=bool(r.m1_slope.min() <= joint["k8b_on_m1"]["m1_slope"] <= r.m1_slope.max()),
                knowledge_effect_random_max=float(r.knowledge_effect.max()))
        k = joint["k8b_on_naming"]["knowledge_effect"]
        log(f"dev: knowledge effect {k['estimate']:.3f} [{k['ci_low']:.3f}, {k['ci_high']:.3f}]")
    (out / "joint.json").write_text(json.dumps(joint, indent=2, default=float), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
