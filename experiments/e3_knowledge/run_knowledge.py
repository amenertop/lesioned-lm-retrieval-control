"""Knowledge-side screen on the definition-naming test (user decisions 2026-09-28), in one process so the
model is loaded once:

1. attribution over all experts and heads on the screening split (knowledge effect: rare-minus-common drop
   in the answer's log-probability);
2. real mean ablation of the top `--top` components, scores averaged over batch compositions; top 4 chosen;
3. joint lesion on the dev split against matched random sets; plus, as a preview of the two between-class
   differences, the control heads (from the M1 screen) on the same naming items and the knowledge set on
   the M1 dev items.

    python experiments/e3_knowledge/run_knowledge.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16

Writes outputs/e3_knowledge/<model>/screen/{attribution.csv, screen.csv, selection.json, joint.json, meta.json}.
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
from lesion import (attribution, baselines, components, damage, knowledge_items as ki, materials, naming,  # noqa: E402
                    runinfo, scoring, screen)
from lesion.components import Component  # noqa: E402

K = 4
BATCHES = (16, 11, 7)


def acceptable(top_rows):
    """A knowledge set is chosen only if there are K verified components, all with a positive knowledge effect."""
    return len(top_rows) == K and bool((top_rows.knowledge_effect > 0).all())


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--top", type=int, default=32)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    commit = common.guard(args)
    all_meta = pd.read_csv(ki.OUT / "items.csv")[["item_id", "zipf"]]      # item properties only, no scores
    median = naming.zipf_median(all_meta)
    scr, dev = ki.load_items("screening"), ki.load_items("dev")
    demos = pd.read_csv(ki.OUT / "demos.csv")
    m1_screen = common.out_dir(args, "screen")
    m1_meta = json.loads((m1_screen / "meta.json").read_text(encoding="utf-8"))
    m1_selection = common.load_selection(m1_screen, m1_meta)
    model, tok, chat, means, ablate = common.load(args)
    out = Path(args.out) if args.out else ROOT / ("outputs/e3_knowledge_tiny" if args.tiny else "outputs/e3_knowledge") / args.model / "screen"
    fields = common.run_fields(args, commit, "knowledge_screen", "screening+dev", len(scr) + len(dev), top=args.top,
                               n_random=args.n_random, batches=list(BATCHES),
                               items_sha256=runinfo.sha256(ki.OUT / "items.csv"), zipf_median=median)
    common.check_resume(out, fields)
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()

    def scorer(prompts):
        def score(comps=()):
            with ablate(list(comps)) if comps else nullcontext():
                return naming.averaged_logprobs(model, tok, prompts, chat, BATCHES)
        return score

    def naming_effects(intact, lesioned, items, boot=False):
        frame = naming.item_frame(naming.item_damage(intact, lesioned), items, median)
        cols = {c: frame[c].to_numpy() for c in frame.columns}
        return damage.bootstrap(frame, naming.effects, args.n_boot) if boot else naming.effects(cols)

    # 1. attribution on the screening split
    p_scr = naming.build_prompts(scr, demos, chat)
    est = attribution.naming_attribution(model, tok, p_scr, chat, means, naming.statistic_weights(scr, median), batch_size=8)
    est.to_csv(out / "attribution.csv", index=False)
    print("attribution top:", est.nlargest(5, "knowledge_effect")[["component", "knowledge_effect"]].to_string(index=False), flush=True)

    # 2. real ablation of the top candidates
    score_scr = scorer(p_scr)
    intact_scr = score_scr()
    rows = []
    for key in est.nlargest(args.top, "knowledge_effect").component:
        e = naming_effects(intact_scr, score_scr([Component.parse(key)]), scr)
        rows.append(dict(component=key, **e))
    summary = pd.DataFrame(rows).sort_values("knowledge_effect", ascending=False)
    summary.to_csv(out / "screen.csv", index=False)
    chosen = summary.head(K).component.tolist()
    ok = acceptable(summary.head(K))
    selection = dict(knowledge=chosen if ok else [], status=dict(knowledge="ok" if ok else "not_found"), k=K, zipf_median=median,
                     rule=f"top {args.top} by attribution, verified by real ablation; the {K} largest knowledge effects, all above 0")
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("knowledge set:", chosen, "status:", selection["status"], flush=True)

    # 3. dev split: joint lesion against random sets, plus the cross-class previews
    joint = dict(selection=selection)
    if ok:
        knowledge = [Component.parse(k) for k in chosen]
        control = [Component.parse(k) for k in m1_selection["control"]]
        kinds = sorted({c.kind for c in knowledge})
        pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
        freq = means.routing_frequency() if "expert" in kinds else None
        routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if freq is not None else None
        randoms = baselines.matched_random_sets(knowledge, pool, args.n_random, seed=5, exclude=knowledge + control,
                                                routing_freq=routing)
        p_dev = naming.build_prompts(dev, demos, chat)
        score_dev = scorer(p_dev)
        intact_dev = score_dev()
        joint["knowledge_set_on_naming"] = naming_effects(intact_dev, score_dev(knowledge), dev, boot=True)
        joint["control_set_on_naming"] = naming_effects(intact_dev, score_dev(control), dev, boot=True)
        rand = [naming_effects(intact_dev, score_dev(s), dev)["knowledge_effect"] for s in randoms]
        target = joint["knowledge_set_on_naming"]["knowledge_effect"]["estimate"]
        joint["random_knowledge_effect"] = dict(n=len(rand), median=float(np.median(rand)), max=float(np.max(rand)),
                                               at_least_target=int(np.sum(np.array(rand) >= target)))
        # the knowledge set on the M1 dev items (control effect), next to the control set's
        m1_dev = materials.load_split("dev")
        _, m1_demos, templates = scoring.load_materials()
        label_prompts = scoring.build_label_prompts(m1_dev, m1_demos, templates, variant="main", chat=chat)
        label_score = screen.make_scorer(model, tok, label_prompts, chat, ablate, args.batch_size, common.SCORERS[args.scorer])
        intact_m1 = label_score()
        fm = damage.frequency_median(scoring.load_materials()[0])
        for name, comps in (("knowledge_set_on_m1", knowledge), ("control_set_on_m1", control)):
            frame = damage.item_frame(damage.item_damage(intact_m1, label_score(comps)), m1_dev, fm)
            joint[name] = damage.bootstrap(frame, damage.effects, args.n_boot)
        joint["preview_delta_C"] = (joint["control_set_on_m1"]["control_effect"]["estimate"]
                                    - joint["knowledge_set_on_m1"]["control_effect"]["estimate"])
        joint["preview_delta_F"] = (joint["knowledge_set_on_naming"]["knowledge_effect"]["estimate"]
                                    - joint["control_set_on_naming"]["knowledge_effect"]["estimate"])
        k = joint["knowledge_set_on_naming"]["knowledge_effect"]
        print(f"dev knowledge effect {k['estimate']:.3f} [{k['ci_low']:.3f}, {k['ci_high']:.3f}]; random max "
              f"{joint['random_knowledge_effect']['max']:.3f}; preview dC {joint['preview_delta_C']:.2f}, dF {joint['preview_delta_F']:.2f}",
              flush=True)
    (out / "joint.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1),
                       selection_sha256=runinfo.sha256(out / "selection.json"),
                       screen_csv_sha256=runinfo.sha256(out / "screen.csv"), **fields)


if __name__ == "__main__":
    main()
