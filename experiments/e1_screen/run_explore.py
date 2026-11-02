"""Development-split explorations of the chosen control set (user decision 2026-09-28):

1. dose-response: mean ablation at alpha = 0.75, 0.5, 0.25, 0 (1 = intact);
2. another way to lesion: zero ablation (alpha = 0);
3. where the control heads look: attention from the answer position to the correct option and to the
   distractor, against heads from the same layers drawn at random.

    python experiments/e1_screen/run_explore.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 --scorer prefix_avg3

Writes outputs/e1_screen/<model>/explore/{dose.json, attention.csv, attention_summary.json, meta.json}.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import common
from lesion import attention, baselines, components, damage, hooks, materials, runinfo, scoring, screen
from lesion.components import Component

ALPHAS = (0.75, 0.5, 0.25, 0.0)


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--n-random-heads", type=int, default=20)
    args = ap.parse_args()

    commit = common.guard(args)
    screen_dir = common.out_dir(args, "screen")
    screen_meta = json.loads((screen_dir / "meta.json").read_text(encoding="utf-8"))
    selection = common.load_selection(screen_dir, screen_meta)
    all_items, demos, templates = scoring.load_materials()
    items = materials.load_split("dev")
    model, tok, chat, means, _ = common.load(args)
    prompts = scoring.build_label_prompts(items, demos, templates, variant="main", chat=chat)
    out = common.out_dir(args, "explore")
    fields = common.run_fields(args, commit, "explore", "dev", len(prompts), alphas=list(ALPHAS),
                               n_random_heads=args.n_random_heads, screen_commit=screen_meta.get("resume_commit"),
                               code_changes_since_screen=common.changes_since(screen_meta.get("resume_commit"), commit))
    # reuses a finished selection (hash checked by load_selection) at a newer commit
    common.check_same_run(screen_meta, fields, screen_dir, same_code=False,
                          keys=[k for k in common.SAME_RUN if k != "analysis_plan_sha256"])
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()
    control = [Component.parse(k) for k in selection["control"]]
    fm = selection["freq_median"]

    # 1-2. dose-response and zero ablation
    def lesion_under(alpha, mode):
        return lambda comps: hooks.ablate(model, comps, None if mode == "zero" else means, alpha=alpha, mode=mode)

    base = screen.make_scorer(model, tok, prompts, chat, lesion_under(1.0, "mean"), args.batch_size, common.SCORERS[args.scorer])
    intact = base()
    dose = []
    for alpha, mode in [(a, "mean") for a in ALPHAS] + [(0.0, "zero")]:
        score = screen.make_scorer(model, tok, prompts, chat, lesion_under(alpha, mode), args.batch_size, common.SCORERS[args.scorer])
        frame = damage.item_frame(damage.item_damage(intact, score(control)), items, fm)
        ci = damage.bootstrap(frame, damage.effects, args.n_boot)
        dose.append(dict(alpha=alpha, mode=mode, **{k: ci[k] for k in ("control_effect", "semantic_damage", "string_damage", "knowledge_effect")}))
        print(f"{mode} alpha={alpha}: control effect {ci['control_effect']['estimate']:.2f} "
              f"[{ci['control_effect']['ci_low']:.2f}, {ci['control_effect']['ci_high']:.2f}]", flush=True)
    (out / "dose.json").write_text(json.dumps(dose, indent=2), encoding="utf-8")

    # 3. attention of the control heads against random heads from the same layers (only if the set is heads)
    heads = [c for c in control if c.kind == "head"]
    if heads:
        pool = components.all_components(model.config, "head")
        randoms = baselines.matched_random_sets(heads, pool, args.n_random_heads, seed=7, exclude=heads)
        random_heads = sorted({h for s in randoms for h in s})
        fm_prompts = [p for p in prompts if p.task in damage.MATCHING]
        import torch

        previous = model.config._attn_implementation
        model.config._attn_implementation = "eager"           # attention weights are only returned by eager attention
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            att = attention.head_attention(model, tok, fm_prompts, chat, heads + random_heads, batch_size=4)
        finally:
            model.config._attn_implementation = previous
        attention_run = dict(implementation="eager", restored_to=previous, heads=[h.key for h in heads],
                             random_heads=[h.key for h in random_heads],
                             peak_gpu_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
        meta = items.set_index("item_id")
        att["high_demand"] = att.item_id.map(meta.high_demand)
        att["group"] = np.where(att["head"].isin([h.key for h in heads]), "control", "random")
        att["wrong_minus_correct"] = att.to_wrong - att.to_correct
        att.to_csv(out / "attention.csv", index=False)
        summary = (att.groupby(["group", "high_demand"])[["to_correct", "to_wrong", "wrong_minus_correct"]].mean()
                   .reset_index().to_dict("records"))
        per_head = (att[att.group == "control"].groupby(["head", "high_demand"])[["to_correct", "to_wrong"]].mean()
                    .reset_index().to_dict("records"))
        (out / "attention_summary.json").write_text(json.dumps(dict(by_group=summary, control_heads=per_head, run=attention_run),
                                                               indent=2), encoding="utf-8")
        print(pd.DataFrame(summary).round(4).to_string(index=False))
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)


if __name__ == "__main__":
    main()
