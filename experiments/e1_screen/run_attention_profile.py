"""Where the control heads look, by prompt region (exploration, dev split; follow-up to run_explore.py).

    python experiments/e1_screen/run_attention_profile.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16

For the chosen control heads and heads from the same layers drawn at random, the share of the answer
position's attention on each part of the prompt (lesion.attention.REGIONS), for high- and low-interference
colour and size items. Writes outputs/e1_screen/<model>/attention_profile/{profile.csv, summary.json, meta.json}.
"""
import argparse
import json
import time

import numpy as np

import common
from lesion import attention, baselines, components, damage, materials, runinfo, scoring
from lesion.components import Component


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--n-random-heads", type=int, default=20)
    args = ap.parse_args()

    commit = common.guard(args)
    screen_dir = common.out_dir(args, "screen")
    screen_meta = json.loads((screen_dir / "meta.json").read_text(encoding="utf-8"))
    selection = common.load_selection(screen_dir, screen_meta)
    heads = [Component.parse(k) for k in selection["control"] if k.startswith("head:")]
    if not heads:
        raise SystemExit("the control set has no attention heads")
    items = materials.load_split("dev")
    _, demos, templates = scoring.load_materials()
    model, tok, chat, _, _ = common.load(args)
    prompts = [p for p in scoring.build_label_prompts(items, demos, templates, variant="main", chat=chat)
               if p.task in damage.MATCHING]
    out = common.out_dir(args, "attention_profile")
    fields = common.run_fields(args, commit, "attention_profile", "dev", len(prompts), n_random_heads=args.n_random_heads,
                               screen_commit=screen_meta.get("resume_commit"))
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()
    pool = components.all_components(model.config, "head")
    random_heads = sorted({h for s in baselines.matched_random_sets(heads, pool, args.n_random_heads, seed=7, exclude=heads) for h in s})
    previous = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    try:
        prof = attention.attention_profile(model, tok, prompts, chat, heads + random_heads, batch_size=4)
    finally:
        model.config._attn_implementation = previous
    prof["high_demand"] = prof.item_id.map(items.set_index("item_id").high_demand)
    prof["group"] = np.where(prof["head"].isin([h.key for h in heads]), "control", "random")
    prof.to_csv(out / "profile.csv", index=False)
    by_group = prof.groupby(["group", "high_demand", "region"]).share.mean().unstack("region").round(4)
    per_head = prof[prof.group == "control"].groupby(["head", "region"]).share.mean().unstack("region").round(4)
    (out / "summary.json").write_text(json.dumps(dict(by_group=by_group.reset_index().to_dict("records"),
                                                      control_heads=per_head.reset_index().to_dict("records"),
                                                      implementation="eager", restored_to=previous), indent=2), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)
    print(by_group.to_string())
    print(per_head.to_string())


if __name__ == "__main__":
    main()
