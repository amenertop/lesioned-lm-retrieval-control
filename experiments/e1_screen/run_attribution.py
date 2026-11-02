"""Estimate every expert's single-ablation effects with one backward pass per batch (ADR 0004).

    python experiments/e1_screen/run_attribution.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16

Writes outputs/e1_screen/<model>/attribution/{attribution.csv, meta.json}: per expert, the first-order
estimate of the four damage statistics (lesion.damage.effects) on the screening split. run_screen.py
--candidates then checks the top experts with real ablations.
"""
import argparse
import time

import common
from lesion import attribution, damage, materials, runinfo, scoring


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--kinds", default="expert,head", help="component kinds to estimate (ADR 0005 adds attention heads)")
    args = ap.parse_args()
    kinds = tuple(args.kinds.split(","))

    commit = common.guard(args)
    all_items, demos, templates = scoring.load_materials()
    items = materials.load_split("screening")
    model, tok, chat, means, _ = common.load(args)
    prompts = scoring.build_label_prompts(items, demos, templates, variant="main", chat=chat)
    out = common.out_dir(args, "attribution")
    fields = common.run_fields(args, commit, "attribution", "screening", len(prompts), method="attribution patching",
                               kinds=list(kinds))
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()
    weights = attribution.statistic_weights(prompts, items, damage.frequency_median(all_items))
    est = attribution.component_attribution(model, tok, prompts, chat, means, weights, args.batch_size, kinds=kinds)
    est.to_csv(out / "attribution.csv", index=False)
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), n_experts=len(est), **fields)
    for col in ("control_effect", "knowledge_effect"):
        print(col, "top:", est.nlargest(5, col)[["component", col]].to_string(index=False))


if __name__ == "__main__":
    main()
