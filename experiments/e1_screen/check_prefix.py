"""Check on a real model that prefix-shared scoring matches full scoring (both with the float32 output
layer), and time both (ADR 0003).

    python experiments/e1_screen/check_prefix.py --model qwen3-30b-a3b-instruct --means outputs/means/<file>.pt

Scores a sample of screening prompts intact and with a few single experts mean-ablated, with both
scorers; writes outputs/e1_screen/<model>/check_prefix.json with the largest and mean logit differences
and, for each lesion, how much the margins move against the intact model under each scorer and how well
those moves agree; plus seconds per full screening pass for each scorer.
"""
import argparse
import json
import time

import numpy as np

import common
from lesion import components, materials, prefix_scoring, scoring

# Fixed before the check is run on a real model (agent2 2026-09-28): bf16 internals make even full scoring
# vary with batch size, so the prefix scorer must stay within that noise floor: its mean |margin
# difference| from full scoring at most 1.5 x the floor + 0.02 logit, and, for lesions that move margins
# by more than 0.01 on average, lesion-induced moves correlating with full scoring's at least as well as
# the floor minus 0.05.
RULE = "prefix vs full <= 1.5 x floor + 0.02; lesion-move correlation >= floor - 0.05 (floor: full at batch 16 vs 32)"


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--n-experts", type=int, default=3)
    args = ap.parse_args()

    _, demos, templates = scoring.load_materials()
    items = materials.load_split("screening")
    model, tok, chat, means, ablate = common.load(args)
    prompts = scoring.build_label_prompts(items, demos, templates, variant="main", chat=chat)
    experts = components.all_components(model.config, "expert")
    counts = means.expert_count
    busiest = sorted(experts, key=lambda c: -int(counts[c.layer][c.index]))[:args.n_experts]

    result = dict(model=args.model, n_prompts=len(prompts), rule=RULE, lesions={})
    margin = lambda s: (s.logit_correct - s.logit_wrong).to_numpy()
    runs = (("full", prefix_scoring.score_label_prompts_full, 32), ("full_alt", prefix_scoring.score_label_prompts_full, 16),
            ("prefix", prefix_scoring.score_label_prompts, args.batch_size))
    intact = {}
    for label, comps in [("intact", [])] + [(c.key, [c]) for c in busiest]:
        timings, m = {}, {}
        for name, fn, bs in runs:
            t0 = time.time()
            if comps:
                with ablate(comps):
                    m[name] = margin(fn(model, tok, prompts, chat, bs))
            else:
                m[name] = margin(fn(model, tok, prompts, chat, bs))
            timings[name] = round(time.time() - t0, 2)
        if not comps:
            intact = dict(m)
        move = {n: m[n] - intact[n] for n in m}
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
        r = dict(noise_floor_mean_abs=float(np.abs(m["full_alt"] - m["full"]).mean()),
                 prefix_vs_full_mean_abs=float(np.abs(m["prefix"] - m["full"]).mean()),
                 lesion_move_mean_abs={n: float(np.abs(v).mean()) for n, v in move.items()},
                 seconds=timings)
        ok = r["prefix_vs_full_mean_abs"] <= 1.5 * r["noise_floor_mean_abs"] + 0.02
        if comps:
            r.update(move_corr_floor=corr(move["full_alt"], move["full"]), move_corr_prefix=corr(move["prefix"], move["full"]))
            if r["lesion_move_mean_abs"]["full"] > 0.01:
                ok = ok and r["move_corr_prefix"] >= r["move_corr_floor"] - 0.05
        r["pass"] = bool(ok)
        result["lesions"][label] = r
        print(label, r, flush=True)
    result["pass"] = all(r["pass"] for r in result["lesions"].values())
    out = common.out_dir(args, "check_prefix")
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_prefix.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
