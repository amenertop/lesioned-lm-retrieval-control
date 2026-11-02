"""GPU session 14b, the component taxonomy of P14 (docs/plans/p14-learned-gates.md), exploratory: complete the damage
profile of every component already screened alone, so that they can be clustered.

    python experiments/e3_knowledge/run_profile_pool.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3

The pool is every component that one of the three single-component screens ablated (the knowledge screen, the
session-11 store screen and the session-12 K8b screen; 79 components). The K8b screen already measured the full
profile; every other pool member is measured here with exactly its procedure (run_rescreen_k8b.py): alone, on the
screening items, naming (knowledge effect and naming damage, averaged over three batch compositions), M1 (semantic,
string and control damage, the margin-scaling slope) and the language-model loss on 40 neutral passages of 128
tokens; experts also get their routing frequency and mean-output norm over the layer median.
Writes outputs/e3_knowledge/<model>/profile_pool/{profiles.csv, meta.json}; profiles.csv holds all 79 rows (the K8b
screen's rows copied, with their source noted).
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
from lesion import damage, knowledge_items as ki, materials, naming, neutral_text, probes, rescreen, runinfo, scoring, screen  # noqa: E402
from lesion.components import Component  # noqa: E402

BATCHES = (16, 11, 7)            # as the knowledge screen and run_rescreen_k8b.py
N_TEXTS, N_TOKENS = 40, 128      # as run_rescreen_k8b.py
FULL = ("knowledge_effect", "naming_damage", "m1_semantic_damage", "m1_string_damage", "m1_control_effect", "m1_slope",
        "lm_loss_change")


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--limit", type=int, default=0, help="profile only the first N missing components (tiny runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit = common.guard(args)
    kroot = ROOT / ("outputs/e3_knowledge_tiny" if args.tiny else "outputs/e3_knowledge") / args.model
    screens = {"knowledge_screen": kroot / "screen/screen.csv", "store_screen": kroot / "store_screen/screen.csv",
               "k8b_screen": kroot / "k8b/screen.csv"}
    present = {k: pd.read_csv(v) for k, v in screens.items() if v.exists()}
    if "k8b_screen" not in present:
        raise SystemExit("the K8b screen is needed (its rows are the full profiles to complete)")
    pool = sorted(set().union(*(set(t.component) for t in present.values())))
    full = present["k8b_screen"]
    missing = [c for c in pool if c not in set(full.component)]
    if args.limit:
        missing = missing[:args.limit]
    out = Path(args.out) if args.out else kroot / "profile_pool"

    all_meta = pd.read_csv(ki.OUT / "items.csv")[["item_id", "zipf"]]
    median = naming.zipf_median(all_meta)
    n_scr = ki.load_items("screening")
    demos = pd.read_csv(ki.OUT / "demos.csv")
    all_m1, m1_demos, templates = scoring.load_materials()
    all_m1 = all_m1.astype({"item_id": str})
    fm = damage.frequency_median(all_m1)
    m_scr = materials.load_split("screening")

    model, tok, chat, means, ablate = common.load(args)
    pad = tok.eos_token_id or 0
    texts = neutral_text.load()[:N_TEXTS]
    lm_seqs = [tok(t, add_special_tokens=False).input_ids[:N_TOKENS] for t in texts]
    fields = common.run_fields(args, commit, "profile_pool", "screening", len(n_scr) + len(m_scr), batches=list(BATCHES),
                               zipf_median=median, pool=pool, missing=missing,
                               screens_sha256={k: runinfo.sha256(v) for k, v in screens.items() if v.exists()})
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()

    prompts_n = naming.build_prompts(n_scr, demos, chat)

    def naming_score(comps=()):
        with ablate(list(comps)) if comps else nullcontext():
            return naming.averaged_logprobs(model, tok, prompts_n, chat, BATCHES)

    prompts_m = scoring.build_label_prompts(m_scr, m1_demos, templates, variant="main", chat=chat)
    label_score = screen.make_scorer(model, tok, prompts_m, chat, ablate, args.batch_size, common.SCORERS[args.scorer])

    def m1_profile(intact, lesioned):
        i, l = intact.astype({"item_id": str}), lesioned.astype({"item_id": str})
        frame = damage.item_frame(damage.item_damage(i, l), all_m1, fm)
        e = rescreen.m1_effects_with_synonym(damage.columns(frame))
        slope = damage.margin_scaling(i, l, all_m1).set_index("task").slope
        return dict(m1_semantic_damage=e["m1_semantic_damage"], m1_string_damage=e["m1_string_damage"],
                    m1_control_effect=e["m1_control_effect"], m1_slope=float(slope["all"]))

    lm_intact = probes.lm_loss_region(model, lm_seqs, [], means, pad_id=pad)
    flags = rescreen.super_expert_flags(means.expert_mean) if means.expert_mean is not None else {}
    freq = means.routing_frequency() if means.expert_count is not None else None

    def expert_info(c):
        if c.kind != "expert":
            return {}
        norms = means.expert_mean[c.layer].float().norm(dim=-1).numpy()
        return dict(routing_frequency=float(freq[c.layer][c.index]) if freq is not None else np.nan,
                    mean_output_norm=float(norms[c.index]), norm_over_layer_median=float(norms[c.index] / np.median(norms)),
                    candidate_super_expert=(c.layer, c.index) in flags)

    intact_n, intact_m = naming_score(), label_score()
    rows = [dict(r, source="k8b_screen") for r in full.to_dict("records")]
    for i, key in enumerate(missing, 1):
        c = [Component.parse(key)]
        frame = naming.item_frame(naming.item_damage(intact_n, naming_score(c)), n_scr, median)
        ne = naming.effects(damage.columns(frame))
        rows.append(dict(component=key, knowledge_effect=ne["knowledge_effect"], knowledge_slope=ne["knowledge_slope"],
                         naming_damage=ne["naming_damage"], **m1_profile(intact_m, label_score(c)),
                         lm_loss_change=probes.lm_loss_region(model, lm_seqs, c, means, pad_id=pad) / lm_intact - 1,
                         **expert_info(c[0]), source="profile_pool"))
        if i % 8 == 0 or i == len(missing):
            pd.DataFrame(rows).to_csv(out / "profiles.csv", index=False)
            print(f"{i}/{len(missing)} components ({time.time() - t0:.0f} s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "profiles.csv", index=False)
    complete = int(table[list(FULL)].notna().all(axis=1).sum())
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), n_complete=complete, **fields)
    print(f"written to {out} ({complete}/{len(table)} complete profiles)")


if __name__ == "__main__":
    main()
