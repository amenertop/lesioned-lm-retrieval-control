"""Experiment 3 on the sealed M3 test concepts, once, at the analysis-plan freeze tag (docs/analysis-plan/experiment3.md):

    python experiments/e3_errors/run_experiment3.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 \
        --freeze-tag analysis-frozen-experiment3

States (full mean ablation, WikiText means): intact, K8, C4, C4 restricted to the answer region, K30, C16.
Per state, on the 160 test concepts, the battery formats (lesion.m3_battery): free naming with five candidates,
the hinted-demo baseline, cue and miscue (on the concepts usable for free naming), the six options answered with
the word, copying, the context-intrusion test (a word to ignore shown before the definition, Q21), six-choice
letters (6 rotations) and word-to-definition matching (2 orders); every score
averaged over three batch compositions; greedy answers for intact, K8 and C4. Then the direct logit attribution of
K8 in the intact model (S4).
Writes outputs/e3_errors/<model>/confirm/{candidates,labels,greedy,dla}.parquet and meta.json (renamed from
confirm.partial when complete; never overwrites). Analysis: analyze_experiment3.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import common  # noqa: E402
import probe_lesions  # noqa: E402
import run_battery as rb  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import hooks, m3_battery as mb, m3_dla, m3_items, m3_scoring as ms, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

FINAL_ITEMS_SHA256 = "d539310755a31f8155cda5db7ae4920760dd36bbde97442800fa0b17df16b98d"
MEANS_SHA256 = "98b3865a71b0c8e85c530cb78d5438f9223a828fdf77c907315785b1f1de0d52"   # WikiText means (analysis plan §2)
POOL_ROLES = ("test", "calibration_a", "calibration_b", "reserve")
GREEDY_STATES = ("intact", "K8", "C4")
# the dose sets as fixed before the freeze (GPU session 6, probe_dose meta; analysis plan §2) -- agent2 2026-10-08
K30 = ("expert:46:61", "expert:46:101", "head:37:24", "expert:45:32", "expert:45:48", "expert:43:70", "expert:42:115",
       "expert:45:93", "head:46:3", "head:41:12", "expert:45:12", "head:33:16", "head:38:5", "head:40:15", "expert:41:79",
       "head:14:13", "head:38:3", "head:31:5", "expert:45:116", "head:19:5", "expert:41:62", "head:40:14", "head:32:4",
       "head:30:14", "expert:21:0", "head:6:14", "expert:27:61", "expert:45:119", "expert:44:22", "expert:2:85")
C16 = ("head:9:15", "head:27:9", "head:25:24", "head:24:5", "head:14:13", "head:33:20", "head:21:19", "expert:35:28",
       "head:6:1", "head:26:16", "head:32:6", "head:21:3", "head:9:3", "head:22:13", "head:26:13", "head:22:17")
FOILS = m3_items.OUT / "w2d_foils_test.csv"   # word-to-definition foils of the test concepts, fixed before the freeze


def check_foils(items, pool):
    """The foils built now must be the ones written down before the freeze."""
    fixed = pd.read_csv(FOILS, keep_default_na=False).set_index("item_id").foil_id.to_dict()
    built = {it.item_id: it.foil_id for it in mb.word_to_definition_items(items, pool) if it.variant == 0}
    if built != fixed:
        raise SystemExit("word-to-definition foils differ from data/materials/m3/w2d_foils_test.csv")


def plan(sets, ablate):
    return [("intact", None, None, None), ("K8", sets["K8"], ablate, None), ("C4", sets["C4"], ablate, None),
            ("C4_answer", sets["C4"], ablate, "answer"), ("K30", sets["K30"], ablate, None), ("C16", sets["C16"], ablate, None)]


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True, help="analysis-plan freeze tag (analysis-frozen-...)")
    args = ap.parse_args()
    if args.tiny:
        raise SystemExit("the confirmatory run needs the main model; tests cover the code")
    if args.ablation_mode != "mean":
        raise SystemExit("the analysis plan fixes mean ablation")
    out = ROOT / "outputs/e3_errors" / args.model / "confirm"
    rb.check_destination(out)
    if not hasattr(hooks, "position_mask"):
        raise SystemExit("C4_answer requires hooks.position_mask")
    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    if runinfo.sha256(m3_items.OUT / "final_items.csv") != FINAL_ITEMS_SHA256:
        raise SystemExit("final_items.csv differs from the hash in the analysis plan")
    if runinfo.sha256(args.means) != MEANS_SHA256:
        raise SystemExit(f"{args.means} is not the means file fixed in the analysis plan")
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role == "test"].reset_index(drop=True)
    pool = final[final.role.isin(POOL_ROLES)].reset_index(drop=True)
    m1_dir = common.out_dir(args, "screen")
    control = list(common.load_selection(m1_dir, json.loads((m1_dir / "meta.json").read_text(encoding="utf-8")))["control"])
    keys = probe_lesions.dose_sets(pd.read_csv(m1_dir / "screen.csv"),
                                   pd.read_csv(ROOT / "outputs/e3_knowledge" / args.model / "screen/screen.csv"))
    if keys["C4"] != control or keys["K8"] != list(rdt.K8) or keys["K30"] != list(K30) or keys["C16"] != list(C16):
        raise SystemExit("the rankings no longer give the sets fixed in the analysis plan")
    sets = {k: [Component.parse(x) for x in v] for k, v in dict(K8=rdt.K8, C4=control, K30=K30, C16=C16).items()}
    check_foils(items, pool)

    model, tok, chat, means, ablate = common.load(args)
    t0 = time.time()
    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    tables, timing = {}, {}

    def save():
        for t, dfs in tables.items():
            tmp = partial / f"{t}.parquet.tmp"
            pd.concat(dfs, ignore_index=True).to_parquet(tmp, index=False)
            tmp.replace(partial / f"{t}.parquet")

    for name, comps, abl, region in plan(sets, ablate):
        t1 = time.time()
        res = rb.run_state(model, tok, chat, items, pool, comps, abl, region, name, greedy=name in GREEDY_STATES, readouts=False,
                           intrusion=True)
        for t, df in res.items():
            if df is not None:
                tables.setdefault(t, []).append(df)
        timing[name] = round(time.time() - t1, 1)
        print(f"{name}: {timing[name]} s", flush=True)
        save()
    prompts = ms.build_naming_prompts(items, ("none",), chat, roles=ms.ERROR_ROLES)
    contrib, final_resid = m3_dla.contributions(model, means, sets["K8"], prompts, tok, chat)
    tables["dla"] = [m3_dla.direct_effects(model, tok, prompts, contrib, final_resid, chat)]
    # total effect on the same scale (first-token logits, own target against the other targets; agent2 2026-10-08)
    totals = [m3_dla.first_token_logits(model, tok, prompts, chat).assign(state="intact")]
    with ablate(sets["K8"]):
        totals.append(m3_dla.first_token_logits(model, tok, prompts, chat).assign(state="K8"))
    tables["dla_total"] = totals
    save()
    runinfo.write_meta(partial, model=args.model, dtype=args.dtype, freeze_tag=args.freeze_tag, resume_commit=commit,
                       means_path=str(args.means), means_sha256=MEANS_SHA256, ablation=dict(mode="mean", alpha=0.0),
                       foils_sha256=runinfo.sha256(FOILS),
                       sets={k: [c.key for c in v] for k, v in sets.items()}, states=[s[0] for s in plan(sets, ablate)],
                       final_items_sha256=FINAL_ITEMS_SHA256, n_items=len(items), timing=timing,
                       seconds=round(time.time() - t0, 1), complete=True)
    partial.rename(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
