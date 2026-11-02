"""Direct logit attribution of the K8 components and C4 heads at the answer position of free naming (exploratory,
80 calibration concepts; lesion.m3_dla):

    python experiments/e3_errors/run_dla.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16

Per concept and component, the direct effect of the removed vector on the logit of the concept's own target word
(first token), of the other concepts' targets, of its associate, coordinate and superordinate. Also the total
effect of each whole set on the same logits (intact minus lesioned), to see how much of K8's naming loss is a
direct push of the specific word. Writes outputs/e3_errors/<model>/dla/{direct.parquet, total.parquet,
summary.json, meta.json}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import common  # noqa: E402
import probe_lesions  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import m3_dla, m3_items, m3_scoring as ms, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402


def ci(x, rng, n=5000):
    x = np.asarray(x, dtype=float)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return dict(mean=round(float(x.mean()), 3), ci=[round(float(np.quantile(b, q)), 3) for q in (0.025, 0.975)])


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    args = ap.parse_args()
    if args.ablation_mode != "mean":
        # the removed vector in lesion.m3_dla is output minus mean; a zero-ablation total effect would not match it
        raise SystemExit("direct logit attribution is defined for mean ablation only")
    out = ROOT / "outputs/e3_errors" / args.model / "dla"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    commit = common.guard(args)
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role.isin(("calibration_a", "calibration_b"))].reset_index(drop=True)
    m1_dir = common.out_dir(args, "screen")
    control = list(common.load_selection(m1_dir, json.loads((m1_dir / "meta.json").read_text(encoding="utf-8")))["control"])
    keys = probe_lesions.dose_sets(pd.read_csv(m1_dir / "screen.csv"),
                                   pd.read_csv(ROOT / "outputs/e3_knowledge" / args.model / "screen/screen.csv"))
    if keys["C4"] != control or keys["K8"] != list(rdt.K8):
        raise SystemExit("the rankings do not start with the frozen sets")
    sets = {k: [Component.parse(x) for x in keys[k]] for k in ("K8", "C4", "K30")}
    model, tok, chat, means, ablate = common.load(args)
    t0 = time.time()
    prompts = ms.build_naming_prompts(items, ("none",), chat, roles=ms.ERROR_ROLES)
    comps = list(dict.fromkeys(sets["K8"] + sets["C4"] + sets["K30"]))
    contrib, final_resid = m3_dla.contributions(model, means, comps, prompts, tok, chat)
    direct = m3_dla.direct_effects(model, tok, prompts, contrib, final_resid, chat)
    totals = [m3_dla.first_token_logits(model, tok, prompts, chat).assign(state="intact")]
    for k in ("K8", "C4", "K30"):
        with ablate(sets[k]):
            totals.append(m3_dla.first_token_logits(model, tok, prompts, chat).assign(state=k))
    total = pd.concat(totals, ignore_index=True)

    rng = np.random.default_rng(0)
    summary = dict(per_component={}, per_set={})
    for key, g in direct.groupby("component"):
        summary["per_component"][key] = {c: ci(g[c], rng) for c in ("target", "specific", "associate", "coordinate", "superordinate", "vocab_mean")}
    I = total[total.state == "intact"].set_index("item_id")
    for k, comps_k in sets.items():
        d = direct[direct.component.isin([c.key for c in comps_k])].groupby("item_id")[["target", "specific", "associate", "coordinate", "superordinate"]].sum()
        T = total[total.state == k].set_index("item_id")
        total_specific = (I.target_logit - I.other_targets_logit) - (T.target_logit - T.other_targets_logit)
        summary["per_set"][k] = dict(direct_target=ci(d.target, rng), direct_specific=ci(d.specific, rng),
                                     direct_minus_associate=ci(d.target - d.associate, rng),
                                     direct_minus_superordinate=ci(d.target - d.superordinate, rng),
                                     total_target_logit_drop=ci(I.target_logit - T.target_logit, rng),
                                     total_specific_drop=ci(total_specific.loc[d.index], rng),
                                     corr_direct_specific_vs_total_specific=round(float(np.corrcoef(d.specific, total_specific.loc[d.index])[0, 1]), 3))
    out.mkdir(parents=True)
    direct.to_parquet(out / "direct.parquet", index=False)
    total.to_parquet(out / "total.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    runinfo.write_meta(out, model=args.model, dtype=args.dtype, means_sha256=runinfo.sha256(args.means), resume_commit=commit,
                       sets={k: [c.key for c in v] for k, v in sets.items()}, n_items=len(items),
                       seconds=round(time.time() - t0, 1), complete=True)
    print(json.dumps(summary["per_set"], indent=1))


if __name__ == "__main__":
    main()
