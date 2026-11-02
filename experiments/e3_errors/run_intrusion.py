"""Context-intrusion test on the 80 calibration concepts (exploratory; user decision Q21, 2026-10-08).

    python experiments/e3_errors/run_intrusion.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16

A word to ignore is shown before the definition: the associate, the unrelated word, or none
(lesion.m3_battery.intrusion_prompts). Prediction: lesioning the control heads lets a shown associate pull the
answer towards itself (pull_assoc rises), although the same lesion leaves unprompted naming intact.
States: intact, K8, C4, C4 at the answer region only, K30, C16 (three batch compositions each), and 10 random
sets matched to C4 and 10 matched to K8 (one composition each, against an intact score of the same composition).
Greedy answers with the associate shown, for intact, K8, C4, C16. Writes outputs/e3_errors/<model>/intrusion/
{candidates, greedy}.parquet, summary.json, meta.json.
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
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import common  # noqa: E402
import probe_lesions  # noqa: E402
import run_battery as rb  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import baselines, components, hooks, m3_battery as mb, m3_items, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

GREEDY_STATES = ("intact", "K8", "C4", "C16")


def ci(x, rng, n=5000):
    x = np.asarray(x, dtype=float)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return dict(mean=round(float(x.mean()), 3), ci=[round(float(np.quantile(b, q)), 3) for q in (0.025, 0.975)])


def summarize(cand, greedy, items):
    """Per state: change of pull_assoc and pull_unrel from intact (same composition for random sets), associate
    specificity (pull_assoc change minus pull_unrel change), and greedy intrusion rates."""
    rng = np.random.default_rng(0)
    m = mb.intrusion_measures(cand)
    out = dict(states={}, intact=dict(pull_assoc=round(float(m.loc["intact"].pull_assoc.mean()), 3),
                                      pull_unrel=round(float(m.loc["intact"].pull_unrel.mean()), 3)))
    for s in m.index.get_level_values(0).unique():
        if s.startswith("intact"):
            continue
        base = m.loc["intact1"] if s.startswith("random") else m.loc["intact"]
        d = m.loc[s].reindex(base.index) - base
        out["states"][s] = dict(pull_assoc_change=ci(d.pull_assoc, rng), pull_unrel_change=ci(d.pull_unrel, rng),
                                associate_specific=ci(d.pull_assoc - d.pull_unrel, rng),
                                target_logprob_change_assoc_shown=ci(d.target_logprob_assoc_shown, rng))
    for k, prefix in (("C4", "randomC_"), ("K8", "randomK_")):
        r = [v["pull_assoc_change"]["mean"] for s, v in out["states"].items() if s.startswith(prefix)]
        if r and k in out["states"]:
            t = out["states"][k]["pull_assoc_change"]["mean"]
            out[f"{k}_vs_random"] = dict(target=t, random_median=round(float(np.median(r)), 3), random_max=round(float(max(r)), 3),
                                         n_random_at_least_target=int(sum(x >= t for x in r)), n_random=len(r))
    if greedy is not None and len(greedy):
        rows = items.set_index("item_id")
        g = greedy.assign(said=[mb.classify_answer(a, rows.loc[i]) for i, a in zip(greedy.item_id, greedy.answer)])
        out["greedy_with_associate_shown"] = {s: x.said.value_counts(normalize=True).round(3).to_dict() for s, x in g.groupby("state")}
    return out


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--n-random", type=int, default=10)
    args = ap.parse_args()
    out = ROOT / "outputs/e3_errors" / args.model / "intrusion"
    rb.check_destination(out)
    commit = common.guard(args)
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role.isin(("calibration_a", "calibration_b"))].reset_index(drop=True)
    m1_dir = common.out_dir(args, "screen")
    control = list(common.load_selection(m1_dir, json.loads((m1_dir / "meta.json").read_text(encoding="utf-8")))["control"])
    keys = probe_lesions.dose_sets(pd.read_csv(m1_dir / "screen.csv"),
                                   pd.read_csv(ROOT / "outputs/e3_knowledge" / args.model / "screen/screen.csv"))
    if keys["C4"] != control or keys["K8"] != list(rdt.K8):
        raise SystemExit("the rankings do not start with the frozen sets")
    sets = {k: [Component.parse(x) for x in keys[k]] for k in ("K8", "C4", "K30", "C16")}
    exclude = [Component.parse(k) for k in set(keys["K30"]) | set(keys["C16"])]
    model, tok, chat, means, ablate = common.load(args)
    t0 = time.time()
    prompts = mb.intrusion_prompts(items, chat)
    freq = means.routing_frequency()
    randoms = {}
    for name, target, seed in (("C", sets["C4"], 41), ("K", sets["K8"], 42)):
        kinds = sorted({c.kind for c in target})
        pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
        routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if "expert" in kinds else None
        for i, s in enumerate(baselines.matched_random_sets(target, pool, args.n_random, seed=seed, exclude=exclude, routing_freq=routing)):
            randoms[f"random{name}_{i:02d}"] = s
    plan = [("intact", None, None), ("K8", sets["K8"], None), ("C4", sets["C4"], None), ("C4_answer", sets["C4"], "answer"),
            ("K30", sets["K30"], None), ("C16", sets["C16"], None)]
    cands, gens = [], []
    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True)
    for name, comps, region in plan:
        with nullcontext() if comps is None else ablate(comps):
            c = rb.averaged(lambda bs: mb.candidate_scores(model, tok, prompts, chat, bs, region), rb.CANDIDATE_BATCHES, ["logprob"])
            cands.append(c.assign(state=name))
            if name in GREEDY_STATES:
                gens.append(mb.greedy_answers(model, tok, [p for p in prompts if p.condition == "distract_assoc"], chat).assign(state=name))
        print(f"{name}: {round(time.time() - t0)} s", flush=True)
    cands.append(mb.candidate_scores(model, tok, prompts, chat, 16).assign(state="intact1"))
    for name, comps in randoms.items():
        with ablate(comps):
            cands.append(mb.candidate_scores(model, tok, prompts, chat, 16).assign(state=name))
        print(f"{name}: {round(time.time() - t0)} s", flush=True)
    cand, greedy = pd.concat(cands, ignore_index=True), pd.concat(gens, ignore_index=True)
    cand.to_parquet(partial / "candidates.parquet", index=False)
    greedy.to_parquet(partial / "greedy.parquet", index=False)
    summary = summarize(cand, greedy, items)
    (partial / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    runinfo.write_meta(partial, model=args.model, dtype=args.dtype, means_sha256=runinfo.sha256(args.means), resume_commit=commit,
                       sets={k: [c.key for c in v] for k, v in {**sets, **randoms}.items()}, n_items=len(items),
                       seconds=round(time.time() - t0, 1), complete=True)
    partial.rename(out)
    print(json.dumps({k: v for k, v in summary.items() if k != "states"}, indent=1, ensure_ascii=False))
    for s in ("K8", "C4", "C4_answer", "K30", "C16"):
        print(s, summary["states"][s]["pull_assoc_change"], summary["states"][s]["associate_specific"])


if __name__ == "__main__":
    main()
