"""Specificity control for the mechanism battery (exploratory, calibration concepts only; 2026-10-08).

Random component sets matched to K8 and to C4: same kind and layer; experts also within 0.25 of the target's
routing-frequency percentile in its layer, as in experiment 2 (lesion.baselines). The selected knowledge and
control candidates (K30, C16) are never drawn. Each state runs the battery's scored and label formats on the 80
calibration concepts with one batch composition, against an intact score from the same composition.

    python experiments/e3_errors/run_random_controls.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16

Questions: is "free naming falls while word-to-definition matching, six-choice and copying hold" specific to K8,
or does any late-layer lesion of the same size do it? Does any mid-layer head lesion of C4's size raise the
miscue effect?
Writes outputs/e3_errors/<model>/random_controls/{candidates,labels}.parquet, sets.json, summary.json, meta.json.
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

import analyze_battery  # noqa: E402
import common  # noqa: E402
import probe_lesions  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import baselines, components, m3_battery as mb, m3_items, m3_scoring as ms, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

CAND_BATCH, LABEL_BATCH = 16, 32
METRICS = ("naming_loss", "w2d_loss", "letter_loss", "word_shown_loss", "copy_loss", "cue_benefit_gain", "miscue_harm_gain")


def top_correct(lab):
    """The correct label beats every other label (not: holds more than half the probability)."""
    return [row[f"logit_{row['correct']}"] > max(row[f"logit_{l}"] for l in ms.LETTERS
                                                 if l != row["correct"] and pd.notna(row.get(f"logit_{l}")))
            for row in lab.to_dict("records")]


def score_state(model, tok, chat, items, pool, comps, ablate, name):
    scored, labels = mb.scored_prompts(items, chat), mb.label_items(items, pool, chat)
    with nullcontext() if comps is None else ablate(comps):
        cand = mb.candidate_scores(model, tok, scored, chat, CAND_BATCH)
        lab = mb.label_scores(model, tok, labels, chat, LABEL_BATCH)
    lab["top_correct"] = top_correct(lab)
    return cand.assign(state=name), lab.assign(state=name)


def state_metrics(pc):
    """Per state: mean losses (intact minus lesion; positive = worse) and hint changes, from per-concept scores."""
    I = pc["intact"]
    out = {}
    for s, T in pc.items():
        if s == "intact":
            continue
        T = T.reindex(I.index)
        cols = ["delta_none_hinted_demos", "delta_cue", "delta_miscue"]
        ok = I[cols].notna().all(axis=1) & T[cols].notna().all(axis=1)
        b = ((T.delta_cue - T.delta_none_hinted_demos) - (I.delta_cue - I.delta_none_hinted_demos))[ok]
        h = -((T.delta_miscue - T.delta_none_hinted_demos) - (I.delta_miscue - I.delta_none_hinted_demos))[ok]
        out[s] = dict(naming_loss=float((I.name_target_logprob - T.name_target_logprob).mean()),
                      w2d_loss=float((I.word_to_definition_margin - T.word_to_definition_margin).mean()),
                      letter_loss=float((I.options_letter_margin - T.options_letter_margin).mean()),
                      word_shown_loss=float((I.options_word_target_logprob - T.options_word_target_logprob).mean()),
                      copy_loss=float((I.copy_target_logprob - T.copy_target_logprob).mean()),
                      cue_benefit_gain=float(b.mean()), miscue_harm_gain=float(h.mean()))
    return out


def compare(metrics, target, prefix):
    """Where the target set falls among its random sets, metric by metric."""
    rand = {k: v for k, v in metrics.items() if k.startswith(prefix)}
    out = {}
    for m in METRICS:
        r = np.array([v[m] for v in rand.values()])
        t = metrics[target][m]
        out[m] = dict(target=round(t, 3), random_median=round(float(np.median(r)), 3), random_min=round(float(r.min()), 3),
                      random_max=round(float(r.max()), 3), n_random_at_least_target=int((r >= t).sum()), n_random=len(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--n-random", type=int, default=20)
    args = ap.parse_args()
    if args.tiny:
        raise SystemExit("needs the frozen sets of the main model")
    out = ROOT / "outputs/e3_errors" / args.model / "random_controls"
    for path in (out, out.with_name(out.name + ".partial")):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing results: {path}")
    commit = common.guard(args)
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role.isin(("calibration_a", "calibration_b"))].reset_index(drop=True)
    pool_items = final[final.role.isin(mb.POOL_ROLES)].reset_index(drop=True)
    m1_dir = common.out_dir(args, "screen")
    control = list(common.load_selection(m1_dir, json.loads((m1_dir / "meta.json").read_text(encoding="utf-8")))["control"])
    keys = probe_lesions.dose_sets(pd.read_csv(m1_dir / "screen.csv"),
                                   pd.read_csv(ROOT / "outputs/e3_knowledge" / args.model / "screen/screen.csv"))
    if keys["C4"] != control or keys["K8"] != list(rdt.K8):
        raise SystemExit("the rankings do not start with the frozen sets")
    K8, C4 = [Component.parse(k) for k in keys["K8"]], [Component.parse(k) for k in keys["C4"]]
    exclude = [Component.parse(k) for k in set(keys["K30"]) | set(keys["C16"])]

    model, tok, chat, means, ablate = common.load(args)
    t0 = time.time()
    freq = means.routing_frequency()
    draws = {}
    for name, target, seed in (("K", K8, 31), ("C", C4, 32)):
        kinds = sorted({c.kind for c in target})
        cand_pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
        routing = {c: float(freq[c.layer][c.index]) for c in cand_pool if c.kind == "expert"} if "expert" in kinds else None
        for i, s in enumerate(baselines.matched_random_sets(target, cand_pool, args.n_random, seed=seed, exclude=exclude,
                                                            routing_freq=routing)):
            draws[f"random{name}_{i:02d}"] = s
    plan = [("intact", None), ("K8", K8), ("C4", C4), *draws.items()]
    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True)
    (partial / "sets.json").write_text(json.dumps({k: [c.key for c in v] for k, v in plan if v is not None}, indent=2),
                                       encoding="utf-8")
    cands, labs = [], []
    for name, comps in plan:
        c, l = score_state(model, tok, chat, items, pool_items, comps, ablate, name)
        cands.append(c)
        labs.append(l)
        print(f"{name}: {round(time.time() - t0)} s", flush=True)
        for t, dfs in (("candidates", cands), ("labels", labs)):
            tmp = partial / f"{t}.parquet.tmp"
            pd.concat(dfs, ignore_index=True).to_parquet(tmp, index=False)
            tmp.replace(partial / f"{t}.parquet")
    pc = analyze_battery.per_concept(pd.concat(cands, ignore_index=True), pd.concat(labs, ignore_index=True))
    metrics = state_metrics(pc)
    summary = dict(per_state=metrics, K8_vs_random=compare(metrics, "K8", "randomK_"), C4_vs_random=compare(metrics, "C4", "randomC_"))
    (partial / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    runinfo.write_meta(partial, model=args.model, dtype=args.dtype, means_path=str(args.means), means_sha256=runinfo.sha256(args.means),
                       ablation=dict(mode=args.ablation_mode, alpha=0.0), n_random=args.n_random, resume_commit=commit,
                       batches=dict(candidates=CAND_BATCH, labels=LABEL_BATCH), n_items=len(items),
                       final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"),
                       seconds=round(time.time() - t0, 1), complete=True)
    partial.rename(out)
    for k in ("K8_vs_random", "C4_vs_random"):
        print(k, json.dumps({m: (v["target"], v["random_median"], v["n_random_at_least_target"]) for m, v in summary[k].items()}))


if __name__ == "__main__":
    main()
