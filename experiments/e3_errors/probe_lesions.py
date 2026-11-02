"""Exploratory, calibration concepts only (2026-10-07): do the frozen lesion sets move the M3 items at all?

The intact model picks the target in the six-choice task with mean probability 0.995 (validity check, GPU session
6), and H3a needs errors after a lesion. This probe applies full mean ablation (alpha 0) of the control heads (C,
the M1 screen selection) and of the 8 knowledge components (K8, as frozen in run_dissociation_test.py) to the
calibration concepts of data/materials/m3/final_items.csv (calibration_a and calibration_b, the sets kept for
severity matching and the degradation model; experiment3-protocol.md §4, §5). No test concept is read.

    python experiments/e3_errors/probe_lesions.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16

Writes outputs/e3_errors/<model>/probe/{choice.parquet, naming.parquet, summary.json, meta.json}.

--dose (added after the first probe left the six-choice task at ceiling under both sets): larger sets taken further
down the same screening rankings, fixed before the run. Control: the components of the M1 screen ranked by
control effect, leaving out those whose string damage exceeds their control effect (ADR 0005), top 4 (= the frozen
set), 8 and 16. Knowledge: the verified knowledge screen ranked by knowledge effect, top 8 (= K8), 16 and all 30
with a positive effect. Writes to probe_dose/.

--errors (user decision 2026-10-07, Q13: read error types from free naming): the dose sets, free naming without a
hint, every calibration concept scored on five candidate answers (target, coordinate, superordinate, associate,
unrelated; lesion.m3_scoring.naming_errors). Writes to probe_errors/.
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

import common  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import m3_items, m3_scoring as ms, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

CALIBRATION = ("calibration_a", "calibration_b")


def calibration_items(path=m3_items.OUT / "final_items.csv"):
    items = pd.read_csv(path, keep_default_na=False, na_values=[""])
    items = items[items.role.isin(CALIBRATION)].reset_index(drop=True)
    assert set(items.role) <= set(CALIBRATION)
    return items


def run_probe(model, tok, chat, items, sets, ablate):
    """Six-choice scores of all items and free-naming scores (three conditions) of the items usable for free
    naming, intact and under each lesion set; one table each, with a `set` column."""
    choice_prompts = ms.build_choice_prompts(items, chat)
    naming_prompts = ms.build_naming_prompts(items[items.free_naming_ok.astype(bool)], chat=chat)
    choice, naming = {}, {}
    for name, comps in {"intact": None, **sets}.items():
        with nullcontext() if comps is None else ablate(comps):
            choice[name] = ms.averaged_choice(model, tok, choice_prompts, chat)
            naming[name] = ms.averaged_naming(model, tok, naming_prompts, chat)
    return (pd.concat(choice, names=["set"]).reset_index(level=0).reset_index(drop=True),
            pd.concat(naming, names=["set"]).reset_index(level=0).reset_index(drop=True))


def _ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    boots = np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(axis=1)
    return dict(mean=round(float(x.mean()), 4), ci_low=round(float(np.quantile(boots, 0.025)), 4),
                ci_high=round(float(np.quantile(boots, 0.975)), 4))


def dose_sets(control_screen, knowledge_screen):
    """Sets of growing size down the screening rankings (see the module docstring); keys C4, C8, C16, K8, K16, K30."""
    c = control_screen[control_screen.string_damage <= control_screen.control_effect]
    c = c.sort_values("control_effect", ascending=False).component.tolist()
    k = knowledge_screen.sort_values("knowledge_effect", ascending=False)
    k = k.component[k.knowledge_effect > 0].tolist()
    sets = {f"C{n}": c[:n] for n in (4, 8, 16)}
    sets.update({f"K{n}": k[:n] for n in (8, 16)}, K30=k[:30])
    return sets


def logit_gaps(choice):
    """Per set and item: logit of each wrong option minus the target's, averaged over rotations, and
    S_log = gap(associate) - mean(gap(coordinate), gap(superordinate)): which error type gains on the log scale,
    whatever the overall error rate (the probability scale is flat when the target holds about 25 logits)."""
    g = choice[["set", "item_id"]].copy()
    for r in ms.ROLES[1:]:
        g[r] = choice[f"logit_{r}"] - choice.logit_target
    g = g.groupby(["set", "item_id"]).mean()
    g["S_log"] = g.associate - (g.coordinate + g.superordinate) / 2
    return g


def run_errors(model, tok, chat, items, sets, ablate):
    """Free-naming scores of the five candidate answers without a hint, intact and under each set."""
    prompts = ms.build_naming_prompts(items, ("none",), chat, roles=ms.ERROR_ROLES)
    naming = {}
    for name, comps in {"intact": None, **sets}.items():
        with nullcontext() if comps is None else ablate(comps):
            naming[name] = ms.averaged_naming(model, tok, prompts, chat)
    return pd.concat(naming, names=["set"]).reset_index(level=0).reset_index(drop=True)


def summarize_errors(naming):
    """Per set: mean renormalised probability of each candidate, share of concepts whose most probable candidate
    is not the target, mean log-probability gaps to the target, S and S_log (whole word and per token); for the
    lesions, the changes from intact with percentile intervals over resampled concepts."""
    e = {s: ms.naming_errors(g).set_index("item_id") for s, g in naming.groupby("set")}
    cols = ["target_logprob", "S", "S_log", "S_log_pt"] + [f"gap_{r}" for r in ms.ERROR_ROLES[1:]]
    out = {}
    for s, t in e.items():
        p = t[[f"p_{r}" for r in ms.ERROR_ROLES]]
        out[s] = dict(n=len(t), mean_probability={r: round(float(t[f"p_{r}"].mean()), 4) for r in ms.ERROR_ROLES},
                      share_target_not_top=round(float((p.idxmax(axis=1) != "p_target").mean()), 4),
                      mean={c: round(float(t[c].mean()), 3) for c in cols})
        if s != "intact":
            out[s]["change"] = {c: _ci(t[c] - e["intact"][c]) for c in cols}
    return out


def summarize(choice, naming, items):
    """Per set: mean probability of each option type, share of concepts whose most probable option is not the
    target, S = P(associate) - [P(coordinate) + P(superordinate)] and its change from intact (H3a); mean delta per
    hint condition and R = delta_cue - delta_miscue and its change (H3b). Means over concepts, with a percentile
    interval over resampled concepts (exploratory)."""
    role = items.set_index("item_id").role
    probs = {s: ms.choice_probabilities(g).set_index("item_id") for s, g in choice.groupby("set")}
    deltas = {s: ms.naming_deltas(g).pivot(index="item_id", columns="condition", values="delta") for s, g in naming.groupby("set")}
    S = {s: p.p_associate - p.p_coordinate - p.p_superordinate for s, p in probs.items()}
    R = {s: d.cue - d.miscue for s, d in deltas.items()}
    gaps = logit_gaps(choice)
    out = {}
    for s in probs:
        p, d = probs[s], deltas[s]
        out[s] = dict(
            n_choice=len(p), n_naming=len(d),
            mean_probability={r: round(float(p[f"p_{r}"].mean()), 4) for r in ms.ROLES},
            share_target_not_top=round(float((p.idxmax(axis=1) != "p_target").mean()), 4),
            share_target_below_half=round(float((p.p_target < 0.5).mean()), 4),
            S=_ci(S[s]), mean_delta={c: round(float(d[c].mean()), 3) for c in ms.CONDITIONS}, R=_ci(R[s]),
            by_role={r: round(float(p.p_target[role[p.index] == r].mean()), 4) for r in CALIBRATION},
            mean_logit_gap={c: round(float(gaps.loc[s][c].mean()), 3) for c in gaps.columns})
        if s != "intact":
            out[s]["logit_gap_change"] = {c: _ci(gaps.loc[s][c] - gaps.loc["intact"][c]) for c in gaps.columns}
            out[s]["S_change"] = _ci(S[s] - S["intact"])
            out[s]["R_change"] = _ci((R[s] - R["intact"]).dropna())
            out[s]["p_target_change"] = _ci(p.p_target - probs["intact"].p_target)
    return out


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--dose", action="store_true", help="larger sets down the screening rankings")
    ap.add_argument("--errors", action="store_true", help="naming errors with the dose sets (implies --dose)")
    args = ap.parse_args()
    if args.tiny:
        raise SystemExit("the probe needs the frozen sets of the main model; tests/test_m3_probe.py covers the code")
    args.dose = args.dose or args.errors
    commit = common.guard(args)
    out = ROOT / "outputs/e3_errors" / args.model / ("probe_errors" if args.errors else "probe_dose" if args.dose else "probe")
    if out.exists():
        raise SystemExit(f"Refusing to run: {out} exists")
    items = calibration_items()
    m1_dir = common.out_dir(args, "screen")
    m1_meta = json.loads((m1_dir / "meta.json").read_text(encoding="utf-8"))
    control = common.load_selection(m1_dir, m1_meta)["control"]
    if args.dose:
        k_dir = ROOT / "outputs/e3_knowledge" / args.model / "screen"
        keys = dose_sets(pd.read_csv(m1_dir / "screen.csv"), pd.read_csv(k_dir / "screen.csv"))
        if keys["C4"] != list(control) or keys["K8"] != list(rdt.K8):
            raise SystemExit(f"the rankings do not start with the frozen sets: {keys['C4']}, {keys['K8']}")
    else:
        keys = {"control": control, "main": list(rdt.K8)}
    sets = {name: [Component.parse(k) for k in v] for name, v in keys.items()}
    model, tok, chat, means, ablate = common.load(args)
    t0 = time.time()
    if args.errors:
        naming = run_errors(model, tok, chat, items, sets, ablate)
        summary = summarize_errors(naming)
        out.mkdir(parents=True)
    else:
        choice, naming = run_probe(model, tok, chat, items, sets, ablate)
        summary = summarize(choice, naming, items)
        out.mkdir(parents=True)
        choice.to_parquet(out / "choice.parquet", index=False)
    naming.to_parquet(out / "naming.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    runinfo.write_meta(out, model=args.model, dtype=args.dtype, means_path=str(args.means),
                       means_sha256=runinfo.sha256(args.means), ablation=dict(mode="mean", alpha=0.0),
                       sets={k: [c.key for c in v] for k, v in sets.items()}, resume_commit=commit,
                       final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"), n_items=len(items),
                       seconds=round(time.time() - t0, 1))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
