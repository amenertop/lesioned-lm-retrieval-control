"""Session 11, main model (exploratory; docs/plans/gpu-session11.md): the clinical scorecard for the new sets next to
K8 and C4, the letter-cued naming, and the frequency effect in synonym judgement counted as accuracy. No GPU:

    python experiments/e3_errors/analyze_new_sets.py [--out reports/explore/qwen_new_sets.json]

Inputs: outputs/e3_errors/<model>/new_sets (run_new_sets.py: intact, K8, C4, C4s, S8), the held-out tests of the new
sets (outputs/e2_confirm/<model>/dissociation_new with S8 and C4s in place of K8 and C4, control_semantic for C4s),
and, for K8 and C4, the knowledge-side report and the dissociation test's scores.

1. Scorecard: the six markers and the blind diagnosis with the rules of analyze_scorecard.py. Markers 1 and 2 come from
   the held-out tests (naming rare minus common; M1 strong minus weak distractor); 3 to 6 from the new_sets battery, so
   all four sets are scored on the same run.
2. Cued naming (Jefferies, Patterson & Lambon Ralph 2008, in text): per state, the share of concepts named correctly
   with j letters given (0, 1, 2 and the final hint, all but the last letter), the benefit of 2 letters and of the
   final hint over none, and, for concepts not named without a hint but named at some length, the share of the word's
   letters given at the first length that worked. Concept-resampled 95% intervals (10,000, seed 0).
3. Synonym judgement by frequency (Hoffman, Rogers & Lambon Ralph 2011): the M1 held-out synonym items, accuracy (the
   correct option's logit above the other's) averaged over the two option orders, for words below and above the
   frequency median; the frequency effect (common minus rare) per state and its change under each lesion.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import analyze_consistency as ac  # noqa: E402
import analyze_scorecard as sc  # noqa: E402
from lesion import damage, m3_cued, m3_items, runinfo, scoring  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
N_BOOT, SEED = 10_000, 0
NEW = ROOT / "outputs/e3_errors" / MODEL / "new_sets"
CONFIRM = ROOT / "outputs/e2_confirm" / MODEL
OLD_DISSOCIATION = ROOT / "outputs/server/e2_confirm" / MODEL / "dissociation"
SETS = ("K8", "C4", "C4s", "S8")


def ci(boots):
    return [round(float(q), 4) for q in np.nanquantile(np.asarray(boots, dtype=float), (0.025, 0.975))]


def markers_1_2(new_results):
    """Frequency and control-demand markers: K8 and C4 from the knowledge report, S8 and C4s from dissociation_new."""
    out = {}
    k = json.loads(sc.KNOWLEDGE.read_text(encoding="utf-8"))["sets"]["main"]
    n = new_results["sets"]["main"] if new_results else None
    for lesion, src, name, where in (("K8", k, "knowledge", "knowledge report"), ("C4", k, "control", "knowledge report"),
                                     ("S8", n, "knowledge", "dissociation_new"), ("C4s", n, "control", "dissociation_new")):
        if src is None:
            continue
        f, m = src["naming"][f"{name}.knowledge_effect"], src["m1"][f"{name}.control_effect"]
        out[lesion] = dict(frequency=sc.entry(f["estimate"], f["ci_low"], f["ci_high"], 240, "answer log-probability, rare minus common", where),
                           control_demand=sc.entry(m["estimate"], m["ci_low"], m["ci_high"], 63, "logit, strong minus weak distractor", where))
    return out


def consistency_marker(lesions):
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    ids = list(items.item_id[items.role == "test"])
    h = ac.damage(pd.read_parquet(NEW / "candidates.parquet"), pd.read_parquet(NEW / "labels.parquet"), ids, lesions)
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT)]
    out = {}
    for s in lesions:
        point = ac.consistency(h[s])[0]
        boots = [ac.consistency(h[s].iloc[i].reset_index(drop=True))[0] for i in draws]
        lo, hi = ci(boots)
        out[s] = sc.entry(point, lo, hi, len(ids), "mean pairwise Spearman correlation", "new_sets battery data")
    return out


def synonym_frequency():
    """M1 held-out synonym items: accuracy for rare and common words per state and the frequency effect."""
    items = scoring.load_materials()[0].astype({"item_id": str})
    fm = damage.frequency_median(items)
    syn = items[items.task == "synonym"].set_index("item_id")
    rare = pd.to_numeric(syn.min_freq) < fm
    sources = {"K8": (OLD_DISSOCIATION, "main"), "C4": (OLD_DISSOCIATION, "control"),
               "S8": (CONFIRM / "dissociation_new", "main"), "C4s": (CONFIRM / "dissociation_new", "control")}
    out = {}
    for lesion, (d, name) in sources.items():
        f = d / "m1_scores.parquet"
        if not f.exists():
            continue
        s = pd.read_parquet(f).astype({"item_id": str})
        s = s[s.item_id.isin(syn.index)]
        acc = s.assign(ok=(s.logit_correct > s.logit_wrong).astype(float)).groupby(["set", "item_id"]).ok.mean()
        res = {}
        for state, key in (("intact", "intact"), (lesion, name)):
            a = acc.loc[key]
            r = rare.reindex(a.index)
            res[state] = dict(rare=round(float(a[r].mean()), 4), common=round(float(a[~r].mean()), 4),
                              frequency_effect=round(float(a[~r].mean() - a[r].mean()), 4), n_rare=int(r.sum()), n_common=int((~r).sum()))
        res["change_in_frequency_effect"] = round(res[lesion]["frequency_effect"] - res["intact"]["frequency_effect"], 4)
        out[lesion] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports/explore/qwen_new_sets.json"))
    args = ap.parse_args(argv)
    meta = json.loads((NEW / "meta.json").read_text(encoding="utf-8"))
    lesions = tuple(s for s in SETS if s in meta["sets"])
    new_file = CONFIRM / "dissociation_new/results.json"
    new_results = json.loads(new_file.read_text(encoding="utf-8")) if new_file.exists() else None
    model = markers_1_2(new_results)
    for s, v in consistency_marker(lesions).items():
        model.setdefault(s, {})["consistency"] = v
    for s, v in sc.from_experiment3(NEW, lesions).items():
        model.setdefault(s, {}).update(v)
    complete = {s: m for s, m in model.items() if set(sc.CLINICAL) <= set(m)}
    control_sem = CONFIRM / "control_semantic/results.json"
    out = dict(note="exploratory (session 11); rules as in analyze_scorecard.py", commit=runinfo.git("rev-parse", "HEAD"),
               sets=meta["sets"], model=model, diagnosis=sc.diagnose(complete),
               random=dict(store_knowledge_effect=new_results.get("random") if new_results else None,
                           semantic_control=(json.loads(control_sem.read_text(encoding="utf-8")).get("random")
                                             if control_sem.exists() else None)),
               string_damage={k: v for k, v in (
                   ("C4s", (new_results or {}).get("sets", {}).get("main", {}).get("m1", {}).get("control.string_damage")),
                   ("S8", (new_results or {}).get("sets", {}).get("main", {}).get("m1", {}).get("knowledge.string_damage"))) if v},
               cued_naming=m3_cued.summarize(pd.read_parquet(NEW / "cued.parquet"), N_BOOT, SEED),
               synonym_frequency=synonym_frequency())
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for s, d in out["diagnosis"].items():
        print(f"{s}: SD {d['matches_SD']}, SA {d['matches_SA']} -> {d['diagnosis']}")
    for s, r in out["cued_naming"].items():
        print(f"cued naming {s}: none {r['accuracy_none']['estimate']}, two {r['accuracy_two']['estimate']}, final {r['accuracy_final']['estimate']}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
