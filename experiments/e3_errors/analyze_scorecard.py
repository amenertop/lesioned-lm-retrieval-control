"""P8 (a): the clinical scorecard and blind diagnosis (docs/plans/p8-clinical-alignment.md (a), "计算细则"; rules
written and committed before this was first computed, 2026-10-13; all data were already unsealed, so the result is
exploratory). No GPU:

    python experiments/e3_errors/analyze_scorecard.py [--out reports/explore/qwen_clinical_scorecard.json]

Six markers of Jefferies & Lambon Ralph 2006, Table 2. For each lesion (K8, C4) and marker, "present" when the 95%
interval excludes 0 with the estimate in the clinical direction; "absent (reversed)" when it excludes 0 the other
way; "absent" otherwise. Each lesion is compared with the SD and the SA column; the column with more matches is its
diagnosis, a tie is "cannot tell". Markers 1-3 come from the confirmatory reports; 4-6 are recomputed per concept
from experiment 3 (10,000 concept resamples, seed 0). Random sets: marker 1 for the 200 knowledge-side sets, marker
2 for the 200 control-side sets, each with its own interval (lesion.damage.bootstrap, 10,000, seed 0).
Supplementary columns (descriptive, not counted): later patient studies and TMS virtual lesions (data/literature/).
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
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))

import analyze_battery  # noqa: E402
import analyze_experiment3 as ae  # noqa: E402
from lesion import damage, e3_analysis as ea, knowledge_items as ki, m3_items, naming, runinfo, scoring  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
N_BOOT, SEED = 10_000, 0
LESIONS = {"K8": "knowledge", "C4": "control"}         # name in the knowledge report's "sets/main" tables
SERVER = ROOT / "outputs/server/e2_confirm" / MODEL
E3 = ROOT / "outputs/e3_errors" / MODEL / "confirm"
KNOWLEDGE = ROOT / "reports/confirm/knowledge/results.json"
CONSISTENCY = ROOT / "reports/explore/qwen_h3c_consistency.json"
LIT = ROOT / "data/literature"

# marker: (label, SD, SA, clinical direction)
CLINICAL = {
    "frequency": ("词频或熟悉度效应", True, False, "冷门的损伤更大"),
    "control_demand": ("控制需求效应", False, True, "强干扰的损伤更大"),
    "consistency": ("跨任务逐项一致性", True, False, "同一概念在不同题型上一起错"),
    "associative_errors": ("联想性命名错误", False, True, "错误中联想词的份额上升"),
    "cue_benefit": ("音素提示获益", False, True, "正确提示带来的获益增大"),
    "comprehension": ("理解受损", True, True, "理解题损伤为正"),
}


def call(estimate, lo, hi):
    """present / absent (reversed) / absent, the clinical direction always being positive."""
    if lo > 0:
        return "present"
    if hi < 0:
        return "absent (reversed)"
    return "absent"


def entry(estimate, lo, hi, n, unit, source):
    return dict(estimate=round(float(estimate), 4), ci=[round(float(lo), 4), round(float(hi), 4)], n=n, unit=unit,
                source=source, call=call(estimate, lo, hi))


def from_reports():
    """Markers 1-3 for K8 and C4."""
    k = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))["sets"]["main"]
    c = json.loads(CONSISTENCY.read_text(encoding="utf-8"))["consistency"]
    out = {}
    for lesion, name in LESIONS.items():
        f = k["naming"][f"{name}.knowledge_effect"]
        m = k["m1"][f"{name}.control_effect"]
        out[lesion] = dict(
            frequency=entry(f["estimate"], f["ci_low"], f["ci_high"], 240, "answer log-probability, rare minus common",
                            f"reports/confirm/knowledge/results.json /sets/main/naming/{name}.knowledge_effect"),
            control_demand=entry(m["estimate"], m["ci_low"], m["ci_high"], 63, "logit, strong minus weak distractor",
                                 f"reports/confirm/knowledge/results.json /sets/main/m1/{name}.control_effect"),
            consistency=entry(c[lesion]["mean"], *c[lesion]["ci"], 160, "mean pairwise Spearman correlation",
                              f"reports/explore/qwen_h3c_consistency.json /consistency/{lesion}"))
    return out


def from_experiment3(source=E3, lesions=tuple(LESIONS)):
    """Markers 4-6 per lesion, per concept, as in analyze_experiment3 (its report gives only the means); source: a
    folder with the battery's candidates and labels tables (experiment 3's confirm run, or run_new_sets.py's)."""
    items = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False).set_index("item_id")
    pc = analyze_battery.per_concept(pd.read_parquet(source / "candidates.parquet"), pd.read_parquet(source / "labels.parquet"))
    test_ids = items.index[items.role == "test"].tolist()
    h3a_ids = [i for i in test_ids if i not in ae.H3A_EXCLUDED]
    h3b_ids = items.index[(items.role == "test") & items.free_naming.astype(str).eq("True")].tolist()
    I = pc["intact"]
    cols = ["delta_none_hinted_demos", "delta_cue", "delta_miscue"]
    out = {}
    for lesion in lesions:
        T = pc[lesion]
        d_s = (T.name_S - I.name_S).reindex(h3a_ids).to_numpy()
        Ii, Ts = I[cols].reindex(h3b_ids).to_numpy(), T[cols].reindex(h3b_ids).to_numpy()
        b = (Ts[:, 1] - Ts[:, 0]) - (Ii[:, 1] - Ii[:, 0])
        w2d = (I.word_to_definition_margin - T.word_to_definition_margin).reindex(test_ids).to_numpy()
        out[lesion] = {}
        for marker, x, unit, what in (
                ("associative_errors", d_s, "five-candidate probability, associate minus coordinate and superordinate",
                 "H3a concepts, lesioned minus intact S"),
                ("cue_benefit", b, "logit, change in correct-cue benefit (baseline: no hint with hinted demos)", "H3b concepts, B as in S3"),
                ("comprehension", w2d, "logit, word-to-definition correct-option margin, intact minus lesioned", "all test concepts")):
            if np.isnan(x).any():
                raise SystemExit(f"{lesion} {marker}: missing values")
            rng = np.random.default_rng(SEED)
            s = ae.summary(x, ae.boot_mean(x, ea.resample_counts(len(x), N_BOOT, rng)))
            out[lesion][marker] = entry(s["estimate"], *s["ci"], s["n"], unit, f"{source.name} battery data; {what}")
    return out


def diagnose(model):
    out = {}
    for lesion, m in model.items():
        hits = {g: [k for k, (_, sd, sa, _) in CLINICAL.items() if (m[k]["call"] == "present") == (sd if g == "SD" else sa)]
                for g in ("SD", "SA")}
        n_sd, n_sa = len(hits["SD"]), len(hits["SA"])
        out[lesion] = dict(matches_SD=n_sd, matches_SA=n_sa, matched_SD=hits["SD"], matched_SA=hits["SA"],
                           diagnosis="SD" if n_sd > n_sa else "SA" if n_sa > n_sd else "cannot tell")
    return out


def summarise_random(rows, marker):
    calls = [r["call"] for r in rows]
    est = np.array([r["estimate"] for r in rows])
    return dict(marker=marker, n=len(rows), n_present=calls.count("present"), share_present=round(calls.count("present") / len(rows), 4),
                n_reversed=calls.count("absent (reversed)"), median=round(float(np.median(est)), 4), max=round(float(est.max()), 4),
                expected_share_present_without_effect=0.025, sets=rows)


def post_hoc_size(model, random):
    """Added after the random sets were seen (labelled post hoc): an interval clear of zero picks up any reliable
    small effect, so for markers 1 and 2 also ask whether the lesion's effect exceeds every random set's, the
    comparison the confirmatory tests used, and how many random sets clear the smallest effect of interest (0.20)."""
    out = dict(note="post hoc, added after the random-set calls were seen; not part of the rule written beforehand")
    for side, marker in (("knowledge", "frequency"), ("control", "control_demand")):
        est = np.array([r["estimate"] for r in random[side]["sets"]])
        lo = np.array([r["ci"][0] for r in random[side]["sets"]])
        hi = np.array([r["ci"][1] for r in random[side]["sets"]])
        out[marker] = dict(random_quantiles={q: round(float(np.quantile(est, q)), 4) for q in (0.05, 0.5, 0.95)},
                           random_max=round(float(est.max()), 4), random_lower_bound_above_0_20=int((lo > 0.20).sum()),
                           random_upper_bound_below_minus_0_20=int((hi < -0.20).sum()),
                           exceeds_every_random_set={lesion: bool(model[lesion][marker]["estimate"] > est.max()) for lesion in LESIONS})
    return out


def random_knowledge():
    """Marker 1 for the 200 knowledge-side random sets: batch-16 scoring against the batch-16 intact scores."""
    meta = json.loads((SERVER / "dissociation/meta.json").read_text(encoding="utf-8"))
    items = pd.read_csv(ki.OUT / "items.csv")
    test = items[items.split == "test"].reset_index(drop=True)
    s = pd.read_parquet(SERVER / "dissociation/naming_scores.parquet")
    single = s[s.scoring == "single"]
    intact = single[single.set == "intact"]
    stat = lambda c: {"knowledge_effect": naming.effects(c)["knowledge_effect"]}
    rows = []
    for name in sorted(x for x in single.set.unique() if x.startswith("random_")):
        frame = naming.item_frame(naming.item_damage(intact, single[single.set == name]), test, meta["params"]["zipf_median"])
        if len(frame) != len(test):
            raise SystemExit(f"{name}: {len(frame)} naming items, expected {len(test)}")
        b = damage.bootstrap(frame, stat, N_BOOT, SEED)["knowledge_effect"]
        rows.append(dict(set=name, estimate=round(b["estimate"], 4), ci=[round(b["ci_low"], 4), round(b["ci_high"], 4)],
                         call=call(b["estimate"], b["ci_low"], b["ci_high"])))
    return summarise_random(rows, "frequency")


def random_control():
    """Marker 2 for the 200 control-side random sets."""
    all_items = scoring.load_materials()[0].astype({"item_id": str})
    fm = damage.frequency_median(all_items)
    intact = pd.read_csv(SERVER / "control/intact.csv").astype({"item_id": str})
    lesioned = pd.read_parquet(SERVER / "control/lesioned.parquet").astype({"item_id": str})
    stat = lambda c: {"control_effect": damage.effects(c)["control_effect"]}
    rows = []
    for name in sorted(x for x in lesioned.set.unique() if x.startswith("random_")):
        frame = damage.item_frame(damage.item_damage(intact, lesioned[lesioned.set == name]), all_items, fm)
        b = damage.bootstrap(frame, stat, N_BOOT, SEED)["control_effect"]
        rows.append(dict(set=name, estimate=round(b["estimate"], 4), ci=[round(b["ci_low"], 4), round(b["ci_high"], 4)],
                         call=call(b["estimate"], b["ci_low"], b["ci_high"])))
    return summarise_random(rows, "control_demand")


def supplementary():
    """Descriptive columns: the later patient studies, TMS virtual lesions, and the model's non-semantic task."""
    def rows(file, picks):
        t = pd.read_csv(LIT / file, keep_default_na=False)
        out = []
        for measure, group in picks:
            r = t[(t.measure == measure) & (t.group == group)]
            if len(r) != 1:
                raise SystemExit(f"{file}: expected one row for {measure!r} / {group!r}, found {len(r)}")
            out.append({k: r.iloc[0][k] for k in ("marker", "measure", "group", "value", "unit", "n", "source", "how", "note")})
        return out
    cue = [(m, g) for m in ("benefit of a two-phoneme cue", "benefit of the final cue") for g in ("SD", "SA")]
    cue += [("SA benefit larger than SD, two-phoneme cue", "SA vs SD"), ("SA benefit larger than SD, final cue", "SA vs SD"),
            ("two-phoneme cue vs no cue", "SD"), ("two-phoneme cue vs no cue", "SA")]
    freq = [("frequency effect, high minus low frequency, mean over imageability", g) for g in ("SD", "SA")]
    freq += [("frequency effect", "SD"), ("frequency effect", "SA"), ("itemwise regression, log frequency beta, step 2", "SA")]
    atl, ifg, pmtg, vx = "rTMS left ATL", "rTMS left IFG", "rTMS left pMTG", "rTMS vertex"
    tms = [("specific-level picture naming RT, after TMS", atl), ("specific-level picture naming RT, no TMS", atl),
           ("basic-level picture naming RT, after TMS", atl), ("basic-level picture naming RT, no TMS", atl),
           ("synonym judgement RT, after TMS", atl), ("synonym judgement RT, no TMS", atl),
           ("number judgement RT, after TMS", atl), ("number judgement RT, no TMS", atl),
           ("TMS x naming task (basic, specific, number)", atl), ("TMS x task (synonym vs number judgement)", atl)]
    tms += [(f"{t}, RT TMS effect", s) for s in (ifg, pmtg, vx) for t in (
        "weak-association semantic judgement", "strong-association semantic judgement", "local Navon letter matching (high control)")]
    tms += [("TMS effect larger for weak than strong associations", s) for s in (ifg, pmtg)]
    k = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))["sets"]["main"]["m1"]
    nonsemantic = {lesion: dict(estimate=round(k[f"{name}.string_damage"]["estimate"], 4),
                                ci=[round(k[f"{name}.string_damage"]["ci_low"], 4), round(k[f"{name}.string_damage"]["ci_high"], 4)],
                                unit="logit, M1 string-matching damage", source=f"reports/confirm/knowledge/results.json /sets/main/m1/{name}.string_damage")
                   for lesion, name in LESIONS.items()}
    return dict(cueing_jefferies2008=rows("jefferies2008_cueing.csv", cue), frequency_hoffman2011=rows("hoffman2011_frequency.csv", freq),
                tms=rows("tms_virtual_lesion.csv", tms), model_nonsemantic_task=nonsemantic)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports/explore/qwen_clinical_scorecard.json"))
    ap.add_argument("--skip-random", action="store_true", help="leave out the random sets (quick check)")
    args = ap.parse_args(argv)
    model = from_reports()
    for lesion, extra in from_experiment3().items():
        model[lesion].update(extra)
    inputs = [KNOWLEDGE, CONSISTENCY, E3 / "candidates.parquet", E3 / "labels.parquet", SERVER / "dissociation/naming_scores.parquet",
              SERVER / "control/lesioned.parquet", SERVER / "control/intact.csv", *sorted(LIT.glob("*.csv"))]
    out = dict(note="exploratory: rules fixed before computing (docs/plans/p8-clinical-alignment.md (a), 计算细则); data unsealed earlier",
               commit=runinfo.git("rev-parse", "HEAD"), n_boot=N_BOOT, seed=SEED,
               inputs={str(p.relative_to(ROOT)).replace("\\", "/"): runinfo.sha256(p) for p in inputs},
               clinical={k: dict(label=v[0], SD="present" if v[1] else "absent", SA="present" if v[2] else "absent",
                                 direction=v[3], source="Jefferies & Lambon Ralph 2006, Table 2") for k, v in CLINICAL.items()},
               model=model, diagnosis=diagnose(model), supplementary=supplementary())
    if not args.skip_random:
        out["random"] = dict(knowledge=random_knowledge(), control=random_control())
        out["post_hoc_size"] = post_hoc_size(model, out["random"])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for lesion, d in out["diagnosis"].items():
        calls = ", ".join(f"{k}={model[lesion][k]['call']}" for k in CLINICAL)
        print(f"{lesion}: SD {d['matches_SD']}, SA {d['matches_SA']} -> {d['diagnosis']} ({calls})")
    for side, r in out.get("random", {}).items():
        print(f"random {side}: {r['n_present']}/{r['n']} present, {r['n_reversed']} reversed")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
