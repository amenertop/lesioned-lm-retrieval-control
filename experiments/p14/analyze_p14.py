"""P14 first reading (exploratory): one summary of session 15's gate training, confirmation and final tests.

    python experiments/p14/analyze_p14.py --freeze-tag analysis-frozen-p14-reading [--root ...] [--out ...]

The sealed naming items are read under an analysis freeze tag (lesion.runinfo.check_analysis_freeze): run it from a
clean checkout of the tag that holds this script; --root then points at the fetched outputs.

Reads the fetched outputs only (no model); the computations are in lesion.p14_analysis (agent2's code review,
2026-10-17, reports/explore/2026-10-17-p14-code-review.md):
- every run's meta.json and gates.csv: side, lambda_sp, seed, chosen k, gates below 0.5, skipped steps;
- confirm/: the chosen set per objective, its minimality and stability, and its overlap with K8 and C4;
- final_knowledge/ and final_hub/ (run_dissociation_test on the sealed items): the stored statistics, plus P14's own
  naming statistic (the rare coefficient with the answer's token count as covariate, as in training and confirmation)
  with an item bootstrap and a comparison with the random sets, and the hub's M1 synonym rare-minus-common effect;
- margin scaling of the control and knowledge sets on the sealed M1 items, margins averaged over the two option orders,
  and the strong-distractor coefficient at equal intact margin (cluster bootstrap over concept groups). All of this
  was added after the data were seen.
Before joining, the current materials must match the final tests' records (M1 files and the naming items), and the
sets in confirm/groups.json must be the ones the final tests used. The output records the analysis commit, every
input's SHA-256, the definitions, the number of draws and the seed.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))

from lesion import knowledge_items as ki, p14_analysis as pa, runinfo, scoring  # noqa: E402
from run_dissociation_test import K8  # noqa: E402

N_BOOT, SEED = 2000, 0


def c4_members():
    """C4 as the frozen control test recorded it."""
    return tuple(json.loads((ROOT / "reports/confirm/control/results.json").read_text(encoding="utf-8"))["control"])


def runs_table(root):
    rows = []
    for d in sorted((root / "runs").iterdir()):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        gates = pd.read_csv(d / "gates.csv") if (d / "gates.csv").exists() else None
        rows.append(dict(run=d.name, side=meta.get("side"), lambda_sp=meta.get("lambda_sp"), seed=meta.get("seed"),
                         pool=meta.get("pool"), status=meta.get("status"), chosen_k=meta.get("chosen_k"),
                         delta=meta.get("delta"), skipped_share=meta.get("skipped_share"),
                         gates_below_half=int((gates.g < 0.5).sum()) if gates is not None else None,
                         n_gates=len(gates) if gates is not None else None))
    return rows


def check_inputs(root, groups):
    """Current materials as the final tests recorded them; the sets of groups.json are the ones tested."""
    for name in ("final_knowledge", "final_hub"):
        meta = json.loads((root / name / "meta.json").read_text(encoding="utf-8"))
        if meta.get("status") != "done":
            raise SystemExit(f"{name} is not done")
        for f, sha in meta["materials_sha256"].items():
            if runinfo.sha256(scoring.M1 / f) != sha:
                raise SystemExit(f"{scoring.M1 / f} differs from the material {name} was scored with")
        if runinfo.sha256(ki.OUT / "items.csv") != meta["params"]["items_sha256"]:
            raise SystemExit(f"{ki.OUT / 'items.csv'} differs from the naming items {name} was scored with")
        res = json.loads((root / name / "results.json").read_text(encoding="utf-8"))
        side = "knowledge" if name == "final_knowledge" else "hub"
        if res["sets"]["main"]["components"] != groups[side]["members"] or res["control"] != groups["control"]["members"]:
            raise SystemExit(f"{name} tested other sets than confirm/groups.json lists")
    params = json.loads((root / "final_knowledge/meta.json").read_text(encoding="utf-8"))["params"]
    return params["zipf_median"], params["freeze_tag"]


def final_summary(path, test_items, median):
    r = json.loads((path / "results.json").read_text(encoding="utf-8"))
    s = r["sets"]["main"]
    pick = lambda block, keys: {k: s[block][k] for k in keys if k in s[block]}
    naming_scores = pd.read_parquet(path / "naming_scores.parquet")
    frame = pa.naming_frame(naming_scores, test_items, median)
    return dict(
        components=s["components"], n_items=r["n_items"], h2=r["h2"], descriptive=r["descriptive"],
        stored=dict(note="lesion's final test (run_dissociation_test): rare minus common naming damage without the "
                         "token-count covariate; comparable with K8's frozen result",
                    naming=pick("naming", ("knowledge.knowledge_effect", "knowledge.naming_damage", "delta_F",
                                           "control.knowledge_effect", "control.naming_damage")),
                    m1=pick("m1", ("knowledge.knowledge_effect", "knowledge.control_effect", "knowledge.semantic_damage",
                                   "knowledge.string_damage", "control.control_effect", "control.control_slope",
                                   "control.semantic_damage", "control.string_damage")),
                    random=r["random"]),
        p14_statistic=dict(note="P14's objective statistic on the sealed naming items (as in training and confirmation)",
                           averaged=pa.rare_coefficient_ci(frame, N_BOOT, SEED),
                           random_comparison=pa.random_comparison(naming_scores, test_items, median)),
        m1_synonym_rare_minus_common=dict(note="the comprehension half of the hub objective, stored by the final test",
                                          **s["m1"]["knowledge.knowledge_effect"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs/p14/qwen3-30b-a3b-instruct"))
    ap.add_argument("--out", default=str(ROOT / "reports/explore/p14_session15.json"))
    ap.add_argument("--freeze-tag", required=True, help="the analysis tag of this checkout (reads the sealed items)")
    args = ap.parse_args()
    root = Path(args.root)
    groups = json.loads((root / "confirm/groups.json").read_text(encoding="utf-8"))
    stability = json.loads((root / "confirm/stability.json").read_text(encoding="utf-8"))
    median, tested_tag = check_inputs(root, groups)
    c4 = c4_members()
    sets = {}
    for side, g in groups.items():
        st = stability.get(side, {})
        freq = st.get("frequency", {})
        sets[side] = dict(run=g["run"], k=g["k"], members=g["members"], **pa.minimality_fields(side, g),
                          select_selectivity=g["select_selectivity"], confirm_selectivity=g["confirm_selectivity"],
                          selectivity_definition=("naming rare coefficient plus M1 synonym rare-minus-common, each over "
                                                  "its intact spread, on the select split; confirm: naming only"
                                                  if side == "hub" else
                                                  "control: strong minus weak distractor damage" if side == "control"
                                                  else "naming rare coefficient with token-count covariate"),
                          stability=dict(n_runs=st.get("n_runs"), mean_jaccard=st.get("mean_jaccard"),
                                         chosen_k=st.get("chosen_k"),
                                         in_every_run=sorted(k for k, f in freq.items() if f == 1.0)),
                          overlap_with_K8=sorted(set(g["members"]) & set(K8)), overlap_with_C4=sorted(set(g["members"]) & set(c4)))
    minim = pd.read_csv(root / "confirm/minimality.csv")
    sets["control"]["minimality_detail"] = pa.control_minimality(minim)
    # the sealed items under this analysis's own tag; check_inputs made sure they are the items the final tests used
    test_items = ki.load_items("test", freeze_tag=args.freeze_tag).astype({"item_id": str})
    m1_items = pa.item_table(scoring.load_materials()[0])
    m1 = pd.read_parquet(root / "final_knowledge/m1_scores.parquet")
    w = pa.order_averaged(m1, m1_items, ["intact", "control", "main"])
    scaling = {name: dict(by_task=pa.margin_scaling(w, s), by_demand=pa.demand_breakdown(w, s),
                          strong_minus_weak_at_equal_intact=pa.demand_at_equal_intact(w, s, N_BOOT, SEED))
               for name, s in (("control_set", "control"), ("knowledge_set", "main"))}
    outputs = sorted(p for p in root.rglob("*") if p.is_file() and p.parent.name in ("confirm", "final_knowledge", "final_hub"))
    repo = [ROOT / "reports/confirm/control/results.json", ki.OUT / "items.csv", scoring.M1 / "items.csv"]
    inputs = {**{f"<root>/{p.relative_to(root).as_posix()}": runinfo.sha256(p) for p in outputs},
              **{p.relative_to(ROOT).as_posix(): runinfo.sha256(p) for p in repo}}
    out = dict(stage="p14_first_reading", evidence="exploratory; the scaling analysis and P14's statistic on the final "
                                                  "items were added after the data were seen",
               analysis_commit=runinfo.commit(), analysis_tag=args.freeze_tag, final_tests_tag=tested_tag,
               n_boot=N_BOOT, seed=SEED,
               root=root.as_posix(), inputs=inputs,
               runs=runs_table(root), sets=sets, C4=list(c4),
               final_knowledge=final_summary(root / "final_knowledge", test_items, median),
               final_hub=final_summary(root / "final_hub", test_items, median), margin_scaling=scaling)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
