"""Knowledge side and the two-way dissociation on the held-out splits (spec.md H2 and G3; user decisions 2026-09-28).

Opens the naming test's held-out split once, at an analysis-plan freeze tag (docs/analysis-plan/knowledge-test.md):

    python experiments/e2_confirm/run_dissociation_test.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3 --freeze-tag analysis-frozen-knowledge

delta_F: knowledge effect (rare-minus-common drop in answer log-probability) of the knowledge set minus that of
the control set, on the held-out naming items. delta_C: control effect of the control set minus that of the
knowledge set, on the held-out M1 items. The main knowledge set is the pre-specified expansion to 8 components
(K8 below); its first 4 (the original knowledge set) are reported only.

Writes outputs/e2_confirm/<model>/dissociation/{naming_scores.parquet, m1_scores.parquet, sets.json,
results.json, meta.json}.
"""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))

import common  # noqa: E402
from lesion import baselines, coder_sources, components, damage, knowledge_items as ki, materials, naming, runinfo, scoring, screen  # noqa: E402
from lesion.components import Component  # noqa: E402

THRESHOLD = 0.20        # smallest effect of interest (spec.md H2): logit on M1, log-probability on the naming test
EQUIVALENCE = 0.10      # "spared" bound (spec.md H2 附加), reported only
MAIN, REPORTED = 8, 4   # knowledge sets: the top 8 verified candidates decide, the top 4 are reported
BATCHES = (16, 11, 7)   # naming scores averaged over these batch compositions, as at the knowledge screen
RANDOM_BATCH = 16       # random sets: one composition each, against an intact score from the same composition
# the main knowledge set as frozen in the analysis plan (Qwen; knowledge screen at d755a4e). Other models (the second
# model, P6) take the top 8 of their own knowledge screen by the same rule.
K8_MODEL = "qwen3-30b-a3b-instruct"
K8 = ("expert:46:61", "expert:46:101", "head:37:24", "expert:45:32",
      "expert:45:48", "expert:43:70", "expert:42:115", "expert:45:93")
ITEMS = "data/materials/m3_knowledge/items.csv"
NAMING_SCORE_FILES = ("src/lesion/naming.py", "data/materials/m3_knowledge/demos.csv")


def verdict(ci, threshold=THRESHOLD):
    """Two tiers (user decision 2026-09-28): the direction is supported when the interval lies above 0; the size
    label says whether it also clears the smallest effect of interest."""
    lo, hi = ci["ci_low"], ci["ci_high"]
    if lo > 0:
        return dict(direction="supported", size="meaningful" if lo > threshold else "small" if hi < threshold else "uncertain")
    return dict(direction="excluded" if hi < threshold else "insufficient", size=None)


def h2(delta_c, delta_f):
    """Both differences supported: two-way dissociation; both excluded: excluded; otherwise insufficient."""
    d = {delta_c["direction"], delta_f["direction"]}
    return "supported" if d == {"supported"} else "excluded" if d == {"excluded"} else "insufficient"


def naming_stat(c):
    k, ctl = naming.effects(c, "knowledge"), naming.effects(c, "control")
    frequent = c["control"][~c["rare"].astype(bool)]
    return {**{f"knowledge.{n}": v for n, v in k.items()}, **{f"control.{n}": v for n, v in ctl.items()},
            "delta_F": k["knowledge_effect"] - ctl["knowledge_effect"],
            "control.common_damage": float(frequent.mean()) if frequent.size else np.nan}


def m1_stat(c):
    k, ctl = damage.effects(c, "knowledge"), damage.effects(c, "control")
    low = c["control"][np.isin(c["task"], damage.MATCHING) & (c["high"] == 0)]
    return {**{f"knowledge.{n}": v for n, v in k.items()}, **{f"control.{n}": v for n, v in ctl.items()},
            "delta_C": ctl["control_effect"] - k["control_effect"],
            "control.low_demand_damage": float(low.mean()) if low.size else np.nan}


def check_screening_items(kmeta, commit, allow_code_change=False):
    """The knowledge sets were chosen on the items that existed at the knowledge screen (the first 300). That file
    must match the recorded hash, each of its items must be unchanged now, and so must the naming code and demos
    (analysis plan §7). allow_code_change: a change to the naming code alone does not stop the run (exploratory reruns
    whose naming scores were shown to be unchanged for this model). Returns the old file's hash."""
    old_commit, want = kmeta["resume_commit"], kmeta["params"]["items_sha256"]
    raw = ((ROOT / ITEMS).read_bytes() if old_commit == commit else
           subprocess.run(["git", "show", f"{old_commit}:{ITEMS}"], cwd=ROOT, capture_output=True, check=True).stdout)
    if hashlib.sha256(raw).hexdigest() != want:
        raise SystemExit(f"{ITEMS} at {old_commit} does not match the knowledge screen's items_sha256")
    read = dict(dtype=str, keep_default_na=False)
    old, now = pd.read_csv(io.BytesIO(raw), **read), pd.read_csv(ROOT / ITEMS, **read).set_index("item_id")
    if not set(old.item_id) <= set(now.index) or not now.loc[old.item_id, old.columns[1:]].reset_index().equals(old):
        raise SystemExit("items used at the knowledge screen were changed afterwards")
    if old_commit != commit and not allow_code_change and runinfo.git("diff", "--name-only", old_commit, commit, "--", *NAMING_SCORE_FILES):
        raise SystemExit(f"{NAMING_SCORE_FILES} changed since the knowledge screen ({old_commit})")
    return want


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True, help="analysis-plan freeze tag (analysis-frozen-...)")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--knowledge-dir", default=None)
    ap.add_argument("--expansion-dir", type=Path, help="completed 8-member development run, required for Coder")
    ap.add_argument("--knowledge-set", default=None,
                    help="comma-separated component keys tested in place of the knowledge screen's top 8, e.g. a set "
                         "chosen by another screen (exploratory re-screens, 2026-10-13); the screen still gives the "
                         "Zipf median and the item checks")
    ap.add_argument("--control-set", default=None,
                    help="comma-separated component keys tested in place of the M1 screen's control set (exploratory)")
    ap.add_argument("--allow-score-code-change", action="store_true",
                    help="exploratory reruns: do not stop when score-relevant code changed since the screens (recorded "
                         "in the run's metadata); only after checking that the change leaves scores identical")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    kdir = Path(args.knowledge_dir) if args.knowledge_dir else ROOT / "outputs/e3_knowledge" / args.model / "screen"
    kmeta = json.loads((kdir / "meta.json").read_text(encoding="utf-8"))
    if kmeta.get("status") != "done" or runinfo.sha256(kdir / "selection.json") != kmeta.get("selection_sha256"):
        raise SystemExit(f"{kdir} is unfinished or its selection changed after screening")
    kselection = json.loads((kdir / "selection.json").read_text(encoding="utf-8"))
    ranked = pd.read_csv(kdir / "screen.csv").sort_values("knowledge_effect", ascending=False).component.tolist()
    sets = {"main": ranked[:MAIN], "reported": ranked[:REPORTED]}
    if args.knowledge_set:
        sets = {"main": args.knowledge_set.split(",")}
    elif kselection["knowledge"] != sets["reported"] or (args.model == K8_MODEL and not args.tiny and tuple(sets["main"]) != K8):
        raise SystemExit(f"the knowledge sets {sets} differ from the ones frozen in the analysis plan")
    expansion_sha256 = None
    if args.model == coder_sources.MODEL or args.expansion_dir:
        coder_sources.knowledge_screen(kdir, require_lock=True)
        if not args.expansion_dir or args.knowledge_set:
            ap.error("Coder requires its completed development expansion and no replacement knowledge set")
        expansion_sha256 = coder_sources.check_expansion(args.expansion_dir, kmeta, sets["main"])
    screening_items_sha256 = check_screening_items(kmeta, commit, args.allow_score_code_change)
    median = kmeta["params"]["zipf_median"]

    m1_dir = common.out_dir(args, "screen")
    m1_meta = json.loads((m1_dir / "meta.json").read_text(encoding="utf-8"))
    control = [Component.parse(k) for k in common.load_selection(m1_dir, m1_meta)["control"]]
    if args.control_set:
        control = [Component.parse(k) for k in args.control_set.split(",")]
    test = ki.load_items("test", freeze_tag=args.freeze_tag)
    m1_test = materials.load_split("test", freeze_tag=args.freeze_tag)
    m1_all, m1_demos, templates = scoring.load_materials()
    fm = damage.frequency_median(m1_all)

    model, tok, chat, means, ablate = common.load(args)
    p_naming = naming.build_prompts(test, pd.read_csv(ki.OUT / "demos.csv"), chat)
    p_m1 = scoring.build_label_prompts(m1_test, m1_demos, templates, variant="main", chat=chat)
    out = Path(args.out) if args.out else ROOT / "outputs/e2_confirm" / args.model / "dissociation"
    fields = common.run_fields(
        args, commit, "dissociation_test", "test", len(p_naming) + len(p_m1), n_random=args.n_random,
        batches=list(BATCHES), random_batch=RANDOM_BATCH, zipf_median=median, knowledge_sets=sets,
        control=[c.key for c in control], freeze_tag=args.freeze_tag, allow_score_code_change=args.allow_score_code_change, knowledge_selection_sha256=kmeta["selection_sha256"],
        screening_items_sha256=screening_items_sha256, items_sha256=runinfo.sha256(ROOT / ITEMS),
        demos_sha256=runinfo.sha256(ki.OUT / "demos.csv"),
        knowledge_screen_sha256=kmeta.get("screen_csv_sha256"), expansion_sha256=expansion_sha256,
        knowledge_screen_commit=kmeta.get("resume_commit"), m1_screen_commit=m1_meta.get("resume_commit"),
        code_changes_since_knowledge_screen=common.changes_since(kmeta.get("resume_commit"), commit))
    for meta, where in ((m1_meta, m1_dir), (kmeta, kdir)):
        common.check_same_run(meta, fields, where, keys=[k for k in common.SAME_RUN if k != "analysis_plan_sha256"],
                              same_code=False, allow_score_change=args.allow_score_code_change)
    common.check_resume(out, fields)
    runinfo.write_meta(out, status="running", **fields)

    lesions = {"control": control, **{name: [Component.parse(k) for k in keys] for name, keys in sets.items()}}
    kinds = sorted({c.kind for c in lesions["main"]})
    pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
    freq = means.routing_frequency() if "expert" in kinds else None
    routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if freq is not None else None
    randoms = baselines.matched_random_sets(lesions["main"], pool, args.n_random, seed=13,
                                            exclude=lesions["main"] + control, routing_freq=routing)
    (out / "sets.json").write_text(json.dumps({**{k: [c.key for c in v] for k, v in lesions.items()},
                                               **{f"random_{i:03d}": [c.key for c in s] for i, s in enumerate(randoms)}},
                                              indent=2), encoding="utf-8")

    t0 = time.time()

    def naming_scores(comps=(), batches=BATCHES):
        with ablate(list(comps)) if comps else nullcontext():
            return naming.averaged_logprobs(model, tok, p_naming, chat, batches)

    label_score = screen.make_scorer(model, tok, p_m1, chat, ablate, args.batch_size, common.SCORERS[args.scorer])
    n_scores = {"intact": naming_scores(), **{k: naming_scores(v) for k, v in lesions.items()}}
    m1_scores = {"intact": label_score(), **{k: label_score(v) for k, v in lesions.items()}}
    single = {"intact": naming_scores(batches=(RANDOM_BATCH,)), "main": naming_scores(lesions["main"], (RANDOM_BATCH,))}
    for i, s in enumerate(randoms):
        single[f"random_{i:03d}"] = naming_scores(s, (RANDOM_BATCH,))
    pd.concat([pd.concat(n_scores, names=["set"]).reset_index(level=0).assign(scoring="averaged"),
               pd.concat(single, names=["set"]).reset_index(level=0).assign(scoring="single")]
              ).to_parquet(out / "naming_scores.parquet", index=False)
    pd.concat(m1_scores, names=["set"]).reset_index(level=0).to_parquet(out / "m1_scores.parquet", index=False)

    results = dict(control=[c.key for c in control], knowledge_sets=sets, sets={}, n_boot=args.n_boot,
                   overlapping_components=sorted(set(sets["main"]) & {c.key for c in control}),
                   n_items=dict(naming=len(test), naming_rare=int((test.zipf < median).sum()), m1=len(m1_test),
                                m1_colour_size=int(m1_test.task.isin(damage.MATCHING).sum())))
    for name in sets:
        hn = pd.concat({"knowledge": naming.item_damage(n_scores["intact"], n_scores[name]),
                        "control": naming.item_damage(n_scores["intact"], n_scores["control"])}, axis=1)
        hm = pd.concat({"knowledge": damage.item_damage(m1_scores["intact"], m1_scores[name]),
                        "control": damage.item_damage(m1_scores["intact"], m1_scores["control"])}, axis=1)
        results["sets"][name] = dict(components=sets[name],
                                     naming=damage.bootstrap(naming.item_frame(hn, test, median), naming_stat, args.n_boot),
                                     m1=damage.bootstrap(damage.item_frame(hm, m1_test, fm), m1_stat, args.n_boot))

    def knowledge_effect(intact, lesioned):
        frame = naming.item_frame(naming.item_damage(intact, lesioned), test, median)
        return naming.effects(damage.columns(frame))["knowledge_effect"]

    main_set = results["sets"]["main"]
    rand = np.array([knowledge_effect(single["intact"], v) for k, v in single.items() if k.startswith("random_")])
    target = main_set["naming"]["knowledge.knowledge_effect"]["estimate"]
    at_least = int(np.sum(rand >= target))
    results["random"] = dict(n=len(rand), matched_to="main", scoring=f"batch {RANDOM_BATCH} only", median=float(np.median(rand)),
                             max=float(np.max(rand)), at_least_target=at_least, empirical_p=(1 + at_least) / (1 + len(rand)),
                             main_single_scoring=knowledge_effect(single["intact"], single["main"]))
    matched_count = int(np.sum(rand >= results["random"]["main_single_scoring"]))
    results["random"].update(at_least_matched_target=matched_count,
                             matched_empirical_p=(1 + matched_count) / (1 + len(rand)))
    dc, df = verdict(main_set["m1"]["delta_C"]), verdict(main_set["naming"]["delta_F"])
    results["h2"] = dict(delta_C=dc, delta_F=df, decision=h2(dc, df), threshold=THRESHOLD)
    results["descriptive"] = dict(knowledge_effect=verdict(main_set["naming"]["knowledge.knowledge_effect"]),
                                  spared={k: dict(ci_high=ci["ci_high"], below_equivalence=ci["ci_high"] < EQUIVALENCE)
                                          for k, ci in (("m1_low_demand", main_set["m1"]["control.low_demand_damage"]),
                                                        ("naming_common", main_set["naming"]["control.common_damage"]))})
    control_test = ROOT / "outputs/e2_confirm" / args.model / "control/results.json"
    if not args.tiny and not args.control_set and control_test.exists():     # the control test of the screen's own set
        results["control_test_control_effect"] = json.loads(control_test.read_text(encoding="utf-8"))["intervals"]["control_effect"]
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)

    for label, ci, v in (("delta_F", main_set["naming"]["delta_F"], df), ("delta_C", main_set["m1"]["delta_C"], dc)):
        print(f"{label} {ci['estimate']:.2f} [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}] -> {v['direction']} ({v['size']})")
    k = main_set["naming"]["knowledge.knowledge_effect"]
    print(f"H2: {results['h2']['decision']}; knowledge effect {k['estimate']:.2f} [{k['ci_low']:.2f}, {k['ci_high']:.2f}]; "
          f"random {at_least}/{len(rand)} at least as large")


if __name__ == "__main__":
    main()
