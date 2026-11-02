"""Control-side test on the held-out M1 split (spec.md 核心证伪实验; user decisions 2026-09-28).

Opens the held-out split once, at an analysis-plan freeze tag (docs/analysis-plan/control-test.md):

    python experiments/e2_confirm/run_control_test.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3 --freeze-tag analysis-frozen-control

Uses the control set chosen on the screening split (outputs/e1_screen/<model>/screen/selection.json) and
writes outputs/e2_confirm/<model>/control/{intact.csv, lesioned.parquet, results.json, meta.json}.
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

import common  # noqa: E402
from lesion import baselines, components, damage, materials, runinfo, scoring, screen  # noqa: E402
from lesion.components import Component  # noqa: E402

THRESHOLD = 0.20


def core_claim(primary, association):
    """The two conditions of control-test.md §4, including an unestimable slope."""
    if "falsified" in (primary, association):
        return "falsified"
    if primary == "supported" and association == "not_falsified":
        return "supported"
    return "insufficient"


def verdict(ci, threshold=THRESHOLD):
    if ci["ci_low"] > threshold:
        return "supported"
    if ci["ci_high"] < threshold:
        return "falsified"
    return "insufficient"


def association_slope(cols):
    """Damage regressed on the probe-distractor similarity (task as covariate), colour and size items."""
    fm = np.isin(cols["task"], damage.MATCHING)
    y, x = cols["h"][fm], cols["w2v_distractor_probe"][fm]
    if len(y) < 4 or np.std(x) == 0:
        return dict(association_slope=np.nan)
    X = np.column_stack([np.ones(len(y)), (x - x.mean()) / x.std(), (cols["task"][fm] == "size").astype(float)])
    return dict(association_slope=float(np.linalg.lstsq(X, y, rcond=None)[0][1]))


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True, help="analysis-plan freeze tag (analysis-frozen-...)")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--control-set", default=None,
                    help="comma-separated component keys tested in place of the screen's control set, e.g. a set "
                         "chosen by another rule (exploratory re-screens, 2026-10-13)")
    ap.add_argument("--allow-score-code-change", action="store_true",
                    help="exploratory reruns: do not stop when score-relevant code changed since the screens (recorded "
                         "in the run's metadata); only after checking that the change leaves scores identical")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    items = materials.load_split("test", freeze_tag=args.freeze_tag)
    all_items, demos, templates = scoring.load_materials()
    screen_dir = common.out_dir(args, "screen")
    screen_meta = json.loads((screen_dir / "meta.json").read_text(encoding="utf-8"))
    selection = common.load_selection(screen_dir, screen_meta)
    if args.control_set:
        control = [Component.parse(k) for k in args.control_set.split(",")]
    elif selection["status"]["control"] != "ok":
        raise SystemExit(f"no control set was chosen at screening: {selection['status']}")
    else:
        control = [Component.parse(k) for k in selection["control"]]

    model, tok, chat, means, ablate = common.load(args)
    prompts = scoring.build_label_prompts(items, demos, templates, variant="main", chat=chat)
    out = Path(args.out) if args.out else ROOT / "outputs/e2_confirm" / args.model / "control"
    fields = common.run_fields(args, commit, "control_test", "test", len(prompts), n_random=args.n_random,
                               associations_sha256=runinfo.sha256(materials.ASSOCIATIONS),
                               freeze_tag=args.freeze_tag, allow_score_code_change=args.allow_score_code_change, screen_commit=screen_meta.get("resume_commit"),
                               code_changes_since_screen=common.changes_since(screen_meta.get("resume_commit"), commit))
    common.check_same_run(screen_meta, fields, screen_dir, same_code=False, allow_score_change=args.allow_score_code_change,
                          keys=[k for k in common.SAME_RUN if k != "analysis_plan_sha256"])
    common.check_resume(out, fields)
    runinfo.write_meta(out, status="running", control=[c.key for c in control], **fields)

    kinds = sorted({c.kind for c in control})
    pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
    freq = means.routing_frequency() if "expert" in kinds else None
    routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if freq is not None else None
    sets = {"control": control}
    for i, s in enumerate(baselines.matched_random_sets(control, pool, args.n_random, seed=11, exclude=control,
                                                        routing_freq=routing)):
        sets[f"random_control_{i:03d}"] = s
    (out / "sets.json").write_text(json.dumps({k: [c.key for c in v] for k, v in sets.items()}, indent=2), encoding="utf-8")

    t0 = time.time()
    score = screen.make_scorer(model, tok, prompts, chat, ablate, args.batch_size, common.SCORERS[args.scorer])
    intact = score()
    intact.to_csv(out / "intact.csv", index=False)
    lesioned = {name: score(comps) for name, comps in sets.items()}
    pd.concat(lesioned, names=["set"]).reset_index(level=0).to_parquet(out / "lesioned.parquet", index=False)

    fm = damage.frequency_median(all_items)
    assoc = pd.read_csv(materials.ASSOCIATIONS, dtype={"item_id": str})
    frame = damage.item_frame(damage.item_damage(intact, lesioned["control"]), items, fm)
    frame = frame.merge(assoc, on="item_id", how="left")
    stat = lambda c: {**damage.effects(c), **association_slope(c)}
    ci = damage.bootstrap(frame, stat, args.n_boot)
    random_effects = [damage.effects(damage.columns(damage.item_frame(damage.item_damage(intact, v), items, fm)))["control_effect"]
                      for n, v in lesioned.items() if n.startswith("random_")]
    at_least = int(np.sum(np.array(random_effects) >= ci["control_effect"]["estimate"]))
    # size relative to the intact model's own interference effect on the same items (TDS effect, G1 units)
    from lesion import g1
    tab = g1.item_table(intact, items)
    intact_e = float(g1.tds_effect_stat(g1.tds_gap(all_items))(tab[tab.task.isin(g1.MATCHING)]))
    results = dict(
        control=[c.key for c in control], intervals=ci,
        primary=dict(statistic="control_effect", threshold=THRESHOLD, verdict=verdict(ci["control_effect"])),
        association=dict(statistic="association_slope",
                         verdict=("not_estimable" if not np.isfinite(ci["association_slope"]["estimate"])
                                  else "falsified" if ci["association_slope"]["ci_high"] <= 0 else "not_falsified")),
        relative_to_intact=dict(intact_tds_effect=intact_e,
                                control_effect_share=ci["control_effect"]["estimate"] / intact_e if intact_e else None),
        random=dict(n=len(random_effects), at_least_target=at_least, empirical_p=(1 + at_least) / (1 + len(random_effects)),
                    median=float(np.median(random_effects)), max=float(np.max(random_effects))),
        n_items=len(items), n_concept_groups=int(items.concept_group.nunique()), n_boot=args.n_boot)
    results["core_claim"] = dict(verdict=core_claim(results["primary"]["verdict"], results["association"]["verdict"]),
                                conditions=["primary", "association"])
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    runinfo.write_meta(out, status="done", control=[c.key for c in control], seconds=round(time.time() - t0, 1), **fields)
    c = ci["control_effect"]
    print(f"control effect {c['estimate']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}] -> {results['primary']['verdict']}; "
          f"association slope {ci['association_slope']['estimate']:.2f} -> {results['association']['verdict']}; "
          f"random {at_least}/{len(random_effects)} at least as large")


if __name__ == "__main__":
    main()
