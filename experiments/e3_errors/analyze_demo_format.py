"""Analysis of the demonstration-format check (exploratory; docs/plans/demo-format-check.md). No GPU.

    python experiments/e3_errors/analyze_demo_format.py [--dir outputs/e3_errors/qwen3-30b-a3b-instruct/demo_format]

Per concept: the target's summed log-probability averaged over the batch layouts (as m3_scoring.averaged_naming does)
and the naming loss, intact minus lesioned. Means with 95% percentile intervals from 10,000 resamples of the concepts
(seed 1910; one index matrix per item set, so every statistic of a set is resampled in the same way; family concepts in
family order, as in lesion.title_gap_analysis.bootstrap_family_mean, so the final family analysis's interval for the
existing family cell is reproduced exactly).

Reported: the cells of this run (new formats: battery without demos, families with demos; with --also-existing also
the existing formats); the existing values from the results ledger (EXISTING ids) and, when the original outputs are
present, recomputed from them with intervals; the format effect, new minus existing format, paired by concept, within
this run (--also-existing) and across runs (this run against the original outputs); the reproduction check; and which
of the reading rules fixed in the plan before the run the result meets (reading()). Writes
reports/explore/demo-format-<YYYYMMDD of the run>/demo-format.json (or --out), in the layout the ledger builder reads
(estimate, ci_low, ci_high, n, n_unit). Exploratory: not part of the confirmatory family of Table S6.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import run_demo_format as rdf  # noqa: E402
from lesion import runinfo  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
N_BOOT, SEED = 10_000, 1910
THRESHOLD = 0.20     # the plan's bound for "above zero"; also the smallest effect of interest of experiment 3's S tests
LESIONS = ("WR8", "SC4")
NEW = dict(battery="no_demos", family="demos")
OLD = dict(battery="demos", family="no_demos")
LEDGER = ROOT / "docs/paper-prep/results_ledger.csv"
# existing naming losses (lesioned against intact, 160 M3 test concepts and 120 final family concepts) as cited
EXISTING = {
    ("battery", "SC4"): "experiment3-b212cfe8d967",                     # report-only C4 naming loss, 0.0289
    ("battery", "WR8"): "experiment3-88fae3396929",                     # S2 K8 naming loss, 0.84 [0.45, 1.26]
    ("family", "SC4"): "title_gap_fine_remaining_core-9a0a976b077d",    # 0.78 [0.62, 0.96]
    ("family", "WR8"): "title_gap_fine_remaining_core-e7c009e99e28",    # 0.90 [0.56, 1.27]
}
ORIGINAL_BATTERY = ROOT / "outputs/e3_errors" / MODEL / "confirm"
ORIGINAL_FAMILY = ROOT / "outputs/title_gap_execution_20261019/fine_final_core"
ORIGINAL_STATES = dict(battery=dict(intact="intact", WR8="K8", SC4="C4"),
                       family=dict(intact="intact", WR8="WR8_a0", SC4="SC4_a0"))


# --- per-concept scores -------------------------------------------------------------------------------------------

def target_scores(scores):
    """(item_set, format, state) -> Series item_id -> target log-probability averaged over the layouts."""
    t = scores[scores.role == "target"]
    layouts = t.groupby(["item_set", "format", "state", "item_id"]).layout.nunique()
    if layouts.nunique() != 1:
        raise ValueError("some concepts lack a batch layout")
    mean = t.groupby(["item_set", "format", "state", "item_id"]).logprob.mean()
    return {key: g.droplevel([0, 1, 2]) for key, g in mean.groupby(level=[0, 1, 2])}


def concept_order(item_set, ids, items=None):
    """Battery concepts by item_id; family concepts by family, as the final family analysis resampled them."""
    ids = list(ids)
    if item_set == "family":
        items = rdf.family_items() if items is None else items
        family = items.set_index("item_id").family
        return sorted(ids, key=lambda i: (family[i], i))
    return sorted(ids)


class Resampler:
    """One index matrix per item set (concepts in concept_order); 95% percentile intervals of means."""

    def __init__(self, n_boot=N_BOOT, seed=SEED):
        self.n_boot, self.seed, self.index = n_boot, seed, {}

    def stat(self, item_set, x, unit="nat"):
        x = np.asarray(x, dtype=float)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("missing or non-finite per-concept values")
        if (item_set, len(x)) not in self.index:
            self.index[(item_set, len(x))] = np.random.default_rng(self.seed).integers(len(x), size=(self.n_boot, len(x)))
        lo, hi = np.quantile(x[self.index[(item_set, len(x))]].mean(1), [0.025, 0.975])
        return dict(estimate=float(x.mean()), ci_low=float(lo), ci_high=float(hi), n=len(x), n_unit="概念", unit=unit)


# --- existing values ------------------------------------------------------------------------------------------------

def ledger_rows(path=LEDGER, ids=tuple(EXISTING.values())):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = {r["编号"]: r for r in csv.DictReader(handle) if r["编号"] in ids}
    missing = set(ids) - set(rows)
    if missing:
        raise SystemExit(f"ledger rows not found: {sorted(missing)}")

    def num(v):
        return float(v) if v != "" else None

    return {i: dict(value=num(r["数值"]), ci_low=num(r["95%区间下限"]), ci_high=num(r["95%区间上限"]),
                    n=int(r["n"]) if r["n"] else None, source=r["来源文件"], pointer=r["指标路径"]) for i, r in rows.items()}


def original_scores(battery_dir=ORIGINAL_BATTERY, family_dir=ORIGINAL_FAMILY):
    """Target log-probabilities of the existing formats from the stored original runs, keyed like target_scores
    (states renamed to intact, WR8, SC4); with the identity checks done. Missing outputs: an empty dict and a note."""
    out, notes, inputs = {}, [], {}
    candidates, meta = Path(battery_dir) / "candidates.parquet", Path(battery_dir) / "meta.json"
    if candidates.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("final_items_sha256") != rdf.BATTERY_SHA256 or m.get("means_sha256") != rdf.MEANS_SHA256:
            raise SystemExit(f"{battery_dir} used other items or means")
        c = pd.read_parquet(candidates)
        c = c[(c.condition == "name_free") & (c.role == "target")]
        for state, original in ORIGINAL_STATES["battery"].items():
            if state != "intact" and m["sets"][original] != list(sets_by_code()[state]):
                raise SystemExit(f"{battery_dir}: {original} is not {state}")
            out[("battery", OLD["battery"], state)] = c[c.state == original].set_index("item_id").logprob
        inputs[relative(candidates)] = runinfo.sha256(candidates)
    else:
        notes.append(f"{battery_dir} not found: battery comparisons use the ledger value only")
    plan = Path(family_dir) / "plan.json"
    if plan.exists():
        p = json.loads(plan.read_text(encoding="utf-8"))
        prov = p["provenance"]
        if prov.get("items_sha256") != rdf.FAMILY_SHA256 or prov.get("means_sha256") != rdf.MEANS_SHA256:
            raise SystemExit(f"{family_dir} used other items or means")
        states = {s["name"]: s for s in p["states"]}
        for state, original in ORIGINAL_STATES["family"].items():
            directory = Path(family_dir) / original
            record = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
            path = directory / "candidates.parquet"
            if runinfo.sha256(path) != record["files"]["candidates.parquet"]:
                raise SystemExit(f"{path} differs from its completion record")
            if state != "intact" and set(states[original]["gates"]) != set(sets_by_code()[state]):
                raise SystemExit(f"{family_dir}: {original} is not {state}")
            c = pd.read_parquet(path)
            c = c[(c.condition == "naming") & (c.role == "target")]
            out[("family", OLD["family"], state)] = c.set_index("item_id").logprob
            inputs[relative(path)] = runinfo.sha256(path)
        out["_family_prompts"] = {q["item_id"]: q["text"] for q in p["naming"] if q["condition"] == "naming"}
        out["_family_batches"] = p.get("naming_batches") or p["batches"]
    else:
        notes.append(f"{family_dir} not found: family comparisons use the ledger value only")
    return out, notes, inputs


def sets_by_code():
    return rdf.component_sets(MODEL)


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


# --- reading rules (docs/plans/demo-format-check.md, 怎么判读, fixed before the run) -------------------------------

RULE_TEXT = {
    "format": "格式造成：去掉示范题后 SC4 在题库上的命名损失升到家族题的水平，家族题加上示范题后降到接近零。"
              "按计划，正文改写为“SC4 在有示范题时不伤命名”，讨论里写示范题如何让模型绕过控制。",
    "concept": "概念造成：两种格式下 SC4 的命名损失都和原来一样（题库仍接近零，家族题仍约 0.78）。"
               "按计划，正文删掉“分不开两种原因”那句，改为“差异来自概念，不来自格式”。",
    "between": "介于两者之间：照实报告两个方向的变化量，不强行归类。",
}


def reading(battery_new, family_new, family_value, battery_value, threshold=THRESHOLD):
    """Which rule of the plan the SC4 result meets.

    battery_new, family_new: SC4 naming loss (estimate, ci_low, ci_high, optionally n) on the battery without demos
    and on the families with demos; family_value, battery_value: the existing SC4 losses (ledger: 0.78 and 0.03).
    The reference values sit under the reading's meta (the ledger builder skips them; they are ledger rows already).
    - format: the battery interval overlaps the family value (contains it) with its lower limit above 0.20, and the
      families with demos are near zero;
    - concept: the battery is still near zero and the family interval still contains the family value;
    - between: anything else; the two changes are reported.
    "Near zero" is not given a bound in the plan; here it means an upper limit below 0.20, the bound the plan's
    first rule uses for "above zero" (and experiment 3's smallest effect of interest)."""
    def contains(s, v):
        return s["ci_low"] <= v <= s["ci_high"]

    def near_zero(s):
        return s["ci_high"] < threshold

    checks = dict(battery_interval_contains_family_value=contains(battery_new, family_value),
                  battery_lower_limit_above_threshold=battery_new["ci_low"] > threshold,
                  family_near_zero=near_zero(family_new),
                  battery_near_zero=near_zero(battery_new),
                  family_interval_contains_family_value=contains(family_new, family_value))
    if (checks["battery_interval_contains_family_value"] and checks["battery_lower_limit_above_threshold"]
            and checks["family_near_zero"]):
        rule = "format"
    elif checks["battery_near_zero"] and checks["family_interval_contains_family_value"]:
        rule = "concept"
    else:
        rule = "between"
    change = {name: dict(estimate=new["estimate"] - old, n=new.get("n"), n_unit="概念", unit="nat")
              for name, new, old in (("battery", battery_new, battery_value), ("family", family_new, family_value))}
    return dict(rule=rule, text=RULE_TEXT[rule], checks=checks, change_from_existing=change,
                meta=dict(threshold=threshold, compared_with=dict(family=family_value, battery=battery_value)))


# --- analysis --------------------------------------------------------------------------------------------------------

def losses(scores_by_cell, item_set, fmt, ids):
    intact = scores_by_cell[(item_set, fmt, "intact")].reindex(ids)
    return {lesion: (intact - scores_by_cell[(item_set, fmt, lesion)].reindex(ids)).to_numpy() for lesion in LESIONS}


def analyse(scores, meta, ledger, original=None, notes=(), n_boot=N_BOOT, seed=SEED, family_items=None):
    """The report (a dict); scores: the run's scores table, meta: its meta.json."""
    original = dict(original or {})
    family_prompts = original.pop("_family_prompts", None)
    family_batches = original.pop("_family_batches", None)
    cells = target_scores(scores)
    res = Resampler(n_boot, seed)
    formats = {s: sorted({f for (s2, f, _) in cells if s2 == s}) for s in ("battery", "family")}
    ids = {}
    for s in formats:
        sets = [set(cells[(s, f, st)].index) for f in formats[s] for st in ("intact", *LESIONS)]
        if any(x != sets[0] for x in sets):
            raise ValueError(f"{s}: cells cover different concepts")
        ids[s] = concept_order(s, sets[0], family_items)
    report = dict(intact_naming={}, naming_loss={}, existing_recomputed={}, format_effect_within_run={},
                  format_effect_cross_run={}, reproduction={})
    loss = {}
    for s, fmts in formats.items():
        for f in fmts:
            report["intact_naming"].setdefault(s, {})[f] = res.stat(s, cells[(s, f, "intact")].reindex(ids[s]))
            loss[(s, f)] = losses(cells, s, f, ids[s])
            report["naming_loss"].setdefault(s, {})[f] = {k: res.stat(s, v) for k, v in loss[(s, f)].items()}
        if OLD[s] in fmts and NEW[s] in fmts:
            new, old = loss[(s, NEW[s])], loss[(s, OLD[s])]
            effect = {k: res.stat(s, new[k] - old[k]) for k in LESIONS}
            effect["intact_target_logprob"] = res.stat(s, (cells[(s, NEW[s], "intact")].reindex(ids[s])
                                                           - cells[(s, OLD[s], "intact")].reindex(ids[s])).to_numpy())
            report["format_effect_within_run"][s] = effect
        if all((s, OLD[s], st) in original for st in ("intact", *LESIONS)):
            if set(original[(s, OLD[s], "intact")].index) != set(ids[s]):
                raise ValueError(f"{s}: the original run covers other concepts")
            old = losses(original, s, OLD[s], ids[s])
            report["existing_recomputed"][s] = {OLD[s]: {k: res.stat(s, v) for k, v in old.items()}}
            report["format_effect_cross_run"][s] = {k: res.stat(s, loss[(s, NEW[s])][k] - old[k]) for k in LESIONS}
            if OLD[s] in fmts:
                rep = {}
                for st in ("intact", *LESIONS):
                    a = cells[(s, OLD[s], st)].reindex(ids[s]).to_numpy()
                    b = original[(s, OLD[s], st)].reindex(ids[s]).to_numpy()
                    rep[st] = dict(target_logprob_mean_difference=float((a - b).mean()),
                                   target_logprob_mean_abs_difference=float(np.abs(a - b).mean()),
                                   target_logprob_r=float(np.corrcoef(a, b)[0, 1]), n=len(a), n_unit="概念")
                    if st != "intact":
                        new_loss, old_loss = loss[(s, OLD[s])][st], old[st]
                        rep[st].update(naming_loss_rerun=float(new_loss.mean()), naming_loss_original=float(old_loss.mean()),
                                       naming_loss_difference=float(new_loss.mean() - old_loss.mean()),
                                       naming_loss_r=float(np.corrcoef(new_loss, old_loss)[0, 1]),
                                       original_in_rerun_interval=bool(
                                           report["naming_loss"][s][OLD[s]][st]["ci_low"] <= float(old_loss.mean())
                                           <= report["naming_loss"][s][OLD[s]][st]["ci_high"]))
                report["reproduction"][s] = {OLD[s]: rep}
    # batch-layout noise of this run: per concept, the SD of the target log-probability over the layouts
    t = scores[scores.role == "target"]
    sd = t.groupby(["item_set", "format", "state", "item_id"]).logprob.std(ddof=0)
    report["layout_noise"] = {s: {f: dict(mean_sd_over_layouts=float(g.mean()), n=int(g.size), n_unit="概念×状态")
                                  for f, g in sd.loc[s].groupby(level=0)} for s in formats}

    value = {k: ledger[i]["value"] for k, i in EXISTING.items()}
    new_b, new_f = report["naming_loss"]["battery"]["no_demos"]["SC4"], report["naming_loss"]["family"]["demos"]["SC4"]
    report["reading"] = reading(new_b, new_f, value[("family", "SC4")], value[("battery", "SC4")])
    if "demos" in formats["battery"] and "no_demos" in formats["family"]:
        within = reading(new_b, new_f, report["naming_loss"]["family"]["no_demos"]["SC4"]["estimate"],
                         report["naming_loss"]["battery"]["demos"]["SC4"]["estimate"])
        report["reading_within_run"] = dict(within, note="the same rules with this run's existing-format cells in "
                                                         "place of the ledger values (secondary, not the plan's rule)")
    checks = {}
    for (s, lesion), i in EXISTING.items():
        rec = report["existing_recomputed"].get(s, {}).get(OLD[s], {}).get(lesion)
        if rec is not None:
            # the ledger keeps the source's precision (0.0289 is rounded to 4 decimals in experiment 3's report)
            checks[i] = dict(ledger=ledger[i]["value"], recomputed=rec["estimate"],
                             agrees=abs(rec["estimate"] - ledger[i]["value"]) < 5e-4)
    prompt_check = None
    if family_prompts is not None:     # the default family prompts are the ones the original run scored
        mine = rdf.cell_prompts("family", "no_demos", rdf.family_items() if family_items is None else family_items)
        prompt_check = all(family_prompts.get(p.item_id) == p.text for p in mine) and len(mine) == len(family_prompts)
    report["meta"] = dict(
        description="demonstration-format check, exploratory, outside the confirmatory family of Table S6 "
                    "(docs/plans/demo-format-check.md, approved by the user 2026-10-28)",
        run=dict(git_commit=meta.get("git_commit"), clean_tree=meta.get("clean_tree"), timestamp=meta.get("timestamp"),
                 model=meta.get("model"), revision=meta.get("revision"), dtype=meta.get("dtype"),
                 moe_forward=meta.get("moe_forward"), tiny=meta.get("tiny"), sets=meta.get("sets"),
                 batches=meta.get("batches"), also_existing=meta.get("also_existing"), gpu=meta.get("gpu")),
        n_boot=n_boot, seed=seed, interval="95% percentile, concepts resampled (families in family order)",
        score="target summed log-probability, averaged over the batch layouts; naming loss = intact minus lesioned",
        format_effect="new minus existing format, paired by concept: battery without minus with demos, families "
                      "with minus without demos",
        existing_ledger={f"{s}/{lesion}": dict(id=i, **ledger[i]) for (s, lesion), i in EXISTING.items()},
        ledger_agreement=checks, original_family_prompts_identical=prompt_check,
        original_family_naming_batches=family_batches, original_battery_naming_batches=list(rdf.BATCHES),
        notes=list(notes))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "outputs/e3_errors" / MODEL / "demo_format")
    ap.add_argument("--original-battery", type=Path, default=ORIGINAL_BATTERY)
    ap.add_argument("--original-family", type=Path, default=ORIGINAL_FAMILY)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    meta = json.loads((args.dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("status") != "done":
        raise SystemExit(f"{args.dir} is not a finished run")
    scores = pd.read_parquet(args.dir / "scores.parquet")
    family_items = None
    if meta.get("tiny"):
        original, notes, inputs = {}, ["tiny run: no comparison with the original outputs"], {}
        family_items = rdf.family_items(meta["n_concepts"]["family"])
    else:
        original, notes, inputs = original_scores(args.original_battery, args.original_family)
    report = analyse(scores, meta, ledger_rows(args.ledger), original, notes, args.n_boot, SEED, family_items)
    for path in (args.dir / "scores.parquet", args.dir / "meta.json", args.ledger):
        inputs[relative(path)] = runinfo.sha256(path)
    report["meta"]["inputs"] = inputs
    report = {"meta": report.pop("meta"), **report}
    date = str(meta.get("timestamp", ""))[:10].replace("-", "") or "undated"
    out = args.out or ROOT / f"reports/explore/demo-format-{date}/demo-format.json"
    if meta.get("tiny") and args.out is None:
        out = ROOT / "outputs/e3_errors_tiny" / MODEL / "demo-format-tiny.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for s, fmts in report["naming_loss"].items():
        for f, v in fmts.items():
            print(f"{s:8s} {f:9s} " + "  ".join(f"{k} {x['estimate']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]"
                                                for k, x in v.items()))
    for s, v in report["format_effect_within_run"].items():
        print(f"within-run format effect {s}: " + "  ".join(f"{k} {x['estimate']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]"
                                                         for k, x in v.items()))
    r = report["reading"]
    print(f"reading rule: {r['rule']}  {r['checks']}\n{r['text']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
