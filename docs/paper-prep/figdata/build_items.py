"""Item-level data for the redesigned Figures 2 to 5, extracted on CPU from completed runs. No model inference.

    python -X utf8 docs/paper-prep/figdata/build_items.py [--check]

Follows build.py: every input file is hashed (SHA-256) into items-manifest.json, every extracted table is checked
against the ledger summary it underlies before anything is written (a failed check stops the script), and --check
rebuilds in memory and compares with the files on disk. The per-item values are the same per-item quantities the
original analyses averaged; this script recomputes a ledger summary only to check it. Nothing here becomes a new
summary number in the paper: the figures read their estimates and intervals from results_ledger.csv by ID.

Outputs (docs/paper-prep/figdata/; renamed 2026-10-28 so that each name matches its panel):
  items_fig2a_screening.csv    7,680 components: gradient (attribution-patching) scores on the frequency and control
                               markers, the 32-candidate shortlists of each direction, WR8 and SC4 membership
  items_fig2b_random_sets.csv  the 200 matched random sets of each marker with their frequency or control effect
  items_fig2d_wr8_concepts.csv the 240 held-out naming concepts: Zipf frequency, rare flag, WR8 and SC4 naming loss
  items_fig3d_graded.csv       the 160 battery concepts: WR8 and SC4 naming and six-option loss at 75, 50, 25 and 0%
                               of output retained
  items_fig4a_patient_errors.csv  per-patient error shares (coordinate, superordinate, associate) of SD and SA
  items_fig4d_patient_cueing.csv  per-patient progressive cueing (8 SD, 6 SA; counts out of 30, rescued shares)
  items_fig5b_families.csv     the 120 final families: naming and definition-choice loss under WR8, SC4, the broad
                               middle-layer lesion at 10% retained and the two learned lesions, with the covariates of
                               the partial correlation
  items_fig5c_badges.csv       where each Fig. 5c lesion acts: gated components and removed output per layer
  items-manifest.json          input hashes, checks, output hashes
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))
import analyze_optogenetics as ao  # noqa: E402  (per-concept losses exactly as the frozen analysis computes them)

LEDGER = ROOT / "docs/paper-prep/results_ledger.csv"
MEMBERS = "docs/paper-prep/component-members.csv"
SERVER = "outputs/server"
MODEL = "qwen3-30b-a3b-instruct"
ATTR_FREQ = f"{SERVER}/e3_knowledge/{MODEL}/screen/attribution.csv"
SCREEN_FREQ = f"{SERVER}/e3_knowledge/{MODEL}/screen/screen.csv"
ATTR_CTRL = f"{SERVER}/outputs/e1_screen/{MODEL}/attribution/attribution.csv"
SCREEN_CTRL = f"{SERVER}/outputs/e1_screen/{MODEL}/screen/screen.csv"
RANDOM_CTRL = f"{SERVER}/e2_confirm/{MODEL}/control/per_set_control_effect.csv"
NAMING = f"{SERVER}/e2_confirm/{MODEL}/dissociation/naming_scores.parquet"
KNOWLEDGE_ITEMS = "data/materials/m3_knowledge/items.csv"
OPTO = f"outputs/e3_errors/{MODEL}/optogenetics_confirm"
OPTO_REPORT = "reports/confirm/optogenetics/result.json"
M3_ITEMS = "data/materials/m3/final_items.csv"
PATIENTS = "data/literature/jefferies2006_individual.csv"
TG = "outputs/title_gap_execution_20261019"
FINE_ITEMS = f"{TG}/core_readout_20261020/fine-target-items.csv"
BROAD_ITEMS = f"{TG}/remaining_core_20261020/broad/target-items.csv"
ZIPF_MEDIAN = 2.99965          # the dissociation run's split (outputs/server/e2_confirm/.../dissociation/meta.json params)
SHORTLIST = 32                 # candidates per direction ablated one by one (Methods 2.3)
TOL = 5.000001e-4              # ledger rows of the optogenetics report are rounded to four decimals
INPUTS: dict[str, str] = {}
CHECKS: dict[str, object] = {}


def tracked(relative):
    path = ROOT / relative
    INPUTS[str(relative).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def ledger():
    with open(tracked("docs/paper-prep/results_ledger.csv"), encoding="utf-8-sig", newline="") as f:
        return {r["编号"]: r for r in csv.DictReader(f)}


L = None


def led(ledger_id, field="数值"):
    return float(L[ledger_id][field])


def match(label, actual, ledger_id, atol=1e-9, field="数值"):
    expected = led(ledger_id, field)
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise SystemExit(f"CHECK FAILED {label}: {actual!r} differs from ledger {ledger_id} {expected!r}")
    CHECKS[label] = dict(ledger_id=ledger_id, ledger=expected, recomputed=float(actual), status="pass")


def require(label, condition, detail):
    if not condition:
        raise SystemExit(f"CHECK FAILED {label}: {detail}")
    CHECKS[label] = dict(detail=detail, status="pass")


def members():
    m = pd.read_csv(tracked(MEMBERS))
    return {code: list(m.component[m.paper_code == code]) for code in ("WR8", "SC4")}


# ------------------------------------------------------------------------------------------------- Figure 2a: screening
def screening():
    """Both gradient screens, one row per component. Scores are copied, not rescaled; the figure scales them."""
    fq = pd.read_csv(tracked(ATTR_FREQ))
    ct = pd.read_csv(tracked(ATTR_CTRL))
    short_f = set(pd.read_csv(tracked(SCREEN_FREQ)).component)
    short_ctrl_union = set(pd.read_csv(tracked(SCREEN_CTRL)).component)
    sets = members()
    require("2a/components", len(fq) == len(ct) == 7680 and set(fq.component) == set(ct.component),
            "7,680 components in both screens (Methods 2.1: 6,144 experts and 1,536 heads)")
    kinds = fq.kind.value_counts().to_dict()
    require("2a/kinds", kinds == {"expert": 6144, "head": 1536}, f"kinds {kinds}")
    require("2a/layers", fq.layer.min() == 0 and fq.layer.max() == 47, "layers numbered 0 to 47")
    require("2a/frequency_shortlist", len(short_f) == SHORTLIST and short_f == set(fq.nlargest(SHORTLIST, "knowledge_effect").component),
            "the 32 frequency candidates are the 32 largest gradient scores of the frequency screen")
    top_c = set(ct.nlargest(SHORTLIST, "control_effect").component)
    require("2a/control_shortlist", top_c <= short_ctrl_union and len(short_ctrl_union) == 61,
            "the 32 control candidates are the 32 largest control scores; the control screen ablated 61 components, "
            "the union of these 32 and the 32 largest scores of an earlier frequency direction (common.shortlist)")
    require("2a/members", set(sets["WR8"]) <= short_f and set(sets["SC4"]) <= top_c,
            "every WR8 member is a frequency candidate and every SC4 member a control candidate")
    t = fq[["component", "kind", "layer", "index"]].merge(
        fq[["component", "knowledge_effect"]].rename(columns={"knowledge_effect": "frequency_score"}), on="component").merge(
        ct[["component", "control_effect"]].rename(columns={"control_effect": "control_score"}), on="component")
    t["frequency_candidate"] = t.component.isin(short_f).astype(int)
    t["control_candidate"] = t.component.isin(top_c).astype(int)
    t["set"] = np.where(t.component.isin(sets["WR8"]), "WR8", np.where(t.component.isin(sets["SC4"]), "SC4", ""))
    t["source_frequency"] = ATTR_FREQ
    t["source_control"] = ATTR_CTRL
    return t.sort_values(["layer", "kind", "index"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------ Figure 2b, 2c: random sets, concepts
def naming_frames():
    scores = pd.read_parquet(tracked(NAMING))
    items = pd.read_csv(tracked(KNOWLEDGE_ITEMS))[["item_id", "zipf"]]
    return scores, items


def freq_effect(intact, lesioned, zipf):
    loss = intact.reindex(lesioned.index) - lesioned
    rare = zipf.reindex(loss.index) < ZIPF_MEDIAN
    return float(loss[rare].mean() - loss[~rare].mean())


def random_sets(scores, items):
    zipf = items.set_index("item_id").zipf
    single = scores[scores.scoring == "single"]
    by = {s: g.set_index("item_id").logprob for s, g in single.groupby("set")}
    intact = by["intact"]
    rows = [dict(marker="frequency", set=s, effect=freq_effect(intact, v, zipf), unit="nat",
                 scoring="single batch layout (batch 16)", source=NAMING)
            for s, v in sorted(by.items()) if s.startswith("random_")]
    f = np.array([r["effect"] for r in rows])
    require("2b/frequency_n", len(f) == 200, "200 matched random sets for the frequency marker")
    match("2b/frequency_median", np.median(f), "knowledge-f8dda0adcc71")
    match("2b/frequency_max", f.max(), "knowledge-8c54fa5ee735")
    match("2b/frequency_at_least_target", int((f >= led("knowledge-f3ce21647c2f")).sum()), "knowledge-f733221fee68")
    match("2b/WR8_single_layout", freq_effect(intact, by["main"], zipf), "knowledge-516697debdde")
    c = pd.read_csv(tracked(RANDOM_CTRL))
    rc = c[c.set.str.startswith("random")]
    require("2b/control_n", len(rc) == 200, "200 matched random sets for the control marker")
    match("2b/control_median", float(np.median(rc.control_effect)), "control-f8dda0adcc71")
    match("2b/control_max", float(rc.control_effect.max()), "control-8c54fa5ee735")
    match("2b/control_at_least_target", int((rc.control_effect >= led("control-b9d87e63d005")).sum()), "control-f733221fee68")
    match("2b/SC4_control_effect", float(c.control_effect[c.set == "control"].iloc[0]), "control-b9d87e63d005")
    rows += [dict(marker="control", set=r.set, effect=float(r.control_effect), unit="logit",
                  scoring="control test, 63 held-out feature-matching items", source=RANDOM_CTRL)
             for r in rc.itertuples()]
    return rows


def concepts(scores, items):
    """Per-concept naming loss (intact minus lesioned log probability of the target), averaged over the three batch
    layouts as in the main estimate; set 'main' is WR8 and set 'control' is SC4."""
    avg = scores[scores.scoring == "averaged"]
    by = {s: g.set_index("item_id") for s, g in avg.groupby("set")}
    t = by["intact"][["target", "n_tokens"]].join(items.set_index("item_id").zipf)
    t["rare"] = (t.zipf < ZIPF_MEDIAN).astype(int)
    t["loss_WR8"] = by["intact"].logprob - by["main"].logprob.reindex(t.index)
    t["loss_SC4"] = by["intact"].logprob - by["control"].logprob.reindex(t.index)
    require("2d/n", len(t) == 240 and t.rare.sum() == 120 and t[["loss_WR8", "loss_SC4"]].notna().all().all(),
            "240 held-out concepts, 120 rare (Zipf below 2.99965), no missing loss")
    r = t.rare.astype(bool)
    match("2d/WR8_frequency_effect", t.loss_WR8[r].mean() - t.loss_WR8[~r].mean(), "knowledge-f3ce21647c2f")
    match("2d/WR8_naming_loss", t.loss_WR8.mean(), "knowledge-a6c62f0364c4")
    match("2d/SC4_frequency_effect", t.loss_SC4[r].mean() - t.loss_SC4[~r].mean(), "knowledge-e06843c7270c")
    match("2d/SC4_naming_loss", t.loss_SC4.mean(), "knowledge-91368a9460f1")
    t = t.reset_index().rename(columns={"index": "item_id"})
    t["scoring"] = "averaged over batch layouts 16, 11, 7"
    t["source"] = NAMING
    return t


# ----------------------------------------------------------------------------------------- Figure 3d: graded lesions
GRADED = {  # (set, retained %) -> (state in the run, ledger ID of the naming mean, ledger ID of the six-option mean)
    ("WR8", 75): ("K8@0.75", "optogenetics-838983539d83", "optogenetics-d7ef1de56138"),
    ("WR8", 50): ("K8@0.5", "optogenetics-d410a5695072", "optogenetics-8d8431b87635"),
    ("WR8", 25): ("K8@0.25", "optogenetics-0d05bc447945", "optogenetics-5ce4f5318f12"),
    ("WR8", 0): ("K8", "optogenetics-8c912a82f7fd", "optogenetics-6f16be6dd1b8"),
    ("SC4", 75): ("C4@0.75", "optogenetics-8f1416b6746b", "optogenetics-8293b3e7a7cc"),
    ("SC4", 50): ("C4@0.5", "optogenetics-32cce154f741", "optogenetics-1a34feaced56"),
    ("SC4", 25): ("C4@0.25", "optogenetics-6fff2487b4d3", "optogenetics-1a837c2b2d93"),
    ("SC4", 0): ("C4", "optogenetics-5d5add9cfe69", "optogenetics-a3782f19a542"),
}


def graded():
    report = json.loads(tracked(OPTO_REPORT).read_text(encoding="utf-8"))
    for name, digest in report["source_tables_sha256"].items():
        if name in ("candidates", "labels"):
            got = hashlib.sha256(tracked(f"{OPTO}/{name}.parquet").read_bytes()).hexdigest()
            require(f"3d/source_{name}", got == digest, f"{name}.parquet is the table the frozen report analysed")
    items = pd.read_csv(tracked(M3_ITEMS))
    ids = list(items.item_id[items.role == "test"])
    require("3d/n", len(ids) == report["n_concepts"] == 160, "160 held-out battery concepts")
    losses, _, _ = ao.load_losses(ROOT / OPTO, ids)
    rows = []
    for (code, kept), (state, id_naming, id_six) in GRADED.items():
        x = losses[state].reindex(ids)
        match(f"3d/{code}@{kept}/naming", x.naming.mean(), id_naming, TOL)
        match(f"3d/{code}@{kept}/six_choice", x.six_choice.mean(), id_six, TOL)
        rows += [dict(item_id=i, set=code, retained_percent=kept, state=state, naming_loss=float(n),
                      six_option_loss=float(s), naming_unit="nat", six_option_unit="logit",
                      source=f"{OPTO}/candidates.parquet; {OPTO}/labels.parquet")
                 for i, n, s in zip(ids, x.naming, x.six_choice)]
    return rows


# ----------------------------------------------------------------------------------------- Figure 4a: patient errors
PATIENT_IDS = {("SD", "coordinate"): "title_gap_patient_markers-22b6a4a99d4b",
               ("SD", "superordinate"): "title_gap_patient_markers-444a8f678f57",
               ("SD", "associative"): "title_gap_patient_markers-c6e192f47af1",
               ("SA", "coordinate"): "title_gap_patient_markers-7cc27b874df4",
               ("SA", "superordinate"): "title_gap_patient_markers-7b4e973bdb7e",
               ("SA", "associative"): "title_gap_patient_markers-80ca3121e20a"}


def patient_errors():
    t = pd.read_csv(tracked(PATIENTS), keep_default_na=False)
    t = t[(t.task == "naming_error_analysis") & t.measure.isin(["coordinate", "superordinate", "associative"])].copy()
    require("4a/denominator", set(t.denominator) == {"single_word_semantic_errors"},
            "shares of single-word semantic errors (Jefferies & Lambon Ralph 2006, Table 5)")
    t["value"] = t.value.astype(float)
    for (group, measure), lid in PATIENT_IDS.items():
        v = t.value[(t.group == group) & (t.measure == measure)]
        require(f"4a/{group}/{measure}/n", len(v) == int(L[lid]["n"]), f"{len(v)} patients")
        match(f"4a/{group}/{measure}", v.mean(), lid)
    return [dict(patient_id=r.patient_id, group=r.group, error_type=r.measure.replace("associative", "associate"),
                 share=float(r.value), printed_value=r.printed_value, source=PATIENTS + ", " + r.source)
            for r in t.sort_values(["group", "measure", "patient_id"]).itertuples()]


# ------------------------------------------------------------------------------------- Figure 4d: patient cueing
PATIENT_CUES = "data/literature/jefferies2008_cueing_individual.csv"
CUE_GROUP_MEANS = "data/literature/jefferies2008_cueing.csv"
CUE_IDS = {("SD", "two"): "title_gap_patient_markers-e6829f2afed2", ("SD", "final"): "title_gap_patient_markers-26778aefa564",
           ("SA", "two"): "title_gap_patient_markers-fdebb8564dd8", ("SA", "final"): "title_gap_patient_markers-2b1342db3549"}
MEAN_ROWS = {"none": "uncued naming", "two": "naming with a two-phoneme cue", "final": "naming at the final cue"}


def patient_cueing():
    """Per-patient progressive cueing (Jefferies et al., 2008, Fig. 2, transcribed to counts out of 30 by
    tools/measure_jefferies2008_fig2.py). Each patient's rescued share is (cued - uncued) / (30 - uncued). The group
    values of the ledger are pooled ratios (all initially failed items together), computed from the group means of
    jefferies2008_cueing.csv; the checks below reproduce those group means exactly and the ledger ratios within the
    three-decimal rounding of the means."""
    t = pd.read_csv(tracked(PATIENT_CUES))
    means = pd.read_csv(tracked(CUE_GROUP_MEANS), keep_default_na=False)
    w = t.pivot(index=["patient", "group"], columns="condition", values="count_correct").reset_index()
    n_items = t.groupby("patient").n_items.unique()
    require("4d/items", all(len(v) == 1 and v[0] == 30 for v in n_items), "every patient named 30 pictures per condition")
    require("4d/patients", (w.group == "SD").sum() == 8 and (w.group == "SA").sum() == 6, "8 SD and 6 SA patients")
    require("4d/monotone", ((w.none <= w.two) & (w.two <= w["final"])).all(), "correct counts never fall as cues grow")
    rows = []
    for group, g in w.groupby("group"):
        for cond, label in MEAN_ROWS.items():
            m = means[(means.group == group) & (means.measure == label)]
            require(f"4d/{group}/{cond}/mean", len(m) == 1 and abs(g[cond].sum() / (30 * len(g)) - float(m.value.iloc[0])) < 5.0001e-4,
                    f"mean proportion {g[cond].sum() / (30 * len(g)):.4f} rounds to the group mean {m.value.iloc[0]}")
        for cond in ("two", "final"):
            pooled = (g[cond].sum() - g.none.sum()) / (30 * len(g) - g.none.sum())
            match(f"4d/{group}/{cond}/pooled_ratio", pooled, CUE_IDS[(group, cond)], atol=1.5e-3)
    for r in w.sort_values(["group", "patient"]).itertuples():
        rows.append(dict(patient=r.patient, group=r.group, uncued_correct=int(r.none), two_correct=int(r.two),
                         final_correct=int(r.final), n_items=30, initial_failures=30 - int(r.none),
                         rescued_two=(r.two - r.none) / (30 - r.none), rescued_final=(r.final - r.none) / (30 - r.none),
                         source=PATIENT_CUES + ", Jefferies et al. 2008, Fig. 2"))
    return rows


# ----------------------------------------------------------------------------------------- Figure 5b: final families
FAMILIES = {  # label -> (frame, state, naming ID, definition ID, partial r ID)
    "WR8": ("fine", "WR8_a0", "title_gap_fine_remaining_core-e7c009e99e28",
            "title_gap_fine_remaining_core-72bb93644bd4", "title_gap_fine_core_readout-648ed349c867"),
    "SC4": ("fine", "SC4_a0", "title_gap_fine_remaining_core-9a0a976b077d",
            "title_gap_fine_remaining_core-3e84cf723965", "title_gap_fine_core_readout-668a9fba898a"),
    "broad_10": ("broad", "broad_middle_f1_a0.1_s0", "title_gap_fine_remaining_core-5a1d470c13fe",
                 "title_gap_fine_remaining_core-f34bb8f3d165", "title_gap_fine_remaining_core-50641c69dc33"),
    "learned_comprehension": ("fine", "comprehension_s2_learned_soft", "title_gap_fine_core_readout-f6f815b1577a",
                              "title_gap_fine_core_readout-bc43f20250b2", "title_gap_fine_core_readout-adcf6e72514c"),
    "learned_hub": ("fine", "hub_s2_learned_soft", "title_gap_fine_core_readout-2b4e87553076",
                    "title_gap_fine_core_readout-36b6a70af49b", "title_gap_fine_core_readout-42b468ab42e8"),
}


def partial_r(frame, covariates):
    """The frozen analysis's partial correlation (title_gap_analysis.partial_correlation, fine_final_source)."""
    x = np.column_stack([np.ones(len(frame)), frame[covariates].to_numpy(float)])
    y = frame[["naming", "definition"]].to_numpy(float)
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    return float(np.corrcoef(residual.T)[0, 1])


def families():
    frames = {"fine": pd.read_csv(tracked(FINE_ITEMS)), "broad": pd.read_csv(tracked(BROAD_ITEMS))}
    tracked(f"{TG}/fine_final_source/src/lesion/title_gap_analysis.py")
    rows = []
    for label, (which, state, id_n, id_d, id_r) in FAMILIES.items():
        g = frames[which][frames[which].state == state]
        cov = ["frequency", "n_tokens", "intact_naming", "intact_definition"]
        if g.semd.notna().all():
            cov.append("semd")
        require(f"5b/{label}/n", len(g) == 120 and g.family.nunique() == 120 and g[["naming", "definition", *cov]].notna().all().all(),
                "120 final families, one row each, complete")
        match(f"5b/{label}/naming", g.naming.mean(), id_n)
        match(f"5b/{label}/definition", g.definition.mean(), id_d)
        match(f"5b/{label}/partial_r", partial_r(g, cov), id_r)
        src = FINE_ITEMS if which == "fine" else BROAD_ITEMS
        rows += [dict(lesion=label, state=state, item_id=r.item_id, family=r.family, target=r.target,
                      naming_loss=float(r.naming), definition_loss=float(r.definition), frequency=float(r.frequency),
                      n_tokens=int(r.n_tokens), intact_naming=float(r.intact_naming),
                      intact_definition=float(r.intact_definition), semd=float(r.semd), readout="target",
                      naming_unit="nat", definition_unit="logit", source=src)
                 for r in g.sort_values("item_id").itertuples()]
    # The text's .17 for the learned lesion aimed at comprehension is the accepted_vs_coordinate readout's partial r
    # (title_gap_fine_core_readout-0ed64d76f6e2); the figure uses the target readout for every lesion, whose naming
    # and definition means are the ones the text reports. Record both so the difference is visible.
    CHECKS["5b/readout_note"] = dict(status="note", target_readout_partial_r=led("title_gap_fine_core_readout-adcf6e72514c"),
                                     accepted_vs_coordinate_partial_r=led("title_gap_fine_core_readout-0ed64d76f6e2"))
    return rows


# ----------------------------------------------------------------------------------------- Figure 5c: lesion badges
PLAN = f"{TG}/fine_final_core/plan.json"
FINE_POINTS = f"{TG}/core_readout_20261020/fine-points.json"
BROAD_POINTS = f"{TG}/remaining_core_20261020/broad/points.json"
BADGE_STATES = ("WR8_a0", "SC4_a0", "broad_middle_f1_a0.1_s0", "comprehension_s2_learned_soft", "hub_s2_learned_soft")


def badges():
    """Where each lesion of Fig. 5c acts: per layer, the number of gated components and the removed share of their
    output (off mass, sum of 1 - gate), from the frozen run plan. Totals are checked against the readouts' records."""
    import collections
    plan = json.loads(tracked(PLAN).read_text(encoding="utf-8"))
    fine = json.loads(tracked(FINE_POINTS).read_text(encoding="utf-8"))["meta"]
    broad = json.loads(tracked(BROAD_POINTS).read_text(encoding="utf-8"))["meta"]
    require("5c/fingerprint", fine["source_fingerprint"] == broad["source_fingerprint"] == plan["fingerprint"],
            "the plan is the one both readouts analysed")
    records = {**fine["interventions"], **broad["interventions"]}
    states = {s["name"]: s for s in plan["states"]}
    sets = members()
    rows = []
    for name in BADGE_STATES:
        gates = states[name]["gates"]
        off, n = collections.defaultdict(float), collections.Counter()
        for comp, g in gates.items():
            layer = int(comp.split(":")[1])
            off[layer] += 1 - float(g)
            n[layer] += 1
        rec = records[name]
        require(f"5c/{name}/gates", len(gates) == rec["n_gates"] and abs(sum(off.values()) - rec["off_mass"]) < 1e-6,
                f"{len(gates)} gates, off mass {sum(off.values()):.4f} as recorded")
        if name in ("WR8_a0", "SC4_a0"):
            code = name.split("_")[0]
            require(f"5c/{name}/members", sorted(gates) == sorted(sets[code]), "gates are the set's members")
        rows += [dict(state=name, layer=layer, n_components=n[layer], off_mass=off[layer], source=PLAN)
                 for layer in sorted(off)]
    return rows


# ------------------------------------------------------------------------------------------------------- assemble
def csv_text(rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    buf = io.StringIO(newline="")
    frame.to_csv(buf, index=False, lineterminator="\n", float_format="%.10g")
    return buf.getvalue()


def build():
    global L
    INPUTS.clear()
    CHECKS.clear()
    L = ledger()
    scores, items = naming_frames()
    data = {
        "items_fig2a_screening.csv": screening(),
        "items_fig2b_random_sets.csv": random_sets(scores, items),
        "items_fig2d_wr8_concepts.csv": concepts(scores, items),
        "items_fig3d_graded.csv": graded(),
        "items_fig4a_patient_errors.csv": patient_errors(),
        "items_fig4d_patient_cueing.csv": patient_cueing(),
        "items_fig5b_families.csv": families(),
        "items_fig5c_badges.csv": badges(),
    }
    for source in ("docs/paper-prep/figdata/build_items.py", "experiments/e3_errors/analyze_optogenetics.py",
                   "experiments/e3_errors/analyze_battery.py", "experiments/e2_confirm/run_dissociation_test.py",
                   "src/lesion/naming.py"):
        tracked(source)
    outputs = {name: csv_text(rows) for name, rows in data.items()}
    manifest = dict(schema_version=1,
                    operation="CPU extraction of item-level values from completed runs for Figures 2 to 4; "
                              "no model inference, no new summary statistics",
                    rule="each file reproduces its ledger summary (checks) before it is written",
                    software=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                    inputs=dict(sorted(INPUTS.items())), checks=CHECKS,
                    outputs={name: dict(rows=int(len(pd.read_csv(io.StringIO(text)))),
                                        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
                             for name, text in outputs.items()})
    outputs["items-manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build()  # every ledger check runs before any file is written
    if args.check:
        different = [name for name, content in outputs.items()
                     if not (OUT / name).exists() or (OUT / name).read_bytes() != content.encode("utf-8")]
        if different:
            raise SystemExit("NOT REPRODUCIBLE: " + ", ".join(different))
        print(f"PASS: {len(outputs) - 1} item files and the manifest match their sources ({len(CHECKS)} checks)")
    else:
        for name, content in outputs.items():
            (OUT / name).write_bytes(content.encode("utf-8"))
        print(f"Wrote {len(outputs) - 1} item files and items-manifest.json; {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
