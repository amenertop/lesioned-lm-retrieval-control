"""Prepare paper figure data from completed runs on CPU. No model inference.

Existing report intervals are copied, not replaced. Additional percentile
intervals are descriptive. --check rebuilds in memory and never writes outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))
import analyze_battery  # noqa: E402
from lesion import damage, degradation as dg, e3_analysis as ea, knowledge_items as ki, m3_scoring as ms  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
E3 = f"outputs/e3_errors/{MODEL}"
CONF = f"{E3}/confirm"
E2 = f"outputs/server/e2_confirm/{MODEL}"
REPORT = "reports/confirm/experiment3/results.json"
KNOWLEDGE = "reports/confirm/knowledge/results.json"
CONTROL = "reports/confirm/control/results.json"
G1 = "reports/g1_qwen/g1_result.json"
N_BOOT, SEED = 10_000, 0
LOGP, LOGODDS, LOGIT = "log probability (nat)", "restricted log odds (nat)", "logit margin"
K8 = {"expert:46:61", "expert:46:101", "head:37:24", "expert:45:32",
      "expert:45:48", "expert:43:70", "expert:42:115", "expert:45:93"}
INPUTS: dict[str, str] = {}
CHECKS: dict[str, object] = {}


def tracked(relative):
    path = ROOT / relative
    INPUTS[str(relative).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(relative):
    return json.loads(tracked(relative).read_text(encoding="utf-8"))


def read_csv(relative, **kwargs):
    return pd.read_csv(tracked(relative), keep_default_na=False, **kwargs)


def read_parquet(relative, **kwargs):
    return pd.read_parquet(tracked(relative), **kwargs)


def ref(path, pointer=""):
    return path + "#" + pointer


def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite values in bootstrap samples")
    return np.quantile(values, [0.025, 0.975]).tolist()


def match(label, actual, expected, atol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, err_msg=label)
    CHECKS[label] = "pass"


def report_interval(node):
    return node["ci"] if "ci" in node else [node["ci_low"], node["ci_high"]]


def check_summary(label, values, counts, node):
    match(label + ":mean", np.mean(values), node["estimate"], 5.000001e-5)
    match(label + ":ci", interval(counts @ values / len(values)), report_interval(node), 5.000001e-5)


def paired_loss(pc, ids, state, score):
    result = (pc["intact"][score] - pc[state][score]).reindex(ids).to_numpy()
    if not np.isfinite(result).all():
        raise ValueError(f"incomplete paired data: {state}/{score}")
    return result


def dissociation(knowledge):
    rows = []
    for state, key in (("K8", "knowledge"), ("C4", "control")):
        for test, task, stat, unit, n, related in (
            ("frequency", "naming", "knowledge_effect", LOGP, 240, "H2_delta_F"),
            ("control", "m1", "control_effect", LOGIT, 63, "H2_delta_C"),
        ):
            pointer = f"/sets/main/{task}/{key}.{stat}"
            node = knowledge["sets"]["main"][task][f"{key}.{stat}"]
            rows.append(dict(set=state, test=test, effect=node["estimate"], ci_low=node["ci_low"],
                             ci_high=node["ci_high"], tier="报", source=ref(KNOWLEDGE, pointer),
                             unit=unit, n=n, related_test=related, ci_source="original_report",
                             note="Individual set estimate. H2 tests the difference between sets."))
    return rows


def formats_and_hints(pc, ids, hint_ids, results):
    rng = np.random.default_rng(SEED)
    counts = ea.resample_counts(len(ids), N_BOOT, rng)
    hint_counts = ea.resample_counts(len(hint_ids), N_BOOT, rng)
    format_specs = (
        ("name_free", "name_target_logprob", "naming_loss", LOGP),
        ("options_word", "options_word_target_logprob", "word_shown_loss", LOGP),
        ("options_letter", "options_letter_margin", "six_choice_loss", LOGIT),
        ("word_to_definition", "word_to_definition_margin", "comprehension_loss", LOGIT),
        ("copy", "copy_target_logprob", "copy_loss", LOGP),
    )
    rows, losses = [], {}
    for state in ("K8", "C4"):
        for fmt, score, key, unit in format_specs:
            loss = paired_loss(pc, ids, state, score)
            losses[state, fmt] = loss
            pointer = f"/report/{state}/{key}"
            mean = results["report"][state][key]
            match(f"formats/{state}/{fmt}", loss.mean(), mean, 5.000001e-5)
            ci = interval(counts @ loss / len(loss))
            tier, ci_source, related = "报", "descriptive_bootstrap_10000_seed0", ""
            if state == "K8" and fmt in ("name_free", "word_to_definition"):
                endpoint = "naming_loss_K8" if fmt == "name_free" else "comprehension_loss_K8"
                node = results["secondary"]["S2"][endpoint]
                check_summary(f"S2/{endpoint}", loss, counts, node)
                pointer = f"/secondary/S2/{endpoint}"
                mean, ci, tier = node["estimate"], node["ci"], "确"
                ci_source, related = "original_report", "S2_joint"
            rows.append(dict(set=state, format=fmt, unit=unit, loss=mean, ci_low=ci[0], ci_high=ci[1],
                             tier=tier, source=ref(REPORT, pointer), n=len(ids), ci_source=ci_source,
                             related_test=related, raw_source=f"{CONF}/candidates.parquet" if unit == LOGP
                             else f"{CONF}/labels.parquet", derived_score=score,
                             raw_selection=f"condition={fmt}, role=target, logprob" if unit == LOGP
                             else f"fmt={fmt}, mean margin over orders/rotations"))
    check_summary("S1/naming", losses["K8", "name_free"] - losses["C4", "name_free"], counts,
                  results["secondary"]["S1"]["S1a_naming"])
    check_summary("S1/six_choice", losses["C4", "options_letter"] - losses["K8", "options_letter"], counts,
                  results["secondary"]["S1"]["S1b_six_choice"])

    cols = ["delta_none_hinted_demos", "delta_cue", "delta_miscue"]
    intact = pc["intact"].reindex(hint_ids)[cols].to_numpy()
    hints, vectors = [], {}
    for state in ("K8", "C4", "C4_answer"):
        lesion = pc[state].reindex(hint_ids)[cols].to_numpy()
        B = (lesion[:, 1] - lesion[:, 0]) - (intact[:, 1] - intact[:, 0])
        H = -((lesion[:, 2] - lesion[:, 0]) - (intact[:, 2] - intact[:, 0]))
        vectors[state] = B, H
        for measure, values in (("B", B), ("H", H)):
            pointer = f"/report/{state}/{measure}"
            mean = results["report"][state][measure]
            match(f"hints/{state}/{measure}", values.mean(), mean, 5.000001e-5)
            ci = interval(hint_counts @ values / len(values))
            tier, ci_source, related = "报", "descriptive_bootstrap_10000_seed0", ""
            if state == "C4_answer":
                node = results["secondary"]["S3"][measure]
                check_summary(f"S3/{measure}", values, hint_counts, node)
                pointer, mean, ci = f"/secondary/S3/{measure}", node["estimate"], node["ci"]
                ci_source = "original_report"
                if measure == "H":
                    tier, related = "确", "S3_joint_A_and_H"
            hints.append(dict(set=state, measure=measure, estimate=mean, ci_low=ci[0], ci_high=ci[1],
                              tier=tier, n=len(hint_ids), unit=LOGODDS, source=ref(REPORT, pointer),
                              ci_source=ci_source, related_test=related, baseline="none_hinted_demos",
                              raw_source=f"{CONF}/candidates.parquet"))
    B, H = vectors["C4_answer"]
    check_summary("S3/A", B - H, hint_counts, results["secondary"]["S3"]["A_C4_answer"])
    x = paired_loss(pc, hint_ids, "K8", "name_target_logprob")
    y = sum(vectors["K8"])
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    match("S6/rho", rho, results["secondary"]["S6"]["spearman"], 5.000001e-5)
    CHECKS["S6_unrounded_rho"] = rho
    scatter = [dict(item_id=i, naming_loss_K8=float(a), dR_K8=float(b), tier="报", related_test="S6",
                    naming_unit=LOGP, dR_unit=LOGODDS, source=f"{CONF}/candidates.parquet",
                    note="Per-concept observations. Confirmatory inference concerns the overall correlation.")
               for i, a, b in zip(hint_ids, x, y)]
    return rows, hints, scatter, counts


def errors(candidates, items, ids, results):
    scores = candidates.rename(columns={"state": "set"})
    scores = scores[scores.condition == "name_free"].assign(condition="none")
    arrays = ea.h3a_tables(scores, ids)
    cal_ids = items.index[items.role == "calibration_b"].tolist()
    if len(cal_ids) != 40 or set(cal_ids) & set(ids):
        raise ValueError("calibration/test separation differs from the frozen plan")
    cal = ea.h3a_tables(read_parquet(f"{E3}/probe_errors/naming.parquet"), cal_ids)
    tracked(ki.SUBTLEX.relative_to(ROOT).as_posix())
    zipf = ki.load_zipf()
    prior = dg.frequency_prior(ea.candidate_zipf(items.loc[ids], zipf).to_numpy())
    prior_cal = dg.frequency_prior(ea.candidate_zipf(items.loc[cal_ids], zipf).to_numpy())
    intact = np.exp(dg.log_softmax(arrays["intact"]))
    grid = np.stack([np.exp(dg.log_predict(arrays["intact"], prior, T, lam)) - intact
                     for T in dg.T_GRID for lam in dg.LAMBDA_GRID], axis=1)
    rng = np.random.default_rng(SEED)
    ct = ea.resample_counts(len(ids), N_BOOT, rng)
    cc = ea.resample_counts(len(cal_ids), N_BOOT, rng)
    rows, summaries = [], {}
    weights = np.array([{"associate": 1, "coordinate": -1, "superordinate": -1}.get(r, 0)
                        for r in ms.ERROR_ROLES])
    for state, short in (("K8", "K"), ("C4", "C")):
        ce = dg.item_cross_entropy_grid(np.exp(dg.log_softmax(cal[state])), cal["intact"], prior_cal)
        T, lam, (i, j) = dg.best(ce.sum(axis=0))
        official = results["primary"]["H3a"]
        match(f"H3a/{state}/fit", [T, lam], [official["fits"][short]["T"], official["fits"][short]["lam"]])
        g0 = i * len(dg.LAMBDA_GRID) + j
        fits = dg.bootstrap_fits(ce, cc)
        gb = fits[:, 0] * len(dg.LAMBDA_GRID) + fits[:, 1]
        obs = np.exp(dg.log_softmax(arrays[state])) - intact
        b_obs = ct @ obs / len(ids)
        b_pred = np.column_stack([np.einsum("bi,ib->b", ct, grid[:, gb, k]) / len(ids)
                                  for k in range(len(ms.ERROR_ROLES))])
        pred = grid[:, g0, :].mean(axis=0)
        point_obs, point_pred = float(obs.mean(axis=0) @ weights), float(pred @ weights)
        boot_obs, boot_pred = b_obs @ weights, b_pred @ weights
        node = official["change"][short]
        match(f"H3a/{state}/observed", point_obs, node["observed"])
        match(f"H3a/{state}/predicted", point_pred, node["predicted"])
        match(f"H3a/{state}/observed_ci", interval(boot_obs), node["observed_ci"])
        match(f"H3a/{state}/predicted_ci", interval(boot_pred), node["predicted_ci"])
        summaries[short] = point_obs, point_pred, boot_obs, boot_pred
        for k, role in enumerate(ms.ERROR_ROLES):
            if role == "target":
                continue
            ci_obs, ci_pred = interval(b_obs[:, k]), interval(b_pred[:, k])
            rows.append(dict(set=state, role=role, observed=float(obs[:, k].mean()),
                             obs_ci_low=ci_obs[0], obs_ci_high=ci_obs[1], predicted=float(pred[k]),
                             pred_ci_low=ci_pred[0], pred_ci_high=ci_pred[1], tier="报", n=len(ids),
                             n_calibration=len(cal_ids), unit="five-candidate probability change",
                             denominator="target + coordinate + superordinate + associate + unrelated",
                             related_test="H3a_aggregate", source=f"{CONF}/candidates.parquet",
                             calibration_source=f"{E3}/probe_errors/naming.parquet",
                             ci_source="descriptive_bootstrap_with_calibration_refit",
                             T=T, lambda_=lam))
    c, k = summaries["C"], summaries["K"]
    for name, point, boots in (("D_obs", c[0] - k[0], c[2] - k[2]),
                               ("D_pred", c[1] - k[1], c[3] - k[3]),
                               ("excess", c[0] - k[0] - c[1] + k[1], c[2] - k[2] - c[3] + k[3])):
        match(f"H3a/{name}", point, results["primary"]["H3a"][name])
        match(f"H3a/{name}/ci", interval(boots), results["primary"]["H3a"][name + "_ci"])
    return rows


def similarity(control):
    items = read_csv("data/materials/m1/items.csv", dtype=str)
    split = read_csv("data/materials/m1/split.csv", dtype=str)
    expected = items[items.concept_group.isin(split.loc[split.split == "test", "concept_group"])]
    scores_path = f"{E2}/dissociation/m1_scores.parquet"
    scores = read_parquet(scores_path)
    scores["item_id"] = scores.item_id.astype(str)
    for state, part in scores.groupby("set"):
        if set(part.item_id) != set(expected.item_id):
            raise ValueError(f"M1 {state} is not precisely the test split")
        if part.duplicated(["item_id", "order"]).any() or not part.groupby("item_id").order.nunique().eq(2).all():
            raise ValueError(f"M1 {state} lacks unique paired option orders")
    CHECKS["M1_test_ids"] = dict(items=len(expected), concept_groups=int(expected.concept_group.nunique()))
    h = damage.item_damage(scores[scores.set == "intact"], scores[scores.set == "control"])
    assoc = read_csv("data/materials/m1/associations.csv", dtype={"item_id": str}).set_index("item_id")
    meta = expected[expected.task.isin(("colour", "size"))].set_index("item_id")
    t = meta.join(h).join(assoc)
    x = t.w2v_distractor_probe.to_numpy(dtype=float)
    z = (x - x.mean()) / x.std()
    cov = (t.task == "size").to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(t)), z, cov])
    beta = np.linalg.lstsq(X, t.h.to_numpy(), rcond=None)[0]
    match("M1/association_slope", beta[1], control["intervals"]["association_slope"]["estimate"])
    CHECKS["M1_association_model"] = dict(n=len(t), intercept=float(beta[0]), slope=float(beta[1]),
                                        size_coefficient=float(beta[2]), raw_x_mean=float(x.mean()), raw_x_sd=float(x.std()),
                                        formula="damage_C4 ~ intercept + standardized_similarity + task_is_size")
    return [dict(item_id=i, similarity=float(raw), damage_C4=float(t.at[i, "h"]),
                 standardized_similarity=float(std), task=t.at[i, "task"], concept_group=t.at[i, "concept_group"],
                 fitted_damage_C4=float(fit), task_adjusted_damage_C4=float(t.at[i, "h"] - beta[2] * size),
                 tier="报", related_test="control_association_falsification_condition", split="test",
                 source=scores_path, similarity_source="data/materials/m1/associations.csv")
            for i, raw, std, fit, size in zip(t.index, x, z, X @ beta, cov)]


def direct_effects(ids, counts, results):
    source = f"{CONF}/dla.parquet"
    dla = read_parquet(source)
    if dla.duplicated(["item_id", "component"]).any():
        raise ValueError("duplicate DLA component/concept rows")
    wide = dla.pivot(index="item_id", columns="component", values="specific").reindex(ids)
    if wide.shape != (160, 8) or set(wide.columns) != K8 or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("unexpected K8 DLA coverage")
    rows = []
    for component in wide:
        v = wide[component].to_numpy()
        ci = interval(counts @ v / len(v))
        rows.append(dict(component=component, specific=float(v.mean()), ci_low=ci[0], ci_high=ci[1],
                         tier="报", n=len(ids), unit="first-token specific logit contribution", source=source,
                         ci_source="descriptive_bootstrap_10000_seed0", related_test="S4_group_sum"))
    total = wide.sum(axis=1).to_numpy()
    node = results["secondary"]["S4"]["direct_specific_K8"]
    check_summary("S4/group_sum", total, counts, node)
    rows.append(dict(component="K8_sum", specific=node["estimate"], ci_low=node["ci"][0], ci_high=node["ci"][1],
                     tier="确", n=len(ids), unit="first-token specific logit contribution",
                     source=ref(REPORT, "/secondary/S4/direct_specific_K8"), ci_source="original_report",
                     related_test="S4_group_sum"))
    CHECKS["DLA_zero_components"] = list(wide.columns[(wide == 0).all()])
    return rows


def tests(results, knowledge, control, g1):
    rows = []

    def add(test, endpoint, prediction, node, criterion, outcome, source, n, unit,
            tier="确", p="", p_scope="not provided / CI decision", family=""):
        ci = report_interval(node)
        rows.append(dict(test=test, endpoint=endpoint, prediction=prediction, estimate=node["estimate"],
                         ci_low=ci[0], ci_high=ci[1], criterion=criterion, p_holm=p, outcome=outcome,
                         tier=tier, n=n, unit=unit, source=source, p_scope=p_scope, family=family,
                         note="G1 revised after development results were observed (ADR 0003)" if test == "H1_G1" else ""))

    model = g1["chosen_version"]
    add("H1_G1", "matching_tds_effect", "完好模型具备可用的干扰效应", g1["versions"][model]["matching"]["tds_effect"],
        "TDS CI lower > .20 AND order-averaged high-demand accuracy CI lower > .50", g1["g1_decision"],
        ref(G1, f"/versions/{model}/matching/tds_effect"), 160, LOGIT, tier="测", family="development_measurement_gate")
    m1 = read_csv("data/materials/m1/items.csv", dtype=str)
    n_hard = int((m1.task.isin(("colour", "size")) & m1.high_demand.eq("1")).sum())
    add("H1_G1", "order_averaged_high_demand_accuracy", "顺序平均后难题正确率可靠地高于机会水平",
        g1["versions"][model]["order_averaged_high_demand_accuracy"],
        "TDS CI lower > .20 AND order-averaged high-demand accuracy CI lower > .50", g1["g1_decision"],
        ref(G1, f"/versions/{model}/order_averaged_high_demand_accuracy"), n_hard, "proportion",
        tier="测", family="development_measurement_gate")
    add("H2_control", "control_effect", "强干扰比弱干扰受损更多", control["intervals"]["control_effect"],
        "CI lower > 0.20", control["primary"]["verdict"], ref(CONTROL, "/intervals/control_effect"), 63, LOGIT)
    add("H2_control", "association_slope", "损伤随诱饵相似度增加", control["intervals"]["association_slope"],
        "Falsify if CI upper <= 0. Task-adjusted standardized similarity", control["association"]["verdict"],
        ref(CONTROL, "/intervals/association_slope"), 63, "logit / 1 SD similarity", tier="预定证伪条件")
    for task, endpoint, n, unit in (("naming", "delta_F", 240, LOGP), ("m1", "delta_C", 63, LOGIT)):
        add("H2_dissociation", endpoint, "K8 的词频效应更大且 C4 的控制效应更大", knowledge["sets"]["main"][task][endpoint],
            "Both CI lower > 0. Meaningful if lower > 0.20", knowledge["h2"]["decision"],
            ref(KNOWLEDGE, f"/sets/main/{task}/{endpoint}"), n, unit)
    for test, unit in (("H3a", "five-candidate probability change"), ("H3b", LOGODDS)):
        group = results["primary"][test]
        for endpoint in ("D_obs", "excess"):
            add(test, endpoint, "C 减 K 的标记变化为正且超出一般退化预测",
                dict(estimate=group[endpoint], ci=group[endpoint + "_ci"]),
                "Both D_obs and excess CI lower > 0 AND family Holm p < .05. Small if excess < "
                + (".05" if test == "H3a" else ".20"), group["verdict"],
                ref(REPORT, f"/primary/{test}/{endpoint}"), group["n_test"], unit,
                p=group["p_holm"], p_scope=f"{test} group p for excess, repeated across endpoints", family="E3_primary_2")
    specs = (
        ("S1", "S1a_naming", "K8 的命名损失大于 C4", "Both S1 endpoint CI lower > 0", LOGP),
        ("S1", "S1b_six_choice", "C4 的六选一损失大于 K8", "Both S1 endpoint CI lower > 0", LOGIT),
        ("S2", "naming_loss_K8", "命名受损且词选定义非劣效", "Naming CI lower > .20 AND comprehension CI upper < .73", LOGP),
        ("S2", "comprehension_loss_K8", "命名受损且词选定义非劣效", "Naming CI lower > .20 AND comprehension CI upper < .73", LOGIT),
        ("S3", "A_C4_answer", "回答区 C4 误导伤害增加且大于提示获益变化", "A CI upper < 0 AND H CI lower > 0", LOGODDS),
        ("S3", "H", "回答区 C4 误导伤害增加", "A CI upper < 0 AND H CI lower > 0", LOGODDS),
        ("S4", "direct_specific_K8", "K8 整组对目标首 token 的特异直接贡献为正", "CI lower > 0", "first-token specific logit contribution"),
        ("S5", "associate_excess_K30", "K30 联想词份额增幅小于退化预测", "Excess CI upper < 0", "five-candidate probability change"),
        ("S6", "spearman", "K8 命名损失与提示依赖增幅正相关", "CI lower > 0", "Spearman rho"),
    )
    for test, endpoint, prediction, criterion, unit in specs:
        group = results["secondary"][test]
        node = group[endpoint]
        pointer = f"/secondary/{test}/{endpoint}"
        if test == "S5":
            node = dict(estimate=node["excess"], ci=node["excess_ci"], n=node["n"])
            pointer += "/excess"
        if test == "S6":
            node = dict(estimate=node, ci=group["ci"], n=group["n"])
        add(test, endpoint, prediction, node, criterion + " AND secondary-family Holm p < .05", group["verdict"],
            ref(REPORT, pointer), node["n"], unit, p=group["p_holm"],
            p_scope=f"{test} group p, not a separate test for each endpoint", family="E3_secondary_6")
    return rows


def inventory_random_scores():
    source = f"{E2}/dissociation/naming_scores.parquet"
    naming = read_parquet(source, columns=["set", "scoring"])
    CHECKS["random_naming_score_inventory"] = {
        scoring: dict(n_sets=int(g.set.nunique()), n_random_sets=int(g.loc[g.set.str.startswith("random_"), "set"].nunique()),
                      n_rows=len(g)) for scoring, g in naming.groupby("scoring")}
    CHECKS["local_control_random_raw_files_present"] = {
        name: (ROOT / E2 / "control" / name).is_file() for name in ("intact.csv", "lesioned.parquet")}


def csv_text(rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def build():
    INPUTS.clear()
    CHECKS.clear()
    results, knowledge, control, g1 = (read_json(p) for p in (REPORT, KNOWLEDGE, CONTROL, G1))
    if results["dry_run"]:
        raise ValueError("formal report is marked dry_run")
    items = read_csv("data/materials/m3/final_items.csv").set_index("item_id")
    ids = items.index[items.role == "test"].tolist()
    hints = items.index[(items.role == "test") & items.free_naming.astype(str).eq("True")].tolist()
    errors_ids = [i for i in ids if i not in ("m000", "m043", "m081", "m124")]
    match("sample_sizes", [len(ids), len(errors_ids), len(hints)], [160, 156, 100])
    candidates, labels = (read_parquet(f"{CONF}/{name}.parquet") for name in ("candidates", "labels"))
    for name, frame in (("candidates", candidates), ("labels", labels)):
        for state, part in frame.groupby("state"):
            if set(part.item_id) != set(ids):
                raise ValueError(f"unexpected {name}/{state} concept coverage")
    pc = analyze_battery.per_concept(candidates, labels)
    formats, hint_rows, scatter, counts = formats_and_hints(pc, ids, hints, results)
    data = {
        "fig2c_dissociation.csv": dissociation(knowledge),
        "fig3a_formats.csv": formats,
        "fig4a_errors.csv": errors(candidates, items, errors_ids, results),
        "fig4b_hints.csv": hint_rows,
        "fig4c_cue_vs_loss.csv": scatter,
        "table1_tests.csv": tests(results, knowledge, control, g1),
        "fig2d_similarity.csv": similarity(control),
        "fig3b_dla.csv": direct_effects(ids, counts, results),
    }
    inventory_random_scores()
    for source in (
        "docs/paper-prep/figdata/build.py", "experiments/e3_errors/analyze_battery.py",
        "experiments/e3_errors/analyze_experiment3.py", "src/lesion/e3_analysis.py", "src/lesion/degradation.py",
        "src/lesion/m3_scoring.py", "src/lesion/knowledge_items.py", "src/lesion/damage.py",
        "experiments/e2_confirm/run_dissociation_test.py", "docs/analysis-plan/control-test.md",
        "docs/analysis-plan/experiment3.md", "docs/adr/0003-qwen-and-revised-g1.md",
    ):
        tracked(source)
    outputs = {name: csv_text(rows) for name, rows in data.items()}
    manifest = dict(schema_version=1, operation="CPU paper preparation from completed results, not new model experiments",
                    bootstrap=dict(n=N_BOOT, seed=SEED, ci="95% percentile", resampling_unit="concept",
                                   secondary_rng="160-concept draws then 100-concept draws, as in formal analyzer",
                                   errors_rng="156 test-concept draws then 40 independent calibration-concept draws",
                                   new_intervals="descriptive only; no new p-values"),
                    software=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                    inputs=dict(sorted(INPUTS.items())), checks=CHECKS,
                    outputs={name: dict(rows=len(data[name]), sha256=hashlib.sha256(value.encode("utf-8")).hexdigest())
                             for name, value in outputs.items()})
    outputs["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build()  # Complete every numeric check before writing any figure file.
    if args.check:
        different = [name for name, content in outputs.items()
                     if not (OUT / name).exists() or (OUT / name).read_bytes() != content.encode("utf-8")]
        if different:
            raise SystemExit("NOT REPRODUCIBLE: " + ", ".join(different))
        print(f"PASS: {len(outputs) - 1} CSVs and manifest match their sources and recorded calculations")
    else:
        for name, content in outputs.items():
            (OUT / name).write_bytes(content.encode("utf-8"))
        print(f"Wrote {len(outputs) - 1} CSVs and manifest; all source-value checks passed")


if __name__ == "__main__":
    main()
