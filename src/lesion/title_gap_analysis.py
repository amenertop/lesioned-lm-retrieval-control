"""Paired concept-level analyses for the new comprehension and production batteries.

Input tables must be complete and keyed. Formats and response modalities are never treated as independent
concepts. Unknown generated responses remain unclassified until a blind review supplies their categories.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import runinfo, title_gap_scoring


def read_run(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if title_gap_scoring._fingerprint({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("plan contents changed after scoring")
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    names = [s["name"] for s in plan["states"]]
    if complete != dict(fingerprint=plan["fingerprint"], states=names):
        raise ValueError("run is incomplete or has different states")
    from .m3_scoring import NamingPrompt
    from .m3_battery import LabelItem
    naming = [NamingPrompt(**p) for p in plan["naming"]]
    labels = [LabelItem(**p) for p in plan["labels"]]
    accumulated = {name: [] for name in ("candidates", "labels", "generated")}
    language = []
    for state in names:
        folder = directory / state
        marker = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        if marker["fingerprint"] != plan["fingerprint"]:
            raise ValueError("mixed state provenance")
        language.append(marker["language"])
        for name, digest in marker["files"].items():
            if runinfo.sha256(folder / name) != digest:
                raise ValueError(f"{state}/{name}: changed source")
        tables = {name: pd.read_parquet(folder / f"{name}.parquet") for name in accumulated}
        title_gap_scoring.validate_tables(tables, naming, labels)
        for name, table in tables.items():
            accumulated[name].append(table.assign(state=state))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in accumulated.items()}
    tables["language"] = pd.DataFrame(language)
    return tables, plan


def paired_damage(table, score, keys):
    if table.duplicated(["state", *keys]).any():
        raise ValueError("duplicate state/item observations")
    wide = table.pivot(index=keys, columns="state", values=score)
    if "intact" not in wide or wide.isna().any().any():
        raise ValueError("missing intact or lesion observations")
    return (wide.intact.to_numpy()[:, None] - wide).rename_axis(columns="state").stack().rename("damage").reset_index()


def bootstrap(frame, statistic, n_boot=2000, seed=1910):
    groups = [g for _, g in frame.groupby("family", sort=True)]
    if n_boot == 0:
        return dict(estimate=float(statistic(frame)), ci_low=None, ci_high=None, n_families=len(groups), status="point_only")
    if len(groups) < 4:
        return dict(estimate=float(statistic(frame)), ci_low=None, ci_high=None, n_families=len(groups), status="too_few_families")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))], ignore_index=True)
        values.append(statistic(sample))
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) < 0.9 * n_boot:
        return dict(estimate=float(statistic(frame)), ci_low=None, ci_high=None, n_families=len(groups), status="unstable")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return dict(estimate=float(statistic(frame)), ci_low=float(lo), ci_high=float(hi), n_families=len(groups), status="estimated")


def partial_correlation(frame, a, b, covariates):
    x = np.column_stack([np.ones(len(frame)), frame[list(covariates)].to_numpy(float)])
    y = frame[[a, b]].to_numpy(float)
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    if np.min(residual.std(0)) < 1e-8:
        return np.nan
    return float(np.corrcoef(residual.T)[0, 1])


def bootstrap_family_mean(frame, column, n_boot=2000, seed=1910):
    """Vectorized paired resampling for a table with exactly one observation per family."""
    if frame.family.duplicated().any():
        raise ValueError("family means require one paired row per family")
    values = frame.sort_values('family')[column].to_numpy(float)
    if not np.isfinite(values).all() or not len(values):
        raise ValueError("missing family-level measurements")
    result = dict(estimate=float(values.mean()), ci_low=None, ci_high=None, n_families=len(values),
                  status="point_only" if n_boot == 0 else "too_few_families")
    if n_boot and len(values) >= 4:
        ix = np.random.default_rng(seed).integers(len(values), size=(n_boot, len(values)))
        lo, hi = np.quantile(values[ix].mean(1), [.025, .975])
        result.update(ci_low=float(lo), ci_high=float(hi), status="estimated")
    return result


def fine_contrast(frame, first, second, n_boot=2000):
    """Direct lesion differences on the same concepts, including differences of residual correlations."""
    a = frame[frame.state == first].set_index("item_id")
    b = frame[frame.state == second].set_index("item_id")
    if set(a.index) != set(b.index) or a.empty:
        raise ValueError("lesion contrasts need the same nonempty item set")
    b = b.loc[a.index]
    joined = a.reset_index()
    tasks = [t for t in ("naming", "definition", "synonym", "property") if t in joined]
    for task in tasks:
        joined[task + "_other"] = b[task].to_numpy()
    result = {}
    for task in tasks:
        paired = joined.dropna(subset=[task, task + "_other"])
        result[task + "_damage_difference"] = bootstrap(paired, lambda t, task=task: (t[task] - t[task + "_other"]).mean(), n_boot)
    for task in tasks[1:]:
        covs = ["frequency", "n_tokens", "intact_naming", f"intact_{task}"]
        if "semd" in joined and joined.semd.notna().all():
            covs.append("semd")
        paired = joined.dropna(subset=["naming", "naming_other", task, task + "_other", *covs])
        result[task + "_consistency_difference"] = bootstrap(paired, lambda t, task=task, covs=covs:
            partial_correlation(t, "naming", task, covs) - partial_correlation(t, "naming_other", task + "_other", covs), n_boot)
    return result


def fine_profiles(tables, items, n_boot=2000):
    labels = tables["labels"].groupby(["state", "item_id", "fmt"], as_index=False).margin.mean()
    naming = tables["candidates"].query("condition == 'naming' and role == 'target'")
    scores = pd.concat([labels.rename(columns={"fmt": "task", "margin": "score"}),
                        naming[["state", "item_id", "logprob"]].rename(columns={"logprob": "score"}).assign(task="naming")])
    d = paired_damage(scores, "score", ["item_id", "task"])
    d = d.pivot(index=["state", "item_id"], columns="task", values="damage").reset_index()
    tokens = naming.query("state == 'intact'")[["item_id", "n_tokens"]]
    base = scores.query("state == 'intact'").pivot(index="item_id", columns="task", values="score").add_prefix("intact_").reset_index()
    # Item text uses the same names as two outcomes. Keep numeric tasks unshadowed by metadata.
    info = items.drop(columns=[task for task in scores.task.unique() if task in items])
    frame = d.merge(info, on="item_id", validate="many_to_one").merge(tokens, on="item_id", validate="many_to_one").merge(base, on="item_id", validate="many_to_one")
    results = {}
    for state, group in frame.groupby("state"):
        if state == "intact":
            continue
        tasks = [t for t in ("naming", "definition", "synonym", "property") if t in group]
        summary = {task: bootstrap(group.dropna(subset=[task]), lambda t, task=task: t[task].mean(), n_boot) for task in tasks}
        for task in tasks[1:]:
            covariates = ["frequency", "n_tokens", "intact_naming", f"intact_{task}"]
            if "semd" in group and group.semd.notna().all():
                covariates.append("semd")
            paired = group.dropna(subset=["naming", task, *covariates])
            summary[f"naming_{task}_partial_r"] = bootstrap(paired, lambda t, task=task, covariates=covariates:
                                                            partial_correlation(t, "naming", task, covariates), n_boot)
        for task in tasks:
            def slope(t, task=task):
                covs = ["frequency", "n_tokens", f"intact_{task}"]
                if "semd" in t and t.semd.notna().all():
                    covs.append("semd")
                x = np.column_stack([np.ones(len(t)), t[covs].to_numpy(float)])
                return np.linalg.lstsq(x, t[task], rcond=None)[0][1]
            summary[f"{task}_frequency_slope"] = bootstrap(group.dropna(subset=[task, f"intact_{task}"]), slope, n_boot)
        results[state] = summary
    return results, frame


def fine_label_accuracy(labels):
    """Average the two option orders per judgement, then count each property's truth judgement."""
    labels = labels.copy()
    labels["claim"] = 0
    prop = labels.fmt == "property"
    if prop.any():
        if "variant" not in labels or labels.loc[prop, "variant"].isna().any():
            raise ValueError("property accuracy requires the true/false variant")
        labels.loc[prop, "claim"] = labels.loc[prop, "variant"].astype(int) // 2
    claims = labels.groupby(["state", "item_id", "fmt", "claim"], as_index=False).margin.mean()
    claims["correct_response"] = (claims.margin > 0).astype(float)
    return claims.groupby(["state", "item_id", "fmt"], as_index=False).correct_response.mean()


def learned_development_scores(frame, states, profiles, answers, language):
    """Apply the written development rule, reusing the exact training objective and screening scales."""
    import torch
    from .fine_objective import TASKS, hub_terms
    base_loss = language.loc[language.state == "intact", "loss"].item()
    n = len(frame[frame.state == "intact"])

    def format_accuracy(state):
        return 1 - answers[state]["categories"].get("format_invalid", {}).get("naming", 0) / n

    rows = []
    for state in states:
        if "training" not in state:
            continue
        name, meta = state["name"], state["training"]
        if meta["mode"] not in ("hub", "comprehension"):
            raise ValueError("unknown trained objective")
        data = frame[frame.state == name]
        scales = torch.tensor(meta["scales"], dtype=torch.float64)
        if scales.shape != (3,) or not torch.isfinite(scales).all() or (scales <= 0).any() or len(data) != n:
            raise ValueError("invalid screening scales or incomplete development concepts")
        damage = torch.tensor(data[list(TASKS)].to_numpy(), dtype=torch.float64) / scales
        intact = torch.tensor(data[['intact_' + t for t in TASKS]].to_numpy(), dtype=torch.float64) / scales
        terms = hub_terms(damage, data.frequency.to_numpy(copy=True), data.n_tokens.to_numpy(copy=True), intact, meta["mode"])
        score = float(terms["frequency"] + .5 * terms["shared"] + .25 * terms["consistency"])
        lm_change = language.loc[language.state == name, "loss"].item() / base_loss - 1
        copy = answers[name]["generated_accuracy"]["copy"]
        eligible = (np.isfinite(score) and lm_change <= .1 and copy is not None
                    and copy >= answers["intact"]["generated_accuracy"]["copy"] - .05
                    and format_accuracy(name) >= format_accuracy("intact") - .05)
        metrics = profiles[name]
        tasks = TASKS if meta["mode"] == "hub" else TASKS[1:]
        complete = all(metrics[t]["estimate"] > 0 and metrics[t + "_frequency_slope"]["estimate"] < 0 for t in tasks)
        if meta["mode"] == "hub":
            complete &= all(metrics['naming_' + t + '_partial_r']["estimate"] > 0 for t in TASKS[1:])
        else:
            covs = ['frequency', 'n_tokens', 'intact_definition', 'intact_synonym', 'semd']
            complete &= partial_correlation(data, 'definition', 'synonym', covs) > 0
            eligible &= float(terms["naming_damage"]) <= .1
        rows.append(dict(state=name, mode=meta["mode"], seed=meta["seed"], kind=state["kind"],
                         n_members=len(state["gates"]), score=score, eligible=bool(eligible),
                         complete_profile=bool(complete), lm_change=lm_change, copy_accuracy=copy,
                         naming_format_accuracy=format_accuracy(name),
                         **{k: float(v) for k, v in terms.items()}))
    if not rows:
        raise ValueError("run contains no learned states with training provenance")
    return pd.DataFrame(rows).sort_values(['mode', 'kind', 'eligible', 'complete_profile', 'score', 'n_members', 'seed', 'state'],
        ascending=[True, True, False, False, False, True, True, True]).reset_index(drop=True)


def fine_ledger_summary(result, frame, n_neutral_texts):
    """Ledger-ready outcomes. Unresolved generated answers stay missing, with their review status retained."""
    states = {}
    for state, metrics in result["results"].items():
        rows = frame[frame.state == state]
        saved = {}
        for name, value in metrics.items():
            correlation = name.endswith("_partial_r")
            task = name.split("_")[1] if correlation else name.split("_")[0]
            unit = "correlation" if correlation else "nat" if task == "naming" else "logit"
            if name.endswith("_frequency_slope"):
                unit += " / log10 BNC frequency"
            saved[name] = {k: v for k, v in value.items() if k != "n_families"}
            saved[name].update(unit=unit, n=int(rows[task].notna().sum()), n_unit="概念")
        for task, value in result["actual_answers"][state]["label_accuracy"].items():
            saved[task + "_accuracy"] = dict(estimate=value, unit="proportion", n=int(rows[task].notna().sum()), n_unit="概念")
        states[state] = saved
    for state, answers in result["actual_answers"].items():
        saved = states.setdefault(state, {})
        rows = frame[frame.state == state]
        for task, value in answers["label_accuracy"].items():
            saved[task + "_accuracy"] = dict(estimate=value, unit="proportion", n=int(rows[task].notna().sum()), n_unit="概念")
        for condition, value in answers["generated_accuracy"].items():
            count = sum(g.get(condition, 0) for g in answers["categories"].values())
            saved[condition + "_generated_accuracy"] = dict(estimate=value, unit="proportion", n=count,
                n_unit="回答", status="fully_classified" if value is not None else "needs_blind_response_review")
        for name, value in answers.items():
            if name.endswith(("_binary_r", "_response_r")):
                saved[name] = {k: v for k, v in value.items() if k != "n_families"}
                saved[name].update(unit="correlation", n=value["n_families"], n_unit="概念")
        errors = answers["naming_error_profile"]
        for category, value in errors["single_word_semantic_proportions"].items():
            saved[category + "_share_of_semantic_errors"] = dict(estimate=value, unit="proportion",
                n=errors["semantic_errors"], n_unit="单词语义错误", status=errors["status"])
    base = next(r["loss"] for r in result["language"] if r["state"] == "intact")
    for row in result["language"]:
        if row["state"] in states:
            states[row["state"]]["lm_change"] = dict(estimate=row["loss"] / base - 1,
                unit="relative change", n=n_neutral_texts, n_unit="中性文本")
    return dict(meta=dict(status=result["status"], provenance=result["provenance"],
        source_fingerprint=result["source_fingerprint"], analysis_sources=result["analysis_sources"],
        auxiliary_sources=result["auxiliary_sources"], n_boot=result["n_boot"],
        generated_answer_status={s: a["naming_accuracy_status"] for s, a in result["actual_answers"].items()}), states=states)


def fine_error_profile(named):
    """Use the clinical single-word semantic-error denominator, never all errors or all responses."""
    counts = named.category.value_counts()
    semantic = {k: int(counts.get(k, 0)) for k in ("coordinate", "superordinate", "associate")}
    denominator = sum(semantic.values())
    unknown = int(counts.get("unclassified", 0))
    errors = int(len(named) - counts.get("correct", 0))
    return dict(n=len(named), unclassified=unknown, semantic_errors=denominator,
                status="needs_blind_response_review" if unknown else "fully_classified",
                semantic_share_of_all_errors=denominator / errors if errors and not unknown else None,
                single_word_semantic_counts=semantic,
                single_word_semantic_proportions={k: v / denominator if denominator and not unknown else None
                                                  for k, v in semantic.items()})


def fine_candidate_specificity(candidates, items, n_boot=2000):
    """Post hoc target-versus-neighbour contrasts, using already scored whole-word probabilities."""
    rows = candidates[candidates.condition == "naming"]
    keys = ["state", "item_id"]
    scores = rows.pivot(index=keys, columns="role", values="logprob")
    words = rows.pivot(index=keys, columns="role", values="word")
    roles = ["target", "synonym", "coordinate"]
    if scores[roles].isna().any().any() or any(words[a].eq(words[b]).any()
                                            for a, b in (("target", "synonym"), ("target", "coordinate"), ("synonym", "coordinate"))):
        raise ValueError("specificity needs distinct, fully scored target, synonym and coordinate words")
    scores["target_vs_coordinate"] = scores.target - scores.coordinate
    scores["accepted_vs_coordinate"] = np.logaddexp(scores.target, scores.synonym) - scores.coordinate
    contrasts = ["target_vs_coordinate", "accepted_vs_coordinate"]
    scores = scores.reset_index().merge(items[["item_id", "family"]], on="item_id", validate="many_to_one")
    long = scores.melt(id_vars=[*keys, "family"], value_vars=contrasts, var_name="contrast", value_name="score")
    damage = paired_damage(long, "score", ["item_id", "contrast"])
    rows = long.merge(damage, on=[*keys, "contrast"], validate="one_to_one")
    states = {}
    for (state, contrast), part in rows.groupby(["state", "contrast"]):
        result = bootstrap_family_mean(part, "damage", n_boot)
        result.pop("n_families")
        states.setdefault(state, {})[contrast + "_damage"] = dict(**result, unit="nat", n=len(part), n_unit="概念")
    return states, rows


def fine_readout_tables(tables, items, readout="target"):
    """Change only the naming readout. Raw candidate tables and the original scoring stay untouched."""
    if readout == "target":
        return tables
    if readout not in ("target_vs_coordinate", "accepted_vs_coordinate"):
        raise ValueError("unknown naming readout")
    _, relative = fine_candidate_specificity(tables['candidates'], items, n_boot=0)
    candidates = tables['candidates'].copy()
    index = candidates.condition.eq('naming') & candidates.role.eq('target')
    values = relative[relative.contrast == readout].set_index(['state', 'item_id']).score
    keys = pd.MultiIndex.from_frame(candidates.loc[index, ['state', 'item_id']])
    candidates.loc[index, 'logprob'] = values.reindex(keys).to_numpy()
    if not np.isfinite(candidates.loc[index, 'logprob']).all():
        raise ValueError('a relative naming score is missing')
    return {**tables, 'candidates': candidates}


def fine_responses(tables, items, n_boot=2000, review=None):
    """Actual answers and cross-task consistency, with unresolved synonyms left for blind review."""
    rows = tables["generated"].merge(items, on="item_id", validate="many_to_one")
    categories = []
    for r in rows.to_dict("records"):
        answer = str(r["answer"]).lower().strip().strip('"\'.,! ')
        accepted = {r["target"], r["synonym"], *str(r.get("accepted_answers", "")).split("|")}
        if not re.fullmatch("[a-z]+", answer):
            category = "format_invalid"
        elif r["condition"].startswith("synonym_word"):
            category = "correct" if answer == r["synonym"] else "incorrect" if answer == r["foil"] else "unclassified"
        elif answer == r["target"] or (r["condition"] == "naming" and answer in accepted):
            category = "correct"
        elif r["condition"] == "copy":
            category = "incorrect"
        elif answer == r["foil"]:
            category = "coordinate"
        elif answer == r.get("superordinate"):
            category = "superordinate"
        else:
            category = "unclassified"
        categories.append(category)
    rows["category"] = categories
    rows["response_id"] = [hashlib.sha256(json.dumps([r["item_id"], r["condition"], r["answer"]]).encode()).hexdigest()[:16]
                           for r in rows.to_dict("records")]
    if review is not None:
        allowed = {"correct", "coordinate", "superordinate", "associate", "phonological", "omission",
                   "unrelated", "format_invalid", "incorrect"}
        if review.response_id.duplicated().any() or set(review.category) - allowed or set(review.response_id) - set(rows.response_id):
            raise ValueError("invalid blind fine-response review")
        judged = rows.response_id.map(review.set_index("response_id").category)
        rows.loc[judged.notna(), "category"] = judged[judged.notna()]
    rows["correct_response"] = rows.category.eq("correct").astype("boolean")
    rows.loc[rows.category.eq("unclassified"), "correct_response"] = pd.NA
    labels = fine_label_accuracy(tables["labels"])
    results = {}
    for state, generated in rows.groupby("state"):
        summary = dict(categories=generated.groupby("condition").category.value_counts().unstack(fill_value=0).to_dict(),
                       naming_accuracy_status="needs_blind_response_review" if generated.query("condition == 'naming'").correct_response.isna().any() else "fully_classified")
        summary["generated_accuracy"] = {condition: float(g.correct_response.mean()) if not g.correct_response.isna().any() else None
                                         for condition, g in generated.groupby("condition")}
        summary["naming_error_profile"] = fine_error_profile(generated.query("condition == 'naming'"))
        named = generated.query("condition == 'naming'")[["item_id", "family", "correct_response"]]
        wide = labels[labels.state == state].pivot(index="item_id", columns="fmt", values="correct_response")
        joined = named.rename(columns={"correct_response": "naming"}).merge(wide, on="item_id", validate="one_to_one")
        pairs = [("definition", "synonym")]
        if "property" in joined:
            pairs += [("definition", "property"), ("synonym", "property")]
        if not joined.naming.isna().any():
            pairs += [("naming", "definition"), ("naming", "synonym")]
            if "property" in joined:
                pairs += [("naming", "property")]
        for a, b in pairs:
            metric = "response_r" if "property" in (a, b) else "binary_r"
            summary[f"{a}_{b}_{metric}"] = bootstrap(joined.dropna(subset=[a, b]), lambda t, a=a, b=b:
                float(np.corrcoef(t[a].astype(float), t[b].astype(float))[0, 1])
                if t[a].nunique() > 1 and t[b].nunique() > 1 else np.nan, n_boot)
        summary["label_accuracy"] = {task: float(g.correct_response.mean()) for task, g in labels[labels.state == state].groupby("fmt")}
        results[state] = summary
    return results, rows


def homonym_scores(tables, items, n_boot=2000):
    """Candidate and choice evidence, separate from what the model actually generates."""
    keys = ["state", "item_id", "condition"]
    candidates = tables["candidates"].pivot(index=keys, columns="role", values="logprob")
    if candidates[["target", "other_meaning"]].isna().any().any():
        raise ValueError("both homonym candidates are required")
    candidate = (candidates.target - candidates.other_meaning).rename("score").reset_index()
    choice = tables["labels"].groupby(["state", "item_id", "fmt"]).margin.mean().rename("score").reset_index()
    choice = choice.rename(columns={"fmt": "condition"})
    rows = pd.concat([candidate.assign(readout="candidate_log_odds"), choice.assign(readout="choice_logit")])
    rows = rows.merge(items, on="item_id", validate="many_to_one")
    rows["context"] = rows.condition.str.rsplit("_", n=1).str[0]
    rows["cue"] = rows.condition.str.rsplit("_", n=1).str[-1]
    results = []
    for (readout, state, context, dominance), group in rows.groupby(["readout", "state", "context", "dominance"]):
        wide = group.pivot(index=["item_id", "family"], columns="cue", values="score").reset_index()
        if set(("none", "correct", "misleading")) - set(wide) or wide.isna().any().any():
            raise ValueError("each homonym needs all three cue conditions")
        wide["correct_gain"] = wide.correct - wide["none"]
        wide["misleading_cost"] = wide["none"] - wide.misleading
        results.append(dict(readout=readout, state=state, context=context, dominance=dominance,
                            endpoint="restricted candidate evidence, not generated accuracy",
                            **{name: bootstrap_family_mean(wide, name, n_boot)
                               for name in ("none", "correct_gain", "misleading_cost")}))
    return results, rows


def binary_response_bounds(rows, weights, n_boot=0, seed=1910):
    """Bounds over unresolved judgements, keeping one label for every repeated response ID."""
    known = rows.semantic_correct.notna().to_numpy()
    weights = np.asarray(weights, dtype=float)
    observed = float(weights[known] @ rows.loc[known, "semantic_correct"].to_numpy(float))
    uncertain = pd.DataFrame(dict(response_id=rows.loc[~known, "response_id"], weight=weights[~known]))
    coefficients = uncertain.groupby("response_id").weight.sum()
    result = dict(lower=observed + float(coefficients.clip(upper=0).sum()),
                  upper=observed + float(coefficients.clip(lower=0).sum()),
                  unresolved_responses=int(uncertain.response_id.nunique()),
                  interpretation="classification bounds, not sampling confidence limits")
    if n_boot:
        family, names = pd.factorize(rows.family, sort=True)
        if len(names) < 4:
            result['sampling_envelope'] = dict(status='too_few_families', n_families=len(names))
            return result
        fixed = np.bincount(family[known], weights=weights[known]*rows.loc[known, 'semantic_correct'].to_numpy(float),
                            minlength=len(names))
        response, unknown = pd.factorize(rows.loc[~known, 'response_id'], sort=True)
        variable = np.zeros((len(names), len(unknown)))
        np.add.at(variable, (family[~known], response), weights[~known])
        ix = np.random.default_rng(seed).integers(len(names), size=(n_boot, len(names)))
        counts = np.zeros((n_boot, len(names)), dtype=int)
        np.add.at(counts, (np.arange(n_boot)[:, None], ix), 1)
        constant, uncertain = counts @ fixed, counts @ variable
        lower = constant + np.minimum(uncertain, 0).sum(1)
        upper = constant + np.maximum(uncertain, 0).sum(1)
        result['sampling_envelope'] = dict(lower=float(np.quantile(lower, .025)),
            upper=float(np.quantile(upper, .975)), n_families=len(names), n_boot=n_boot, seed=seed,
            interpretation='outer percentile interval with paired-family sampling and tied unresolved labels')
    return result


def homonym_rescue_bounds(group):
    """Conservative rescue bounds. Unknown baseline errors remain in the possible denominator."""
    wide = group.pivot(index="item_id", columns="cue", values="semantic_correct")
    ids = group.pivot(index="item_id", columns="cue", values="response_id")
    wrong = wide["none"].eq(False).fillna(False)
    can_be_wrong = wrong | wide["none"].isna()
    rescued = wrong & wide.correct.eq(True).fillna(False)
    can_be_rescued = can_be_wrong & ~wide.correct.eq(False).fillna(False) & ids["none"].ne(ids.correct)
    n_min, n_max = int(wrong.sum()), int(can_be_wrong.sum())
    return dict(denominator_min=n_min, denominator_max=n_max,
                lower=float(rescued.sum() / n_max) if n_max else None,
                upper=min(1., float(can_be_rescued.sum() / max(1, n_min))) if n_max else None,
                interpretation="conservative classification bounds, not sampling confidence limits")


def homonym_recovery(generated, items, lexicon, n_boot=2000, review=None):
    from .homonym_items import classify, normalize_answer
    if generated.duplicated(["state", "item_id", "condition"]).any():
        raise ValueError("duplicate generated responses")
    rows = generated.merge(items, on="item_id", validate="many_to_one")
    rows["category"] = [classify(r["answer"], r, lexicon) for r in rows.to_dict("records")]
    rows['initial_category'] = rows.category
    # Blinding key has no state or cue. The same response to the same intended meaning receives one judgement.
    rows["response_id"] = [hashlib.sha256(json.dumps([r["word"], int(r["sense"]), normalize_answer(r["answer"])]).encode()).hexdigest()[:16]
                           for r in rows.to_dict("records")]
    if review is not None:
        if review.response_id.duplicated().any() or set(review.category) - {"same_meaning", "other_meaning", "unrelated", "ambiguous_response", "unclassified"}:
            raise ValueError("invalid or duplicate blind response judgements")
        if set(review.response_id) - set(rows.loc[rows.initial_category == 'unclassified', 'response_id']):
            raise ValueError("response review may only resolve initially unclassified answers")
        mapping = review.set_index("response_id").category
        judged = rows.response_id.map(mapping)
        rows.loc[judged.notna(), "category"] = judged[judged.notna()]
    rows["exact_correct"] = rows.answer.map(normalize_answer).eq(rows.target)
    rows["semantic_correct"] = rows.category.isin(["target", "same_meaning"]).astype("boolean")
    rows.loc[rows.category.isin(["unclassified", "ambiguous_response"]), "semantic_correct"] = pd.NA
    rows["context"] = rows.condition.str.rsplit("_", n=1).str[0]
    rows["cue"] = rows.condition.str.rsplit("_", n=1).str[-1]
    results = []
    for (state, context, dominance), group in rows.groupby(["state", "context", "dominance"]):
        wide = group.pivot(index=["item_id", "family"], columns="cue", values="exact_correct")
        if set(wide.columns) != {"none", "correct", "misleading"} or wide.isna().any().any():
            raise ValueError("each item needs all three cue conditions")
        paired = wide.reset_index().rename(columns={"correct": "cued"})
        paired["gain"] = paired.cued.astype(int) - paired["none"].astype(int)
        wrong = paired[~paired["none"]]
        entry = dict(state=state, context=context, dominance=dominance, n=len(paired),
                     endpoint="exact designated associate, secondary to reviewed semantic accuracy",
                     semantic_accuracy_status="needs_blind_response_review" if group.semantic_correct.isna().any() else "fully_classified",
                     unclassified=int(group.semantic_correct.isna().sum()),
                     exact_target_no_cue=float(paired["none"].mean()), exact_target_correct_cue=float(paired.cued.mean()),
                     exact_target_misleading=float(paired.misleading.mean()),
                     rescue_denominator=len(wrong), rescued=int(wrong.cued.sum()),
                     rescue_fraction=float(wrong.cued.mean()) if len(wrong) else None,
                     cue_gain=bootstrap_family_mean(paired, 'gain', n_boot),
                     categories=group.groupby("cue").category.value_counts().unstack(fill_value=0).to_dict())
        entry["semantic_accuracy_bounds"] = {cue: binary_response_bounds(g, np.ones(len(g)) / len(g))
                                              for cue, g in group.groupby("cue")}
        entry["semantic_rescue_bounds"] = homonym_rescue_bounds(group)
        if not group.semantic_correct.isna().any():
            semantic = group.pivot(index=["item_id", "family"], columns="cue", values="semantic_correct").astype(bool)
            wrong_semantic = semantic[~semantic["none"]]
            entry["semantic_accuracy"] = {cue: float(semantic[cue].mean()) for cue in ("none", "correct", "misleading")}
            entry["semantic_rescue"] = dict(denominator=len(wrong_semantic), rescued=int(wrong_semantic.correct.sum()),
                                             fraction=float(wrong_semantic.correct.mean()) if len(wrong_semantic) else None)
        results.append(entry)
    return results, rows


def homonym_contrast(rows, first, second, n_boot=2000, endpoint="semantic_correct"):
    """Paired difference of subordinate-minus-dominant impairment and of cue gains, per homonym family."""
    table = rows[rows.state.isin([first, second])].copy()
    results = {}
    for context, g in table.groupby("context"):
        wide = g.pivot(index="family", columns=["state", "dominance", "cue"], values=endpoint)
        expected = {(s, d, c) for s in (first, second) for d in ("dominant", "subordinate")
                    for c in ("none", "correct", "misleading")}
        if set(wide.columns) != expected or len(g) != len(wide) * len(expected):
            raise ValueError("all states, meanings and cue conditions are required for the contrast")
        if g[endpoint].isna().any():
            if endpoint != "semantic_correct":
                raise ValueError("missing numeric homonym evidence")
            sign = np.where(g.state == first, 1., -1.) / len(wide)
            subordinate = g.dominance.eq("subordinate").to_numpy()
            none, correct, misleading = [g.cue.eq(c).to_numpy() for c in ("none", "correct", "misleading")]
            weights = dict(subordinate_extra=sign * np.where(subordinate, -1., 1.) * none,
                           cue_gain=sign * subordinate * (correct.astype(float) - none),
                           misleading_cost=sign * subordinate * (none.astype(float) - misleading))
            results[context] = dict(status="needs_blind_response_review", unclassified=int(g[endpoint].isna().sum()), n_families=len(wide),
                                    classification_bounds={k: binary_response_bounds(g, w, n_boot) for k, w in weights.items()})
            continue
        wide = wide.astype(float)
        a, b = wide[first], wide[second]
        # Positive means first has the larger subordinate-specific deficit.
        extra = (a["dominant"]["none"] - a["subordinate"]["none"]) - (b["dominant"]["none"] - b["subordinate"]["none"])
        gain = (a["subordinate"]["correct"] - a["subordinate"]["none"]) - (b["subordinate"]["correct"] - b["subordinate"]["none"])
        miscue = (a["subordinate"]["none"] - a["subordinate"]["misleading"]) - (b["subordinate"]["none"] - b["subordinate"]["misleading"])
        paired = pd.DataFrame(dict(family=wide.index, subordinate_extra=extra.to_numpy(),
                                   cue_gain=gain.to_numpy(), misleading_cost=miscue.to_numpy()))
        results[context] = {col: bootstrap_family_mean(paired, col, n_boot)
                            for col in ("subordinate_extra", "cue_gain", "misleading_cost")}
    return results


def homonym_ledger_summary(result):
    """Save candidate evidence, actual-answer bounds and their distinct uncertainty definitions."""
    def metric(value, unit):
        return {**{k: v for k, v in value.items() if k != "n_families"},
                "unit": unit, "n": value["n_families"], "n_unit": "词义家族"}

    evidence, contrasts, answers, semantic = {}, {}, {}, {}
    units = dict(candidate_log_odds="nat，指定两候选之差", choice_logit="logit")
    for row in result["candidate_and_choice"]:
        name = f"{row['state']}/{row['context']}/{row['dominance']}"
        evidence.setdefault(row['readout'], {})[name] = {
            k: metric(row[k], units[row['readout']]) for k in ('none', 'correct_gain', 'misleading_cost')}
    for readout, comparisons in result["candidate_and_choice_contrasts"].items():
        contrasts[readout] = {pair: {context: {k: metric(v, units[readout]) for k, v in values.items()}
                                   for context, values in groups.items()} for pair, groups in comparisons.items()}
    for row in result['results']:
        name = f"{row['state']}/{row['context']}/{row['dominance']}"
        saved = {}
        for cue, bounds in row['semantic_accuracy_bounds'].items():
            for end in ('lower', 'upper'):
                saved[f'{cue}_accuracy_classification_{end}'] = dict(estimate=bounds[end], unit="proportion，分类界限非置信区间",
                    n=row['n'], n_unit="词义家族", status=row['semantic_accuracy_status'])
        for category, cues in row['categories'].items():
            for cue, count in cues.items():
                saved[f'{cue}_observed_{category}_fraction'] = dict(estimate=count / row['n'], unit="proportion，已判类别占全部回答",
                                                                  n=row['n'], n_unit="回答")
        answers[name] = saved
    for pair, contexts in result['contrasts'].items():
        semantic[pair] = {}
        for context, values in contexts.items():
            if 'classification_bounds' in values:
                semantic[pair][context] = {f'{name}_{end}': dict(estimate=bounds[end],
                    unit="proportion difference，分类界限非置信区间", n=values['n_families'], n_unit="词义家族")
                    for name, bounds in values['classification_bounds'].items() for end in ('lower', 'upper')}
                for name, bounds in values['classification_bounds'].items():
                    envelope = bounds.get('sampling_envelope', {})
                    for end in ('lower', 'upper'):
                        if end in envelope:
                            semantic[pair][context][f'{name}_sampling_envelope_{end}'] = dict(estimate=envelope[end],
                                unit='proportion difference，配对抽样与分类不确定性外界',
                                n=values['n_families'], n_unit='词义家族')
            else:
                semantic[pair][context] = {k: metric(v, 'proportion difference') for k, v in values.items()}
    return dict(meta={k: result[k] for k in ('status', 'provenance', 'source_fingerprint', 'analysis_sources',
                                           'auxiliary_sources', 'n_boot')},
                candidate_and_choice=evidence, candidate_and_choice_contrasts=contrasts,
                generated_answers=answers, generated_contrasts=semantic,
                matched_random=result.get('matched_random', {}))


def homonym_random_comparison(rows, states, endpoint='semantic_correct'):
    """Matched-group ranks retain semantic classification bounds. Each group uses the same intact answers."""
    controls = [s for s in states if s.get('kind') == 'matched_random']
    result = {}
    for target in sorted({s['target'] for s in controls}):
        names = [s['name'] for s in controls if s['target'] == target]
        comparisons = {name: homonym_contrast(rows, name, 'intact', 0, endpoint) for name in [target, *names]}
        saved = {}
        for context, metrics in comparisons[target].items():
            for metric in ('subordinate_extra', 'cue_gain', 'misleading_cost'):
                def bounds(name):
                    values = comparisons[name][context]
                    if 'classification_bounds' in values:
                        b = values['classification_bounds'][metric]
                        return b['lower'], b['upper']
                    value = values[metric]['estimate']
                    return value, value
                lower, upper = bounds(target)
                random = np.array([bounds(name) for name in names])
                guaranteed = int((random[:, 0] >= upper).sum())
                possible = int((random[:, 1] >= lower).sum())
                unit = ('proportion difference, classification bound' if endpoint == 'semantic_correct'
                        else {'candidate_log_odds': 'nat', 'choice_logit': 'logit'}[rows.readout.iloc[0]])
                saved[f'{context}/{metric}'] = dict(
                    target_lower=dict(estimate=lower, unit=unit), target_upper=dict(estimate=upper, unit=unit),
                    guaranteed_exceedances=dict(estimate=guaranteed, unit='matched groups'),
                    possible_exceedances=dict(estimate=possible, unit='matched groups'),
                    rank_lower=dict(estimate=(1+guaranteed)/(1+len(names)), unit='upper-tail empirical rank lower bound',
                                    n=len(names), n_unit='matched groups'),
                    rank_upper=dict(estimate=(1+possible)/(1+len(names)), unit='upper-tail empirical rank upper bound',
                                    n=len(names), n_unit='matched groups'),
                    rows=[dict(state=name, lower=float(lo), upper=float(hi)) for name, (lo, hi) in zip(names, random)])
        result[target] = saved
    return result


def property_response_diagnostics(labels, items, n_boot=2000, seed=1910):
    """Separate truth discrimination from endorsement after averaging both answer orders.

    AUC compares the true-response score on true and false claims. It does not recalibrate the
    original decision threshold. Resampling keeps a concept's two claims and every state together.
    """
    source = labels[labels.fmt == 'property']
    wide = source.pivot(index=['state', 'item_id'], columns='variant', values='margin')
    if set(wide.columns) != {0, 1, 2, 3} or wide.isna().any().any():
        raise ValueError('property diagnostics require both truths and both answer orders')
    frame = pd.DataFrame(dict(true_score=(wide[2]+wide[3])/2,
                              false_score=-(wide[0]+wide[1])/2)).reset_index()
    frame = frame.merge(items[['item_id', 'family']], on='item_id', validate='many_to_one')
    ids = sorted(frame.item_id.unique())
    if frame[['item_id', 'family']].drop_duplicates().family.duplicated().any():
        raise ValueError('paired property bootstrap requires one concept per family')
    arrays = {s: g.set_index('item_id').reindex(ids)[['true_score', 'false_score']].to_numpy()
              for s, g in frame.groupby('state')}
    if 'intact' not in arrays or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('all property states require the same intact concept pairs')
    n = len(ids)
    draws = np.random.default_rng(seed).integers(n, size=(n_boot, n))
    keys = ('true_accuracy', 'false_accuracy', 'accuracy', 'true_response_rate',
            'auc', 'within_concept_order', 'score_separation', 'true_score_offset')

    def statistics(a):
        true, false = a.T
        hit, reject = np.mean(true > 0), np.mean(false < 0)
        pair = true[:, None]-false[None, :]
        auc = np.mean((pair > 0)+.5*(pair == 0))
        return np.array([hit, reject, (hit+reject)/2, (hit+np.mean(false > 0))/2,
                         auc, np.mean(true > false), np.mean(true-false), np.mean((true+false)/2)])

    points = {s: statistics(a) for s, a in arrays.items()}
    boots = {s: np.array([statistics(a[ix]) for ix in draws]).reshape(n_boot, len(keys))
             for s, a in arrays.items()}

    def summary(point, boot):
        bounds = np.quantile(boot, [.025, .975], axis=0) if n_boot else np.full((2, len(keys)), np.nan)
        return {key: dict(estimate=float(point[j]), ci_low=float(bounds[0, j]) if n_boot else None,
                          ci_high=float(bounds[1, j]) if n_boot else None,
                          unit='logit' if j >= 6 else 'proportion', n=n, n_unit='概念对',
                          status='posthoc_paired_concept_bootstrap' if n_boot else 'point_only')
                for j, key in enumerate(keys)}

    results = dict(states={s: summary(p, boots[s]) for s, p in points.items()},
                   lesion_minus_intact={s: summary(p-points['intact'], boots[s]-boots['intact'])
                                        for s, p in points.items() if s != 'intact'})
    return results, frame


def read_runs(directories):
    """Union complete runs of the same battery, verifying every repeated state exactly.

    A multi-run plan is explicitly an analysis union. Each physical run keeps its own provenance.
    """
    if not directories:
        raise ValueError('at least one complete run is required')
    runs = [(Path(path), *read_run(path)) for path in directories]
    if len(runs) == 1:
        return runs[0][1:]
    base = runs[0][2]
    provenance = lambda plan: {k: v for k, v in plan['provenance'].items() if k not in ('commit', 'states_sha256')}
    settings = ('chat', 'batches', 'max_new_tokens', 'naming', 'labels', 'neutral_sequences')
    if 'naming_batches' in base:
        settings += ('naming_batches',)
    states, collected, sources = {}, {}, []
    for path, tables, plan in runs:
        if (provenance(plan) != provenance(base) or any(plan.get(k) != base[k] for k in settings)
                or plan.get('naming_batches') != base.get('naming_batches')):
            raise ValueError('cannot combine different run materials, means, prompts or scoring settings')
        sources.append(dict(path=str(path), fingerprint=plan['fingerprint'], provenance=plan['provenance']))
        for state in plan['states']:
            name = state['name']
            rows = {key: value[value.state == name].reset_index(drop=True) for key, value in tables.items()}
            if name in states:
                if state != states[name]:
                    raise ValueError(f'{name}: repeated state has a different intervention')
                for key, value in rows.items():
                    pd.testing.assert_frame_equal(collected[name][key], value, check_exact=True)
            else:
                states[name], collected[name] = state, rows
    combined = {key: pd.concat([value[key] for value in collected.values()], ignore_index=True)
                for key in runs[0][1]}
    plan = dict(kind='analysis_union', provenance=provenance(base), states=list(states.values()),
                source_runs=sources, **{k: base[k] for k in settings})
    plan['fingerprint'] = title_gap_scoring._fingerprint(plan)
    return combined, plan
