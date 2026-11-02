"""Clinical evidence, explicit denominators and held-out-family comparisons of measured lesion states."""
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def classify_cued(generated, items, review=None):
    """Complete-word judgements, with unknown answers left for condition-blind semantic review."""
    rows = generated.merge(items, on="item_id", validate="many_to_one")
    rows["letters"] = rows.condition.str.removeprefix("cue").astype(int)
    rows["n_letters"] = rows.target.str.len()
    categories = []
    for r in rows.to_dict("records"):
        word = str(r["answer"]).lower().strip().strip('.!,?"\'')
        if word in ("", "cannot tell", "unknown", "i don't know"):
            category = "omission"
        elif not re.fullmatch("[a-z]+", word):
            category = "format_invalid"
        else:
            category = "unclassified"
            for role in ("target", "coordinate", "superordinate", "associate", "unrelated"):
                candidate = r[role].lower()
                forms = {candidate, candidate + "s", candidate + "es"}
                if candidate.endswith("s"):
                    forms.add(candidate[:-1])
                if word in forms:
                    category = "correct" if role == "target" else role
                    break
        categories.append(category)
    rows["category"] = categories
    rows["response_id"] = [hashlib.sha256(json.dumps([r["item_id"], r["answer"]]).encode()).hexdigest()[:16]
                           for r in rows.to_dict("records")]
    if review is not None:
        allowed = {"correct", "coordinate", "superordinate", "associate", "unrelated", "phonological", "omission", "format_invalid"}
        if review.response_id.duplicated().any() or set(review.category) - allowed or set(review.response_id) - set(rows.response_id):
            raise ValueError("invalid clinical response judgements")
        judged = rows.response_id.map(review.set_index("response_id").category)
        rows.loc[judged.notna(), "category"] = judged[judged.notna()]
    return rows


def cue_profiles(rows, n_boot=10000, seed=1910):
    """Stop at the first correct answer, as in progressive clinical cueing. Keep independent-cue reversals."""
    results = {}
    for state, group in rows.groupby("state"):
        if group.duplicated(["item_id", "letters"]).any():
            raise ValueError("duplicate clinical cue observation")
        for _, item in group.groupby("item_id"):
            if item.n_letters.nunique() != 1 or item.n_letters.iloc[0] < 3 or set(item.letters) != set(range(item.n_letters.iloc[0])):
                raise ValueError("incomplete progressive cue sequence")
        if (group.category == "unclassified").any():
            results[state] = dict(status="needs_blind_response_review", unclassified=int((group.category == "unclassified").sum()))
            continue
        values = []
        for _, item in group.sort_values("letters").groupby("item_id", sort=True):
            correct = item.category.eq("correct").to_numpy()
            values.append((correct[0], correct[:3].any(), correct.any(), correct[2], correct[-1]))
        x = np.asarray(values, dtype=float)
        n = len(x)
        rng = np.random.default_rng(seed)
        draws = rng.integers(n, size=(n_boot, n))

        def estimate(value, denominator=None):
            denominator = np.ones(n) if denominator is None else denominator
            total = denominator.sum()
            point = float(value.sum() / total) if total else None
            if n_boot and total:
                d = denominator[draws].sum(1)
                boot = value[draws].sum(1) / np.where(d > 0, d, np.nan)
                finite = boot[np.isfinite(boot)]
                ci = np.quantile(finite, [.025, .975]).tolist() if len(finite) >= .9 * n_boot else [None, None]
            else:
                ci = [None, None]
            return dict(estimate=point, numerator=int(value.sum()), ci_low=ci[0], ci_high=ci[1],
                        n=int(total), unit="proportion")

        wrong = 1 - x[:, 0]
        results[state] = dict(status="fully_classified", n=n,
            accuracy_none=estimate(x[:, 0]), accuracy_two=estimate(x[:, 1]), accuracy_final=estimate(x[:, 2]),
            rescue_two=estimate(wrong * x[:, 1], wrong), rescue_final=estimate(wrong * x[:, 2], wrong),
            independent_two=estimate(x[:, 3]), independent_final=estimate(x[:, 4]),
            initially_correct_lost_at_two=estimate(x[:, 0] * (1-x[:, 3]), x[:, 0]))
    return results


def relative_loss(score, healthy, chance):
    if not np.isfinite([score, healthy, chance]).all() or healthy <= chance:
        raise ValueError("a finite healthy baseline above chance is required")
    return (healthy - score) / (healthy - chance)


def evidence_audit(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.csv")):
        table = pd.read_csv(path, keep_default_na=False)
        for i, row in enumerate(table.to_dict("records")):
            value = pd.to_numeric(row.get("value", ""), errors="coerce")
            method = row.get("how", "")
            individual = bool(row.get("patient_id", ""))
            rows.append(dict(file=path.name, source_row=i + 2, marker=row.get("marker", ""),
                             measure=row.get("measure", ""), group=row.get("group", ""),
                             value=row.get("value", ""), unit=row.get("unit", ""), n=row.get("n", ""),
                             source=row.get("source", ""), how=method,
                             level="individual" if individual else "group_or_test_statistic",
                             patient_id=row.get("patient_id", ""), task=row.get("task", ""),
                             denominator=row.get("denominator", ""), missing_reason=row.get("missing_reason", ""),
                             numeric=bool(pd.notna(value)), figure_estimate="figure" in method,
                             model_scale_mapping="not_established", individual_fit=False,
                             reason="matched model tasks and calibration required" if individual else
                                    "group summaries need matched task, healthy baseline and chance or a rank analysis"))
    return pd.DataFrame(rows)


def patient_tasks(table):
    """Keep missing observations explicit and calibrate only tasks with a documented chance level."""
    if table.duplicated(["patient_id", "task", "measure"]).any():
        raise ValueError("duplicate patient observations")
    out = table.copy()
    for col in ("value", "healthy_mean", "chance_count", "maximum"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if ((out.value.isna()) & (out.missing_reason == "")).any():
        raise ValueError("missing patient scores require the published missingness code")
    out["relative_loss"] = np.nan
    ix = out[["value", "healthy_mean", "chance_count"]].notna().all(axis=1)
    if (out.loc[ix, "healthy_mean"] <= out.loc[ix, "chance_count"]).any():
        raise ValueError("healthy scores must exceed chance")
    out.loc[ix, "relative_loss"] = ((out.loc[ix, "healthy_mean"] - out.loc[ix, "value"]) /
                                     (out.loc[ix, "healthy_mean"] - out.loc[ix, "chance_count"]))
    # Values below zero or above one are retained: patients may exceed the healthy mean or fall below chance.
    return out


def patient_group_profiles(table, n_boot=10000, seed=1910):
    """Resample published patient rows, keeping tasks paired within each patient and missing cells explicit."""
    if table.duplicated(["patient_id", "task", "measure"]).any():
        raise ValueError("duplicate patient observations")
    rng = np.random.default_rng(seed)
    summaries, draws = {}, {}
    for group, data in table.groupby("group", sort=True):
        matrix = data.pivot(index="patient_id", columns=["task", "measure"], values="value")
        values = matrix.to_numpy(float)
        ix = rng.integers(len(values), size=(n_boot, len(values)))
        sampled = values[ix]
        counts = np.isfinite(sampled).sum(axis=1)
        means = np.divide(np.nansum(sampled, axis=1), counts,
                          out=np.full(counts.shape, np.nan), where=counts > 0)
        for j, (task, measure) in enumerate(matrix.columns):
            source = data[(data.task == task) & (data.measure == measure)]
            for field in ("unit", "denominator"):
                if source[field].nunique() != 1:
                    raise ValueError("inconsistent clinical scale")
            finite = means[:, j][np.isfinite(means[:, j])]
            key = task, measure
            summaries[group, key] = dict(group=group, task=task, measure=measure,
                estimate=float(np.nanmean(values[:, j])), ci_low=float(np.percentile(finite, 2.5)),
                ci_high=float(np.percentile(finite, 97.5)), n=int(np.isfinite(values[:, j]).sum()),
                unit=source.unit.iloc[0], denominator=source.denominator.iloc[0])
            draws[group, key] = means[:, j]
    rows = list(summaries.values())
    shared = {key for group, key in summaries if group == "SD"} & {key for group, key in summaries if group == "SA"}
    for key in sorted(shared):
        sd, sa = summaries["SD", key], summaries["SA", key]
        if (sd["unit"], sd["denominator"]) != (sa["unit"], sa["denominator"]):
            raise ValueError("clinical group units differ")
        delta = draws["SA", key] - draws["SD", key]
        delta = delta[np.isfinite(delta)]
        rows.append(dict(group="SA_minus_SD", task=key[0], measure=key[1],
            estimate=sa["estimate"]-sd["estimate"], ci_low=float(np.percentile(delta, 2.5)),
            ci_high=float(np.percentile(delta, 97.5)), n=sd["n"]+sa["n"],
            unit=sd["unit"], denominator=sd["denominator"]))
    return pd.DataFrame(rows)


def published_marker_profiles(directory, n_boot=10000, seed=1910):
    """Align three published marker families. Figure precision is not a patient confidence interval.

    Cue rescue is the ratio of group-mean recovered items to group-mean initial errors, not the mean
    of unavailable individual rescue ratios. Error compositions retain the printed rounded proportions.
    """
    directory = Path(directory)
    rows = []
    frequency = pd.read_csv(directory / "hoffman2011_frequency.csv", keep_default_na=False)
    cueing = pd.read_csv(directory / "jefferies2008_cueing.csv", keep_default_na=False)
    calibration = json.loads((directory / 'clinical_calibration.json').read_text(encoding='utf-8'))
    healthy = float(calibration['frequency']['healthy_accuracy'])
    chance = 1 / int(calibration['frequency']['n_choices'])
    frequency_precision = float(calibration['frequency']['figure_mean_tolerance'])
    cue_precision = float(calibration['cueing']['figure_mean_tolerance'])
    if not 0 <= chance < healthy <= 1 or not 0 < frequency_precision < 1 or not 0 < cue_precision < 1:
        raise ValueError('invalid published clinical calibration')

    def observation(table, group, measure):
        source = table[(table.group == group) & (table.measure == measure)]
        if len(source) != 1:
            raise ValueError(f"one published {group}/{measure} observation is required")
        row = source.iloc[0]
        return float(row.value), row, int(source.index[0]) + 2

    for group in ("SD", "SA"):
        gap, row, source_row = observation(frequency, group,
            "frequency effect, high minus low frequency, mean over imageability")
        high_cells, low_cells = [], []
        for imageability in ('high', 'medium', 'low'):
            high, _, _ = observation(frequency, group, f'synonym judgement, high frequency, {imageability} imageability')
            low, _, _ = observation(frequency, group, f'synonym judgement, low frequency, {imageability} imageability')
            high_cells.append(high)
            low_cells.append(low)
        if not np.isclose(gap, np.mean(high_cells)-np.mean(low_cells), atol=.0005, rtol=0):
            raise ValueError('published frequency difference disagrees with its rounded source cells')
        distance = healthy-chance
        rows.append(dict(group=group, marker="frequency_gap", family="frequency", value=gap / distance,
            raw_value=gap, denominator=distance, denominator_kind="healthy_accuracy_minus_chance",
            denominator_status="documented_calibration",
            ci_low=None, ci_high=None, sensitivity_low=(gap-2*frequency_precision)/distance,
            sensitivity_high=(gap+2*frequency_precision)/distance,
            high_accuracy=float(np.mean(high_cells)), low_accuracy=float(np.mean(low_cells)),
            healthy_accuracy=healthy, chance_accuracy=chance, figure_mean_tolerance=frequency_precision,
            calibration_file='clinical_calibration.json',
            sensitivity="each frequency mean shifted by the documented tolerance in opposite directions",
            n=int(row.n), unit="normalized_accuracy_gap", source_file="hoffman2011_frequency.csv",
            source_rows=str(source_row), source=row.source, how=row.how,
            calibration_source="Hoffman 2011 author manuscript pp. 13, 17: three choices, healthy accuracy .95"))
        uncued, baseline, baseline_row = observation(cueing, group, "uncued naming")
        for endpoint, measure in (("two", "naming with a two-phoneme cue"), ("final", "naming at the final cue")):
            cued, row, source_row = observation(cueing, group, measure)
            if not 0 <= uncued <= cued <= 1 or uncued == 1 or row.n != baseline.n:
                raise ValueError("progressive clinical cueing requires paired group endpoints and initial errors")
            benefit_measure = 'benefit of a two-phoneme cue' if endpoint == 'two' else 'benefit of the final cue'
            benefit, benefit_row, _ = observation(cueing, group, benefit_measure)
            if benefit_row.n != row.n or not np.isclose(benefit, cued-uncued, atol=.0005, rtol=0):
                raise ValueError('published cue benefit disagrees with its paired source endpoints')
            low = max(0., (cued-cue_precision-(uncued+cue_precision)) / (1-(uncued+cue_precision)))
            high = min(1., (cued+cue_precision-(uncued-cue_precision)) / (1-(uncued-cue_precision)))
            rows.append(dict(group=group, marker="rescue_"+endpoint, family="cueing", value=(cued-uncued)/(1-uncued),
                raw_value=cued-uncued, denominator=1-uncued, denominator_kind="group_mean_initial_error_proportion",
                denominator_status="derived_from_group_mean", uncued_accuracy=uncued, cued_accuracy=cued,
                figure_mean_tolerance=cue_precision, calibration_file='clinical_calibration.json',
                ci_low=None, ci_high=None, sensitivity_low=low, sensitivity_high=high,
                sensitivity="uncued and cued group means each shifted by the documented tolerance",
                n=int(row.n), unit="proportion", source_file="jefferies2008_cueing.csv",
                source_rows=f"{baseline_row}|{source_row}", source=row.source, how=row.how,
                calibration_source="progressive phoneme cueing, ratio of group means"))

    table = pd.read_csv(directory / "jefferies2006_individual.csv", keep_default_na=False)
    selected = table[(table.task == "naming_error_analysis") &
                     table.measure.isin(["coordinate", "superordinate", "associative"])]
    if set(selected.denominator) != {"single_word_semantic_errors"}:
        raise ValueError("clinical error composition requires the single-word semantic-error denominator")
    profiles = patient_group_profiles(patient_tasks(table), n_boot, seed)
    profiles = profiles[(profiles.task == 'naming_error_analysis') & profiles.measure.isin(selected.measure.unique())]
    for row in profiles[profiles.group.isin(["SD", "SA"])].itertuples():
        source = selected[(selected.group == row.group) & (selected.measure == row.measure)]
        rows.append(dict(group=row.group, marker="error_"+row.measure.replace("associative", "associate"),
            family="semantic_errors", value=row.estimate, raw_value=row.estimate, denominator=None,
            denominator_kind="within_patient_single_word_semantic_errors_then_patient_mean",
            denominator_status="counts_not_reported_patient_proportions_available",
            ci_low=row.ci_low, ci_high=row.ci_high, sensitivity_low=None, sensitivity_high=None,
            sensitivity="printed rounded proportions retained without renormalization",
            n=row.n, unit="proportion", source_file="jefferies2006_individual.csv",
            source_rows="|".join(str(i+2) for i in source.index), source="Table 5, p. 2141",
            how="patient table, patient bootstrap", calibration_source="no error counts reconstructed"))
    return pd.DataFrame(rows)


def figure_scenarios(patients):
    """Enumerate figure-precision corners, sharing the uncued baseline between each group's endpoints."""
    groups = ['SD', 'SA']
    if set(patients.group) != set(groups) or patients.duplicated(['group', 'marker']).any():
        raise ValueError('one complete marker profile per clinical group is required')
    source = patients.set_index(['group', 'marker'])
    for i, signs in enumerate(itertools.product((-1, 1), repeat=4*len(groups))):
        adjusted = source.copy()
        endpoints = []
        for j, group in enumerate(groups):
            frequency, none, two, final = signs[4*j:4*j+4]
            f = source.loc[(group, 'frequency_gap')]
            adjusted.loc[(group, 'frequency_gap'), 'value'] = f['sensitivity_low' if frequency < 0 else 'sensitivity_high']
            precision = source.loc[(group, 'rescue_two'), 'figure_mean_tolerance']
            if precision != source.loc[(group, 'rescue_final'), 'figure_mean_tolerance']:
                raise ValueError('cue endpoints require the same documented transcription tolerance')
            u = source.loc[(group, 'rescue_two'), 'uncued_accuracy'] + none*precision
            c2 = source.loc[(group, 'rescue_two'), 'cued_accuracy'] + two*precision
            cf = source.loc[(group, 'rescue_final'), 'cued_accuracy'] + final*precision
            if not 0 <= u <= c2 <= cf <= 1 or u == 1:
                raise ValueError('figure perturbations must preserve progressive cue endpoints')
            for endpoint, value in [('two', c2), ('final', cf)]:
                adjusted.loc[(group, 'rescue_'+endpoint), 'value'] = (value-u)/(1-u)
            endpoints.append(dict(group=group, frequency_gap=adjusted.loc[(group, 'frequency_gap'), 'value'],
                                  uncued_accuracy=u, two_accuracy=c2, final_accuracy=cf))
        yield f'corner_{i:03d}', adjusted.reset_index()[['group', 'marker', 'family', 'value']], endpoints


def compare_figure_sensitivity(patients, predictions, selection=None):
    """Refit development at every coherent corner, or transfer each corner's frozen states to final items."""
    fits, corners = [], []
    scenarios = list(figure_scenarios(patients))
    if selection is not None and set(selection.scenario) != {name for name, _, _ in scenarios}:
        raise ValueError('frozen sensitivity must retain every development figure scenario')
    for name, profile, endpoints in scenarios:
        fit = (compare_measured_profiles(profile, predictions) if selection is None else
               predict_frozen_profiles(profile, predictions, selection[selection.scenario == name]))
        fits.append(fit.assign(scenario=name))
        corners.extend(dict(scenario=name, **row) for row in endpoints)
    return pd.concat(fits, ignore_index=True), pd.DataFrame(corners)


def measured_marker_profiles(clinical_rows, labels, items):
    """Use observed cue recovery, uncued semantic errors and a fixed intact synonym baseline.

    labels use the clinical state names after the caller has checked identical interventions and prompts.
    Missing judgements or zero error denominators remain missing, with reasons in the returned table.
    """
    from .title_gap_analysis import fine_error_profile, fine_label_accuracy

    accuracy = fine_label_accuracy(labels).query("fmt == 'synonym'").merge(
        items[['item_id', 'frequency_bin']], on='item_id', validate='many_to_one')
    baseline = accuracy[accuracy.state == 'intact']
    if baseline.empty or any(set(part.item_id) != set(baseline.item_id) for _, part in accuracy.groupby('state')):
        raise ValueError('all synonym states require the same concepts and an intact baseline')
    healthy = float(baseline.correct_response.mean())
    if healthy <= .5:
        raise ValueError('synonym baseline must exceed the two-choice chance level')
    if any(not {'low', 'high'} <= set(part.frequency_bin) for _, part in accuracy.groupby('state')):
        raise ValueError('frequency comparison needs both prespecified frequency bins')
    cues = cue_profiles(clinical_rows, n_boot=0)
    rows = []
    for state, group in clinical_rows.groupby('state', sort=True):
        subset = accuracy[accuracy.state == state]
        means = subset.groupby('frequency_bin').correct_response.mean()
        gap = float(means['high']-means['low']) if not subset.empty else None
        rows.append(dict(state=state, marker='frequency_gap', family='frequency',
            value=gap/(healthy-.5) if gap is not None else None, raw_value=gap,
            denominator=healthy-.5, n=int(subset.frequency_bin.isin(['high', 'low']).sum()),
            n_contrast=int(subset.frequency_bin.isin(['high', 'low']).sum()), n_baseline=len(baseline),
            n_high=int(subset.frequency_bin.eq('high').sum()), n_low=int(subset.frequency_bin.eq('low').sum()),
            high_accuracy=float(means['high']) if gap is not None else None,
            low_accuracy=float(means['low']) if gap is not None else None,
            healthy_accuracy=healthy, chance_accuracy=.5,
            calibration_source='same-run intact synonym accuracy, two-choice order-averaged task',
            status='measured' if gap is not None else 'missing_same_state_synonym_run'))
        for endpoint in ('two', 'final'):
            summary = cues[state].get('rescue_'+endpoint, {})
            value = summary.get('estimate')
            rows.append(dict(state=state, marker='rescue_'+endpoint, family='cueing', value=value,
                raw_value=summary.get('numerator'), denominator=summary.get('n'), n=group.item_id.nunique(),
                rescued_count=summary.get('numerator'), initial_error_count=summary.get('n'), rescue_rate=value,
                uncued_accuracy=cues[state].get('accuracy_none', {}).get('estimate'),
                cued_accuracy=cues[state].get('accuracy_'+endpoint, {}).get('estimate'),
                status='measured' if value is not None else ('no_initial_errors' if summary else cues[state]['status'])))
        errors = fine_error_profile(group[group.letters == 0])
        for category, value in errors['single_word_semantic_proportions'].items():
            rows.append(dict(state=state, marker='error_'+category, family='semantic_errors', value=value,
                raw_value=errors['single_word_semantic_counts'][category], denominator=errors['semantic_errors'], n=errors['n'],
                status='measured' if value is not None else ('no_semantic_errors' if not errors['unclassified'] else errors['status'])))
    return pd.DataFrame(rows)


def complete_curve_profiles(markers, curves):
    """Keep missing markers explicit. Each fold will require only its training markers."""
    if markers.empty or curves.empty:
        raise ValueError('measured markers and explicit curves are required')
    if curves.duplicated(['model', 'curve', 'state']).any() or markers.duplicated(['state', 'marker']).any():
        raise ValueError('duplicate curve state or measured marker')
    names = set(markers.marker)
    rows, kept = [], []
    for row in curves.itertuples(index=False):
        profile = markers[markers.state == row.state]
        missing = sorted(names - set(profile.marker) | set(profile.loc[profile.value.isna(), 'marker']))
        reasons = sorted(set(profile.loc[profile.value.isna(), 'status']))
        rows.append(dict(model=row.model, curve=row.curve, state=row.state, complete=not missing,
                         missing='|'.join(missing), reason='|'.join(reasons) or ('state_not_measured' if missing else '')))
        completed = profile.set_index('marker').reindex(sorted(names)).reset_index()
        kept.append(completed[['marker', 'value']].assign(state=row.state, model=row.model, curve=row.curve))
    columns = ['state', 'marker', 'value', 'model', 'curve']
    return pd.concat(kept, ignore_index=True)[columns] if kept else pd.DataFrame(columns=columns), pd.DataFrame(rows)


def compare_profiles(frame, single_ratio=1.0):
    """Legacy linear approximation, not the measured-trajectory clinical comparison.

    Columns: marker, family, observed, retrieval, control. single_ratio fixes the control/retrieval mixture
    using independent calibration data. The single-axis fit has one nonnegative severity. The two-axis fit
    has two nonnegative severities. No per-marker scaling or test-fold fitting is allowed.
    Returns every held-out error, not a claim that any particular dynamical clinical model was defeated.
    """
    if frame.marker.duplicated().any() or frame.family.nunique() < 3:
        raise ValueError("distinct markers from at least three families are required")
    if not np.isfinite(single_ratio) or single_ratio < 0:
        raise ValueError("single_ratio must be a finite nonnegative calibration value")
    y = frame.observed.to_numpy(float)
    x = frame[["retrieval", "control"]].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("missing/non-finite clinical or model scores")
    rows = []
    for family in frame.family.unique():
        test = (frame.family == family).to_numpy()
        train = ~test
        if train.sum() < 3 or np.linalg.matrix_rank(x[train]) < 2:
            raise ValueError(f"{family}: training markers do not identify two axes")
        axis = (x[:, 0] + single_ratio * x[:, 1])[:, None]
        one = nnls(axis[train], y[train])[0]
        two = nnls(x[train], y[train])[0]
        pred1, pred2 = (axis[test] @ one), (x[test] @ two)
        for r, a, b in zip(frame.loc[test].itertuples(), pred1, pred2):
            rows.append(dict(marker=r.marker, family=family, observed=r.observed, single=float(a), dual=float(b),
                             single_error=float((a - r.observed) ** 2), dual_error=float((b - r.observed) ** 2),
                             single_severity=float(one[0]), retrieval_severity=float(two[0]), control_severity=float(two[1])))
    return pd.DataFrame(rows)


def compare_measured_profiles(patients, predictions):
    """Leave whole marker families out while selecting among actually measured dose states.

    patients: group, marker, family, value. predictions: model, curve, state, marker, value.
    A curve is shared by the clinical groups, but each group selects its own severity/state. A two-axis
    model supplies an actual measured grid as its curve. No interpolation or per-marker scaling is fitted.
    All markers must already share the documented clinical/model scale. Selection uses training families
    only, with equal weight per family and per patient group. State names break exact ties deterministically.
    """
    if patients.duplicated(["group", "marker"]).any() or predictions.duplicated(["model", "curve", "state", "marker"]).any():
        raise ValueError("duplicate clinical or model marker")
    if patients.groupby("marker").family.nunique().max() != 1 or patients.family.nunique() < 3:
        raise ValueError("each marker needs one family, with at least three families")
    markers = sorted(patients.marker.unique())
    groups = sorted(patients.group.unique())
    if len(groups) < 2 or any(set(g.marker) != set(markers) for _, g in patients.groupby("group")):
        raise ValueError("clinical groups must share all markers")
    if set(predictions.marker) != set(markers) or any(set(g.marker) != set(markers) for _, g in
            predictions.groupby(["model", "curve", "state"])):
        raise ValueError("every measured state must contain the same complete marker set")
    if not np.isfinite(patients.value.to_numpy(float)).all() or np.isinf(predictions.value.to_numpy(float)).any():
        raise ValueError("clinical targets must be finite and model scores cannot be infinite")
    families = patients.drop_duplicates("marker").set_index("marker").family.reindex(markers)
    observed = patients.pivot(index="group", columns="marker", values="value").reindex(index=groups, columns=markers)
    rows = []
    for held_family in sorted(families.unique()):
        held = families.eq(held_family).to_numpy()
        train_families = families[~held]
        weights = (1 / train_families.map(train_families.value_counts())).to_numpy(float)
        weights = weights / weights.sum()
        for model, candidates in predictions.groupby("model", sort=True):
            fits = []
            for curve, part in candidates.groupby("curve", sort=True):
                matrix = part.pivot(index="state", columns="marker", values="value").sort_index().reindex(columns=markers)
                matrix = matrix.loc[matrix.iloc[:, ~held].notna().all(axis=1)]
                if matrix.empty:
                    continue
                values = matrix.to_numpy(float)
                errors = ((values[None, :, ~held] - observed.to_numpy(float)[:, None, ~held]) ** 2 * weights).sum(2)
                best = errors.argmin(1)
                fits.append((float(errors[np.arange(len(groups)), best].mean()), curve, matrix, best))
            if not fits:
                for group in groups:
                    for marker in families.index[held]:
                        rows.append(dict(model=model, group=group, family=held_family, marker=marker,
                            observed=float(observed.loc[group, marker]), predicted=np.nan, error=np.nan,
                            curve=None, state=None, training_error=np.nan,
                            training_families=int(families.nunique()-1), status='missing_training_markers'))
                continue
            fit_loss, curve, matrix, best = min(fits, key=lambda x: (x[0], x[1]))
            for group, state_index in zip(groups, best):
                state = matrix.index[state_index]
                for marker in families.index[held]:
                    value, prediction = float(observed.loc[group, marker]), float(matrix.loc[state, marker])
                    rows.append(dict(model=model, group=group, family=held_family, marker=marker,
                        observed=value, predicted=prediction, error=(prediction-value)**2,
                        curve=curve, state=state, training_error=fit_loss,
                        training_families=int(families.nunique()-1),
                        status='measured' if np.isfinite(prediction) else 'missing_held_out_marker'))
    return pd.DataFrame(rows)


def predict_frozen_profiles(patients, predictions, selection):
    """Transfer development-selected states to independent items without refitting to their scores."""
    keys = ['model', 'group', 'family', 'marker']
    if selection.duplicated(keys).any() or predictions.duplicated(['model', 'curve', 'state', 'marker']).any():
        raise ValueError('duplicate frozen prediction')
    expected = {(model, row.group, row.family, row.marker) for model in predictions.model.unique()
                for row in patients.itertuples()}
    if set(selection[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('frozen selection must retain every model, clinical group and held-out marker')
    if (selection.groupby(['model', 'family']).curve.nunique().ne(1).any()
            or selection.groupby(['model', 'group', 'family']).state.nunique().ne(1).any()
            or selection[['curve', 'state']].isna().any().any()
            or not selection.training_families.eq(patients.family.nunique()-1).all()
            or not np.isfinite(selection.training_error.to_numpy(float)).all()
            or selection.training_error.lt(0).any()):
        raise ValueError('frozen selection must share each curve and retain each group severity within a fold')
    chosen = selection[keys+['curve', 'state', 'training_error', 'training_families']]
    actual = patients[['group', 'family', 'marker', 'value']].rename(columns={'value': 'observed'})
    result = chosen.merge(actual, on=['group', 'family', 'marker'], how='left', validate='many_to_one')
    result = result.merge(predictions.rename(columns={'value': 'predicted'}),
                          on=['model', 'curve', 'state', 'marker'], how='left', validate='many_to_one')
    if not np.isfinite(result.observed.to_numpy(float)).all():
        raise ValueError('all frozen clinical targets must remain available')
    result['status'] = np.where(np.isfinite(result.predicted), 'measured', 'missing_held_out_marker')
    result['error'] = (result.predicted-result.observed)**2
    return result
