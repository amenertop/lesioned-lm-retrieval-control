"""Compare published group markers with actual dose curves, preserving missing profiles and source checks."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import clinical_alignment as ca, clinical_resampling as cr, runinfo, title_gap_analysis as ta


def operation(state):
    intervention = dict(gates=state['gates'], region=state.get('region') or 'all', weight_mask=state.get('weight_mask'))
    return hashlib.sha256(json.dumps(intervention, sort_keys=True).encode()).hexdigest()


def final_material_digest(path):
    """Bind the approved 120-concept final set, never an implicit development-file fallback."""
    approved = ROOT / 'data/materials/title_gap/fine_final/items.csv'
    if path is None or path.resolve() != approved.resolve():
        raise ValueError('explicit approved fine_final/items.csv is required')
    manifest = json.loads(approved.with_name('manifest.json').read_text(encoding='utf-8'))
    digest = runinfo.sha256(path)
    items = pd.read_csv(path, keep_default_na=False).query("split == 'test' and review == 'approved'")
    if (manifest['status'] != 'approved_for_execution' or manifest['items_sha256'] != digest
            or len(items) != 120 or items.family.nunique() != 120
            or items.frequency_bin.value_counts().to_dict() != {'low': 40, 'middle': 40, 'high': 40}):
        raise ValueError('final fine materials differ from the approved 120-concept set')
    return digest


def load_labels(paths, clinical_plan, items_path):
    shared = ('model', 'revision', 'dtype', 'means_sha256', 'split', 'tiny', 'moe')
    expected = {k: clinical_plan['provenance'][k] for k in shared}
    names = {operation(s): s['name'] for s in clinical_plan['states']}
    if len(names) != len(clinical_plan['states']):
        raise ValueError('clinical curve contains duplicate interventions')
    collected, sources, prompt = {}, [], None
    for path in paths:
        tables, plan = ta.read_run(path)
        provenance = plan['provenance']
        if ({k: provenance[k] for k in shared} != expected or provenance['battery'] != 'fine'
                or provenance['items_sha256'] != runinfo.sha256(items_path)):
            raise ValueError('fine and clinical runs must share model, means, precision and split')
        signature = {k: plan[k] for k in ('labels', 'batches', 'chat')}
        if prompt is not None and signature != prompt:
            raise ValueError('do not combine different synonym prompts or scoring batches')
        prompt = signature
        sources.append(dict(path=str(path), fingerprint=plan['fingerprint'], scoring=signature))
        for state in plan['states']:
            name = names.get(operation(state))
            if name is None:
                continue
            rows = tables['labels'][tables['labels'].state == state['name']]
            rows = rows.assign(state=name).sort_values(['item_id', 'fmt', 'variant']).reset_index(drop=True)
            if name in collected:
                pd.testing.assert_frame_equal(collected[name], rows, check_exact=True)
            else:
                collected[name] = rows
    if 'intact' not in collected:
        raise ValueError('the common intact synonym baseline is required')
    return pd.concat(collected.values(), ignore_index=True), sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clinical-run', type=Path, action='append', required=True)
    ap.add_argument('--fine-run', type=Path, action='append', required=True)
    ap.add_argument('--fine-items', type=Path, default=ROOT / 'data/materials/title_gap/fine/items.csv')
    ap.add_argument('--final-fine-items', type=Path, help='on development runs, bind the approved final material file')
    ap.add_argument('--review', type=Path)
    ap.add_argument('--curves', type=Path, required=True, help='CSV: model, curve, state')
    ap.add_argument('--frozen-selection', type=Path)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=1910)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.frozen_selection and args.final_fine_items:
        ap.error('final material transfer is fixed by the development selection')
    try:
        final_digest = final_material_digest(args.fine_items if args.frozen_selection else args.final_fine_items)
    except ValueError as error:
        ap.error(str(error))
    tables, plan = ta.read_runs(args.clinical_run)
    provenance = plan['provenance']
    clinical_path = ROOT / 'data/materials/m3/final_items.csv'
    fine_path = args.fine_items
    if provenance['battery'] != 'clinical_m3' or provenance['items_sha256'] != runinfo.sha256(clinical_path):
        ap.error('clinical materials do not match the scored battery')
    if (provenance['split'] == 'test') != bool(args.frozen_selection):
        ap.error('independent test profiles require a development selection, development profiles must select afresh')
    rows = ca.classify_cued(tables['generated'], pd.read_csv(clinical_path, keep_default_na=False),
                           pd.read_csv(args.review) if args.review else None)
    labels, sources = load_labels(args.fine_run, plan, fine_path)
    markers = ca.measured_marker_profiles(rows, labels, pd.read_csv(fine_path, keep_default_na=False))
    patients = ca.published_marker_profiles(ROOT / 'data/literature')
    curves = pd.read_csv(args.curves)
    predictions, coverage = ca.complete_curve_profiles(markers, curves)
    source_hashes = {name: runinfo.sha256(ROOT / 'data/literature' / name) for name in patients.source_file.unique()}
    source_hashes['clinical_calibration.json'] = runinfo.sha256(ROOT / 'data/literature/clinical_calibration.json')
    protocol = dict(version=2, method='measured states, training-family eligibility, shared curve, per-group severity',
        weighting='equal clinical group and marker family', ties='curve and state names',
        clinical_scoring={k: plan[k] for k in ('batches', 'chat', 'max_new_tokens')},
        fine_scoring={k: sources[0]['scoring'][k] for k in ('batches', 'chat')},
        implementation={p: runinfo.sha256(ROOT / p) for p in ('src/lesion/clinical_alignment.py',
            'src/lesion/clinical_resampling.py', 'src/lesion/m3_cued.py', 'src/lesion/m3_scoring.py',
            'src/lesion/fine_semantics.py', 'src/lesion/title_gap_scoring.py',
            'src/lesion/title_gap_analysis.py', 'experiments/title_gap/compare_clinical.py')})
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    operations = {s['name']: operation(s) for s in plan['states']}
    measurement = {k: provenance[k] for k in ('model', 'revision', 'dtype', 'means_sha256', 'tiny', 'moe')}
    fit, sensitivity, corners = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    status = 'incomplete_marker_coverage'
    selected = None
    if args.frozen_selection:
        frozen = json.loads(args.frozen_selection.read_text(encoding='utf-8'))
        tag = provenance.get('freeze_tag')
        if not tag or not tag.startswith(runinfo.ANALYSIS_TAG_PREFIX):
            ap.error('final clinical profiles require a recorded analysis freeze tag')
        relative = args.frozen_selection.resolve().relative_to(ROOT).as_posix()
        if json.loads(runinfo.git_bytes('show', f'{tag}:{relative}')) != frozen:
            ap.error('clinical selection differs from the version saved in the analysis freeze')
        if (frozen['meta']['status'] != 'development_leave_family_out' or frozen['meta']['split'] != 'dev'
                or frozen['meta']['measurement'] != measurement or frozen['meta']['patient_sources'] != source_hashes
                or frozen['meta']['curves_sha256'] != runinfo.sha256(args.curves)
                or frozen['meta'].get('final_fine_items_sha256') != final_digest
                or frozen['meta'].get('protocol_sha256') != protocol_hash):
            ap.error('frozen selection has different clinical targets or curve definitions')
        selected = pd.DataFrame(frozen['predictions'])
        selected_sensitivity = pd.DataFrame(frozen['figure_sensitivity'])
        required_states = set(selected.state) | set(selected_sensitivity.state)
        if selected.empty or any(operations.get(s) != frozen['meta']['state_operations'].get(s) for s in required_states):
            ap.error('frozen lesion states are missing or have changed')
        # A zero error denominator on final items stays missing. Never select a replacement dose.
        measured = curves.merge(markers[['state', 'marker', 'value']], on='state', validate='many_to_many')
        fit = ca.predict_frozen_profiles(patients, measured, selected)
        sensitivity, corners = ca.compare_figure_sensitivity(patients, measured, selected_sensitivity)
        if corners.to_dict('records') != frozen['figure_corners']:
            ap.error('frozen figure sensitivity definitions have changed')
        status = ('frozen_development_selection_on_independent_items' if
                  fit.status.eq('measured').all() and sensitivity.status.eq('measured').all() else
                  'incomplete_frozen_marker_coverage')
    elif set(predictions.model) == set(curves.model):
        fit = ca.compare_measured_profiles(patients, predictions)
        sensitivity, corners = ca.compare_figure_sensitivity(patients, predictions)
        status = ('development_leave_family_out' if fit.status.eq('measured').all() and
                  sensitivity.status.eq('measured').all() else 'incomplete_development_marker_coverage')
    args.out.mkdir(parents=True, exist_ok=True)
    patients.to_csv(args.out / 'patient-markers.csv', index=False)
    markers.to_csv(args.out / 'model-markers.csv', index=False)
    coverage.to_csv(args.out / 'coverage.csv', index=False)
    fit.to_csv(args.out / 'predictions.csv', index=False)
    sensitivity.to_csv(args.out / 'figure-sensitivity.csv', index=False)
    corners.to_csv(args.out / 'figure-corners.csv', index=False)
    uncertainty, draws, choices, point = cr.bootstrap_comparison(rows, labels,
        pd.read_csv(fine_path, keep_default_na=False), patients,
        pd.read_csv(ROOT / 'data/literature/jefferies2006_individual.csv', keep_default_na=False), curves,
        args.n_boot, args.seed, selected)
    pd.testing.assert_frame_equal(fit.reset_index(drop=True), point.reset_index(drop=True))
    draws.to_csv(args.out / 'sampling-draws.csv', index=False)
    choices.to_csv(args.out / 'sampling-selections.csv', index=False)
    payload = dict(meta=dict(status=status, split=provenance['split'], measurement=measurement, source_fingerprint=plan['fingerprint'],
        clinical_sources=plan.get('source_runs', [dict(path=str(args.clinical_run[0]), fingerprint=plan['fingerprint'])]),
        evaluation_type='published cross-study clinical group markers with model task proxies',
        protocol=protocol, protocol_sha256=protocol_hash,
        fine_sources=sources, fine_items_sha256=runinfo.sha256(fine_path),
        final_fine_items_sha256=final_digest,
        patient_sources=source_hashes, state_operations=operations,
        review_sha256=runinfo.sha256(args.review) if args.review else None,
        curves_sha256=runinfo.sha256(args.curves),
        selection_sha256=runinfo.sha256(args.frozen_selection) if args.frozen_selection else None,
        analysis_commit=runinfo.commit(), code_sha256={p: runinfo.sha256(ROOT / p) for p in
            ('src/lesion/clinical_alignment.py', 'src/lesion/clinical_resampling.py', 'experiments/title_gap/compare_clinical.py')}),
        sampling=uncertainty,
        coverage=coverage.to_dict('records'),
        predictions=fit.astype(object).where(pd.notna(fit), None).to_dict('records'),
        figure_sensitivity=sensitivity.astype(object).where(pd.notna(sensitivity), None).to_dict('records'),
        figure_corners=corners.to_dict('records'))
    (args.out / 'result.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(status, f'{coverage.complete.sum()}/{len(coverage)} complete curve states')


if __name__ == '__main__':
    main()
