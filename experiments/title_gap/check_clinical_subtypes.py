"""Check whether a blinded, known-wrong response's unresolved subtype changes clinical endpoints."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True, help='unchanged clinical execution checkout')
    ap.add_argument('--analysis', type=Path, required=True, help='original frozen analysis with unknowns retained')
    ap.add_argument('--review', type=Path, required=True, help='all adjudicated categories, including unknowns')
    ap.add_argument('--classified', type=Path, required=True, help='classified rows supplied to the frozen analysis')
    ap.add_argument('--binary', type=Path, required=True, help='separate blinded incorrect judgements')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source / 'src'))
    from lesion import clinical_alignment as ca, runinfo, title_gap_analysis as ta

    original = json.loads((args.analysis / 'result.json').read_text(encoding='utf-8'))
    protocol = json.loads((source / 'docs/analysis-plan/title-gap-clinical.json').read_text(encoding='utf-8'))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if (commit != original['meta']['analysis_commit']
            or any(runinfo.sha256(source / p) != digest for p, digest in protocol['sources_sha256'].items())
            or runinfo.sha256(args.classified) != original['meta']['review_sha256']):
        ap.error('frozen analysis sources or original classifications differ')
    tables, plan = ta.read_runs([Path(r['path']) for r in original['meta']['source_runs']])
    if plan['fingerprint'] != original['meta']['source_fingerprint']:
        ap.error('clinical raw source differs from the strict analysis')
    review = pd.read_csv(args.review, keep_default_na=False)
    classified = pd.read_csv(args.classified, keep_default_na=False)
    pd.testing.assert_frame_equal(review[review.category != 'unclassified'].reset_index(drop=True), classified)
    rows = ca.classify_cued(tables['generated'], pd.read_csv(source / 'data/materials/m3/final_items.csv', keep_default_na=False), classified)
    binary = pd.read_csv(args.binary, keep_default_na=False)
    unknown = rows.category.eq('unclassified')
    if (binary.response_id.duplicated().any() or set(binary.correctness) != {'incorrect'}
            or set(binary.response_id) != set(rows.loc[unknown, 'response_id'])
            or set(review.loc[review.category == 'unclassified', 'response_id']) != set(binary.response_id)):
        ap.error('every unresolved subtype needs an independent, explicit incorrect judgement')

    args.out.mkdir(parents=True, exist_ok=False)
    rows.to_csv(args.out / 'responses-with-unresolved-subtype.csv', index=False)
    scenarios, profiles, uncued_inputs = {}, [], []
    for category in ('coordinate', 'superordinate', 'associate', 'phonological', 'unrelated'):
        assumed = review.copy()
        assumed.loc[assumed.category == 'unclassified', ['category', 'reason']] = [category, 'Sensitivity assumption only. Original subtype remains unresolved.']
        review_path = args.out / f'{category}-review.csv'
        assumed.to_csv(review_path, index=False)
        scenario = rows.copy()
        scenario.loc[unknown, 'category'] = category
        pd.testing.assert_series_equal(scenario.category.eq('correct'), rows.category.eq('correct'))
        values = ca.cue_profiles(scenario, n_boot=protocol['n_boot'], seed=protocol['seed'])
        for state, group in scenario[scenario.letters == 0].groupby('state'):
            values[state]['naming_error_profile'] = ta.fine_error_profile(group)
        profiles.append(values)
        uncued_inputs.append(scenario.loc[scenario.letters == 0, ['state', 'item_id', 'category']])
        meta = dict(original['meta'], review_sha256=runinfo.sha256(review_path),
                    classification_assumption=category, original_subtype='unclassified',
                    binary_judgement_sha256=runinfo.sha256(args.binary), sensitivity_only=True)
        result_path = args.out / f'{category}-result.json'
        result_path.write_text(json.dumps(dict(meta=meta, states=values), indent=2, allow_nan=False)+'\n', encoding='utf-8')
        scenarios[category] = dict(review=review_path.as_posix(), review_sha256=runinfo.sha256(review_path),
                                   result=result_path.as_posix(), result_sha256=runinfo.sha256(result_path))
    cue_invariant = all({s: {k: v for k, v in p[s].items() if k != 'naming_error_profile'} for s in p}
                        == {s: {k: v for k, v in profiles[0][s].items() if k != 'naming_error_profile'} for s in profiles[0]}
                        for p in profiles[1:])
    uncued_invariant = all(table.equals(uncued_inputs[0]) for table in uncued_inputs[1:])
    result = dict(status='subtype_sensitivity_complete', cue_endpoints_invariant=cue_invariant,
        uncued_classification_inputs_invariant=uncued_invariant,
        complete_profiles_invariant=all(p == profiles[0] for p in profiles[1:]),
        unknown_observations=rows.loc[unknown, ['response_id', 'state', 'item_id', 'letters']].to_dict('records'),
        sources={p.as_posix(): runinfo.sha256(p) for p in
            (args.analysis / 'result.json', args.review, args.classified, args.binary, Path(__file__).resolve())},
        frozen_source=source.as_posix(), frozen_commit=commit, scenarios=scenarios,
        note='Assumed subtypes are not new labels. Correctness vectors are identical in every scenario. '
             'If uncued categories are also identical, clinical marker resampling receives identical inputs.')
    (args.out / 'invariance.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k.endswith('_invariant')}))


if __name__ == '__main__':
    main()
