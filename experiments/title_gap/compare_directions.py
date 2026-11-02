"""Build the five-family development scorecard from complete, operation-matched batteries."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import (clinical_alignment as ca, clinical_directions as cd, damage, legacy_analysis,
                    runinfo, scoring, semantic_diversity as semd, title_gap_analysis as ta)


def operation(state):
    return json.dumps(dict(gates=state['gates'], region=state.get('region') or 'all',
                           weight_mask=state.get('weight_mask')), sort_keys=True)


def align(plan, reference, names):
    fields = ('model', 'revision', 'dtype', 'moe', 'split', 'tiny')
    if any(plan['provenance'][key] != reference['provenance'][key] for key in fields):
        raise ValueError('batteries use different model conditions or splits')
    means = plan['provenance'].get('means_sha256')
    if means is None:
        means = {digest for path, digest in plan['provenance']['source_sha256'].items() if path.endswith('.pt')}
        if len(means) != 1:
            raise ValueError('legacy component means are ambiguous')
        means = means.pop()
    if means != reference['provenance']['means_sha256']:
        raise ValueError('component means differ between batteries')
    expected = {operation(state): state['name'] for state in reference['states'] if state['name'] in names}
    mapped = {state['name']: expected[operation(state)] for state in plan['states'] if operation(state) in expected}
    if len(set(mapped.values())) != len(mapped):
        raise ValueError('one battery contains duplicate interventions')
    return mapped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fine-run', type=Path, action='append', required=True)
    ap.add_argument('--fine-items', type=Path, required=True)
    ap.add_argument('--norms', type=Path, required=True)
    ap.add_argument('--control-run', type=Path)
    ap.add_argument('--homonym-run', type=Path, action='append')
    ap.add_argument('--homonym-review', type=Path)
    ap.add_argument('--lexicon', type=Path)
    ap.add_argument('--clinical-run', type=Path, action='append')
    ap.add_argument('--clinical-review', type=Path)
    ap.add_argument('--sd', required=True)
    ap.add_argument('--sa', required=True)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    tables, plan = ta.read_runs(args.fine_run)
    if (plan['provenance']['split'] != 'dev' or plan['provenance']['battery'] != 'fine'
            or plan['provenance']['tiny'] or plan['provenance']['items_sha256'] != runinfo.sha256(args.fine_items)):
        ap.error('this entry requires the real, complete fine development battery')
    names = {args.sd, args.sa, 'intact'}
    if len(names) != 3 or not names <= {s['name'] for s in plan['states']}:
        ap.error('two distinct lesioned fine states and intact are required')
    tables = {key: rows[rows.state.isin(names)] for key, rows in tables.items()}
    items = pd.read_csv(args.fine_items, keep_default_na=False).query("split == 'dev'")
    items = semd.match_words(items, semd.load_norms(args.norms))
    items['frequency'] = items.log_bnc_freq
    _, fine = ta.fine_profiles(tables, items, 0)
    sources = dict(fine=plan['fingerprint'])
    hashes = {str(path): runinfo.sha256(path) for path in (args.fine_items, args.norms)}
    control, homonyms, clinical = None, None, None
    conditions = {}

    if args.control_run:
        all_items = scoring.load_materials()[0]
        _, cp = legacy_analysis.analyze(args.control_run,
            pd.read_csv(ROOT / 'data/materials/m3_knowledge/items.csv'), all_items)
        mapping = align(cp, plan, names)
        for path, digest in cp['provenance']['source_sha256'].items():
            if '/data/materials/' in path:
                local = ROOT / ('data/materials/' + path.split('/data/materials/', 1)[1])
                if runinfo.sha256(local) != digest:
                    ap.error('legacy materials differ from the scored inputs')
        if names <= set(mapping.values()):
            raw = {new: pd.read_parquet(args.control_run / old / 'control.parquet') for old, new in mapping.items()}
            h = pd.DataFrame({key: damage.item_damage(raw['intact'], raw[state])
                              for key, state in [('h_sd', args.sd), ('h_sa', args.sa)]})
            control = damage.item_frame(h, all_items, damage.frequency_median(all_items))
            control = control[control.task.isin(damage.MATCHING)]
        sources['control'] = cp['fingerprint']
        conditions['control_scorer'] = cp['provenance']['scorer']

    if args.homonym_run:
        if not args.lexicon:
            ap.error('homonym data require the human meaning lexicon')
        ht, hp = ta.read_runs(args.homonym_run)
        path = ROOT / 'data/materials/title_gap/homonyms/items.csv'
        if hp['provenance']['battery'] != 'homonyms' or hp['provenance']['items_sha256'] != runinfo.sha256(path):
            ap.error('homonym material mismatch')
        mapping = align(hp, plan, names)
        lexicon = {(r['word'], r['answer']): r['senses'] for r in json.loads(args.lexicon.read_text(encoding='utf-8'))}
        review = pd.read_csv(args.homonym_review) if args.homonym_review else None
        _, rows = ta.homonym_recovery(ht['generated'], pd.read_csv(path, keep_default_na=False), lexicon, 0, review)
        if {args.sd, args.sa} <= set(mapping.values()):
            homonyms = rows[rows.state.isin(mapping)].copy()
            homonyms['state'] = homonyms.state.map(mapping)
        sources['homonyms'] = hp['fingerprint']
        hashes.update({str(p): runinfo.sha256(p) for p in (path, args.lexicon, args.homonym_review) if p})

    if args.clinical_run:
        ct, cp = ta.read_runs(args.clinical_run)
        path = ROOT / 'data/materials/m3/final_items.csv'
        if cp['provenance']['battery'] != 'clinical_m3' or cp['provenance']['items_sha256'] != runinfo.sha256(path):
            ap.error('clinical material mismatch')
        mapping = align(cp, plan, names)
        material = pd.read_csv(path, keep_default_na=False)
        rows = ca.classify_cued(ct['generated'], material)
        review = pd.read_csv(args.clinical_review) if args.clinical_review else None
        if review is not None:
            if set(review.response_id) - set(rows.loc[rows.category == 'unclassified', 'response_id']):
                ap.error('clinical review may not replace an automatic classification')
            rows = ca.classify_cued(ct['generated'], material, review)
        if {args.sd, args.sa} <= set(mapping.values()):
            clinical = rows[rows.state.isin(mapping)].copy()
            clinical['state'] = clinical.state.map(mapping)
        sources['clinical'] = cp['fingerprint']
        hashes.update({str(p): runinfo.sha256(p) for p in (path, args.clinical_review) if p})

    result = cd.compare(fine, control, homonyms, clinical, args.sd, args.sa, args.n_boot)
    literature = ROOT / 'data/literature/jefferies2006_markers.csv'
    hashes[str(literature)] = runinfo.sha256(literature)
    result['meta'] = dict(status='development_directional_comparison', sources=sources,
        source_sha256=hashes, measurement=plan['provenance'], conditions=conditions,
        state_operations={s['name']: json.loads(operation(s)) for s in plan['states'] if s['name'] in names},
        code_sha256={p: runinfo.sha256(ROOT / p) for p in ('src/lesion/clinical_directions.py',
            'src/lesion/fine_confirmation.py', 'src/lesion/title_gap_analysis.py', 'src/lesion/damage.py',
            'src/lesion/clinical_alignment.py', 'experiments/title_gap/compare_directions.py')},
        analysis_commit=runinfo.commit(), random_reference_status='not_measured_on_all_five_families')
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result['markers']).to_csv(args.out / 'markers.csv', index=False)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (args.out / 'ledger.json').write_text(json.dumps(cd.ledger_report(result), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['directions']))


if __name__ == '__main__':
    main()
