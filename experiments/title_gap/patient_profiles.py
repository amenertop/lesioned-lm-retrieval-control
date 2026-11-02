"""Summarize the transcribed individual clinical tables, without inventing missing patient measurements."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import clinical_alignment, runinfo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1910)
    args = ap.parse_args()
    source = ROOT / "data/literature/jefferies2006_individual.csv"
    patients = clinical_alignment.patient_tasks(pd.read_csv(source, keep_default_na=False))
    profiles = clinical_alignment.patient_group_profiles(patients, args.n_boot, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.out / "patient-profiles.csv", index=False)
    result = dict(meta=dict(source=str(source.relative_to(ROOT)), source_sha256=runinfo.sha256(source),
                           analysis_commit=runinfo.commit(), n_boot=args.n_boot, seed=args.seed,
                           status="published_patient_data_reanalysis", model_fit=False,
                           missing="frequency and cueing lack individual measurements in this cohort",
                           code_sha256=runinfo.sha256(ROOT / "src/lesion/clinical_alignment.py")), groups={})
    for row in profiles.to_dict("records"):
        result["groups"].setdefault(row.pop("group"), {}).setdefault(row.pop("task"), {})[row.pop("measure")] = row
    (args.out / "patient-profiles.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    markers = clinical_alignment.published_marker_profiles(source.parent, args.n_boot, args.seed)
    markers.to_csv(args.out / "clinical-markers.csv", index=False)
    aligned = dict(meta=dict(analysis_commit=runinfo.commit(), n_boot=args.n_boot, seed=args.seed,
        status="published_group_marker_alignment", model_fit=False,
        cohorts="frequency, cueing and error composition come from different published cohorts",
        figure_precision="sensitivity bounds are not patient confidence intervals",
        sources={name: runinfo.sha256(source.parent / name) for name in markers.source_file.unique()},
        code_sha256=runinfo.sha256(ROOT / "src/lesion/clinical_alignment.py")), groups={}, figure_sensitivity={})
    aligned['meta']['marker_definitions'] = []
    for row in markers.astype(object).where(pd.notna(markers), None).to_dict("records"):
        group, marker, value = row.pop("group"), row.pop("marker"), row.pop("value")
        summary = {k: row[k] for k in ('ci_low', 'ci_high', 'n', 'unit')}
        summary['n_unit'] = '患者'
        aligned["groups"].setdefault(group, {})[marker] = dict(estimate=value, **summary)
        if row['sensitivity_low'] is not None:
            bounds = {edge: dict(estimate=row['sensitivity_'+edge], n=row['n'], n_unit='患者', unit=row['unit'],
                                note='读图精度敏感性界限，不是患者置信区间') for edge in ('low', 'high')}
            aligned['figure_sensitivity'].setdefault(group, {})[marker] = bounds
        aligned['meta']['marker_definitions'].append(dict(group=group, marker=marker, **row))
    (args.out / "clinical-marker-profiles.json").write_text(json.dumps(aligned, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(profiles[profiles.measure.isin(["associative", "correct"])].to_string(index=False))


if __name__ == "__main__":
    main()
