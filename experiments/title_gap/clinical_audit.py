"""Audit what the existing clinical tables can identify before attempting a numerical fit."""
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
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/title_gap_offline/clinical")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    source = ROOT / "data/literature"
    audit = clinical_alignment.evidence_audit(source)
    args.out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out / "evidence.csv", index=False)
    individual = pd.read_csv(source / "jefferies2006_individual.csv", keep_default_na=False)
    patients = clinical_alignment.patient_tasks(individual)
    patients.to_csv(args.out / "patient_tasks.csv", index=False)
    result = dict(status="individual_tables_recovered_model_calibration_pending", rows=len(audit),
                  numeric_rows=int(audit.numeric.sum()), figure_rows=int(audit.figure_estimate.sum()),
                  individual_rows=len(patients), patients=int(patients.patient_id.nunique()),
                  missing_rows=int(patients.value.isna().sum()),
                  calibrated_rows=int(patients.relative_loss.notna().sum()),
                  sources={p.name: runinfo.sha256(p) for p in source.glob("*.csv")},
                  missing=["matched model tasks and independently fixed scale mapping",
                           "patient-level frequency contrasts and cueing scores for this same cohort",
                           "measured two-axis dose surface and matched random profiles"],
                  overlap="2008 SA patients overlap 2006, the 2006 cueing SD cases differ from its main cohort",
                  analysis_commit=runinfo.commit(),
                  code_sha256={p: runinfo.sha256(ROOT / p) for p in
                               ("src/lesion/clinical_alignment.py", "experiments/title_gap/clinical_audit.py")})
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.report:
        summary = dict(meta=result, patients=dict(estimate=result["patients"], unit="patients"),
                       individual_observations=dict(estimate=result["individual_rows"], unit="rows"),
                       missing_observations=dict(estimate=result["missing_rows"], unit="rows"),
                       chance_calibrated=dict(estimate=result["calibrated_rows"], unit="rows"))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(result["status"], result["rows"])


if __name__ == "__main__":
    main()
