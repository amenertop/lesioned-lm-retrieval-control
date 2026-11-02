"""Summarize a coarse/component lesion battery without loading a model."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import legacy_analysis, runinfo, scoring


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--report", type=Path, help="ledger-ready summary, original units and point estimates")
    args = ap.parse_args()
    words = pd.read_csv(ROOT / "data/materials/m3_knowledge/items.csv")
    runs = [legacy_analysis.analyze(path, words, scoring.load_materials()[0], args.partial) for path in args.run]
    plan = runs[0][1]
    for _, other in runs[1:]:
        fields = ("chat", "batches", "words", "control", "comprehension", "neutral_sequences")
        if any(plan[k] != other[k] for k in fields):
            ap.error("runs use different materials or scoring settings")
        fields = ("model", "revision", "dtype", "moe", "scorer", "split")
        if any(plan["provenance"][k] != other["provenance"][k] for k in fields):
            ap.error("runs use different models or evaluation settings")
    table = pd.concat([t if i == 0 else t[t.state != "intact"] for i, (t, _) in enumerate(runs)], ignore_index=True)
    if table.state.duplicated().any():
        ap.error("lesioned states overlap between runs")
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "effects.csv", index=False)
    result = dict(status="development_partial" if args.partial else "complete", split=plan["provenance"]["split"],
                  source_fingerprints=[p["fingerprint"] for _, p in runs], analysis_commit=runinfo.commit(),
                  code_sha256=runinfo.sha256(ROOT / "src/lesion/legacy_analysis.py"),
                  repeated_baselines=[t[t.state == "intact"].to_dict("records")[0] for t, _ in runs],
                  state_definitions=[s for _, p in runs for s in p["states"] if s["name"] != "intact"],
                  states=table.to_dict("records"))
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.report:
        if args.partial:
            ap.error("partial runs cannot be written as a completed grid report")
        payload = legacy_analysis.ledger_summary(table, [p for _, p in runs], scoring.load_materials()[0])
        payload["meta"]["analysis_commit"] = runinfo.commit()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(table[["state", "naming_frequency", "control_effect", "comprehension_damage", "lm_change"]].to_string(index=False))


if __name__ == "__main__":
    main()
