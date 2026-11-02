"""Offline SemD coverage and paired regressions on already collected M1 and M3 scores."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import models, runinfo, semantic_diversity as sd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--norms", type=Path, default=ROOT / "outputs/title_gap_sources/SemD.xlsx")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/title_gap_offline/semd")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--report", type=Path, help="optional public statistical summary, without norm rows")
    args = ap.parse_args()
    tok = models.load_tokenizer("qwen3-30b-a3b-instruct")
    frames, cov, sources = sd.existing_frames(ROOT, args.norms, tok)
    results = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for task, frame in frames.items():
        controls = ("frequency", "n_tokens", "intact", "different_pair") if task == "m1_synonym" else ("frequency", "n_tokens", "intact")
        results[task] = sd.paired_regression(frame, controls, args.n_boot)
        # Per-word norm values remain in outputs, not in the public report.
        frame.to_csv(args.out / f"{task}.csv", index=False)
    sources.update({p: runinfo.sha256(ROOT / p) for p in ("src/lesion/semantic_diversity.py", "experiments/title_gap/analyze_semd.py")})
    payload = dict(status="exploratory_existing_scores", analysis_commit=runinfo.commit(), sources=sources,
                   coverage=cov, results=results, random_control_status="M3 comprehension random scores absent, scheduled with A",
                   imageability="not available on the full matched sample, not included", norm_redistribution="not established")
    (args.out / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(sd.public_summary(payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: dict(n=v.get("n"), status=v["status"]) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
