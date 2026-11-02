"""Merge the shards of a two-GPU optogenetics run (run_optogenetics.py --shard i/n; user decision 2026-10-12) and
check the GPUs against each other. No GPU:

    python experiments/e3_errors/merge_optogenetics_shards.py --model qwen3-30b-a3b-instruct --out-name optogenetics_confirm --shards 2

lesion.shards.merge does the work: every part complete and run under the same plan (same commit on a clean tree,
freeze tag, role, model, revision, dtype, inputs, components, concepts), the shards together holding the plan's
states (lesion.m3_optogenetics_plan.state_names, not the shards' own record) exactly once; each part's tables and
the merged tables checked against the plan and the materials (m3_optogenetics_plan.check_run_tables); then the
cross-GPU check (analysis plan §8): every intact score of each intact-check rerun against the merged intact state,
largest absolute difference at most 0.05 and median at most 0.01. Only a passed merge becomes <out-name>; a failed
one is kept as <out-name>.crossgpu_failed-<time> for inspection and this script exits with code 2.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import m3_optogenetics_plan as op, shards  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--role", choices=("calibration", "test"), default="test")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)
    base = ROOT / ("outputs/e3_errors_tiny" if args.tiny else "outputs/e3_errors") / args.model
    check = lambda d, states, parts: op.check_run_tables(d, args.role, args.tiny, states, parts)
    out, result = shards.merge(base, args.out_name, args.shards, list(op.state_names(args.role)), check)
    print(f"merged into {out}; cross-GPU check {'passed' if result['passed'] else 'FAILED'}: {result['results']}")
    if not result["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
