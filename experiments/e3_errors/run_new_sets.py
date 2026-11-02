"""Experiment 3's battery and a patient-style cued naming on the 160 test concepts, for component sets given on the
command line (exploratory, user decisions 2026-10-13; docs/plans/gpu-session11.md):

    python experiments/e3_errors/run_new_sets.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 \
        --set K8=expert:46:61,... --set C4=head:9:15,... --set S8=... --set C4s=...

States: intact, then each set (full mean ablation). Per state:
- the battery as in experiment 3 (lesion.m3_battery via run_battery.run_state: free naming with five candidates,
  cue and miscue, options answered with the word, copying, context intrusion, six-choice letters, word-to-definition),
  with greedy answers;
- cued naming as in Jefferies, Patterson & Lambon Ralph 2008, in text: the hint gives the target's first j letters,
  j = 0 (no hint) up to all but the last letter, the demos' hints the same length; the model's greedy answer is
  classified (target, coordinate, ...), so accuracy can be counted at each length, as for the patients.
Writes outputs/e3_errors/<model>/new_sets/{candidates,labels,greedy,cued}.parquet and meta.json; the folder is
new_sets.partial until every state is done, and a rerun resumes after the last finished state.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import common  # noqa: E402
import run_battery as rb  # noqa: E402
from lesion import m3_battery as mb, m3_cued, m3_items, rescreen, runinfo  # noqa: E402

POOL_ROLES = ("test", "calibration_a", "calibration_b", "reserve")    # foils for word-to-definition, as in experiment 3
TABLES = ("candidates", "labels", "greedy", "cued")


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--set", action="append", default=[], help="NAME=comma-separated component keys; repeatable")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sets = rescreen.parse_named_sets(args.set)
    except ValueError as e:
        raise SystemExit(str(e))
    commit = common.guard(args)
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role == "test"].reset_index(drop=True)
    pool = final[final.role.isin(POOL_ROLES)].reset_index(drop=True)
    out = Path(args.out) if args.out else ROOT / ("outputs/e3_errors_tiny" if args.tiny else "outputs/e3_errors") / args.model / "new_sets"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    partial = out.with_name(out.name + ".partial")
    fields = dict(model=args.model, dtype=args.dtype, means_sha256=runinfo.sha256(args.means), scorer=args.scorer, tiny=args.tiny,
                  sets={k: [c.key for c in v] for k, v in sets.items()}, final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"))
    tables, done = {t: [] for t in TABLES}, []
    if partial.exists():
        import json
        old = json.loads((partial / "meta.json").read_text(encoding="utf-8"))
        if {k: old.get(k) for k in fields} != fields:
            raise SystemExit(f"{partial} was started with other settings; move it away to start again")
        for t in TABLES:
            if (partial / f"{t}.parquet").exists():
                tables[t] = [pd.read_parquet(partial / f"{t}.parquet")]
        done = list(old.get("done", []))
        print(f"resuming after {done}", flush=True)
    partial.mkdir(parents=True, exist_ok=True)
    model, tok, chat, means, ablate = common.load(args)
    cued = m3_cued.prompts(items, chat)
    t0 = time.time()

    def save(state_done):
        for t, dfs in tables.items():
            if dfs:
                tmp = partial / f"{t}.parquet.tmp"
                pd.concat(dfs, ignore_index=True).to_parquet(tmp, index=False)
                tmp.replace(partial / f"{t}.parquet")
        runinfo.write_meta(partial, status="running", resume_commit=commit, done=state_done, **fields)

    save(done)
    for name, comps in [("intact", None), *sets.items()]:
        if name in done:
            continue
        t1 = time.time()
        res = rb.run_state(model, tok, chat, items, pool, comps, ablate, None, name, greedy=True, readouts=False, intrusion=True)
        for t, df in res.items():
            if df is not None and t in tables:
                tables[t].append(df)
        with ablate(comps) if comps else _nothing():
            tables["cued"].append(m3_cued.label(mb.greedy_answers(model, tok, cued, chat), cued, items).assign(state=name))
        done.append(name)
        save(done)
        print(f"{name}: {time.time() - t1:.0f} s", flush=True)
    runinfo.write_meta(partial, status="done", resume_commit=commit, done=done, seconds=round(time.time() - t0, 1),
                       n_items=len(items), **fields)
    partial.rename(out)
    print(f"written to {out}")


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
