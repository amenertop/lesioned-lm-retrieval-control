"""Merging the shards of a run split over several GPUs, and the cross-GPU check (user decision 2026-10-12: two
GPUs; agent2 reviews 2026-10-12, -13).

Each GPU runs a disjoint part of one plan into <out>.shard<i>of<n>; GPUs 1..n-1 also rerun the intact state into
<out>.intactcheck<i>of<n>. merge() refuses unless every part is complete and all were run under the same plan (the
SAME fields of their meta.json), the shards hold every planned state exactly once, and one shard holds the router
counts; it then concatenates the tables, has the caller's check verify them against the plan, and compares every
intact score of each rerun with the merged intact state. The merged folder becomes <out> only when that check
passes; otherwise it is kept as <out>.crossgpu_failed-<time> with complete=false, so <out> never holds a failed
merge and a later attempt (after the cause is fixed and the intact check rerun) starts afresh. The merge records
the SHA-256 of its own tables and of every part's. joined_problems() checks later that a merged run's tables are
still the parts' tables joined, by content; lesion.m3_optogenetics_plan.evidence_problems uses it, with the parts'
own units and the cross-GPU check recomputed from the shard that holds the intact state, rather than trust the
record (agent3 reviews 2026-10-13).
"""
import json
import shutil
import time

import numpy as np
import pandas as pd

from . import runinfo, state_store

SAME = ("git_commit", "clean_tree", "freeze_tag", "role", "model", "revision", "dtype", "tiny", "means_path",
        "means_sha256", "final_items_sha256", "w2d_foils_sha256", "analysis_plan_sha256", "protocol_sha256", "sets",
        "random_controls", "all_states", "item_roles", "n_neutral_texts", "neutral_tokens", "ablation_mode")
TABLES = ("candidates", "labels", "greedy", "lm_loss", "router")
CROSS_GPU_MAX, CROSS_GPU_MEDIAN = 0.05, 0.01      # analysis plan §8; the same numbers as the implant's fidelity rule


def read_meta(d):
    if not (d / "meta.json").exists():
        raise SystemExit(f"missing or unfinished: {d}")
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if not m.get("complete"):
        raise SystemExit(f"not complete: {d}")
    return m


def same_plan(metas):
    differ = [k for k in SAME if any(m.get(k) != metas[0].get(k) for m in metas)]
    if differ:
        raise SystemExit(f"not run under the same plan: {differ}")
    if not metas[0].get("tiny") and not metas[0].get("clean_tree"):
        raise SystemExit("a part ran on an uncommitted tree")


def intact_scores(d):
    """Candidate log-probabilities and label margins of the intact state, one Series with a unique index."""
    key = lambda df, cols: df[cols].astype(str).agg("|".join, axis=1)
    c = pd.read_parquet(d / "candidates.parquet")
    c = c[c.state == "intact"]
    lab = pd.read_parquet(d / "labels.parquet")
    lab = lab[lab.state == "intact"]
    s = pd.concat([pd.Series(c.logprob.to_numpy(), index="cand|" + key(c, ["item_id", "condition", "role", "word"])),
                   pd.Series(lab.margin.to_numpy(), index="label|" + key(lab, ["item_id", "fmt", "variant"]))])
    if s.index.duplicated().any():
        raise SystemExit(f"{d.name}: duplicate intact scores")
    return s


def cross_gpu(merged, checks):
    """Compare the merged run's intact state with each intact-check rerun."""
    ref = intact_scores(merged)
    out = {}
    for d in checks:
        other = intact_scores(d)
        if set(other.index) != set(ref.index):
            raise SystemExit(f"{d.name}: the intact rerun does not cover the same scores")
        diff = (other.reindex(ref.index) - ref).abs().to_numpy()
        out[d.name] = dict(max=float(diff.max()), median=float(np.median(diff)), n=int(len(diff)),
                           passed=bool(diff.max() <= CROSS_GPU_MAX and np.median(diff) <= CROSS_GPU_MEDIAN))
    return out


def merge(base, out_name, n, all_states, check):
    """Merge base/<out_name>.shard<i>of<n> into base/<out_name>. all_states: the full state list of the plan (not
    taken from the shards' records); check(folder, states, parts): raises ValueError unless the folder's tables are
    what the plan says. Returns (folder written, cross-GPU check)."""
    shards = [base / f"{out_name}.shard{i}of{n}" for i in range(n)]
    checks = [base / f"{out_name}.intactcheck{i}of{n}" for i in range(1, n)]
    out = base / out_name
    if out.exists():
        raise SystemExit(f"Refusing to overwrite: {out}")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():                          # left by a stopped merge; rebuilt from the shards
        shutil.rmtree(partial)
    metas = [read_meta(s) for s in shards]
    check_metas = [read_meta(c) for c in checks]
    same_plan(metas + check_metas)
    if metas[0]["all_states"] != list(all_states):
        raise SystemExit("the shards were run under another state list than the plan's")
    seen = [s for m in metas for s in m["states"]]
    if len(seen) != len(set(seen)) or set(seen) != set(all_states):
        raise SystemExit("the shards do not hold every state of the plan exactly once")
    routers = [m for m in metas if "router" in (m.get("parts") or [])]
    if len(routers) != 1:
        raise SystemExit("exactly one shard must hold the router counts")
    try:
        for s, m in zip(shards, metas):
            check(s, m["states"], m["parts"])
        for c in checks:
            check(c, ["intact"], ["states"])
    except ValueError as e:
        raise SystemExit(f"a part's tables differ from the plan: {e}")
    partial.mkdir(parents=True)
    for t in TABLES:
        parts = [pd.read_parquet(s / f"{t}.parquet") for s in shards if (s / f"{t}.parquet").exists()]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(partial / f"{t}.parquet", index=False)
    parts = sorted({p for m in metas for p in m["parts"]})
    try:
        check(partial, list(all_states), parts)
    except ValueError as e:
        raise SystemExit(f"merged tables differ from the plan: {e}")
    result = cross_gpu(partial, checks)
    passed = all(v["passed"] for v in result.values())
    meta = {k: metas[0].get(k) for k in SAME}
    meta.update(states=list(all_states), router_experts=routers[0].get("router_experts"), parts=parts,
                n_items=len(metas[0]["item_roles"]), complete=passed,
                cross_gpu_check=dict(tolerance=dict(max=CROSS_GPU_MAX, median=CROSS_GPU_MEDIAN), results=result, passed=passed),
                shards=[dict(dir=s.name, shard=m.get("shard"), states=m["states"], gpu=m.get("gpu"), host=m.get("host"),
                             timestamp=m.get("timestamp"), timing=m.get("timing")) for s, m in zip(shards, metas)],
                intact_checks=[c.name for c in checks], tables_sha256=state_store.tables_sha256(partial),
                source_tables_sha256={p.name: state_store.tables_sha256(p) for p in shards + checks},
                merged_by_commit=runinfo.commit(), merged_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    (partial / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    dest = out if passed else out.with_name(f"{out.name}.crossgpu_failed-{time.strftime('%Y%m%d-%H%M%S')}")
    partial.rename(dest)
    return dest, meta["cross_gpu_check"]


def joined_problems(merged, parts):
    """Tables of the merged folder that are not the parts' tables joined in the parts' order, as merge() writes
    them: compared by content (rows in order, columns, values), not by file bytes (agent3 2026-10-13, F1)."""
    out = []
    names = {f.name[:-len(".parquet")] for d in [merged, *parts] for f in d.glob("*.parquet")}
    for t in sorted(names):
        frames = [pd.read_parquet(p / f"{t}.parquet") for p in parts if (p / f"{t}.parquet").exists()]
        f = merged / f"{t}.parquet"
        if not frames or not f.exists() or not state_store.same_content(f, pd.concat(frames, ignore_index=True)):
            out.append(t)
    return [f"merged tables {out} are not the shards' tables joined"] if out else []
