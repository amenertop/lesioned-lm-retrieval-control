"""Provenance and row coverage of the frozen experiment-3 outputs, without running the model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import m3_battery as mb, m3_items, runinfo

FILES = ("candidates.parquet", "labels.parquet", "dla.parquet", "dla_total.parquet", "greedy.parquet")


def check_rows(table, columns, expected, name, values=()):
    """No missing, extra or duplicate observations, and finite scores on the tested rows."""
    keys = list(table[list(columns)].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError(f"{name}: missing, extra or duplicate rows")
    if values and not np.isfinite(table[list(values)].to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: nonfinite scores")


def check_source(directory, final_items, sets):
    """Verify the source freeze, materials, fixed sets and the complete test battery.

    Analysis may run at a later commit. The source run, not today's checkout, must match its frozen plan.
    Only the recorded hashes of means are read here, not the weights or a model.
    """
    d = Path(directory)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    tag = meta.get("freeze_tag", "")
    frozen = runinfo.git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}") if tag else ""
    if not meta.get("complete") or not tag.startswith("analysis-frozen") or not frozen:
        raise ValueError("experiment 3: incomplete run or missing analysis freeze")
    if any(meta.get(k) != frozen for k in ("resume_commit", "git_commit")):
        raise ValueError("experiment 3: run commit differs from its freeze tag")
    if meta.get("analysis_plan_sha256") != runinfo.analysis_plan_sha256_at(frozen):
        raise ValueError("experiment 3: analysis plan differs from the frozen source")
    material_paths = dict(final_items_sha256=Path(final_items), foils_sha256=m3_items.OUT / "w2d_foils_test.csv")
    for key, path in material_paths.items():
        if meta.get(key) != runinfo.sha256(path) or meta[key] != runinfo.sha256_at(frozen, path.relative_to(runinfo.ROOT)):
            raise ValueError(f"experiment 3: material differs from frozen source: {path.name}")
    if meta.get("sets") != {k: list(v) for k, v in sets.items()}:
        raise ValueError("experiment 3: component sets differ from the fixed test")
    states = ("intact", "K8", "C4", "C4_answer", "K30", "C16")
    if meta.get("states") != list(states):
        raise ValueError("experiment 3: unexpected states")
    all_items = pd.read_csv(final_items, keep_default_na=False)
    items = all_items[all_items.role == "test"]
    ids = items.item_id.tolist()
    if len(ids) != 160 or meta.get("n_items") != len(ids):
        raise ValueError("experiment 3: incomplete test split")
    pool = all_items[all_items.role.isin(("test", "calibration_a", "calibration_b", "reserve"))]
    prompts = mb.scored_prompts(items, intrusion=True)
    labels = mb.label_items(items, pool)
    tables = {name: pd.read_parquet(d / name) for name in FILES}
    check_rows(tables[FILES[0]], ("state", "item_id", "condition", "role", "word"),
               ((s, p.item_id, p.condition, r, w) for s in states for p in prompts for r, w in p.candidates.items()),
               "candidates", ("logprob",))
    check_rows(tables[FILES[1]], ("state", "item_id", "fmt", "variant", "correct"),
               ((s, p.item_id, p.fmt, p.variant, p.correct) for s in states for p in labels), "labels", ("margin",))
    check_rows(tables[FILES[2]], ("item_id", "component"), ((i, c) for i in ids for c in sets["K8"]), "dla", ("specific",))
    check_rows(tables[FILES[3]], ("state", "item_id"), ((s, i) for s in ("intact", "K8") for i in ids),
               "dla_total", ("target_logit", "other_targets_logit"))
    check_rows(tables[FILES[4]], ("state", "item_id"), ((s, i) for s in ("intact", "K8", "C4") for i in ids), "greedy")
    return dict(source_freeze_tag=tag, source_commit=frozen, analysis_commit=runinfo.commit(),
                analysis_code_sha256=runinfo.sha256(runinfo.ROOT / "experiments/e3_errors/analyze_experiment3.py"),
                validator_sha256=runinfo.sha256(__file__),
                inputs_sha256={n: runinfo.sha256(d / n) for n in ("meta.json", *FILES)},
                materials_sha256={p.name: runinfo.sha256(p) for p in material_paths.values()})


def check_output(path, dry_run, external_dla):
    if dry_run and Path(path).resolve().is_relative_to((runinfo.ROOT / "reports/confirm").resolve()):
        raise ValueError("a calibration dry run must write outside reports/confirm")
    if external_dla and not dry_run:
        raise ValueError("external DLA is allowed only in a calibration dry run")
