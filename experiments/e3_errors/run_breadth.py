"""GPU session 14, P13 (docs/plans/p13-knowledge-breadth.md), exploratory, one model load: knowledge lesions of
growing breadth, to see whether comprehension and cross-task consistency start to fail as in semantic dementia.

    python experiments/e3_errors/run_breadth.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 \
        --scorer prefix_avg3 --freeze-tag analysis-frozen-session14

Sets (lesion.rescreen.breadth_sets): K8; K30, the knowledge screen's 30 positive components (the session-9 dose
probe's K30); S64, S128, S256, K30 plus attribution candidates with both attributed effects positive, excluding C4,
candidate super experts and the components session 12's K8b screen excluded. n_random sets matched to each of K30 and
the S sets (same kind and layer, experts by routing frequency; drawn outside S256, C4 and the excluded components).
Per state, all on held-out items with the scoring of earlier runs:
- naming on the 240 knowledge test concepts, averaged over the dissociation test's batch compositions;
- experiment 3's candidate and label scores on its 160 test concepts (run_battery.run_state, no greedy answers, no
  read-outs), for unhinted naming, word-to-definition and six-choice;
- language-model loss on the 40 neutral passages (256 tokens);
- held-out M1 with the dissociation test's scorer (--scorer).
Writes outputs/e3_errors/<model>/breadth/{sets.json, ranked_candidates.csv, naming.parquet, candidates.parquet,
labels.parquet, m1.parquet, lm_loss.json, meta.json}; the folder is breadth.partial until every state is done, and
a rerun resumes after the last finished state.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import common  # noqa: E402
import run_battery as rb  # noqa: E402
from lesion import (baselines, components, knowledge_items as ki, m3_items, materials, naming, neutral_text,  # noqa: E402
                    rescreen, runinfo, scoring, screen)
from lesion.components import Component  # noqa: E402
from lesion.m3_optogenetics import lm_loss  # noqa: E402

POOL_ROLES = ("test", "calibration_a", "calibration_b", "reserve")    # word-to-definition foils, as in experiment 3
NAMING_BATCHES = (16, 11, 7)                                           # as the dissociation test
N_TEXTS, N_TOKENS = 40, 256
TABLES = ("naming", "candidates", "labels", "m1")
SEED = 41


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--anchor", type=int, default=rescreen.ANCHOR)
    ap.add_argument("--n-random", type=int, default=5, help="random sets matched to each set from the anchor up")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    suf = "_tiny" if args.tiny else ""
    kdir = ROOT / f"outputs/e3_knowledge{suf}" / args.model
    out = Path(args.out) if args.out else ROOT / f"outputs/e3_errors{suf}" / args.model / "breadth"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    m1_dir = common.out_dir(args, "screen")
    means_sha = runinfo.sha256(args.means)
    sources = {}
    for name, d in (("knowledge_screen", kdir / "screen"), ("k8b", kdir / "k8b"), ("m1_screen", m1_dir)):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        problems = [] if meta.get("status") == "done" else ["missing or unfinished"]
        if not args.tiny:      # the source runs must match this run's model, means and precision
            problems += [f"{k} {meta.get(k)!r} != {v!r}" for k, v in (("revision", models_revision(args)),
                         ("means_sha256", means_sha), ("dtype", args.dtype)) if meta.get(k) != v]
        if problems:
            raise SystemExit(f"{d}: " + "; ".join(problems))
        sources[name] = dict(resume_commit=meta.get("resume_commit"), analysis_plan_sha256=meta.get("analysis_plan_sha256"))
    k8b = json.loads((kdir / "k8b/selection.json").read_text(encoding="utf-8"))
    c4 = list(common.load_selection(m1_dir, json.loads((m1_dir / "meta.json").read_text(encoding="utf-8")))["control"])

    # 2. a stopped run resumes only with the same settings; compared before the model is loaded (agent2 review)
    partial = out.with_name(out.name + ".partial")
    fields = dict(stage="breadth", model=args.model, revision=models_revision(args), dtype=args.dtype, tiny=args.tiny,
                  scorer=args.scorer, freeze_tag=args.freeze_tag, run_commit=commit, means_sha256=means_sha,
                  sizes=args.sizes, anchor=args.anchor, n_random=args.n_random, seed=SEED, sources=sources,
                  final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"),
                  knowledge_items_sha256=runinfo.sha256(ki.OUT / "items.csv"),
                  screen_sha256=runinfo.sha256(kdir / "screen/screen.csv"),
                  attribution_sha256=runinfo.sha256(kdir / "screen/attribution.csv"),
                  k8b_selection_sha256=runinfo.sha256(kdir / "k8b/selection.json"), n_texts=N_TEXTS, neutral_tokens=N_TOKENS)
    old = None
    if partial.exists():
        if not (partial / "meta.json").exists():
            raise SystemExit(f"{partial} has no meta.json; move it away to start again")
        old = json.loads((partial / "meta.json").read_text(encoding="utf-8"))
        if {k: old.get(k) for k in fields} != json.loads(json.dumps(fields)):
            raise SystemExit(f"{partial} was started with other settings; move it away to start again")
    model, tok, chat, means, ablate = common.load(args)
    flagged = ([f"expert:{l}:{e}" for (l, e) in rescreen.super_expert_flags(means.expert_mean)]
               if means.expert_mean is not None else [])
    exclude = sorted(set(c4) | set(flagged) | set(k8b.get("excluded_lm_loss", [])) | set(k8b.get("excluded_scaling", [])))
    sets, ranked = rescreen.breadth_sets(pd.read_csv(kdir / "screen/screen.csv"), pd.read_csv(kdir / "screen/attribution.csv"),
                                         exclude=exclude, sizes=args.sizes, anchor=args.anchor)
    largest = sets[f"S{max(args.sizes)}"]
    freq = means.routing_frequency()
    randoms = {}
    for i, name in enumerate(k for k in sets if k != "K8"):
        targets = [Component.parse(c) for c in sets[name]]
        kinds = sorted({c.kind for c in targets})
        pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
        routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if "expert" in kinds else None
        # outside the largest knowledge set and the exclusions, so that knowledge and random sets share one pool;
        # a tiny model has too few heads for that, so a rehearsal excludes only the set itself
        banned = set(sets[name]) if args.tiny else set(largest) | set(exclude)
        drawn = baselines.matched_random_sets(targets, pool, args.n_random, seed=SEED + i, routing_freq=routing,
                                              exclude=[Component.parse(c) for c in banned],
                                              max_pct_gap=1.0 if args.tiny else 0.25)
        randoms.update({f"{name}_random_{j}": [c.key for c in s] for j, s in enumerate(drawn)})
    states = {"intact": [], **sets, **randoms}

    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role == "test"].reset_index(drop=True)
    pool_items = final[final.role.isin(POOL_ROLES)].reset_index(drop=True)
    test = ki.load_items("test", freeze_tag=args.freeze_tag)
    p_naming = naming.build_prompts(test, pd.read_csv(ki.OUT / "demos.csv"), chat)
    m1_test = materials.load_split("test", freeze_tag=args.freeze_tag)
    _, m1_demos, templates = scoring.load_materials()
    p_m1 = scoring.build_label_prompts(m1_test, m1_demos, templates, variant="main", chat=chat)
    label_score = screen.make_scorer(model, tok, p_m1, chat, ablate, args.batch_size, common.SCORERS[args.scorer])
    texts = neutral_text.load()[:N_TEXTS]

    derived = dict(exclude=exclude, super_expert_flags=flagged, n_m3_concepts=len(items), n_naming=len(test),
                   n_m1=len(m1_test))
    if old is not None and {k: old.get(k) for k in derived} != json.loads(json.dumps(derived)):
        raise SystemExit(f"{partial} was started with other derived sets or items; move it away to start again")
    fields.update(derived)
    tables, loss, done = {t: [] for t in TABLES}, {}, []
    if old is not None:
        for t in TABLES:
            if (partial / f"{t}.parquet").exists():
                tables[t] = [pd.read_parquet(partial / f"{t}.parquet")]
        loss = json.loads((partial / "lm_loss.json").read_text(encoding="utf-8")) if (partial / "lm_loss.json").exists() else {}
        done = list(old.get("done", []))
        print(f"resuming after {done}", flush=True)
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "sets.json").write_text(json.dumps(states, indent=1), encoding="utf-8")
    ranked.to_csv(partial / "ranked_candidates.csv", index=False)
    t0 = time.time()

    def save():
        for t, dfs in tables.items():
            if dfs:
                tmp = partial / f"{t}.parquet.tmp"
                pd.concat(dfs, ignore_index=True).to_parquet(tmp, index=False)
                tmp.replace(partial / f"{t}.parquet")
        (partial / "lm_loss.json").write_text(json.dumps(loss, indent=1), encoding="utf-8")
        runinfo.write_meta(partial, status="running", done=done, **fields)

    save()
    for name, keys in states.items():
        if name in done:
            continue
        t1 = time.time()
        comps = [Component.parse(c) for c in keys] or None
        res = rb.run_state(model, tok, chat, items, pool_items, comps, ablate, None, name, greedy=False, readouts=False)
        for t in ("candidates", "labels"):
            tables[t].append(res[t])
        with ablate(comps) if comps else _Nothing():
            tables["naming"].append(naming.averaged_logprobs(model, tok, p_naming, chat, NAMING_BATCHES).assign(state=name))
            loss[name] = lm_loss(model, tok, texts, max_tokens=N_TOKENS)
        tables["m1"].append(label_score(comps or ()).assign(state=name))
        done.append(name)
        save()
        print(f"{name}: {time.time() - t1:.0f} s (lm loss {loss[name]:.4f})", flush=True)
    runinfo.write_meta(partial, status="done", done=done, seconds=round(time.time() - t0, 1), **fields)
    partial.rename(out)
    print(f"written to {out}")


def models_revision(args):
    from lesion import models

    return models.MODELS[args.model][1]


class _Nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
