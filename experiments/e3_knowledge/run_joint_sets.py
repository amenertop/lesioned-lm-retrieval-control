"""GPU session 14b (user, 2026-10-15): K8 and K8b-2 lesioned together. Do the two non-overlapping knowledge sets add up
(two parallel routes) or overlap (the same job done twice)? Exploratory, one model load.

    python experiments/e3_knowledge/run_joint_sets.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3 --freeze-tag analysis-frozen-session14b

States in one run, so that their intact scores are the same: intact, K8, K8b-2 (session 13's K8b without the two heads
it shares with S8), both together, and n_random sets matched to the joint set (same kind and layer, experts by
routing frequency). Per state: naming on the 240 held-out knowledge concepts, held-out M1 with the dissociation
test's scorer, and the language-model loss on the 40 neutral passages (256 tokens). Naming is scored as in the
dissociation test: the four main states averaged over its batch compositions ("averaged") and also at one composition
("single"); the random sets at that one composition only, to be compared with the main states' single scores. Writes outputs/e3_knowledge/<model>/joint_sets/{sets.json, naming.parquet, m1.parquet,
lm_loss.json, meta.json}.
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))

import common  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402  (the frozen K8)
from lesion import baselines, components, knowledge_items as ki, materials, naming, neutral_text, runinfo, scoring, screen  # noqa: E402
from lesion.components import Component  # noqa: E402
from lesion.m3_optogenetics import lm_loss  # noqa: E402

BATCHES = rdt.BATCHES
RANDOM_BATCH = rdt.RANDOM_BATCH
N_TEXTS, N_TOKENS = 40, 256
SEED = 51


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True)
    ap.add_argument("--n-random", type=int, default=100, help="random sets matched to the joint set (as the dissociation test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    suf = "_tiny" if args.tiny else ""
    split_sets = ROOT / f"outputs/e3_knowledge{suf}" / args.model / "k8b_split/sets.json"
    old_sets = ROOT / f"outputs/e2_confirm{suf}" / args.model / "dissociation/sets.json"
    k8 = list(rdt.K8) if not args.tiny else json.loads(old_sets.read_text(encoding="utf-8"))["main"]
    k8bm = json.loads(split_sets.read_text(encoding="utf-8"))["K8b_minus_shared"]
    if set(k8) & set(k8bm):
        raise SystemExit(f"K8 and K8b-2 overlap: {sorted(set(k8) & set(k8bm))}")
    joint = k8 + k8bm
    out = Path(args.out) if args.out else ROOT / f"outputs/e3_knowledge{suf}" / args.model / "joint_sets"
    model, tok, chat, means, ablate = common.load(args)
    targets = [Component.parse(c) for c in joint]
    kinds = sorted({c.kind for c in targets})
    pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
    freq = means.routing_frequency()
    routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"}
    m1_dir = common.out_dir(args, "screen")
    c4 = list(common.load_selection(m1_dir, json.loads((m1_dir / "meta.json").read_text(encoding="utf-8")))["control"])
    randoms = baselines.matched_random_sets(targets, pool, args.n_random, seed=SEED, routing_freq=routing,
                                            exclude=[Component.parse(c) for c in joint + c4],
                                            max_pct_gap=1.0 if args.tiny else 0.25)
    states = {"intact": [], "K8": k8, "K8b_minus_shared": k8bm, "joint": joint,
              **{f"random_{i:03d}": [c.key for c in s] for i, s in enumerate(randoms)}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "sets.json").write_text(json.dumps(states, indent=1), encoding="utf-8")
    fields = dict(stage="joint_sets", model=args.model, tiny=args.tiny, freeze_tag=args.freeze_tag, run_commit=commit,
                  means_sha256=runinfo.sha256(args.means), n_random=args.n_random, seed=SEED, scorer=args.scorer,
                  split_sets_sha256=runinfo.sha256(split_sets), batches=list(BATCHES), random_batch=RANDOM_BATCH,
                  sets_sha256=runinfo.sha256(out / "sets.json"))
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()

    test = ki.load_items("test", freeze_tag=args.freeze_tag)
    p_naming = naming.build_prompts(test, pd.read_csv(ki.OUT / "demos.csv"), chat)
    m1_test = materials.load_split("test", freeze_tag=args.freeze_tag)
    _, m1_demos, templates = scoring.load_materials()
    p_m1 = scoring.build_label_prompts(m1_test, m1_demos, templates, variant="main", chat=chat)
    label_score = screen.make_scorer(model, tok, p_m1, chat, ablate, args.batch_size, common.SCORERS[args.scorer])
    texts = neutral_text.load()[:N_TEXTS]
    names, m1s, loss = [], [], {}
    for name, keys in states.items():
        comps = [Component.parse(c) for c in keys]
        with ablate(comps) if comps else nullcontext():
            if not name.startswith("random_"):
                names.append(naming.averaged_logprobs(model, tok, p_naming, chat, BATCHES).assign(state=name, scoring="averaged"))
            names.append(naming.averaged_logprobs(model, tok, p_naming, chat, (RANDOM_BATCH,)).assign(state=name, scoring="single"))
            loss[name] = lm_loss(model, tok, texts, max_tokens=N_TOKENS)
        m1s.append(label_score(comps).assign(state=name))
        print(f"{name}: lm loss {loss[name]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(names, ignore_index=True).to_parquet(out / "naming.parquet", index=False)
    pd.concat(m1s, ignore_index=True).to_parquet(out / "m1.parquet", index=False)
    (out / "lm_loss.json").write_text(json.dumps(loss, indent=1), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
