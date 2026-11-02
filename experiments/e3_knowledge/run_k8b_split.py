"""GPU session 13, item 1 (docs/plans/gpu-session13.md), exploratory, one model load: K8b taken apart.

    python experiments/e3_knowledge/run_k8b_split.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --scorer prefix_avg3 --freeze-tag analysis-frozen-session13

K8b (session 12, outputs/e3_knowledge/<model>/k8b/selection.json) shares head:38:3 and head:38:5 with the store set S8
(session 11, dissociation_new/sets.json). Sets: K8b, K8b without the shared pair, the pair alone, S8 without the pair.
1. Language-model loss on the 40 neutral passages (256 tokens, as qwen_lm_check.py in session 11) for intact, the four
   sets, and random sets matched (same kind and layer; experts by routing frequency) to K8b without the pair and to the
   pair, n_random each.
2. M1 on the held-out items, both option orders, for the same states, with one scoring function
   (lesion.probes.label_scores_in_positions: one batch composition, no prefix cache), for semantic and string damage
   and the per-item margin scaling.
3. Rare or long words (session 12 item 6 repeated for K8b): naming of the 645 unused eligible words for intact, K8b,
   K8b without the pair, the pair, and two random sets matched to K8b.
4. Where the target falls in the whole vocabulary's change (lesion.probes.target_change_rank), on the 240 held-out
   naming prompts at the answer position, for K8, K8b, K8b without the pair, the pair, and the first three random sets
   of the K8 dissociation test.

Writes outputs/e3_knowledge/<model>/k8b_split/{sets.json, lm_loss.json, m1_scores.parquet, naming_length.parquet,
target_rank.parquet, meta.json}. The held-out dissociation tests of the new sets are run_dissociation_test.py runs.
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

import common  # noqa: E402
from lesion import (baselines, components, hooks, knowledge_items as ki, materials, naming, neutral_text,  # noqa: E402
                    probes, runinfo, scoring)
from lesion.components import Component  # noqa: E402
from lesion.m3_optogenetics import lm_loss  # noqa: E402

N_TEXTS, N_TOKENS = 40, 256
LENGTH_BATCHES = (16, 11, 7)      # as session 12 item 6
SHARED = ("head:38:3", "head:38:5")
SEEDS = dict(k8b_minus=21, pair=22, k8b=23)


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True)
    ap.add_argument("--n-random", type=int, default=20, help="random sets matched to K8b without the pair, and to the pair")
    ap.add_argument("--pair", default=None, help="comma-separated, instead of K8b and S8's shared components (tiny runs)")
    ap.add_argument("--length-items", type=int, default=0, help="the length check on the first N unused words only (tiny runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    suf = "_tiny" if args.tiny else ""
    kdir = ROOT / f"outputs/e3_knowledge{suf}" / args.model / "k8b"
    cdir = ROOT / f"outputs/e2_confirm{suf}" / args.model
    out = Path(args.out) if args.out else ROOT / f"outputs/e3_knowledge{suf}" / args.model / "k8b_split"

    # the tiny K8 dissociation folder of earlier rehearsals holds only sets.json, so it is not checked in tiny runs
    for d in (kdir, cdir / "dissociation_new", *([] if args.tiny else [cdir / "dissociation"])):
        meta = d / "meta.json"
        if not meta.exists() or json.loads(meta.read_text(encoding="utf-8")).get("status") != "done":
            raise SystemExit(f"{d} is missing or unfinished")
    k8b = json.loads((kdir / "selection.json").read_text(encoding="utf-8"))["k8b"]
    new = json.loads((cdir / "dissociation_new/sets.json").read_text(encoding="utf-8"))
    old = json.loads((cdir / "dissociation/sets.json").read_text(encoding="utf-8"))
    s8 = new["main"]
    if not k8b or not s8 or not old.get("main"):
        raise SystemExit("K8b, S8 or K8 is empty in the source files")
    pair = args.pair.split(",") if args.pair else sorted(set(k8b) & set(s8))
    if not args.tiny and set(pair) != set(SHARED):
        raise SystemExit(f"K8b and S8 share {pair}, not {SHARED} as in the plan")
    if not pair or not set(pair) < set(k8b):
        raise SystemExit(f"the pair {pair} must be a proper subset of K8b {k8b}")
    sets = {"K8b": k8b, "K8b_minus_shared": [c for c in k8b if c not in pair], "shared_pair": list(pair),
            "S8_minus_pair": [c for c in s8 if c not in pair], "K8": old["main"]}

    model, tok, chat, means, _ = common.load(args)
    pad = tok.eos_token_id or 0
    freq = means.routing_frequency()
    exclude = [Component.parse(c) for c in set(k8b) | set(s8)]

    def matched(name, n, seed):
        targets = [Component.parse(c) for c in sets[name]]
        kinds = sorted({c.kind for c in targets})
        pool = [c for kind in kinds for c in components.all_components(model.config, kind)]
        routing = {c: float(freq[c.layer][c.index]) for c in pool if c.kind == "expert"} if "expert" in kinds else None
        return [[c.key for c in s] for s in baselines.matched_random_sets(targets, pool, n, seed=seed, exclude=exclude,
                                                                          routing_freq=routing)]

    randoms = {f"rand_k8bm_{i:03d}": s for i, s in enumerate(matched("K8b_minus_shared", args.n_random, SEEDS["k8b_minus"]))}
    randoms.update({f"rand_pair_{i:03d}": s for i, s in enumerate(matched("shared_pair", args.n_random, SEEDS["pair"]))})
    rand_k8b = {f"rand_k8b_{i:03d}": s for i, s in enumerate(matched("K8b", 2, SEEDS["k8b"]))}
    k8_randoms = {k: old[k] for k in sorted(k for k in old if k.startswith("random_"))[:3]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "sets.json").write_text(json.dumps(dict(**sets, **randoms, **rand_k8b, **k8_randoms), indent=2), encoding="utf-8")
    fields = dict(stage="k8b_split", model=args.model, tiny=args.tiny, freeze_tag=args.freeze_tag, run_commit=commit,
                  means_path=str(args.means), means_sha256=runinfo.sha256(args.means), pair=list(pair),
                  n_random=args.n_random, seeds=SEEDS, n_texts=N_TEXTS, neutral_tokens=N_TOKENS,
                  m1_score_method="probes.label_scores_in_positions, one batch composition, no prefix cache (not --scorer)",
                  k8b_selection_sha256=runinfo.sha256(kdir / "selection.json"),
                  new_sets_sha256=runinfo.sha256(cdir / "dissociation_new/sets.json"),
                  old_sets_sha256=runinfo.sha256(cdir / "dissociation/sets.json"))
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()
    log = lambda m: print(f"{m} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    lesion = lambda keys: ([Component.parse(c) for c in keys], means) if keys else None  # noqa: E731
    main_states = {k: sets[k] for k in ("K8b", "K8b_minus_shared", "shared_pair", "S8_minus_pair")}

    # --- 1 and 2: language-model loss and held-out M1 -----------------------------------------------------------------
    texts = neutral_text.load()[:N_TEXTS]
    m1_test = materials.load_split("test", freeze_tag=args.freeze_tag)
    _, m1_demos, templates = scoring.load_materials()
    prompts = scoring.build_label_prompts(m1_test, m1_demos, templates, variant="main", chat=chat)
    loss, m1 = {}, []
    for name, keys in {"intact": [], **main_states, **randoms}.items():
        with hooks.ablate(model, *lesion(keys)) if keys else nullcontext():
            loss[name] = lm_loss(model, tok, texts, max_tokens=N_TOKENS)
        m1.append(probes.label_scores_in_positions(model, tok, prompts, chat, None, args.batch_size, lesion(keys))
                  .assign(state=name))
        log(f"language and M1: {name} (loss {loss[name]:.4f})")
    (out / "lm_loss.json").write_text(json.dumps(dict(lm_loss=loss, n_texts=len(texts), neutral_tokens=N_TOKENS), indent=1),
                                      encoding="utf-8")
    pd.concat(m1, ignore_index=True).to_parquet(out / "m1_scores.parquet", index=False)

    # --- 3: rare or long words (session 12 item 6, for K8b) -----------------------------------------------------------
    demos_df = pd.read_csv(ki.OUT / "demos.csv")
    used = set(pd.read_csv(ki.OUT / "items.csv").target) | set(demos_df.target)
    pool = pd.read_csv(ki.OUT / "eligible.csv")
    pool = pool[~pool.target.isin(used)].reset_index(drop=True)
    if args.length_items:
        pool = pool.head(args.length_items)
    pool["item_id"] = [f"u{i:03d}" for i in range(len(pool))]
    pool_prompts = naming.build_prompts(pool, demos_df, chat)
    length = []
    for name, keys in {"intact": [], "K8b": sets["K8b"], "K8b_minus_shared": sets["K8b_minus_shared"],
                       "shared_pair": sets["shared_pair"], **rand_k8b}.items():
        with hooks.ablate(model, *lesion(keys)) if keys else nullcontext():
            length.append(naming.averaged_logprobs(model, tok, pool_prompts, chat, LENGTH_BATCHES).assign(state=name))
        log(f"word length check: {name}")
    length = pd.concat(length, ignore_index=True).merge(pool[["item_id", "target", "zipf"]], on=["item_id", "target"])
    length.to_parquet(out / "naming_length.parquet", index=False)

    # --- 4: the target's place in the whole vocabulary's change -------------------------------------------------------
    test = ki.load_items("test", freeze_tag=args.freeze_tag)
    nprompts = naming.build_prompts(test, demos_df, chat)
    pairs = [naming.encode_pair(tok, p, chat) for p in nprompts]
    seqs = [p for p, _ in pairs]
    first = [a[0] for _, a in pairs]
    intact = probes.last_logprobs(model, seqs, args.batch_size, pad)
    rank = []
    for name, keys in {"K8": sets["K8"], "K8b": sets["K8b"], "K8b_minus_shared": sets["K8b_minus_shared"],
                       "shared_pair": sets["shared_pair"], **k8_randoms}.items():
        with hooks.ablate(model, *lesion(keys)):
            lesioned = probes.last_logprobs(model, seqs, args.batch_size, pad)
        rank.append(probes.target_change_rank(intact, lesioned, first)
                    .assign(state=name, item_id=[str(p.item_id) for p in nprompts], target=[p.target for p in nprompts]))
        log(f"target rank: {name}")
    pd.concat(rank, ignore_index=True).to_parquet(out / "target_rank.parquet", index=False)

    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
