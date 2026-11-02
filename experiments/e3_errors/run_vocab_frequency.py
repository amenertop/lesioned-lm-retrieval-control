"""GPU session 12, items 2 and 4 (docs/plans/gpu-session12.md), exploratory, one model load:

    python experiments/e3_errors/run_vocab_frequency.py --model qwen3-30b-a3b-instruct --means <fp16 means> \
        --dtype float16 --freeze-tag analysis-frozen-session12

Item 2, why rare words depend more on K8. States: intact, K8, C4 and three random sets matched to K8 (the
dissociation test's sets.json). On the 240 held-out naming prompts, at the answer position: the log-probability of
each target word's first token (main analysis: its change against the word's Zipf, controlling the word's token count
and the intact log-probability), and the whole-vocabulary shift against word frequency (lesion.probes.vocab_shift,
exploratory; tokens that are one English word without a leading space, as a chat reply begins). The same shift on the
40 neutral passages (256 tokens), at every 8th position, for words with a leading space.

Item 4, do the control heads carry which task is asked. The C4 heads' outputs at the answer position (the o_proj
input slice that mean ablation replaces) averaged per task over the development colour and size items give a colour
vector and a size vector. On the held-out colour and size items, only at the answer position, C4's outputs are
replaced by: the neutral-text mean (a lesion at the answer only), the same task's vector, the other task's vector, or
a random direction as long as the same task's vector; and four random heads from C4's layers by their own other-task
vectors. Each item is scored with both option orders.

Item 6 (added after agent4's review, user 2026-10-14), rare words or long words: naming of every eligible word used
nowhere else (645 words), for the same states, so the frequency effect can be separated from the number of tokens.

Writes outputs/e3_errors/<model>/mechanisms/{naming_first.parquet, vocab_naming.parquet, vocab_neutral.parquet,
naming_length.parquet, patch_scores.parquet, task_vectors.pt, results.json, meta.json}.
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))

import common  # noqa: E402
from lesion import (baselines, components, damage, hooks, knowledge_items as ki, materials, naming, neutral_text,  # noqa: E402
                    probes, runinfo, scoring)
from lesion.components import Component  # noqa: E402

N_TEXTS, N_TOKENS, STEP = 40, 256, 8
LENGTH_BATCHES = (16, 11, 7)   # naming scores averaged over these batch compositions, as in the dissociation test
TASKS = ("colour", "size")
SEED = 0


def neutral_shift(model, seqs, comps, means, zipf, batch_size, pad_id):
    """vocab_shift rows for the lesion `comps` at every STEP-th position of each passage (all passages have N_TOKENS)."""
    device = next(model.parameters()).device
    cols = list(range(STEP - 1, N_TOKENS - 1, STEP))
    rows = []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            ids = torch.tensor(seqs[start:start + batch_size], device=device)
            intact = probes.position_logprobs(model, ids, cols).cpu()
            with hooks.ablate(model, comps, means):
                lesioned = probes.position_logprobs(model, ids, cols).cpu()
            rows.append(probes.vocab_shift(intact.reshape(-1, intact.shape[-1]), lesioned.reshape(-1, lesioned.shape[-1]), zipf))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--freeze-tag", required=True)
    ap.add_argument("--n-random-heads", type=int, default=1, help="random head sets for the item 4 control")
    ap.add_argument("--k8", default=None, help="comma-separated, instead of the dissociation test's main set (tiny runs)")
    ap.add_argument("--length-items", type=int, default=0, help="item 6 on the first N unused words only (tiny runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit = runinfo.check_analysis_freeze(args.freeze_tag)
    from wordfreq import zipf_frequency

    out = Path(args.out) if args.out else ROOT / ("outputs/e3_errors_tiny" if args.tiny else "outputs/e3_errors") / args.model / "mechanisms"
    sets_file = ROOT / ("outputs/e2_confirm_tiny" if args.tiny else "outputs/e2_confirm") / args.model / "dissociation/sets.json"
    sets = json.loads(sets_file.read_text(encoding="utf-8"))
    k8 = args.k8.split(",") if args.k8 else sets["main"]
    randoms = sorted(k for k in sets if k.startswith("random_"))[:3]
    states = {"K8": k8, "C4": sets["control"], **{k: sets[k] for k in randoms}}
    model, tok, chat, means, _ = common.load(args)
    pad = tok.eos_token_id or 0
    fields = dict(stage="mechanisms", model=args.model, tiny=args.tiny, freeze_tag=args.freeze_tag, run_commit=commit,
                  means_path=str(args.means), means_sha256=runinfo.sha256(args.means), states=states,
                  sets_sha256=runinfo.sha256(sets_file), n_texts=N_TEXTS, neutral_tokens=N_TOKENS, step=STEP, seed=SEED)
    runinfo.write_meta(out, status="running", **fields)
    t0 = time.time()
    log = lambda m: print(f"{m} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731

    # --- item 2: naming answer position -------------------------------------------------------------------------
    test = ki.load_items("test", freeze_tag=args.freeze_tag)
    prompts = naming.build_prompts(test, pd.read_csv(ki.OUT / "demos.csv"), chat)
    pairs = [naming.encode_pair(tok, p, chat) for p in prompts]
    ends = naming._end_ids(tok)
    seqs = [p for p, _ in pairs]
    vocab = model.get_output_embeddings().weight.shape[0]
    zipf_bare = probes.token_zipf(tok, lambda w: zipf_frequency(w, "en"), leading_space=False, size=vocab)
    zipf_space = probes.token_zipf(tok, lambda w: zipf_frequency(w, "en"), leading_space=True, size=vocab)
    intact = probes.last_logprobs(model, seqs, args.batch_size, pad)
    first = [a[0] for _, a in pairs]
    n_word = [sum(t not in ends for t in a) for _, a in pairs]
    rows = [dict(item_id=str(p.item_id), target=p.target, state="intact", first_logprob=float(intact[i, first[i]]),
                 n_tokens=n_word[i]) for i, p in enumerate(prompts)]
    shifts = []
    for name, comps in states.items():
        with hooks.ablate(model, [Component.parse(c) for c in comps], means):
            lesioned = probes.last_logprobs(model, seqs, args.batch_size, pad)
        rows += [dict(item_id=str(p.item_id), target=p.target, state=name, first_logprob=float(lesioned[i, first[i]]),
                      n_tokens=n_word[i]) for i, p in enumerate(prompts)]
        shifts.append(probes.vocab_shift(intact, lesioned, zipf_bare).assign(state=name, item_id=[str(p.item_id) for p in prompts]))
        log(f"naming vocabulary: {name}")
    pd.DataFrame(rows).to_parquet(out / "naming_first.parquet", index=False)
    pd.concat(shifts, ignore_index=True).to_parquet(out / "vocab_naming.parquet", index=False)

    # --- item 6: rare words or long words (agent4's review; user 2026-10-14) ------------------------------------------
    # Rare targets are more often split into several tokens (Zipf and token count correlate -0.54 on the held-out
    # items), so K8's frequency effect could come from word length. Naming on every eligible word not used anywhere
    # (not an item, not a demo), scored as the dissociation test scores naming, for the same states.
    demos_df = pd.read_csv(ki.OUT / "demos.csv")
    used = set(pd.read_csv(ki.OUT / "items.csv").target) | set(demos_df.target)
    pool = pd.read_csv(ki.OUT / "eligible.csv")
    pool = pool[~pool.target.isin(used)].reset_index(drop=True)
    if args.length_items:
        pool = pool.head(args.length_items)
    pool["item_id"] = [f"u{i:03d}" for i in range(len(pool))]
    pool_prompts = naming.build_prompts(pool, demos_df, chat)
    length = []
    for name, comps in {"intact": [], **states}.items():
        with hooks.ablate(model, [Component.parse(c) for c in comps], means) if comps else nullcontext():
            length.append(naming.averaged_logprobs(model, tok, pool_prompts, chat, LENGTH_BATCHES).assign(state=name))
        log(f"word length check: {name}")
    length = pd.concat(length, ignore_index=True).merge(pool[["item_id", "target", "zipf"]], on=["item_id", "target"])
    length.to_parquet(out / "naming_length.parquet", index=False)

    texts = neutral_text.load()[:N_TEXTS]
    nseqs = [s for s in (tok(t, add_special_tokens=False).input_ids[:N_TOKENS] for t in texts) if len(s) == N_TOKENS]
    neutral = [neutral_shift(model, nseqs, [Component.parse(c) for c in comps], means, zipf_space, 8, pad).assign(state=name)
               for name, comps in states.items()]
    pd.concat(neutral, ignore_index=True).to_parquet(out / "vocab_neutral.parquet", index=False)
    log("neutral vocabulary")

    # --- item 4: task vectors at the answer position -----------------------------------------------------------------
    _, demos, templates = scoring.load_materials()
    dev = materials.load_split("dev")
    held = materials.load_split("test", freeze_tag=args.freeze_tag)
    c4 = [Component.parse(c) for c in sets["control"]]
    heads_c4 = [c for c in c4 if c.kind == "head"]
    pool = components.all_components(model.config, "head")
    rand_heads = baselines.matched_random_sets(heads_c4, pool, args.n_random_heads, seed=11, exclude=heads_c4)
    rand_heads = [[Component.parse(k) if isinstance(k, str) else k for k in s] for s in rand_heads]
    all_heads = heads_c4 + [h for s in rand_heads for h in s]
    dev_prompts = scoring.build_label_prompts(dev, demos, templates, variant="main", chat=chat, tasks=TASKS)
    captured = probes.head_outputs_last(model, [scoring.encode(tok, p.text, chat) for p in dev_prompts], all_heads,
                                        args.batch_size, pad)
    is_task = {t: torch.tensor([p.task == t for p in dev_prompts]) for t in TASKS}
    vectors = {t: {h: captured[h.key][is_task[t]].mean(0) for h in all_heads} for t in TASKS}
    gen = torch.Generator().manual_seed(SEED)
    random_dir = {}
    for t in TASKS:
        random_dir[t] = {}
        for h in heads_c4:
            d = torch.randn(vectors[t][h].shape, generator=gen)
            random_dir[t][h] = d / d.norm() * vectors[t][h].norm()
    torch.save(dict(vectors={t: {h.key: v for h, v in vs.items()} for t, vs in vectors.items()},
                    random_direction={t: {h.key: v for h, v in vs.items()} for t, vs in random_dir.items()},
                    n_dev={t: int(is_task[t].sum()) for t in TASKS}), out / "task_vectors.pt")
    held_prompts = scoring.build_label_prompts(held, demos, templates, variant="main", chat=chat, tasks=TASKS)
    patch = [probes.label_scores_at_answer(model, tok, held_prompts, chat, args.batch_size).assign(state="intact")]
    for t in TASKS:
        other = [o for o in TASKS if o != t][0]
        ps = [p for p in held_prompts if p.task == t]
        plan = {"neutral_mean": (heads_c4, means),
                "same_task": (heads_c4, probes.patched_means(means, {h: vectors[t][h] for h in heads_c4})),
                "other_task": (heads_c4, probes.patched_means(means, {h: vectors[other][h] for h in heads_c4})),
                "random_direction": (heads_c4, probes.patched_means(means, random_dir[t]))}
        for s, heads in enumerate(rand_heads):
            plan[f"random_heads_{s}_other_task"] = (heads, probes.patched_means(means, {h: vectors[other][h] for h in heads}))
        for name, lesion in plan.items():
            patch.append(probes.label_scores_at_answer(model, tok, ps, chat, args.batch_size, lesion).assign(state=name))
            log(f"task patch: {t} {name}")
    patch = pd.concat(patch, ignore_index=True)
    patch.to_parquet(out / "patch_scores.parquet", index=False)

    # --- summaries (the full analysis is a separate CPU script) --------------------------------------------------------
    items_all = scoring.load_materials()[0].astype({"item_id": str})
    fm = damage.frequency_median(items_all)
    base = patch[patch.state == "intact"]
    summary = {}
    for name in patch.state.unique():
        if name == "intact":
            continue
        les = patch[patch.state == name]
        b = base[base.item_id.isin(set(les.item_id))]
        frame = damage.item_frame(damage.item_damage(b, les), items_all, fm)
        summary[name] = damage.bootstrap(frame, lambda c: {"control_effect": damage.effects(c)["control_effect"],
                                                           "mean_damage": float(np.mean(c["h"]))}, args.n_boot)
    result = dict(states=states, random_heads=[[h.key for h in s] for s in rand_heads], patch_summary=summary,
                  vocab_naming_mean={k: float(v) for k, v in pd.concat(shifts).groupby("state").spearman.mean().items()},
                  vocab_neutral_mean={k: float(v) for k, v in pd.concat(neutral).groupby("state").spearman.mean().items()})
    (out / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    runinfo.write_meta(out, status="done", seconds=round(time.time() - t0, 1), **fields)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
