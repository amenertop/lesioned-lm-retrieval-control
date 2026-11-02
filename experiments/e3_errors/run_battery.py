"""Cross-format mechanism battery on the M3 calibration concepts, one GPU session (experiment3-protocol.md §5.5;
user decision Q18, 2026-10-08). Calibration concepts only; no test concept is read.

    python experiments/e3_errors/run_battery.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16

Lesion states (all full mean ablation, alpha 0):
- intact; K8; C4; K30; C16 (WikiText means, as in experiments 1 and 2);
- K8 and C4 restricted to the answer region or to the prompt region (requires hooks.position_mask);
- K8 and C4 with task-matched means collected on the calibration_a prompts of the battery.
Per state: candidate scores of every scored format and label scores (each averaged over three batch
compositions, ADR 0004); greedy answers for name_free (whole-sequence states); logit lens for intact, K8, C4, K30;
attention of the C4 heads to the hint letters for intact and K8.
Writes outputs/e3_errors/<model>/battery/{candidates,labels,greedy,lens,hint_attention}.parquet and meta.json;
never overwrites an existing run or battery.partial; checkpoints are atomically replaced during this run,
and the directory is renamed when meta.json (complete) is written.
"""
import argparse
import functools
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
sys.path.insert(0, str(ROOT / "experiments/e2_confirm"))
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import common  # noqa: E402
import probe_lesions  # noqa: E402
import run_dissociation_test as rdt  # noqa: E402
from lesion import hooks, m3_battery as mb, m3_items, m3_mechanism as mm, m3_scoring as ms, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

CANDIDATE_BATCHES = ms.NAMING_BATCHES
LABEL_BATCHES = (32, 21, 13)
LENS_STATES = ("intact", "K8", "C4", "K30")
ATTENTION_STATES = ("intact", "K8")


def averaged(fn, batch_sizes, value_cols):
    """fn(batch_size) -> table; the value columns averaged over the batch sizes (rows in the same order)."""
    tables = [fn(bs) for bs in batch_sizes]
    out = tables[0].copy()
    for c in value_cols:
        out[c] = np.mean([t[c].to_numpy() for t in tables], axis=0)
    return out


def states(sets, wiki, task, positional):
    """(name, components, ablate factory, region) for every lesion state of the battery."""
    out = [("intact", None, None, None)]
    out += [(k, sets[k], wiki, None) for k in ("K8", "C4", "K30", "C16")]
    if positional:
        out += [(f"{k}_{r}", sets[k], wiki, r) for k in ("K8", "C4") for r in ("answer", "prompt")]
    if task is not None:
        out += [(f"{k}_taskmean", sets[k], task, None) for k in ("K8", "C4")]
    return out


def run_state(model, tok, chat, items, pool, comps, ablate, region, name, heads=(), greedy=True, readouts=True,
              intrusion=False):
    """Every table of one lesion state; heads: the attention heads whose hint attention is read; readouts=False
    skips the logit lens and the attention read-out (the confirmatory run on the test concepts does not use them)."""
    scored = mb.scored_prompts(items, chat, intrusion=True) if intrusion else mb.scored_prompts(items, chat)
    labels = mb.label_items(items, pool, chat)
    with nullcontext() if comps is None else ablate(comps):
        cand = averaged(lambda bs: mb.candidate_scores(model, tok, scored, chat, bs, region), CANDIDATE_BATCHES, ["logprob"])
        lab = averaged(lambda bs: mb.label_scores(model, tok, labels, chat, bs, region), LABEL_BATCHES,
                       ["margin"] + [f"logit_{l}" for l in ms.LETTERS])
        # Winning among six labels does not require more than half their total
        # probability. Recompute correctness from the averaged label logits.
        lab["top_correct"] = [row[f"logit_{row['correct']}"] >
                              max(row[f"logit_{l}"] for l in ms.LETTERS
                                  if l != row["correct"] and pd.notna(row.get(f"logit_{l}")))
                              for row in lab.to_dict("records")]
        gen = None
        if greedy and region is None:
            free = [p for p in scored if p.condition == "name_free"]
            gen = mb.greedy_answers(model, tok, free, chat)
            rows = items.set_index("item_id")
            gen["kind"] = [mb.classify_answer(a, rows.loc[i]) for i, a in zip(gen.item_id, gen.answer)]
        lens = mm.logit_lens(model, tok, [p for p in scored if p.condition == "name_free"], chat) if readouts and name in LENS_STATES else None
        att = None
        if readouts and name in ATTENTION_STATES:
            previous = model.config._attn_implementation
            model.config._attn_implementation = "eager"
            try:
                hinted = [p for p in scored if p.condition in ("cue", "miscue")]
                att = mm.hint_attention(model, tok, hinted, list(heads), chat)
            finally:
                model.config._attn_implementation = previous
    return {t: (df.assign(state=name) if df is not None else None)
            for t, df in dict(candidates=cand, labels=lab, greedy=gen, lens=lens, hint_attention=att).items()}



def task_mean_prompts(items, pool, chat):
    """Reference prompts use calibration_a and reserve foils, never b definitions."""
    a_items = items[items.role == "calibration_a"]
    mean_pool = pool[pool.role.isin(("calibration_a", "reserve"))]
    return mb.scored_prompts(a_items, chat) + mb.label_items(a_items, mean_pool, chat)


def check_destination(out):
    """Fail before loading a model if either finished or interrupted output exists."""
    for path in (out, out.with_name(out.name + ".partial")):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing results: {path}")


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    args = ap.parse_args()
    if args.tiny:
        raise SystemExit("the battery needs the frozen sets of the main model; tests/test_m3_battery.py covers the code")
    out = ROOT / "outputs/e3_errors" / args.model / "battery"
    check_destination(out)
    if not hasattr(hooks, "position_mask"):
        raise SystemExit("The full mechanism battery requires hooks.position_mask")
    commit = common.guard(args)
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role.isin(("calibration_a", "calibration_b"))].reset_index(drop=True)
    pool = final[final.role.isin(mb.POOL_ROLES)].reset_index(drop=True)
    m1_dir = common.out_dir(args, "screen")
    m1_meta = json.loads((m1_dir / "meta.json").read_text(encoding="utf-8"))
    control = list(common.load_selection(m1_dir, m1_meta)["control"])
    keys = probe_lesions.dose_sets(pd.read_csv(m1_dir / "screen.csv"),
                                   pd.read_csv(ROOT / "outputs/e3_knowledge" / args.model / "screen/screen.csv"))
    if keys["C4"] != control or keys["K8"] != list(rdt.K8):
        raise SystemExit("the rankings do not start with the frozen sets")
    sets = {k: [Component.parse(x) for x in keys[k]] for k in ("C4", "K8", "C16", "K30")}

    model, tok, chat, means, wiki = common.load(args)
    t0 = time.time()
    # task-matched means: every battery prompt of the calibration_a concepts, as the chat template writes it
    mean_prompts = task_mean_prompts(items, pool, chat)
    texts = [tok.apply_chat_template([{"role": "user", "content": p.text}], tokenize=False, add_generation_prompt=True)
             if chat else p.text for p in mean_prompts]
    from lesion import means as means_mod, models

    task_means = means_mod.collect_means(model, tok, texts, max_tokens=1024, batch_size=8, model_name=args.model,
                                         revision=models.MODELS[args.model][1])
    task = functools.partial(hooks.ablate, model, means=task_means, alpha=0.0, mode="mean")
    positional = True
    plan = states(sets, wiki, task, positional)

    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    tables, timing = {}, {}
    for name, comps, ablate, region in plan:
        t1 = time.time()
        res = run_state(model, tok, chat, items, pool, comps, ablate, region, name, heads=sets["C4"])
        for t, df in res.items():
            if df is not None:
                tables.setdefault(t, []).append(df)
        timing[name] = round(time.time() - t1, 1)
        print(f"{name}: {timing[name]} s", flush=True)
        for t, dfs in tables.items():          # save as we go, so an interruption keeps the finished states
            checkpoint = partial / f"{t}.parquet.tmp"
            pd.concat(dfs, ignore_index=True).to_parquet(checkpoint, index=False)
            checkpoint.replace(partial / f"{t}.parquet")
    task_means.save(partial / "task_means.pt")
    runinfo.write_meta(partial, model=args.model, dtype=args.dtype, means_path=str(args.means),
                       means_sha256=runinfo.sha256(args.means), task_means_n_prompts=len(texts),
                       item_roles=dict(zip(items.item_id, items.role)),
                       task_mean_foil_ids=sorted({p.foil_id for p in mean_prompts if isinstance(p, mb.LabelItem) and p.foil_id}),
                       sets={k: [c.key for c in v] for k, v in sets.items()}, states=[s[0] for s in plan],
                       positional=positional, resume_commit=commit, timing=timing,
                       final_items_sha256=runinfo.sha256(m3_items.OUT / "final_items.csv"), n_items=len(items),
                       seconds=round(time.time() - t0, 1), complete=True)
    partial.rename(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
