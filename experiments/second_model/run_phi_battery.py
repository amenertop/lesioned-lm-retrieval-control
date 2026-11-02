"""Phi-3.5-MoE on the main model's clinical battery (exploratory second model; user decision 2026-11-02, plan
docs/plans/phi-battery.md).

    python experiments/second_model/run_phi_battery.py --model phi35-moe-instruct \
        --means outputs/means/phi35-moe-instruct@4368845__fp16.pt --dtype float16

Phi's retrieval set (the six components left of its retrieval search once the two experts it cannot lose are removed,
Supplementary S1.1) and its control set (four heads) were found by the same searches as WR8 and SC4. This run gives
them the battery of Results 3.2 and 3.3 on the same 160 held-out concepts, with the same prompts, scorers, batch
layouts and mean ablation (Phi's own WikiText means). Three parts, each resumable state by state:

  battery    intact, PR6, PC4, PR6+PC4, PR6@0.75, PC4@0.75 (all positions), PR6 and PC4 at the answer positions:
             every candidate-scored prompt (naming without hint, with hinted demos, cue and miscue, the six options
             answered with the word, copying), the six-option and word-to-definition labels, greedy answers (all-position
             states) and the neutral-text loss (first 40 WikiText passages, 256 tokens; all-position states).
  random     PC4 at the answer positions against N random four-head sets, one head in each of PC4's layers (seed
             1910 + i), on the hinted naming prompts only (no hint with hinted demos, cue, miscue).
  generated  greedy answers on the 240 held-out naming concepts of the dissociation test, intact, PR6 and PC4.
  bias       the naming-test log-probabilities of the same 240 concepts for intact, PR6 and PC4, with and without
             Phi's output bias, to measure how much leaving the bias out changed the earlier Phi results.

Every scorer here adds Phi's output bias (output_bias.with_output_bias), which the shared scorers leave out; the
earlier Phi runs (Supplementary S1.1) were scored without it.

Writes outputs/second_model/phi35-moe-instruct/battery/<part>/<state>/*.parquet and meta.json.
"""
import argparse
import functools
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("experiments/e1_screen", "experiments/e3_errors"):
    sys.path.insert(0, str(ROOT / sub))
import common  # noqa: E402
import run_battery  # noqa: E402
from output_bias import with_output_bias  # noqa: E402
from lesion import hooks, knowledge_items as ki, m3_battery as mb, m3_items, m3_optogenetics as mo, m3_scoring as ms  # noqa: E402
from lesion import models, naming, neutral_text, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

MODEL = "phi35-moe-instruct"
PR6 = ["expert:26:6", "expert:27:8", "head:10:5", "head:14:21", "head:17:9", "head:18:26"]
PC4 = ["head:0:2", "head:1:1", "head:6:11", "head:6:28"]
DROPPED = ["expert:3:7", "expert:5:10"]       # the two experts Phi cannot lose (Supplementary S1.1)
FINAL_ITEMS = m3_items.OUT / "final_items.csv"
FINAL_SHA256 = "d539310755a31f8155cda5db7ae4920760dd36bbde97442800fa0b17df16b98d"
TEST_POOL_ROLES = ("test", "calibration_a", "calibration_b", "reserve")
KNOWLEDGE_TAG = "analysis-frozen-knowledge"
N_NEUTRAL, NEUTRAL_TOKENS = 40, 256
SEED = 1910
BATTERY = [("intact", None, 0.0, None), ("PR6", PR6, 0.0, None), ("PC4", PC4, 0.0, None),
           ("PR6+PC4", PR6 + PC4, 0.0, None), ("PR6@0.75", PR6, 0.75, None), ("PC4@0.75", PC4, 0.75, None),
           ("PR6_answer", PR6, 0.0, "answer"), ("PC4_answer", PC4, 0.0, "answer")]


@contextmanager
def _null():
    yield


def random_heads(n, n_heads):
    """n sets of four heads, one in each of PC4's layers, never a PC4 head, no head twice within a set."""
    taken = set(PC4)
    out = []
    for i in range(n):
        rng = np.random.default_rng(SEED + i)
        chosen = []
        for key in PC4:
            layer = Component.parse(key).layer
            pool = [h for h in range(n_heads) if f"head:{layer}:{h}" not in taken | set(chosen)]
            chosen.append(f"head:{layer}:{int(rng.choice(pool))}")
        out.append(chosen)
    return out


def held_out_naming():
    """The 240 held-out concepts of the naming test (lesion.knowledge_items items.csv, split test). The split was opened
    at the knowledge-test freeze by the dissociation test; this exploratory rerun reads the same rows again without
    that tag check, as run_demo_format.family_items does for the family concepts."""
    items = pd.read_csv(ki.OUT / "items.csv")
    test = items[items.split == "test"].reset_index(drop=True)
    if len(test) != 240:
        raise SystemExit(f"expected 240 held-out naming concepts, found {len(test)}")
    return test


def hinted(items, chat):
    fn = items[items.free_naming_ok.astype(bool)]
    return mb.name_hinted_demos_prompts(fn, chat) + ms.build_naming_prompts(fn, ("cue", "miscue"), chat)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(ap)
    ap.add_argument("--parts", default="battery,random,generated,bias")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/second_model" / MODEL / "battery")
    args = ap.parse_args()
    if args.model != MODEL:
        raise SystemExit(f"this run is for {MODEL}")
    if runinfo.sha256(FINAL_ITEMS) != FINAL_SHA256:
        raise SystemExit("final items differ from experiment 3")
    commit = common.guard(args)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    final = pd.read_csv(FINAL_ITEMS, keep_default_na=False)
    items = final[final.role == "test"].reset_index(drop=True)
    pool = final[final.role.isin(TEST_POOL_ROLES)].reset_index(drop=True)
    if args.tiny:      # local smoke test: tiny random Phi, its own means, two concepts with hints and two without
        global PR6, PC4, BATTERY
        PR6, PC4 = ["expert:1:0", "head:1:1"], ["head:0:2", "head:1:3"]
        BATTERY = [(n, None if c is None else (PR6 if "PR6" in n else PC4) + (PC4 if n == "PR6+PC4" else []), a, r)
                   for n, c, a, r in BATTERY]
        from lesion import means as means_mod
        model, tok = models.load_tiny(MODEL)
        chat = models.uses_chat(MODEL)
        means = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)
        fn = items.free_naming_ok.astype(bool)
        items = pd.concat([items[fn].head(2), items[~fn].head(2)]).reset_index(drop=True)
    else:
        model, tok, chat, means, _ = common.load(args)
    n_heads = model.config.num_attention_heads
    rand = random_heads(args.n_random, n_heads)
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    fixed = dict(model=MODEL, revision=models.MODELS[MODEL][1], dtype=args.dtype, moe_forward=args.moe, tiny=args.tiny,
                 means_sha256=None if args.tiny else runinfo.sha256(args.means), final_items_sha256=FINAL_SHA256, PR6=PR6, PC4=PC4,
                 dropped=DROPPED, random_sets=rand, seed=SEED)
    for k, v in fixed.items():
        if k in meta and meta[k] != v:
            raise SystemExit(f"{args.out} was started with another {k}")
    meta.update(fixed, commits=sorted(set(meta.get("commits", [])) | {commit}))
    meta.setdefault("timing", {})

    def save_meta():
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def ablator(alpha):
        return functools.partial(hooks.ablate, model, means=means, alpha=alpha, mode="mean")

    texts = neutral_text.load()[:3 if args.tiny else N_NEUTRAL]
    patch = with_output_bias(model)
    meta["output_bias_added"] = patch.__enter__()
    if "battery" in parts:
        for name, comps, alpha, region in BATTERY:
            d = args.out / "battery" / name
            if (d / "complete.json").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            parsed = None if comps is None else [Component.parse(c) for c in comps]
            res = run_battery.run_state(model, tok, chat, items, pool, parsed, ablator(alpha), region, name,
                                        greedy=True, readouts=False)
            for t in ("candidates", "labels", "greedy"):
                if res[t] is not None:
                    res[t].to_parquet(d / f"{t}.parquet", index=False)
            loss = None
            if region is None:
                if parsed is None:
                    loss = mo.lm_loss(model, tok, texts, max_tokens=NEUTRAL_TOKENS)
                else:
                    with ablator(alpha)(parsed):
                        loss = mo.lm_loss(model, tok, texts, max_tokens=NEUTRAL_TOKENS)
            seconds = round(time.monotonic() - start, 1)
            (d / "complete.json").write_text(json.dumps(dict(state=name, components=comps, alpha=alpha, region=region,
                                                             lm_loss=loss, seconds=seconds)), encoding="utf-8")
            meta["timing"][f"battery/{name}"] = seconds
            save_meta()
            print(f"battery/{name}: {seconds} s, lm_loss {loss}", flush=True)
    if "random" in parts:
        ps = hinted(items, chat)
        plan = [("intact", None), ("PC4_answer", PC4)] + [(f"randomPC4_{i:02d}_answer", s) for i, s in enumerate(rand)]
        for name, comps in plan:
            d = args.out / "random" / name
            if (d / "complete.json").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            region = None if comps is None else "answer"

            def scores(bs):
                return mb.candidate_scores(model, tok, ps, chat, bs, region)
            if comps is None:
                table = run_battery.averaged(scores, run_battery.CANDIDATE_BATCHES, ["logprob"])
            else:
                with ablator(0.0)([Component.parse(c) for c in comps]):
                    table = run_battery.averaged(scores, run_battery.CANDIDATE_BATCHES, ["logprob"])
            table.assign(state=name).to_parquet(d / "candidates.parquet", index=False)
            seconds = round(time.monotonic() - start, 1)
            (d / "complete.json").write_text(json.dumps(dict(state=name, components=comps, seconds=seconds)), encoding="utf-8")
            meta["timing"][f"random/{name}"] = seconds
            save_meta()
            print(f"random/{name}: {seconds} s", flush=True)
    if "generated" in parts:
        test = held_out_naming()
        test = test.head(4) if args.tiny else test
        prompts = naming.build_prompts(test, pd.read_csv(ki.OUT / "demos.csv"), chat)
        for name, comps in [("intact", None), ("PR6", PR6), ("PC4", PC4)]:
            d = args.out / "generated" / name
            if (d / "complete.json").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            if comps is None:
                gen = mb.greedy_answers(model, tok, prompts, chat, 16, 8)
            else:
                with ablator(0.0)([Component.parse(c) for c in comps]):
                    gen = mb.greedy_answers(model, tok, prompts, chat, 16, 8)
            gen.assign(state=name, target=[p.target for p in prompts]).to_parquet(d / "generated.parquet", index=False)
            seconds = round(time.monotonic() - start, 1)
            (d / "complete.json").write_text(json.dumps(dict(state=name, seconds=seconds)), encoding="utf-8")
            meta["timing"][f"generated/{name}"] = seconds
            save_meta()
            print(f"generated/{name}: {seconds} s", flush=True)
    patch.__exit__(None, None, None)
    if "bias" in parts:
        test = held_out_naming()
        test = test.head(4) if args.tiny else test
        prompts = naming.build_prompts(test, pd.read_csv(ki.OUT / "demos.csv"), chat)
        for biased in (False, True):
            for name, comps in [("intact", None), ("PR6", PR6), ("PC4", PC4)]:
                d = args.out / "bias" / f"{name}_{'with' if biased else 'without'}_bias"
                if (d / "complete.json").exists():
                    continue
                d.mkdir(parents=True, exist_ok=True)
                start = time.monotonic()
                with (with_output_bias(model) if biased else _null()):
                    if comps is None:
                        table = naming.averaged_logprobs(model, tok, prompts, chat, (16, 11, 7))
                    else:
                        with ablator(0.0)([Component.parse(c) for c in comps]):
                            table = naming.averaged_logprobs(model, tok, prompts, chat, (16, 11, 7))
                table.assign(state=name, bias=biased).to_parquet(d / "naming.parquet", index=False)
                seconds = round(time.monotonic() - start, 1)
                (d / "complete.json").write_text(json.dumps(dict(state=name, bias=biased, seconds=seconds)), encoding="utf-8")
                meta["timing"][f"bias/{d.name}"] = seconds
                save_meta()
                print(f"bias/{d.name}: {seconds} s", flush=True)
    meta["status"] = "done" if set(parts) == {"battery", "random", "generated", "bias"} else "partial"
    save_meta()
    print("done", flush=True)


if __name__ == "__main__":
    main()
