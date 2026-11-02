"""Random-head control for the answer-position SC4 lesion (exploratory, user 2026-11-02).

    python experiments/e3_errors/run_random_answer_heads.py --model qwen3-30b-a3b-instruct \
        --means outputs/means/qwen3-30b-a3b-instruct@0d7cf23__42ae809__fp16.pt --dtype float16 --moe sorted

Lesioning SC4 only at the answer positions raised the miscue cost by 0.40 nats and changed the cue benefit much less
(H4d, Results 3.3). Is that asymmetry specific to SC4, or would any four heads in the same layers produce it when
lesioned at the answer? This run applies the same answer-position lesion (mean ablation, alpha 0, prompt segment
"answer" as in the optogenetics confirmation) to SC4 and to the first N matched random sets of the final family runs
(experiments/title_gap/configs/fine-final-random-SC4_a0.json: four heads, one in each of SC4's layers, seed 1910),
and scores the hinted naming prompts of experiment 3 on its 160 test concepts: no hint with hinted demos, cue and
miscue, candidates target and associate. All states run in this one run with the same prompts and batch layouts, so
SC4 is compared with the random sets within the run. analyze_random_answer_heads.py computes per state the cue
benefit change B, the miscue harm change H and H - B against the intact model, as analyze_optogenetics.py does.

Writes outputs/e3_errors/<model>/random_answer_heads/<state>.parquet (one per state, resumable) and meta.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
import common  # noqa: E402
from lesion import m3_battery as mb, m3_items, m3_scoring as ms, models, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

FINAL_ITEMS = m3_items.OUT / "final_items.csv"
FINAL_SHA256 = "d539310755a31f8155cda5db7ae4920760dd36bbde97442800fa0b17df16b98d"
MEANS_SHA256 = "98b3865a71b0c8e85c530cb78d5438f9223a828fdf77c907315785b1f1de0d52"
RANDOM_CONFIG = ROOT / "experiments/title_gap/configs/fine-final-random-SC4_a0.json"
SC4 = ["head:9:15", "head:27:9", "head:25:24", "head:24:5"]
REGION = ("answer",)
BATCHES = ms.NAMING_BATCHES        # (16, 11, 7), the candidate layouts of experiment 3


def prompts(items, chat):
    fn = items[items.free_naming_ok.astype(bool)]
    return mb.name_hinted_demos_prompts(fn, chat) + ms.build_naming_prompts(fn, ("cue", "miscue"), chat)


def states(n_random):
    cfg = json.loads(RANDOM_CONFIG.read_text(encoding="utf-8"))
    ref = next(s for s in cfg["states"] if s["name"] == "SC4_a0")
    if sorted(ref["gates"]) != sorted(SC4):
        raise SystemExit("SC4 in the random config differs from the frozen SC4")
    rnd = [s for s in cfg["states"] if s.get("kind") == "matched_random"][:n_random]
    if len(rnd) < n_random:
        raise SystemExit(f"only {len(rnd)} random sets in {RANDOM_CONFIG}")
    return [("intact", None), ("SC4_answer", SC4)] + [(f"{s['name']}_answer", list(s["gates"])) for s in rnd]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(ap)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.tiny:
        raise SystemExit("real model only; the scoring path is the one tested by tests/test_m3_battery.py")
    if runinfo.sha256(FINAL_ITEMS) != FINAL_SHA256 or runinfo.sha256(args.means) != MEANS_SHA256:
        raise SystemExit("items or means differ from experiment 3")
    commit = common.guard(args)
    final = pd.read_csv(FINAL_ITEMS, keep_default_na=False)
    items = final[final.role == "test"].reset_index(drop=True)
    plan = states(args.n_random)
    out = args.out or ROOT / "outputs/e3_errors" / args.model / "random_answer_heads"
    out.mkdir(parents=True, exist_ok=True)
    model, tok, chat, _, ablate = common.load(args)
    ps = prompts(items, chat)
    meta = dict(stage="random_answer_heads", model=args.model, revision=models.MODELS[args.model][1], commit=commit,
                dtype=args.dtype, moe_forward=args.moe, means_sha256=runinfo.sha256(args.means), region=list(REGION),
                batches=list(BATCHES), n_prompts=len(ps), n_concepts=len({p.item_id for p in ps}),
                random_config_sha256=runinfo.sha256(RANDOM_CONFIG),
                states={name: comps for name, comps in plan}, timing={})
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for k in ("model", "revision", "dtype", "moe_forward", "means_sha256", "region", "batches", "n_prompts"):
            if old[k] != meta[k]:
                raise SystemExit(f"{out} was started with another {k}")
        meta["timing"] = old.get("timing", {})
    for name, comps in plan:
        path = out / f"{name}.parquet"
        if path.exists():
            continue
        start = time.monotonic()
        ctx = ablate([Component.parse(c) for c in comps]) if comps else None
        tables = []
        if ctx is None:
            for bs in BATCHES:
                tables.append(mb.candidate_scores(model, tok, ps, chat, bs, None))
        else:
            with ctx:
                for bs in BATCHES:
                    tables.append(mb.candidate_scores(model, tok, ps, chat, bs, REGION))
        table = tables[0].copy()
        table["logprob"] = np.mean([t.logprob.to_numpy() for t in tables], axis=0)
        table.assign(state=name).to_parquet(path, index=False)
        meta["timing"][name] = round(time.monotonic() - start, 1)
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{name}: {meta['timing'][name]} s", flush=True)
    meta["status"] = "done"
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
