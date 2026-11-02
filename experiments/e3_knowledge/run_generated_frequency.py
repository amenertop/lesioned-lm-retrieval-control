"""What the lesioned model says on the held-out naming concepts (exploratory, user 2026-11-02).

    python experiments/e3_knowledge/run_generated_frequency.py --model qwen3-30b-a3b-instruct \
        --means outputs/means/qwen3-30b-a3b-instruct@0d7cf23__42ae809__fp16.pt --dtype float16 --moe sorted

The frequency effect of WR8 (Results 3.1) was measured on the log-probability of the correct name. This run asks
whether it also shows in the words the model produces: greedy answers on the 240 held-out naming concepts of the
dissociation test (same prompts: lesion.naming.build_prompts with the knowledge-test demos), for the intact model, WR8
and SC4 (mean ablation, alpha 0, all positions). analyze_generated_frequency.py grades the answers case-insensitively
and compares the accuracy loss on rare and common names (the Zipf median of the knowledge screen).

Writes outputs/e3_knowledge/<model>/generated_frequency/{generated.parquet, meta.json}; a finished state is kept.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
import common  # noqa: E402
from lesion import knowledge_items as ki, m3_battery as mb, models, naming, runinfo  # noqa: E402
from lesion.components import Component  # noqa: E402

FREEZE_TAG = "analysis-frozen-knowledge"
MEANS_SHA256 = "98b3865a71b0c8e85c530cb78d5438f9223a828fdf77c907315785b1f1de0d52"
MEMBERS = ROOT / "docs/paper-prep/component-members.csv"
BATCH, MAX_NEW = 16, 8


def component_sets():
    members = pd.read_csv(MEMBERS, keep_default_na=False)
    return {code: list(members[members.paper_code == code].sort_values("order").component) for code in ("WR8", "SC4")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(ap)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.tiny:
        raise SystemExit("real model only")
    if runinfo.sha256(args.means) != MEANS_SHA256:
        raise SystemExit("means differ from the confirmatory runs")
    commit = common.guard(args)
    # The held-out split was opened at the knowledge-test freeze by the dissociation test; this exploratory rerun reads the
    # same rows again without that tag check, as run_demo_format.family_items does for the family concepts.
    items = pd.read_csv(ki.OUT / "items.csv")
    test = items[items.split == "test"].reset_index(drop=True)
    if len(test) != 240:
        raise SystemExit(f"expected 240 held-out naming concepts, found {len(test)}")
    sets = component_sets()
    out = args.out or ROOT / "outputs/e3_knowledge" / args.model / "generated_frequency"
    out.mkdir(parents=True, exist_ok=True)
    model, tok, chat, _, ablate = common.load(args)
    prompts = naming.build_prompts(test, pd.read_csv(ki.OUT / "demos.csv"), chat)
    path = out / "generated.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    timing = {}
    for name, comps in [("intact", None), ("WR8", sets["WR8"]), ("SC4", sets["SC4"])]:
        if len(done) and name in set(done.state):
            continue
        start = time.monotonic()
        if comps is None:
            gen = mb.greedy_answers(model, tok, prompts, chat, BATCH, MAX_NEW)
        else:
            with ablate([Component.parse(c) for c in comps]):
                gen = mb.greedy_answers(model, tok, prompts, chat, BATCH, MAX_NEW)
        gen = gen.assign(state=name, target=[p.target for p in prompts])
        done = pd.concat([done, gen], ignore_index=True)
        done.to_parquet(path, index=False)
        timing[name] = round(time.monotonic() - start, 1)
        print(f"{name}: {timing[name]} s", flush=True)
    (out / "meta.json").write_text(json.dumps(dict(
        stage="generated_frequency", model=args.model, revision=models.MODELS[args.model][1], commit=commit,
        dtype=args.dtype, moe_forward=args.moe, means_sha256=runinfo.sha256(args.means), freeze_tag=FREEZE_TAG,
        sets=sets, batch=BATCH, max_new_tokens=MAX_NEW, n_concepts=len(test),
        items_sha256=runinfo.sha256(ki.OUT / "items.csv"), timing=timing, status="done"), indent=1), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
