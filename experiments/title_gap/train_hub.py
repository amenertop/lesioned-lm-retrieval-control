"""Train component gates on screening concepts. Optional seeds share one model load, with separate outputs."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
import common
from lesion import fine_semantics, fine_objective, models, neutral_text, runinfo, semantic_diversity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    common.add_args(ap)
    ap.set_defaults(batch_size=16)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--norms", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True, help="JSON with members and model revision")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs='+', help="run independent seeds in separate subdirectories with one model load")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--mode", choices=("hub", "comprehension"), default="hub")
    ap.add_argument("--naming-readout", choices=("target", "accepted_vs_coordinate"), default="target")
    ap.add_argument("--sparsity", type=float, default=0.03)
    ap.add_argument("--checkpointing", action="store_true", help="reuse P14 non-reentrant checkpointing")
    ap.add_argument("--resident-means", action="store_true", help="keep fixed gate means on the model device")
    ap.add_argument("--check-resident-means", action="store_true", help="check one paired forward/backward before training")
    args = ap.parse_args()
    if args.naming_readout != "target" and args.mode != "hub":
        ap.error("the relative-candidate plan adds only a hub branch")
    seeds = args.seeds or [args.seed]
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        ap.error('seeds must be distinct nonnegative integers')
    commit = common.guard(args)
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("training output must be empty, use a different run directory")
    if args.tiny and not args.out.resolve().is_relative_to((ROOT / "outputs/title_gap_tiny").resolve()):
        ap.error("tiny runs must use outputs/title_gap_tiny")
    items = fine_semantics.load_items(args.items, "screening", rehearsal=args.tiny)
    items = semantic_diversity.match_words(items, semantic_diversity.load_norms(args.norms))
    items["frequency"] = items.log_bnc_freq
    if items.frequency.isna().any():
        ap.error("missing screening frequencies")
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    if pool["model"] != args.model or pool["revision"] != models.MODELS[args.model][1]:
        ap.error("candidate pool belongs to a different model")
    model, tok, chat, means, _ = common.load(args)
    data = fine_objective.FineData(items, tok, chat, args.naming_readout)
    resource_check = None
    if args.check_resident_means:
        resource_check = fine_objective.check_resident_means(model, means, data, pool['members'],
            min(args.batch_size, len(items)), args.checkpointing)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'resource-check.json').write_text(json.dumps(resource_check, indent=2)+'\n', encoding='utf-8')
        print('Resident means check:', resource_check, flush=True)
        if not resource_check['passed'] or resource_check['speedup'] <= 1:
            print('Transfer check or timing gain did not pass, retain original host means', flush=True)
            args.resident_means = False
    if args.resident_means:
        means = fine_objective.resident_means(model, means)
    seqs = [tok.encode(t, add_special_tokens=False)[:128] for t in neutral_text.load()[40:120]]
    for seed in seeds:
        out = args.out / f'seed{seed}' if args.seeds else args.out
        print(f'Train {args.mode}, seed {seed}', flush=True)
        gates, log = fine_objective.train(model, means, data, pool["members"], seqs, args.steps,
                                        min(args.batch_size, len(items)), seed, args.mode, args.sparsity,
                                        checkpointing=args.checkpointing)
        out.mkdir(parents=True, exist_ok=True)
        gates.to_csv(out / "gates.csv", index=False)
        log.to_csv(out / "log.csv", index=False)
        meta = dict(complete=True, commit=commit, model=args.model, revision=models.MODELS[args.model][1],
                    tiny=args.tiny, mode=args.mode, seed=seed, steps=args.steps, sparsity=args.sparsity,
                    naming_readout=args.naming_readout,
                    objective_plan_sha256=runinfo.sha256(ROOT / 'docs/plans/title-gap-relative-hub.md')
                        if args.naming_readout != 'target' else None,
                    batch_size=min(args.batch_size, len(items)), dtype=args.dtype, moe=args.moe,
                    checkpointing=args.checkpointing,
                    resident_means=args.resident_means, resource_check=resource_check,
                    items_sha256=runinfo.sha256(args.items), norms_sha256=runinfo.sha256(args.norms),
                    manifest_sha256=runinfo.sha256(args.items.with_name("manifest.json")),
                    means_sha256=runinfo.sha256(args.means), pool_sha256=runinfo.sha256(args.pool),
                    gates_sha256=runinfo.sha256(out / "gates.csv"),
                    scales=data.scales.tolist(), selection_split="screening")
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
