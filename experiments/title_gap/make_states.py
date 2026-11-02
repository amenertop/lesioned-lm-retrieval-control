"""Prepare explicit lesion states without loading model weights. Random states require a saved means file."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import models, runinfo, title_gap_states as ts, weight_masks
from lesion.components import all_components
from lesion.means import Means


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3-30b-a3b-instruct")
    ap.add_argument("--config", type=Path, help="local pinned config directory")
    ap.add_argument("--kind", required=True, choices=("references", "broad", "random", "variants", "combination", "pool", "learned", "weight_masks"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 12, 20, 28, 36, 44])
    ap.add_argument("--weight-families", nargs="+", default=["attention", "ffn"], choices=(*weight_masks.FAMILIES, *weight_masks.PROJECTIONS))
    ap.add_argument("--severities", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--source", type=Path, help="input states or learned gates CSV")
    ap.add_argument("--additional-sources", type=Path, nargs='+', default=[], help="additional learned gate files for one shared development run")
    ap.add_argument("--state", help="state selected from source")
    ap.add_argument("--second-state", help="comprehension state for the full factorial")
    ap.add_argument("--extra", nargs="*", default=[], help="independently selected add-one component keys")
    ap.add_argument("--means", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1910)
    ap.add_argument("--band", choices=("early", "middle", "late"), default="middle")
    args = ap.parse_args()
    config = AutoConfig.from_pretrained(args.config or models.local_dir(args.model), local_files_only=True)
    states = [dict(name="intact", kind="intact", gates={})]
    source_hash = None
    training_sources = []
    if args.additional_sources and args.kind != 'learned':
        ap.error('additional sources apply only to learned gate runs')
    if args.kind == "references":
        if args.model != "qwen3-30b-a3b-instruct":
            ap.error("a second model needs independently selected components, never copied coordinates")
        members = pd.read_csv(ROOT / "docs/paper-prep/component-members.csv")
        groups = {name: members.loc[members.paper_code == name, "component"].tolist() for name in ("WR8", "WR30", "SC4")}
        states = ts.reference_states(groups)
    elif args.kind == "broad":
        states += ts.broad_states(config)
    elif args.kind == "weight_masks":
        states += weight_masks.grid(config, args.layers, args.weight_families, args.severities, (args.seed, args.seed + 1))
    elif args.kind == "pool":
        layers = ts.layer_bands(config)[args.band]
        members = [c.key for kind in ("expert", "head") for c in all_components(config, kind) if c.layer in layers]
        payload = dict(model=args.model, revision=models.MODELS[args.model][1], band=args.band, layers=layers,
                       members=members, selection_rule="all experts and heads in fixed layer band, no test scores")
    elif args.kind == "learned":
        settings, identities = None, set()
        for path in [args.source, *args.additional_sources]:
            meta = json.loads((path.parent / "meta.json").read_text(encoding="utf-8"))
            ts.check_source(meta, args.model, models.MODELS[args.model][1])
            if not meta.get("complete") or meta.get("selection_split") != "screening":
                ap.error("learned gates require a complete screening run")
            if meta.get("gates_sha256") != runinfo.sha256(path):
                ap.error("learned gate hash differs from training metadata")
            current = {k: meta[k] for k in ('items_sha256', 'means_sha256', 'norms_sha256', 'dtype', 'moe', 'steps', 'sparsity')}
            readout = meta.get('naming_readout', 'target')
            if readout not in ('target', 'accepted_vs_coordinate'):
                ap.error('unknown training naming readout')
            current.update(naming_readout=readout, objective_plan_sha256=meta.get('objective_plan_sha256'))
            identity = meta['mode'], meta['seed']
            if (settings is not None and settings != current) or identity in identities:
                ap.error('learned sources need matching settings and distinct mode/seed pairs')
            settings = current
            identities.add(identity)
            table = pd.read_csv(path)
            if table.component.duplicated().any():
                ap.error("duplicate learned gates")
            prefix = f'{meta["mode"]}_s{meta["seed"]}_' if args.additional_sources else ''
            if readout != 'target':
                prefix = 'relative_' + (prefix or f'{meta["mode"]}_s{meta["seed"]}_')
            added = [dict(name=prefix+"learned_soft", kind="learned", gates=dict(zip(table.component, table.retained)))]
            order = table.sort_values(["retained", "component"]).component.tolist()
            for size in (4, 8, 16, 32, 64):
                if size <= len(order):
                    added.append(dict(name=f"{prefix}learned_hard_{size}", kind="learned_hard", gates=dict.fromkeys(order[:size], 0.0)))
            for state in added:
                state["training"] = {k: meta[k] for k in ("seed", "mode", "steps", "sparsity", "scales", "gates_sha256")}
                state["training"]["meta_sha256"] = runinfo.sha256(path.parent / "meta.json")
                state["training"].update(naming_readout=readout, objective_plan_sha256=meta.get('objective_plan_sha256'))
            states.extend(added)
            training_sources.append(dict(path=str(path), meta_sha256=runinfo.sha256(path.parent / 'meta.json'),
                                         gates_sha256=runinfo.sha256(path), mode=meta['mode'], seed=meta['seed'],
                                         naming_readout=readout))
        source_hash = runinfo.sha256(args.source)
    else:
        source = json.loads(args.source.read_text(encoding="utf-8"))
        if source.get("model") != args.model or source.get("revision") != models.MODELS[args.model][1]:
            ap.error("source states belong to a different model revision")
        by_name = {s["name"]: s for s in source["states"]}
        target = by_name[args.state]
        if args.kind == "random":
            if not args.means:
                ap.error("random expert matching needs saved routing frequencies")
            means = Means.load(args.means)
            if means.meta.get("model_name") != args.model or means.meta.get("revision") != models.MODELS[args.model][1]:
                ap.error("means were collected on a different model revision")
            states += [target] + ts.matched_states(target, config, means, args.n_random, args.seed)
        elif args.kind == "variants":
            states += [target] + ts.variants(target, args.extra)
        else:
            other = by_name[args.second_state]
            if len(set(target["gates"].values())) != 1 or len(set(other["gates"].values())) != 1:
                ap.error("use uniform-dose factors for this factorial design")
            states = ts.combination_states(target["gates"], other["gates"], next(iter(target["gates"].values())), next(iter(other["gates"].values())))
        source_hash = runinfo.sha256(args.source)
    if args.kind != "pool":
        ts.validate_states(states, config)
        payload = dict(model=args.model, revision=models.MODELS[args.model][1], states=states, seed=args.seed,
                       source_sha256=source_hash, code_commit=runinfo.commit(), role="development_plan_not_test_selection")
        if training_sources:
            payload.update(training_sources=training_sources, means_sha256=settings['means_sha256'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        ap.error("output exists, use a new path to preserve the earlier plan")
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(args.out, len(payload.get("states", payload.get("members", []))))


if __name__ == "__main__":
    main()
