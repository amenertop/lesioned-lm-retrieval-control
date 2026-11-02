"""Locate production-control candidates on development families, then check held-aside development families."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
import common
from lesion import (attribution, fine_semantics, homonym_items, models, neutral_text, production_control as pc,
                    runinfo, title_gap_analysis as ta, title_gap_scoring as ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    common.add_args(ap)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--partition", type=Path, required=True)
    ap.add_argument("--fine-items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    commit = common.guard(args)
    if args.tiny and not args.out.resolve().is_relative_to((ROOT / "outputs/title_gap_tiny").resolve()):
        ap.error("tiny runs must use outputs/title_gap_tiny")
    items = homonym_items.load_items(args.items, "dev", rehearsal=args.tiny)
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    if partition != dict(items_sha256=runinfo.sha256(args.items), **pc.split_families(items)):
        ap.error("family partition differs from the fixed rule or materials")
    if args.tiny:
        partition = {**partition, "screening": partition["screening"][:2], "dev": partition["dev"][:2]}
    scr, dev = (items[items.family.isin(partition[key])] for key in ("screening", "dev"))
    model, tok, chat, means, _ = common.load(args)
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = dict(commit=commit, model=args.model, revision=models.MODELS[args.model][1], tiny=args.tiny,
                      dtype=args.dtype, moe=args.moe, split="dev", battery="homonyms", purpose="production_search",
                      items_sha256=runinfo.sha256(args.items), means_sha256=runinfo.sha256(args.means),
                      manifest_sha256=runinfo.sha256(args.items.with_name("manifest.json")),
                      partition=partition, partition_sha256=runinfo.sha256(args.partition))
    record = args.out / "meta.json"
    if record.exists() and json.loads(record.read_text(encoding="utf-8"))["provenance"] != provenance:
        ap.error("search directory belongs to different inputs")
    record.write_text(json.dumps(dict(status="running", provenance=provenance), indent=2) + "\n", encoding="utf-8")
    texts = neutral_text.load()[:2 if args.tiny else 40]
    seqs = [tok.encode(t, add_special_tokens=False)[:128] for t in texts]
    batches = (4, 3, 2) if args.tiny else (16, 11, 7)
    jobs, weights = pc.attribution_prompts(scr)
    estimated = attribution.naming_attribution(model, tok, jobs, chat, means, weights, batch_size=2, word_only=True)
    estimated.to_csv(args.out / "attribution.csv", index=False)
    keys = estimated.sort_values(["context_damage", "component"], ascending=[False, True]).head(2 if args.tiny else 32).component
    singles = [dict(name="intact", kind="intact", gates={})] + [
        dict(name=k.replace(":", "_"), kind="single", gates={k: 0.}) for k in keys]
    prompts, labels = homonym_items.prompts(scr)
    ts.run_states(model, tok, means, [p for p in prompts if p.condition.endswith("_none")],
                  [p for p in labels if p.fmt.endswith("_none")], singles, args.out / "singles",
                  dict(provenance, families=partition["screening"]), chat, batches, neutral_sequences=seqs)
    screened, _ = pc.screen_summary(args.out / "singles", scr)
    screened.to_csv(args.out / "screen.csv", index=False)
    states, ranking = pc.joint_states(screened, singles)
    selected, copy_checks = None, []
    if len(states) > 1:
        prompts, labels = homonym_items.prompts(dev)
        ts.run_states(model, tok, means, prompts, labels, states, args.out / "dev",
                      dict(provenance, families=partition["dev"]), chat, batches, neutral_sequences=seqs)
        compared, _ = pc.screen_summary(args.out / "dev", dev)
        compared.to_csv(args.out / "development.csv", index=False)
        fine = fine_semantics.load_items(args.fine_items, "dev", rehearsal=args.tiny)
        fine = fine.head(2) if args.tiny else fine
        lookup = {s["name"]: s for s in states}
        for name in pc.rank_joint(compared, states).state:
            copy_out = args.out / "copy" / name
            ts.run_states(model, tok, means, fine_semantics.naming_prompts(fine, copy=True),
                          fine_semantics.label_prompts(fine), [states[0], lookup[name]], copy_out,
                          dict(provenance, battery="fine", items_sha256=runinfo.sha256(args.fine_items),
                               manifest_sha256=runinfo.sha256(args.fine_items.with_name("manifest.json"))),
                          chat, batches, neutral_sequences=seqs)
            tables, _ = ta.read_run(copy_out)
            generated = tables["generated"].merge(fine[["item_id", "target"]], on="item_id", validate="many_to_one")
            generated["correct"] = generated.answer.str.lower().str.strip().str.strip('\"\'.,! ') == generated.target.str.lower()
            accuracy = generated.groupby("state").correct.mean()
            drop = float(accuracy["intact"] - accuracy[name])
            copy_checks.append(dict(state=name, copy_drop=drop))
            if drop <= .05:
                selected = lookup[name]
                break
    result = dict(status="ok" if selected else "not_found", state=selected, single_ranking=ranking,
                  copy_checks=copy_checks, candidates=states, provenance=provenance)
    (args.out / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    record.write_text(json.dumps(dict(status="done", provenance=provenance,
                                      selection_sha256=runinfo.sha256(args.out / "selection.json")), indent=2) + "\n", encoding="utf-8")
    print("production selection:", result["status"], None if selected is None else selected["name"], flush=True)


if __name__ == "__main__":
    main()
