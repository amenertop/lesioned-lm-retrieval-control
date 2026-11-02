"""Score a reviewed fine-comprehension or homonym battery against an explicit state file."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))
import common
from lesion import fine_semantics, homonym_items, models, neutral_text, runinfo, title_gap_scoring, title_gap_states


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    common.add_args(ap)
    ap.add_argument("--battery", required=True, choices=("fine", "homonyms"))
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--split", choices=("screening", "dev", "test"), required=True)
    ap.add_argument("--freeze-tag")
    ap.add_argument("--protocol", choices=("relative-hub",), help="separate fine-battery mechanism freeze")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.protocol and (args.battery != 'fine' or args.split != 'test'):
        ap.error('an explicit mechanism protocol requires the final fine battery')
    if args.ablation_mode != "mean":
        ap.error("this plan specifies mean lesions")
    if args.tiny and not args.out.resolve().is_relative_to((ROOT / "outputs/title_gap_tiny").resolve()):
        ap.error("tiny runs must use outputs/title_gap_tiny")
    commit = common.guard(args)
    state_source = json.loads(args.states.read_text(encoding="utf-8"))
    title_gap_states.check_source(state_source, args.model, models.MODELS[args.model][1], args.tiny)
    if state_source.get('means_sha256') and state_source['means_sha256'] != runinfo.sha256(args.means):
        ap.error('state selection and scoring use different component means')
    protocol = None
    if args.split == 'test':
        runinfo.check_analysis_freeze(args.freeze_tag)
        protocol_path = ROOT / f'docs/analysis-plan/title-gap-{args.protocol or args.battery}.json'
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        if (protocol.get('status') != 'frozen' or protocol.get('model') != args.model
                or protocol.get('revision') != models.MODELS[args.model][1]
                or protocol.get('dtype') != args.dtype or protocol.get('moe') != args.moe
                or runinfo.sha256(args.items) != protocol.get('items_sha256')
                or runinfo.sha256(args.states) not in protocol.get('states_sha256', {}).values()
                or runinfo.sha256(args.means) != protocol.get('means_sha256')):
            ap.error('test inputs differ from the concrete battery freeze')
        for path, digest in protocol['sources_sha256'].items():
            if runinfo.sha256(ROOT / path) != digest:
                ap.error(f'freeze source changed: {path}')
    module = fine_semantics if args.battery == "fine" else homonym_items
    items = module.load_items(args.items, args.split, args.freeze_tag, rehearsal=args.tiny)
    if args.limit:
        if not args.tiny:
            ap.error("limit is for tiny rehearsals only")
        items = items.head(args.limit)
    model, tok, chat, means, _ = common.load(args)
    if args.battery == "fine":
        naming = (fine_semantics.naming_prompts(items) + fine_semantics.naming_prompts(items, copy=True)
                  + fine_semantics.word_answer_prompts(items))
        labels = fine_semantics.label_prompts(items)
    else:
        naming, labels = homonym_items.prompts(items)
    states = state_source["states"]
    manifest = args.items.with_name("manifest.json")
    texts = neutral_text.load()[:2 if args.tiny else 40]
    seqs = [tok.encode(t, add_special_tokens=False)[:128] for t in texts]
    provenance = dict(commit=commit, model=args.model, revision=models.MODELS[args.model][1], tiny=args.tiny,
                      dtype=args.dtype, moe=args.moe, split=args.split, freeze_tag=args.freeze_tag,
                      items_sha256=runinfo.sha256(args.items), means_sha256=runinfo.sha256(args.means),
                      manifest_sha256=runinfo.sha256(manifest) if manifest.exists() else None,
                      states_sha256=runinfo.sha256(args.states), battery=args.battery)
    if protocol is not None:
        provenance['protocol_sha256'] = runinfo.sha256(protocol_path)
        if args.protocol:
            provenance['protocol_path'] = protocol_path.relative_to(ROOT).as_posix()
    naming_batches = protocol.get('naming_batches') if protocol is not None and args.battery == 'fine' else None
    title_gap_scoring.run_states(model, tok, means, naming, labels, states, args.out, provenance, chat,
                                 batches=(4, 3, 2) if args.tiny else (16, 11, 7), neutral_sequences=seqs,
                                 naming_batches=naming_batches)


if __name__ == "__main__":
    main()
