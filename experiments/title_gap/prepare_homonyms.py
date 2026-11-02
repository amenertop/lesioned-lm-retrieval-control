"""Combine authored contexts with identified human meanings and human association responses. No model is run."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import homonym_items as hi, runinfo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, default=ROOT / "outputs/title_gap_sources")
    ap.add_argument("--out", type=Path, default=ROOT / "data/materials/title_gap/homonyms")
    ap.add_argument("--seed", type=int, default=1910)
    args = ap.parse_args()
    if (args.out / "items.csv").exists():
        ap.error("items.csv exists, use a new draft directory to preserve reviewed materials")
    table = pd.read_csv(args.out / "contexts.tsv", sep="\t")
    contexts = {word: {r.sense: (r.low, r.high) for r in group.itertuples()} for word, group in table.groupby("word")}
    norms = pd.read_csv(args.sources / "uk_norms.csv")
    responses = pd.read_csv(args.sources / "uk_responses.csv")
    items, audit = hi.select_normed(norms, responses, contexts, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    items.to_csv(args.out / "items.csv", index=False)
    audit.to_csv(args.sources / "homonym_audit.csv", index=False)
    lexicon = [{"word": w, "answer": a, "senses": list(s)} for (w, a), s in hi.response_lexicon(responses).items()]
    (args.sources / "homonym_lexicon.json").write_text(json.dumps(lexicon, indent=1) + "\n", encoding="utf-8")
    meta = dict(status="draft_for_review", seed=args.seed, families=int(items.family.nunique()),
                source_modality="spoken British English, homographs only", source_doi="10.5334/joc.194",
                sources={p.name: runinfo.sha256(p) for p in (args.sources / "uk_norms.csv", args.sources / "uk_responses.csv")},
                items_sha256=runinfo.sha256(args.out / "items.csv"), contexts_sha256=runinfo.sha256(args.out / "contexts.tsv"))
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(items.groupby(["split", "dominance"]).size().to_string())


if __name__ == "__main__":
    main()
