"""Build a reproducible WordNet review pool and a fresh, family-separated draft battery. No model is run."""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import fine_semantics as fs, runinfo, semantic_diversity as sd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, default=ROOT / "outputs/title_gap_sources")
    ap.add_argument("--out", type=Path, default=ROOT / "data/materials/title_gap/fine")
    ap.add_argument("--seed", type=int, default=1910)
    args = ap.parse_args()
    if (args.out / "items.csv").exists():
        ap.error("items.csv exists, use a new draft directory to preserve reviewed materials")
    corpus = args.sources / "corpora/wordnet.zip"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    if not corpus.exists():
        shutil.copyfile(args.sources / "wordnet.zip", corpus)
    nltk.data.path.insert(0, str(args.sources.resolve()))
    from nltk.corpus import wordnet as wn
    norms = sd.load_norms(args.sources / "SemD.xlsx")
    frequency = norms.set_index("word").log_bnc_freq.to_dict()
    pool = fs.wordnet_pool(wn, fs.old_words(ROOT / "data/materials"), frequency, norms.set_index("word").semd.to_dict())
    items = fs.select_pool(pool, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # Restricted numeric norms and the complete pool stay outside the public materials.
    pool.to_csv(args.sources / "fine_pool.csv", index=False)
    items.drop(columns=["frequency", "semd"]).to_csv(args.out / "items.csv", index=False)
    with zipfile.ZipFile(corpus) as z:
        (args.out / "WORDNET-LICENSE.txt").write_bytes(z.read("wordnet/LICENSE"))
    meta = dict(status="draft_for_review", seed=args.seed, wordnet=wn.get_version(),
                sources={p.name: runinfo.sha256(p) for p in (corpus, args.sources / "SemD.xlsx")},
                counts=items.groupby(["split", "frequency_bin"]).size().unstack().to_dict(),
                joint_counts=items.groupby(["split", "frequency_bin", "semd_bin"]).size().reset_index(name="n").to_dict("records"),
                items_sha256=runinfo.sha256(args.out / "items.csv"))
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(items.groupby(["split", "frequency_bin"]).size().to_string())


if __name__ == "__main__":
    main()
