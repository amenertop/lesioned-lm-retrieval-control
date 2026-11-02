"""Definition-naming items for the knowledge test (user decisions 2026-09-28; docs/plans/decisions-pending.md).

The model reads a WordNet 3.0 gloss ("toothless mammal ... long snout for feeding on ants") and should name
the word. Targets run from very common to very rare (SUBTLEX-US Zipf frequency), so that an intact model
shows a frequency effect and knowledge lesions can hit rare words first, as in semantic dementia.

A target qualifies when
- it is the first lemma of its synset, and any other single-word lemma is in SUBTLEX-US and at least 1 Zipf
  unit rarer (so there is one clearly best answer; a single-word rival of unknown frequency drops the item;
  multiword alternatives such as "domestic dog" are allowed),
- the synset is the word's first noun sense and really its main meaning: either the word has a single noun
  sense, or this sense has the highest SemCor tag count among its noun senses and has been tagged at least
  once (WordNet orders untagged senses almost arbitrarily: "tuna" comes first as a prickly pear),
- the word has more noun senses than verb, adjective and adverb senses together (mainly used as a noun),
- the synset is a concrete non-human thing (animal or plant, artifact, food, natural object: the usual
  naming-test categories; people are left out),
- the gloss has 6 to 30 words and gives no word starting with the target (e.g. "boxes" for "box") and,
  for targets of five letters or more, no word sharing its first five letters.

Build:  python -m lesion.knowledge_items   (needs data/raw/subtlex_us and data/raw/nltk_data; see SOURCES.md)
Writes data/materials/m3_knowledge/{eligible.csv, items.csv, demos.csv}.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUBTLEX = ROOT / "data/raw/subtlex_us/SUBTLEXus74286wordstextversion.tsv"
SUBTLEX_SHA256 = "c5f86f065fc5d057fbf366433b8c5ca550aa7c24e128362dea4394f2b29c86e4"
NLTK_DIR = ROOT / "data/raw/nltk_data"
OUT = ROOT / "data/materials/m3_knowledge"
CATEGORIES = ("organism.n.01", "artifact.n.01", "food.n.02", "natural_object.n.01")
ZIPF_BINS = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 7.5)   # SUBTLEX-US Zipf: 1 = once in 50 million words, 7 = "the"
PER_BIN, N_DEMOS, SEED = 50, 4, 0            # 6 bins x 50 = 300 items (user decision 2026-09-28)
EXTRA_SEEDS = (1,)                            # a second round of 300 new items, same rules (user decision 2026-09-28)
SPLIT_SHARES = (("screening", 0.4), ("dev", 0.2), ("test", 0.4))
STEM = 5


def load_zipf(path=SUBTLEX):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SUBTLEX_SHA256:
        raise RuntimeError(f"{path} differs from the pinned SUBTLEX-US file; see data/raw/SOURCES.md")
    t = pd.read_csv(path, sep="\t", keep_default_na=False)
    t = t[t.Word.str.fullmatch(r"[a-z]+")]
    return dict(zip(t.Word, np.log10(t.SUBTLWF.astype(float)) + 3))


def wordnet():
    import nltk

    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    from nltk.corpus import wordnet as wn

    return wn


def category_of(synset, wn):
    wanted = {wn.synset(c) for c in CATEGORIES}
    for path in synset.hypernym_paths():
        for s in path:
            if s in wanted:
                return s.name().split(".")[0]
    return None


def leaks(gloss, word, stem=STEM):
    """Whether the gloss gives the answer away: a word starting with the target ("boxes" for "box") or, for
    targets of `stem` letters or more, a word sharing its first `stem` letters."""
    key = word[:stem]
    return any(w.startswith(word) or (len(word) >= stem and w.startswith(key)) for w in re.findall(r"[a-z]+", gloss.lower()))


def eligible(wn, zipf):
    person = wn.synset("person.n.01")
    rows = []
    for s in wn.all_synsets("n"):
        lemmas = s.lemma_names()
        w = lemmas[0]
        if not (w.isalpha() and w.islower() and len(w) >= 3) or w not in zipf:
            continue
        single = [l.lower() for l in lemmas[1:] if "_" not in l and "-" not in l]
        rivals = [l for l in single if l not in zipf or zipf[l] > zipf[w] - 1]
        if rivals:
            continue
        nouns = wn.synsets(w, pos="n")
        if nouns[0] != s or len(nouns) <= sum(len(wn.synsets(w, pos=p)) for p in ("v", "a", "s", "r")):
            continue
        if len(nouns) > 1:
            counts = [sum(l.count() for l in n.lemmas() if l.name().lower() == w) for n in nouns]
            if counts[0] < 1 or counts[0] < max(counts):
                continue
        if any(person in path for path in s.hypernym_paths()):
            continue
        gloss = s.definition()
        n_words = len(gloss.split())
        if not 6 <= n_words <= 30 or leaks(gloss, w):
            continue
        cat = category_of(s, wn)
        if cat is None:
            continue
        rows.append(dict(target=w, definition=gloss, zipf=round(float(zipf[w]), 4), category=cat, synset=s.name(),
                         n_noun_senses=len(nouns), n_other_senses=sum(len(wn.synsets(w, pos=p)) for p in ("v", "a", "s", "r")),
                         n_gloss_words=n_words))
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def sample(pool, seed=SEED, per_bin=PER_BIN, n_demos=N_DEMOS, exclude=(), first_id=0):
    """(items, demos): per_bin targets from each Zipf bin; demos are common words used only as nouns, kept out
    of the items. With exclude (targets already used) and n_demos=0, draws a further round of items whose ids
    continue from first_id."""
    rng = np.random.default_rng(seed)
    pool = pool.assign(zipf_bin=pd.cut(pool.zipf, ZIPF_BINS, right=False, labels=False))
    pool = pool[~pool.target.isin(set(exclude))]
    common = pool[(pool.zipf >= 4.5) & (pool.n_other_senses == 0)]
    demos = common.iloc[rng.choice(len(common), n_demos, replace=False)].sort_values("target") if n_demos else common.iloc[:0]
    rest = pool.drop(demos.index)
    picks = []
    for _, g in rest.groupby("zipf_bin"):
        chosen = g.iloc[rng.choice(len(g), min(per_bin, len(g)), replace=False)].copy()
        # split within each frequency bin: 40% screening, 20% dev, 40% held-out test (each item is its own concept)
        cuts = np.cumsum([round(share * len(chosen)) for _, share in SPLIT_SHARES])[:-1]
        chosen["split"] = np.concatenate([[name] * n for (name, _), n in
                                          zip(SPLIT_SHARES, np.diff(np.concatenate([[0], cuts, [len(chosen)]])))])
        picks.append(chosen)
    items = pd.concat(picks).sort_values(["zipf", "target"]).reset_index(drop=True)
    items.insert(0, "item_id", [f"k{first_id + i:03d}" for i in range(len(items))])
    return items, demos.drop(columns="zipf_bin").reset_index(drop=True)


def load_items(split=None, freeze_tag=None, path=None):
    """Knowledge-test items, optionally one split. The held-out test split needs the analysis-plan freeze tag
    (as for M1: lesion.runinfo.check_analysis_freeze)."""
    items = pd.read_csv(path or OUT / "items.csv")
    if split is None:
        if freeze_tag is None:
            raise PermissionError("all items include the held-out test split; pass split= or the freeze tag")
    elif split not in {s for s, _ in SPLIT_SHARES}:
        raise ValueError(f"unknown split {split!r}")
    if split in (None, "test"):
        from . import runinfo

        runinfo.check_analysis_freeze(freeze_tag)
    return items if split is None else items[items.split == split].reset_index(drop=True)


def build(out=OUT):
    wn = wordnet()
    pool = eligible(wn, load_zipf())
    items, demos = sample(pool)
    for seed in EXTRA_SEEDS:        # further rounds: new targets only, ids continue, demos unchanged
        more, _ = sample(pool, seed=seed, n_demos=0, exclude=set(items.target) | set(demos.target), first_id=len(items))
        items = pd.concat([items, more], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out / "eligible.csv", index=False, lineterminator="\n")
    items.to_csv(out / "items.csv", index=False, lineterminator="\n")
    demos.to_csv(out / "demos.csv", index=False, lineterminator="\n")
    return pool, items, demos


if __name__ == "__main__":
    pool, items, demos = build()
    print(f"eligible {len(pool)}; items {len(items)}; demos {', '.join(demos.target)}")
    print(items.groupby(pd.cut(items.zipf, ZIPF_BINS, right=False), observed=True).size())
