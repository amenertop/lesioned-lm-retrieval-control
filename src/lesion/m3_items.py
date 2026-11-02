"""M3 and M3-cal concepts for experiment 3: H3a error types and H3b cueing (docs/plans/materials-m2-m3.md v0.3 §5;
docs/plans/experiment3-protocol.md v0.4; user decisions 2026-09-29 and 2026-10-07).

Each concept gives a WordNet 3.0 gloss and five word options ("cannot tell" is added when prompts are built):

| option        | example (camel) | source and rule                                                                 |
| ------------- | --------------- | ------------------------------------------------------------------------------- |
| target        | camel           | the word the gloss defines (concept rules below)                                |
| coordinate    | llama           | a WordNet sibling or cousin in the same everyday category, similar frequency    |
| superordinate | animal          | the everyday category name (LABELS) or, if the gloss says it, a broader one     |
| associate     | desert          | a SWOW-EN18 first response to the target that is a noun outside its category    |
| unrelated     | violin          | a concrete noun from another domain, similar frequency and length (automatic)   |

Concept rules: a SWOW cue taken in its most frequent noun sense in SemCor (WordNet's first noun sense when that is
also the most frequent, else the uniquely most frequent one; 2026-10-07); mostly used as a noun,
counted in SemCor rather than by dictionary senses ("hammer" has as many verb as noun senses but is mostly a noun);
a concrete non-human thing with an everyday category; not one of the naming test's screening words (on which the
knowledge components were chosen) or demos, and not an M1 word. When cues share a sense ("sofa", "couch") the most
frequent is kept, and the others never become options.

The definition is drafted by agent1 from the WordNet gloss (user decision 2026-10-07: verbatim glosses often give
a rare sense or already name the category): 8 to 20 words, the sense people usually mean, and none of the target,
its synonyms or the options (the naming test's leak rule).

Two steps (user decisions 2026-10-07: agent1 may choose options, but only from the databases' candidates):

1. `python -m lesion.m3_items menus` lists, per concept, the candidates each database offers and an automatic
   choice, in a fixed order (Zipf bin, then a seeded shuffle): data/materials/m3/menus.csv.
2. agent1 reviews the concepts in that order, without seeing which will be test or calibration items, and records
   keep or drop, the chosen options and the drafted definition in data/materials/m3/curation.csv.
   `python -m lesion.m3_items items` checks the curation (options from the menus, definition rules), adds the
   unrelated option and fills each bin's test and calibration quotas in order (quotas in proportion to the kept
   concepts per bin), the rest kept as numbered reserves: data/materials/m3/items.csv.

3. After the intact-model validity check (§5.8; experiments/e3_errors/check_items.py on the server),
   `python -m lesion.m3_items final <check dir>` replaces the concepts that failed with reserves and draws the free
   naming concepts: data/materials/m3/final_items.csv, the file the analysis plan freezes.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import knowledge_items as ki

ROOT = Path(__file__).resolve().parents[2]
SWOW_R1 = ROOT / "data/raw/swow_en/extracted/strength.SWOW-EN.R1.20180827.csv"
SWOW_SHA256 = "55afc854d584bb8a4c5c81ef104689e1e961b16d60ae26012ec8698d20703881"
M1_ITEMS = ROOT / "data/materials/m1/items.csv"
M1_SPLIT = ROOT / "data/materials/m1/split.csv"
M1_DEMOS = ROOT / "data/materials/m1/demos.csv"
OUT = ROOT / "data/materials/m3"
M1_SENSE_REVIEW = OUT / "m1_sense_review.csv"

# Everyday category names: (word, WordNet synset, the broader name in the same family or None). A concept's
# category is the nearest one above it, ties going to the earlier entry ("vehicle" before "container" for wagons).
LABELS = (("bird", "bird.n.01", "animal"), ("fish", "fish.n.01", "animal"), ("insect", "insect.n.01", "animal"),
          ("snake", "snake.n.01", "animal"), ("dog", "dog.n.01", "animal"), ("animal", "animal.n.01", None),
          ("tree", "tree.n.01", "plant"), ("flower", "flower.n.01", "plant"), ("plant", "plant.n.02", None),
          ("fruit", "edible_fruit.n.01", "food"), ("vegetable", "vegetable.n.01", "food"), ("meat", "meat.n.01", "food"),
          ("bread", "bread.n.01", "food"), ("cheese", "cheese.n.01", "food"), ("candy", "candy.n.01", "food"),
          ("dessert", "dessert.n.01", "food"), ("sauce", "sauce.n.01", "food"), ("spice", "spice.n.02", "food"),
          ("cereal", "cereal.n.03", "food"), ("drink", "beverage.n.01", None), ("food", "food.n.02", None),
          ("food", "food.n.01", None),
          ("boat", "boat.n.01", "vehicle"), ("ship", "ship.n.01", "vehicle"), ("aircraft", "aircraft.n.01", "vehicle"),
          ("vehicle", "vehicle.n.01", None),
          ("shoe", "shoe.n.01", "clothing"), ("hat", "hat.n.01", "clothing"), ("clothing", "clothing.n.01", None),
          ("jewelry", "jewelry.n.01", None), ("cosmetic", "cosmetic.n.01", None),
          ("gun", "gun.n.01", "weapon"), ("weapon", "weapon.n.01", None),
          ("bed", "bed.n.01", "furniture"), ("seat", "seat.n.03", "furniture"), ("furniture", "furniture.n.01", None),
          ("instrument", "musical_instrument.n.01", None), ("toy", "plaything.n.01", None), ("ball", "ball.n.01", None),
          ("book", "book.n.02", None), ("medicine", "medicine.n.02", None), ("drug", "drug.n.01", None),
          ("room", "room.n.01", None), ("shop", "shop.n.01", None), ("building", "building.n.01", None),
          ("road", "road.n.01", None), ("structure", "structure.n.01", None),
          ("knife", "knife.n.01", "tool"), ("utensil", "kitchen_utensil.n.01", None), ("tool", "tool.n.01", None),
          ("dish", "dish.n.01", "container"), ("bag", "bag.n.01", "container"), ("box", "box.n.01", "container"),
          ("lamp", "lamp.n.01", None), ("computer", "computer.n.01", "machine"),
          ("fabric", "fabric.n.01", None), ("metal", "metal.n.01", None), ("rock", "rock.n.01", None),
          ("container", "container.n.01", None), ("equipment", "equipment.n.01", None),
          ("machine", "machine.n.01", None), ("device", "device.n.01", None))
LABEL_WORDS = {w for w, _, _ in LABELS}
# categories too broad for "same category" to mean a coordinate (a party horn and an elevator are both devices):
# their coordinates must be close WordNet relatives
BROAD = {"device", "machine", "equipment", "structure", "container"}
MIN_ASSOCIATION = 0.02     # associate: at least 2% of first responses to the cue
MAX_COORD_GAP = 0.75       # coordinate: Zipf within this of the target (menus list up to MENU_COORD_GAP)
MENU_COORD_GAP = 1.0
MAX_UNRELATED_GAP = 0.3    # unrelated: Zipf within this of the target, length within MAX_LEN_GAP letters
FALLBACK_UNRELATED_GAP = 0.5
# never unrelated options (besides category names): particles and words that are unpleasant or point to scavenging,
# found in a review of the drawn words on 2026-10-07
UNRELATED_STOPLIST = {"ion", "proton", "electron", "neutron", "fetus", "parasite", "carrion", "article"}
MAX_LEN_GAP = 2
WEAK = 0.02                # unrelated words: SWOW strength below this in both directions
COORD_WEAK = 0.05          # coordinates: below this in both directions, so they stay distinct from associates
MENU_SIZE = 10
ASSOCIATE_MENU_SIZE = 15
ZIPF_BINS = (1.0, 3.0, 4.0, 7.5)
N_TEST, N_CAL_A, N_CAL_B_PER_DIRECTION, SEED = 160, 40, 20, 0


# --- sources -------------------------------------------------------------------------------------------------

def load_swow(path=SWOW_R1):
    """SWOW-EN18 first-response strengths: cue, response, strength (lowercase single words only)."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SWOW_SHA256:
        raise RuntimeError(f"{path} differs from the pinned SWOW-EN18 file; see data/raw/SOURCES.md")
    t = pd.read_csv(path, sep="\t", keep_default_na=False)
    t = t.rename(columns={"R1.Strength": "strength"})[["cue", "response", "strength"]]
    t["cue"], t["response"] = t.cue.str.lower().str.strip(), t.response.str.lower().str.strip()
    return t[t.response.str.fullmatch(r"[a-z]+")].reset_index(drop=True)


def excluded_sources():
    """(naming synsets, M1 words) M3 may not reuse (user decisions 2026-10-07): the screening items of the naming
    test and of M1, on which the knowledge and control components were chosen, and both tests' demos, which every
    prompt shows. Their dev and held-out words may be reused. Naming items carry their WordNet sense; M1 words do not."""
    naming = pd.read_csv(ki.OUT / "items.csv")
    naming = pd.concat([naming[naming.split == "screening"], pd.read_csv(ki.OUT / "demos.csv")])
    m1 = pd.read_csv(M1_ITEMS, dtype=str, keep_default_na=False)
    split = pd.read_csv(M1_SPLIT, dtype=str, keep_default_na=False)
    screening = m1[m1.concept_group.isin(split.concept_group[split.split == "screening"])]
    m1_demos = pd.read_csv(M1_DEMOS, dtype=str, keep_default_na=False)
    m1_words = set()
    for table, cols in ((screening, ("probe", "option1", "option2", "word1", "word2", "target", "distractor")),
                        (m1_demos, ("probe", "option_a", "option_b", "word1", "word2", "target", "distractor"))):
        for col in cols:
            m1_words |= {w.lower() for w in table[col] if w}
    return dict(zip(naming.target, naming.synset)), m1_words


def excluded_words():
    naming, m1_words = excluded_sources()
    return set(naming) | m1_words


def load_sense_review(path=M1_SENSE_REVIEW):
    """{(M1 word, synset name): "exclude" | "allow"}: decisions on M3 concepts that share a sense of an M1 word
    other than its main one (agent2 2026-10-07). The original M1 materials record no word senses, so a decision
    can only be inferred from the item; the rule is to exclude unless the item's context rules out the M3
    concept's sense ("lead" next to "pencil" and "graphite" is not a dog lead). Each row states its reason."""
    if not Path(path).exists():
        return {}
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not set(t.decision) <= {"exclude", "allow"}:
        raise ValueError(f"{path}: decision must be exclude or allow")
    return {(r.m1_word, r.synset): r.decision for r in t.itertuples()}


# --- WordNet helpers -----------------------------------------------------------------------------------------

def label_table(wn):
    return [(word, wn.synset(name), parent) for word, name, parent in LABELS]


def hypernym_dist(s):
    """{hypernym: smallest number of steps up from s} (s itself excluded)."""
    out = {}
    for h, d in s.hypernym_distances():
        if h != s and d > 0:
            out[h] = min(d, out.get(h, d))
    return out


def labels_above(s, table):
    """[(steps, order, word, synset name, broader name)] of the everyday categories above s, nearest first."""
    up = hypernym_dist(s)
    return sorted((up[syn], i, word, syn.name(), parent) for i, (word, syn, parent) in enumerate(table) if syn in up)


def category_key(s, table):
    found = labels_above(s, table)
    return found[0][3] if found else None


def semcor_counts(w, wn, pos):
    return sum(l.count() for p in pos for s in wn.synsets(w, pos=p) for l in s.lemmas() if l.name().lower() == w)


def main_sense(w, wn):
    """w's first noun sense when it is also the most frequent in SemCor (tagged at least once), or the only one."""
    nouns = wn.synsets(w, pos="n")
    if not nouns:
        return None
    if len(nouns) > 1:
        counts = [sum(l.count() for l in n.lemmas() if l.name().lower() == w) for n in nouns]
        if counts[0] < 1 or counts[0] < max(counts):
            return None
    return nouns[0]


def concept_sense(w, wn):
    """The sense a concept is taken in: its main sense, or, when WordNet lists a rarer sense first, the noun sense
    tagged most often in SemCor (a unique maximum, tagged at least once). Option words keep the stricter main_sense."""
    s = main_sense(w, wn)
    if s is not None:
        return s
    nouns = wn.synsets(w, pos="n")
    counts = [sum(l.count() for l in n.lemmas() if l.name().lower() == w) for n in nouns]
    if counts and max(counts) >= 1 and counts.count(max(counts)) == 1:
        return nouns[counts.index(max(counts))]
    return None


def mostly_noun(w, wn):
    """More SemCor tokens as a noun than as anything else; without SemCor tokens, at least as many noun senses."""
    if not wn.synsets(w, pos="n"):
        return False
    nouns, others = semcor_counts(w, wn, "n"), semcor_counts(w, wn, "vasr")
    if nouns or others:
        return nouns > others
    return len(wn.synsets(w, pos="n")) >= sum(len(wn.synsets(w, pos=p)) for p in "vasr")


def close_relatives(a, b, steps=2):
    """Whether one is above the other, or they share a hypernym at most `steps` above both."""
    ua, ub = hypernym_dist(a), hypernym_dist(b)
    if a == b or a in ub or b in ua:
        return True
    return any(ua[h] <= steps and ub[h] <= steps for h in set(ua) & set(ub))


def overlaps(w, others):
    return any(w in o or o in w for o in others if o)


def rhymes(a, b, n=4):
    """Shared ending of n letters or more ("footstool", "toadstool"): a sound or spelling response, not a meaning one."""
    return len(a) >= n and len(b) >= n and a[-n:] == b[-n:]


# --- concepts and menus --------------------------------------------------------------------------------------

def concepts(wn, zipf, cues, naming_synsets, m1_words, review=None, log=None):
    """Concept rules of the module docstring, over the SWOW cues; log counts why the others were dropped. When
    several cues share a sense ("sofa", "couch"), the most frequent is kept; the others never become options.

    Excluded sources (excluded_sources) block their word forms, singular or plural ("dinosaurs" for "dinosaur";
    agent2 2026-10-07: a string check alone let plurals through), and their senses: a naming item's recorded
    synset, an M1 word's main sense, and any other sense of an M1 word that `review` marks "exclude". Overlaps
    with another sense of an M1 word that `review` does not cover are dropped and listed in log["pending_review"]."""
    person, table = wn.synset("person.n.01"), label_table(wn)
    review = review or {}
    rows, dropped, pending = [], {}, []
    lemma = lambda x: wn.morphy(x, wn.NOUN) or x
    exclude = set(naming_synsets) | set(m1_words)
    excluded_forms = {lemma(x) for x in exclude} | exclude
    m1_forms = {lemma(x) for x in m1_words} | set(m1_words)
    excluded_senses = {wn.synset(s) for s in naming_synsets.values()}
    excluded_senses |= {s for x in m1_forms for s in [concept_sense(x, wn)] if s is not None}
    other_m1_senses = {}     # sense -> the M1 words (as M1 spells them) that have it
    for x in m1_words:
        for f in {x, lemma(x)}:
            for s in wn.synsets(f, pos="n"):
                if s not in excluded_senses:
                    other_m1_senses.setdefault(s, set()).add(x)

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for w in sorted(cues):
        if not (w.isalpha() and len(w) >= 3) or w not in zipf:
            continue
        if w in excluded_forms or lemma(w) in excluded_forms:
            drop("naming-test or M1 word")
            continue
        s = concept_sense(w, wn)
        if s is None:
            drop("no clear main noun sense")
            continue
        if s in excluded_senses:
            drop("same sense as a naming-test or M1 word")
            continue
        if not mostly_noun(w, wn):
            drop("mostly not a noun")
            continue
        if any(person in p for p in s.hypernym_paths()) or ki.category_of(s, wn) is None:
            drop("not a concrete non-human thing")
            continue
        found = labels_above(s, table)
        if w in LABEL_WORDS or not found:
            drop("category name itself, or no everyday category")
            continue
        if s in other_m1_senses:
            decisions = {review.get((x, s.name())) for x in other_m1_senses[s]}
            if None in decisions:
                pending.append(f"{w} ({s.name()}) shares a sense with M1 word(s) {sorted(other_m1_senses[s])}")
                drop("awaiting review of an M1 word's other sense")
                continue
            if "exclude" in decisions:
                drop("same sense as an M1 word (reviewed)")
                continue
        synonyms = [l.lower() for l in s.lemma_names() if l.lower() != w]
        rows.append(dict(target=w, wordnet_gloss=s.definition(), zipf=round(float(zipf[w]), 4),
                         domain=ki.category_of(s, wn), synset=s.name(), category=found[0][2],
                         category_synset=found[0][3], synonyms="; ".join(synonyms)))
    t = pd.DataFrame(rows)
    shared = t.sort_values(["zipf", "target"], ascending=[False, True]).duplicated("synset")
    if log is not None:
        log.update(concepts=int((~shared).sum()), dropped={**dropped, "a more frequent cue has the same sense": int(shared.sum())},
                   pending_review=pending)
    return t[~shared.reindex(t.index)].reset_index(drop=True)


def category_members(wn, zipf, table):
    """{category synset name: [(word, zipf)]}: concrete nouns (first lemma of their main sense, mostly nouns) in
    each everyday category, the pool for same-category coordinates."""
    person = wn.synset("person.n.01")
    out = {}
    for s in wn.all_synsets("n"):
        w = s.lemma_names()[0]
        if not (w.isalpha() and w.islower() and len(w) >= 3) or w not in zipf:
            continue
        key = category_key(s, table)
        if key is None:
            continue
        main = main_sense(w, wn)
        if main is None or main != s or not mostly_noun(w, wn) or any(person in p for p in s.hypernym_paths()):
            continue
        out.setdefault(key, []).append((w, float(zipf[w])))
    return out


def coordinate_menu(row, wn, zipf, strength, table, members):
    """Same-category words of similar frequency, weakly associated with the target: WordNet siblings (level 1),
    cousins sharing a hypernym below the category (level 2), then other members of the category (level 3; not for
    the BROAD categories, where "both are devices" does not make a party horn and an elevator coordinates)."""
    s, key = wn.synset(row.synset), wn.synset(row.category_synset)
    labels = {syn for _, syn, _ in table}
    up_s = hypernym_dist(s)
    avoid = [row.target] + [x for x in str(row.synonyms).split("; ") if x]
    out = {}

    def consider(w, level, c=None):
        if w in out or not (w.isalpha() and w.islower() and len(w) >= 3) or w not in zipf:
            return
        gap = abs(zipf[w] - row.zipf)
        if gap > MENU_COORD_GAP or overlaps(w, avoid) or rhymes(w, row.target):
            return
        main = main_sense(w, wn)
        if main is None or (c is not None and main != c) or main == s or not mostly_noun(w, wn):
            return
        if category_key(main, table) != row.category_synset or main in up_s or s in hypernym_dist(main):
            return
        if strength.get((row.target, w), 0) >= COORD_WEAK or strength.get((w, row.target), 0) >= COORD_WEAK:
            return
        out[w] = dict(word=w, level=level, gap=round(gap, 2))

    for c in (c for h in s.hypernyms() for c in h.hyponyms()):
        consider(c.lemma_names()[0], 1, c)
    for c in (c for h in s.hypernyms() for g in h.hypernyms() for p in g.hyponyms() for c in p.hyponyms()):
        up_c = hypernym_dist(c)
        shared = [h for h in set(up_s) & set(up_c) if up_s[h] <= 2 and up_c[h] <= 2]
        if any(h not in labels and key in hypernym_dist(h) for h in shared):
            consider(c.lemma_names()[0], 2, c)
    if row.category not in BROAD:
        for w, _ in members.get(row.category_synset, []):
            consider(w, 3)
    return sorted(out.values(), key=lambda d: (d["level"], d["gap"], d["word"]))[:MENU_SIZE]


def associate_menu(row, swow, wn, zipf, table):
    """SWOW first responses to the target (strength >= MIN_ASSOCIATION), mostly nouns, outside its category and
    not close WordNet relatives, not one of its own category names, not sound-alikes. Other category names are
    allowed ("tree" for woodpecker)."""
    s = wn.synset(row.synset)
    avoid = [row.target] + [x for x in str(row.synonyms).split("; ") if x]
    own = {w for _, _, w, _, parent in labels_above(s, table) for w in (w, parent) if w}
    out = []
    resp = swow[swow.cue == row.target].sort_values(["strength", "response"], ascending=[False, True])
    for a, st in resp[["response", "strength"]].itertuples(index=False):
        if st < MIN_ASSOCIATION or len(out) >= ASSOCIATE_MENU_SIZE:
            break
        if (len(a) < 3 or a not in zipf or a in own or overlaps(a, avoid) or rhymes(a, row.target)
                or not mostly_noun(a, wn)):
            continue
        if any(category_key(n, table) == row.category_synset or close_relatives(n, s) for n in wn.synsets(a, pos="n")):
            continue
        out.append(dict(word=a, strength=round(float(st), 3)))
    return out


def superordinate_menu(row, wn, zipf, table):
    """The everyday category name and its broader family name, then single-word WordNet hypernyms (up to six
    steps, Zipf >= 3)."""
    s = wn.synset(row.synset)
    found = labels_above(s, table)
    words = [found[0][2], found[0][4]] if found else []
    up = hypernym_dist(s)
    words += [l.lower() for h, d in sorted(up.items(), key=lambda x: (x[1], x[0].name())) if d <= 6
              for l in h.lemma_names() if l.isalpha()]
    out = []
    for w in words:
        if w and w not in out and w != row.target and zipf.get(w, 0) >= 3.0 and not overlaps(w, [row.target]):
            out.append(w)
    return out[:MENU_SIZE]


def automatic(menu, label_names):
    """The default choice: the nearest coordinate within MAX_COORD_GAP, the strongest associate, the everyday
    category name (or its family name)."""
    coord = next((c["word"] for c in menu["coordinate"] if c["gap"] <= MAX_COORD_GAP), None)
    assoc = menu["associate"][0]["word"] if menu["associate"] else None
    sup = next((w for w in menu["superordinate"] if w in label_names), None)
    return coord, sup, assoc


def fmt(entries, key):
    return "; ".join(f"{e['word']} ({e[key]})" for e in entries)


def menus(wn, zipf, swow, sources, review=None, log=None, seed=SEED):
    """One row per concept in the fixed review order (Zipf bin, then a seeded shuffle within the bin);
    sources is excluded_sources()."""
    table = label_table(wn)
    strength = {(c, r): st for c, r, st in swow.itertuples(index=False)}
    members = category_members(wn, zipf, table)
    cons = concepts(wn, zipf, set(swow.cue), *sources, review=review, log=log)
    cons["zipf_bin"] = pd.cut(cons.zipf, ZIPF_BINS, right=False, labels=False)
    rng = np.random.default_rng(seed)
    cons = pd.concat([g.iloc[rng.permutation(len(g))] for _, g in cons.sort_values("target").groupby("zipf_bin")])
    rows = []
    for order, r in enumerate(cons.itertuples()):
        menu = dict(coordinate=coordinate_menu(r, wn, zipf, strength, table, members),
                    associate=associate_menu(r, swow, wn, zipf, table),
                    superordinate=superordinate_menu(r, wn, zipf, table))
        coord, sup, assoc = automatic(menu, LABEL_WORDS)
        rows.append(dict(order=order, target=r.target, zipf=r.zipf, zipf_bin=r.zipf_bin, domain=r.domain,
                         category=r.category, wordnet_gloss=r.wordnet_gloss, synset=r.synset, synonyms=r.synonyms,
                         auto_coordinate=coord, auto_superordinate=sup, auto_associate=assoc,
                         coordinate_menu=fmt(menu["coordinate"], "gap"), associate_menu=fmt(menu["associate"], "strength"),
                         superordinate_menu="; ".join(menu["superordinate"])))
    return pd.DataFrame(rows)


# --- curation, unrelated option, assignment ------------------------------------------------------------------

# avoid_unrelated: words the automatic unrelated option must not be for this concept ("; "-separated), when a
# reviewer finds a hidden link the association norms miss ("calf" is also the lower leg that tights cover)
CURATION_COLUMNS = ("target", "decision", "coordinate", "superordinate", "associate", "definition", "note",
                    "avoid_unrelated")
DEFINITION_WORDS = (8, 20)


def menu_words(text):
    return [part.split(" (")[0].strip() for part in str(text).split(";") if part.strip()] if isinstance(text, str) else []


def definition_problems(definition, words):
    """Why a drafted definition is unusable: length outside DEFINITION_WORDS, or it contains one of `words` (the
    target, its synonyms, the options) by the naming test's leak rule."""
    if not isinstance(definition, str) or not definition.strip():
        return ["no definition"]
    out = []
    n = len(definition.split())
    if not DEFINITION_WORDS[0] <= n <= DEFINITION_WORDS[1]:
        out.append(f"definition has {n} words")
    out += [f"definition contains {w!r}" for w in words if w and w.isalpha() and ki.leaks(definition, w)]
    return out


def check_curation(cur, menu):
    """Problems with the curation: unknown targets, bad decisions, options not taken from the concept's menu,
    repeated options, unusable definitions. An empty list means it can be applied."""
    problems = []
    rows = menu.set_index("target")
    if list(cur.columns[:len(CURATION_COLUMNS)]) != list(CURATION_COLUMNS):
        return [f"curation columns must start with {CURATION_COLUMNS}"]
    for r in cur.itertuples():
        if r.target not in rows.index:
            problems.append(f"{r.target}: not in the menus")
            continue
        if r.decision not in ("keep", "drop"):
            problems.append(f"{r.target}: decision must be keep or drop")
        elif r.decision == "drop":
            if not isinstance(r.note, str) or not r.note.strip():
                problems.append(f"{r.target}: a dropped concept needs a reason in the note")
        else:
            m = rows.loc[r.target]
            for col, menu_col in (("coordinate", "coordinate_menu"), ("superordinate", "superordinate_menu"),
                                  ("associate", "associate_menu")):
                if getattr(r, col) not in menu_words(m[menu_col]):
                    problems.append(f"{r.target}: {col} {getattr(r, col)!r} is not in its menu")
            if len({r.target, r.coordinate, r.superordinate, r.associate}) < 4:
                problems.append(f"{r.target}: options repeat")
            words = [r.target, r.coordinate, r.superordinate, r.associate] + menu_words(m["synonyms"])
            problems += [f"{r.target}: {p}" for p in definition_problems(r.definition, words)]
    if cur.target.duplicated().any():
        problems.append(f"targets curated twice: {sorted(cur.target[cur.target.duplicated()])}")
    return problems


def unrelated(target, gloss, domain, avoid, nouns, strength, gaps=(MAX_UNRELATED_GAP, FALLBACK_UNRELATED_GAP)):
    """A concrete noun from another domain, Zipf within MAX_UNRELATED_GAP (FALLBACK_UNRELATED_GAP when no word
    qualifies, which happens among the most frequent targets) and length within MAX_LEN_GAP, with no SWOW
    association either way; drawn with a seed derived from the target, so it does not depend on order."""
    z = nouns.loc[target, "zipf"]
    length = np.abs(nouns.index.str.len().to_numpy() - len(target))
    for gap in gaps:
        pool = nouns[(nouns.domain != domain).to_numpy() & ((nouns.zipf - z).abs() <= gap).to_numpy()
                     & (length <= MAX_LEN_GAP)]
        ok = [w for w in pool.index if not overlaps(w, [target, *avoid]) and not ki.leaks(gloss, w)
              and strength.get((target, w), 0) < WEAK and strength.get((w, target), 0) < WEAK]
        if ok:
            rng = np.random.default_rng(int(hashlib.sha256(target.encode()).hexdigest()[:8], 16))
            return sorted(ok)[rng.integers(len(ok))]
    return None


def quotas(bin_sizes, n):
    """n split over bins in proportion to their sizes (largest remainders)."""
    raw = bin_sizes / bin_sizes.sum() * n
    q = np.floor(raw).astype(int)
    q[(raw - q).sort_values(ascending=False).index[: n - q.sum()]] += 1
    return q


def assign(kept, n_test=N_TEST, n_cal_a=N_CAL_A, n_b=N_CAL_B_PER_DIRECTION):
    """Roles in review order (experiment3-protocol.md §4.1; materials-m2-m3.md §5.5; agent2 2026-10-07):

    1. calibration_b, the part that fits the degradation model for both question types: the first n_b concepts
       usable for free naming whose target is more common than its associate, and the first n_b whose target is
       rarer (the degradation model's two parameters are told apart by this direction);
    2. per Zipf bin, in proportion to the concepts left: test, then calibration_a (sets the lesion strength), then
       numbered reserves.

    kept needs order, zipf_bin, free_naming_ok and target_more_common. Returns (items, shortfall)."""
    k = kept.sort_values("order").copy()
    k["role"] = None
    short = {}
    usable = k.free_naming_ok.astype(bool)
    for name, mask in (("target more common", usable & k.target_more_common), ("target rarer", usable & ~k.target_more_common)):
        chosen = k.index[mask & k.role.isna()][:n_b]
        k.loc[chosen, "role"] = "calibration_b"
        if len(chosen) < n_b:
            short[f"calibration_b, {name}"] = int(n_b - len(chosen))
    rest = k[k.role.isna()]
    sizes = rest.groupby("zipf_bin").size()
    q_test, q_a = quotas(sizes, n_test), quotas(sizes, n_cal_a)
    for b, g in rest.groupby("zipf_bin"):
        nt, na, n = q_test.get(b, 0), q_a.get(b, 0), len(g)
        k.loc[g.index, "role"] = (["test"] * min(nt, n) + ["calibration_a"] * min(na, max(n - nt, 0))
                                  + ["reserve"] * max(n - nt - na, 0))
        if n < nt + na:
            short[f"bin {int(b)}"] = int(nt + na - n)
    k["rank"] = k.groupby(["role", "zipf_bin"]).cumcount()
    items = k.reset_index(drop=True)
    items.insert(0, "item_id", [f"m{i:03d}" for i in range(len(items))])
    return items, short


def token_counts(words, tokenizer, chat=True):
    """Tokens per word as the answer is scored: chat models start the assistant turn with the word itself; base
    models score it after "Answer:" with a leading space (experiment3-protocol.md §7.2)."""
    return [len(tokenizer.encode(w if chat else " " + w, add_special_tokens=False)) for w in words]


MAX_TOKEN_GAP = 1   # user decision 2026-10-07: target and associate may differ by one token (was: equal)


def free_naming_ok(items, tokenizer, chat=True):
    """H3b needs target and associate within MAX_TOKEN_GAP tokens of each other and with different first two
    letters (§5.2). The H3b statistic subtracts conditions within a concept, so a fixed length difference cancels;
    per-token scores are reported as a robustness check (experiment3-protocol.md §7.2)."""
    gap = np.abs(np.array(token_counts(items.target, tokenizer, chat)) - np.array(token_counts(items.associate, tokenizer, chat)))
    return (gap <= MAX_TOKEN_GAP) & (items.target.str[:2] != items.associate.str[:2]).to_numpy()


def build_menus(out=OUT):
    log = {}
    menu = menus(ki.wordnet(), ki.load_zipf(), load_swow(), excluded_sources(), load_sense_review(), log)
    if log["pending_review"]:
        raise ValueError("record a decision in m1_sense_review.csv for:\n" + "\n".join(log["pending_review"]))
    log.update(menus=len(menu), by_bin=menu.groupby("zipf_bin").size().to_dict(),
               with_automatic_choice=int(menu[["auto_coordinate", "auto_superordinate", "auto_associate"]].notna().all(axis=1).sum()))
    out.mkdir(parents=True, exist_ok=True)
    menu.to_csv(out / "menus.csv", index=False, lineterminator="\n")
    (out / "menus_log.json").write_text(json.dumps(log, indent=2, default=int), encoding="utf-8")
    return menu, log


def build_items(out=OUT, tokenizer=None, chat=True):
    menu = pd.read_csv(out / "menus.csv", keep_default_na=False, na_values=[""])
    cur = pd.read_csv(out / "curation.csv", keep_default_na=False, na_values=[""])
    problems = check_curation(cur, menu)
    if problems:
        raise ValueError("curation problems:\n" + "\n".join(problems))
    wn, zipf, swow = ki.wordnet(), ki.load_zipf(), load_swow()
    strength = {(c, r): st for c, r, st in swow.itertuples(index=False)}
    # unrelated options come from familiar concrete nouns: SWOW cues (agent2 2026-10-07: "pinche" was a WordNet
    # monkey few readers know)
    # and never a concept dropped in review (misspelled cues such as "calender", odd senses)
    kept_targets = set(cur.target[cur.decision == "keep"])
    nouns = ki.eligible(wn, zipf).drop_duplicates("target").set_index("target")[["zipf", "category"]].rename(columns={"category": "domain"})
    nouns = pd.concat([nouns, menu[menu.target.isin(kept_targets)].set_index("target")[["zipf", "domain"]]])
    nouns = nouns.loc[~nouns.index.duplicated()]
    nouns = nouns[nouns.index.isin(set(swow.cue)) & ~nouns.index.isin(set(cur.target[cur.decision == "drop"]))
                  & ~nouns.index.isin(LABEL_WORDS | UNRELATED_STOPLIST)]
    kept = menu.merge(cur[cur.decision == "keep"][list(CURATION_COLUMNS[:6]) + ["avoid_unrelated"]], on="target").drop(columns="decision")
    kept["unrelated"] = [unrelated(r.target, r.definition, r.domain,
                                   [r.coordinate, r.superordinate, r.associate, *menu_words(str(r.synonyms)),
                                    *menu_words(r.avoid_unrelated)], nouns, strength)
                         for r in kept.itertuples()]
    no_unrelated = kept.target[kept.unrelated.isna()].tolist()
    kept = kept.dropna(subset=["unrelated"]).copy()
    kept["associate_zipf"] = kept.associate.map(zipf)
    kept["target_more_common"] = kept.zipf > kept.associate_zipf
    if tokenizer is None:
        raise ValueError("the item assignment needs the model's tokenizer (free naming eligibility)")
    kept["free_naming_ok"] = free_naming_ok(kept, tokenizer, chat)
    items, short = assign(kept)
    cols = ["item_id", "role", "rank", "zipf_bin", "target", "definition", "coordinate", "superordinate", "associate",
            "unrelated", "zipf", "associate_zipf", "target_more_common", "free_naming_ok", "domain", "category",
            "synset", "wordnet_gloss", "order"]
    items = items[cols]
    log = dict(curated=len(cur), kept=int((cur.decision == "keep").sum()), dropped=int((cur.decision == "drop").sum()),
               no_unrelated=no_unrelated, by_role=items.role.value_counts().to_dict(), shortfall=short,
               free_naming_ok_by_role=items.groupby("role").free_naming_ok.sum().astype(int).to_dict(),
               target_more_common_and_free_naming=int((items.free_naming_ok & items.target_more_common).sum()))
    items.to_csv(out / "items.csv", index=False, lineterminator="\n")
    (out / "items_log.json").write_text(json.dumps(log, indent=2, default=int), encoding="utf-8")
    return items, log


N_FREE_TEST = 100   # free-naming test concepts (materials-m2-m3.md §5.2: 100 concepts x 3 conditions)
REPLACE_ORDER = ("test", "calibration_a", "calibration_b")
MAIN_MODEL, MAIN_DTYPE = "qwen3-30b-a3b-instruct", "float16"   # the check runs on the intact main model (§5.8)


def finalize(items, check, n_free=N_FREE_TEST, seed=SEED):
    """Final roles after the intact-model validity check (materials-m2-m3.md §5.8), from items.csv and the check
    table of lesion.m3_scoring.validity (choice_ok and naming_ok per item_id). Rules written before the check:

    - test and calibration_a concepts need choice_ok (rule 1); one that fails is replaced by the first reserve, in
      review order, of the same Zipf bin that passes;
    - calibration_b concepts are used in both formats and need choice_ok and naming_ok (rules 1 and 2); one that
      fails is replaced by the first reserve of the same frequency direction that is usable for free naming and
      passes both;
    - failures are filled set by set in REPLACE_ORDER (the confirmatory set first when reserves run short), each in
      review order (user decision 2026-10-07); a slot no reserve can fill stays empty and is reported, never filled
      from another set;
    - free naming (H3b): all calibration_b concepts, and n_free test concepts drawn with default_rng(seed) from the
      final test concepts usable for free naming that pass rule 2. A test concept that fails only rule 2 stays in
      the six-choice test and is just not drawn for free naming (user decision 2026-10-07).

    The check must cover every item exactly once. Returns (final items with planned_role, role, replaces and
    free_naming columns; log)."""
    ids, checked = set(items.item_id), check.item_id
    if checked.duplicated().any() or set(checked) != ids:
        raise ValueError(f"the check does not cover each item once: {len(ids - set(checked))} missing, "
                         f"{len(set(checked) - ids)} unknown, {int(checked.duplicated().sum())} repeated")
    c = check.set_index("item_id")
    t = items.sort_values("order").copy()
    t["planned_role"] = t.role
    t["replaces"] = ""
    choice_ok = t.item_id.map(c.choice_ok).fillna(False).astype(bool)
    naming_ok = t.item_id.map(c.naming_ok).fillna(False).astype(bool) & t.free_naming_ok.astype(bool)
    passes = {"test": choice_ok, "calibration_a": choice_ok, "calibration_b": choice_ok & naming_ok}
    available = t.role == "reserve"
    replaced, unfilled = [], []
    for role in REPLACE_ORDER:
        for i in t.index[(t.planned_role == role) & ~passes[role]]:
            key = "target_more_common" if role == "calibration_b" else "zipf_bin"
            cand = t.index[available & passes[role] & (t[key] == t.at[i, key])]
            t.at[i, "role"] = "excluded"
            entry = dict(role=role, failed=t.at[i, "item_id"], target=t.at[i, "target"], **{key: t.at[i, key]})
            if len(cand):
                j = cand[0]
                t.at[j, "role"], t.at[j, "replaces"] = role, t.at[i, "item_id"]
                available[j] = False
                replaced.append(dict(entry, replacement=t.at[j, "item_id"], replacement_target=t.at[j, "target"]))
            else:
                unfilled.append(entry)
    # what is left to replace with: unused reserves that pass the rules of each set (agent2 2026-10-07)
    reserves_left = dict(
        unused=int(available.sum()),
        passing_rule_1_by_bin={int(b): int((available & choice_ok & (t.zipf_bin == b)).sum()) for b in sorted(t.zipf_bin.unique())},
        passing_rules_1_and_2_for_calibration_b={
            "target more common": int((available & passes["calibration_b"] & t.target_more_common).sum()),
            "target rarer": int((available & passes["calibration_b"] & ~t.target_more_common).sum())})
    pool = t[(t.role == "test") & naming_ok]
    drawn = pool.item_id.to_numpy()[np.random.default_rng(seed).permutation(len(pool))[:n_free]]
    t["free_naming"] = (t.role == "calibration_b") | t.item_id.isin(drawn)
    t["rank"] = t.groupby(["role", "zipf_bin"]).cumcount()
    t = t.sort_values("item_id").reset_index(drop=True)
    planned, final = t[t.planned_role == "calibration_b"], t[t.role == "calibration_b"]
    short = {k: int(v) for k, v in {
        **{r: (t.planned_role == r).sum() - (t.role == r).sum() for r in ("test", "calibration_a")},
        "calibration_b, target more common": planned.target_more_common.sum() - final.target_more_common.sum(),
        "calibration_b, target rarer": (~planned.target_more_common).sum() - (~final.target_more_common).sum(),
        "free naming, test": n_free - len(drawn)}.items() if v > 0}
    log = dict(by_role=t.role.value_counts().to_dict(),
               by_role_and_bin={r: g.zipf_bin.value_counts().sort_index().to_dict() for r, g in t.groupby("role")},
               planned_by_role_and_bin={r: g.zipf_bin.value_counts().sort_index().to_dict() for r, g in t.groupby("planned_role")},
               free_naming_by_role=t[t.free_naming].role.value_counts().to_dict(),
               free_naming_test_pool=len(pool), replaced=replaced, unfilled=unfilled, shortfall=short,
               reserves_left=reserves_left)
    return t, log


def check_run_problems(meta, items_sha256):
    """Why a check run cannot set the final items: only one complete run of the intact main model at its pinned
    revision and dtype, on all items of this items.csv from a clean tree, counts (agent2 2026-10-07)."""
    from . import models

    want = dict(model=MAIN_MODEL, revision=models.MODELS[MAIN_MODEL][1], dtype=MAIN_DTYPE, tiny=False, limit=None,
                complete=True, clean_tree=True, items_sha256=items_sha256)
    return [f"{k} is {meta.get(k)!r}, needs {v!r}" for k, v in want.items() if meta.get(k) != v]


def write_final(check_dir, out=OUT):
    """Apply finalize to the check run in check_dir (validity.csv, meta.json) and store the check alongside."""
    check_dir = Path(check_dir)
    items = pd.read_csv(out / "items.csv", keep_default_na=False, na_values=[""])
    check = pd.read_csv(check_dir / "validity.csv")
    meta = json.loads((check_dir / "meta.json").read_text(encoding="utf-8"))
    problems = check_run_problems(meta, hashlib.sha256((out / "items.csv").read_bytes()).hexdigest())
    if problems:
        raise ValueError(f"{check_dir} cannot set the final items:\n" + "\n".join(problems))
    final, log = finalize(items, check)
    log["check_meta"] = meta
    check.to_csv(out / "validity.csv", index=False, lineterminator="\n")
    final.to_csv(out / "final_items.csv", index=False, lineterminator="\n")
    (out / "final_log.json").write_text(json.dumps(log, indent=2, default=int, ensure_ascii=False), encoding="utf-8")
    return final, log


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "menus":
        print(json.dumps(build_menus()[1], indent=2, default=int))
    elif step == "items":
        from . import models

        print(json.dumps(build_items(tokenizer=models.load_tokenizer(MAIN_MODEL))[1], indent=2, default=int))
    elif step == "final" and len(sys.argv) > 2:
        print(json.dumps(write_final(sys.argv[2])[1], indent=2, default=int, ensure_ascii=False))
    else:
        sys.exit("usage: python -m lesion.m3_items menus|items|final <check dir>")
