"""New-concept comprehension materials and prompts for experiment A.

WordNet supplies a review pool, not automatically approved experimental items. A concept's naming,
synonym, close-definition and property questions always stay together. No model scores enter selection.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer

from . import runinfo
from .m3_battery import LabelItem
from .m3_scoring import NamingPrompt

LEXICAL = ("target", "synonym", "foil", "superordinate")
OLD_WORD_COLUMNS = ("probe", "target", "distractor", "word1", "word2", "coordinate", "superordinate", "associate",
                    "unrelated", "option_a", "option_b", "option1", "option2")
STEM = PorterStemmer()


def old_words(directory):
    words = set()
    for path in Path(directory).rglob("*.csv"):
        if "title_gap" in path.parts:
            continue
        table = pd.read_csv(path, keep_default_na=False)
        for col in set(table) & set(OLD_WORD_COLUMNS):
            words.update(str(w).lower().strip() for w in table[col] if re.fullmatch("[A-Za-z]+", str(w)))
    return words


def lexemes(row):
    words = [str(row[c]).lower() for c in LEXICAL if c in row and row[c]]
    words += str(row.get("family_words", "")).split("|")
    words += str(row.get("accepted_answers", "")).split("|")
    return {STEM.stem(w) for w in words if w}


def validate(items):
    required = ["item_id", "family", "split", "target", "synonym", "foil", "definition", "foil_definition", "review",
                "synset", "foil_synset", "accepted_answers", "frequency_bin", "semd_bin"]
    if set(required) - set(items) or items[required].isna().any().any() or items[required].eq("").any().any():
        raise ValueError("missing fine-semantics fields")
    if items.item_id.duplicated().any() or items.target.duplicated().any():
        raise ValueError("duplicate fine-semantics item or target")
    if set(items.split) - {"screening", "dev", "test"}:
        raise ValueError("unknown split")
    if set(items.frequency_bin) - {"low", "middle", "high"} or set(items.semd_bin) - {"low", "high"}:
        raise ValueError("unknown frequency or semantic-diversity bin")
    if (items.groupby("family").split.nunique() != 1).any():
        raise ValueError("a concept family crosses splits")
    owners = {}
    for row in items.to_dict("records"):
        if len({row[c].lower() for c in ("target", "synonym", "foil")}) != 3:
            raise ValueError(f"{row['item_id']}: target, synonym and foil must differ")
        for word in lexemes(row) - {STEM.stem(row.get("superordinate", ""))}:
            if word in owners and owners[word] != row["split"]:
                raise ValueError(f"{word}: lexical family crosses splits")
            owners[word] = row["split"]


def load_items(path, split, freeze_tag=None, rehearsal=False):
    if split == "test":
        runinfo.check_analysis_freeze(freeze_tag)
    table = pd.read_csv(path, keep_default_na=False)
    table = table[table.review != "excluded"].copy()
    validate(table)
    out = table[table.split == split].copy()
    if out.empty or (not rehearsal and not (out.review == "approved").all()):
        raise ValueError("empty split or materials not approved")
    return out


def naming_prompts(items, copy=False):
    prompts = []
    for row in items.to_dict("records"):
        text = (f"Repeat this word exactly. Reply with the word only.\nWord: {row['target']}\nAnswer:" if copy else
                "Name the thing or concept described. Reply with one word only.\n"
                f"Description: {row['definition']}\nAnswer:")
        candidates = {"target": row["target"]} if copy else {
            "target": row["target"], "synonym": row["synonym"], "coordinate": row["foil"]}
        prompts.append(NamingPrompt(row["item_id"], "copy" if copy else "naming", text, candidates))
    return prompts


def label_prompts(items):
    out = []
    for r in items.to_dict("records"):
        for task, question, right, wrong in (
            ("definition", f"Which description defines {r['target']}?", r["definition"], r["foil_definition"]),
            ("synonym", f"Which word can mean the same as {r['target']}?", r["synonym"], r["foil"]),
        ):
            for order in (0, 1):
                a, b = (right, wrong) if order == 0 else (wrong, right)
                text = f"{question} Reply with the letter only.\nA. {a}\nB. {b}\nAnswer:"
                out.append(LabelItem(r["item_id"], task, order, text, "AB"[order], ("A", "B")))
        # Property claims are independently written during review, not copied from the definition.
        if r.get("property_true") and r.get("property_false"):
            for truth, field in ((True, "property_true"), (False, "property_false")):
                for order in (0, 1):
                    a, b = ("true", "false") if order == 0 else ("false", "true")
                    text = (f"For {r['target']}, is this claim true? {r[field]}\n"
                            f"Reply with the letter only.\nA. {a}\nB. {b}\nAnswer:")
                    correct = "AB"[order if truth else 1 - order]
                    out.append(LabelItem(r["item_id"], "property", 2 * int(truth) + order, text, correct, ("A", "B")))
    return out


def word_answer_prompts(items):
    out = []
    for r in items.to_dict("records"):
        for order in (0, 1):
            a, b = (r["synonym"], r["foil"]) if order == 0 else (r["foil"], r["synonym"])
            text = (f"Which word can mean the same as {r['target']}?\n"
                    f"Choose between these two words: {a}, {b}.\n"
                    "Reply by repeating the chosen word only. Do not add a letter or explanation.\nChosen word:")
            out.append(NamingPrompt(r["item_id"], f"synonym_word_{order}", text,
                                    {"target": r["synonym"], "coordinate": r["foil"]}))
    return out


def wordnet_pool(wn, excluded, frequency, semd):
    """Close siblings under the same immediate hypernym, with a distinct single-word synonym.

    The output keeps synset IDs so every proposed synonym/foil can be checked against its exact sense.
    """
    excluded = {STEM.stem(w) for w in excluded}
    rows = []
    for syn in wn.all_synsets("n"):
        names = sorted(w.lower() for w in syn.lemma_names() if re.fullmatch("[a-z]+", w))
        names = [w for w in names if STEM.stem(w) not in excluded and w in frequency]
        if len(names) < 2 or not 5 <= len(syn.definition().split()) <= 35:
            continue
        primary = [w for w in names if wn.synsets(w, "n")[0] == syn]
        if not primary:
            continue
        target = min(primary, key=lambda w: (frequency[w], w))
        synonym = max((w for w in names if STEM.stem(w) != STEM.stem(target)),
                      key=lambda w: (frequency[w], w), default=None)
        if synonym is None:
            continue
        siblings = sorted({s for parent in syn.hypernyms() for s in parent.hyponyms() if s != syn}, key=lambda s: s.name())
        for foil in siblings:
            options = [w for w in foil.lemma_names() if re.fullmatch("[a-z]+", w) and w in frequency
                       and STEM.stem(w) not in excluded and STEM.stem(w) not in {STEM.stem(n) for n in names}]
            if not options or not 5 <= len(foil.definition().split()) <= 35:
                continue
            word = min(options, key=lambda w: (abs(frequency[w] - frequency[target]), w))
            if (set(wn.synsets(target)) | set(wn.synsets(synonym))) & set(wn.synsets(word)):
                continue
            family = {lemma.name().lower() for s in (syn, foil) for lemma in s.lemmas()}
            family |= {other.name().lower() for s in (syn, foil) for lemma in s.lemmas()
                       for other in lemma.derivationally_related_forms()}
            family = {w for w in family if re.fullmatch("[a-z]+", w)}
            if {STEM.stem(w) for w in family} & excluded:
                continue
            if re.search(rf"\b{re.escape(target)}\b", syn.definition(), re.I):
                continue
            rows.append(dict(target=target, synonym=synonym, foil=word, synset=syn.name(), foil_synset=foil.name(),
                             definition=syn.definition().replace(";", ","), foil_definition=foil.definition().replace(";", ","),
                             frequency=frequency[target], semd=semd[target], family=syn.name(), superordinate="",
                             family_words="|".join(sorted(family)),
                             property_true="", property_false="", review="draft"))
            break
    return pd.DataFrame(rows).drop_duplicates("target")


def select_pool(pool, n=360, seed=1910):
    """Stratify by lexical frequency, with disjoint stems for target, synonym and close foil.

    The selection is a draft. Review does not consult lesion outputs. Counts are exact when the pool permits.
    """
    rng = np.random.default_rng(seed)
    pool = pool.copy().sort_values(["frequency", "target"])
    pool["frequency_bin"] = pd.qcut(np.arange(len(pool)), 3, labels=("low", "middle", "high")).astype(str)
    if n % 6:
        raise ValueError("balanced frequency by SemD selection needs a multiple of six")
    pool["semd_bin"] = pool.groupby("frequency_bin").semd.transform(
        lambda s: pd.qcut(s.rank(method="first"), 2, labels=("low", "high")).astype(str))
    picked, used, families = [], set(), set()
    cells = [(f, s) for f in ("low", "middle", "high") for s in ("low", "high")]
    for group, diversity in cells:
        choices = pool[(pool.frequency_bin == group) & (pool.semd_bin == diversity)]
        for i in rng.permutation(choices.index):
            row = choices.loc[i].to_dict()
            words = lexemes(row)
            if words & used or row["family"] in families:
                continue
            picked.append(row)
            used.update(words)
            families.add(row["family"])
            if sum(r["frequency_bin"] == group and r["semd_bin"] == diversity for r in picked) == n // 6:
                break
    if len(picked) != n:
        raise ValueError(f"only {len(picked)} disjoint concepts available for requested {n}")
    out = pd.DataFrame(picked)
    out["split"] = ""
    # Six frequency x SemD cells, keeping 160/80/120 concepts and every family together.
    for (group, diversity), n_screen in zip(cells, (27, 27, 26, 27, 26, 27)):
        ix = out.index[(out.frequency_bin == group) & (out.semd_bin == diversity)].to_numpy()
        if n != 360:
            n_screen = round(len(ix) * 4 / 9)
        n_test = len(ix) // 3
        labels = ["screening"] * n_screen + ["dev"] * (len(ix) - n_screen - n_test) + ["test"] * n_test
        out.loc[rng.permutation(ix), "split"] = labels
    out.insert(0, "item_id", [f"f{i:03d}" for i in range(len(out))])
    validate(out)
    return out


def review_table(items, wn, excluded):
    """Source-based review flags for every draft. Flags never approve an item or use model performance."""
    rows, family_splits = [], {}
    for r in items.to_dict("records"):
        syn, foil = wn.synset(r["synset"]), wn.synset(r["foil_synset"])
        family = {STEM.stem(w.lower()) for s in (syn, foil) for w in s.lemma_names() if re.fullmatch("[a-z]+", w.lower())}
        for word in family:
            family_splits.setdefault(word, set()).add(r["split"])
        flags = []
        if syn != wn.synsets(r["target"], "n")[0]:
            flags.append("target_nonprimary_sense")
        if set(wn.synsets(r["target"])) & set(wn.synsets(r["foil"])):
            flags.append("target_and_foil_share_another_sense")
        if set(wn.synsets(r["synonym"])) & set(wn.synsets(r["foil"])):
            flags.append("answer_and_foil_share_another_sense")
        if family & {STEM.stem(w) for w in excluded}:
            flags.append("additional_synset_lemma_overlaps_old_materials")
        rows.append(dict(item_id=r["item_id"], target=r["target"], synonym=r["synonym"], foil=r["foil"],
                         split=r["split"], synset=r["synset"], source_synonym_valid=r["synonym"] in syn.lemma_names(),
                         shared_hypernym=bool(set(syn.hypernyms()) & set(foil.hypernyms())),
                         flags=flags, family_lexemes=sorted(family),
                         semantic_review="pending independent review of answer uniqueness and definition contrast",
                         property_status="authored" if r.get("property_true") and r.get("property_false") else "not_authored"))
    for row in rows:
        if any(len(family_splits[w]) > 1 for w in row["family_lexemes"]):
            row["flags"].append("additional_synset_lemma_crosses_splits")
        row["flags"] = "|".join(row["flags"])
        row["family_lexemes"] = "|".join(row["family_lexemes"])
    return pd.DataFrame(rows)
