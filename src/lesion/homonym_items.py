"""Meaning-specific related-word production, using explicitly identified human-norm meanings.

Dominance is never inferred from a language model. Context, cue and dominance remain separate factors.
Candidate likelihoods and genuine generated answers are separate outcomes.
"""
import re

import numpy as np
import pandas as pd

from . import runinfo
from .m3_battery import LabelItem
from .m3_scoring import NamingPrompt

CONTEXTS = ("low", "high")
CUES = ("none", "correct", "misleading")


def validate(items):
    required = ["item_id", "family", "split", "word", "sense", "dominance", "target", "competitor",
                "context_low", "context_high", "norm_source", "review", "other_sense", "meaning_probability",
                "norm_n_valid", "target_n_responses", "target_within_meaning_fraction", "target_frequency_rank",
                "context_low_words", "context_high_words"]
    if set(required) - set(items) or items[required].isna().any().any() or items[required].eq("").any().any():
        raise ValueError("missing homonym fields")
    if set(items.split) - {"dev", "test"}:
        raise ValueError("unknown homonym split")
    for col in ("meaning_probability", "target_within_meaning_fraction"):
        if not items[col].between(0, 1, inclusive="right").all():
            raise ValueError("invalid human-norm probability")
    for col in ("sense", "other_sense", "norm_n_valid", "target_n_responses", "target_frequency_rank"):
        if not ((items[col] >= 1) & (items[col] % 1 == 0)).all():
            raise ValueError("invalid human-norm count or sense")
    if (items.target_n_responses > items.norm_n_valid).any() or (items.sense == items.other_sense).any():
        raise ValueError("inconsistent human-norm fields")
    if items.item_id.duplicated().any():
        raise ValueError("duplicate homonym item")
    if not set(items.dominance) <= {"dominant", "subordinate"}:
        raise ValueError("unknown dominance")
    if (items.groupby("family").split.nunique() != 1).any():
        raise ValueError("homonym senses cross splits")
    for family, group in items.groupby("family"):
        if len(group) != 2 or set(group.dominance) != {"dominant", "subordinate"}:
            raise ValueError(f"{family}: need exactly two different normed senses")
    for r in items.itertuples():
        if len(r.context_low.split()) != len(r.context_high.split()) or any(
                len(getattr(r, 'context_' + c).split()) != getattr(r, 'context_' + c + '_words')
                for c in CONTEXTS):
            raise ValueError(f"{r.item_id}: context word counts differ")
        if r.target[:2] == r.competitor[:2] or min(len(r.target), len(r.competitor)) <= 2:
            raise ValueError(f"{r.item_id}: indistinguishable or complete-word cues")
        for context in (r.context_low, r.context_high):
            for word in (r.target, r.competitor):
                if re.search(rf"\b{re.escape(word)}\b", context, re.I):
                    raise ValueError(f"{r.item_id}: context contains a candidate answer")


def load_items(path, split, freeze_tag=None, rehearsal=False):
    if split == "test":
        runinfo.check_analysis_freeze(freeze_tag)
    table = pd.read_csv(path, keep_default_na=False)
    validate(table)
    out = table[table.split == split].copy()
    if out.empty or (not rehearsal and not (out.review == "approved").all()):
        raise ValueError("empty split or materials not approved")
    return out


def prompts(items):
    naming, labels = [], []
    for r in items.to_dict("records"):
        for context in CONTEXTS:
            for cue in CUES:
                condition = f"{context}_{cue}"
                hint = ""
                if cue != "none":
                    word = r["target"] if cue == "correct" else r["competitor"]
                    hint = f'\nAn additional hint suggests a word starting with "{word[:2]}". The hint may be wrong.'
                body = f"Word: {r['word']}\nIntended meaning: {r['context_' + context]}{hint}"
                text = "Give one word closely associated with the intended meaning. Reply with that word only.\n" + body + "\nAnswer:"
                naming.append(NamingPrompt(r["item_id"], condition, text,
                                           {"target": r["target"], "other_meaning": r["competitor"]}))
                for order in (0, 1):
                    a, b = ((r["target"], r["competitor"]) if order == 0 else (r["competitor"], r["target"]))
                    text = ("Choose the word associated with the intended meaning. Reply with the letter only.\n"
                            + body + f"\nA. {a}\nB. {b}\nAnswer:")
                    labels.append(LabelItem(r["item_id"], condition, order, text, "AB"[order], ("A", "B")))
    return naming, labels


def normalize_answer(answer):
    return str(answer).strip().lower().strip('.!,?"\'')


def classify(answer, row, lexicon):
    """Exact complete single-word responses only, with a pre-existing human association lexicon.

    Unknown answers need blinded review. They are not all semantic errors. Extra prose is a format failure.
    A cue cannot change which meaning an answer belongs to.
    """
    word = normalize_answer(answer)
    if not word:
        return "empty"
    if not re.fullmatch("[a-z]+", word):
        return "format_invalid"
    if word == row["target"]:
        return "target"
    if word == row["word"]:
        return "ambiguous_response"
    senses = set(lexicon.get((row["word"], word), ()))
    if len(senses) > 1:
        return "ambiguous_response"
    if senses == {int(row["sense"])}:
        return "same_meaning"
    if senses == {int(row["other_sense"])}:
        return "other_meaning"
    return "unclassified"


def response_lexicon(responses):
    table = responses.dropna(subset=["EditedResponse", "MeaningCode"])
    table = table[table.MeaningCode > 0].copy()
    table["EditedResponse"] = table.EditedResponse.astype(str).str.lower().str.strip()
    return {(str(word).lower(), answer): tuple(sorted(int(x) for x in g.MeaningCode.unique()))
            for (word, answer), g in table.groupby(["AmbiguousWord", "EditedResponse"])}


def select_normed(norms, responses, contexts, n=80, seed=1910):
    """Select homographs with >=100 responses and a measurable two-meaning bias.

    UK auditory norms are the source of the rank, not a claim of written-US equivalence. Heterographic
    homophones are excluded. The draft keeps all source meaning IDs and source counts in a private audit.
    Contexts are authored independently of model outputs. The answer words are frequent, meaning-specific
    human associates absent from both contexts. Their initial two letters must differ.
    """
    lexicon = response_lexicon(responses)
    rows, audit = [], []
    for r in norms.to_dict("records"):
        word = r["AmbiguousWord"]
        if word not in contexts or r["nonHomograph"] != 0 or r["nValidResp"] < 100:
            continue
        senses = sorted([(float(r[f"pM{i}"]), i) for i in range(1, 9) if pd.notna(r[f"pM{i}"])], reverse=True)[:2]
        if len(senses) < 2 or senses[0][0] - senses[1][0] < 0.10 or senses[1][0] < 0.03:
            continue
        if {s for _, s in senses} != set(contexts[word]):
            raise ValueError(f"{word}: contexts do not match the two normed meanings")
        answers, sense_counts = [], {}
        all_context = " ".join(c for pair in contexts[word].values() for c in pair)
        for _, sense in senses:
            counts = responses[(responses.AmbiguousWord == word) & (responses.MeaningCode == sense)].EditedResponse.dropna().str.lower().value_counts()
            sense_counts[sense] = counts
            eligible = [w for w in counts.index if re.fullmatch("[a-z]{3,}", w) and w != word
                        and not re.search(rf"\b{re.escape(w)}\b", all_context, re.I)
                        and lexicon.get((word, w)) == (sense,)]
            answers.append(eligible)
        pair = next(((a, b) for a in answers[0] for b in answers[1] if a[:2] != b[:2]), None)
        if pair is None:
            audit.append(dict(word=word, status="no_distinct_answer_pair"))
            continue
        for j, (p, sense) in enumerate(senses):
            low, high = contexts[word][sense]
            counts = sense_counts[sense]
            count = int(counts[pair[j]])
            rows.append(dict(item_id=f"h_{word}_{sense}", family=word, word=word, sense=sense,
                             other_sense=senses[1 - j][1], dominance=("dominant", "subordinate")[j],
                             target=pair[j], competitor=pair[1 - j], context_low=low, context_high=high,
                             meaning_probability=p, norm_n_valid=int(r['nValidResp']), target_n_responses=count,
                             target_within_meaning_fraction=count / int(counts.sum()),
                             target_frequency_rank=1 + int((counts > count).sum()),
                             context_low_words=len(low.split()), context_high_words=len(high.split()),
                             norm_source="Gilbert and Rodd 2022, explicit meaning code", review="draft"))
            audit.append(dict(word=word, sense=sense, probability=p, n_valid=r["nValidResp"], status="eligible"))
    table = pd.DataFrame(rows)
    families = sorted(table.family.unique()) if not table.empty else []
    if len(families) < n:
        raise ValueError(f"only {len(families)} suitable homonym families, requested {n}")
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(families)[:n]
    split = {word: "dev" if i < n // 2 else "test" for i, word in enumerate(chosen)}
    table = table[table.family.isin(chosen)].copy()
    table["split"] = table.family.map(split)
    validate(table)
    return table, pd.DataFrame(audit)
