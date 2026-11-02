"""Cross-format battery on the M3 calibration concepts (docs/plans/experiment3-protocol.md §5.5; user decision Q18,
2026-10-08): the same concept tested with and without the need to produce its name, to tell a word-retrieval
lesion from partial semantic loss and from an output-format effect.

Formats (prompt builders return lesion.m3_scoring.NamingPrompt for scored answers, ChoicePrompt-like LabelItem for
label answers):

| format               | input                                    | answer                    | what it needs              |
| -------------------- | ---------------------------------------- | ------------------------- | -------------------------- |
| name_free            | definition                               | the word (5 candidates)   | meaning -> word            |
| name_hinted_demos    | definition; demos carry correct hints    | target vs associate       | control for the demo shift |
| cue / miscue         | definition + first two letters           | target vs associate       | (existing H3b conditions)  |
| options_word         | definition + the six options             | the chosen word           | meaning -> word, word shown|
| options_letter       | definition + the six options             | A to F                    | meaning -> option          |
| word_to_definition   | the word + its definition and a foil     | A or B                    | word -> meaning            |
| copy                 | the word                                 | the same word             | word form only             |

The foil in word_to_definition is the definition of another calibration or reserve concept of the same category
(else the same domain, else any), the next one in review order, never one that names the target. Both orders are run.
Greedy generation (up to 6 tokens) on name_free shows what the model actually says.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import knowledge_items as ki, prefix_scoring, scoring
from .m3_scoring import (CANNOT_TELL, CHOICE_INSTRUCTION, DEMOS, ERROR_ROLES, LETTERS, ROLES, NamingPrompt,
                         _choice_block, _naming_block, base_order, build_naming_prompts)

OPTIONS_WORD_RULE = " Reply with the chosen word only, in lowercase."
LABEL_RULE = " Reply with the letter only."
W2D_INSTRUCTION = "Choose the description that defines the given word."
COPY_INSTRUCTION = "Repeat the given word exactly."
NAMING_RULE = " Reply with the word only, in lowercase."
POOL_ROLES = ("calibration_a", "calibration_b", "reserve")


@dataclass
class LabelItem:
    item_id: str
    fmt: str
    variant: int
    text: str
    correct: str              # the correct label
    labels: tuple             # the labels read
    foil_id: str | None = None
    foil_match: str | None = None   # category / domain / any, for interpretation


# --- prompts ---------------------------------------------------------------------------------------------------

def name_hinted_demos_prompts(items, chat=True):
    """No hint on the test item, but the demos carry their (correct) hints, as in the cue and miscue conditions, so
    that cue minus this baseline is not confounded with a change of demos (agent2 2026-10-08)."""
    out = []
    instruction = "Name the thing that matches each description." + (NAMING_RULE if chat else "")
    demos = [_naming_block(d["definition"], d["target"][:2], d["target"]) for d in DEMOS]
    for row in items.to_dict("records"):
        text = "\n\n".join([instruction, *demos, _naming_block(row["definition"])])
        out.append(NamingPrompt(row["item_id"], "none_hinted_demos", text, dict(target=row["target"], associate=row["associate"])))
    return out


def options_word_prompts(items, chat=True):
    """The six-choice prompt, base order only, answered with the chosen word; candidates: the six options."""
    instruction = CHOICE_INSTRUCTION.format(ct=CANNOT_TELL) + (OPTIONS_WORD_RULE if chat else "")
    demos = []
    for d in DEMOS:
        lines = [f"Description: {d['definition']}", *(f"{l}. {w}" for l, w in zip(LETTERS, d["options"])), "Answer:"]
        demos.append("\n".join(lines) + f" {d['target']}")
    out = []
    for row in items.to_dict("records"):
        roles = base_order(row["item_id"])
        words = {r: (CANNOT_TELL if r == "cannot_tell" else row[r]) for r in roles}
        text = "\n\n".join([instruction, *demos, _choice_block(row["definition"], [words[r] for r in roles])])
        out.append(NamingPrompt(row["item_id"], "options_word", text, {r: words[r] for r in ROLES}))
    return out


def foils(pool):
    """For each concept in the pool (rows with item_id, target, definition, category, domain, order): the item_id
    of the concept whose definition serves as its foil."""
    p = pool.sort_values("order").reset_index(drop=True).assign(any="")
    out = {}
    for i, row in p.iterrows():
        for key in ("category", "domain", "any"):
            same = p[(p[key] == row[key]) & (p.item_id != row.item_id)]
            same = same[[not ki.leaks(d, row.target) for d in same.definition]]
            if len(same):
                after = same[same.index > i]
                out[row.item_id] = (after if len(after) else same).iloc[0].item_id
                break
    return out


def word_to_definition_items(items, pool, chat=True):
    """Two orders per concept: its own definition at A, then at B."""
    fo = foils(pool)
    by_id = pool.set_index("item_id")
    defs = by_id.definition
    instruction = W2D_INSTRUCTION + (LABEL_RULE if chat else "")
    demo_pairs = [(DEMOS[0], DEMOS[1], "A"), (DEMOS[1], DEMOS[2], "B"), (DEMOS[2], DEMOS[0], "A")]
    demos = []
    for d, foil, at in demo_pairs:
        a, b = (d["definition"], foil["definition"]) if at == "A" else (foil["definition"], d["definition"])
        demos.append(f'Word: {d["target"]}\nA. {a}\nB. {b}\nAnswer: {at}')
    out = []
    for row in items.to_dict("records"):
        own, other = row["definition"], defs[fo[row["item_id"]]]
        foil_id = fo[row["item_id"]]
        foil = by_id.loc[foil_id]
        match = next((k for k in ("category", "domain") if row[k] == foil[k]), "any")
        for variant, (a, b, correct) in enumerate(((own, other, "A"), (other, own, "B"))):
            text = "\n\n".join([instruction, *demos, f'Word: {row["target"]}\nA. {a}\nB. {b}\nAnswer:'])
            out.append(LabelItem(row["item_id"], "word_to_definition", variant, text, correct, ("A", "B"), foil_id, match))
    return out


def options_letter_items(items, chat=True):
    """The existing six-choice prompts (6 rotations) as label items; correct = the target's letter."""
    from .m3_scoring import build_choice_prompts

    out = []
    for p in build_choice_prompts(items, chat):
        correct = next(l for l, r in p.letter_to_role.items() if r == "target")
        out.append(LabelItem(p.item_id, "options_letter", p.rotation, p.text, correct, LETTERS))
    return out


def copy_prompts(items, chat=True):
    instruction = COPY_INSTRUCTION + (NAMING_RULE if chat else "")
    demos = [f"Word: {d['target']}\nAnswer: {d['target']}" for d in DEMOS]
    return [NamingPrompt(r["item_id"], "copy", "\n\n".join([instruction, *demos, f"Word: {r['target']}\nAnswer:"]),
                         dict(target=r["target"])) for r in items.to_dict("records")]


INTRUSION_CONDITIONS = ("distract_none", "distract_assoc", "distract_unrel")
INTRUSION_DEMO_WORDS = ("alarm", "onion", "stairs")   # shown before the clock, kite, elevator demos: related, unrelated, related


def intrusion_prompts(items, chat=True):
    """Context-intrusion test (user decision Q21, 2026-10-08): free naming with a word to ignore shown before the
    description, the associate (distract_assoc), the unrelated word (distract_unrel), or none (distract_none).
    The demos always show a word to ignore (two related, one unrelated) and answer with their own target.
    Candidates: target, associate, unrelated. The pull of the associate is measured against the same prompt with
    the unrelated word shown, so the format is identical (intrusion_measures)."""
    instruction = ("Name the thing that matches each description. Ignore the word shown before it."
                   + (NAMING_RULE if chat else ""))
    demos = [f"Ignore this word: {w}\n" + _naming_block(d["definition"], None, d["target"])
             for d, w in zip(DEMOS, INTRUSION_DEMO_WORDS)]
    out = []
    for cond in INTRUSION_CONDITIONS:
        for row in items.to_dict("records"):
            shown = {"distract_none": None, "distract_assoc": row["associate"], "distract_unrel": row["unrelated"]}[cond]
            block = (f"Ignore this word: {shown}\n" if shown else "") + _naming_block(row["definition"])
            out.append(NamingPrompt(row["item_id"], cond, "\n\n".join([instruction, *demos, block]),
                                    dict(target=row["target"], associate=row["associate"], unrelated=row["unrelated"])))
    return out


def intrusion_measures(cand):
    """Per state and concept, from candidate scores of intrusion_prompts:
    pull_assoc = [logP(associate) - logP(target)] with the associate shown minus the same with the unrelated
    word shown; pull_unrel likewise for the unrelated word; and the target's log-probability with the associate
    shown (index: state, item_id)."""
    c = cand[cand.condition.isin(INTRUSION_CONDITIONS)]
    t = c.pivot_table(index=["state", "item_id"], columns=["condition", "role"], values="logprob")
    a_shown, u_shown = t["distract_assoc"], t["distract_unrel"]
    return pd.DataFrame({
        "pull_assoc": (a_shown.associate - a_shown.target) - (u_shown.associate - u_shown.target),
        "pull_unrel": (u_shown.unrelated - u_shown.target) - (a_shown.unrelated - a_shown.target),
        "target_logprob_assoc_shown": a_shown.target,
    })


def scored_prompts(items, chat=True, intrusion=False):
    """All candidate-scored prompts of the battery, with their format in `condition`; intrusion=True adds the
    context-intrusion test."""
    free = build_naming_prompts(items, ("none",), chat, roles=ERROR_ROLES)
    for p in free:
        p.condition = "name_free"
    fn = items[items.free_naming_ok.astype(bool)]
    return (free + name_hinted_demos_prompts(fn, chat) + build_naming_prompts(fn, ("cue", "miscue"), chat)
            + options_word_prompts(items, chat) + copy_prompts(items, chat)
            + (intrusion_prompts(items, chat) if intrusion else []))


def label_items(items, pool, chat=True):
    return options_letter_items(items, chat) + word_to_definition_items(items, pool, chat)


# --- scoring ---------------------------------------------------------------------------------------------------

SEGMENTS = ("context", "probe", "middle", "answer")


def is_segments(region):
    """True when region names prompt segments (a tuple or set drawn from SEGMENTS) rather than answer/prompt."""
    return isinstance(region, (tuple, list, set, frozenset))


def prompt_segments(tokenizer, text, chat=True):
    """(n_context, n_probe): token counts at the start of the encoded prompt.

    context = everything before the last block (instruction and demonstrations); probe = the last block's first
    line ("Description: ..." or "Word: ..."); middle = the rest of the prompt before its last position (hint or
    option lines, "Answer:" and the chat template's tail); answer = the last prompt position and the answer tokens,
    the same positions as answer_mask's `answer`. A token belongs to the segment where it starts.
    """
    ids = scoring.encode(tokenizer, text, chat)
    full = (tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            if chat else tokenizer.convert_ids_to_tokens(ids[:1])[0] + text)
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc.input_ids) != list(ids):
        raise ValueError("re-tokenised prompt does not match the scored prompt")
    start = max(full.rfind("\nDescription: "), full.rfind("\nWord: ")) + 1
    if start == 0:
        raise ValueError("prompt has no Description or Word block")
    end = full.index("\n", start)
    starts = [a for a, _ in enc.offset_mapping]
    n_context = sum(a < start for a in starts)
    n_probe = sum(start <= a < end for a in starts)
    if n_probe == 0 or n_context + n_probe >= len(ids) - 1:
        raise ValueError("could not place the probe line inside the prompt")
    return n_context, n_probe


def answer_mask(lengths, answer_lengths, width, region, bounds=None):
    """Bool mask (batch, width) for left-padded rows: `answer` = the last prompt position and the answer tokens,
    `prompt` = the prompt tokens before it, None or "all" = every real token. region may also be a set of
    SEGMENTS; bounds then gives (n_context, n_probe) per row (prompt_segments)."""
    import torch

    mask = torch.zeros((len(lengths), width), dtype=torch.bool)
    if is_segments(region):
        segs = set(region)
        if not segs or not segs <= set(SEGMENTS):
            raise ValueError(f"unknown segments {sorted(segs)}; use {SEGMENTS}")
        if bounds is None or len(bounds) != len(lengths):
            raise ValueError("segment regions need (n_context, n_probe) bounds for every row")
    for r, (n, a) in enumerate(zip(lengths, answer_lengths)):
        start = width - n
        if is_segments(region):
            c, p = bounds[r]
            spans = dict(context=(start, start + c), probe=(start + c, start + c + p),
                         middle=(start + c + p, width - a - 1), answer=(width - a - 1, width))
            for s in segs:
                lo, hi = spans[s]
                mask[r, lo:hi] = True
        elif region == "answer":
            mask[r, width - a - 1:] = True
        elif region == "prompt":
            mask[r, start:width - a - 1] = True
        else:
            mask[r, start:] = True
    return mask


def _region_context(model, mask):
    from contextlib import nullcontext

    if mask is None:
        return nullcontext()
    from . import hooks

    return hooks.position_mask(model, mask.to(next(model.parameters()).device))


def label_scores(model, tokenizer, items, chat=True, batch_size=32, region=None):
    """One row per label item: float32 logits of its labels at the answer position (full forward passes, so a
    position-restricted lesion applies to exactly the chosen positions)."""
    import torch

    device = next(model.parameters()).device
    seqs = [scoring.encode(tokenizer, it.text, chat) for it in items]
    labels = sorted({l for it in items for l in it.labels})
    lab_ids = scoring.label_token_ids(tokenizer, chat, labels)
    rows = [None] * len(items)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
            bounds = [prompt_segments(tokenizer, items[i].text, chat) for i in idx] if is_segments(region) else None
            pm = None if region is None else answer_mask([len(seqs[i]) for i in idx], [0] * len(idx), ids.shape[1], region, bounds)
            with _region_context(model, pm):
                logits = prefix_scoring.label_logits(model, [lab_ids[l] for l in labels], input_ids=ids,
                                                     attention_mask=mask, position_ids=pos).cpu()
            for r, i in enumerate(idx):
                it = items[i]
                lg = {l: float(logits[r, labels.index(l)]) for l in it.labels}
                others = np.array([v for l, v in lg.items() if l != it.correct])
                m = others.max()
                rows[i] = dict(item_id=it.item_id, fmt=it.fmt, variant=it.variant, correct=it.correct,
                               margin=lg[it.correct] - (m + np.log(np.exp(others - m).sum())),
                               top_correct=bool(lg[it.correct] > others.max()), foil_id=it.foil_id, foil_match=it.foil_match,
                               **{f"logit_{l}": v for l, v in lg.items()})
    return pd.DataFrame(rows)


def candidate_scores(model, tokenizer, prompts, chat=True, batch_size=16, region=None):
    """Like lesion.m3_scoring.score_naming_prompts (summed float32 log-probability of each candidate as the
    answer), with an optional position-restricted lesion region (answer_mask: "answer", "prompt" or a set of
    SEGMENTS)."""
    import torch

    from .m3_scoring import encode_answer

    device = next(model.parameters()).device
    jobs = [(i, role, word, *encode_answer(tokenizer, p.text, word, chat))
            for i, p in enumerate(prompts) for role, word in p.candidates.items()]
    seqs = [pre + ans for *_, pre, ans in jobs]
    segments = [prompt_segments(tokenizer, p.text, chat) for p in prompts] if is_segments(region) else None
    weight = model.get_output_embeddings().weight.float()
    out = [None] * len(jobs)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[j] for j in idx], tokenizer.eos_token_id or 0, device)
            width = ids.shape[1]
            bounds = [segments[jobs[j][0]] for j in idx] if is_segments(region) else None
            pm = None if region is None else answer_mask([len(seqs[j]) for j in idx], [len(jobs[j][4]) for j in idx], width, region, bounds)
            with _region_context(model, pm):
                hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
            for row, j in enumerate(idx):
                i, role, word, _, ans = jobs[j]
                n = len(ans)
                logp = torch.log_softmax(hidden[row, width - n - 1:width - 1].float() @ weight.T, dim=-1)
                answer = ids[row, width - n:]
                first = logp[0]
                out[j] = dict(item_id=prompts[i].item_id, condition=prompts[i].condition, role=role, word=word,
                              logprob=float(logp.gather(1, answer[:, None]).sum()), n_tokens=n,
                              first_token_rank=int((first > first[answer[0]]).sum()))
    return pd.DataFrame(out)


def greedy_answers(model, tokenizer, prompts, chat=True, batch_size=16, max_new_tokens=6):
    """What the model actually says: greedy continuation of each prompt, up to max_new_tokens."""
    import torch

    device = next(model.parameters()).device
    seqs = [scoring.encode(tokenizer, p.text, chat) for p in prompts]
    pad = tokenizer.eos_token_id or 0
    out = [None] * len(prompts)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, _ = scoring._pad_left([seqs[i] for i in idx], pad, device)
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=pad)
            for r, i in enumerate(idx):
                out[i] = dict(item_id=prompts[i].item_id, answer=tokenizer.decode(gen[r, ids.shape[1]:], skip_special_tokens=True))
    return pd.DataFrame(out)


def classify_answer(answer, row):
    """Which candidate a generated answer is: target, coordinate, superordinate, associate, unrelated or other.
    Lowercase, first word, trailing punctuation removed; a plural or singular form of a candidate counts as it."""
    words = str(answer).strip().lower().split()
    w = words[0].strip(".,;:!?\"'") if words else ""

    def forms(x):
        x = x.lower()
        return {x, x + "s", x + "es", x[:-1] if x.endswith("s") else x, x[:-2] if x.endswith("es") else x}

    for role in ERROR_ROLES:
        if w in forms(row[role]):
            return role
    return "other" if w else "empty"
