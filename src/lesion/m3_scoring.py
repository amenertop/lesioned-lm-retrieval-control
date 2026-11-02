"""Prompts and scoring for experiment 3 (docs/plans/experiment3-protocol.md §6, §7) and the intact-model
validity check on its items (docs/plans/materials-m2-m3.md §5.8).

H3a, definition naming with six options. A definition and six labelled options: the target, coordinate,
superordinate, associate, unrelated word and "cannot tell". Each item runs in 6 cyclic rotations of an
item-specific base order (seeded by the item id), so every option sits once at every letter; the six label logits
are read in float32 at the answer position (lesion.prefix_scoring, shared prefix computed once).

H3b, free naming. The definition, then optionally a hint with the first two letters of the target (cue) or of
the associate (miscue); the log-probabilities of the target and of the associate as the answer are compared
(restricted scoring). Chat models: the answer starts the assistant turn, without a leading space, and only the
word's tokens are scored, not the end marker; base models: " word" after "Answer:" (§7.2). The answer tokens are
taken from the chat template and must equal lesion.m3_items.token_counts' encoding, the one the materials were
checked with.

Demos: DEMOS, three concepts used nowhere else; neither they nor their options occur in M1, the naming test or M3.
The answers sit at different letters; "cannot tell" is never the demo answer. In the hint conditions the demo
hints are always correct; without hints the demos have none (§7.1).

This module does not know about ablations: callers wrap the scoring calls in an ablation context.
"""
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import naming, prefix_scoring, scoring

LETTERS = ("A", "B", "C", "D", "E", "F")
ROLES = ("target", "coordinate", "superordinate", "associate", "unrelated", "cannot_tell")
CANNOT_TELL = "cannot tell"
CHOICE_INSTRUCTION = 'Choose the word that the description defines. If none of the words fits, choose "{ct}".'
CHOICE_REPLY_RULE = " Reply with the letter only."
NAMING_INSTRUCTION = "Name the thing that matches each description."
NAMING_REPLY_RULE = " Reply with the word only, in lowercase."
HINT = 'Hint: the word starts with "{xx}".'
CONDITIONS = ("none", "cue", "miscue")
CHOICE_BATCHES = (400, 128, 48)   # as prefix_avg3 in experiments 1 and 2 (ADR 0004)
NAMING_BATCHES = (16, 11, 7)      # as the naming scores of the knowledge test

# Demo concepts. options in the order shown, A to F: coordinate, superordinate, associate and unrelated word as in
# the items; none of these words occurs in M1, the naming test or M3 (tests/test_m3_scoring.py).
DEMOS = (
    dict(target="clock", definition="an object on a wall or desk with a numbered face and moving hands that show the time",
         options=("watch", "clock", "pillow", "timepiece", CANNOT_TELL, "alarm")),
    dict(target="kite", definition="a light frame covered with paper or plastic that children fly high on a long cord",
         options=("plaything", "sky", "doll", CANNOT_TELL, "kite", "onion")),
    dict(target="elevator", definition="a moving compartment that carries people up and down between the floors of a building",
         options=("stairs", CANNOT_TELL, "machine", "elevator", "blanket", "escalator")),
)


@dataclass
class ChoicePrompt:
    item_id: str
    rotation: int
    text: str
    letter_to_role: dict      # {"A": "associate", ...}


@dataclass
class NamingPrompt:
    item_id: str
    condition: str            # none, cue or miscue
    text: str
    candidates: dict          # {"target": word, "associate": word}


def base_order(item_id):
    """Item-specific order of the six roles, seeded by the item id (independent of the item's position)."""
    rng = np.random.default_rng(int(hashlib.sha256(str(item_id).encode()).hexdigest()[:8], 16))
    return [ROLES[i] for i in rng.permutation(len(ROLES))]


def _choice_block(definition, words, answer=None):
    lines = [f"Description: {definition}", *(f"{l}. {w}" for l, w in zip(LETTERS, words)), "Answer:"]
    return "\n".join(lines) + (f" {answer}" if answer else "")


def build_choice_prompts(items, chat=True):
    """6 prompts per item, the cyclic rotations of its base order."""
    instruction = CHOICE_INSTRUCTION.format(ct=CANNOT_TELL) + (CHOICE_REPLY_RULE if chat else "")
    demos = [_choice_block(d["definition"], d["options"], LETTERS[d["options"].index(d["target"])]) for d in DEMOS]
    prompts = []
    for row in items.to_dict("records"):
        order = base_order(row["item_id"])
        for r in range(len(ROLES)):
            roles = order[len(order) - r:] + order[:len(order) - r]
            words = [CANNOT_TELL if x == "cannot_tell" else row[x] for x in roles]
            text = "\n\n".join([instruction, *demos, _choice_block(row["definition"], words)])
            prompts.append(ChoicePrompt(row["item_id"], r, text, dict(zip(LETTERS, roles))))
    return prompts


def _choice_rows(prompts, logits):
    rows = []
    for p, lg in zip(prompts, logits):
        row = dict(item_id=p.item_id, rotation=p.rotation)
        row.update({f"logit_{l}": float(v) for l, v in zip(LETTERS, lg)})
        row.update({f"logit_{p.letter_to_role[l]}": float(v) for l, v in zip(LETTERS, lg)})
        rows.append(row)
    return pd.DataFrame(rows)


def score_choice_prompts(model, tokenizer, prompts, chat=True, batch_size=64):
    """One row per prompt: the six label logits (float32), keyed by letter and by role."""
    ids = scoring.label_token_ids(tokenizer, chat, LETTERS)
    seqs = [scoring.encode(tokenizer, p.text, chat) for p in prompts]
    logits = prefix_scoring.next_token_logits(model, seqs, [0] * len(seqs), [ids[l] for l in LETTERS], batch_size,
                                              tokenizer.eos_token_id or 0)
    return _choice_rows(prompts, logits.tolist())


def averaged_choice(model, tokenizer, prompts, chat=True, batch_sizes=CHOICE_BATCHES):
    """score_choice_prompts averaged over several batch compositions (Qwen's logits move with them; ADR 0004)."""
    tables = [score_choice_prompts(model, tokenizer, prompts, chat, bs) for bs in batch_sizes]
    return _choice_rows(prompts, np.mean([t[[f"logit_{l}" for l in LETTERS]].to_numpy() for t in tables], axis=0))


def choice_probabilities(scores):
    """Per item: probability of each role, softmax over the six labels within a rotation, then averaged over the
    rotations (§6). Columns item_id, p_target, ..., p_cannot_tell."""
    logits = scores[[f"logit_{r}" for r in ROLES]].to_numpy()
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    t = pd.DataFrame(p, columns=[f"p_{r}" for r in ROLES]).assign(item_id=scores.item_id.to_numpy())
    return t.groupby("item_id", sort=False).mean().reset_index()


def _naming_block(definition, hint=None, answer=None):
    lines = [f"Description: {definition}"] + ([HINT.format(xx=hint)] if hint else []) + ["Answer:"]
    return "\n".join(lines) + (f" {answer}" if answer else "")


def build_naming_prompts(items, conditions=CONDITIONS, chat=True, roles=("target", "associate")):
    """One prompt per item and condition; the demos carry correct hints in the hint conditions. roles: the words
    scored as candidate answers (H3b: target and associate; naming errors: ERROR_ROLES)."""
    instruction = NAMING_INSTRUCTION + (NAMING_REPLY_RULE if chat else "")
    prompts = []
    for cond in conditions:
        demos = [_naming_block(d["definition"], None if cond == "none" else d["target"][:2], d["target"]) for d in DEMOS]
        for row in items.to_dict("records"):
            hint = {"none": None, "cue": row["target"][:2], "miscue": row["associate"][:2]}[cond]
            text = "\n\n".join([instruction, *demos, _naming_block(row["definition"], hint)])
            prompts.append(NamingPrompt(row["item_id"], cond, text, {r: row[r] for r in roles}))
    return prompts


def candidate_ids(tokenizer, word, chat=True):
    """Tokens of an answer word as the materials were checked (lesion.m3_items.token_counts)."""
    return tokenizer.encode(word if chat else " " + word, add_special_tokens=False)


def encode_answer(tokenizer, text, word, chat=True):
    """(prompt ids, answer ids). Chat models: the assistant turn as the chat template writes it, cut before its end
    token; base models: " word" after the prompt. Raises if the answer tokens differ from candidate_ids."""
    if chat:
        prompt, answer = naming.encode_pair(tokenizer, naming.NamingPrompt(None, word, text), chat=True)
        if answer and answer[-1] in naming._end_ids(tokenizer):
            answer = answer[:-1]
    else:
        prompt, answer = scoring.encode(tokenizer, text, chat=False), candidate_ids(tokenizer, word, chat=False)
    if list(answer) != list(candidate_ids(tokenizer, word, chat)):
        raise ValueError(f"{word!r}: the chat template encodes the answer as {answer}, the materials check as "
                         f"{candidate_ids(tokenizer, word, chat)}")
    return list(prompt), list(answer)


def score_naming_prompts(model, tokenizer, prompts, chat=True, batch_size=16):
    """One row per prompt and candidate: summed float32 log-probability of the candidate's tokens as the answer,
    its token count, and the rank of its first token among all tokens (0 = the model's top choice)."""
    import torch

    device = next(model.parameters()).device
    jobs = [(i, role, word, *encode_answer(tokenizer, p.text, word, chat))
            for i, p in enumerate(prompts) for role, word in p.candidates.items()]
    seqs = [pre + ans for *_, pre, ans in jobs]
    weight = model.get_output_embeddings().weight.float()
    out = [None] * len(jobs)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[j] for j in idx], tokenizer.eos_token_id or 0, device)
            hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
            width = ids.shape[1]
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


def averaged_naming(model, tokenizer, prompts, chat=True, batch_sizes=NAMING_BATCHES):
    """score_naming_prompts with the log-probabilities averaged over several batch compositions (ADR 0004); the
    first-token rank is taken from the first composition."""
    tables = [score_naming_prompts(model, tokenizer, prompts, chat, bs) for bs in batch_sizes]
    out = tables[0].copy()
    out["logprob"] = np.mean([t.logprob.to_numpy() for t in tables], axis=0)
    return out


def naming_deltas(scores):
    """Per item and condition: delta = log P(target) - log P(associate); also per token (robustness, §7.2) and the
    target's first-token rank."""
    t = scores.pivot_table(index=["item_id", "condition"], columns="role",
                           values=["logprob", "n_tokens", "first_token_rank"], sort=False)
    return pd.DataFrame({
        "delta": t[("logprob", "target")] - t[("logprob", "associate")],
        "delta_per_token": (t[("logprob", "target")] / t[("n_tokens", "target")]
                            - t[("logprob", "associate")] / t[("n_tokens", "associate")]),
        "target_first_rank": t[("first_token_rank", "target")],
    }).reset_index()


# --- naming errors (H3a in free naming; user decision 2026-10-07, Q13) --------------------------------------
# The six-choice task leaves the target about 25 logits ahead of every wrong option even under 16 control heads or
# 30 knowledge components (calibration probe), so error types are read from free naming instead: without options or
# hints, how probable is each candidate word as the answer.

ERROR_ROLES = ("target", "coordinate", "superordinate", "associate", "unrelated")


def naming_errors(scores):
    """Per item (one condition): for each wrong candidate, its log-probability as the answer minus the target's
    (gap_<role>, whole word; gap_pt_<role>, per token), the candidates' probabilities renormalised over the five
    candidates (p_<role>), S = p_associate - (p_coordinate + p_superordinate) and, on the log scale,
    S_log = gap_associate - (gap_coordinate + gap_superordinate) / 2 (per token: S_log_pt); and the target's own
    log-probability (target_logprob)."""
    lp = scores.pivot_table(index="item_id", columns="role", values="logprob", sort=False)[list(ERROR_ROLES)]
    nt = scores.pivot_table(index="item_id", columns="role", values="n_tokens", sort=False)[list(ERROR_ROLES)]
    pt = lp / nt
    out = pd.DataFrame(index=lp.index)
    out["target_logprob"] = lp.target
    p = np.exp(lp.to_numpy() - lp.to_numpy().max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    for j, r in enumerate(ERROR_ROLES):
        out[f"p_{r}"] = p[:, j]
    for r in ERROR_ROLES[1:]:
        out[f"gap_{r}"] = lp[r] - lp.target
        out[f"gap_pt_{r}"] = pt[r] - pt.target
    out["S"] = out.p_associate - out.p_coordinate - out.p_superordinate
    out["S_log"] = out.gap_associate - (out.gap_coordinate + out.gap_superordinate) / 2
    out["S_log_pt"] = out.gap_pt_associate - (out.gap_pt_coordinate + out.gap_pt_superordinate) / 2
    return out.reset_index()


# --- validity check on the intact model (materials-m2-m3.md §5.8, rules written before any model run) ---------

MIN_P_TARGET = 0.50                             # rule 1: mean p(target) over the rotations, and the highest option
SENSITIVITY_RANK, SENSITIVITY_DELTA = 10, 1.0   # rule 4: first token in the top 10 and delta >= 1.0 (report only)


def validity(items, choice_scores, naming_scores):
    """Per item: the six option probabilities and choice_ok (rule 1); for items scored in free naming, the no-hint
    delta, naming_ok (delta > 0, rule 2) and sensitivity_ok (rule 4). Items not scored in free naming get NaN
    there and naming_ok False."""
    probs = choice_probabilities(choice_scores)
    p = probs.set_index("item_id")[[f"p_{r}" for r in ROLES]]
    probs["choice_ok"] = ((p.p_target >= MIN_P_TARGET) & (p.idxmax(axis=1) == "p_target")).to_numpy()
    d = naming_deltas(naming_scores)
    d = d[d.condition == "none"].drop(columns="condition")
    d["naming_ok"] = d.delta > 0
    d["sensitivity_ok"] = (d.target_first_rank < SENSITIVITY_RANK) & (d.delta >= SENSITIVITY_DELTA)
    t = items[["item_id", "role", "zipf_bin", "target", "free_naming_ok", "target_more_common", "order"]]
    t = t.merge(probs, on="item_id", how="left").merge(d, on="item_id", how="left")
    t["naming_ok"] = t.naming_ok.fillna(False).astype(bool)
    t["sensitivity_ok"] = t.sensitivity_ok.fillna(False).astype(bool)
    return t


def hint_effects(naming_scores):
    """Intact-model check that the hints work (report only, no rule): per item, delta with the cue and with the
    miscue minus delta without a hint, and R = delta_cue - delta_miscue (experiment3-protocol.md §7.2)."""
    cols = ["item_id", "cue_effect", "miscue_effect", "R"]
    if not set(CONDITIONS) <= set(naming_scores.get("condition", ())):
        return pd.DataFrame(columns=cols)
    d = naming_deltas(naming_scores).pivot(index="item_id", columns="condition", values="delta")
    return pd.DataFrame({"cue_effect": d.cue - d["none"], "miscue_effect": d.miscue - d["none"],
                         "R": d.cue - d.miscue}).reset_index()


def _pass_counts(t, flag):
    """Per set and Zipf bin (as planned before the check): concepts checked, passed and failed."""
    return {f"{r}, bin {int(b)}": dict(n=len(g), passed=int(g[flag].sum()), failed=int((~g[flag]).sum()))
            for (r, b), g in t.groupby(["role", "zipf_bin"])}


def validity_summary(check, hints=None):
    """What §5.8 rule 5 asks to report: per set and Zipf bin, how many concepts pass each rule (free naming among
    the concepts usable for it); the mean probability of every option type; and, if given, the intact-model hint
    effects (mean and share above zero)."""
    t = check.copy()
    fn = t[t.free_naming_ok.astype(bool)]
    out = dict(
        choice_pass=_pass_counts(t, "choice_ok"),
        naming_pass=_pass_counts(fn, "naming_ok"),
        sensitivity_pass=_pass_counts(fn, "sensitivity_ok"),
        mean_probability={r: round(float(t[f"p_{r}"].mean()), 4) for r in ROLES},
        mean_probability_failed_choice={r: round(float(t.loc[~t.choice_ok, f"p_{r}"].mean()), 4) for r in ROLES}
        if (~t.choice_ok).any() else {},
    )
    if hints is not None and len(hints):
        out["hint_effects"] = {c: dict(n=int(hints[c].notna().sum()), mean=round(float(hints[c].mean()), 3),
                                       share_positive=round(float((hints[c] > 0).mean()), 3))
                               for c in ("cue_effect", "miscue_effect", "R")}
    return out
