"""Prompt assembly and scoring for Experiment 0 (docs/plans/pilot-protocol.md §4, ticket pilot 04).

Label scoring: both options are shown, the model answers A or B; we read the logits of the two
label tokens at the answer position. Every item runs in both option orders. Cloze scoring
(colour and size only) compares the summed log-probability of the two option words.

This module does not know about ablations: callers wrap `score_*` in an ablation context
(src/lesion/hooks.py) when needed.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
M1 = ROOT / "data/materials/m1"
LABELS = ("A", "B")
DOC_START = "<|endoftext|>"  # OLMo tokenizers use it as the document boundary; the Instruct chat template starts with it too
# gpt-oss (P12, docs/plans/second-model-g1-p12.md): the official harmony template writes today's date, which is fixed
# here, and the scoring prompt adds an empty analysis channel and opens the final channel, where the label is read.
HARMONY_DATE = "2026-10-15"
HARMONY_SUFFIX = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
# P12b (user decision 2026-10-16): with "harmony", 77% of gpt-oss's demo prompts began with a label, below the 90% gate,
# and 10 of the other 11 began "Answer: A". "harmony_answer" also opens the final message with "Answer:", so the next
# token is the label with a space (" A"), as for base models. Chosen after seeing the format check; reported as such.
HARMONY_ANSWER = "Answer:"
FORMATS = ("plain", "chat_template", "harmony", "harmony_answer")


def load_materials(m1_dir=M1):
    items = pd.read_csv(m1_dir / "items.csv", dtype=str, keep_default_na=False)
    demos = pd.read_csv(m1_dir / "demos.csv", dtype=str, keep_default_na=False)
    templates = yaml.safe_load((m1_dir / "templates.yaml").read_text(encoding="utf-8"))
    return items, demos, templates


@dataclass
class LabelPrompt:
    item_id: str
    task: str
    order: int            # 0: option1 shown as A; 1: option2 shown as A
    text: str
    correct_label: str    # "A" or "B"
    label_to_option: dict  # {"A": "option1" or "option2", "B": ...}


def _block(templates, task, row, a, b):
    fields = dict(probe=row.get("probe", ""), word1=row.get("word1", ""), word2=row.get("word2", ""), a=a, b=b)
    return templates["blocks"][task].format(**fields)


def _demo_blocks(templates, demos, task, order):
    """Demo blocks with answers. Demos are fixed except for synonym, where A/B name the response options
    themselves, so in the swapped order the demo options and answers swap as well."""
    blocks = []
    for d in demos[demos.task == task].sort_values("position", key=lambda s: s.astype(int)).to_dict("records"):
        a, b, label = d["option_a"], d["option_b"], d["correct_label"]
        if task == "synonym" and order == 1:
            a, b, label = b, a, {"A": "B", "B": "A"}[label]
        if task == "synonym":
            a, b = templates["synonym_options"][a], templates["synonym_options"][b]
        blocks.append(_block(templates, task, d, a, b) + templates["demo_answer"].format(label=label))
    return blocks


def build_label_prompts(items, demos, templates, variant="main", chat=False, tasks=None):
    prompts = []
    for row in items.to_dict("records"):
        task = row["task"]
        if tasks and task not in tasks:
            continue
        instruction = templates["instructions"][task][variant]
        if chat:
            instruction += templates["instruct"]["reply_rule"]
        for order in (0, 1):
            first, second = ("option1", "option2") if order == 0 else ("option2", "option1")
            a, b = row[first], row[second]
            if task == "synonym":
                a, b = templates["synonym_options"][a], templates["synonym_options"][b]
            test = _block(templates, task, row, a, b)
            text = "\n\n".join([instruction, *_demo_blocks(templates, demos, task, order), test])
            label_to_option = {"A": first, "B": second}
            correct = "A" if label_to_option["A"] == row["correct_option"] else "B"
            prompts.append(LabelPrompt(row["item_id"], task, order, text, correct, label_to_option))
    return prompts


def harmony_text(tokenizer, text, answer=False):
    """gpt-oss scoring prompt as a string: the official template (reasoning effort low) with the date line fixed to
    HARMONY_DATE, then HARMONY_SUFFIX, so that the next token is the answer in the final channel; with answer, also
    HARMONY_ANSWER ("harmony_answer"), so that the next token is the label after "Answer:"."""
    import re

    s = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                      reasoning_effort="low")
    s, n = re.subn(r"Current date: \d{4}-\d{2}-\d{2}\n", f"Current date: {HARMONY_DATE}\n", s)
    if n != 1 or not s.endswith("<|start|>assistant"):
        raise ValueError("the gpt-oss chat template no longer has the expected date line or generation prompt")
    return s + HARMONY_SUFFIX + (HARMONY_ANSWER if answer else "")


def encode(tokenizer, text, chat, fmt=None):
    """Token ids of the full prompt; the next token is the answer. fmt (lesion.models.prompt_format's first element)
    overrides chat: "plain", "chat_template", "harmony" or "harmony_answer"."""
    if fmt is not None and fmt not in FORMATS:
        raise ValueError(f"unknown prompt format {fmt!r}")
    if fmt in ("harmony", "harmony_answer"):
        return tokenizer(harmony_text(tokenizer, text, answer=fmt == "harmony_answer"), add_special_tokens=False).input_ids
    if (chat if fmt is None else fmt == "chat_template"):
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    start = tokenizer.convert_tokens_to_ids([DOC_START])
    # an unknown token may come back as the unknown-token id rather than None, so ask the vocabulary when there is one
    known = DOC_START in tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else start[0] is not None
    if not known:                                # no <|endoftext|> (DeepSeek-V2): its own beginning-of-sequence token
        if tokenizer.bos_token_id is None:
            raise ValueError(f"the tokenizer has neither {DOC_START} nor a beginning-of-sequence token")
        start = [tokenizer.bos_token_id]
    return start + tokenizer.encode(text, add_special_tokens=False)


def label_token_ids(tokenizer, chat, labels=LABELS, prefix=None):
    """Base models continue "Answer:" with " A"; chat models start the assistant turn with "A". prefix ("bare" or
    "space", lesion.models.prompt_format's second element) overrides that rule."""
    if prefix not in (None, "bare", "space"):
        raise ValueError(f"unknown label prefix {prefix!r}")
    space = (not chat) if prefix is None else prefix == "space"
    ids = {}
    for lab in labels:
        enc = tokenizer.encode(" " + lab if space else lab, add_special_tokens=False)
        assert len(enc) == 1, f"label {lab!r} is not a single token: {enc}"
        ids[lab] = enc[0]
    return ids


def _batches(seqs, batch_size):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def _pad_left(seqs, pad_id, device):
    import torch

    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), positions.to(device)


def next_token_logits(model, seqs, token_ids, batch_size=16, pad_id=0):
    """Logits of `token_ids` at the position after each sequence, as float32 (n_seqs x n_tokens)."""
    import torch

    device = next(model.parameters()).device
    out = torch.empty((len(seqs), len(token_ids)), dtype=torch.float32)
    with torch.no_grad():
        for idx in _batches(seqs, batch_size):
            ids, mask, pos = _pad_left([seqs[i] for i in idx], pad_id, device)
            # only the last position is projected onto the vocabulary (the same logits; a 201k-token vocabulary at
            # every position of a batch does not fit in memory)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
            out[idx] = logits[:, token_ids].float().cpu()
    return out


def score_label_prompts(model, tokenizer, prompts, chat, batch_size=16, fmt=None, label_prefix=None):
    """One row per prompt: logits of A and B, which is correct, and the model's choice. fmt and label_prefix: see
    encode and label_token_ids (None keeps the rule given by chat)."""
    lab_ids = label_token_ids(tokenizer, chat, prefix=label_prefix)
    seqs = [encode(tokenizer, p.text, chat, fmt) for p in prompts]
    logits = next_token_logits(model, seqs, [lab_ids[l] for l in LABELS], batch_size, tokenizer.eos_token_id or 0)
    rows = []
    for p, (la, lb) in zip(prompts, logits.tolist()):
        wrong = "B" if p.correct_label == "A" else "A"
        by_label = {"A": la, "B": lb}
        rows.append(dict(item_id=p.item_id, task=p.task, order=p.order, correct_label=p.correct_label,
                         logit_A=la, logit_B=lb, logit_correct=by_label[p.correct_label], logit_wrong=by_label[wrong],
                         choice="A" if la > lb else "B", chosen_option=p.label_to_option["A" if la > lb else "B"],
                         option_A=p.label_to_option["A"]))
    return pd.DataFrame(rows)


# --- Cloze scoring (secondary, colour and size only; pilot protocol §4.2) -------------------

def build_cloze_prompts(items, demos, templates):
    """(item_id, task, prefix text, {option1: word, option2: word}) for colour and size items."""
    out = []
    for row in items[items.task.isin(["colour", "size"])].to_dict("records"):
        task = row["task"]
        block = templates["cloze"]["block"]
        demo_rows = demos[demos.task == task].sort_values("position", key=lambda s: s.astype(int)).to_dict("records")
        demo_text = [block.format(probe=d["probe"], dimension=task) + templates["cloze"]["demo_answer"].format(target=d["target"])
                     for d in demo_rows]
        test = block.format(probe=row["probe"], dimension=task)
        text = "\n\n".join([templates["cloze"]["instructions"][task], *demo_text, test])
        out.append((row["item_id"], task, text, {"option1": row["option1"], "option2": row["option2"]}))
    return out


def continuation_logprob(model, prefix_ids, cont_ids_list, batch_size=16, pad_id=0):
    """Summed log-probability of each continuation given its prefix, as a list of floats."""
    import torch

    device = next(model.parameters()).device
    seqs = [p + c for p, c in zip(prefix_ids, cont_ids_list)]
    result = [0.0] * len(seqs)
    with torch.no_grad():
        for idx in _batches(seqs, batch_size):
            ids, mask, pos = _pad_left([seqs[i] for i in idx], pad_id, device)
            # left padding puts every continuation at the end of its row, so only the last `keep` positions are
            # projected onto the vocabulary (the same log-probabilities; a 201k-token vocabulary at every position
            # of a batch does not fit in memory)
            keep = max(len(cont_ids_list[i]) for i in idx) + 1
            logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                           logits_to_keep=keep).logits.float(), -1)
            width = ids.shape[1]
            offset = width - keep                # column j of the full sequence is column j - offset here
            for row, i in enumerate(idx):
                n_cont = len(cont_ids_list[i])
                start = width - n_cont  # first continuation token position
                targets = ids[row, start:]
                result[i] = logp[row, start - 1 - offset:width - 1 - offset].gather(-1, targets[:, None]).sum().item()
    return result


def score_cloze(model, tokenizer, cloze_prompts, batch_size=16):
    prefixes, conts, keys = [], [], []
    for item_id, task, text, words in cloze_prompts:
        prefix = encode(tokenizer, text, chat=False)
        for opt, word in words.items():
            prefixes.append(prefix)
            conts.append(tokenizer.encode(" " + word, add_special_tokens=False))
            keys.append((item_id, task, opt, len(conts[-1])))
    lp = continuation_logprob(model, prefixes, conts, batch_size, tokenizer.eos_token_id or 0)
    return pd.DataFrame([dict(item_id=i, task=t, option=o, n_tokens=n, logprob=v) for (i, t, o, n), v in zip(keys, lp)])
