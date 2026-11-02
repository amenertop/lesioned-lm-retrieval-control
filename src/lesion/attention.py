"""Where do given attention heads look at the answer position? (control-head exploration, 2026-09-28)

For two-choice M1 prompts, the answer is read at the last prompt position. For each listed head this
returns the attention weight from that position to the tokens of the correct option word and of the
distractor word in the test block. A control head that suppresses the associate would put little weight
on the distractor in high-interference items, or more weight on the target.
"""
import pandas as pd

from . import scoring


def option_spans(tokenizer, prompt, chat):
    """Token index ranges of the option A and option B words in the last block of the prompt."""
    import re

    ids = scoring.encode(tokenizer, prompt.text, chat)
    text = (tokenizer.apply_chat_template([{"role": "user", "content": prompt.text}], tokenize=False, add_generation_prompt=True)
            if chat else tokenizer.convert_ids_to_tokens(ids[:1])[0] + prompt.text)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc.input_ids) != list(ids):
        raise ValueError("re-tokenised prompt does not match the scored prompt")
    last = text.rfind("\nA: ")
    spans = {}
    for label in ("A", "B"):
        m = re.compile(rf"\n{label}: ([^\n]+)").search(text, last)
        lo, hi = m.span(1)
        spans[label] = [i for i, (a, b) in enumerate(enc.offset_mapping) if a < hi and b > lo]
    return ids, spans


def head_attention(model, tokenizer, prompts, chat, heads, batch_size=8):
    """One row per prompt x head: attention from the answer position to the correct and the wrong option word."""
    import torch

    device = next(model.parameters()).device
    prepared = [option_spans(tokenizer, p, chat) for p in prompts]
    rows = []
    with torch.no_grad():
        for idx in scoring._batches([ids for ids, _ in prepared], batch_size):
            seqs = [prepared[i][0] for i in idx]
            ids, mask, pos = scoring._pad_left(seqs, tokenizer.eos_token_id or 0, device)
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_attentions=True)
            width = ids.shape[1]
            for row, i in enumerate(idx):
                p, (seq, spans) = prompts[i], prepared[i]
                shift = width - len(seq)
                correct = p.correct_label
                wrong = "B" if correct == "A" else "A"
                for h in heads:
                    att = out.attentions[h.layer][row, h.index, -1].float()
                    rows.append(dict(item_id=p.item_id, task=p.task, order=p.order, head=h.key,
                                     to_correct=float(att[[shift + j for j in spans[correct]]].sum()),
                                     to_wrong=float(att[[shift + j for j in spans[wrong]]].sum())))
    return pd.DataFrame(rows)


REGIONS = ("start", "template", "instruction", "demos", "probe_line", "correct_option", "wrong_option", "cue")


def regions(tokenizer, prompt, chat):
    """(token ids, {region: token indices}) covering every prompt token once: the first token (where many
    heads park attention), chat-template tokens, the instruction, the demos, the test block's first line,
    the correct and the wrong option word, and the rest of the test block (labels and "Answer:")."""
    ids, spans = option_spans(tokenizer, prompt, chat)
    text = (tokenizer.apply_chat_template([{"role": "user", "content": prompt.text}], tokenize=False, add_generation_prompt=True)
            if chat else tokenizer.convert_ids_to_tokens(ids[:1])[0] + prompt.text)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    lo = text.find(prompt.text)
    hi = lo + len(prompt.text)
    first_break = text.find("\n\n", lo)
    last_block = text.rfind("\n\n", lo, hi) + 2
    probe_end = text.find("\n", last_block)
    wrong = "B" if prompt.correct_label == "A" else "A"
    out = {r: [] for r in REGIONS}
    taken = set(spans["A"]) | set(spans["B"])
    out["correct_option"], out["wrong_option"] = list(spans[prompt.correct_label]), list(spans[wrong])
    for i, (a, b) in enumerate(enc.offset_mapping):
        if i in taken:
            continue
        mid = (a + b) / 2
        if i == 0:
            r = "start"
        elif mid < lo or mid >= hi:
            r = "template"
        elif mid < first_break:
            r = "instruction"
        elif mid < last_block:
            r = "demos"
        elif mid < probe_end:
            r = "probe_line"
        else:
            r = "cue"
        out[r].append(i)
    return ids, out


def attention_profile(model, tokenizer, prompts, chat, heads, batch_size=4):
    """One row per prompt x head x region: the share of the answer position's attention that the head puts on
    that part of the prompt (shares over REGIONS sum to 1)."""
    import torch

    device = next(model.parameters()).device
    prepared = [regions(tokenizer, p, chat) for p in prompts]
    rows = []
    with torch.no_grad():
        for idx in scoring._batches([ids for ids, _ in prepared], batch_size):
            seqs = [prepared[i][0] for i in idx]
            ids, mask, pos = scoring._pad_left(seqs, tokenizer.eos_token_id or 0, device)
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_attentions=True)
            width = ids.shape[1]
            for row, i in enumerate(idx):
                seq, reg = prepared[i]
                shift = width - len(seq)
                for h in heads:
                    att = out.attentions[h.layer][row, h.index, -1].float()
                    for r, toks in reg.items():
                        rows.append(dict(item_id=prompts[i].item_id, task=prompts[i].task, order=prompts[i].order,
                                         head=h.key, region=r, share=float(att[[shift + j for j in toks]].sum()) if toks else 0.0))
    return pd.DataFrame(rows)
