"""Mechanism readouts for the experiment 3 battery (docs/plans/experiment3-protocol.md §5.5; user decision Q18):

- logit_lens: at the answer position, every layer's output passed through the final norm and the output layer
  (float32): log-probability and rank of the target's first answer token. Where along the depth the intact model
  already "has" the word, and where a lesion takes it away. Layer outputs are captured with forward hooks on the
  decoder layers (before the final norm), so the readout does not depend on how a transformers version fills
  hidden_states.
- hint_attention: attention from the answer position to the two hint letters and to the definition of the test
  block, for given heads (eager attention, set by the caller).
"""
import re

import pandas as pd

from . import scoring
from .m3_scoring import encode_answer


def logit_lens(model, tokenizer, prompts, chat=True, batch_size=8, role="target"):
    """One row per prompt and layer: log-probability and rank of the first token of the `role` candidate."""
    import torch

    device = next(model.parameters()).device
    layers = model.model.layers
    seqs, firsts = [], []
    for p in prompts:
        pre, ans = encode_answer(tokenizer, p.text, p.candidates[role], chat)
        seqs.append(pre)
        firsts.append(ans[0])
    weight = model.get_output_embeddings().weight.float()
    captured = {}
    handles = [layer.register_forward_hook(lambda _m, _i, out, k=k: captured.__setitem__(k, (out[0] if isinstance(out, tuple) else out)[:, -1].detach()))
               for k, layer in enumerate(layers)]
    rows = []
    try:
        with torch.no_grad():
            for idx in scoring._batches(seqs, batch_size):
                ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
                captured.clear()
                model.model(input_ids=ids, attention_mask=mask, position_ids=pos)
                for k in range(len(layers)):
                    h = model.model.norm(captured[k]).float()
                    logp = torch.log_softmax(h @ weight.T, dim=-1)
                    for r, i in enumerate(idx):
                        t = firsts[i]
                        rows.append(dict(item_id=prompts[i].item_id, condition=prompts[i].condition, layer=k,
                                         logprob=float(logp[r, t]), rank=int((logp[r] > logp[r, t]).sum())))
    finally:
        for h in handles:
            h.remove()
    return pd.DataFrame(rows)


def hint_spans(tokenizer, text, chat=True):
    """(token ids, hint-letter token indices, definition token indices) of the last block of a hinted prompt."""
    ids = scoring.encode(tokenizer, text, chat)
    full = (tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            if chat else tokenizer.convert_ids_to_tokens(ids[:1])[0] + text)
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc.input_ids) != list(ids):
        raise ValueError("re-tokenised prompt does not match the scored prompt")
    last = full.rfind("Description: ")
    d = re.compile(r"Description: ([^\n]+)").search(full, last)
    h = re.compile(r'starts with "([a-z]+)"').search(full, last)

    def tokens(lo, hi):
        return [i for i, (a, b) in enumerate(enc.offset_mapping) if a < hi and b > lo]

    return ids, tokens(*h.span(1)), tokens(*d.span(1))


def hint_attention(model, tokenizer, prompts, heads, chat=True, batch_size=4):
    """One row per prompt and head: attention mass from the answer position to the hint letters and to the
    definition. The caller sets eager attention (attention weights are only returned by it)."""
    import torch

    device = next(model.parameters()).device
    prepared = [hint_spans(tokenizer, p.text, chat) for p in prompts]
    rows = []
    with torch.no_grad():
        for idx in scoring._batches([x[0] for x in prepared], batch_size):
            seqs = [prepared[i][0] for i in idx]
            ids, mask, pos = scoring._pad_left(seqs, tokenizer.eos_token_id or 0, device)
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_attentions=True)
            width = ids.shape[1]
            for r, i in enumerate(idx):
                seq, hint, definition = prepared[i]
                shift = width - len(seq)
                for h in heads:
                    att = out.attentions[h.layer][r, h.index, -1].float()
                    rows.append(dict(item_id=prompts[i].item_id, condition=prompts[i].condition, head=h.key,
                                     to_hint=float(att[[shift + j for j in hint]].sum()),
                                     to_definition=float(att[[shift + j for j in definition]].sum())))
    return pd.DataFrame(rows)
