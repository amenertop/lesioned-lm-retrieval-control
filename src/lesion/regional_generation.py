"""Greedy generation with the same answer/prompt positions as m3_battery's candidate scorer.

The library still handles generation, caches and stopping. A temporary pre-hook only updates the lesion mask
for each prefill or cached forward. The final prompt token predicts the first answer token.
"""
from contextlib import ExitStack, contextmanager

import pandas as pd
import torch

from . import hooks, m3_battery, scoring


@contextmanager
def generation_region(model, prompt_width, region):
    if region not in ("answer", "prompt", "all"):
        raise ValueError("generation region must be answer, prompt or all")
    with ExitStack() as positions:
        def before_forward(module, args, kwargs):
            ids, attention = kwargs["input_ids"], kwargs["attention_mask"]
            width = ids.shape[1]
            absolute = torch.arange(attention.shape[1] - width, attention.shape[1], device=ids.device)
            mask = attention[:, -width:].bool()
            if region == "answer":
                mask = mask & (absolute >= prompt_width - 1)
            elif region == "prompt":
                mask = mask & (absolute < prompt_width - 1)
            positions.close()
            positions.enter_context(hooks.position_mask(model, mask))

        handle = model.register_forward_pre_hook(before_forward, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()


def greedy_answers(model, tokenizer, prompts, chat=True, batch_size=16, max_new_tokens=8, region=None):
    if region is None:
        return m3_battery.greedy_answers(model, tokenizer, prompts, chat, batch_size, max_new_tokens)
    device = next(model.parameters()).device
    seqs = [scoring.encode(tokenizer, p.text, chat) for p in prompts]
    pad = tokenizer.eos_token_id or 0
    out = [None] * len(prompts)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, _ = scoring._pad_left([seqs[i] for i in idx], pad, device)
            with generation_region(model, ids.shape[1], region):
                gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=pad, use_cache=True)
            for r, i in enumerate(idx):
                out[i] = dict(item_id=prompts[i].item_id,
                              answer=tokenizer.decode(gen[r, ids.shape[1]:], skip_special_tokens=True))
    return pd.DataFrame(out)
