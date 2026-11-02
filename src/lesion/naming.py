"""Definition naming: prompts and answer log-probabilities (knowledge test; user decisions 2026-09-28).

The model reads a definition and should give the word. The score is the log-probability of the whole
answer (the word's tokens plus the end-of-answer marker), with the output layer in float32, so that a
lesion that weakens a word's knowledge lowers its score even when the word stays the top answer.
Rare words are split into more tokens, which lowers their score by itself; analyses keep n_tokens as a
covariate.
"""
from dataclasses import dataclass

import pandas as pd

from . import scoring

INSTRUCTION = "Name the single English word that each definition describes."
REPLY_RULE = " Reply with the word only, in lowercase."
BLOCK = "Definition: {definition}\nWord:"


@dataclass
class NamingPrompt:
    item_id: str
    target: str
    text: str


def build_prompts(items, demos, chat=False):
    demo_text = [BLOCK.format(definition=d.definition) + f" {d.target}" for d in demos.itertuples()]
    instruction = INSTRUCTION + (REPLY_RULE if chat else "")
    return [NamingPrompt(r.item_id, r.target, "\n\n".join([instruction, *demo_text, BLOCK.format(definition=r.definition)]))
            for r in items.itertuples()]


def _end_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>"):     # <|end|>: Phi-3's end of turn
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tokenizer.unk_token_id:
            ids.add(i)
    return ids - {None}


def encode_pair(tokenizer, prompt, chat):
    """(prompt ids, answer ids). Base models: " word" then the "\\n\\n" that starts the next block.
    Chat models: the assistant turn as the chat template writes it, up to and including its end token."""
    if not chat:
        p = scoring.encode(tokenizer, prompt.text, chat=False)
        a = tokenizer.encode(" " + prompt.target, add_special_tokens=False)
        return p, a + tokenizer.encode("\n\n", add_special_tokens=False)[:1]
    user = [{"role": "user", "content": prompt.text}]
    p = tokenizer.apply_chat_template(user, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": prompt.target}], tokenize=True)
    if list(full[:len(p)]) != list(p):
        raise ValueError("the chat template does not extend the generation prompt with the answer")
    a = list(full[len(p):])
    ends = _end_ids(tokenizer)
    cut = next((i for i, t in enumerate(a) if t in ends), None)
    return list(p), a[:cut + 1] if cut is not None else a


def answer_logprobs(model, tokenizer, prompts, chat, batch_size=16):
    """One row per prompt: summed float32 log-probability of the answer (word plus end marker), the number of
    tokens of the word itself, and the rank of the word's first token among all tokens at the answer position
    (0 = the model's top choice)."""
    import torch

    device = next(model.parameters()).device
    pairs = [encode_pair(tokenizer, p, chat) for p in prompts]
    seqs = [p + a for p, a in pairs]
    weight = model.get_output_embeddings().weight.float()
    out = [None] * len(prompts)
    with torch.no_grad():
        for idx in scoring._batches(seqs, batch_size):
            ids, mask, pos = scoring._pad_left([seqs[i] for i in idx], tokenizer.eos_token_id or 0, device)
            hidden = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
            width = ids.shape[1]
            for row, i in enumerate(idx):
                n = len(pairs[i][1])
                h = hidden[row, width - n - 1:width - 1].float()          # positions that predict each answer token
                logp = torch.log_softmax(h @ weight.T, dim=-1)
                answer = ids[row, width - n:]
                first = logp[0]
                out[i] = dict(item_id=prompts[i].item_id, target=prompts[i].target,
                              logprob=float(logp.gather(1, answer[:, None]).sum()), n_tokens=n - 1,
                              first_token_rank=int((first > first[answer[0]]).sum()))
    return pd.DataFrame(out)


# --- does the test work? (user rule 2026-09-28: items go to human review only after they show the effect) ---

MIN_TOP1_RARE = 0.20   # rarest Zipf bin: at least this share of answers must be the model's first choice


def frequency_check(scores, items, bins, n_boot=2000, seed=0):
    """Whether the intact model shows the frequency effect the knowledge test relies on.

    slope: change in answer log-probability per Zipf unit, with the word's token count as covariate
      (rare words are split into more tokens); interval from resampling items. Must lie above 0.
    rare_top1: share of the rarest bin whose first answer token is the model's top choice; must reach
      MIN_TOP1_RARE, so rare words are not simply unknown (floor).
    """
    import numpy as np

    t = scores.merge(items[["item_id", "zipf"]], on="item_id")
    X = np.column_stack([np.ones(len(t)), t.zipf, t.n_tokens])
    y = t.logprob.to_numpy()
    fit = lambda idx: np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][1]
    rng = np.random.default_rng(seed)
    boot = np.array([fit(rng.integers(len(t), size=len(t))) for _ in range(n_boot)])
    slope = dict(estimate=float(fit(np.arange(len(t)))), ci_low=float(np.percentile(boot, 2.5)),
                 ci_high=float(np.percentile(boot, 97.5)))
    t["bin"] = pd.cut(t.zipf, bins, right=False)
    by_bin = t.groupby("bin", observed=True).agg(n=("item_id", "size"), mean_logprob=("logprob", "mean"),
                                                 top1=("first_token_rank", lambda r: float((r == 0).mean())))
    rare_top1 = float(by_bin.top1.iloc[0])
    return dict(slope=slope, rare_top1=rare_top1, by_bin=by_bin.reset_index().astype({"bin": str}).to_dict("records"),
                passed=bool(slope["ci_low"] > 0 and rare_top1 >= MIN_TOP1_RARE))


# --- lesion damage on the naming test (knowledge side; user decisions 2026-09-28) --------------------------

NAMING_EFFECTS = ("knowledge_effect", "knowledge_slope", "naming_damage")


def zipf_median(items):
    """Median Zipf of all knowledge items; fixed, so every split uses the same rare/common boundary."""
    return float(items.zipf.median())


def item_damage(intact, lesioned):
    """h per item = intact answer log-probability - lesioned (larger = more harm); both tables cover the same items."""
    a, b = intact.set_index("item_id").logprob, lesioned.set_index("item_id").logprob
    if set(a.index) != set(b.index):
        raise ValueError("intact and lesioned scores cover different items")
    return (a - b.reindex(a.index)).rename("h")


def item_frame(h, items, median):
    """One row per item with its damage column(s), Zipf, token count and rare flag; each item is its own group."""
    h = h.to_frame() if isinstance(h, pd.Series) else h
    meta = items.set_index("item_id")[["zipf"]]
    t = h.join(meta, how="left")
    if t.zipf.isna().any():
        raise ValueError("damage given for items not in the item table")
    t["rare"] = t.zipf < median
    t = t.reset_index()
    t["concept_group"] = t.item_id
    return t


def effects(cols, h="h"):
    """knowledge_effect: mean h on the rarer half minus the more common half; knowledge_slope: h regressed on
    -Zipf (standardised; positive = rare words hurt more); naming_damage: mean h."""
    import numpy as np

    y, rare = cols[h], cols["rare"].astype(bool)
    z = cols["zipf"]
    slope = np.nan
    if len(y) >= 4 and np.std(z) > 0:
        X = np.column_stack([np.ones(len(y)), -(z - z.mean()) / z.std()])
        slope = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    return dict(knowledge_effect=float(y[rare].mean() - y[~rare].mean()) if rare.any() and (~rare).any() else np.nan,
                knowledge_slope=slope, naming_damage=float(y.mean()) if len(y) else np.nan)


def statistic_weights(items_in_order, median):
    """{statistic: per-item weights v}: under a lesion, S = sum_i v_i logprob_i changes by that statistic of h."""
    import numpy as np

    rare = (items_in_order.zipf < median).to_numpy()
    n = len(rare)
    ke = np.where(rare, 1.0 / max(rare.sum(), 1), -1.0 / max((~rare).sum(), 1))
    return {"knowledge_effect": -ke, "naming_damage": -np.full(n, 1.0 / n)}


def averaged_logprobs(model, tokenizer, prompts, chat, batch_sizes=(16, 7, 3)):
    """answer_logprobs averaged over several batch compositions (Qwen's scores move with the composition; ADR 0004)."""
    import numpy as np

    tables = [answer_logprobs(model, tokenizer, prompts, chat, bs) for bs in batch_sizes]
    out = tables[0].copy()
    out["logprob"] = np.mean([t.logprob.to_numpy() for t in tables], axis=0)
    return out
