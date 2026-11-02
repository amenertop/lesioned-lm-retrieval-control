"""Pinned intact-model census. Coder uses the existing loader and lesion pipeline after passing G1.

For the original Qwen3 checkpoints, the official template's enable_thinking=False flag is explicit. The
Thinking-only model keeps its native thought process and gets a format precheck, not a forced label score.
"""
import json
import re
from pathlib import Path

import pandas as pd

from . import g1_revised, models, scoring

CANDIDATES = {
    "qwen3-coder-30b-a3b-instruct": dict(repo=models.MODELS["qwen3-coder-30b-a3b-instruct"][0],
                                         revision=models.MODELS["qwen3-coder-30b-a3b-instruct"][1], mode="direct", replication=True),
    "qwen3-30b-a3b": dict(repo="Qwen/Qwen3-30B-A3B", revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", mode="nonthinking", replication=False),
    "qwen3-32b": dict(repo="Qwen/Qwen3-32B", revision="9216db5781bf21249d130ec9da846c4624c16137", mode="nonthinking", replication=False),
    "qwen3-30b-a3b-thinking-2507": dict(repo="Qwen/Qwen3-30B-A3B-Thinking-2507", revision="144afc2f379b542fdd4e85a1fcd5e1f79112d95d", mode="native_thinking", replication=False),
}


def verified_candidate_path(name, directory=None):
    """Census-only checkpoints use the same fixed-revision verification as registered lesion models."""
    path = Path(directory) if directory else models.VERIFIED_DIR / name
    record = json.loads((path / "VERIFIED.json").read_text(encoding="utf-8"))
    if not record.get("all_safetensors_match_hf") or record.get("hf_revision") != CANDIDATES[name]["revision"]:
        raise ValueError("census weights do not match the fixed candidate revision")
    return path


class NonThinkingTokenizer:
    """Delegate everything except the one documented template flag, leaving the frozen scorer unchanged."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **{**kwargs, "enable_thinking": False})


def format_answer(text, native_thinking=False):
    if native_thinking:
        if "</think>" not in text:
            return None
        text = text.rsplit("</think>", 1)[1]
    return text.strip() if re.fullmatch(r"\s*[AB]\s*", text) else None


def demo_items(demos):
    """The same 24 demonstration items and two orders as the existing P12 format precheck."""
    rows = []
    for d in demos.to_dict("records"):
        syn = d["task"] == "synonym"
        rows.append(dict(item_id=f"demo_{d['task']}_{d['position']}", task=d["task"], probe=d["probe"],
                         option1="similar" if syn else d["option_a"], option2="different" if syn else d["option_b"],
                         correct_option="option1" if d["correct_label"] == "A" else "option2",
                         word1=d["word1"], word2=d["word2"]))
    return pd.DataFrame(rows)


def format_check(model, tok, prompts, native_thinking=False, max_new_tokens=16):
    import torch
    rows = []
    device = next(model.parameters()).device
    label_ids = set(scoring.label_token_ids(tok, True).values())
    for p in prompts:
        ids = torch.tensor([scoring.encode(tok, p.text, True)], device=device)
        with torch.no_grad():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                       max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id or 0)
        continuation = generated[0, ids.shape[1]:].tolist()
        text = tok.decode(continuation, skip_special_tokens=True)
        first_is_label = bool(continuation) and continuation[0] in label_ids
        rows.append(dict(item_id=p.item_id, order=p.order, generated=text,
                         answer=format_answer(text, native_thinking), first_is_label=first_is_label,
                         completion_tokens=len(continuation)))
    table = pd.DataFrame(rows)
    share = table.answer.notna().mean() if native_thinking else table.first_is_label.mean()
    return table, dict(n=len(table), valid_fraction=float(share),
                      pass_format=bool(share >= 0.9), native_thinking=native_thinking,
                      rule="complete final label after native thought" if native_thinking else "P12 first greedy token is A or B",
                      token_budget=max_new_tokens)


def measurement(model, tok, prompts, items, batch_size=16, n_boot=10000):
    scores = scoring.score_label_prompts(model, tok, prompts, True, batch_size)
    return scores, g1_revised.analyze({"candidate": scores}, items, n=n_boot)
