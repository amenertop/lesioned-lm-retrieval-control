"""Prefix-shared scoring gives the same logits as full scoring (ADR 0003), with and without ablations."""
import pytest
import torch

from lesion import hooks, means as means_mod, models, neutral_text, prefix_scoring, scoring
from lesion.components import Component

CASES = [("olmoe-0924-base", False), ("qwen3-30b-a3b-instruct", True), ("olmo2-1b", False)]


@pytest.fixture(scope="module")
def materials():
    items, demos, templates = scoring.load_materials()
    pick = items.groupby("task").head(5)          # 20 items, 40 prompts, four tasks and both orders
    return pick, demos, templates


@pytest.mark.parametrize("name,chat", CASES)
def test_same_logits_as_full_scoring(name, chat, materials):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    items, demos, templates = materials
    prompts = scoring.build_label_prompts(items, demos, templates, chat=chat)
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)
    if getattr(model.config, "num_experts", 0):
        lesion = [Component("expert", 0, 1), Component("expert", 1, 3), Component("head", 1, 2)]
    else:
        lesion = [Component("mlp", 0, 0), Component("head", 1, 2)]
    contexts = {"intact": None, "mean": dict(means=m, mode="mean"), "zero": dict(means=None, mode="zero")}
    for label, kw in contexts.items():
        def run(fn, bs):
            if kw is None:
                return fn(model, tok, prompts, chat=chat, batch_size=bs)
            with hooks.ablate(model, lesion, alpha=0.0, **kw):
                return fn(model, tok, prompts, chat=chat, batch_size=bs)
        full = run(scoring.score_label_prompts, 8)
        fast = run(prefix_scoring.score_label_prompts, 7)
        full32 = run(prefix_scoring.score_label_prompts_full, 5)
        assert (torch.tensor(full[["logit_A", "logit_B"]].to_numpy())
                - torch.tensor(full32[["logit_A", "logit_B"]].to_numpy())).abs().max().item() < 1e-4
        cols = ["logit_A", "logit_B"]
        assert (full.item_id.tolist(), full.order.tolist()) == (fast.item_id.tolist(), fast.order.tolist())
        diff = (torch.tensor(full[cols].to_numpy()) - torch.tensor(fast[cols].to_numpy())).abs().max().item()
        assert diff < 1e-4, (label, diff)
        same = ["item_id", "task", "order", "correct_label", "option_A"]
        assert full[same].equals(fast[same])
        clear = (full.logit_A - full.logit_B).abs() > 1e-3        # choices agree wherever they are not ties
        assert (full.choice[clear] == fast.choice[clear]).all()


def test_split_prefixes_keeps_an_own_token():
    seqs = [[1, 2, 3, 4], [1, 2, 3, 5, 6], [1, 2, 3]]
    groups = prefix_scoring.split_prefixes(seqs, ["a", "a", "a"])
    prefix, own = groups["a"]
    assert prefix == [1, 2] and [o for _, o in own] == [[3, 4], [3, 5, 6], [3]]
