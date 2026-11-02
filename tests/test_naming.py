"""Definition-naming items and scoring (knowledge test, user decisions 2026-09-28)."""
import pandas as pd
import pytest
import torch

from lesion import knowledge_items as ki, models, naming

ITEMS = ki.OUT / "items.csv"


@pytest.fixture(scope="module")
def materials():
    return pd.read_csv(ITEMS), pd.read_csv(ki.OUT / "demos.csv")


def test_items_span_frequency_and_do_not_leak(materials):
    items, demos = materials
    assert len(items) == 600 and items.item_id.is_unique and items.target.is_unique
    bins = pd.cut(items.zipf, ki.ZIPF_BINS, right=False)
    assert (bins.value_counts() == 100).all()
    per_bin = items.groupby([bins, "split"], observed=True).size().unstack()
    assert (per_bin["screening"] == 40).all() and (per_bin["dev"] == 20).all() and (per_bin["test"] == 40).all()
    assert not set(demos.target) & set(items.target)
    assert not any(ki.leaks(r.definition, r.target) for r in items.itertuples())


@pytest.mark.parametrize("name,chat", [("olmoe-0924-base", False), ("qwen3-30b-a3b-instruct", True), ("olmoe-0924-instruct", True)])
def test_answer_encoding(name, chat, materials):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    tok = models.load_tokenizer(name)
    items, demos = materials
    p = naming.build_prompts(items.head(3), demos, chat=chat)[2]
    prompt_ids, answer_ids = naming.encode_pair(tok, p, chat)
    text = tok.decode(answer_ids)
    assert p.target in text and len(answer_ids) >= 2
    assert answer_ids[-1] in naming._end_ids(tok) or text.endswith("\n\n")


@pytest.mark.parametrize("name,chat", [("olmoe-0924-base", False), ("qwen3-30b-a3b-instruct", True)])
def test_logprob_matches_full_softmax(name, chat, materials):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    items, demos = materials
    prompts = naming.build_prompts(items.head(5), demos, chat=chat)
    got = naming.answer_logprobs(model, tok, prompts, chat, batch_size=2)
    for p, row in zip(prompts, got.itertuples()):
        prompt_ids, answer_ids = naming.encode_pair(tok, p, chat)
        ids = torch.tensor([prompt_ids + answer_ids])
        with torch.no_grad():
            logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
        n = len(answer_ids)
        want = sum(float(logp[len(prompt_ids) - 1 + j, answer_ids[j]]) for j in range(n))
        assert abs(row.logprob - want) < 1e-3 and row.n_tokens == n - 1


def test_leak_check_catches_short_word_forms():
    assert ki.leaks("a container; boxes are often made of cardboard", "box")
    assert not ki.leaks("a small rectangular container", "box")
    assert ki.leaks("seeds of the anteaters' favourite plant", "anteater")


def test_held_out_knowledge_items_are_locked():
    assert len(ki.load_items("screening")) == 240 and len(ki.load_items("dev")) == 120
    with pytest.raises(PermissionError):
        ki.load_items("test")
    with pytest.raises(PermissionError):
        ki.load_items()
