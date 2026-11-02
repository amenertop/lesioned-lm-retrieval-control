"""Option spans and head attention read-out (control-head exploration)."""
import numpy as np
import pytest

from lesion import attention, models, scoring
from lesion.components import Component


@pytest.mark.parametrize("name,chat", [("olmoe-0924-base", False), ("qwen3-30b-a3b-instruct", True)])
def test_option_spans_cover_the_option_words(name, chat):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    tok = models.load_tokenizer(name)
    items, demos, templates = scoring.load_materials()
    for p in scoring.build_label_prompts(items[items.task == "colour"].head(3), demos, templates, chat=chat):
        ids, spans = attention.option_spans(tok, p, chat)
        words = {lab: tok.decode([ids[j] for j in spans[lab]]).strip() for lab in "AB"}
        row = items.set_index("item_id").loc[p.item_id]
        assert words[p.correct_label] == row[p.label_to_option[p.correct_label]]
        assert max(spans["A"] + spans["B"]) < len(ids)


def test_head_attention_rows():
    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    model.config._attn_implementation = "eager"          # attention weights are only returned by eager attention
    items, demos, templates = scoring.load_materials()
    prompts = scoring.build_label_prompts(items[items.task == "size"].head(2), demos, templates, chat=True)
    out = attention.head_attention(model, tok, prompts, True, [Component("head", 1, 2), Component("head", 0, 0)], batch_size=2)
    assert len(out) == len(prompts) * 2
    assert ((out.to_correct >= 0) & (out.to_correct <= 1) & (out.to_wrong >= 0)).all()


@pytest.mark.parametrize("name,chat", [("olmoe-0924-base", False), ("qwen3-30b-a3b-instruct", True)])
def test_regions_partition_the_prompt(name, chat):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    tok = models.load_tokenizer(name)
    items, demos, templates = scoring.load_materials()
    p = scoring.build_label_prompts(items[items.task == "size"].head(1), demos, templates, chat=chat)[0]
    ids, reg = attention.regions(tok, p, chat)
    flat = sorted(j for toks in reg.values() for j in toks)
    assert flat == list(range(len(ids)))                       # every token in exactly one region
    assert reg["start"] == [0] and reg["instruction"] and reg["demos"] and reg["probe_line"] and reg["cue"]
    assert tok.decode([ids[j] for j in reg["probe_line"]]).strip().startswith("Word:")


def test_attention_profile_shares_sum_to_one():
    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    model.config._attn_implementation = "eager"
    items, demos, templates = scoring.load_materials()
    prompts = scoring.build_label_prompts(items[items.task == "colour"].head(2), demos, templates, chat=True)
    out = attention.attention_profile(model, tok, prompts, True, [Component("head", 1, 2)], batch_size=2)
    totals = out.groupby(["item_id", "order", "head"]).share.sum()
    assert np.allclose(totals, 1.0, atol=1e-4)
