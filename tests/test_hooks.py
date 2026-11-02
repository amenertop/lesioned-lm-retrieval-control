"""Focused smoke tests for pilot 03; full architecture coverage is contractual."""

import pytest
import torch

from lesion import models, neutral_text
from lesion.components import Component

hooks = pytest.importorskip("lesion.hooks")
means_mod = pytest.importorskip("lesion.means")


@pytest.fixture(scope="module")
def model_and_means():
    if not models.local_dir("olmoe-0924-base").exists():
        pytest.skip("tokenizer files not downloaded")
    model, tokenizer = models.load_tiny("olmoe-0924-base")
    means = means_mod.collect_means(model, tokenizer, neutral_text.load()[:2], max_tokens=20, batch_size=2)
    return model, tokenizer, means


def test_context_restores_and_zero_mode_needs_no_means(model_and_means):
    model, tokenizer, means = model_and_means
    ids = torch.tensor([tokenizer.encode(neutral_text.load()[0], add_special_tokens=False)[:32]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        base = model(input_ids=ids, attention_mask=mask).logits
    # Ablate the whole first-layer expert set so the test cannot randomly pick
    # an expert that receives no token in this particular input.
    expert = [Component("expert", 0, i) for i in range(means.expert_count.shape[1])]
    with hooks.ablate(model, expert, means, alpha=0.0):
        with torch.no_grad():
            damaged = model(input_ids=ids, attention_mask=mask).logits
    assert not torch.equal(base, damaged)
    with hooks.ablate(model, expert, None, alpha=0.0, mode="zero"):
        pass
    with torch.no_grad():
        restored = model(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(base, restored)


def test_router_mask_conflicts_with_ablation(model_and_means):
    model, _tokenizer, means = model_and_means
    expert = [Component("expert", 0, i) for i in range(means.expert_count.shape[1])]
    with hooks.mask_router(model, expert):
        with pytest.raises(RuntimeError):
            with hooks.ablate(model, expert, means):
                pass
