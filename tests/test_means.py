"""Focused smoke tests for pilot 09; the cross-architecture contract is in test_contract_hooks_means."""

import numpy as np
import pytest

from lesion import models, neutral_text

means_mod = pytest.importorskip("lesion.means")


@pytest.fixture(scope="module")
def collected():
    if not models.local_dir("olmoe-0924-base").exists():
        pytest.skip("tokenizer files not downloaded")
    model, tokenizer = models.load_tiny("olmoe-0924-base")
    texts = neutral_text.load()[:3]
    return means_mod.collect_means(model, tokenizer, texts, max_tokens=24, batch_size=2)


def test_means_are_reproducible_across_batching(collected):
    assert collected.n_tokens > 0
    model, tokenizer = models.load_tiny("olmoe-0924-base")
    texts = neutral_text.load()[:3]
    one = means_mod.collect_means(model, tokenizer, texts, max_tokens=24, batch_size=1)
    assert one.n_tokens == collected.n_tokens
    assert np.array_equal(one.expert_count.numpy(), collected.expert_count.numpy())
    assert np.allclose(one.expert_mean.numpy(), collected.expert_mean.numpy(), atol=1e-5)
    assert np.allclose(one.head_mean.numpy(), collected.head_mean.numpy(), atol=1e-5)


def test_means_round_trip(collected, tmp_path):
    path = tmp_path / "means.pt"
    collected.save(path)
    restored = means_mod.Means.load(path)
    assert restored.meta == collected.meta
    assert restored.n_tokens == collected.n_tokens
    assert np.array_equal(restored.expert_count.numpy(), collected.expert_count.numpy())
