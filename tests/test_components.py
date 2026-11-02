"""Component names and enumeration (docs/plans/code-architecture.md §3.1)."""
from types import SimpleNamespace

import pytest

from lesion.components import Component, all_components, moe_layers, n_experts


def test_key_round_trip_and_order():
    c = Component("expert", 3, 17)
    assert c.key == "expert:3:17" and Component.parse(c.key) == c
    assert sorted([Component("expert", 1, 0), Component("expert", 0, 5)])[0] == Component("expert", 0, 5)


@pytest.mark.parametrize("bad", [("neuron", 0, 0), ("expert", -1, 0), ("mlp", 0, 1)])
def test_invalid(bad):
    with pytest.raises(ValueError):
        Component(*bad)


def test_olmoe_sizes():
    cfg = SimpleNamespace(num_hidden_layers=16, num_experts=64, num_attention_heads=16)
    assert len(all_components(cfg, "expert")) == 1024
    assert len(all_components(cfg, "head")) == 256
    assert all_components(cfg, "mlp") == []


def test_qwen_dense_layers_and_dense_model():
    qwen = SimpleNamespace(num_hidden_layers=4, num_experts=8, num_attention_heads=2, mlp_only_layers=[1], decoder_sparse_step=1)
    assert moe_layers(qwen) == [0, 2, 3]
    assert [c.layer for c in all_components(qwen, "mlp")] == [1]
    dense = SimpleNamespace(num_hidden_layers=3, num_attention_heads=2)
    assert all_components(dense, "expert") == [] and len(all_components(dense, "mlp")) == 3


def test_phi_style_config_counts_local_experts():
    phi = SimpleNamespace(num_hidden_layers=3, num_local_experts=16, num_attention_heads=2)
    assert n_experts(phi) == 16 and moe_layers(phi) == [0, 1, 2]
    assert len(all_components(phi, "expert")) == 48 and all_components(phi, "mlp") == []
