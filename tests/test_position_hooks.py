"""Position-restricted lesions on random models, independent of downloaded tokenizers."""
from types import SimpleNamespace

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM, PhimoeConfig, PhimoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from lesion import hooks
from lesion.components import Component, moe_block


@pytest.fixture(scope="module", params=["qwen", "olmoe", "phi"])
def tiny(request):
    torch.manual_seed(319)
    config = dict(vocab_size=43, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
                  max_position_embeddings=64, attention_dropout=0.0, use_cache=False)
    if request.param == "qwen":
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config, head_dim=8, moe_intermediate_size=24))
    elif request.param == "phi":
        config["num_local_experts"] = config.pop("num_experts")
        model = PhimoeForCausalLM(PhimoeConfig(**config, rope_scaling=None))
    else:
        model = OlmoeForCausalLM(OlmoeConfig(**config))
    model.eval()
    model.config._attn_implementation = "eager"
    means = SimpleNamespace(expert_mean=torch.randn(2, 4, 32) * .03,
                            head_mean=torch.randn(2, 4, 8) * .03)
    return model, means


def batch(padded):
    ids = torch.tensor([[2, 7, 3, 9, 5], [6, 4, 8, 2, 1]])
    attention = torch.ones_like(ids)
    if padded:
        ids[1, :2] = 0
        attention[1, :2] = 0
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    return dict(input_ids=ids, attention_mask=attention, position_ids=positions)


def logits(model, inputs):
    with torch.no_grad():
        return model(**inputs).logits


def components(kind):
    # All experts ensures the chosen position uses an ablated expert.
    return ([Component("expert", 0, e) for e in range(4)] if kind == "expert"
            else [Component("head", 0, 1)])


@pytest.mark.parametrize("kind", ["expert", "head"])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("mode", ["mean", "zero"])
def test_all_and_empty_masks_match_existing_behaviour(tiny, kind, padded, mode):
    model, means = tiny
    inputs = batch(padded)
    base = logits(model, inputs)
    with hooks.ablate(model, components(kind), means, mode=mode):
        plain = logits(model, inputs)
        with hooks.position_mask(model, torch.ones_like(inputs["input_ids"], dtype=torch.bool)):
            full = logits(model, inputs)
        with hooks.position_mask(model, torch.zeros_like(inputs["input_ids"], dtype=torch.bool)):
            empty = logits(model, inputs)
        restored = logits(model, inputs)
    assert torch.equal(full, plain)
    assert torch.equal(empty, base)
    assert torch.equal(restored, plain)
    assert torch.equal(logits(model, inputs), base)
    assert model._lesion_position_mask is None


@pytest.mark.parametrize("kind", ["expert", "head"])
@pytest.mark.parametrize("layer", [0, 1])
def test_partial_mask_preserves_causal_prefix_and_other_batch_row(tiny, kind, layer):
    model, means = tiny
    inputs = batch(True)
    selected = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
    selected[1, 3] = True
    base = logits(model, inputs)
    comps = [Component(c.kind, layer, c.index) for c in components(kind)]
    with hooks.ablate(model, comps, means), hooks.position_mask(model, selected):
        changed = logits(model, inputs)
    # A downstream MoE can regroup unchanged earlier tokens into different expert
    # GEMM batches. OLMoE then differs by <=3e-8 despite identical causal inputs.
    if layer == 1:
        assert torch.equal(base[0], changed[0])
        assert torch.equal(base[1, 2:3], changed[1, 2:3])
    else:
        torch.testing.assert_close(base[0], changed[0], rtol=0, atol=5e-8)
        torch.testing.assert_close(base[1, 2:3], changed[1, 2:3], rtol=0, atol=5e-8)
    assert not torch.equal(base[1, 3], changed[1, 3])


def test_expert_rows_follow_topk_rank_order_not_sorted_token_order(tiny):
    model, means = tiny
    block = moe_block(model.model.layers[0])
    x = torch.randn(2, 4, 32)
    # Expert 0 gets rank-0 tokens 1, 3, 5, 7, then rank-1 tokens 0, 2, 4, 6.
    forced = torch.tensor([[3., 4., -4., -5.], [4., 3., -4., -5.]]).repeat(4, 1)
    selected = torch.zeros(2, 4, dtype=torch.bool)
    selected[0, 0] = True
    selected[1, 3] = True
    before, after = [], []
    gate = block.gate.register_forward_hook(lambda _m, _i, _o: forced)
    first = block.experts[0].register_forward_hook(lambda _m, _i, out: before.append(out.detach().clone()))
    try:
        with hooks.ablate(model, [Component("expert", 0, 0)], means, alpha=.4):
            last = block.experts[0].register_forward_hook(lambda _m, _i, out: after.append(out.detach().clone()))
            try:
                with torch.no_grad(), hooks.position_mask(model, selected):
                    block(x)
            finally:
                last.remove()
    finally:
        first.remove()
        gate.remove()
    row_mask = selected.reshape(-1)[torch.tensor([1, 3, 5, 7, 0, 2, 4, 6])]
    expected = before[0].clone()
    expected[row_mask] = means.expert_mean[0, 0] + .4 * (expected[row_mask] - means.expert_mean[0, 0])
    assert torch.equal(after[0], expected)


def test_phi_routing_matches_sparsemixer_on_exact_ties(tiny):
    model, _ = tiny
    block = moe_block(model.model.layers[0])
    if not hasattr(block, "router_jitter_noise"):
        pytest.skip("Phi only")
    from transformers.models.phimoe.modeling_phimoe import sparsemixer
    tied = torch.tensor([[2., 2., 2., -5.], [1., 3., 3., 3.], [0., 0., 0., 0.], [4., -1., 4., 4.]])
    random = torch.randn(64, 4).round()  # rounding to integers makes many exact ties
    for router_logits in (tied, random):
        expected = sparsemixer(router_logits, jitter_eps=block.router_jitter_noise, training=False)[1]
        assert torch.equal(hooks._routed_experts(block, router_logits, 2), expected)


def test_mask_validation_and_restoration_after_nested_exception(tiny):
    model, means = tiny
    inputs = batch(False)
    outer = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(TypeError, match="bool"):
        with hooks.position_mask(model, torch.ones(2, 5)):
            pass
    with pytest.raises(ValueError, match="batch.*seq"):
        with hooks.position_mask(model, torch.ones(10, dtype=torch.bool)):
            pass
    with hooks.position_mask(model, outer):
        with pytest.raises(RuntimeError, match="sentinel"):
            with hooks.position_mask(model, ~outer):
                raise RuntimeError("sentinel")
        assert model._lesion_position_mask is outer
    assert model._lesion_position_mask is None
    base = logits(model, inputs)
    for kind in ("expert", "head"):
        with pytest.raises(ValueError, match="shape"):
            with hooks.ablate(model, components(kind), means), hooks.position_mask(model, outer.reshape(1, 10)):
                logits(model, inputs)
        assert model._lesion_position_mask is None
        assert not model._lesion_ablate_active
    # A failing forward must not leave stale expert routing maps behind.
    with hooks.ablate(model, components("expert"), means), hooks.position_mask(model, ~outer):
        assert torch.equal(logits(model, inputs), base)
