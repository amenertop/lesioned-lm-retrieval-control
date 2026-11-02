"""lesion.gates on tiny random Qwen3-MoE and Phi-MoE models: g = 0 is lesion.hooks.ablate's mean ablation, g = 1 the
intact model, any g matches hooks.ablate at alpha = g, components are gated independently, the gradients reach the
gates and agree with finite differences, and no hook is left behind."""
from types import SimpleNamespace

import pytest
import torch
from transformers import PhimoeConfig, PhimoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from lesion import gates, hooks
from lesion.components import Component

IDS = torch.tensor([[2, 7, 3, 9, 5, 11, 4], [6, 4, 8, 1, 3, 5, 7]])
COMPS = [Component("head", 0, 1), Component("head", 1, 3), Component("expert", 0, 2), Component("expert", 1, 0)]


@pytest.fixture(scope="module", params=["qwen", "phi"])
def tiny(request):
    torch.manual_seed(5)
    config = dict(vocab_size=43, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
                  attention_dropout=0.0, use_cache=False)
    if request.param == "qwen":
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config, head_dim=8, moe_intermediate_size=24))
    else:
        config["num_local_experts"] = config.pop("num_experts")
        model = PhimoeForCausalLM(PhimoeConfig(**config, rope_scaling=None))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    means = SimpleNamespace(expert_mean=torch.randn(2, 4, 32) * .3, head_mean=torch.randn(2, 4, head_dim) * .3)
    return model, means


def logits(model, ctx=None):
    with torch.no_grad():
        if ctx is None:
            return model(input_ids=IDS).logits
        with ctx:
            return model(input_ids=IDS).logits


def test_zero_one_and_alpha_match_hooks_ablate(tiny):
    model, means = tiny
    g = gates.Gates(COMPS)
    intact = logits(model)
    for alpha in (0.0, 0.3, 1.0):
        g.set_values([alpha] * len(COMPS))
        mine = logits(model, gates.gated(model, g, means))
        ref = logits(model, hooks.ablate(model, COMPS, means, alpha=alpha))
        assert torch.allclose(mine, ref, atol=1e-5), alpha
    assert torch.allclose(logits(model, gates.gated(model, g, means)), intact, atol=1e-5)    # g = 1 left last


def test_components_are_gated_independently(tiny):
    model, means = tiny
    g = gates.Gates(COMPS)
    for k, c in enumerate(COMPS):
        values = [1.0] * len(COMPS)
        values[k] = 0.0
        g.set_values(values)
        mine = logits(model, gates.gated(model, g, means))
        assert torch.allclose(mine, logits(model, hooks.ablate(model, [c], means)), atol=1e-5), c.key


def test_gradients_reach_the_gates_and_match_finite_differences(tiny):
    model, means = tiny
    g = gates.Gates(COMPS, init=0.4)
    with gates.gated(model, g, means):
        loss = model(input_ids=IDS).logits.float().pow(2).mean()
    loss.backward()
    grad = g.logits.grad.clone()
    assert torch.isfinite(grad).all() and (grad.abs() > 0).all()
    eps = 1e-3
    for k in range(len(COMPS)):
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                g.logits[k] += sign * eps
                with gates.gated(model, g, means):
                    values.append(model(input_ids=IDS).logits.float().pow(2).mean().item())
                g.logits[k] -= sign * eps
        assert abs((values[0] - values[1]) / (2 * eps) - grad[k].item()) < 1e-3 + 1e-2 * abs(grad[k].item())


def test_no_hook_is_left_behind_even_after_an_error(tiny):
    model, means = tiny
    g = gates.Gates(COMPS)
    g.set_values([0.0] * len(COMPS))
    intact = logits(model)
    with pytest.raises(RuntimeError):
        with gates.gated(model, g, means):
            raise RuntimeError("stop")
    assert torch.allclose(logits(model), intact, atol=0)


def test_gates_reject_bad_components():
    with pytest.raises(ValueError):
        gates.Gates([Component("head", 0, 1), Component("head", 0, 1)])
    with pytest.raises(ValueError):
        gates.Gates([Component("mlp", 0, 0)])
