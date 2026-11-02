"""Coarse lesions must change only the requested weights and leave the loaded model exactly restorable."""
from types import SimpleNamespace

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from lesion import fast_moe, title_gap_states as states, weight_masks as wm


@pytest.fixture(params=["qwen", "olmoe"])
def model(request):
    torch.manual_seed(4)
    config = dict(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
                  num_experts=4, num_experts_per_tok=2, attention_dropout=0.0, use_cache=False)
    if request.param == "qwen":
        m = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config, head_dim=8, moe_intermediate_size=16))
    else:
        m = OlmoeForCausalLM(OlmoeConfig(**config))
    return m.eval().requires_grad_(False)


def forward(model):
    with torch.no_grad():
        return model(torch.tensor([[1, 4, 7, 2, 8]])).logits


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_zero_dose_and_restore_are_bit_identical(model):
    before, logits = snapshot(model), forward(model)
    for family in wm.FAMILIES:
        with wm.masked(model, 0, family, 0, 19):
            assert torch.equal(forward(model), logits)
        with wm.masked(model, 0, family, 1, 19):
            assert not torch.equal(forward(model), logits)
        assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
        assert torch.equal(forward(model), logits)


def test_all_experts_and_only_selected_layer_changed(model):
    before = snapshot(model)
    targets = wm.matrices(model, 0, "ffn")
    assert len(targets) == 3 * model.config.num_experts
    selected = {id(p) for _, p in targets}
    with wm.masked(model, 0, "ffn", 1, 3):
        for name, p in model.named_parameters():
            assert torch.count_nonzero(p) == 0 if id(p) in selected else torch.equal(p, before[name])


def test_masks_are_nested_reproducible_and_do_not_rescale(model):
    original = dict(wm.matrices(model, 0, "ffn"))
    originals = {n: p.clone() for n, p in original.items()}
    rng_state = torch.get_rng_state().clone()
    seen = []
    for severity, seed in [(0.25, 7), (0.25, 7), (0.75, 7), (0.25, 8)]:
        with wm.masked(model, 0, "ffn", severity, seed):
            seen.append({n: p.clone() for n, p in original.items()})
            for n, p in original.items():
                assert torch.all((p == 0) | (p == originals[n]))
                assert 0 < torch.count_nonzero(p) < p.numel()
    assert torch.equal(rng_state, torch.get_rng_state())
    assert all(torch.equal(seen[0][n], seen[1][n]) for n in original)
    assert all(torch.all((seen[2][n] == 0) | (seen[0][n] != 0)) for n in original)
    assert any(not torch.equal(seen[0][n], seen[3][n]) for n in original)
    first, second = list(seen[0].values())[:2]
    assert not torch.equal(first != 0, second != 0)
    with wm.masked(model, 0, "up", 0.25, 7):
        assert all(torch.equal(p, seen[0][n]) for n, p in wm.matrices(model, 0, "up"))


def test_exception_restores_in_place_with_grouped_weight_views(model):
    fast_moe.install(model, "grouped")
    before, logits = snapshot(model), forward(model)
    identities = {n: (id(p), p.data_ptr()) for n, p in model.named_parameters()}
    with pytest.raises(RuntimeError, match="scoring failed"):
        with wm.masked(model, 1, "ffn", 0.5, 9):
            assert not torch.equal(forward(model), logits)
            raise RuntimeError("scoring failed")
    assert torch.equal(forward(model), logits)
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]) and (id(p), p.data_ptr()) == identities[n]
    fast_moe.uninstall(model)


@pytest.mark.parametrize("changes", [dict(layer=-1), dict(family="router"), dict(severity=float("nan")),
                                     dict(severity=1.01), dict(seed=-1)])
def test_bad_requests_fail_before_mutation(model, changes):
    args = dict(layer=0, family="ffn", severity=0.5, seed=7)
    args.update(changes)
    before = snapshot(model)
    with pytest.raises(ValueError):
        with wm.masked(model, **args):
            pytest.fail("invalid request entered")
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_grid_and_common_state_context(model):
    full = wm.grid(SimpleNamespace(num_hidden_layers=48))
    assert len(full) == 84 and len([s for s in full if s["weight_mask"]["severity"] == 1]) == 12
    plan = [dict(name="intact", kind="intact", gates={})] + wm.grid(model.config, layers=(0,))
    states.validate_states(plan, model.config)
    before = snapshot(model)
    with states.apply_state(model, None, plan[1]):
        assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    plan[1]["gates"] = {"head:0:0": 0}
    with pytest.raises(ValueError, match="no activation gates"):
        states.validate_states(plan, model.config)
