"""lesion.gate_objective and lesion.p14 (P14): the differentiable scores agree with the no-gradient scorers on tiny
random Qwen3-MoE and Phi-MoE models, carry gradients to the gates, and the P14 helpers (splits, pool, objective
pieces, set-size rule) do what their docstrings say."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import PhimoeConfig, PhimoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from lesion import gate_objective as go, gates, naming, p14, probes, scoring
from lesion.components import Component

PAIRS = [([2, 7, 3, 9, 5], [11, 4, 1]), ([6, 4, 8], [3, 1]), ([1, 3, 5, 7, 2, 2], [9, 1]), ([9, 9, 2], [4, 6, 8, 1])]
SEQS = [[2, 7, 3, 9, 5, 11], [6, 4, 8], [1, 3, 5, 7], [9, 9, 2, 4, 6]]


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


def test_naming_logprobs_match_answer_logprobs(tiny, monkeypatch):
    model, _ = tiny
    prompts = [SimpleNamespace(item_id=f"k{i}", target="w", pair=p) for i, p in enumerate(PAIRS)]
    monkeypatch.setattr(naming, "encode_pair", lambda tok, p, chat: p.pair)
    ref = naming.answer_logprobs(model, SimpleNamespace(eos_token_id=0), prompts, False, batch_size=4)
    with torch.no_grad():
        mine = go.naming_logprobs(model, PAIRS, pad_id=0)
    assert np.allclose(mine.numpy(), ref.logprob.to_numpy(), atol=1e-4)


def test_label_margins_match_the_label_scorer(tiny, monkeypatch):
    model, _ = tiny
    monkeypatch.setattr(scoring, "encode", lambda tok, text, chat, fmt=None: list(text))
    monkeypatch.setattr(scoring, "label_token_ids", lambda tok, chat, *a, **k: {"A": 5, "B": 9})
    prompts = [SimpleNamespace(item_id=str(i), task="colour", order=i % 2, text=s, correct_label="AB"[i % 2])
               for i, s in enumerate(SEQS)]
    ref = probes.label_scores_in_positions(model, SimpleNamespace(eos_token_id=0), prompts, False, batch_size=4)
    correct = [5 if p.correct_label == "A" else 9 for p in prompts]
    wrong = [9 if p.correct_label == "A" else 5 for p in prompts]
    with torch.no_grad():
        mine = go.label_margins(model, SEQS, correct, wrong, pad_id=0)
    assert np.allclose(mine.numpy(), (ref.logit_correct - ref.logit_wrong).to_numpy(), atol=1e-4)


def test_lm_loss_matches_the_probe(tiny):
    model, means = tiny
    ref = probes.lm_loss_region(model, SEQS, [], means, batch_size=4)
    with torch.no_grad():
        mine = float(go.lm_loss(model, SEQS))
    assert abs(mine - ref) < 1e-4


def test_scores_carry_gradients_to_the_gates(tiny):
    model, means = tiny
    g = gates.Gates([Component("head", 0, 1), Component("expert", 1, 2), Component("expert", 0, 3)], init=0.5)
    with gates.gated(model, g, means):
        loss = -go.naming_logprobs(model, PAIRS).mean() + go.lm_loss(model, SEQS) \
            - go.label_margins(model, SEQS, [5] * 4, [9] * 4).mean()
    loss.backward()
    assert torch.isfinite(g.logits.grad).all() and (g.logits.grad.abs() > 0).any()


def test_checkpointing_gives_the_same_gradients_and_counts(tiny):
    """Gradient checkpointing recomputes the forward pass in backward: run inside the gated context it gives the same
    gate gradients, and with recording off by then the routed-token counts are not doubled (train_gates.train)."""
    model, means = tiny
    comps = [Component("head", 0, 1), Component("expert", 1, 2), Component("expert", 0, 3)]

    def grads(checkpointing):
        g = gates.Gates(comps, init=0.4)
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
        try:
            with gates.gated(model, g, means):
                g.recording = True
                loss = -go.naming_logprobs(model, PAIRS).mean()
                g.recording = False
                loss.backward()
        finally:
            if checkpointing:
                model.gradient_checkpointing_disable()
                model.eval()
        return g.logits.grad.clone(), g.routed.clone()

    plain, routed_plain = grads(False)
    ckpt, routed_ckpt = grads(True)
    assert torch.allclose(plain, ckpt, atol=1e-6)
    assert torch.equal(routed_plain, routed_ckpt) and int(routed_plain.sum()) > 0


def test_routed_tokens_are_counted_only_while_recording(tiny):
    model, means = tiny
    g = gates.Gates([Component("expert", 0, e) for e in range(4)])
    with torch.no_grad(), gates.gated(model, g, means):
        model(input_ids=torch.tensor(SEQS[:1]))
    assert int(g.routed.sum()) == 0
    g.recording = True
    with torch.no_grad(), gates.gated(model, g, means):
        model(input_ids=torch.tensor(SEQS[:1]))
    # each of the 6 tokens goes to 2 of the 4 experts of layer 0
    assert int(g.routed.sum()) == 6 * 2


# --- lesion.p14 -------------------------------------------------------------------------------------------------------

def test_unused_split_is_disjoint_balanced_and_fixed():
    words = pd.DataFrame(dict(target=[f"w{i}" for i in range(11)], zipf=np.linspace(1, 4, 11), definition="d"))
    words = p14.unused_words(pd.DataFrame(dict(target=["w0"])), pd.DataFrame(dict(target=["w1"])), words)
    assert len(words) == 9 and words.item_id.iloc[0] == "u000"
    a, b = p14.split_unused(words), p14.split_unused(words)
    assert a["select"].item_id.tolist() == b["select"].item_id.tolist()
    assert set(a["select"].item_id).isdisjoint(a["confirm"].item_id)
    assert len(a["select"]) == 5 and len(a["confirm"]) == 4
    assert abs(a["select"].zipf.mean() - a["confirm"].zipf.mean()) < 0.5


def test_pool_has_the_top_the_screened_and_sentinels_but_no_excluded():
    att = pd.DataFrame(dict(component=[f"head:{l}:{h}" for l in range(4) for h in range(8)]))
    att["kind"], att["layer"] = "head", [int(c.split(":")[1]) for c in att.component]
    att["knowledge_effect"] = np.arange(len(att))[::-1] / 10.0
    members, sources = p14.build_pool(att, "knowledge_effect", 5, screened=["head:3:7"], exclude=["head:0:1"], n_sentinels=4)
    top = [c for c in att.component if c != "head:0:1"][:5]
    assert set(top) <= set(members) and "head:3:7" in members and "head:0:1" not in members
    assert sum(s == ["sentinel"] for s in sources.values()) == 4
    assert len(members) == 5 + 1 + 4


def test_rare_coefficient_is_the_least_squares_coefficient():
    rng = np.random.default_rng(0)
    rare = rng.integers(2, size=40).astype(bool)
    ntok = rng.integers(1, 5, size=40).astype(float)
    h = 0.7 * rare + 0.2 * ntok + rng.normal(0, .1, 40)
    X = np.column_stack([np.ones(40), rare, (ntok - ntok.mean()) / ntok.std(ddof=1)])
    expected = np.linalg.lstsq(X, h, rcond=None)[0][1]
    assert abs(p14.rare_coefficient_np(h, rare, ntok) - expected) < 1e-6
    assert np.isnan(p14.rare_coefficient_np(h, np.ones(40, bool), ntok))
    # a token count collinear with the rare flag is dropped: the coefficient is then the difference of means
    collinear = rare.astype(float) * 2 + 1
    assert abs(p14.rare_coefficient_np(h, rare, collinear) - (h[rare].mean() - h[~rare].mean())) < 1e-6


def test_control_difference_averages_the_two_tasks():
    h = torch.tensor([3.0, 1.0, 5.0, 1.0, 9.0])
    task = ["colour", "colour", "size", "size", "string"]
    high = torch.tensor([True, False, True, False, False])
    assert float(p14.control_difference(h, task, high)) == pytest.approx(((3 - 1) + (5 - 1)) / 2)


def test_violation_and_choose_size():
    assert float(p14.violation(torch.tensor(0.1), 0.2)) == 0
    assert float(p14.violation(torch.tensor(0.3), 0.2)) == pytest.approx(0.25)
    base = dict(common_naming=0.0, lm_change=0.0, m1_semantic=0.0, m1_string=0.0)
    curve = pd.DataFrame([dict(k=1, selectivity=0.2, **base), dict(k=2, selectivity=0.9, **base),
                          dict(k=4, selectivity=1.0, **base), dict(k=8, selectivity=2.0, **{**base, "lm_change": 0.05})])
    curve["selectivity_se"] = [0.3, 0.2, 0.12, 0.01]
    assert p14.choose_size(curve, "knowledge", delta=0.15) == (2, 0.15)   # k=8 is infeasible; 0.9 is within 0.15 of 1.0
    assert p14.choose_size(curve, "knowledge", delta=0.05) == (4, 0.05)
    assert p14.choose_size(curve, "knowledge") == (2, 0.12)              # delta: the se at the best feasible row
    assert p14.choose_size(curve.assign(lm_change=1.0), "knowledge") == (None, None)
    assert p14.matched_order({"head:1:0": 0.5, "head:0:0": 0.1, "expert:0:1": 0.5}) == ["head:0:0", "expert:0:1", "head:1:0"]
