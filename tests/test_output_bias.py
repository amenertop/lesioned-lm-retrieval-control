"""The output-bias patch of the Phi scorers (experiments/second_model/output_bias.py), on a tiny random Phi-3.5-MoE.

With the bias set to zero the patched scorers return the originals' numbers; with a random bias they match the
log-probabilities of the model's own forward pass, which includes the bias. CPU only.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/second_model"))

from lesion import m3_battery, m3_items, models, naming, prefix_scoring, scoring  # noqa: E402

output_bias = pytest.importorskip("output_bias")


@pytest.fixture(scope="module")
def tiny():
    try:
        model, tok = models.load_tiny("phi35-moe-instruct")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no tiny Phi: {e}")
    model.eval()
    return model, tok


def prompts(n=3):
    final = pd.read_csv(m3_items.OUT / "final_items.csv", keep_default_na=False)
    items = final[final.role == "test"].head(n)
    return [p for p in m3_battery.scored_prompts(items, True) if p.condition == "name_free"]


def test_zero_bias_reproduces_originals(tiny):
    model, tok = tiny
    head = model.get_output_embeddings()
    assert head.bias is not None
    with torch.no_grad():
        head.bias.zero_()
    ps = prompts()
    original = m3_battery.candidate_scores(model, tok, ps, True, 4)
    with output_bias.with_output_bias(model) as patched:
        assert patched
        new = m3_battery.candidate_scores(model, tok, ps, True, 4)
    assert m3_battery.candidate_scores is not output_bias.candidate_scores     # restored on exit
    pd.testing.assert_frame_equal(original, new)


def test_bias_matches_full_forward(tiny):
    model, tok = tiny
    head = model.get_output_embeddings()
    with torch.no_grad():
        head.bias.copy_(torch.randn_like(head.bias) * 0.5)
    ps = prompts(2)
    with output_bias.with_output_bias(model):
        table = m3_battery.candidate_scores(model, tok, ps, True, 4)
        ids = scoring.label_token_ids(tok, True, ("A", "B"))
        seq = scoring.encode(tok, ps[0].text, True)
        lab = prefix_scoring.label_logits(model, [ids["A"], ids["B"]], input_ids=torch.tensor([seq]))
    full = model(input_ids=torch.tensor([seq])).logits[0, -1].float()
    assert torch.allclose(lab[0].float(), full[[ids["A"], ids["B"]]], atol=1e-4)
    for row in table.head(4).itertuples():
        p = next(q for q in ps if q.item_id == row.item_id)
        pre, ans = m3_battery.ms.encode_answer(tok, p.text, row.word, True) if hasattr(m3_battery, "ms") else \
            __import__("lesion.m3_scoring", fromlist=["x"]).encode_answer(tok, p.text, row.word, True)
        logits = model(input_ids=torch.tensor([pre + ans])).logits[0].float()
        logp = torch.log_softmax(logits[len(pre) - 1:len(pre) + len(ans) - 1], dim=-1)
        expected = float(logp.gather(1, torch.tensor(ans)[:, None]).sum())
        assert abs(expected - row.logprob) < 1e-3


def test_naming_patch_restored(tiny):
    model, _ = tiny
    before = naming.answer_logprobs
    with output_bias.with_output_bias(model):
        assert naming.answer_logprobs is output_bias.answer_logprobs
    assert naming.answer_logprobs is before
