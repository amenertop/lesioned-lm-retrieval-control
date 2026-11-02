"""Attribution and damage statistics on the definition-naming test (knowledge side)."""
import numpy as np
import pandas as pd
import pytest

from lesion import attribution, hooks, knowledge_items as ki, means as means_mod, models, naming, neutral_text
from lesion.components import Component

NAME = "qwen3-30b-a3b-instruct"


def test_naming_weights_reproduce_effects():
    items = ki.load_items("screening")
    med = naming.zipf_median(items)
    rng = np.random.default_rng(0)
    h = pd.Series(rng.normal(size=len(items)), index=items.item_id, name="h")
    frame = naming.item_frame(h, items, med)
    ref = naming.effects({c: frame[c].to_numpy() for c in frame.columns})
    for s, v in naming.statistic_weights(items, med).items():
        # S = sum v * logprob changes by the statistic of h, since h = -(change in logprob)
        assert float((v * -h.reindex(items.item_id).to_numpy()).sum()) == pytest.approx(ref[s], abs=1e-9)


def test_naming_attribution_is_the_first_order_effect():
    """Matches real small-step ablations for an expert and a head on a tiny Qwen."""
    if not models.local_dir(NAME).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(NAME)
    items = ki.load_items("screening").iloc[::15].reset_index(drop=True)      # 8 items across frequency
    demos = pd.read_csv(ki.OUT / "demos.csv")
    prompts = naming.build_prompts(items, demos, chat=True)
    weights = naming.statistic_weights(items, naming.zipf_median(items))
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)
    est = attribution.naming_attribution(model, tok, prompts, True, m, weights, batch_size=3).set_index("component")
    intact = naming.answer_logprobs(model, tok, prompts, True, 4).logprob.to_numpy()
    # Answer log-probabilities are large (about -40 on a random model), so float32 cannot resolve a 0.1% step:
    # take the components with the largest estimates and a 5% step towards the mean, allowing 10% error.
    eps = 0.05
    top = [est[est.kind == k].knowledge_effect.abs().idxmax() for k in ("expert", "head")]
    for c in [Component.parse(k) for k in top]:
        with hooks.ablate(model, [c], m, alpha=1 - eps):
            moved = naming.answer_logprobs(model, tok, prompts, True, 4).logprob.to_numpy()
        for s, v in weights.items():
            actual = float((v * (moved - intact)).sum()) / eps
            predicted = est.at[c.key, s]
            assert abs(actual - predicted) <= 0.10 * abs(predicted) + 1e-3, (c, s, actual, predicted)
