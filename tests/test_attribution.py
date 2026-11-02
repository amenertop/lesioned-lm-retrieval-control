"""Attribution estimates match real ablations to first order (ADR 0004)."""
import numpy as np
import pandas as pd
import pytest

from lesion import attribution, damage, hooks, materials, means as means_mod, models, neutral_text, prefix_scoring, scoring
from lesion.components import Component


@pytest.mark.parametrize("name,chat", [("olmoe-0924-base", False), ("qwen3-30b-a3b-instruct", True)])
def test_attribution_is_the_first_order_effect(name, chat):
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    all_items, demos, templates = scoring.load_materials()
    items = materials.load_split("screening")
    syn = items[items.task == "synonym"].assign(f=lambda d: d.min_freq.astype(float)).sort_values("f")
    pick = items[items.task.isin(["colour", "size"])].groupby(["task", "high_demand"]).head(2)
    pick = pd.concat([pick, syn.head(2).drop(columns="f"), syn.tail(2).drop(columns="f"), items[items.task == "string"].head(2)])
    prompts = scoring.build_label_prompts(pick, demos, templates, chat=chat)
    fm = damage.frequency_median(all_items)
    weights = attribution.statistic_weights(prompts, pick, fm)
    assert all(np.abs(v).sum() > 0 for v in weights.values())
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)
    est = attribution.component_attribution(model, tok, prompts, chat, m, weights, batch_size=3).set_index("component")

    margin = lambda s: (s.logit_correct - s.logit_wrong).to_numpy()
    intact = margin(prefix_scoring.score_label_prompts_full(model, tok, prompts, chat, 4))
    eps = 1e-3                                             # alpha = 1 - eps: a tiny step towards the mean
    for c in [Component("expert", 0, 1), Component("expert", 1, 5), Component("head", 0, 2), Component("head", 1, 3)]:
        with hooks.ablate(model, [c], m, alpha=1 - eps):
            moved = margin(prefix_scoring.score_label_prompts_full(model, tok, prompts, chat, 4))
        for s, v in weights.items():
            actual = float((v * (moved - intact)).sum()) / eps
            predicted = est.at[c.key, s]
            assert abs(actual - predicted) <= 0.02 * abs(predicted) + 1e-4, (c, s, actual, predicted)


def test_weights_reproduce_damage_effects():
    """sum_p v_p * (lesioned - intact margin) equals the damage statistics of lesion.damage."""
    import fakes

    all_items = scoring.load_materials()[0]
    items = materials.load_split("screening")
    _, demos, templates = scoring.load_materials()
    prompts = scoring.build_label_prompts(items, demos, templates)
    fm = damage.frequency_median(all_items)
    rng = np.random.default_rng(0)
    drop = {i: rng.normal() for i in items.item_id}
    intact = fakes.label_scores(items, lambda r: 2.0)
    lesioned = fakes.label_scores(items, lambda r: 2.0 - drop[r["item_id"]])
    by = lambda s: s.set_index(["item_id", "order"])
    key = [(p.item_id, p.order) for p in prompts]
    dm = (by(lesioned).loc[key, "logit_correct"] - by(lesioned).loc[key, "logit_wrong"]).to_numpy() - 2.0
    ref = damage.effects(damage.columns(damage.item_frame(damage.item_damage(intact, lesioned), items, fm)))
    for s, v in attribution.statistic_weights(prompts, items, fm).items():
        assert float((v * dm).sum()) == pytest.approx(ref[s], abs=1e-9)


def test_padding_does_not_change_attribution():
    """Batches mixing prompt lengths (left padding) give the same estimates as one prompt at a time."""
    name = "qwen3-30b-a3b-instruct"
    if not models.local_dir(name).exists():
        pytest.skip("tokenizer files not downloaded")
    model, tok = models.load_tiny(name)
    all_items, demos, templates = scoring.load_materials()
    items = materials.load_split("screening").groupby("task").head(2)     # four tasks: prompt lengths differ
    prompts = scoring.build_label_prompts(items, demos, templates, chat=True)
    assert len({len(scoring.encode(tok, p.text, True)) for p in prompts}) > 1
    weights = attribution.statistic_weights(prompts, items, damage.frequency_median(all_items))
    m = means_mod.collect_means(model, tok, neutral_text.load()[:3], max_tokens=32, batch_size=3)
    one = attribution.component_attribution(model, tok, prompts, True, m, weights, batch_size=1).set_index("component")
    padded = attribution.component_attribution(model, tok, prompts, True, m, weights, batch_size=len(prompts)).set_index("component")
    for s in weights:
        assert np.allclose(one[s], padded[s], atol=1e-5), s
