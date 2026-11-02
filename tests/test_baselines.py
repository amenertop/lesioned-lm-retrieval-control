"""Matched random component sets (lesion.baselines)."""
import numpy as np
import pytest

from lesion.baselines import layer_percentiles, matched_random_sets
from lesion.components import Component

ALL = [Component("expert", l, e) for l in range(4) for e in range(16)]
FREQ = {c: (c.index + 1) / 100 for c in ALL}   # within each layer, higher index = routed more often


def test_layer_matched_and_disjoint_from_targets():
    targets = [Component("expert", 1, 3), Component("expert", 1, 9), Component("expert", 3, 0)]
    exclude = [Component("expert", 1, 4)]
    sets = matched_random_sets(targets, ALL, n_sets=20, seed=0, exclude=exclude)
    assert len(sets) == 20
    for s in sets:
        assert [c.layer for c in s] == [t.layer for t in targets]
        assert len(set(s)) == len(s) and not set(s) & (set(targets) | set(exclude))
    assert sets == matched_random_sets(targets, ALL, n_sets=20, seed=0, exclude=exclude)


def test_routing_frequency_percentile_gap():
    pct = layer_percentiles(FREQ)
    targets = [Component("expert", 2, 15)]
    for s in matched_random_sets(targets, ALL, n_sets=50, routing_freq=FREQ, max_pct_gap=0.2):
        assert abs(pct[s[0]] - pct[targets[0]]) <= 0.2


def test_too_few_candidates():
    with pytest.raises(ValueError):
        matched_random_sets([Component("expert", 0, 15)], ALL, n_sets=1, routing_freq=FREQ, max_pct_gap=0.0)


def test_overlapping_pools_still_matched():
    """Two targets whose only candidates are the same two components: greedy order must not dead-end."""
    layer = [Component("expert", 0, e) for e in range(4)]
    freq = {Component("expert", 0, 0): 0.1, Component("expert", 0, 1): 0.2,
            Component("expert", 0, 2): 0.8, Component("expert", 0, 3): 0.9}
    targets = [Component("expert", 0, 2), Component("expert", 0, 3)]
    sets = matched_random_sets(targets, layer, n_sets=30, routing_freq=freq, max_pct_gap=1.0)
    assert all(set(s) == {Component("expert", 0, 0), Component("expert", 0, 1)} for s in sets)
    with pytest.raises(ValueError, match="no matched set"):
        matched_random_sets(targets, layer, n_sets=1, routing_freq=freq, max_pct_gap=0.4)
