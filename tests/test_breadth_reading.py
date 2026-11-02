"""P13's pre-written reading (experiments/e3_errors/analyze_breadth.py) on made-up states."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/e3_errors"))

import analyze_breadth as ab  # noqa: E402

SIZES = ["K30", "S64", "S128", "S256"]


def states(wtd, six, naming, cons, lm=None, rand=0.0):
    """Knowledge sets with the given values per size; five random sets per size at `rand` (consistency 0.1)."""
    out = {}
    for i, s in enumerate(SIZES):
        base = dict(n_components=1, knowledge_effect=0.0, m3_naming=0.0, m1_semantic=0.0, m1_string=0.0, m1_slope=1.0)
        out[s] = dict(base, lm_change=(lm or [0.0] * 4)[i], naming_damage=naming[i], m3_word_to_definition=wtd[i],
                      m3_options_letter=six[i], consistency=cons[i])
        for j in range(5):
            out[f"{s}_random_{j}"] = dict(base, lm_change=0.0, naming_damage=rand, m3_word_to_definition=rand,
                                          m3_options_letter=rand, consistency=0.1)
    return out


def verdict(st):
    return ab.reading(ab.by_size(st, SIZES), SIZES)["verdict"]


def test_progression_two_routes_partial_and_the_collapse_line():
    rising = [0.1, 0.5, 0.9, 1.2]
    assert verdict(states(rising, rising, [2, 2, 2, 2], [0.0, 0.3, 0.4, 0.5])) == "progression"
    assert verdict(states([0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0])) == "two routes"
    assert verdict(states(rising, [0, 0, 0, 0], [2, 2, 2, 2], [0.3] * 4)) == "partial"
    # comprehension fails only where language has collapsed: not counted, so naming alone decides
    assert verdict(states([0, 0, 0, 5], [0, 0, 0, 5], [2, 2, 2, 2], [0, 0, 0, 0.5], lm=[0, 0, 0, 0.5])) == "two routes"
    # rising but consistency never above the random sets: partial, not progression
    assert verdict(states(rising, rising, [2, 2, 2, 2], [0.0] * 4)) == "partial"
