"""Clinical scorecard (experiments/e3_errors/analyze_scorecard.py): the present/absent call and the blind diagnosis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/e3_errors"))

import analyze_scorecard as sc  # noqa: E402


def test_call_needs_an_interval_clear_of_zero_in_the_clinical_direction():
    assert sc.call(0.5, 0.1, 0.9) == "present"
    assert sc.call(-0.5, -0.9, -0.1) == "absent (reversed)"
    assert sc.call(0.5, -0.1, 0.9) == "absent"
    assert sc.call(0.0, 0.0, 0.2) == "absent"          # touching zero is not clear of it


def _model(present):
    return {k: dict(call="present" if k in present else "absent") for k in sc.CLINICAL}


def test_diagnosis_counts_matches_with_each_column():
    sd_like = _model({"frequency", "consistency", "comprehension"})
    sa_like = _model({"control_demand", "associative_errors", "cue_benefit", "comprehension"})
    d = sc.diagnose({"sd": sd_like, "sa": sa_like})
    assert (d["sd"]["matches_SD"], d["sd"]["matches_SA"], d["sd"]["diagnosis"]) == (6, 1, "SD")
    assert (d["sa"]["matches_SD"], d["sa"]["matches_SA"], d["sa"]["diagnosis"]) == (1, 6, "SA")


def test_reversed_counts_as_absent():
    plain = _model({"frequency", "control_demand"})
    reversed_ = _model({"frequency", "control_demand"})
    reversed_["consistency"]["call"] = "absent (reversed)"
    d = sc.diagnose({"plain": plain, "reversed": reversed_})
    assert d["plain"] == d["reversed"]
    assert (d["plain"]["matches_SD"], d["plain"]["matches_SA"]) == (3, 2)


def test_five_markers_separate_the_columns_so_no_tie_is_possible():
    # every marker but comprehension differs between SD and SA, so the two counts always sum to an odd number
    for comp in (True, False):
        d = sc.diagnose({"x": _model({"comprehension"} if comp else set())})["x"]
        assert (d["matches_SD"] + d["matches_SA"]) % 2 == 1


def test_random_summary_counts_calls():
    rows = [dict(set=f"r{i}", estimate=e, ci=[0, 0], call=c)
            for i, (e, c) in enumerate([(0.3, "present"), (-0.2, "absent (reversed)"), (0.0, "absent"), (0.1, "absent")])]
    s = sc.summarise_random(rows, "frequency")
    assert (s["n"], s["n_present"], s["n_reversed"], s["share_present"]) == (4, 1, 1, 0.25)
