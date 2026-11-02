"""Checks the Experiment 0 demo items and templates against docs/plans/pilot-protocol.md §3-4.

Run with pytest, or directly: python tests/test_demos.py
"""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
M1_RAW = ROOT / "data/raw/wu_hoffman_2022/Item_properties.csv"
DEMOS = ROOT / "data/materials/m1/demos.csv"
TEMPLATES = ROOT / "data/materials/m1/templates.yaml"

TASKS = ["colour", "size", "synonym", "string"]
PATTERN = ["A", "B", "B", "A", "B", "A"]


def load_demos():
    return pd.read_csv(DEMOS, dtype=str, keep_default_na=False)


def demo_words(demos):
    cols = ["probe", "target", "distractor", "word1", "word2"]
    return [w.lower() for w in demos[cols].to_numpy().ravel() if w]


def test_positions_and_answer_pattern():
    demos = load_demos()
    for task in TASKS:
        rows = demos[demos.task == task].sort_values("position")
        assert rows.position.tolist() == [str(i) for i in range(1, 7)], task
        assert rows.correct_label.tolist() == PATTERN, task


def test_matching_options_agree_with_answer():
    demos = load_demos()
    for r in demos[demos.task != "synonym"].itertuples():
        correct = r.option_a if r.correct_label == "A" else r.option_b
        wrong = r.option_b if r.correct_label == "A" else r.option_a
        assert (correct, wrong) == (r.target, r.distractor), (r.task, r.position)


def test_demand_and_synonym_balance():
    demos = load_demos()
    for task in ["colour", "size"]:
        assert demos[demos.task == task].demand.value_counts().to_dict() == {"high": 3, "low": 3}, task
    syn = demos[demos.task == "synonym"]
    assert syn.synonym_type.value_counts().to_dict() == {"similar": 3, "different": 3}
    assert all((s == "similar") == (lab == "A") for s, lab in zip(syn.synonym_type, syn.correct_label))


def test_no_overlap_with_m1_and_no_reuse():
    m1 = pd.read_csv(M1_RAW, encoding="utf-8-sig")
    vocab = set()
    for col in ["Probe", "Target", "D1", "Op1", "Op2"]:
        vocab |= set(m1[col].dropna().astype(str).str.lower())
    words = demo_words(load_demos())
    assert not sorted(set(words) & vocab), sorted(set(words) & vocab)
    dupes = sorted({w for w in words if words.count(w) > 1})
    assert not dupes, dupes


def test_string_demos_follow_m1_rules():
    """Each string demo must match the signature of one of M1's five spelling rules, derived from M1 itself."""
    from lesion.materials import OUT, string_rule_table, string_signature
    items = pd.read_csv(OUT / "items.csv", dtype=str, keep_default_na=False)
    table = string_rule_table(items)
    covered = set()
    for r in load_demos().query("task == 'string'").itertuples():
        sig = (len(r.probe), string_signature(r.probe, r.target), string_signature(r.probe, r.distractor))
        matches = [rule for rule, rule_sig in table.items() if rule_sig == sig]
        assert matches, (r.position, sig)
        covered.update(matches)
    assert covered == set(table), f"rules not covered: {set(table) - covered}"


def test_templates_complete():
    t = yaml.safe_load(TEMPLATES.read_text(encoding="utf-8"))
    for task in TASKS:
        assert set(t["instructions"][task]) == {"main", "alt1", "alt2"}, task
        assert "{a}" in t["blocks"][task] and "{b}" in t["blocks"][task], task
        assert t["blocks"][task].endswith("Answer:"), task
    assert set(t["cloze"]["instructions"]) == {"colour", "size"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
