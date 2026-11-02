"""M1 formatting and concept-group split (docs/plans/pilot-protocol.md §3, ticket pilot 01).

Reads Wu & Hoffman (2022) Item_properties.csv and writes:
  data/materials/m1/items.csv  one row per item in a task-independent format
  data/materials/m1/split.csv  one row per concept group: screening, dev or test

Run: python -m lesion.materials   (from the repo root, with src/ on PYTHONPATH)
No model is run here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/wu_hoffman_2022/Item_properties.csv"
OUT = ROOT / "data/materials/m1"
SEED = 0

MATCHING_TASKS = ("colour", "size")


def load_raw(path=RAW):
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"Correct": str})
    # The file has two columns named "Probe"; pandas renames the second to "Probe.1".
    both = raw[["Probe", "Probe.1"]].dropna()
    assert (both["Probe"] == both["Probe.1"]).all(), "the two Probe columns disagree"
    return raw


def format_items(raw):
    """Map the raw columns onto one explicit item format."""
    rows = []
    for r in raw.itertuples(index=False):
        task = r.Task
        row = dict(
            item_id=int(r.Item),
            task=task,
            probe="",
            option1="",
            option2="",
            correct_option="",
            word1="",
            word2="",
            target="",
            distractor="",
            synonym_type="",
            spelling_rule="",
            tds=r.TDS,
            min_freq=r.Min_freq,
            young_acc=r.Young_Acc,
            old_acc=r.Old_Acc,
            difficulty_z=r.Ave_groups_Z,
        )
        if task in MATCHING_TASKS or task == "string":
            row["probe"] = str(r.Probe)
            row["option1"], row["option2"] = str(r.Op1), str(r.Op2)
            row["target"] = str(r.Target)
            row["distractor"] = str(r.D1)
            row["correct_option"] = {"1": "option1", "2": "option2"}[r.Correct]
            correct_word = row["option1"] if r.Correct == "1" else row["option2"]
            assert correct_word == row["target"], f"item {r.Item}: correct option is not the target"
        if task == "string":
            row["spelling_rule"] = str(int(r.Spelling_rule))
            row["tds"] = np.nan
        if task == "synonym":
            # Options are the response labels, as in demos.csv; the two words go in word1/word2.
            row["word1"], row["word2"] = str(r.Op1), str(r.Op2)
            row["option1"], row["option2"] = "similar", "different"
            row["synonym_type"] = {"s": "similar", "d": "different"}[r.Correct]
            row["correct_option"] = "option1" if row["synonym_type"] == "similar" else "option2"
        rows.append(row)
    items = pd.DataFrame(rows)

    # High control demand: TDS below the within-task median (colour and size only).
    items["high_demand"] = ""
    for task in MATCHING_TASKS:
        mask = items.task == task
        median = items.loc[mask, "tds"].median()
        items.loc[mask, "high_demand"] = np.where(items.loc[mask, "tds"] < median, "1", "0")

    items["concept_group"] = assign_concept_groups(items)
    return items.sort_values("item_id").reset_index(drop=True)


def assign_concept_groups(items):
    """Colour and size items share groups by probe; synonym items by word pair; string items are their own group."""
    groups = pd.Series(index=items.index, dtype=object)
    matching = items.task.isin(MATCHING_TASKS)
    groups[matching] = "fm_" + items.loc[matching, "probe"].str.lower()
    syn = items.task == "synonym"
    groups[syn] = [
        "syn_" + "|".join(sorted((a.lower(), b.lower())))
        for a, b in zip(items.loc[syn, "word1"], items.loc[syn, "word2"])
    ]
    string = items.task == "string"
    groups[string] = "str_" + items.loc[string, "item_id"].astype(str)
    return groups


# Three-way split (user decision 2026-09-27): about 40% screening (find components),
# 20% dev (tune parameters), 40% test (held out for the confirmatory tests).
SPLIT_CYCLE = ("screening", "test", "dev", "screening", "test")


def split_groups(items, seed=SEED):
    """Assign each concept group to screening, dev or test (about 40/20/40) within each stratum."""
    g = items.groupby("concept_group").agg(
        tasks=("task", lambda t: "+".join(sorted(set(t)))),
        tds=("tds", "mean"),
        min_freq=("min_freq", "mean"),
        spelling_rule=("spelling_rule", "first"),
        n_items=("item_id", "size"),
    ).reset_index()

    g["stratum"] = ""
    fm = g.tasks.str.contains("colour|size")
    g.loc[fm, "stratum"] = g.loc[fm, "tasks"] + "_tds" + tercile(g.loc[fm, "tds"])
    syn = g.tasks == "synonym"
    g.loc[syn, "stratum"] = "synonym_freq" + tercile(g.loc[syn, "min_freq"])
    string = g.tasks == "string"
    g.loc[string, "stratum"] = "string_rule" + g.loc[string, "spelling_rule"]

    rng = np.random.default_rng(seed)
    g = g.sort_values("concept_group").reset_index(drop=True)
    g["split"] = ""
    for stratum in sorted(g.stratum.unique()):
        idx = g.index[g.stratum == stratum].to_numpy()
        order = idx[rng.permutation(len(idx))]
        # Cycle through SPLIT_CYCLE from a random starting point, so small strata are not always tilted the same way.
        first = rng.integers(len(SPLIT_CYCLE))
        for k, i in enumerate(order):
            g.loc[i, "split"] = SPLIT_CYCLE[(k + first) % len(SPLIT_CYCLE)]
    return g[["concept_group", "tasks", "stratum", "n_items", "split"]]


def tercile(values):
    """Label values 1..3 by tercile, computed over the values given."""
    ranks = values.rank(method="first")
    return pd.cut(ranks, 3, labels=["1", "2", "3"]).astype(str)


SPLITS = ("screening", "dev", "test")


def load_split(split, m1_dir=OUT, freeze_tag=None):
    """M1 items of one split, with the split column attached.

    The held-out test split is only read by confirmatory runs after the analysis plan is frozen
    (spec.md, data split table): freeze_tag must name the analysis-plan freeze tag, and HEAD must be
    that tag on a clean tree (lesion.runinfo.check_analysis_freeze). Record the tag in the run's meta.json.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "test":
        from . import runinfo

        runinfo.check_analysis_freeze(freeze_tag)
    items = pd.read_csv(m1_dir / "items.csv", dtype=str, keep_default_na=False)
    groups = pd.read_csv(m1_dir / "split.csv", dtype=str, keep_default_na=False)[["concept_group", "split"]]
    items = items.merge(groups, on="concept_group", how="left", validate="many_to_one")
    assert (items.split != "").all() and items.split.notna().all(), "every item needs a split"
    return items[items.split == split].reset_index(drop=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = format_items(load_raw())
    split = split_groups(items)
    items.to_csv(OUT / "items.csv", index=False, lineterminator="\n", float_format="%.6f")
    split.to_csv(OUT / "split.csv", index=False, lineterminator="\n")
    print(f"items: {len(items)}  concept groups: {len(split)}")
    print(split.groupby(["tasks", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()


# --- String-matching demos (ticket pilot 02) ---------------------------------------------

def string_signature(probe, option):
    """(same-position letters, displaced letters, first letter kept) of option relative to probe."""
    same = sum(a == b for a, b in zip(probe, option))
    displaced = sum(1 for i, ch in enumerate(option) if ch in probe and not (i < len(probe) and probe[i] == ch))
    return same, displaced, probe[0] == option[0]


def string_rule_table(items):
    """Map each M1 spelling rule to its (length, target signature, distractor signature); asserts each rule is uniform."""
    table = {}
    for rule, g in items[items.task == "string"].groupby("spelling_rule"):
        sigs = {(len(r.probe), string_signature(r.probe, r.target), string_signature(r.probe, r.distractor))
                for r in g.itertuples()}
        assert len(sigs) == 1, f"rule {rule} is not uniform: {sigs}"
        table[rule] = sigs.pop()
    return table


def make_string_item(rule_sig, rng, letters="abcdefghijklmnopqrstuvwxyz"):
    """Generate one (probe, target, distractor) with the given rule signature."""
    n, (t_same, t_disp, _), (d_same, d_disp, _) = rule_sig
    probe = "".join(rng.choice(list(letters), n, replace=False))
    outside = [c for c in letters if c not in probe]

    def build(n_same, n_disp):
        keep = {0} | set(rng.choice(np.arange(1, n), n_same - 1, replace=False)) if n_same else set()
        free = [i for i in range(n) if i not in keep]
        disp_pos = set(rng.choice(free, n_disp, replace=False)) if n_disp else set()
        word = []
        for i in range(n):
            if i in keep:
                word.append(probe[i])
            elif i in disp_pos:
                word.append(rng.choice([c for j, c in enumerate(probe) if j != i and c not in word]))
            else:
                word.append(rng.choice([c for c in outside if c not in word]))
        return "".join(word)

    target, distractor = build(t_same, t_disp), build(d_same, d_disp)
    assert string_signature(probe, target)[:2] == (t_same, t_disp)
    assert string_signature(probe, distractor)[:2] == (d_same, d_disp)
    return probe, target, distractor


STRING_DEMO_RULES = ["1", "2", "3", "4", "5", "3"]  # positions 1-6; answers follow A,B,B,A,B,A


def regenerate_string_demos(seed=20260927):
    """Replace the string rows of demos.csv with items generated from M1's rule table."""
    items = pd.read_csv(OUT / "items.csv", dtype=str, keep_default_na=False)
    table = string_rule_table(items)
    m1_strings = set(items.loc[items.task == "string", ["probe", "target", "distractor"]].to_numpy().ravel())
    demos = pd.read_csv(OUT / "demos.csv", dtype=str, keep_default_na=False)
    rng = np.random.default_rng(seed)
    for pos, rule in enumerate(STRING_DEMO_RULES, start=1):
        while True:
            probe, target, distractor = make_string_item(table[rule], rng)
            if not {probe, target, distractor} & m1_strings:
                break
        i = demos.index[(demos.task == "string") & (demos.position == str(pos))][0]
        label = demos.at[i, "correct_label"]
        a, b = (target, distractor) if label == "A" else (distractor, target)
        demos.loc[i, ["probe", "option_a", "option_b", "target", "distractor", "note"]] = [
            probe, a, b, target, distractor, f"M1 spelling rule {rule}"]
    demos.to_csv(OUT / "demos.csv", index=False, lineterminator="\n")
    return demos[demos.task == "string"]


# --- probe-option associations for the control test (spec.md, 核心证伪实验) --------------------------

ASSOCIATIONS = OUT / "associations.csv"


def write_associations(raw_path=RAW, out=ASSOCIATIONS):
    """Word2vec similarity of the probe to the target and to the distractor (Wu & Hoffman 2022 columns
    w2vTP and w2vD1P) for the colour and size items, in a file of its own so items.csv stays unchanged."""
    raw = load_raw(raw_path)
    raw = raw[raw.Task.isin(MATCHING_TASKS)]
    t = pd.DataFrame(dict(item_id=raw.Item.astype(int), w2v_target_probe=raw.w2vTP.astype(float),
                          w2v_distractor_probe=raw.w2vD1P.astype(float)))
    t.sort_values("item_id").to_csv(out, index=False, lineterminator="\n", float_format="%.6f")
    return t
