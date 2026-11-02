"""GPU session 11, main model, exploratory: one report of the two re-screens and their held-out tests
(experiments/e3_errors/run_qwen_rescreens.sh): the store screen on the development items (S8), the dissociation test
with S8 and the semantic-only control set C4s in place of K8 and C4, the control test of C4s, damage by task and
control demand for C4s and S8 (lesion.damage.demand_breakdown, as for C4 and Phi in analyze_phi.py), and the
language-model loss check run during the session (experiments/e2_confirm/qwen_lm_check.py). No GPU:

    python experiments/e2_confirm/analyze_s11_qwen.py [--out reports/explore/qwen_s11_heldout.json]

The clinical scorecard of the new sets, the cued naming and the synonym judgement by frequency are in
reports/explore/qwen_new_sets.json (experiments/e3_errors/analyze_new_sets.py).

Two checks added in the joint reading with agent2 (reports/explore/2026-10-14-session11-joint/), for K8 and C4 (session
10's dissociation test) as well as S8 and C4s:
- margin_scaling (lesion.damage): per M1 task, lesioned margin regressed on intact margin. A slope below 1 on every
  task with an intercept near 0 means the lesion shrinks every answer by the same factor, whatever its content.
- naming_frequency_vs_difficulty: is the naming frequency effect only the rarer words being harder already? The
  lesion's damage per concept regressed on -Zipf and the intact answer log-probability (both standardised); the -Zipf
  coefficient with a concept-resampled 95% interval (4,000, seed 0).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import damage, runinfo, scoring  # noqa: E402

MODEL = "qwen3-30b-a3b-instruct"
STORE = ROOT / "outputs/e3_knowledge" / MODEL / "store_screen"
CONFIRM = ROOT / "outputs/e2_confirm" / MODEL
NEW, SEMANTIC = CONFIRM / "dissociation_new", CONFIRM / "control_semantic"
OLD = ROOT / "outputs/server/e2_confirm" / MODEL / "dissociation"
KNOWLEDGE_META = ROOT / "outputs/server/e3_knowledge" / MODEL / "screen/meta.json"
KNOWLEDGE_ITEMS = ROOT / "data/materials/m3_knowledge/items.csv"
LM = ROOT / "outputs/explore_s11_qwen/qwen_lm_check.json"
N_BOOT, SEED = 4_000, 0


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def interval(v):
    return [round(v["estimate"], 3), round(v["ci_low"], 3), round(v["ci_high"], 3)]


def intervals(d):
    """estimate and 95% interval of every statistic in a (possibly nested) result block."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and "estimate" in v:
            out[k] = interval(v)
        elif isinstance(v, dict):
            out.update({f"{k}.{kk}": vv for kk, vv in intervals(v).items()})
    return out


def breakdown(intact, lesioned, items):
    t = damage.demand_breakdown(intact.astype({"item_id": str}), lesioned.astype({"item_id": str}), items)
    return t.round(3).to_dict(orient="records")


def scaling(m1, set_name, items):
    m1 = m1.astype({"item_id": str})
    t = damage.margin_scaling(m1[m1.set == "intact"], m1[m1.set == set_name], items)
    return t.round(3).to_dict(orient="records")


def frequency_vs_difficulty(naming, set_name, knowledge_items, median):
    """Naming damage per concept (intact minus lesioned answer log-probability, order-averaged scoring) regressed on
    -Zipf and the intact log-probability, both standardised."""
    naming = naming[naming.scoring == "averaged"].astype({"item_id": str})
    a = naming[naming.set == "intact"].set_index("item_id").logprob
    b = naming[naming.set == set_name].set_index("item_id").logprob
    if set(a.index) != set(b.index):
        raise ValueError("intact and lesioned naming scores cover different concepts")
    t = pd.DataFrame(dict(intact=a, h=a - b.reindex(a.index))).join(knowledge_items.set_index("item_id")[["zipf"]])
    rare = t.zipf < median
    z = lambda v: ((v - v.mean()) / v.std()).to_numpy()  # noqa: E731
    X, y = np.column_stack([np.ones(len(t)), z(-t.zipf), z(t.intact)]), t.h.to_numpy()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(SEED)
    boot = [np.linalg.lstsq(X[i], y[i], rcond=None)[0][1] for i in rng.integers(0, len(t), (N_BOOT, len(t)))]
    slope, intercept = np.polyfit(t.intact, t.intact - t.h, 1)
    return dict(n=len(t), knowledge_effect=round(float(t.h[rare].mean() - t.h[~rare].mean()), 3),
                rarity_coef_given_intact=[round(float(coef[1]), 3)] + [round(float(q), 3) for q in np.percentile(boot, [2.5, 97.5])],
                intact_logprob_coef=round(float(coef[2]), 3),
                corr_intact_logprob_damage=round(float(np.corrcoef(t.intact, t.h)[0, 1]), 3),
                lesioned_on_intact=dict(slope=round(float(slope), 3), intercept=round(float(intercept), 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports/explore/qwen_s11_heldout.json"))
    args = ap.parse_args()
    items = scoring.load_materials()[0]
    new, sets = read(NEW / "results.json"), read(NEW / "sets.json")
    semantic = read(SEMANTIC / "results.json")
    m1 = pd.read_parquet(NEW / "m1_scores.parquet")
    lesioned = pd.read_parquet(SEMANTIC / "lesioned.parquet")
    lm = read(LM)
    old_m1 = pd.read_parquet(OLD / "m1_scores.parquet")
    naming_new, naming_old = pd.read_parquet(NEW / "naming_scores.parquet"), pd.read_parquet(OLD / "naming_scores.parquet")
    knowledge_items = pd.read_csv(KNOWLEDGE_ITEMS, dtype={"item_id": str})
    median = read(KNOWLEDGE_META)["params"]["zipf_median"]
    report = dict(
        status="exploratory: re-screens chosen after session 10 (docs/plans/gpu-session11.md)",
        store_screen=read(STORE / "selection.json"),
        store_screen_dev=intervals({k: v for k, v in read(STORE / "joint.json").items() if k != "selection"}),
        sets=dict(S8=sets["main"], C4s=sets["control"]),
        dissociation_new=dict(sets={name: intervals(s) for name, s in new["sets"].items()},
                              random=new["random"], h2=new["h2"]),
        control_semantic=dict(intervals={k: interval(v) for k, v in semantic["intervals"].items()},
                              primary=semantic["primary"], random=semantic["random"],
                              relative_to_intact=semantic["relative_to_intact"]),
        by_demand=dict(
            C4s_control_test=breakdown(pd.read_csv(SEMANTIC / "intact.csv"), lesioned[lesioned.set == "control"], items),
            C4s_dissociation=breakdown(m1[m1.set == "intact"], m1[m1.set == "control"], items),
            S8_dissociation=breakdown(m1[m1.set == "intact"], m1[m1.set == "main"], items)),
        margin_scaling=dict(K8=scaling(old_m1, "main", items), C4=scaling(old_m1, "control", items),
                            S8=scaling(m1, "main", items), C4s=scaling(m1, "control", items)),
        naming_frequency_vs_difficulty=dict(
            zipf_median=median,
            K8=frequency_vs_difficulty(naming_old, "main", knowledge_items, median),
            C4=frequency_vs_difficulty(naming_old, "control", knowledge_items, median),
            S8=frequency_vs_difficulty(naming_new, "main", knowledge_items, median),
            C4s=frequency_vs_difficulty(naming_new, "control", knowledge_items, median)),
        lm_loss=dict(sets=lm["sets"], n_texts=lm["n_texts"], neutral_tokens=lm["neutral_tokens"],
                     loss={k: round(v, 4) for k, v in lm["lm_loss"].items()},
                     change={k: round(v / lm["lm_loss"]["intact"] - 1, 4) for k, v in lm["lm_loss"].items()}),
        inputs={str(p.relative_to(ROOT)).replace("\\", "/"): runinfo.sha256(p) for p in (
            STORE / "selection.json", STORE / "joint.json", NEW / "results.json", NEW / "sets.json",
            NEW / "m1_scores.parquet", SEMANTIC / "results.json", SEMANTIC / "intact.csv",
            SEMANTIC / "lesioned.parquet", NEW / "naming_scores.parquet", OLD / "m1_scores.parquet",
            OLD / "naming_scores.parquet", KNOWLEDGE_META, KNOWLEDGE_ITEMS, LM)},
    )
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
