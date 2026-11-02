"""GPU session 11, second model (Phi-3.5-MoE-instruct), exploratory: one report of the replication run
(experiments/second_model/run_phi_replication.sh) and of the knowledge-set check run after it (language-model loss on
neutral text, naming split into the word and the end-of-turn token, greedy answers). No GPU:

    python experiments/second_model/analyze_phi.py [--out reports/explore/phi_replication.json]

The control side failed the measurement check (reports/g1_phi), so nothing here is confirmatory. For the comparison
of the control sets' damage by task and control demand, the main model's control test (C4, held-out items) is broken
down the same way (lesion.damage.demand_breakdown).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import damage, runinfo, scoring  # noqa: E402

PHI, QWEN = "phi35-moe-instruct", "qwen3-30b-a3b-instruct"
SCREEN = ROOT / "outputs/e1_screen" / PHI
KNOWLEDGE = ROOT / "outputs/e3_knowledge" / PHI
HELD_OUT = ROOT / "outputs/e2_confirm" / PHI
CHECK = ROOT / "outputs/explore_s11/phi_k8_check.json"
QWEN_CONTROL = ROOT / "outputs/server/e2_confirm" / QWEN / "control"
KEYS = ("knowledge_effect", "naming_damage", "control_effect", "low_demand_damage", "semantic_damage", "string_damage")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def interval(v):
    return [round(v["estimate"], 3), round(v["ci_low"], 3), round(v["ci_high"], 3)]


def key_intervals(s):
    """estimate and 95% interval of the main statistics of one set in a dissociation-style result."""
    out = {}
    for part in ("naming", "m1"):
        for k, v in s.get(part, {}).items():
            if isinstance(v, dict) and "estimate" in v and k.split(".")[-1] in KEYS:
                out[f"{part}.{k}"] = interval(v)
    return out


def precision_checks():
    out = {}
    for tag in ("fp16", "bf16"):
        c = read(SCREEN / f"check_prefix_failed_{tag}/check_prefix.json")
        lesions = c["lesions"]
        out[tag] = dict(passed=c["pass"], noise_floor_intact=round(lesions["intact"]["noise_floor_mean_abs"], 3),
                        lesions_passed=sum(bool(v["pass"]) for v in lesions.values()), lesions=len(lesions))
    return out


def breakdown(folder, set_name="control"):
    """Damage of one control set by task and demand, on the held-out M1 items its control test scored."""
    items = scoring.load_materials()[0]
    intact = pd.read_csv(folder / "intact.csv", dtype={"item_id": str})
    lesioned = pd.read_parquet(folder / "lesioned.parquet")
    lesioned = lesioned[lesioned.set == set_name].astype({"item_id": str})
    t = damage.demand_breakdown(intact, lesioned, items)
    return t.round(3).to_dict(orient="records")


def knowledge_check():
    c = read(CHECK)
    intact = c["lm_loss"]["intact"]
    naming = {}
    for state, rows in c["naming"].items():
        r = pd.DataFrame(rows)
        starts = r.apply(lambda x: x.greedy.strip().lower().startswith(x.target.lower()), axis=1)
        naming[state] = dict(word_logprob=round(r.word_logprob.mean(), 3), end_logprob=round(r.end_logprob.mean(), 3),
                             greedy_starts_with_target=round(float(starts.mean()), 3),
                             examples=r[["target", "greedy"]].head(8).to_dict(orient="records"))
    return dict(n_texts=c["n_texts"], neutral_tokens=c["neutral_tokens"], n_naming=c["n_naming"],
                lm_loss={k: round(v, 4) for k, v in c["lm_loss"].items()},
                lm_loss_change={k: round(v / intact - 1, 4) for k, v in c["lm_loss"].items()}, naming=naming)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports/explore/phi_replication.json"))
    args = ap.parse_args()
    control_sel = read(SCREEN / "screen/selection.json")
    knowledge_sel = read(KNOWLEDGE / "screen/selection.json")
    dissociation = read(HELD_OUT / "dissociation/results.json")
    k8 = dissociation["sets"]["main"]["components"]
    excluded = control_sel["supplementary"]["excluded_domain_general"]
    expand = read(KNOWLEDGE / "expand8_dev120/results.json")
    control = read(HELD_OUT / "control/results.json")
    inputs = [SCREEN / "screen/selection.json", SCREEN / "joint/results.json", KNOWLEDGE / "check/summary.json",
              KNOWLEDGE / "screen/selection.json", KNOWLEDGE / "expand8_dev120/results.json",
              HELD_OUT / "control/results.json", HELD_OUT / "dissociation/results.json", CHECK,
              HELD_OUT / "control/intact.csv", HELD_OUT / "control/lesioned.parquet",
              QWEN_CONTROL / "intact.csv", QWEN_CONTROL / "lesioned.parquet"]
    report = dict(
        status="exploratory: the control side failed G1 (reports/g1_phi); the knowledge side passed its item check",
        precision_checks=precision_checks(),
        precision_used="float16 (PHI_PRECISION; docs/plans/gpu-session11.md)",
        control_set=control_sel["control"],
        control_dev=dict(control_effect=round(read(SCREEN / "joint/results.json")["points"]["control"]["control_effect"], 3)),
        knowledge_item_check=read(KNOWLEDGE / "check/summary.json"),
        knowledge_k4=knowledge_sel["knowledge"], knowledge_k8=k8,
        k8_in_control_screen_domain_general_list=[c for c in k8 if c in excluded],
        knowledge_dev120={name: key_intervals(s) for name, s in expand["sets"].items()},
        control_test=dict(intervals={k: interval(v) for k, v in control["intervals"].items()},
                          random=control["random"], verdict=control["primary"]["verdict"],
                          intact_tds_effect=round(control["relative_to_intact"]["intact_tds_effect"], 3)),
        dissociation_test=dict(sets={name: key_intervals(s) for name, s in dissociation["sets"].items()},
                               random=dissociation["random"], h2=dissociation["h2"]),
        control_by_demand=dict(phi=breakdown(HELD_OUT / "control"), qwen_C4=breakdown(QWEN_CONTROL)),
        knowledge_check=knowledge_check(),
        inputs={str(p.relative_to(ROOT)).replace("\\", "/"): runinfo.sha256(p) for p in inputs},
    )
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
