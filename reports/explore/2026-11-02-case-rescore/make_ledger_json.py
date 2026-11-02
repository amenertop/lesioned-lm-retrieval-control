"""Ledger-ready form of the case-insensitive family scores (case-rescore.json) and the demonstration-format check
(../2026-11-02-gpu-session/demo-format-case.json): one node per reported value with its estimate, 95% interval, unit
and sample size, so that build_ledger.py reads them like the other sources.

    python reports/explore/2026-11-02-case-rescore/make_ledger_json.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNITS = {"naming": "log probability（nat）", "definition": "logit", "synonym": "logit",
         "naming_definition_partial_r": "相关系数", "naming_synonym_partial_r": "相关系数",
         "naming_frequency_slope": "nat / log10 BNC 词频"}


def node(estimate, ci, unit, n):
    return dict(estimate=estimate, ci=ci, unit=unit, n=n) if ci else dict(estimate=estimate, unit=unit, n=n)


def case():
    d = json.loads((HERE / "case-rescore.json").read_text(encoding="utf-8"))
    out = dict(scoring="log(P(word) + P(Word))，家族命名两种拼写合并", runs={})
    for run, r in d["runs"].items():
        states = {}
        for state, v in r["states"].items():
            ci = v["case_insensitive"]
            states[state] = {m: node(ci[m], ci.get(m + "_ci"), UNITS[m], 120 if m != "naming_frequency_slope" else 120)
                             for m in UNITS if m in ci}
        contrasts = {}
        for pair, c in r["contrasts"]["case_insensitive"].items():
            entry = {}
            for metric, val in c.items():
                if metric == "semd":
                    continue
                unit = "log probability（nat）" if metric.startswith("naming_damage") else (
                    "相关系数差" if "consistency" in metric else "logit")
                entry[metric] = node(val["estimate"], val["ci"], unit, 44 if metric.startswith("property") else 120)
            if "semd" in c:
                entry["semd_slopes"] = {
                    task: {f"{x['state']}_{x['predictor']}": node(x["estimate"], [x["ci_low"], x["ci_high"]],
                                                                  ("nat" if task == "naming" else "logit") + " / 语义多样性标准差",
                                                                  44 if task == "property" else 120)
                           for x in s["coefficients"] if x["predictor"] == "semd"}
                    for task, s in c["semd"].items()}
            contrasts[pair] = entry
        out["runs"][run] = dict(
            intact_capitalized_share=node(r["intact_capitalized_share"], None, "比例（0–1）", 120),
            capitalized_share={s: node(v, None, "比例（0–1）", 120) for s, v in r["capitalized_share_by_state"].items()
                               if s in ("intact", "WR8_a0", "SC4_a0", "hub_s2_learned_soft")},
            states=states, contrasts=contrasts)
    return out


def demo():
    d = json.loads((HERE.parent / "2026-11-02-gpu-session/demo-format-case.json").read_text(encoding="utf-8"))
    out = {}
    for cell, v in d.items():
        if cell == "reading_rules":
            continue
        key = cell.replace("/", "__")
        if cell.endswith("capitalized_share"):
            out[key] = {s: node(x, None, "比例（0–1）", 120) for s, x in v.items()}
        else:
            out[key] = {s: node(x["mean"], x["ci"], "log probability（nat）", x["n"]) for s, x in v.items()}
    return out


if __name__ == "__main__":
    c = case()
    (HERE / "case-ledger.json").write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    (HERE / "demo-format-ledger.json").write_text(json.dumps(demo(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("written")
