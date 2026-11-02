"""Ledger-ready form of the 2026-11-02 GPU-session analyses (experiments/analyze_gpu_20261102.py): one node per
reported value with estimate, 95% interval, unit and sample size.

    python reports/explore/2026-11-02-gpu-session/make_ledger_json.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def node(estimate, ci=None, unit="", n=None):
    out = dict(estimate=estimate, unit=unit, n=n)
    if ci is not None:
        out["ci"] = ci
    return out


def generated(g):
    out = dict(accuracy={}, frequency_effect={})
    for s, a in g["accuracy"].items():
        out["accuracy"][s] = {k: node(v, unit="比例（0–1）", n=g["n_concepts"] if k == "all" else g["n_rare"]) for k, v in a.items()}
    for s, e in g["frequency_effect"].items():
        out["frequency_effect"][s] = node(e["mean"], e["ci"], "正确率差（罕见减常见）", sum(e["n"]))
    k = next(k for k in g if k.startswith("frequency_effect_") and k != "frequency_effect")
    out[k] = node(g[k]["mean"], g[k]["ci"], "正确率差（罕见减常见）", sum(g[k]["n"]))
    return out


def heads(h):
    out = dict(observed={k: node(v["mean"], v["ci"], "log odds（nat）", v["n"]) for k, v in h["observed_stats"].items()},
               random={})
    for k, v in h["random"].items():
        out["random"][k] = dict(observed=node(v["observed"], unit="log odds（nat）", n=h["n_concepts"]),
                                random_median=node(v["random_median"], unit="log odds（nat）", n=h["n_concepts"]),
                                random_max=node(v["random_max"], unit="log odds（nat）", n=h["n_concepts"]),
                                n_random_at_least_observed=node(v["n_random_at_least_observed"], unit="随机组件组数", n=h["n_random"]),
                                p_upper=node(v["p_upper"], unit="经验 p（单侧）", n=h["n_random"]))
    return out


UNIT = dict(naming="log probability（nat）", word_shown="log probability（nat）", copy="log probability（nat）",
            six_choice="logit", comprehension="logit", B="log odds（nat）", H="log odds（nat）", H_minus_B="log odds（nat）")


def phi(d):
    out = dict(n_valid=node(d["n_valid"], unit="概念数", n=d["n_concepts"]),
               intact={k: node(v, unit="比例（0–1）", n=d["n_concepts"]) for k, v in d["intact"].items()},
               lm_loss_relative={s: node(v, unit="相对语言损失变化（比例，0.1=10%）", n=40) for s, v in d["lm_loss_relative"].items()
                                 if v is not None and s != "intact"},
               subsets={})
    for sub, s in d["subsets"].items():
        losses = {st: {m: node(v["mean"], v["ci"], UNIT[m], v["n"]) for m, v in L.items()} for st, L in s["losses"].items()}
        rules = {}
        for r, v in s["rules"].items():
            rules[r] = {m: (node(x["mean"], x["ci"], "nat 或 logit（见规则）", x["n"]) if isinstance(x, dict) else
                            node(int(bool(x)), unit="判定（1=一致）", n=s["n"])) for m, x in v.items() if m != "margin"}
        out["subsets"][sub] = dict(losses=losses, rules=rules)
    out["generated"] = generated(d["R7_generated"])
    out["random_heads"] = heads(d["R4_random_heads"])
    out["output_bias_check"] = {lab: {st: {m: node(v, unit="log probability（nat）", n=240) for m, v in x.items()}
                                      for st, x in states.items()} for lab, states in d["output_bias_check"].items()}
    return out


if __name__ == "__main__":
    for src, fn, dst in (("generated-frequency.json", generated, "ledger-generated-frequency.json"),
                         ("random-answer-heads.json", heads, "ledger-random-answer-heads.json"),
                         ("phi-battery.json", phi, "ledger-phi-battery.json")):
        path = HERE / src
        if not path.exists() or fn is None:
            continue
        (HERE / dst).write_text(json.dumps(fn(json.loads(path.read_text(encoding="utf-8"))), indent=1, ensure_ascii=False),
                                encoding="utf-8")
        print("written", dst)
