"""G1 report under the revised rules (docs/adr/0003-qwen-and-revised-g1.md).

Scoring is unchanged: run experiments/e0_baseline/run.py at the `g1-revised-frozen` tag, then
    python experiments/e0_revised/analyze.py --outputs outputs/e0_qwen --report-dir reports/g1_qwen
Writes <report-dir>/g1_result.json and report.md (plain Chinese).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import g1, g1_revised, scoring  # noqa: E402

DECISION_TEXT = {
    "pass": "通过：按修订后的门槛，尺子是准的，进入下一步（找零件）。",
    "insufficient": "证据不足：干扰效应的区间跨过了 0.20 的门槛。",
    "exclude": "排除：这个模型加这套题测不到控制需求。",
    "measurement_fail": "测量不合格：两种选项顺序平均后，难题答对比例的区间没有整体高于瞎猜（50%）。",
}


def fmt(ci):
    return f"{ci['estimate']:.3f}（95% 区间 {ci['ci_low']:.3f} 至 {ci['ci_high']:.3f}）"


def report_md(res):
    chosen = res["chosen_version"]
    v = res["versions"][chosen]
    m = v["matching"]
    lines = [
        "# 预实验（实验零）G1 报告：修订门槛",
        "",
        f"**结论：{DECISION_TEXT[res['g1_decision']]}**",
        "",
        "按 2026-09-28 的用户决定（`docs/adr/0003-qwen-and-revised-g1.md`）：原定的 OLMoE-0924 和备用的 OLMoE-0125 都没通过原来的 G1，"
        "主模型改为 Qwen3-30B-A3B-Instruct-2507，门槛同时修订。修订是在看过这些结果之后做的，论文中如实说明。"
        f"按原门槛，这个模型的判定是 `{v['frozen_rule_decision']}`。",
        "",
        f"模型版本：`{chosen}`（字符串题正确率 {v['string']['accuracy']:.3f}，换顺序一致率 {v['string']['order_consistency']:.3f}）。",
        "",
        "## G1 的标准（只看颜色题和尺寸题）",
        "",
        "| 标准 | 结果 | 门槛 |",
        "| --- | --- | --- |",
        f"| 干扰效应：难题比易题的 logit 差低多少 | {fmt(m['tds_effect'])} | 区间下界高于 0.20（未改） |",
        f"| 难题答对比例（两种选项顺序的 logit 差平均后大于 0 算对） | {fmt(v['order_averaged_high_demand_accuracy'])} | 区间下界高于 0.50（修订） |",
        "",
        "## 只报告、不参与判定的结果",
        "",
        f"- 难题逐条正确率（原门槛用的算法）：{fmt(m['high_demand_accuracy'])}",
        f"- 换选项顺序后答案不变的比例：{m['order_consistency']:.3f}（原门槛要求不低于 0.90，修订后只报告）",
        f"- 颜色题正确率 {m['accuracy_by_task']['colour']:.3f}，尺寸题正确率 {m['accuracy_by_task']['size']:.3f}",
        f"- 同义判断的词频效应：{fmt(v['synonym_frequency'])}；偏向回答“意思相近”的程度 {v['synonym_frequency']['similar_bias']:.3f} logit，已扣除",
        "- 和人类难易的一致性（逐题 Spearman 相关）：",
    ]
    for task, h in v["human_agreement"].items():
        lines.append(f"  - {task}：{h['spearman']:.3f}（n = {h['n']}，p = {h['p']:.3g}）")
    if v["cloze_tds_effect"]:
        lines.append(f"- 完形填空计分下的干扰效应：{fmt(v['cloze_tds_effect'])}")
    lines += ["", f"TDS 两半均值之差（换算系数）：{res['tds_gap']:.4f}。区间按概念组整群重抽样 {g1.N_BOOT} 次。"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--variant", default="main")
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    items = scoring.load_materials()[0]
    labels, cloze = {}, {}
    for d in sorted(Path(args.outputs).iterdir()):
        f = d / f"label_{args.variant}.csv"
        if d.name != "smoke" and f.exists():
            labels[d.name] = pd.read_csv(f, dtype={"item_id": str})
            if (d / "cloze.csv").exists():
                cloze[d.name] = pd.read_csv(d / "cloze.csv", dtype={"item_id": str})
    if not labels:
        sys.exit(f"no label scores under {args.outputs}; run experiments/e0_baseline/run.py first")

    res = g1_revised.analyze(labels, items, cloze or None)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g1_result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(report_md(res), encoding="utf-8")
    print(report_md(res))


if __name__ == "__main__":
    main()
