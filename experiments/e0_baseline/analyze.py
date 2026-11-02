"""G1 report for Experiment 0 (docs/plans/pilot-protocol.md §5-7; ticket pilot 05 / 08).

Run from the repo root after run.py:  python experiments/e0_baseline/analyze.py
Writes reports/g1/g1_result.json and reports/g1/report.md (plain Chinese, for the user).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import g1, scoring  # noqa: E402

DECISION_TEXT = {
    "pass": "通过：尺子是准的，进入下一步（找零件）。",
    "insufficient": "证据不足：效应的范围跨过了 0.20 的门槛。按方案执行一次预先规定的补充：加两个换了措辞的提示模板，三个模板平均后重新判定。",
    "exclude": "排除：这个模型加这套题测不到控制需求。按方案换备用模型 OLMoE-0125 再试一次。",
    "measurement_fail": "测量不合格：难题正确率或换顺序后的一致率没达标。按方案换备用模型 OLMoE-0125 再试一次。",
}


def fmt(ci):
    return f"{ci['estimate']:.3f}（95% 区间 {ci['ci_low']:.3f} 至 {ci['ci_high']:.3f}）"


def report_md(res):
    chosen = res["chosen_version"]
    v = res["versions"][chosen]
    m = v["matching"]
    lines = [
        "# 预实验（实验零）G1 报告",
        "",
        f"**结论：{DECISION_TEXT[res['g1_decision']]}**",
        "",
        "## 选了哪个版本",
        "",
        f"选定版本：`{chosen}`。挑版本只看字符串题（比较两串无意义字母，和词义无关），看模型能不能看懂二选一的答题格式；"
        f"字符串题格式检查{'通过' if res['string_format_check_passed'] else '两个版本都没通过，选了相对较好的一个'}。",
        "",
        "| 版本 | 字符串题正确率 | 字符串题换顺序一致率 |",
        "| --- | --- | --- |",
    ]
    for name, vv in res["versions"].items():
        lines.append(f"| {name} | {vv['string']['accuracy']:.3f} | {vv['string']['order_consistency']:.3f} |")
    lines += [
        "",
        "## G1 的三条标准（只看选定版本的颜色题和尺寸题）",
        "",
        "| 标准 | 结果 | 门槛 |",
        "| --- | --- | --- |",
        f"| 干扰效应：难题比易题的 logit 差低多少 | {fmt(m['tds_effect'])} | 区间下界高于 0.20 |",
        f"| 难题正确率 | {fmt(m['high_demand_accuracy'])} | 不低于 0.70，且区间下界高于 0.50 |",
        f"| 换选项顺序后答案不变的比例 | {m['order_consistency']:.3f} | 不低于 0.90 |",
        "",
        f"颜色题正确率 {m['accuracy_by_task']['colour']:.3f}，尺寸题正确率 {m['accuracy_by_task']['size']:.3f}；"
        f"尺寸维度{'保留' if v['size_dimension_kept'] else '不保留（正确率低于 0.70），新题改用颜色和材质'}。",
        "",
        "## 只报告、不参与判定的结果",
        "",
        f"- 同义判断的词频效应（词越常见，模型越有把握的程度）：{fmt(v['synonym_frequency'])}；"
        f"模型整体偏向回答“意思相近”的程度为 {v['synonym_frequency']['similar_bias']:.3f} logit，已扣除。",
        "- 和人类难易的一致性（逐题 Spearman 相关，正数表示人觉得容易的题，模型也更有把握）：",
    ]
    for task, h in v["human_agreement"].items():
        lines.append(f"  - {task}：{h['spearman']:.3f}（n = {h['n']}，p = {h['p']:.3g}）")
    if v["cloze_tds_effect"]:
        lines.append(f"- 完形填空计分下的干扰效应：{fmt(v['cloze_tds_effect'])}")
    lines += ["", f"TDS 两半均值之差（换算系数）：{res['tds_gap']:.4f}。区间按概念组整群重抽样 {g1.N_BOOT} 次。"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs/e0"))
    ap.add_argument("--variant", default="main")
    ap.add_argument("--report-dir", default=str(ROOT / "reports/g1"))
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
        sys.exit(f"no label scores under {args.outputs}; run run.py first")

    res = g1.analyze(labels, items, cloze or None)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g1_result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(report_md(res), encoding="utf-8")
    print(report_md(res))


if __name__ == "__main__":
    main()
