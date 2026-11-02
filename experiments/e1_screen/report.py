"""MVP one-page report in plain Chinese (docs/plans/mvp.md §5; ticket pilot 11).

    python experiments/e1_screen/report.py --model <G1 version> --g1-result reports/g1_backup/g1_result.json

Reads outputs/e1_screen/<model>/{screen,joint}/ and writes reports/mvp/<model>/{report.md, verdict.json}.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import screen  # noqa: E402

DIRECTION = {"control": "控制", "knowledge": "知识"}
YES = {True: "满足", False: "不满足"}
KIND = {"expert": "专家", "head": "注意力头", "mlp": "MLP"}
OUTCOME = {
    "signal": ("想法有苗头", "开始编 M3 和校准题，准备在 M1 保留测试集上做核心检验"),
    "no_signal": ("测量没问题，但专家这个粒度没找到分工", "加筛注意力头，或改用层段作为单位，再决定是否编新题"),
    "g1_fail": ("这个模型没通过 G1（尺子不准），筛选结果不能用来判断想法", "按预实验方案处理：换备用模型，或按 decisions-pending 的决定走"),
}


def ci(e):
    return f"{e['estimate']:.2f} [{e['ci_low']:.2f}, {e['ci_high']:.2f}]"


def build(model, g1, screen_dir, joint_dir, tiny=False):
    summary = pd.read_csv(screen_dir / "screen.csv")
    selection = json.loads((screen_dir / "selection.json").read_text(encoding="utf-8"))
    res = json.loads((joint_dir / "results.json").read_text(encoding="utf-8"))
    iv, pts, rank = res["intervals"], res["points"], res["random_rank"]
    g1_decision = g1["g1_decision"] if g1["chosen_version"] == model else f"not_chosen ({g1['chosen_version']} was chosen)"
    verdict = screen.mvp_verdict(g1_decision, iv)
    title, next_step = OUTCOME[verdict["outcome"]]

    L = [f"# MVP 报告：{model}", ""]
    if tiny:
        L += ["> 本地检查：随机初始化的小模型，结果没有任何意义，只用来确认流程跑得通。", ""]
    L += [f"**结论：{title}。** 下一步：{next_step}。", "",
          "## 这份报告回答什么", "",
          "MVP 的第二个问题：模型里有没有这样的“专家”（OLMoE 每层 64 个小网络，每个词只交给其中 8 个处理）——"
          "把它关掉以后，主要伤害强干扰题（控制的苗头），或者主要伤害冷门词题（知识的苗头）。"
          "“关掉”是均值消融：把这个专家的输出换成它在普通文本上的平均输出，好比换上一个只会给平均答案的假零件。", "",
          "做法分两步，用的题互不重叠：先在 M1 的**筛选集**上逐个关掉每个专家，挑出每个方向最突出的 4 个；"
          "再在**开发集**上把这 4 个一起关掉，看效应还在不在，并和 20 组随机挑的专家比较。保留测试集没有碰。", "",
          "**损伤量 h**：同一道题，完好模型给正确选项的分数比错误选项高多少，关掉专家后这个差距缩小了多少（单位 logit，"
          "可以理解为模型“偏向正确答案的把握”）。h 越大，伤害越大。", "",
          "- **控制效应**：强干扰题的平均 h 减弱干扰题的平均 h（颜色、尺寸两个任务分别算，再平均）。",
          "- **知识效应**：同义判断题里，冷门词的平均 h 减常见词的平均 h。", "",
          f"判定（mvp.md 第 5 节）：G1 通过，并且至少一类专家在开发集上的效应点估计超过 {screen.THRESHOLD} logit，算“有苗头”。"
          f"方括号里是 95% 置信区间：按概念组重新抽样 {res['n_boot']} 次得到的范围，范围越窄、越远离 0，结果越可靠。", "",
          f"G1 结论：`{g1_decision}`。", "",
          "## 第一步：筛选集上挑出的零件", "",
          f"共核实了 {len(summary)} 个零件。每个零件先按两个效应的标准分归到较突出的一边，再各取原始效应最大的 {selection['k']} 个。", "",
          "| 方向 | 零件（种类 层:编号） | 控制效应 | 知识效应 | 语义题平均损伤 | 字符串题平均损伤 |", "| --- | --- | --- | --- | --- | --- |"]
    by_key = summary.set_index("component")
    for d in screen.PRIMARY:
        for k in selection.get(d, []):
            r = by_key.loc[k]
            L.append(f"| {DIRECTION[d]} | {KIND.get(r.kind, r.kind)} {r.layer}:{r['index']} | {r.control_effect:.2f} | {r.knowledge_effect:.2f} | "
                     f"{r.semantic_damage:.2f} | {r.string_damage:.2f} |")
    q = summary[["control_effect", "knowledge_effect"]].quantile([0.5, 0.9, 0.99]).round(2)
    L += ["", f"被核实零件的分布：控制效应中位数 {q.control_effect[0.5]}、前 10% 分界 {q.control_effect[0.9]}、前 1% 分界 {q.control_effect[0.99]}；"
          f"知识效应中位数 {q.knowledge_effect[0.5]}、前 10% 分界 {q.knowledge_effect[0.9]}、前 1% 分界 {q.knowledge_effect[0.99]}。"
          f"单个零件的控制效应超过 {screen.THRESHOLD} 的有 {(summary.control_effect > screen.THRESHOLD).sum()} 个，"
          f"知识效应超过的有 {(summary.knowledge_effect > screen.THRESHOLD).sum()} 个。筛选集上挑出来的最大值会偏高"
          "（从一千多个里挑最大的，运气成分也被挑进来了），所以要到开发集上再看。", "",
          "## 第二步：开发集上一起关掉这 4 个", "",
          f"开发集 {res['n_items']} 道题、{res['n_concept_groups']} 个概念组。", "",
          "| 关掉的零件组 | 控制效应 | 知识效应 | 语义题平均损伤 | 字符串题平均损伤 |", "| --- | --- | --- | --- | --- |"]
    for d in screen.PRIMARY:
        if f"{d}.control_effect" in iv:
            L.append(f"| {DIRECTION[d]}组 | {ci(iv[f'{d}.control_effect'])} | {ci(iv[f'{d}.knowledge_effect'])} | "
                     f"{ci(iv[f'{d}.semantic_damage'])} | {ci(iv[f'{d}.string_damage'])} |")
    if "delta_C" in iv:
        L += ["", "两类专家组的直接比较（spec 中 H2 的两个差值，这里只在开发集上预看）：",
              f"- 控制效应，控制组减知识组：{ci(iv['delta_C'])}",
              f"- 知识效应，知识组减控制组：{ci(iv['delta_F'])}"]
    L += ["", "## 和随机零件比较（只报告，不作门槛）", "",
          f"随机组：每个被选中的零件配一个同一层的同类零件（专家还要求在普通文本上被调用的频率相近），每个方向 {res['n_random']} 组。"
          "如果随机组也常常达到同样的效应，说明被选中的专家并不特别。", "",
          "| 方向 | 选中组的效应 | 随机组效应（中位数，最小至最大） | 随机组达到或超过选中组的比例 |", "| --- | --- | --- | --- |"]
    for d, e in screen.PRIMARY.items():
        if d in rank:
            rand = pd.Series([v[e] for n, v in pts.items() if n.startswith(f"random_{d}_")])
            L.append(f"| {DIRECTION[d]} | {pts[d][e]:.2f} | {rand.median():.2f}（{rand.min():.2f} 至 {rand.max():.2f}） | "
                     f"{rank[d]['share_at_least_target']:.0%}（共 {rank[d]['n_random']} 组） |")
    sel_supp = selection.get("supplementary")
    if sel_supp:
        excluded = sel_supp.get("excluded_domain_general", [])
        L += ["", "## 补充：挑控制型零件时不排除“领域一般”的零件（ADR 0005）", "",
              "主分析挑控制型零件时，排除了对字符串题（和词义无关的格式题）伤害大于自身控制效应的零件，"
              f"共 {len(excluded)} 个" + (f"，例如 {'、'.join(excluded[:5])}" if excluded else "") + "。"]
        supp = res.get("supplementary_unfiltered_control")
        if supp:
            si = supp["intervals"]
            L.append(f"不排除时挑出的控制组（{'、'.join(supp['components'])}）在开发集上："
                     f"控制效应 {ci(si['control.control_effect'])}，字符串题平均损伤 {ci(si['control.string_damage'])}，"
                     f"随机组达到或超过的比例 {supp['random_rank']['control']['share_at_least_target']:.0%}"
                     f"（共 {supp['random_rank']['control']['n_random']} 组）。")
        else:
            L.append("排除之后挑出的控制组和不排除时相同，所以没有单独的补充结果。")
    L += ["", "## 需要说明的简化", "",
          "- 随机组只按层位和路由频率匹配；spec 要求的输出范数和语言建模损失匹配留到正式的实验二。",
          f"- G2 在 spec 中还要求“另一效应的区间上界低于 {screen.PRESERVED}”：控制组{YES[verdict['g2_preserved_below']['control']]}，"
          f"知识组{YES[verdict['g2_preserved_below']['knowledge']]}。MVP 的结论不看这一条。",
          "- 开发集题目少，区间较宽；MVP 只回答“有没有苗头”，确认性的结论要等保留测试集。"]
    return "\n".join(L) + "\n", verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--g1-result", required=True, help="g1_result.json of the G1 run that chose this version")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    base = ROOT / ("outputs/e1_screen_tiny" if args.tiny else "outputs/e1_screen") / args.model
    g1 = json.loads(Path(args.g1_result).read_text(encoding="utf-8"))
    text, verdict = build(args.model, g1, base / "screen", base / "joint", args.tiny)
    out = ROOT / ("reports/mvp_tiny" if args.tiny else "reports/mvp") / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{verdict['outcome']} -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
