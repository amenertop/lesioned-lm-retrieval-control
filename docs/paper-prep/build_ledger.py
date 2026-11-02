"""Build the paper's numeric ledger from existing JSON reports, without rerunning analyses.

Run with the project's lesion Python. --check verifies the checked-in CSV against
the reports. JSON Pointer (RFC 6901) fields identify the exact unrounded source
value, interval, and p-value. Empty cells mean not supplied, never zero.
Only Python's standard library is required. This script does not modify reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("results_ledger.csv")
MANIFEST = Path(__file__).with_name("results_ledger_manifest.json")
INPUT_HASHES = {}
SOURCES = {
    "control": ("reports/confirm/control/results.json", "控制侧正式检验", "M1 test"),
    "knowledge": ("reports/confirm/knowledge/results.json", "知识侧正式检验", "知识 test / M1 test"),
    "experiment3": ("reports/confirm/experiment3/results.json", "实验三正式检验", "M3 test"),
    "g1_qwen": ("reports/g1_qwen/g1_result.json", "Qwen G1 测量检查", "M1 测量检查"),
    "g1_olmoe": ("reports/g1/g1_result.json", "OLMoE 历史测量检查", "历史 M1 测量检查"),
    **{f"association_{s}": (f"reports/explore/qwen_control_association_{s}.json", f"联想效应 {s}", f"M1 {s}")
       for s in ("screening", "dev", "heldout")},
    **{s: (f"reports/explore/qwen_m3_{s}.json", label, "M3 calibration")
       for s, label in (("battery", "校准跨格式测验"), ("random_controls", "校准随机组件对照"),
                        ("dla", "校准直接归因"), ("intrusion", "校准上下文干扰"))},
    "zero": ("reports/explore/qwen_knowledge_expand8_zero_ablation.json", "清零消融稳健性", "知识 dev / M1 dev"),
}
NEW_SOURCES = {
    "title_gap_fine_final_semd": ("reports/explore/2026-10-19-title-gap-execution/fine-final-semd.json", "补实验丙的最终语义多样性回归", "独立最终概念，六对冻结比较，配对回归已完成，随机排名按用户决定不运行"),
    "title_gap_coarse_final_core": ("reports/explore/2026-10-19-title-gap-execution/coarse-final-core.json", "补实验己的最终粗损伤核心读数", "独立最终概念，九个粗损伤的冻结状态统计及与 SC4 的配对差，匹配随机排名不运行"),
    "title_gap_fine_remaining_core": ("reports/explore/2026-10-19-title-gap-execution/fine-remaining-core-20261020.json", "补实验甲的其余原定核心读数", "独立最终概念，渐进、联合、同大小与直接配对比较，状态冻结，甲随机族按用户决定不运行"),
    "title_gap_fine_core_readout": ("reports/explore/2026-10-19-title-gap-execution/fine-core-readout-20261020.json", "补实验甲的最终核心读数", "独立最终概念，按用户决定提前读取核心以选择后续随机族，匹配随机比较未完成"),
    "title_gap_clinical_scale_sensitivity": ("reports/explore/2026-10-19-title-gap-execution/clinical-calibration-sensitivity.json", "补实验丁固定状态的标定敏感性", "事后描述性点估计，保持原开发选择，不重选状态、不重抽样"),
    "title_gap_clinical_final_calibration": ("reports/explore/2026-10-19-title-gap-execution/clinical-final-calibration.json", "补实验丁的最终原始词频与标定基准", "固定组级转移已有输出的描述性导出，不新增计分、选组或抽样"),
    "title_gap_clinical_final": ("reports/explore/2026-10-19-title-gap-execution/clinical-final-profile-comparison.json", "补实验丁的固定临床组级转移", "开发选择固定，历史 M3 test 提示与命名错误，加独立最终概念的同义词频标记"),
    "title_gap_relative_dev": ("reports/explore/2026-10-19-title-gap-execution/relative-hub-dev.json", "补实验甲候选竞争目标的开发实际作答", "全部预定候选的开发点估计，遮蔽审核保留未定判断及分类界限"),
    "title_gap_relative_readouts_dev": ("reports/explore/2026-10-19-title-gap-execution/relative-hub-readout-diagnostic.json", "补实验甲候选竞争目标的开发读数", "固定选择后的两个候选，三套读数的开发点估计，不是最终检验"),
    "title_gap_clinical_final_cues": ("reports/explore/2026-10-19-title-gap-execution/clinical-final-cues.json", "补实验丁的最终提示与错误读数", "M3 test，历史用过的固定资格概念，遮蔽二元裁决与错误子类不变性分析"),
    "title_gap_answer_completion": ("reports/explore/2026-10-19-title-gap-execution/answer-completion-20261021.json", "补实验甲与己的实际命名作答端点补齐", "120 个独立最终概念，13 个弃权回答经遮蔽二元裁决后用原冻结函数重算，一万次抽样，种子 1910"),
    "title_gap_homonyms_final": ("reports/explore/2026-10-19-title-gap-execution/homonyms-final-ledger.json", "补实验乙的词义产出最终检验", "40 个最终词义家族，15 个核心状态与 200 组匹配随机，遮蔽审核与分类界限；随机组逐组读数见 homonyms-final.json"),
    "title_gap_clinical_cue_search": ("reports/explore/2026-10-19-title-gap-execution/clinical-cue-search.json", "补实验甲三个搜索候选的提示测验", "M3 test，与丁相同的 132 个概念和冻结临床入口，回答全部遮蔽分类"),
    "title_gap_learned_dev": ("reports/explore/2026-10-19-title-gap-execution/learned-dev.json", "补实验甲的完整候选开发比较", "全部六个预定起点，开发点估计与遮蔽回答审核"),
    "title_gap_learned_readouts": ("reports/explore/2026-10-19-title-gap-execution/learned-readout-sensitivity.json", "补实验甲的候选读数敏感性", "原开发输出的离线诊断，沿用原尺度，不替换原选择"),
    "title_gap_search_stability": ("reports/explore/2026-10-19-title-gap-execution/search-stability.json", "补实验甲的搜索起点与资源实现稳定性描述", "同一模型与找组材料，两个已记录执行版本，不是功能复现"),
    "title_gap_clinical_directions_dev": ("reports/explore/2026-10-19-title-gap-execution/clinical-directions-dev.json", "补实验丁的五族方向开发比较", "相同实际损伤在各开发任务内配对，尚无五族完整随机参照"),
    "title_gap_clinical_extended_dev": ("reports/explore/2026-10-19-title-gap-execution/clinical-extended-dev.json", "补实验丁的扩展临床开发比较", "M3 dev 与新概念 dev，加入强剂量与粗损伤后留出整族标记"),
    "title_gap_cue_content_dev": ("reports/explore/2026-10-19-title-gap-execution/cue-content-dev.json", "补实验的提示内容诊断", "M3 dev，原定义、移除定义与直接补词配对比较"),
    "title_gap_coarse_fine_dev": ("reports/explore/2026-10-19-title-gap-execution/coarse-fine-dev.json", "补实验己的粗损伤新概念开发比较", "新概念 dev，遮蔽回答审核"),
    "title_gap_coarse_property_dev": ("reports/explore/2026-10-19-title-gap-execution/coarse-property-dev.json", "补实验己的粗损伤属性诊断", "新概念 dev，开发后真伪分解"),
    "title_gap_coarse_specificity_dev": ("reports/explore/2026-10-19-title-gap-execution/coarse-fine-specificity.json", "补实验己的粗损伤候选词竞争", "新概念 dev，开发后分析已有分数"),
    "title_gap_calibrated_search": ("reports/explore/2026-10-19-title-gap-execution/calibrated-search.json", "补实验甲两种搜索的首起点校准", "找组集优化，尚不是开发选择或最终检验"),
    "title_gap_precision_plan": ("reports/explore/2026-10-19-title-gap-execution/precision-plan.json", "补实验最终题量的设计投影", "开发家族噪声与相关检验近似，不是新模型结果"),
    "token_covariate": ("reports/explore/token-covariate-20261028/token-covariate.json", "WR8 词频效应的词元数协变量检验", "知识 test 240 个概念，确认性运行已存的逐概念分数离线重算，未运行模型，2026-10-28 用户同意"),
    "title_gap_production_dev": ("reports/explore/2026-10-19-title-gap-execution/production-dev.json", "补实验乙的独立产出控制开发比较", "开发词义家族二次划分，选组后遮蔽判断"),
    "title_gap_clinical_dev": ("reports/explore/2026-10-19-title-gap-execution/clinical-dev.json", "补实验丁的实际临床开发比较", "M3 dev 与新概念 dev，留出完整标记族"),
    "title_gap_model_census": ("reports/explore/2026-10-19-title-gap-execution/model-census.json", "补实验戊的模型测量普查", "固定版本，原格式与测量门槛"),
    "title_gap_property_diagnostic_dev": ("reports/explore/2026-10-19-title-gap-execution/property-diagnostic-dev.json", "补实验甲的属性判断与应答偏向诊断", "新概念 dev，开发后分析，不替换原终点"),
    "title_gap_fine_extended_dev": ("reports/explore/2026-10-19-title-gap-execution/fine-extended-dev.json", "补实验甲的强剂量与联合开发比较", "新概念 dev，遮蔽回答审核"),
    "title_gap_fine_specificity_dev": ("reports/explore/2026-10-19-title-gap-execution/fine-extended-specificity.json", "补实验甲的候选词竞争诊断", "新概念 dev，事后分析已有分数"),
    "title_gap_submatrix_dev": ("reports/explore/2026-10-19-title-gap-execution/submatrix-dev.json", "补实验己的子矩阵开发网格", "旧审核材料 dev，单批控制计分"),
    "title_gap_patient_markers": ("reports/explore/2026-10-19-title-gap-execution/clinical-marker-profiles.json", "补实验丁的三族病人参照", "三个已发表队列，按任务分母标定"),
    "title_gap_hub_calibration": ("reports/explore/2026-10-19-title-gap-execution/hub-calibration.json", "补实验甲的轮毂首起点校准", "找组概念，优化校准"),
    "title_gap_fine_wordformat_dev": ("reports/explore/2026-10-19-title-gap-execution/fine-wordformat-dev.json", "补实验甲的无字母标签词回答检查", "新概念 dev，格式修复后的三个状态"),
    "title_gap_homonyms_dev": ("reports/explore/2026-10-19-title-gap-execution/homonyms-dev.json", "补实验乙的词义产出开发比较", "词义家族 dev，遮蔽审核与分类界限"),
    "title_gap_fine_dev": ("reports/explore/2026-10-19-title-gap-execution/fine-dev.json", "补实验甲的细理解开发比较", "新概念 dev，开发点估计"),
    "title_gap_fine_dev_semd": ("reports/explore/2026-10-19-title-gap-execution/fine-dev-semd.json", "补实验丙的新概念开发比较", "新概念 dev，配对回归"),
    "title_gap_patient_profiles": ("reports/explore/2026-10-19-title-gap-execution/patient-profiles.json", "补实验丁的患者组均值与区间", "Jefferies 2006 表4/5，逐患者数据再分析"),
    "title_gap_coarse_dev": ("reports/explore/2026-10-19-title-gap-execution/coarse-dev.json", "补实验己的完整开发网格", "旧审核材料 dev，开发点估计"),
    "title_gap_semd": ("reports/explore/2026-10-19-title-gap-preparation/semd.json", "补实验丙的既有逐题探索", "M1/M3 既有 test，事后分析"),
    "title_gap_clinical": ("reports/explore/2026-10-19-title-gap-preparation/clinical.json", "补实验丁的逐患者资料", "Jefferies 2006 表4/5，资料转录"),
    "optogenetics": ("reports/confirm/optogenetics/result.json", "调光与恢复干预", "M3 test，按指标区分 160/100/132"),
    "case_rescore": ("reports/explore/2026-11-02-case-rescore/case-ledger.json", "家族命名两种拼写合并的重算", "120 个最终家族，事后计分修正，探索性"),
    "generated_frequency": ("reports/explore/2026-11-02-gpu-session/ledger-generated-frequency.json", "生成答案的词频效应", "知识测验 240 个保留概念，贪心生成，大小写不敏感判分，探索性"),
    "random_answer_heads": ("reports/explore/2026-11-02-gpu-session/ledger-random-answer-heads.json", "回答位置的随机四头对照", "M3 test 100 个提示概念，100 组同层随机头，探索性"),
    "phi_battery": ("reports/explore/2026-11-02-gpu-session/ledger-phi-battery.json", "Phi 跑主模型的临床测验", "Phi-3.5-MoE，M3 test 160 个概念（完好可做对的 156 个）与知识测验 240 个概念，探索性（第二模型，G1 不合格）"),
    "demo_format": ("reports/explore/2026-11-02-case-rescore/demo-format-ledger.json", "示范题格式检查（含两种拼写）", "题库 160 个封存概念与 120 个最终家族，探索性"),
    **{name: (f"reports/explore/{name}.json", title, dataset) for name, title, dataset in (
        ("qwen_s11_heldout", "第11次新组件检验", "Qwen，dev/test 分支见路径"),
        ("qwen_new_sets", "新组件病人式测验", "Qwen，事后新组件"),
        ("qwen_clinical_scorecard", "临床记分卡", "模型与文献记分，非独立重复实验"),
        ("session12", "第12次探索", "Phi/Qwen，子实验见路径"),
        ("session13", "第13次探索", "Qwen，子实验见路径"),
        ("breadth", "P13 扩大知识组件范围", "Qwen，探索性跨格式测验"),
        ("session14b", "第14b次探索", "Qwen，联合损伤与位置检查"),
        ("p14_session15", "P14 搜索与最终检查", "Qwen，select/confirm/final 见路径"),
        ("k30_synonym", "K30 同义词补充分析", "M1 test，事后分析"),
        ("phi_replication", "Phi 第二模型探索", "Phi，G1 测量检查不合格"),
    )},
    **{name: (f"reports/{name}/g1_result.json", title, "M1，开发阶段测量检查")
       for name, title in (("g1_phi", "Phi G1"), ("g1_deepseek", "DeepSeek G1"),
                           ("g1_gptoss_p12b", "gpt-oss P12b 格式检查后的事后重试"))},
}
SOURCES.update(NEW_SOURCES)
NEW_SKIP = {"meta", "inputs", "source_tables_sha256", "components", "members", "in_every_run",
            "knowledge_set", "knowledge_sets", "necessary", "greedy_examples_K8", "rows", "items",
            "n_boot", "n_valid", "development_families", "seed", "threshold", "margin", "equivalence_margin", "tds_gap",
            "lambda_sp", "exceed", "collapse", "n", "n_items", "n_concepts", "n_hint_concepts",
            "n_groups", "n_runs", "n_words", "n_texts", "n_naming", "neutral_tokens"}
TRIPLE_SOURCES = {"qwen_s11_heldout", "session12", "session13", "session14b", "phi_replication"}
# Sources whose report JSON was deleted from the repository by a user decision. Their rows are carried over verbatim
# from the checked-in ledger, so the ids that the manuscript cites stay valid, and they are checked against that copy.
ARCHIVED = {"g1_gptoss_p12b": "report deleted 2026-10-26 by user decision (commit 4a1b0a0)"}


def archived_rows(filename):
    if not OUT.exists():
        raise FileNotFoundError(f"{filename} is archived and no checked-in ledger holds its rows")
    with OUT.open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["来源文件"] == filename]


CODE_REVIEW = {
    "date": "2026-10-18",
    "reviewed_sha256": {
        "experiments/e0_baseline/run.py": "50bed06980f653162d2f4e1449489e3c15965c72f8ee76e55eadcd2e2dc91266",
        "experiments/e2_confirm/run_control_test.py": "69f8b99f5181877b8567245cce498ee8df4159eb29f22e743e2ce317bead267f",
        "src/lesion/scoring.py": "de3fc80f2e3eeefb36f2309a12a348665880519dddd874434336cacc24568b3b",
    },
    "decision": "三处代码变化不影响已入账历史报告数字，可以刷新当前代码指纹。当前哈希是审计时的依赖快照，不是历史运行所用代码。报告 JSON 的原值仍是数字来源。",
    "experiments/e0_baseline/run.py": "逐查 97f2aec、50a1c97、0d3a161、303933c、017cc4d。新增 P12 提示格式、格式门禁、tiny 及材料/variant 绑定，Qwen 原 chat_template 和旧 OLMo 提示分支保留，不改已保存 G1 报告。",
    "experiments/e2_confirm/run_control_test.py": "逐查 fc9c447、2f9de0a。2026-10-19 另核对 core_claim 合并判定，复用原主效应和斜率判据。相对修复前报告逐个比对既有数字，数值和区间未变。新增字段不改变历史检验。",
    "src/lesion/scoring.py": "逐查 97f2aec、5d33d1e、0d3a161、699a526、303933c。新格式和标签前缀用于 P12，旧 chat 分支不变，plain 的文档起始 token 对旧 OLMo 保留。next_token_logits 只投影末位置，continuation_logprob 只投影续写所需位置并平移索引，数学评分定义未变。未以当前实现重跑或覆盖历史数字，也不声称跨硬件逐位相同。",
    "regression": "2026-10-18，lesion Python：pytest tests/test_prompt_formats.py tests/test_prefix_scoring.py -q -p no:cacheprovider，15 passed。含提示格式、完整/局部投影与前缀共享评分回归。",
}
FIELDS = ["编号", "结论一句话", "数值", "95%区间下限", "95%区间上限", "单位", "n", "n单位", "n说明",
          "性质", "数据集", "指标路径", "来源文件", "来源字段路径", "区间字段路径", "来源SHA256",
          "p原始", "p校正", "p字段路径", "判定或检验族", "备注"]
SKIP = {"n", "n_items", "n_test", "n_calibration", "n_concepts", "n_concept_groups", "n_boot", "n_valid", "seed",
        "threshold", "noninferiority_bound", "tds_gap", "meta", "fits", "rules", "components", "control",
        "knowledge_sets", "h2", "descriptive", "primary", "association", "greedy_examples_K8", "greedy_n"}
# primary and control are meaningful data containers in some sources, handled below.
ANCILLARY = {"ci", "ci_low", "ci_high", "p", "p_one_sided", "p_holm", "empirical_p"}
LABELS = {
    "intervals": "效应", "sets": "组件组", "main": "K8 主组", "reported": "K4 补充组",
    "naming": "命名", "m1": "M1", "change": "损伤减完好", "report": "只报告",
    "primary": "主要检验", "secondary": "次要检验", "states": "状态", "intact": "完好",
    "control_effect": "强干扰减弱干扰损伤", "knowledge_effect": "低频减高频损伤",
    "control_slope": "控制需求斜率", "knowledge_slope": "词频斜率", "naming_damage": "平均命名损伤",
    "semantic_damage": "语义题平均损伤", "string_damage": "字符串题平均损伤",
    "delta_F": "双重分离 ΔF", "delta_C": "双重分离 ΔC", "D_obs": "观测组间差",
    "D_pred": "退化模型预测组间差", "excess": "观测超出退化预测", "observed": "观测变化",
    "predicted": "预测变化", "S1a_naming": "命名损伤 K8 减 C4", "S1b_six_choice": "选择损伤 C4 减 K8",
    "naming_loss_K8": "K8 命名损伤", "comprehension_loss_K8": "K8 词选定义损伤",
    "direct_specific_K8": "K8 目标首 token 特异直接贡献", "A_C4_answer": "C4 回答区 A",
    "name_target_logprob": "目标词 log probability", "name_p_target": "五候选目标份额",
    "name_S": "联想减分类份额", "options_word_target_logprob": "给选项说词 log probability",
    "copy_target_logprob": "照抄 log probability", "options_letter_margin": "六选一 margin",
    "word_to_definition_margin": "词选定义 margin", "naming_loss": "命名损失",
    "six_choice_loss": "六选一损失", "comprehension_loss": "词选定义损失", "word_shown_loss": "给选项说词损失",
    "copy_loss": "照抄损失", "cue_benefit_change": "线索获益变化", "miscue_harm_change": "误导伤害变化",
    "B": "线索获益变化 B", "H": "误导伤害变化 H", "A": "B 减 H", "R_change": "提示摆动变化",
    "spearman": "Spearman 相关", "rho": "Spearman 相关", "mean": "均值", "estimate": "估计",
    "hints_vs_none_hinted_demos": "带提示示范的无提示基线", "hints_vs_none": "普通无提示基线",
    "greedy_target_rate": "实际生成说对比例", "specific": "目标首 token 特异直接贡献",
    "direct_specific": "目标首 token 特异直接贡献", "total_specific_drop": "同尺度总损伤",
    "target_coef": "正确项联想系数", "distractor_coef": "诱饵联想系数",
    "target_coef_margin_fixed": "固定原 margin 后正确项联想系数",
    "distractor_coef_margin_fixed": "固定原 margin 后诱饵联想系数",
}


def pointer(parts):
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def get(document, parts):
    for part in parts:
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def input_bytes(relative):
    raw = (ROOT / relative).read_bytes()
    INPUT_HASHES[relative] = hashlib.sha256(raw).hexdigest()
    return raw


def csv_rows(relative):
    return list(csv.DictReader(io.StringIO(input_bytes(relative).decode("utf-8-sig"))))


def m1_rows(split=None):
    rows = csv_rows("data/materials/m1/items.csv")
    if split is not None:
        groups = {r["concept_group"] for r in csv_rows("data/materials/m1/split.csv") if r["split"] == split}
        rows = [r for r in rows if r["concept_group"] in groups]
    return rows


def metric_path(parts):
    return parts[:-1] if parts[-1] in ("estimate", "mean", "rho") else parts


def classification(source, parts):
    p = "/".join(parts)
    if source.startswith("g1"):
        return "测量检查（开发阶段）"
    if source == "control":
        if p.startswith("intervals/association_slope/"):
            return "预先规定的证伪条件（正式题）"
        return "确认性主检验" if p.startswith("intervals/control_effect/") else "只报告（正式题）"
    if source == "knowledge":
        return "确认性主检验" if p.startswith("sets/main/") and ("delta_F/" in p or "delta_C/" in p) else "只报告（正式题）"
    if source == "experiment3":
        if parts[0] == "primary":
            return "确认性主检验" if len(parts) == 3 and parts[2] in ("D_obs", "excess") else "只报告（正式题，主检验分量）"
        if parts[0] == "secondary":
            if parts[:3] == ("secondary", "S3", "B"):
                return "只报告（正式题，非检验族）"
            if parts[:3] == ("secondary", "S5", "associate_excess_K30") and parts[-1] in ("observed", "predicted"):
                return "只报告（正式题，次要检验分量）"
            return "确认性次要检验"
        if "test_refit" in p:
            return "探索性（正式题描述性重拟合）"
        return "只报告（正式题，非检验族）"
    return "探索性"


def unit(source, parts):
    p = "/".join(metric_path(parts))
    last = metric_path(parts)[-1]
    if last in ("n_random_at_least_target", "at_least_target", "n_random"):
        return "组件组数"
    if source == "experiment3" and (p.startswith("primary/H3b/") or p.startswith("secondary/S3/")
                                    or last in ("B", "H", "A")):
        return "限定候选 log odds 变化（nat）"
    if source == "control" and "association_slope" in parts:
        return "logit / 标准化诱饵相似度"
    if source.startswith("g1") and "cloze_tds_effect" in parts:
        return "log probability 差（nat）"
    if source.startswith("g1") and metric_path(parts)[-1] == "synonym_frequency":
        return "logit / 标准化词频"
    if any(s in p for s in ("spearman", "rho", "corr_", "item_consistency", "naming_loss_vs_")):
        return "相关系数"
    if "change_in_intact_sd" in p:
        return "完好概念间 SD"
    if (last in ("a", "b_cue", "b_miscue") and ("evidence_weights" in p or "/weights/" in p)) or any(
        s in p for s in ("share_of_whole", "lesioned_per_intact", "effect_share")
    ):
        return "无量纲"
    if any(s in p for s in ("accuracy", "consistency", "p_target", "name_S", "still_all_correct", "greedy", "share_improved",
                            "role_probability", "top1_by_layer", "all_correct", "hint_attention")) or last == "dS":
        return "比例（0–1）或比例差"
    if source == "experiment3" and (p.startswith("primary/H3a/") or "S5" in p) and "S_log" not in p:
        return "五候选概率差"
    if any(s in p for s in ("logprob", "log_prob", "naming_loss", "word_shown_loss", "copy_loss", "S1a_naming")):
        return "log probability（nat）"
    if source in ("knowledge", "zero") and "naming" in parts:
        return "log probability（nat）"
    if source in ("knowledge", "zero") and (parts[0] == "random" or "random_knowledge_effect" in parts):
        return "log probability（nat）"
    if source.startswith("association") and last in ("target_coef", "distractor_coef", "target_coef_margin_fixed", "distractor_coef_margin_fixed", "margin_coef"):
        return "logit / 标准化预测变量"
    if any(s in p for s in ("H3a_S_log_pt",)):
        return "nat/token 差"
    if "H3a_S_log" in p:
        return "log probability 差（nat）"
    if any(s in p for s in ("S1b_six_choice", "tds_effect", "synonym_frequency")):
        return "logit"
    if any(s in p for s in ("margin", "logit", "direct", "specific", "_loss", "pull_", "H3b", "A_C", "delta_", "residual_R", "_dR", "residual_by_condition")):
        return "logit"
    if last in ("B", "H", "A", "R_change", "cue_benefit_change", "miscue_harm_change", "cue_benefit_gain", "miscue_harm_gain"):
        return "log odds 变化（nat）"
    if source in ("control", "knowledge", "zero") or source.startswith("association"):
        return "logit"
    if source in ("dla", "intrusion"):
        return "logit"
    if source == "random_controls":
        if "naming_loss" in parts or "word_shown_loss" in parts or "copy_loss" in parts:
            return "log probability（nat）"
        return "logit"
    return "待核（勿直接作图）"


def m1_n(metric, split):
    rows = m1_rows(split)
    suffix = metric.rsplit(".", 1)[-1]
    if suffix in ("control_effect", "control_slope", "delta_C", "association_slope", "intact_tds_effect",
                  "control_effect_share", "control_test_control_effect"):
        rows = [r for r in rows if r["task"] in ("colour", "size")]
    elif suffix == "low_demand_damage":
        rows = [r for r in rows if r["task"] in ("colour", "size") and r["high_demand"] == "0"]
    elif suffix in ("knowledge_effect", "knowledge_slope"):
        rows = [r for r in rows if r["task"] == "synonym"]
    elif suffix == "string_damage":
        rows = [r for r in rows if r["task"] == "string"]
    elif suffix == "semantic_damage":
        rows = [r for r in rows if r["task"] != "string"]
    else:
        return "", "", "统计子集样本量待核"
    return len(rows), "题目", f"据 M1 材料及 split.csv 的 {split} 划分计算，bootstrap 按概念组"


def g1_n(doc, parts):
    """Sample counts from task coverage in the same report and the original selection rules."""
    rows = m1_rows()
    version = doc["versions"][parts[1]]
    # Every material item has human accuracy. Thus the agreement n also documents
    # complete label-score coverage of each task in this particular report.
    if any(not r["young_acc"] for r in rows):
        raise ValueError("G1 count derivation needs all material items to have human accuracy")
    for task in ("colour", "size", "string", "synonym"):
        if sum(r["task"] == task for r in rows) != version["human_agreement"][task]["n"]:
            raise ValueError(f"G1 {parts[1]} has incomplete {task} coverage")
    if "string" in parts:
        selected = [r for r in rows if r["task"] == "string"]
    elif "synonym_frequency" in parts:
        selected = [r for r in rows if r["task"] == "synonym"]
    elif "accuracy_by_task" in parts:
        selected = [r for r in rows if r["task"] == parts[-1]]
    else:
        selected = [r for r in rows if r["task"] in ("colour", "size")]
        if any("high_demand_accuracy" in part for part in parts):
            selected = [r for r in selected if r["high_demand"] == "1"]
    if "cloze_tds_effect" in parts:
        evidence = ("按 scoring.build_cloze_prompts 与 e0_baseline/run.py 的完整颜色/尺寸题规则计数，"
                    "原始 cloze 分数未在本地复查")
    else:
        evidence = (f"据 /versions/{parts[1]}/human_agreement/*/n 核实各任务完整覆盖，"
                    "按 src/lesion/g1.py、g1_revised.py 及 M1 items.csv 选取子集")
    return len(selected), "题目", evidence + "，两种顺序不重复计 n，区间按概念组抽样"


def association_n(source, doc, parts):
    split = source.removeprefix("association_")
    split = "test" if split == "heldout" else split
    rows = [r for r in m1_rows(split) if r["task"] in ("colour", "size")]
    available = {r["item_id"] for r in csv_rows("data/materials/m1/associations.csv")}
    rows = [r for r in rows if r["item_id"] in available]
    if len(rows) != doc[parts[0]]["n_items"]:
        raise ValueError(f"{source}/{parts[0]} sample differs from material selection")
    metric = metric_path(parts)[-1]
    rule = "颜色与尺寸全部题目"
    if metric == "share_improved_low_demand":
        rows = [r for r in rows if r["high_demand"] != "1"]
        rule = "high_demand != 1"
    elif metric == "share_improved_high_demand":
        rows = [r for r in rows if r["high_demand"] == "1"]
        rule = "high_demand == 1"
    elif metric == "mean_h_easiest_third":
        values = sorted(float(r["tds"]) for r in rows)
        pos = (len(values) - 1) * (2 / 3)
        lo = math.floor(pos)
        threshold = values[lo] + (pos - lo) * (values[math.ceil(pos)] - values[lo])
        rows = [r for r in rows if float(r["tds"]) >= threshold]
        rule = f"TDS >= 子集的 2/3 分位数 {threshold:.17g}，含边界并列题"
    return len(rows), "题目", (f"据 M1 items/split/associations.csv 的 {split} 子集，{rule}。"
                              "沿用 explore_association.py，bootstrap 按概念组，易题分位数每次重算")


def sample_size(source, doc, parts, ancestors):
    p = "/".join(parts)
    # A random-set count is not the number of evaluated concepts.
    if any(s in p for s in ("vs_random", "/random/", "random_knowledge_effect")) or parts[0] == "random":
        for path, node in reversed(ancestors):
            for k in ("n_random", "n"):
                if numeric(node.get(k)):
                    return node[k], "随机组件组", "来源 " + pointer((*path, k))
    for path, node in reversed(ancestors):
        if numeric(node.get("n")):
            return node["n"], "概念/题目", "来源 " + pointer((*path, "n"))
    if source.startswith("g1"):
        return g1_n(doc, parts)
    if source in ("control", "knowledge", "zero"):
        mp = metric_path(parts)
        if "naming" in mp:
            n = doc.get("n_items", {}).get("naming") if source == "knowledge" else None
            evidence = "来源 /n_items/naming"
            if source == "zero":
                meta_path = "outputs/e3_knowledge/qwen3-30b-a3b-instruct/expand8_zero/meta.json"
                meta = json.loads(input_bytes(meta_path))
                if meta["split"] != "dev" or meta["status"] != "done" or meta["ablation"]["mode"] != "zero":
                    raise ValueError("zero-ablation provenance differs from the reported dev run")
                n = sum(r["split"] == meta["split"] for r in csv_rows("data/materials/m3_knowledge/items.csv"))
                evidence = f"据 {meta_path} 的 split=dev 及 m3_knowledge/items.csv，run_expand8.py 未再筛选题目"
            if mp[-1].endswith("common_damage"):
                n = n // 2 if n else None
                evidence += "，仅高频半数"
            return n or "", "命名概念", evidence if n else "源 JSON 未提供命名样本量，待核"
        return m1_n(mp[-1], "dev" if source == "zero" else "test")
    if source == "experiment3":
        if parts[:2] == ("primary", "H3a"):
            return doc["n"]["h3a"], "概念", "来源 /n/h3a"
        if parts[:2] == ("primary", "H3b"):
            return doc["n"]["h3b"], "概念", "来源 /n/h3b"
        if "evidence_weights_calibration_to_test" in p and "weights" in parts:
            return 40, "校准概念", "分析计划 experiment3.md：calibration_b，不能记作正式题 n"
        if any(s in p for s in ("test_refit", "predicted_dR", "observed_dR", "residual_by_condition")) or metric_path(parts)[-1] in ("B", "H", "A"):
            return doc["n"]["h3b"], "概念", "来源 /n/h3b"
        if metric_path(parts)[-1] == "dS" or "H3a_S_log" in p:
            return doc["n"]["h3a"], "概念", "来源 /n/h3a"
        return doc["n"]["test"], "概念", "来源 /n/test"
    if source.startswith("association"):
        return association_n(source, doc, parts)
    if source == "battery":
        if parts[0] == "greedy":
            return doc["greedy_n"][parts[1]], "概念", "来源 " + pointer(("greedy_n", parts[1]))
        if "states" in parts:
            n = doc["states"][parts[1]]["n_concepts"]
            if "hints_vs_" in p:
                n = doc["states"][parts[1]][next(x for x in parts if x.startswith("hints_vs_"))]["R_change"]["n"]
            return n, "概念", "对应 state 的 n_concepts 或 hints 的 n"
        return 80, "校准概念", "run_battery 的 calibration_a+b，taskmean 子集另列"
    if source in ("random_controls", "dla", "intrusion"):
        if source == "random_controls" and any(s in p for s in ("cue_benefit", "miscue_harm")):
            return 76, "校准概念", "校准 free_naming_ok 子集，与 battery 一致"
        return 80, "校准概念", "对应 runner 的 calibration_a+b"
    return "", "", "源 JSON 未提供该指标的 n，待核"


def resolve(doc, field):
    node = doc
    for part in field.lstrip("/").split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def new_unit(source, parts, ancestors):
    path = "/".join(parts)
    last = parts[-1]
    for _, node in reversed(ancestors):
        if isinstance(node.get("unit"), str):
            return node["unit"]
    if last in ("n_random", "at_least_target", "n_present", "n_reversed") or "random_lower_bound" in last or "random_upper_bound" in last:
        return "随机组件组数"
    if last == "rank_from_top":
        return "随机对照中的名次（1起）"
    if last in ("n_components", "k", "chosen_k", "n_redundant", "singles", "singles_point_estimate_up", "singles_ci_excludes_zero", "gates_below_half", "n_gates", "n_eligible") or "chosen_k" in parts:
        return "组件数"
    if last.startswith("pairs_") or "pairs" in parts:
        return "组件对数"
    if last.startswith("matches_"):
        return "符合的临床标记数"
    if "n_tokens" in parts or "n_two_token" in parts or last in ("n_synonym_rare", "n_synonym_common", "n_rare", "n_common", "n_target_in_top"):
        return "词/题目数"
    if last in ("n_rows", "n_complete", "n_large", "lesions", "lesions_passed"):
        return "记录数" if "taxonomy" not in path else "组件数"
    if last == "n_clusters":
        return "聚类数"
    if "tokens_per_part" in parts or last == "n_routed":
        return "token 数"
    if "jaccard" in path.lower():
        return "Jaccard 重叠比例（0–1）"
    if any(text in path for text in ("spearman", "correlation", "corr_", "consistency")) or last in ("r", "rho"):
        return "相关系数"
    if any(text in path for text in ("cosine", "silhouette")):
        return "无量纲"
    if last in ("quantile", "target_quantile", "chance", "top1", "rare_top1", "lost_most_in_top") or any(part.endswith("_share") for part in parts) or any(text in path for text in (
        "accuracy", "correct", "share_", "top_quarter_overlap", "p_target", "top1_rate")):
        return "比例（0–1）或比例差"
    if any(text in path for text in ("ratio", "sum_of_parts_over_all", "slope_range")) or "slope" in parts or "m1_slope" in path:
        return "无量纲（损伤后/完好分数比或逐题缩放斜率）"
    if any(text in path for text in ("lm_loss_change", "lm_change", "language/change", "lm_loss/change")) or (
        "language" in parts and "vs_random" in path) or ("lm_loss" in parts and last == "relative"):
        return "相对语言损失变化（比例，0.1=10%）"
    if last in ("lm_loss", "lm_intact") or "lm_loss/loss" in path or (source == "phi_replication" and "lm_loss" in parts):
        return "语言损失（nat/token）"
    if any(text in path for text in ("frequency_median", "zipf_median")):
        return "Zipf 词频"
    if "per_token_rare_minus_common" in path:
        return "log probability 差（nat/token）"
    if any(text in path for text in ("knowledge_slope", "rarity_coef")):
        return "logit / 标准化稀有度" if "m1" in path else "log probability（nat）/ 标准化稀有度"
    if "n_tokens_coef" in path:
        return "log probability（nat）/ 标准化 token 数"
    if last in ("relative_difference", "intact_logprob_coef"):
        return "无量纲"
    if "mean_change_by_bin" in parts or last == "mean_target_change":
        return "log probability 变化（nat）"
    if "control_slope" in path:
        return "logit / 标准化控制需求"
    if "association_slope" in path:
        return "logit / 标准化诱饵相似度"
    if source == "optogenetics":
        if any(text in path for text in ("/B", "/H", "/dR", "_dR")):
            return "限定候选 log odds 变化（nat）"
        if any(text in path for text in ("naming", "word_shown", "copy", "K8_dose_by_frequency")):
            return "log probability 损失或差（nat）"
        if any(text in path for text in ("six_choice", "comprehension")):
            return "logit 差额变化"
    if source == "p14_session15":
        if "p14_statistic" in parts:
            return "log probability（nat），控制 token 数后的冷门词系数"
        if "stored" in parts and "random" in parts:
            return "log probability 差（nat），未调整冷门减常用"
        if last in ("select_selectivity", "confirm_selectivity", "delta"):
            side = next((node.get("side") for _, node in reversed(ancestors) if node.get("side")), None)
            if "control" in parts or side == "control":
                return "logit，强干扰减弱干扰损伤"
            if "hub" in parts and last == "select_selectivity" or side == "hub":
                return "无量纲，命名与理解效应各除以完好分数离散度后相加"
            return "log probability（nat），控制 token 数后的冷门词系数"
    if any(text in path for text in ("control_effect", "control_demand", "delta_C", "semantic_damage", "string_damage", "synonym", "m1_semantic", "m1_string", "low_demand_damage", "intact_tds_effect")):
        return "logit"
    if any(text in path for text in ("knowledge_effect", "naming_damage", "delta_F", "m3_naming", "logprob", "naming/", "naming.", "two_token_rare_minus_common", "mean_change_by_zipf_fifth")):
        return "log probability 损失或差（nat）" if "m1" not in path else "logit"
    if source == "k30_synonym":
        return "logit"
    if any(text in path for text in ("margin", "m3_word_to_definition", "m3_options_letter")) or last in ("h", "damage", "intercept"):
        return "logit"
    if source in ("qwen_new_sets", "qwen_clinical_scorecard") and "random" in path:
        return "log probability 差（nat）" if any(text in path for text in ("knowledge", "frequency")) else "logit"
    if (source == "session13" and "item2_control_positions" in parts and "random" in parts or
            source == "session14b" and "step2_label_parts/random" in path):
        return "logit，控制效应"
    if source == "session12" and parts[0] == "item3_shared_removed":
        return "logit，控制效应"
    if "random" in parts and any(text in path for text in ("dissociation", "held_out", "item_1b")):
        return "log probability 差（nat）"
    if "random" in parts and any(text in path for text in ("control_semantic", "control_test")):
        return "logit，控制效应"
    if source == "session12" and parts[0] == "item4_task_vectors" and "vector_geometry" not in parts:
        return "logit"
    return "待核（勿直接作图）"


def new_sample_size(source, doc, parts, ancestors):
    path = "/".join(parts)
    if parts[-1] in ("target_single", "main_single_scoring"):
        if source == "p14_session15":
            return doc[parts[0]]["n_items"]["naming"], "命名题", "来源 " + pointer((parts[0], "n_items", "naming"))
        return "", "", "目标组单次评分的题数未提供，不用随机组数量代替"
    for prefix, node in reversed(ancestors):
        if parts[-1] == "two_token_rare_minus_common" and isinstance(node.get("n_two_token"), list):
            return sum(node["n_two_token"]), "双 token 词", "来源 " + pointer((*prefix, "n_two_token")) + " 两组数量相加"
        if parts[-1] == "lost_most_in_top" and numeric(node.get("n_target_in_top")):
            return node["n_target_in_top"], "目标进入候选的题目", "来源 " + pointer((*prefix, "n_target_in_top"))
        if "synonym_frequency" in parts and numeric(node.get("n_rare")) and numeric(node.get("n_common")):
            if "rare" in parts[-1] and "common" not in parts[-1]:
                key = "n_rare"
            elif "common" in parts[-1] and "rare" not in parts[-1]:
                key = "n_common"
            else:
                return node["n_rare"] + node["n_common"], "同义题", "来源 " + pointer(prefix) + "/n_rare 与 n_common 相加"
            return node[key], "同义题", "来源 " + pointer((*prefix, key))
    for prefix, node in reversed(ancestors):
        random = any(part in ("random", "random_comparison") or part.endswith("_vs_random") for part in prefix)
        for key in (("n", "n_random") if random else ("n", "n_items")):
            if numeric(node.get(key)):
                n_unit = node.get("n_unit", "随机组件组" if random else "概念/题目")
                if source == "title_gap_patient_profiles":
                    n_unit = "患者"
                if "clusters" in prefix:
                    n_unit = "组件"
                return node[key], n_unit, "来源 " + pointer((*prefix, key))
        if numeric(node.get("n_words")) and any(text in path for text in ("word_length", "item6_word_length")):
            return node["n_words"], "词", "来源 " + pointer((*prefix, "n_words"))
        if numeric(node.get("n_texts")) and "lm_loss" in path:
            return node["n_texts"], "中性文本", "来源 " + pointer((*prefix, "n_texts"))
        if numeric(node.get("n_naming")) and "naming" in parts:
            return node["n_naming"], "命名题", "来源 " + pointer((*prefix, "n_naming"))
    if source == "p14_session15" and parts[0].startswith("final_"):
        counts = doc[parts[0]]["n_items"]
        if "naming" in path or "p14_statistic" in parts:
            key = "naming"
        elif "control_effect" in path or "delta_C" in path:
            key = "m1_colour_size"
        else:
            key = None
        if key:
            return counts[key], "题目", "来源 " + pointer((parts[0], "n_items", key))
    if source == "p14_session15" and "stability" in parts:
        prefix = parts[:3]
        return doc[parts[0]][parts[1]]["stability"]["n_runs"], "搜索运行", "来源 " + pointer((*prefix, "n_runs"))
    if source == "k30_synonym" and ("synonym" in path or "K30_bootstrap" in parts):
        state = parts[1] if parts[0] == "states" else "K30"
        return doc["states"][state]["synonym"]["n"], "同义题", "来源 " + pointer(("states", state, "synonym", "n"))
    if source == "optogenetics" and "spearman_B_at_0_vs_naming_rescued_at_0.5" in parts:
        return "", "", "源 JSON 未提供相关分析配对后的 n，不用全体概念数代替"
    return "", "", "源 JSON 未提供该指标对应的 n，留空，不以 bootstrap 次数或全体 n 代替"


def new_rows(source, doc, digest):
    filename, title, dataset = SOURCES[source]
    rows = []

    def walk(obj, parts=(), ancestors=(), interval=None):
        if isinstance(obj, dict):
            chain = (*ancestors, (parts, obj))
            for key, value in obj.items():
                if key in NEW_SKIP or key in ANCILLARY or key.endswith("_ci"):
                    continue
                if source == "session14b" and parts == ("step3_taxonomy",) and key == "sets":
                    continue
                walk(value, (*parts, key), chain)
            return
        if isinstance(obj, list):
            if source in TRIPLE_SOURCES and len(obj) == 3 and all(numeric(value) for value in obj):
                walk(obj[0], (*parts, "0"), ancestors, (obj[1:], [pointer((*parts, str(index))) for index in (1, 2)]))
            else:
                for index, value in enumerate(obj):
                    walk(value, (*parts, str(index)), ancestors)
            return
        if not numeric(obj):
            return
        parent_path, parent = ancestors[-1]
        metric = parts[:-1] if interval or parts[-1] in ("estimate", "mean", "rho", "value") else parts
        ci, ci_refs = interval if interval else (None, [])
        if not interval and parts[-1] in ("estimate", "mean", "rho", "spearman"):
            if isinstance(parent.get("ci"), list):
                ci = parent["ci"]
                ci_refs = [pointer((*parent_path, "ci", str(index))) for index in (0, 1)]
            elif "ci_low" in parent and "ci_high" in parent:
                ci = [parent["ci_low"], parent["ci_high"]]
                ci_refs = [pointer((*parent_path, key)) for key in ("ci_low", "ci_high")]
        if not interval and isinstance(parent.get(parts[-1] + "_ci"), list):
            ci = parent[parts[-1] + "_ci"]
            ci_refs = [pointer((*parent_path, parts[-1] + "_ci", str(index))) for index in (0, 1)]
        if source == "k30_synonym":
            key = {("states", "K30", "synonym", "ratio"): "synonym_ratio",
                   ("states", "K30", "synonym_rare_minus_common"): "synonym_rare_minus_common",
                   ("K30_synonym_excess",): "synonym_excess"}.get(parts)
            if key:
                ci_refs = [pointer(("K30_bootstrap", key, bound)) for bound in ("ci_low", "ci_high")]
                ci = [resolve(doc, field) for field in ci_refs]
        if ci is not None and (len(ci) != 2 or not all(numeric(value) for value in ci)):
            ci, ci_refs = None, []
        n, n_unit, n_note = new_sample_size(source, doc, metric, ancestors)
        nature = "探索性（第二模型，G1 不合格）" if source == "phi_replication" else "探索性"
        if source == "title_gap_clinical_final_cues":
            nature = "补实验最终读数（固定状态，含未定子类敏感性）"
        if source == "title_gap_clinical_final":
            nature = "补实验最终组级转移（开发选择固定）"
        if source == "title_gap_clinical_final_calibration":
            nature = "补实验最终标定的原始描述读数"
        if source == "title_gap_clinical_scale_sensitivity":
            nature = "事后量尺敏感性（固定状态，只有点估计）"
        if source == "title_gap_fine_core_readout":
            nature = "补实验最终核心读数（用于后续随机族选择）"
        if source == "title_gap_fine_final_semd":
            nature = "补实验最终配对回归（比较与协变量预先固定）"
        if source == "title_gap_coarse_final_core":
            nature = "补实验最终粗损伤读数（状态冻结，与 SC4 配对比较）"
        if source == "title_gap_fine_remaining_core":
            nature = "补实验最终核心读数（渐进、联合、同大小与直接配对，状态冻结）"
        if source == "title_gap_answer_completion":
            nature = "补实验最终作答端点（遮蔽二元裁决补齐弃权，原冻结函数）"
        if source == "title_gap_homonyms_final":
            nature = "补实验最终检验（冻结方案，内部保留题，含分类界限）"
        if source == "title_gap_clinical_cue_search":
            nature = "补实验最终提示读数（搜索候选，状态冻结，用户决定加测）"
        if source in ("title_gap_patient_profiles", "title_gap_patient_markers"):
            nature = "文献数据再分析"
            if parts[0] == "figure_sensitivity":
                nature = "读图精度敏感性界限"
        notes = ["仅提取来源已有精度，不补回来源已舍入的小数"]
        if source == "title_gap_fine_core_readout":
            notes.append("候选在开发阶段固定，后续随机族将依据本次核心读数选择")
            if parent.get("status") == "pending_bootstrap":
                notes.append("原定抽样区间正在计算，空区间不表示零")
        if source == "optogenetics":
            if parts[0] == "rules":
                nature = "确认性次要检验（R1–R6，第三族 Holm 校正）"
                notes.append("n 按指标从规则原值读取，命名/选择 160，提示指标 100。分量 p 与规则级 Holm p 不属同一层")
            else:
                nature = "只报告（正式题，非检验族）"
                if "O3" in parts:
                    nature = "探索性"
                if n == 132:
                    nature = "探索性（132 个可用概念的探索读数）"
                    notes.append("提示派生指标使用 132 个可用概念，不得当作预定 100 个概念的正式提示检验")
                if any("spearman_B_at_0" in part for part in parts):
                    nature = "探索性（可用概念提示读数，配对 n 未提供）"
                if "lm_loss" in parts:
                    notes.append("分段干预的语言损失是整段替代检查，不是经证明的上界")
        if source == "p14_session15":
            if "stored" in parts:
                notes.append("旧未调整统计 stored，不用于与 P14 自身目标作同口径阶段比较")
            if "p14_statistic" in parts:
                notes.append("P14 自身统计，控制 token 数后的冷门词系数，与 stored 分开使用")
            if parts[:2] == ("sets", "hub"):
                notes.append("hub 的 confirm/minimality 仅检查命名侧，select 为双目标，不混作同口径")
        if source == "session12" and parts[0] == "item1_phi_split":
            notes.append("Phi 第二模型，G1 不合格")
        for prefix, node in ancestors:
            for key in ("note", "statistic", "selectivity_definition"):
                if isinstance(node.get(key), str):
                    notes.append(pointer((*prefix, key)) + ": " + node[key])
            if isinstance(node.get("task"), str):
                notes.append("task=" + node["task"])
            if isinstance(node.get("run"), str):
                notes.append("run=" + node["run"])
        p_raw, p_adj, p_refs, verdict = "", "", {}, ""
        for prefix, node in reversed(ancestors):
            if p_raw == "":
                for key in ("p_one_sided", "empirical_p", "p"):
                    if numeric(node.get(key)):
                        p_raw, p_refs["原始"] = node[key], pointer((*prefix, key))
                        break
            if p_adj == "" and numeric(node.get("p_holm")):
                p_adj, p_refs["校正"] = node["p_holm"], pointer((*prefix, "p_holm"))
            if not verdict and "verdict" in node:
                verdict = pointer(prefix) + ": " + str(node["verdict"]) + "（组级判定）"
        description = " / ".join(LABELS.get(part, part) for part in metric)
        if "stored" in parts:
            description = "旧未调整统计 / " + description
        elif "p14_statistic" in parts:
            description = "P14 自身调整统计 / " + description
        rows.append(dict(zip(FIELDS, [
            source + "-" + hashlib.sha256(pointer(parts).encode()).hexdigest()[:12],
            f"{title}，{description} = {obj}", obj, ci[0] if ci else "", ci[1] if ci else "",
            new_unit(source, metric, ancestors), n, n_unit, n_note, nature, dataset, pointer(metric), filename,
            pointer(parts), json.dumps(ci_refs, ensure_ascii=False) if ci_refs else "", digest,
            p_raw, p_adj, json.dumps(p_refs, ensure_ascii=False) if p_refs else "", verdict, "。".join(notes)
        ])))

    walk(doc)
    return rows


def build_rows():
    INPUT_HASHES.clear()
    for path, reviewed_digest in CODE_REVIEW["reviewed_sha256"].items():
        if hashlib.sha256(input_bytes(path)).hexdigest() != reviewed_digest:
            raise ValueError(f"Code changed since historical-number review: {path}. Review before refreshing provenance.")
    for path in (
        "docs/paper-prep/build_ledger.py", "src/lesion/g1.py", "src/lesion/g1_revised.py", "src/lesion/scoring.py",
        "src/lesion/knowledge_items.py", "experiments/e0_baseline/run.py", "experiments/e0_baseline/analyze.py",
        "experiments/e0_revised/analyze.py", "experiments/e2_confirm/explore_association.py",
        "experiments/e2_confirm/run_control_test.py", "experiments/e3_knowledge/run_expand8.py",
        "docs/analysis-plan/control-test.md", "docs/analysis-plan/experiment3.md", "docs/adr/0003-qwen-and-revised-g1.md",
        "docs/analysis-plan/optogenetics.md",
    ):
        input_bytes(path)
    rows = []
    for source, (filename, title, dataset) in SOURCES.items():
        if source in ARCHIVED and not (ROOT / filename).exists():
            rows.extend(archived_rows(filename))
            continue
        raw = input_bytes(filename)
        doc = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if source in NEW_SOURCES and not source.startswith("g1"):
            rows.extend(new_rows(source, doc, digest))
            continue

        def walk(obj, parts=(), ancestors=()):
            if isinstance(obj, dict):
                chain = (*ancestors, (parts, obj))
                for key, value in obj.items():
                    skip = key in SKIP
                    if key == "primary" and source == "experiment3":
                        skip = False
                    if key == "control" and source.startswith("association"):
                        skip = False
                    if skip or key in ANCILLARY or key.endswith("_ci"):
                        continue
                    walk(value, (*parts, key), chain)
                return
            if isinstance(obj, list):
                for i, value in enumerate(obj):
                    walk(value, (*parts, str(i)), ancestors)
                return
            if not numeric(obj):
                return
            parent_path, parent = ancestors[-1]
            ci, ci_ptrs = None, []
            if parts[-1] in ("estimate", "mean", "rho", "spearman"):
                if isinstance(parent.get("ci"), list):
                    ci, ci_ptrs = parent["ci"], [pointer((*parent_path, "ci", str(i))) for i in (0, 1)]
                elif "ci_low" in parent and "ci_high" in parent:
                    ci, ci_ptrs = [parent["ci_low"], parent["ci_high"]], [pointer((*parent_path, k)) for k in ("ci_low", "ci_high")]
            elif isinstance(parent.get(parts[-1] + "_ci"), list):
                ci = parent[parts[-1] + "_ci"]
                ci_ptrs = [pointer((*parent_path, parts[-1] + "_ci", str(i))) for i in (0, 1)]
            if ci is not None and (len(ci) != 2 or not all(numeric(x) for x in ci)):
                ci, ci_ptrs = None, []
            n, n_unit, n_note = sample_size(source, doc, parts, ancestors)
            units = unit(source, parts)
            p, p_adj, p_refs, verdict = "", "", {}, ""
            for ap, node in reversed(ancestors):
                if p == "":
                    for k in ("p_one_sided", "empirical_p", "p"):
                        if numeric(node.get(k)):
                            p, p_refs["原始"] = node[k], pointer((*ap, k))
                            break
                if p_adj == "" and numeric(node.get("p_holm")):
                    p_adj, p_refs["校正"] = node["p_holm"], pointer((*ap, "p_holm"))
                if not verdict and "verdict" in node:
                    verdict = f"{pointer(ap)}: {node['verdict']}（组级判定）"
            mparts = metric_path(parts)
            description = " / ".join(LABELS.get(x, x) for x in mparts)
            notes = []
            if ci is None:
                notes.append("源报告未给此值的95%区间")
            if p_refs:
                notes.append("p及判定可能属于上层联合检验，按字段路径使用，勿当作本行独立检验")
            if source == "battery" and "change" in parts:
                notes.append("方向为损伤减完好，正值不一定表示损伤")
            if source == "g1_qwen":
                notes.append("G1 按 ADR 0003 修订规则通过，旧规则字段保留，非所有原标记均通过")
            if source.startswith("association"):
                notes.append("事后联想分析，heldout 数据不使该分析自动成为确认性")
            if source == "g1_gptoss_p12b":
                notes.append("看过格式检查后的事后重试，使用 harmony_answer，不是预先固定格式的一次独立确认")
            rows.append(dict(zip(FIELDS, [
                source + "-" + hashlib.sha256(pointer(parts).encode()).hexdigest()[:12],
                f"{title}，{description} = {obj}", obj, ci[0] if ci else "", ci[1] if ci else "", units,
                n, n_unit, n_note, classification(source, parts), dataset, pointer(mparts), filename,
                pointer(parts), json.dumps(ci_ptrs, ensure_ascii=False) if ci_ptrs else "", digest,
                p, p_adj, json.dumps(p_refs, ensure_ascii=False) if p_refs else "", verdict, "。".join(notes)
            ])))

        walk(doc)
    if len({r["编号"] for r in rows}) != len(rows):
        raise ValueError("duplicate ledger ids")
    return rows


def serialize(rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def validate_values(content):
    documents = {filename: json.loads((ROOT / filename).read_bytes()) for filename, _, _ in SOURCES.values()
                 if (ROOT / filename).exists()}
    checked = {}
    for row in csv.DictReader(io.StringIO(content.decode("utf-8-sig"))):
        filename = row["来源文件"]
        if filename not in documents:
            continue
        document = documents[filename]
        references = [(row["来源字段路径"], row["数值"])]
        references.extend(zip(json.loads(row["区间字段路径"] or "[]"),
                              (row["95%区间下限"], row["95%区间上限"])))
        references.extend((reference, row["p原始" if label == "原始" else "p校正"])
                          for label, reference in json.loads(row["p字段路径"] or "{}").items())
        for reference, text in references:
            value = resolve(document, reference)
            equal = int(text) == value if isinstance(value, int) else float(text).hex() == value.hex()
            if not equal:
                raise ValueError(f"Value mismatch: {filename} {reference}")
        checked.setdefault(filename, []).append({"pointer": row["来源字段路径"], "value": resolve(document, row["来源字段路径"])})
    return checked


def provenance(content, rows):
    checked = validate_values(content)
    summaries = {}
    for source, (filename, _, _) in SOURCES.items():
        selected = [row for row in rows if row["来源文件"] == filename]
        summaries[source] = dict(path=filename, rows=len(selected), new_source=source in NEW_SOURCES,
                                 evidence=sorted({row["性质"] for row in selected}),
                                 unresolved_units=sum(row["单位"].startswith("待核") for row in selected),
                                 missing_n=sum(row["n"] == "" for row in selected),
                                 verified_examples=checked.get(filename, [])[:3],
                                 archived=ARCHIVED.get(source, ""))
    document = dict(schema_version=2, operation="Read-only report extraction with documented sample-size derivation",
                    historical_code_review=CODE_REVIEW,
                    validation="Every serialized value, CI endpoint and attached p-value is resolved against its JSON pointer. Integers compare exactly, floats by float.hex(), including signed zero. Examples show three rows per source, not the full verification scope.",
                    extraction_notes="Skip item/component lists and bootstrap bookkeeping. Three-number arrays in designated sources are estimate/CI. Missing n is blank. R1-R6 use source-specific 160/100 denominators and Holm rule-level p-values. Available-concept hint readings with n=132 remain exploratory. P14 stored and p14_statistic are separate metrics.",
                    sources=summaries,
                    n_semantics="Items/concepts or random component sets as identified per row, never bootstrap repetitions",
                    cloze_n_limit="G1 cloze n follows the complete-task generation and analysis code. Raw cloze coverage was not rechecked locally.",
                    inputs=dict(sorted(INPUT_HASHES.items())),
                    output=dict(path=OUT.relative_to(ROOT).as_posix(), rows=len(rows), sha256=hashlib.sha256(content).hexdigest()))
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rows = build_rows()
    content = serialize(rows)
    manifest = provenance(content, rows)
    if args.check:
        if not OUT.exists() or OUT.read_bytes() != content:
            raise SystemExit("Ledger differs from source reports. Run build_ledger.py to regenerate.")
        if not MANIFEST.exists() or MANIFEST.read_bytes() != manifest:
            raise SystemExit("Ledger provenance differs from current sources. Run build_ledger.py to regenerate.")
    else:
        temp = OUT.with_suffix(".csv.tmp")
        temp.write_bytes(content)
        temp.replace(OUT)
        MANIFEST.write_bytes(manifest)
    unresolved = sum(r["单位"].startswith("待核") or r["n"] == "" for r in rows)
    print(f"{'Checked' if args.check else 'Built'} {len(rows)} rows from {len(SOURCES)} reports: {OUT}")
    print(f"Rows with unresolved unit or sample size: {unresolved}; explicitly marked, never guessed.")


if __name__ == "__main__":
    main()
