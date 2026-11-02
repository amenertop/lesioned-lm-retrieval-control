# 示范题格式检查（计划）

**状态：用户 2026-10-28 批准；代码已写好并在本机小模型上测通；等用户提供 GPU。**

本机环境修复（2026-10-28）：`lesion` 环境的 transformers 缺了路径含 "gpt" 的 58 个源文件（导致本机无法建 Qwen-MoE 小模型），已从固定版本 4.57.6 的安装包补回，每个文件都与安装记录 RECORD 里的哈希核对一致。

## 用什么显卡（2026-10-28 补）

RTX PRO 6000 暂时没有。模型在 float16 下约 61 GB，代码把整个模型放在一张卡上（`src/lesion/models.py` 的 `device_map` 是单卡），所以要选**单张显存不少于 80 GB 的卡**：A100 80G、A800 80G、H800、H100 或 H20 96G 都可以。48 GB 及以下的卡（L20、RTX 4090、RTX 5090）放不下；拆到两张小卡上要改加载和损伤钩子的代码并重新验证，不建议。现有环境是 PyTorch 2.8.0 加 CUDA 12.8，对 Ampere（A100、A800）和 Hopper（H800、H100、H20）同样适用。不同架构的显卡在 float16 下的舍入会有细小差别，所以一定加 `--also-existing`：新旧两种格式在同一张卡、同一次运行里比较，结论不受换卡影响；现有两个值（0.03、0.78）的复现若有小偏差，报告里照写。

## 开卡时怎么跑

先把这次的代码提交（运行脚本要求 `src/`、`experiments/` 等目录没有未提交的改动），同步到服务器，然后在服务器上：

```bash
cd /root/autodl-tmp/main
# 只在克隆或重启清空了 /dev/shm 时需要：重灌模型缓存（约 6 分钟）
/root/miniconda3/envs/lesion/bin/python /root/autodl-tmp/restore-shm-cache-westb.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf LESION_MODEL_DIR=/root/autodl-tmp/astra-model-cache-20261019
/root/miniconda3/envs/lesion/bin/python experiments/e3_errors/run_demo_format.py --model qwen3-30b-a3b-instruct \
    --means outputs/means/qwen3-30b-a3b-instruct@0d7cf23__42ae809__fp16.pt --dtype float16 --moe sorted --also-existing
```

跑之前可以加 `--dry-run` 先看提示数和每格一道示例题（不加载模型）。结果写到 `outputs/e3_errors/qwen3-30b-a3b-instruct/demo_format/`（`scores.parquet` 是每个概念、每种格式、每个状态、每种批量布局的分数；`prompts.parquet` 是实际用的提示原文；`meta.json` 记模型版本、git 提交、组件名单和耗时）。中途断了，用同一条命令在同一个提交上重跑，会从最后完成的状态接着跑。取回本机后：

```bash
C:\Users\wanghe\.conda\envs\lesion\python.exe experiments/e3_errors/analyze_demo_format.py
```

它写出 `reports/explore/demo-format-<运行日期>/demo-format.json`（账本构建脚本能读的格式，入账时在 `build_ledger.py` 加一个来源），并在屏幕上打印符合下面哪条事先写定的判读规则。

**题量和时间。** 新增的两种格式共 280 道提示（题库 160、家族 120），按原运行的候选词数打分（题库每题 5 个候选、家族每题 3 个），每种批量布局 1,160 条序列，三种布局、三个状态（完好、WR8、SC4）共 10,440 条、约 70 万个词元。加 `--also-existing` 后再把两种现有格式也跑一遍，共 560 道提示、20,880 条序列、约 167 万个词元。按实验三实测的速度（每秒约 2,900 个词元，当时用的还是较慢的 hf 前向）估计：只跑新格式约 4 分钟，加 `--also-existing` 约 10 分钟；加载模型约 1 到 2 分钟（缓存需要重灌再加约 6 分钟）。整次开卡半小时以内。

**建议加 `--also-existing`。** 多花约 6 分钟，换来两件事：一是在同一次运行里重算 0.03 和 0.78，核对新代码走的是同一条路；二是两种格式在同一次运行、同一种批量布局下逐概念配对比较，不受原运行批量设置不同的影响（见下面第 1 条）。

**写代码时发现、会影响比较的细节，以及开跑前定下的做法。**

1. **批量布局。** 原运行里，题库命名用 16、11、7 三种布局（实验三），家族命名用的是 64、47、31（冻结协议 `title-gap-fine.json` 的 `naming_batches`），不是 16、11、7；两套原运行还都把命名提示和别的题型（抄写、同义词作答等）混在一起分批。Qwen 的分数会随同批的其他题轻微变动（ADR 0004）。本次按计划两套题都用 16、11、7，每种格式单独分批打分，所以本次算出的"现有格式"数值会和 0.03、0.78 有小差别，分析脚本把这个差别单独报告（`reproduction`）。
2. **建议加 `--also-existing`**，让两种格式在同一次运行、同一种批量布局下逐概念比较（`format_effect_within_run`），不必拿新格式去比批量设置不同的旧运行。
3. **提示怎么造。** `src/lesion/m3_scoring.py` 和 `src/lesion/fine_semantics.py` 一个字节都不改：冻结分析协议（`docs/analysis-plan/title-gap-*.json`）和图表数据清单记录了它们的哈希。新格式由 `experiments/e3_errors/demo_format_prompts.py` 构造，它从这两个模块导入示范题、题块格式和指令，复制原构造函数并只加一个"带不带示范题"的开关；每次运行前先核对它在现有格式下与原函数输出逐字节相同，测试另核对新旧格式之间只差那段示范题文字。两套题的指令本来就不同（题库 "Name the thing that matches each description. Reply with the word only, in lowercase."，家族 "Name the thing or concept described. Reply with one word only."），指令和描述之间题库空一行、家族不空行，这些保留不动：题库新格式 = 原提示去掉三道示范题那段文字；家族新格式 = 原提示在指令行和描述之间插入同一段文字。
4. **判读界限（开跑前定下，不看结果再改）。** 计划没写"接近零"的界限，定为"95% 区间上限低于 0.20 nat"，与规则另一半用的 0.20 一致（0.20 也是实验三次要检验的最小有意义效应）；"与 0.78 重叠"按"区间包含 0.78"判。
5. **对准的是哪个 0.03。** 本检验针对正文 3.2 节的 M3 命名题库值：SC4 在 160 个封存概念上的命名损失 0.0289（账本 `experiment3-b212cfe8d967`）。正文 3.1 节也有一个 0.03 [−0.05, 0.12]，那是知识测验 240 个概念上的命名损伤（`knowledge-91368a9460f1`），提示来自另一套材料，不在这次检验范围内。家族一侧对照的是 0.78 [0.62, 0.96]（`title_gap_fine_remaining_core-9a0a976b077d`）。

---

2026-10-28，agent1。来源：评审团编辑综合的修改路线图 C1（`reports/review/2026-10-28-full-draft/panel/6-editorial-decision.md`），方法席、期刊契合席、可解释性席和魔鬼代言人都提到。

## 为什么要做

SC4 损伤在命名题库上几乎不伤命名（0.03 nat），在家族题上却伤了 0.78 nat。两套题有两处不同：概念不同；格式不同，题库每道题前有三道示范题，家族题没有。正文现在写"这次运行分不开两种原因"。审稿人会问："SC4 不伤命名"是这组零件的性质，还是示范题让模型绕过了它？这直接关系到"SC4 只管在候选里选、不管说出词"这条结论。

打个比方：一个人在有例句提示时能正常造句，没有例句时就造不出来。我们需要知道 SC4 损伤后的模型是不是这样。

## 怎么做

只做前向计算，模型权重不变，不训练。

| 题目 | 现有格式 | 新增格式 | 状态 |
| --- | --- | --- | --- |
| 命名题库 160 个概念 | 三道示范题（已跑） | 去掉示范题，只留指令和定义 | 完好、WR8、SC4 |
| 家族题 120 个概念的命名 | 无示范题（已跑） | 加同样三道示范题 | 完好、WR8、SC4 |

读数：每个概念的命名损失（目标词对数概率之和，与正文同一计分），按概念重抽样 10,000 次给 95% 区间，批量布局与正式检验相同（16、11、7 三种取平均）。

## 怎么判读（事先写定）

- 如果去掉示范题后，SC4 在题库上的命名损失升到与家族题相近（区间与 0.78 重叠，且下限高于 0.20），并且家族题加示范题后 SC4 的命名损失降到接近零，就说明是**格式**造成的：正文改写为"SC4 在有示范题时不伤命名"，并在讨论里写示范题如何让模型绕过控制。
- 如果两种格式下 SC4 的损失都和原来一样（题库仍接近零、家族题仍约 0.78），就说明是**概念**不同造成的：正文删掉"分不开两种原因"那句，改为"差异来自概念，不来自格式"。
- 介于两者之间的结果，照实报告两个方向的变化量，不强行归类。
- 这是探索性检验，不进入表 S6 的确认性检验族。

## 成本与安排

- 一次开卡：160 × 3 状态 + 120 × 3 状态，每种两格式中新增的一种，约 840 个提示乘三种批量布局，单卡 RTX PRO 6000 预计不到一小时（含加载模型）。
- 要改的代码：原计划在 `src/lesion/m3_scoring.py` 和 `src/lesion/fine_semantics.py` 里加选项；实际改为不动这两个冻结模块，新格式放在 `experiments/e3_errors/demo_format_prompts.py`（见上面第 3 条）；运行脚本 `experiments/e3_errors/run_demo_format.py`、分析脚本 `experiments/e3_errors/analyze_demo_format.py`、测试 `tests/test_demo_format.py`，已在本机小随机模型上跑通。
- 跑完立刻取回输出、关机（按项目规则），结果写入 `reports/explore/` 并入账，再改正文。
