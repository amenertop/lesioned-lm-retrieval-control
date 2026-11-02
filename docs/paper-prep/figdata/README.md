# 论文图表数据

2026-10-20 更新。主图只读 `main-confirmatory.csv`，表 1 读 `table1-confirmatory.csv`。表 2 已有的临床输入读 `table2-clinical-evidence.csv`。它们由 `../build_writing_assets.py` 从 `../results_ledger.csv` 复制，保留账本编号、来源路径、证据性质和样本量。确认性检验同时保留原 p 值。`main-manifest.json` 记录账本和产物的哈希。

## 主图清单

| 文件 | 用途 | 内容 |
| --- | --- | --- |
| `main-confirmatory.csv` | 图 2 至图 4 | 只收确认性终点，figure 列指定图号 |
| `table1-confirmatory.csv` | 表 1 | 全部已完成确认性检验的分量，包括未获支持的预测 |
| `table2-clinical-evidence.csv` | 表 2 的取数底表 | 三族患者参照与模型最终提示、错误构成，保留各自证据性质 |
| `../component-members.csv` | 图 1b | 零件实际成员与坐标，没有效应量 |
| `main-manifest.json` | 产物核对 | 文件哈希和历史文件的补充材料归属 |

图 1 是设计示意。表 2 的底表尚未包含模型最终词频、固定组级预测及其余临床标记，也没有替模型选择对应哪组病人的剂量。读图精度上下界与抽样区间分列。患者的提示恢复取组均值之比，错误构成取逐患者比例的均值，模型分母随每一行列出。开发结果不进入这份底表。

## 历史文件

以 `fig` 开头的旧 CSV、`build.py` 及其旧 provenance 文件保留，均按补充材料使用。它们含有只报告项或探索性分析，不能整张直接放入主图。旧构建程序会读取词频原始来源，本轮没有运行。旧文件的来源哈希反映当时生成的报告，报告元数据修订后不能把旧哈希误称为当前全文件哈希。

主要区别如下。

- 双向分离画预定的两个组间差值。四个单组均值放补充材料。
- WR8 直接贡献画整组总和。各成员的点不共享整组显著性。
- 错误类型画预定聚合对比与超出整体退化的量。四类候选份额放补充材料。
- 提示依赖正式检验使用预先选定的可自由命名概念。探索性 O2、O4、O5 使用可用概念，不能换成正式检验的样本数。
- 原脚本变量 K8、C4 保留。论文标签从 `../component-code-map.md` 取 WR8、SC4。
- 正文可以写两组效应远超全部匹配随机组、理解没有变化。图表中的估计、区间和检验条件照账本列出。

## 重建与核对

用 lesion 环境运行 `docs/paper-prep/build_writing_assets.py`，加 `--check` 只比对，不写文件。此入口只读账本，不读原始下载、不重跑模型，也不改统计判据。完整图表安排见 `../04-figures-tables.md`。

## 逐项数据（2026-10-28 第二轮重画图 2 至图 4；第三轮按新面板改名，图 5 新增）

`build_items.py` 从已完成运行的存档里抽出逐项数据（每个组件、每组随机零件、每个概念、每个家族、每位病人），写成 `items_*.csv`，并把每个输入文件的 SHA-256、每项核对和输出哈希写进 `items-manifest.json`。规则与 `build.py` 相同：每个文件先复现它在账本里的汇总值（例如 200 组随机零件的中位数 0.0367 与最大值 0.388、WR8 在 240 个概念上的词频效应 1.137、五个损伤在 120 个家族上的均值与偏相关），任何一项不符就停下、不写文件；`--check` 只比对不写。图上的估计值与区间仍只按编号读账本，逐项文件只提供散点、直方图和逐概念折线。

| 文件 | 图 | 内容 | 来源 |
| --- | --- | --- | --- |
| `items_fig2a_screening.csv` | 图 2a | 7,680 个组件在两个标记上的梯度分数、每个方向的 32 个候选、WR8 与 SC4 成员 | `outputs/server/e3_knowledge/.../screen/attribution.csv`、`screen.csv`；`outputs/server/outputs/e1_screen/.../attribution/attribution.csv`、`screen/screen.csv`；`component-members.csv` |
| `items_fig2b_random_sets.csv` | 图 2b | 两个标记各 200 组匹配随机零件的效应 | `outputs/server/e2_confirm/.../dissociation/naming_scores.parquet`（单一批次计分，逐组重算后与账本核对）；`.../control/per_set_control_effect.csv` |
| `items_fig2d_wr8_concepts.csv` | 图 2d | 240 个封存概念的 Zipf、低频标记、WR8 与 SC4 命名损失（三种批次平均） | `naming_scores.parquet`；`data/materials/m3_knowledge/items.csv` |
| `items_fig3d_graded.csv` | 图 3c、补充图 S3 | 160 个题库概念在 WR8、SC4 保留 75、50、25、0% 时的命名与六选一损失 | `outputs/e3_errors/.../optogenetics_confirm/candidates.parquet`、`labels.parquet`（哈希与冻结报告一致） |
| `items_fig4a_patient_errors.csv` | 图 4a | 10 位 SD、8 位 SA 病人的同类、上位、联想错误份额 | `data/literature/jefferies2006_individual.csv`（Jefferies & Lambon Ralph 2006 表 5） |
| `items_fig5b_families.csv` | 图 5c | 120 个最终家族在 WR8、SC4、中层 10% 保留、两个学习损伤下的命名与释义选择损失及偏相关协变量（target 读数） | `outputs/title_gap_execution_20261019/core_readout_20261020/fine-target-items.csv`、`remaining_core_20261020/broad/target-items.csv` |
| `items_fig4d_patient_cueing.csv` | 图 4d | 8 位 SD、6 位 SA 病人在无提示、两音素提示、末档提示下 30 题的答对数与各自的救回比例（细线）；组值仍用账本的合并比值（所有初始失败合在一起），核对：逐人均值四舍五入后等于 `jefferies2008_cueing.csv` 的组均值，合并比值与账本相差不到 0.0015 | `data/literature/jefferies2008_cueing_individual.csv`（Jefferies 等 2008 图 2 逐条转录，`tools/measure_jefferies2008_fig2.py`） |
| `items_fig5c_badges.csv` | 图 5c（徽章） | 图 5 五个编号损伤（WR8、SC4、中层 10% 保留、两个学习损伤）各自作用的层、每层组件数与关闭量；核对：指纹与 fine-points.json、broad points.json 一致，门数与关闭量与读数中的干预记录一致，WR8、SC4 的门等于成员表 | `outputs/title_gap_execution_20261019/fine_final_core/plan.json` |

图 4c 的逐概念散点用 `build.py` 已有的 `fig4c_cue_vs_loss.csv`（100 个自由命名概念，Spearman 与正式报告一致），图 5b 只读账本（32 个损伤状态的释义选择损失与偏相关，target 读数）；补充图 S2 只读账本（九个权重掩蔽损伤与 WR8、SC4 的命名损失和频率斜率），补充图 S3 用 `fig3b_dla.csv` 与 `items_fig3d_graded.csv`。
