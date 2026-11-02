# 从已发表文献摘出的病人数据与健康人“虚拟损伤”数据

原用途是论文表 2 的记分卡和“盲诊断”（`docs/plans/p8-clinical-alignment.md` 第 (a) 项，用户 2026-10-12 定 P8，2026-10-13 定 P9 甲）。旧记分卡把两组病人的数值和两类组件损伤的数值逐项并排，只比较方向和有无，计算见 `experiments/e3_errors/analyze_scorecard.py`。病人的正确率和模型的对数概率不直接比较大小。2026-10-19 批准的组级数值比较另用相同类型的行为读数，明确分母、标定与读图精度，见本页末节。

| 文件 | 来源 | 内容 |
| --- | --- | --- |
| `jefferies2006_markers.csv` | Jefferies, E., & Lambon Ralph, M. A. (2006). Semantic impairment in stroke aphasia versus semantic dementia: a case-series comparison. *Brain*, 129, 2132–2147 | 语义性痴呆（SD）10 例与卒中后语义障碍（原文称 CVA，本项目称 SA）10 例：病灶、理解测验、命名、熟悉度效应、控制需求效应、跨任务一致性、命名错误类型、首音提示。盲诊断的参照列只用这一篇的表 2 |
| `jefferies2008_cueing.csv` | Jefferies, E., Patterson, K., & Lambon Ralph, M. A. (2008). Deficits of knowledge versus executive control in semantic cognition: Insights from cued naming. *Neuropsychologia*, 46, 649–658 | SD 8 例、SA 6 例看图命名时逐个加音素的提示：不提示、给两个音素、给到只差最后一个音素时的正确率，提示获益，组间检验，词频相关 |
| `jefferies2008_cueing_individual.csv` | 同上一篇，图 2 | 上一行组均值背后的逐人数值：SD 8 例、SA 6 例 × 不提示、给两个音素、只差最后一个音素，每格是 30 题中答对的题数与比例。相邻 JSON 记录读图方法、坐标标定、核对结果和 PDF 哈希，由 `tools/measure_jefferies2008_fig2.py` 生成（2026-10-28） |
| `hoffman2011_frequency.csv` | Hoffman, P., Rogers, T. T., & Lambon Ralph, M. A. (2011). Semantic diversity accounts for the “missing” word frequency effect in stroke aphasia. *Journal of Cognitive Neuroscience*, 23, 2432–2446 | SD、SA 各 13 例的同义判断：高、低词频 × 三档具体性的正确率，词频效应的检验，逐题回归中词频与语义多样性的系数 |
| `tms_virtual_lesion.csv` | Pobric, G., Jefferies, E., & Lambon Ralph, M. A. (2007). *PNAS*, 104, 20137–20141；Whitney, C., Kirk, M., O’Sullivan, J., Lambon Ralph, M. A., & Jefferies, E. (2011). *Cerebral Cortex*, 21, 1066–1075 | 健康人的经颅磁刺激“虚拟损伤”：刺激左前颞叶后命名与同义判断变慢、数字任务不变（Pobric）；刺激左额下回、左后颞中回后只有弱联系的语义判断变慢，强联系与字母任务不变，头顶对照点不变（Whitney） |

每一行写明：
- `source`：表号、图号，或结果部分的小节名（TMS 文件的来源列前面带作者和年份）；
- `how`：`table`（照表抄录）、`text`（照正文抄录，统计量原样保留）、`figure estimate`（从图上读出）、`figure estimate (difference)`（两个读图值相减）。旧记分卡只用读图值比较方向，新组级比较同时传播读图精度。
- `n`：该数值所基于的人数，原文没有明确给出的留空。

读图的精度：
- Jefferies 2006 图 2、图 4：约 2 个百分点。
- Jefferies 2008 图 2：按像素量出每位病人的柱高，都落在 1/30 的整数倍上（测验 30 题），各人平均后与图中“Average”两根柱子相差不超过 0.002。表中的组均值就是这样算出的。
- Hoffman 2011 图 3：按像素量出标记的中心，约 0.02；三角形标记约偏低 0.01。
- Pobric 2007 图 1：按像素量出柱高，约 5 毫秒。同义判断从 1697 到 1866 毫秒，慢 10.0%，与正文的 9.9% 相符。
- Whitney 2011 图 5：Europe PMC 提供的图只有 493 × 518 像素，约 10 毫秒；显著性以正文的统计量为准。

需要注意的几点：
- 首音提示（Jefferies 2006 图 4）的 SD 组是另外 3 位病人（EK、BS、GE），不是表 4、表 5 的那 10 位。原文说明，SD 组没有这项测验的数据，所以另找了 3 位作对照。
- Markdown 版 Jefferies 2006 表 5 的语义错误小类中有几个“100”，按同一行的其余数值应为 1.00。组均值不受影响。
- Jefferies 2008 用的是作者存档的投稿版（White Rose 3725），不是期刊排版版；数字可能与正式版有出入，引用前对照正式版。它的 SA 6 例是 Jefferies 2006 那 10 例中的 6 例；SD 8 例中的 EK、GE 也出现在 2006 年的图 4 里。
- Jefferies 2008 里 SD 的提示获益“小但显著”（两个音素：t(7) = 4.5），SA 的获益更大。2006 年表 2 把 SD 记为“无”。记分卡的参照列照 2006 年，两篇的差别在补充列里照写。
- Jefferies 2008 图 2 的逐人表（2026-10-28 用户批准抄录）：42 根病人柱子离最近的 1/30 都不超过 0.0025，与 PDF 里的矢量坐标相差不超过 0.0005。每根“Average”柱子都正好对应各人答对题数之和，SA 不提示那根与各人平均差 0.0022：原图的坐标存在约 0.004 的网格上，所以上面“不超过 0.002”对这一根略紧。用逐人数值重算的配对 t 是 3.8、4.2、2.7 和 4.6（原文 4.5），组间 F 是 7.6、4.8（原文 8.0、5.0）。不加连续性校正的 McNemar 检验重现了原文的显著人数，但两个音素时的 p 值范围（原文两组都印作 .06 到 .0002）和 SD 最后提示的最小 p（.0005，原文 .0002）没有重现。论文和账本里的救回比例（SD .18、.36，SA .57、.90）是组均值之比；逐人比例的平均数是 SD .23、.42，SA .57、.85。
- Hoffman 2011 的 SA 组只有在控制语义多样性之后才显出正向的词频效应；不控制时没有（t(12) = 0.18），高具体性的词甚至反过来。
- Pobric 2007 的命名实验：交互作用的自由度原文印作 F(2,10)，照抄。“具体层级比基本层级更慢”是特异性效应，不是词频效应，记分卡里只作近似。
- 原文 PDF 与文字版在 `article/`（不进 git）：`jefferies2006`、`jefferies2008`、`hoffman2011`、`pobric2007`（.pdf 与 .md），`whitney2011`（Europe PMC 全文 XML 与 .md）。Almaghyuli 等 2012、Lambon Ralph 等 1998 没有找到开放获取的版本，未摘。

摘录由 agent1 完成（2026-10-13）。同日 agent2（经用户的中转，agent3，xhigh）逐行对照 `article/` 的原文核对了四个文件：没有发现数值、统计量、自由度、p 值、人数、单位或组别的抄录错误；读图估计在内部计算上一致，并符合正文给出的约束。按它的意见：
- 2006 年图 4 的第三位额外 SD 病人：正文只点名 BS 和 EK，已对照原文图 4 的柱子标签确认是 EK、BS、GE；
- Hoffman 2011 的一个值由 `-0.0` 改写为 `0.0`；
- 补摘了它列出的几项统计：2008 年最后提示对两音素提示的组内检验、个体病人的 McNemar 范围、复合语义分与命名的相关；Hoffman 2011 的词频主效应与词频 × 具体性交互；Pobric 2007 的任务与 TMS 主效应；Whitney 2011 错误率的 ANOVA。

旧记分卡中的读图估计只用于比较方向。新数值比较将它们作为带精度范围的估计值，不能称为原文报告的精确均值。

## 2026-10-19 补充的逐患者表

`jefferies2006_individual.csv` 由 agent3 对照 Zotero 原版 PDF 的表 4、5 转录。相邻 JSON 保存论文 DOI、原文件哈希、机会水平来源和转换规则。每行有病例代号、组别、任务、单位、分母、印刷值和页码。`TA` 表示测试中止，`NT` 表示未测，没有补成零。

表 5 中的三个 `100` 在原版 PDF 中也是这样印的，数值列按比例记为 `1.00`，印刷列保留 `100`。表 4 的命名总分与表 5 的命名比例没有强行互相改写。原始错误次数没有完整给出，不根据舍入比例倒造次数。

新表提供患者的多任务总分，不能重建逐题对错或逐题熟悉度效应。数值拟合的共同任务与尺度要求见 `docs/plans/title-gap-runbook.md`。旧四个文件及原记分卡的转录没有修改。

## 组级数值比较的后续用途

用户授权的补实验丁将三个队列的词频、提示恢复和错误构成组成组级参照，方法见 `docs/plans/clinical-profile-comparison.md`。前述读图数值用于带明确精度范围的敏感性分析，不改称原文精确均值。输出保留源行与任务分母，不将不同队列拼成同一个患者。语音提示与模型的字母提示按功能近似比较，原始任务名称同时保留。

2026-10-19 找到曼彻斯特大学保存的 [Jefferies 2008 正式版入口](https://research.manchester.ac.uk/en/publications/deficits-of-knowledge-versus-executive-control-in-semantic-cognit/)。PDF 下载返回 403，未完成正式版图值的逐柱核对。提示数字继续使用已核对的作者存档稿转录，未将来源版本改写为正式排版版。
