# Dissociating word retrieval from semantic control in lesioned language models

Code and data for He Wang, Jingbo Chen and Wei Yuan, *Dissociating word retrieval from semantic control in lesioned language
models* (preprint, link to be added).

We located components of a pretrained mixture-of-experts language model (Qwen3-30B-A3B-Instruct-2507) by the clinical
markers of semantic dementia (SD) and semantic aphasia (SA) and lesioned them. Lesioning seven experts and one
attention head (WR8) impaired naming from definition most for rare words while word-to-definition matching stayed
intact. Lesioning four attention heads (SC4) impaired choices against strong distractors and, at the answer, made
misleading cues more costly.

## Contents

| Path | Contents |
| --- | --- |
| `src/lesion/` | Model loading at pinned revisions (`models.py`), component ablation (mean replacement and graded ablation), scoring, statistics |
| `experiments/` | Entry scripts and configurations, one folder per experiment (see below) |
| `tests/` | Tests of the ablation hooks and scoring on a tiny random model; no GPU needed |
| `env/` | Environment setup for a Linux GPU server and for Windows |
| `data/materials/` | The item sets. M1 (semantic control judgments) is rebuilt from its source, see below. How the items were made and reviewed: `data/materials/ITEM_REVIEW.md` |
| `data/literature/` | Patient data transcribed from published papers; every row names its table or figure |
| `data/neutral_text/` | 200 passages from the WikiText-103 validation split, for component means and language-model loss |
| `data/raw/SOURCES.md` | Where each third-party source comes from, with checksums |
| `docs/analysis-plan/` | Analysis rules and component selections fixed before the held-out items were examined |
| `FREEZE.md` | The freeze tags of the analysis plan, with commits, dates and the checksum of every frozen file |
| `docs/paper-prep/results_ledger.csv` | Every number reported in the paper, with the output file and field it was read from |
| `docs/paper-prep/figdata/`, `docs/paper-prep/figures/r/` | Data and R scripts for Figures 2 to 5 |
| `docs/paper-prep/component-code-map.md`, `component-members.csv` | Members of every component set, and repository codes against paper codes |
| `manuscript/Supplementary_Material.pdf` | The Supplementary Material of the paper |
| `reports/explore/2026-11-02-case-rescore/`, `reports/explore/2026-11-02-gpu-session/` | Results of the analyses added on 2 and 3 November 2026 (see below), in the form the results ledger reads |
| `docs/plans/phi-battery.md`, `docs/plans/demo-format-check.md` | Plans of two exploratory checks, with their reading rules fixed before the runs |

The repository uses its own codes for the component sets: K8 is WR8, C4 is SC4, K30 is WR30, K8b is WR8b, C4s is SC4b,
and the P14 control set is SC32 (full table in `docs/paper-prep/component-code-map.md`). WR8 is `expert:46:61`,
`expert:46:101`, `head:37:24`, `expert:45:32`, `expert:45:48`, `expert:43:70`, `expert:42:115` and `expert:45:93`;
SC4 is `head:9:15`, `head:27:9`, `head:25:24` and `head:24:5` (layer and index, zero based).

The column names of the results ledger are in Chinese: 编号 id, 结论一句话 claim, 数值 value, 95%区间下限 and
95%区间上限 the 95% interval, 单位 unit, n, n单位 unit of n, n说明 note on n, 性质 status (confirmatory or
exploratory), 数据集 data set, 指标路径 measure, 来源文件 source file, 来源字段路径 field, 区间字段路径 interval field,
来源SHA256 checksum of the source file, p原始 raw p, p校正 corrected p, p字段路径 p field, 判定或检验族 test family,
备注 remarks.

## Setup

Python 3.12 with PyTorch 2.8.0 (CUDA 12.8) and the packages in `requirements.txt`; `env/setup_cloud.sh` and
`env/setup_local.ps1` build the environment. The model is loaded at revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` and needs about 61 GB of GPU memory in 16-bit precision; we used one NVIDIA
RTX PRO 6000 with 96 GB. The tests run on a CPU:

    python -m pytest -q

## Third-party sources

Some sources may not be redistributed, so they are not in this repository. `python scripts/fetch_sources.py`
downloads the Wu and Hoffman (2022) files from OSF and rebuilds `data/materials/m1/items.csv` and
`data/materials/m1/associations.csv`, checking their SHA-256 against the files used in the paper. The other sources
(SUBTLEX-US, WordNet 3.0, the Small World of Words norms SWOW-EN18, the University of South Florida free association
norms and WikiText-103) are listed with their download addresses and checksums in `data/raw/SOURCES.md`.
`data/materials/m3/menus.csv`, an intermediate file of the M3 curation that holds association strengths from SWOW-EN18
and the USF norms, is left out for the same reason. The final M3 items are included.

## Reproducing the results

Each row of the results ledger names the output file and field it came from. The scripts in `experiments/` write these
outputs to `outputs/`, which is not included (1.3 GB). Some scripts cite planning documents under `docs/plans/`, which
are not part of this release.

| Folder | What it runs |
| --- | --- |
| `experiments/e1_screen/` | Model screening and component scoring (attribution patching against single ablations) |
| `experiments/e2_confirm/` | Held-out tests of the control set, position-restricted lesions, language-model loss checks |
| `experiments/e3_knowledge/` | Tests of the retrieval set: expansion, split and joint lesions |
| `experiments/e3_errors/` | Naming-error and cueing batteries, item consistency, frequency analyses |
| `experiments/title_gap/` | Final batteries on the semantic families, homograph families, learned lesions and the patient comparison |
| `experiments/second_model/` | The exploratory second model, Phi-3.5-MoE |
| `experiments/p14/` | Learned ablation gates |
| `experiments/prep_means/` | Component means on the neutral text |
| `experiments/e0_baseline/` and the `gpu_session*.sh` scripts | Earlier baselines and the batch scripts of each GPU session |

Analyses added on 2 and 3 November 2026, all exploratory:

| Script | What it does |
| --- | --- |
| `experiments/title_gap/rescore_case.py`, `analyze_case.py` | Scores the family names with the first letter capitalized and recomputes every family statistic with both spellings, log[P(word) + P(Word)] (Supplementary Material S3.12) |
| `experiments/e3_errors/run_demo_format.py` | The naming battery without and the families with the three demonstration items |
| `experiments/e3_errors/run_random_answer_heads.py` | SC4 against 100 matched random sets of four heads, lesioned at the answer positions |
| `experiments/e3_knowledge/run_generated_frequency.py` | Greedy answers on the held-out naming concepts, for the frequency effect in what the model says |
| `experiments/second_model/run_phi_battery.py`, `output_bias.py` | Phi-3.5-MoE on the naming battery, with its output-layer bias added to every score (Supplementary Material S1.1) |
| `experiments/analyze_gpu_20261102.py` | Analysis of the last four, including the reading rules of the Phi plan |

## Licenses

Code: MIT (`LICENSE`). Our data (item sets, results ledger, figure data, analysis plan): CC BY 4.0
(`DATA_LICENSE.md`). `data/neutral_text/` is derived from WikiText-103 and keeps its CC BY-SA 3.0 license. The M3 items
quote WordNet 3.0 glosses under the WordNet license. The homograph norms come from Gilbert and Rodd (2022), and the
values in `data/literature/` from the publications each row cites; please cite those sources when you use them.

## Citation

Wang, H., Chen, J., and Yuan, W. (2026). Dissociating word retrieval from semantic control in lesioned language models. Preprint.
