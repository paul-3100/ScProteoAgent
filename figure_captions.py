# -*- coding: utf-8 -*-
r"""Data-driven publication-grade figure caption generator (figure_upgrade_20260904 Step 03).

Builds ``figures_captions.md`` — one structured caption per key figure, each with:

- ``图号 + 图题`` (figure number + title)
- ``面板逐项说明`` (panel-by-panel description of what is drawn)
- ``如何阅读`` (how to read the encoding channels: colour / size / direction)
- ``阈值与样本量`` (significance thresholds, display scale, sample sizes)
- ``统计口径`` (statistical basis: two-sided limma, BH/FDR correction, effect size)
- ``一句话解读`` (one-sentence interpretation tied to the run's main claim)
- ``证据边界`` (evidence boundary: current_matrix / offline_enrichment / extension)
- ``对应原论文图`` (original-paper figure mapping, when available)

Inputs (all from the run folder / project, no network):

1. ``figure_manifest.json`` — per-figure ``caption_metadata`` promoted by the plot
   functions (Step 02), plus ``params`` for runs generated before that migration.
2. ``analysis_design.used.yaml`` (fallback ``analysis_design.inferred.yaml``) —
   differential thresholds, FDR method and primary group column.
3. ``evaluation_evidence/*.csv`` — sample composition (``group_composition_qc.csv``),
   per-contrast up/down counts and top proteins (``core_story_evidence.csv``),
   curated modules (``curated_module_group_summary.csv``).
4. ``report_story_outline.json`` — the run-level main claim used to tie the
   interpretation to the report's storyline.
5. ``docs/figure_upgrade_20260904/original_figure_map.json`` (Step 01) — the
   original-paper figure mapping (``matched_agent_plots`` inverted per figure file).

Every caption field is rendered from data or carries an explicit "not recorded"
sentence — the module never emits empty cells (工程纪律：表格不输出空单元格) and
never invents numbers. ``figures.md`` / ``figures_preview_local.md`` are slimmed to
图号标题 + 嵌图 + 一行指向 ``figures_captions.md`` (render helpers included here so
``main_agent.py`` only orchestrates file writing).

The module depends only on the standard library plus the sibling ``report_language``
registry, so it stays importable in any environment, including minimal scoring venvs.

Offline sample on an existing run (writes into a preview directory, never touches
the accepted run):

    python figure_captions.py --run-folder runs_v3\\<run>\\<dataset>_gpt-5-mini_<date> \\
        --figures qc_sample_overview.png,pca_plot.png,... \\
        --out-dir docs\\figure_upgrade_20260904\\preview\\<dataset>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from report_language import register, t

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTIONS_FILENAME = "figures_captions.md"
FIGURES_MD_FILENAME = "figures.md"
FIGURES_LOCAL_MD_FILENAME = "figures_preview_local.md"
ORIGINAL_FIGURE_MAP_REL = os.path.join("docs", "figure_upgrade_20260904", "original_figure_map.json")

CATEGORY_ORDER = ("QC", "Sample structure", "Differential evidence", "Mechanism evidence", "Enrichment evidence")

# ---------------------------------------------------------------------------
# reader-facing text (zh released verbatim, en opt-in)
# ---------------------------------------------------------------------------

# One flat table of (zh, en) templates. The zh column repeats the released literals
# character for character, so a zh run still renders the same figures_captions.md;
# the en column is the added language and never falls back to Chinese. Sentences that
# interpolate a number, a group or a contrast are assembled in code from the fragments
# below, which is what keeps every registered zh value a literal of the released file.
_CAPTIONS = {
    # ---- shared punctuation (the released zh text uses full-width marks) ----
    "punct_period": ("。", "."),
    "punct_semicolon": ("；", "; "),
    "punct_enum": ("、", ", "),
    "punct_comma": ("，", ", "),
    "punct_paren_open": ("（", " ("),
    "punct_paren_close": ("）", ")"),

    # ---- evidence boundaries (the released BOUNDARY_* module strings) ----
    "boundary_matrix": (
        "current_matrix：图中数字由当前定量矩阵直接计算，可在 run 目录内复现。",
        "current_matrix: the numbers in this figure are computed directly from the current "
        "quantification matrix and can be reproduced inside the run directory.",
    ),
    "boundary_enrichment_a": (
        "offline_enrichment：富集条目来自当前矩阵差异蛋白的离线通路注释，支撑机制解释，",
        "offline_enrichment: the enrichment terms come from offline pathway annotation of the "
        "differential proteins in the current matrix and support mechanistic interpretation, ",
    ),
    "boundary_enrichment_b": (
        "不构成矩阵内的直接测量或因果证据。",
        "but are neither a direct measurement in the matrix nor causal evidence.",
    ),
    "boundary_mixed_a": (
        "current_matrix + offline_enrichment：热图数值来自当前矩阵的组均值 row Z-score，",
        "current_matrix + offline_enrichment: the heatmap values are group-mean row Z-scores "
        "from the current matrix, ",
    ),
    "boundary_mixed_b": (
        "蛋白集合来自离线富集通路条目。",
        "and the protein set comes from an offline enrichment pathway term.",
    ),
    "boundary_extension": (
        "- extension：启发式推测和后续验证建议只作为可检验假设，不进入图注数字。",
        "- extension: heuristic inference and suggestions for follow-up validation are "
        "testable hypotheses only and never enter caption numbers.",
    ),
    "paper_ref_fallback": (
        "本图为 Agent 标准分析图，原论文中没有直接对应的面板（映射基线见 original_figure_map 提取记录）。",
        "This is a standard Agent analysis figure with no directly corresponding panel in the "
        "original paper (for the mapping baseline see the original_figure_map extraction record).",
    ),
    "paper_ref_prefix": ("对应原论文 ", "Corresponding original-paper panels: "),

    # ---- figure titles ----
    "title_qc_sample_overview": ("样本与缺失率 QC", "Sample and missingness QC"),
    "title_pca": ("PCA 样本结构", "PCA sample structure"),
    "title_umap": ("UMAP 样本结构", "UMAP sample structure"),
    "title_heatmap": ("差异蛋白热图", "Differential protein heatmap"),
    "title_differential_summary_barplot": (
        "差异蛋白数量概览",
        "Overview of differential protein counts",
    ),
    "title_key_protein_overview": ("关键蛋白总览", "Key protein overview"),
    "title_protein_contrast_bubble": (
        "候选蛋白对比气泡图",
        "Candidate protein contrast bubble plot",
    ),
    "title_mechanism_top_protein_group_means": (
        "核心蛋白组均值图",
        "Group mean abundance of core proteins",
    ),
    "title_mechanism_enrichment_dotplot": (
        "机制富集 dotplot",
        "Mechanism enrichment dot plot",
    ),
    "title_mechanism_dep_overlap_upset": (
        "差异蛋白重叠 UpSet 图",
        "UpSet plot of overlapping differential proteins",
    ),
    "title_mechanism_contrast_evidence": (
        "单对比综合证据面板",
        "Integrated evidence panel for a single contrast",
    ),
    "title_enrichment_bubble": ("富集气泡图", "Enrichment bubble plot"),
    "title_mechanism_pathway_gene_heatmap": (
        "通路基因热图",
        "Pathway gene heatmap",
    ),
    "volcano_title_suffix": (" 火山图", " volcano plot"),
    "ce_title_suffix": (" 综合证据面板", " integrated evidence panel"),
    "pathway_heatmap_title_prefix": ("通路基因热图：", "Pathway gene heatmap: "),
    "enrichment_bubble_title_suffix": ("富集气泡图", " enrichment bubble plot"),
    "enrichment_bubble_title_fallback_suffix": (" 富集气泡图", " enrichment bubble plot"),
    "figure_no_prefix": ("图", "Figure "),

    # ---- QC panel names (the drawn panel titles are English, so en reuses them) ----
    "qc_panel_detected_proteins": (
        "检出蛋白面板（Detected Proteins Per Sample）",
        "Detected Proteins Per Sample",
    ),
    "qc_panel_missing_rate": (
        "缺失率面板（Missing Rate By Cluster）",
        "Missing Rate By Cluster",
    ),
    "qc_panel_sample_counts": (
        "样本数面板（Sample Count By Cluster）",
        "Sample Count By Cluster",
    ),
    "qc_panel_sample_correlation": (
        "相关性面板（Sample Correlation）",
        "Sample Correlation",
    ),

    # ---- small shared labels ----
    "unnamed_group": ("未命名分组", "Unnamed group"),
    "group_col_default": ("分组", "group"),
    "group_a_default": ("A 组", "group A"),
    "group_b_default": ("B 组", "group B"),
    "not_recorded": ("未记录", "not recorded"),
    "not_recorded_info": (
        "figure_manifest 未记录该信息。",
        "figure_manifest does not record this information.",
    ),
    "no_threshold_recorded": (
        "figure_manifest 未记录该图阈值",
        "figure_manifest records no threshold for this figure",
    ),
    "fdr_correction_suffix": (" 校正）", " correction)"),
    "relative_to": (" 相对 ", " versus "),
    "direction_upregulated": ("上调", "up-regulated"),
    "direction_downregulated": ("下调", "down-regulated"),
    "direction_default": ("对应方向", "the matching direction"),
    "contrast_default": ("对应对比", "the matching contrast"),

    # ---- sample size / threshold / paper reference / focus genes ----
    "sample_missing": (
        "样本量：run 内缺少 group_composition_qc.csv，未记录分组样本数（可在 evaluation_evidence 目录核对）。",
        "Sample size: group_composition_qc.csv is missing from the run, so per-group sample "
        "counts were not recorded (check the evaluation_evidence directory).",
    ),
    "sample_size_prefix": ("样本量：N=", "Sample size: N = "),
    "sample_group_col": ("分组列 ", "grouping column: "),
    "mean_detected_prefix": ("平均每样本检出 ", "mean detected proteins per sample: "),
    "mean_missing_prefix": ("平均缺失率 ", "mean missing rate: "),
    "protein_count_suffix": (" 个蛋白", " proteins"),
    "threshold_label": ("显著性阈值：", "Significance thresholds: "),
    "threshold_not_recorded": (
        "显著性阈值：figure_manifest 未记录该图阈值（差异分析默认口径见文档总览）。",
        "Significance thresholds: figure_manifest records no threshold for this figure "
        "(see the documentation overview for the differential-analysis defaults).",
    ),
    "focus_genes_prefix": ("报告 focus 基因（", "Report focus genes ("),
    "focus_genes_more": ("等", " and others"),
    "focus_genes_suffix": ("）可在图中对照追踪。", ") can be traced in the figure."),

    # ---- QC caption ----
    "qc_interp_base": (
        "QC 面板给出后续所有差异与模块分析的样本基础",
        "The QC panels define the sample basis for every later differential and module analysis",
    ),
    "qc_total_prefix": ("共 ", "a total of "),
    "qc_total_suffix": (" 个样本（", " samples ("),
    "qc_unannotated_prefix": ("另有 ", "a further "),
    "qc_unannotated_suffix": (
        " 个样本缺少分组注释，图中统一显示为 Unannotated",
        " samples lack a group annotation and are shown as Unannotated in the figure",
    ),
    "qc_large_n_note": (
        "样本量超过 600 时检出蛋白面板自动切换为排序折线（附 Q1/中位数/Q3 分位参考线），缺失率散点按固定随机种子抽样",
        "above 600 samples the detected-protein panel switches to a sorted line plot (with "
        "Q1/median/Q3 reference lines) and the missing-rate points are subsampled with a "
        "fixed random seed",
    ),
    "qc_panel_detected_line": (
        "检出蛋白面板（Detected Proteins Per Sample）：每个样本一根柱，柱高为该样本检出蛋白数。",
        "Detected proteins panel (Detected Proteins Per Sample): one bar per sample, its "
        "height being the number of proteins detected in that sample.",
    ),
    "qc_panel_detected_line_large": (
        "检出蛋白面板（Detected Proteins Per Sample）：样本按检出蛋白数排序绘制折线，标注 Q1/中位数/Q3 分位参考线与四分位带（样本量较大时替代逐样本柱状图）。",
        "Detected proteins panel (Detected Proteins Per Sample): samples are drawn as a "
        "line plot sorted by detected-protein count, with Q1/median/Q3 reference lines and "
        "an interquartile band (replaces the per-sample bar chart at large sample sizes).",
    ),
    "qc_panel_missing_line": (
        "缺失率面板（Missing Rate By Cluster）：按分组汇总的平均缺失率。",
        "Missing rate panel (Missing Rate By Cluster): mean missingness summarised by group.",
    ),
    "qc_panel_missing_line_large": (
        "缺失率面板（Missing Rate By Cluster）：按分组汇总的缺失率箱线图，叠加样本散点（样本量较大时散点按固定随机种子抽样，控制图形密度）。",
        "Missing rate panel (Missing Rate By Cluster): boxplots of missingness summarised by "
        "group with the sample points overlaid (at large sample sizes the points are "
        "subsampled with a fixed random seed to control the density).",
    ),
    "qc_panel_counts_line": (
        "样本数面板（Sample Count By Cluster）：各分组样本数。",
        "Sample count panel (Sample Count By Cluster): number of samples in each group.",
    ),
    "qc_panel_correlation_line": (
        "相关性面板（Sample Correlation）：样本两两 Spearman 相关系数热图，颜色范围 -1 至 1。",
        "Sample correlation panel (Sample Correlation): heatmap of pairwise Spearman "
        "correlations between samples, colour scale from -1 to 1.",
    ),
    "panel_omitted_prefix": ("面板省略说明：", "Omitted panels: "),
    "panel_omitted_suffix_qc": (
        " 因无匹配数据未绘制，不使用占位图。",
        " were not drawn because no matching data were available; no placeholder panel is used.",
    ),
    "qc_reading_a": (
        "各面板依次回答：每个样本测到多少蛋白、缺失了多少、分组是否均衡、样本间重复性如何；",
        "The panels answer, in turn, how many proteins each sample yielded, how much is "
        "missing, whether the groups are balanced and how reproducible the samples are; ",
    ),
    "qc_reading_b": (
        "相关系数越接近 1，两个样本的蛋白图谱越一致。",
        "the closer the correlation is to 1, the more similar the protein profiles of two samples.",
    ),
    "qc_threshold_note": (
        "统计性质：描述性 QC 汇总，不设显著性阈值。",
        "Statistical nature: a descriptive QC summary; no significance threshold applies.",
    ),
    "qc_statistics": (
        "描述性统计；相关性面板使用 Spearman 相关系数。",
        "Descriptive statistics; the correlation panel uses Spearman correlation coefficients.",
    ),

    # ---- PCA caption ----
    "pca_interp_prefix": (
        "PCA 展示 ",
        "PCA shows the overall separation and dispersion along ",
    ),
    "pca_interp_suffix": (
        " 维度的整体分离与离散程度，为后续各对比差异方向提供矩阵内结构参照。",
        ", providing an in-matrix structural reference for the direction of the differences "
        "tested later.",
    ),
    "claim_prefix": ("结合主结论「", "Read it together with the main claim, “"),
    "claim_suffix": ("」阅读。", "”."),
    "pca_panel_scatter": (
        "PCA 散点：每个点为一个样本，横轴 PC1、纵轴 PC2，轴标签附各主成分解释方差百分比。",
        "PCA scatter: one point per sample, PC1 on the x axis and PC2 on the y axis, with "
        "the explained-variance percentage of each component appended to the axis labels.",
    ),
    "pca_panel_groups": (
        "分组中心与椭圆：同组样本的中心位置与离散范围（按组着色）。",
        "Group centres and ellipses: centre and spread of the samples of each group, coloured by group.",
    ),
    "pca_reading_a": (
        "点间距离越近表示两个样本的蛋白丰度总模式越相似；颜色对应 ",
        "The closer two points lie, the more similar the overall protein abundance profiles "
        "of the two samples; colour encodes the ",
    ),
    "pca_reading_b": (
        " 分组；同组椭圆与异组椭圆重叠越少，组间整体分离越明显。",
        " grouping; the less the ellipses of one group overlap those of the other groups, the "
        "clearer the overall separation between groups.",
    ),
    "pca_threshold_note": (
        "降维设置：基于当前矩阵蛋白丰度的方差结构，不设显著性阈值。",
        "Dimensionality-reduction settings: based on the variance structure of protein "
        "abundance in the current matrix; no significance threshold applies.",
    ),
    "pca_statistics": (
        "描述性降维（PCA），不涉及假设检验；PC1/PC2 解释方差比例标注于轴标签。",
        "Descriptive dimensionality reduction (PCA) without hypothesis testing; the "
        "explained-variance fractions of PC1/PC2 are annotated on the axis labels.",
    ),

    # ---- UMAP caption ----
    "umap_interp_prefix": (
        "UMAP 展示 ",
        "UMAP shows the neighbourhood structure of the samples across ",
    ),
    "umap_interp_suffix": (
        " 维度样本的邻域结构，可与 PCA 互为对照判断分组结构是否稳健。",
        ", giving a cross-check against PCA for whether the grouping structure is robust.",
    ),
    "umap_panel_scatter": (
        "UMAP 散点：每个点为一个样本，横轴 UMAP 1、纵轴 UMAP 2，按组着色并附图例。",
        "UMAP scatter: one point per sample, UMAP 1 on the x axis and UMAP 2 on the y axis, "
        "coloured by group with a legend.",
    ),
    "umap_reading_a": (
        "相邻点表示蛋白丰度模式相似的样本；颜色对应 ",
        "Neighbouring points are samples with similar protein abundance profiles; colour "
        "encodes the ",
    ),
    "umap_reading_b": (
        " 分组；UMAP 保留局部邻域结构，同组点的聚集比全局距离更可比较。",
        " grouping; UMAP preserves local neighbourhood structure, so the clustering of "
        "points within a group is more comparable than absolute distances.",
    ),
    "umap_threshold_note": (
        "降维设置：基于当前矩阵蛋白丰度的邻域结构，不设显著性阈值。",
        "Dimensionality-reduction settings: based on the neighbourhood structure of protein "
        "abundance in the current matrix; no significance threshold applies.",
    ),
    "umap_statistics": (
        "描述性降维（UMAP），不涉及假设检验。",
        "Descriptive dimensionality reduction (UMAP) without hypothesis testing.",
    ),

    # ---- differential protein heatmap ----
    "heatmap_scale_top_prefix": ("行数为按信息量挑选的 top ", "rows are the top "),
    "heatmap_scale_top_suffix": (
        " 个蛋白",
        " proteins selected by information content",
    ),
    "heatmap_scale_all": (
        "行数为信息量最高的差异蛋白",
        "rows are the differential proteins with the highest information content",
    ),
    "heatmap_panel_prefix": ("聚类热图：", "Clustered heatmap: "),
    "heatmap_panel_mid": ("，列为样本（按 ", "; columns are samples, grouped by the "),
    "heatmap_panel_suffix": (" 分组并附列颜色条）。", " grouping, with a column colour bar."),
    "heatmap_panel_colorbar": (
        "颜色条：Row Z-score（按蛋白行标准化后的丰度）。",
        "Colour bar: row Z-score (abundance standardised within each protein row).",
    ),
    "heatmap_reading_a": (
        "红色表示该蛋白在对应样本中丰度高于其跨样本平均水平，蓝色表示低于；",
        "Red marks a protein that is more abundant in that sample than its cross-sample "
        "average and blue less abundant; ",
    ),
    "heatmap_reading_b": (
        "列的聚类与颜色条可同时反映分组结构是否在差异蛋白上重现。",
        "the column clustering and the colour bar together show whether the group structure "
        "is reproduced among the differential proteins.",
    ),
    "heatmap_statistics": (
        "展示层为 row Z-score（按蛋白行标准化），不涉及组间假设检验。",
        "The display layer is a row Z-score (standardised within each protein row); no "
        "between-group hypothesis test is performed.",
    ),
    "heatmap_interp_prefix": ("top 差异蛋白在 ", "Abundance structure of the top differential proteins across the "),
    "heatmap_interp_mid": (
        " 分组上的丰度结构；行方向（蛋白）与列方向（样本）的聚集",
        " grouping; the clustering along rows (proteins) and columns (samples) ",
    ),
    "heatmap_interp_tail": (
        "一起说明组间差异是否一致、是否存在混杂亚群。",
        "together indicates whether the between-group differences are consistent and whether "
        "a confounding subpopulation is present.",
    ),

    # ---- volcano caption ----
    "volcano_count_missing": (
        "该对比的上下调计数在 figure_manifest 与 core_story_evidence.csv 中均未记录，以图内图例计数为准。",
        "The up/down counts of this contrast are recorded neither in figure_manifest nor in "
        "core_story_evidence.csv; the counts in the figure legend are authoritative.",
    ),
    "count_sig_prefix": ("显著差异蛋白 ", "Significant differential proteins: "),
    "count_sig_paren": (" 个（", " ("),
    "count_source_close": ("）：", "): "),
    "count_up_label": ("上调 ", "up "),
    "count_down_label": ("、下调 ", ", down "),
    "count_direction_prefix": ("（方向为 ", " (direction: "),
    "count_direction_close": ("）。", ")."),
    "interp_total_prefix": (" 共 ", ": "),
    "interp_count_open": (" 个显著差异蛋白（上调 ", " significant differential proteins ("),
    "interp_count_down": (" / 下调 ", " up / "),
    "interp_count_close": ("）", " down)"),
    "interp_up_genes_prefix": ("上调端代表蛋白：", "Representative up-regulated proteins: "),
    "interp_down_genes_prefix": ("下调端代表蛋白：", "Representative down-regulated proteins: "),
    "interp_direction_fallback": (
        "方向与显著性以图内散点和阈值参考线为准",
        "Direction and significance follow the points and the threshold reference lines in the figure",
    ),
    "volcano_panel_points_a": (
        "散点：每个点为一个蛋白，横轴 log2 Fold Change（",
        "Points: one point per protein, log2 fold change on the x axis (",
    ),
    "volcano_panel_points_b": (
        "），纵轴 -log10(adj.P.Val)。",
        ") and -log10(adj.P.Val) on the y axis.",
    ),
    "volcano_panel_thresholds": (
        "阈值参考线：两条竖虚线为 |log2FC| 阈值，一条横虚线为 adj.P.Val 阈值。",
        "Threshold reference lines: the two vertical dashed lines mark the |log2FC| threshold "
        "and the horizontal dashed line the adj.P.Val threshold.",
    ),
    "volcano_panel_colors": (
        "配色：红色 Upregulated、蓝色 Downregulated、灰色 Not significant，图例附各类计数。",
        "Colours: red for Upregulated, blue for Downregulated and grey for Not significant, "
        "with the count of each class in the legend.",
    ),
    "volcano_panel_labels": (
        "基因标注：每侧标注效应量×显著性评分最高的基因名（adjust_text 防重叠）。",
        "Gene labels: each side carries the genes with the highest effect-size score times "
        "significance (adjust_text prevents overlap).",
    ),
    "volcano_reading_a": (
        "右上（红色）为 ",
        "The upper right (red) holds the proteins increased in ",
    ),
    "volcano_reading_b": (
        " 中升高的蛋白，左上（蓝色）为 ",
        "; the upper left (blue) holds the proteins decreased in ",
    ),
    "volcano_reading_c": (" 中降低的蛋白；", ". "),
    "volcano_reading_d": (
        "点越靠上校正后越显著，越靠右/左效应量越大；灰色点不进入候选与富集分析。",
        "The higher a point, the more significant it is after correction, and the further "
        "right or left, the larger its effect size; grey points enter neither candidate "
        "selection nor enrichment analysis.",
    ),
    "volcano_statistics_prefix": (
        "limma 线性模型双侧检验；多重校正 ",
        "Two-sided tests from the limma linear model; multiple-testing correction: ",
    ),
    "volcano_statistics_suffix": (
        "（adj.P.Val）；效应量为 log2 fold change。",
        " (adj.P.Val); the effect size is the log2 fold change.",
    ),

    # ---- differential-count bar chart ----
    "bar_count_label": (
        "各对比计数（方向为对比名中 A 组相对 B 组）：",
        "Counts per contrast (direction: the first group in the contrast name relative to the second): ",
    ),
    "bar_count_up": (" 上调 ", ": up "),
    "bar_count_down": (" / 下调 ", " / down "),
    "bar_count_missing": (
        "各对比计数见图中柱上标注数字（figure_manifest 未记录计数元数据）。",
        "Counts per contrast are given by the numbers annotated on the bars "
        "(figure_manifest records no count metadata).",
    ),
    "bar_interp_base": (
        "汇总各对比通过阈值的差异数规模",
        "Scale of the differential counts that pass the thresholds in each contrast",
    ),
    "bar_interp_biggest_prefix": (
        "；差异规模最大的对比为 ",
        "; the contrast with the largest differential count is ",
    ),
    "bar_interp_biggest_mid": ("（", " ("),
    "bar_interp_biggest_suffix": (" 个显著差异蛋白）", " significant differential proteins)"),
    "bar_interp_tail": (
        "，规模差异提示主对比与对照间对比的信号强度不同。",
        ", indicating that signal strength differs between the primary contrast and the comparators.",
    ),
    "bar_panel_bars": (
        "柱状图：每个对比两根柱，红色 Up（上调蛋白数，向上）、蓝色 Down（下调蛋白数，向下）。",
        "Bar chart: two bars per contrast, red Up (number of up-regulated proteins, drawn "
        "upwards) and blue Down (number of down-regulated proteins, drawn downwards).",
    ),
    "bar_panel_labels": (
        "数字标注：柱端标出各方向的确切计数。",
        "Numeric labels: the exact count of each direction is printed at the end of its bar.",
    ),
    "bar_reading_a": (
        "柱长为该对比同时满足 adj.P.Val 与 |log2FC| 阈值的蛋白数；向上为对比名中 A 组升高、",
        "Bar length is the number of proteins in that contrast passing both the adj.P.Val and "
        "the |log2FC| threshold; bars pointing up are proteins higher in the first group "
        "named in the contrast and ",
    ),
    "bar_reading_b": (
        "向下为 A 组降低；上下规模不对称时说明方向性偏移。",
        "bars pointing down are those lower in it; an asymmetry between the two indicates a "
        "directional shift.",
    ),
    "bar_threshold_prefix": ("计数口径：adj.P.Val < ", "Count definition: adj.P.Val < "),
    "bar_threshold_mid": (" 且 |log2FC| ≥ ", " and |log2FC| ≥ "),
    "bar_threshold_missing": (
        "计数口径：figure_manifest 未记录该图阈值（差异分析默认口径见文档总览）。",
        "Count definition: figure_manifest records no threshold for this figure (see the "
        "documentation overview for the differential-analysis defaults).",
    ),
    "bar_statistics_prefix": (
        "各对比先经 limma 双侧检验 + ",
        "Each contrast is first tested with the two-sided limma test and corrected by ",
    ),
    "bar_statistics_suffix": (
        " 校正，再计数；本图本身不执行新的检验。",
        " before the counts are taken; this figure performs no new test itself.",
    ),

    # ---- key protein overview ----
    "select_label": ("筛选口径：", "Selection: "),
    "per_contrast_top_prefix": ("；每对比取评分 top ", "; top "),
    "per_contrast_top_suffix": (" 个蛋白", " proteins per contrast by score"),
    "key_protein_interp": (
        "并排查看各对比显著蛋白的方向与强度：红色 Up、蓝色 Down 两类点分别对应 A 组升高与降低。",
        "Side-by-side view of the direction and strength of the significant proteins of every "
        "contrast: red Up and blue Down points mark proteins that are higher and lower in the "
        "first group of each contrast.",
    ),
    "key_panel_points": (
        "散点：横轴为对比（Contrast，按评分排序），纵轴 log2 Fold Change；红色 Up、蓝色 Down。",
        "Points: contrasts on the x axis (Contrast, ordered by score) and log2 fold change on "
        "the y axis; red Up and blue Down.",
    ),
    "key_panel_size": (
        "点大小：-log10(adj.P.Val)，越显著点越大；深灰横线为该对比 log2FC 的中位数。",
        "Point size: -log10(adj.P.Val), the more significant the larger; the dark grey "
        "horizontal line is the median log2FC of that contrast.",
    ),
    "key_panel_labels": (
        "基因标注：每对比标注评分最高的若干基因名。",
        "Gene labels: each contrast is labelled with its highest-scoring genes.",
    ),
    "key_reading_a": (
        "同一对比内，点越高表示该蛋白在 A 组升高越多（Down 相反）；不同对比的横轴位置不可横向比较，",
        "Within one contrast, a higher point means the protein is more strongly increased in "
        "the first group (the reverse for Down); x-axis positions of different contrasts are "
        "not comparable, so ",
    ),
    "key_reading_b": (
        "应以纵轴 log2FC 数值与点大小为准。",
        "read the y-axis log2FC value and the point size instead.",
    ),
    "limma_stats_prefix": (
        "log2FC 与 adj.P.Val 来自 limma 双侧检验（",
        "log2FC and adj.P.Val come from the two-sided limma test (",
    ),

    # ---- candidate protein contrast bubble plot ----
    "per_contrast_take_prefix": ("每对比取评分 top ", "top "),
    "per_contrast_take_suffix": (" 个蛋白", " proteins per contrast by score"),
    "max_proteins_prefix": ("最多展示 ", "at most "),
    "max_proteins_suffix": (" 个蛋白", " proteins shown"),
    "bubble_interp_a": (
        "跨对比并排比较候选蛋白：颜色一致（同为红/蓝）说明方向在对比间稳定，颜色翻转说明方向相反，",
        "Candidate proteins compared side by side across contrasts: an identical colour (red or "
        "blue in both) means a stable direction and a flipped colour the opposite direction, "
        "while ",
    ),
    "bubble_interp_b": (
        "空白表示该蛋白未进入该对比的 top 集合。",
        "a blank cell means the protein did not enter the top set of that contrast.",
    ),
    "bubble_panel_axes": (
        "气泡图：横轴为对比（Contrast），纵轴为关键蛋白（Key protein）。",
        "Bubble plot: contrasts on the x axis (Contrast) and key proteins on the y axis (Key protein).",
    ),
    "bubble_panel_color": (
        "颜色通道：log2 Fold Change，红正蓝负，色标以 0 为中心对称。",
        "Colour channel: log2 fold change, red positive and blue negative, with the colour "
        "scale centred symmetrically on 0.",
    ),
    "bubble_panel_size": (
        "大小通道：-log10(adj.P.Val)，越显著气泡越大。",
        "Size channel: -log10(adj.P.Val), the more significant the larger the bubble.",
    ),
    "bubble_reading_a": (
        "横向读一个蛋白跨对比的方向与显著性变化；纵向读一个对比内候选蛋白的组成；",
        "Read horizontally for the change in direction and significance of one protein across "
        "contrasts and vertically for the composition of candidate proteins within one "
        "contrast; ",
    ),
    "bubble_reading_b": (
        "颜色深浅与气泡大小要同时读，避免只按显著性排序。",
        "colour intensity and bubble size should be read together rather than ranking on "
        "significance alone.",
    ),

    # ---- group mean abundance of core proteins ----
    "module_cross_prefix": (
        "可与 curated module（",
        "can be cross-checked against the group means of the curated module (",
    ),
    "module_cross_mid": (" 个匹配基因，", " matched genes, "),
    "module_cross_tail": (
        "见 evaluation_evidence/curated_module_group_summary.csv）的分组均分交叉验证",
        "see evaluation_evidence/curated_module_group_summary.csv)",
    ),
    "module_cross_plain": (
        "可与 evaluation_evidence/curated_module_group_summary.csv 的分组均分交叉验证",
        "can be cross-checked against the group means in "
        "evaluation_evidence/curated_module_group_summary.csv",
    ),
    "top_means_scale_a": ("各对比 adj.P.Val < ", "across contrasts, adj.P.Val < "),
    "top_means_scale_b": (" 的 top ", " selects the top "),
    "top_means_scale_c": (" 差异蛋白", " differential proteins"),
    "top_means_scale_all": ("各对比 top 差异蛋白", "the top differential proteins of each contrast"),
    "protein_set_label": ("蛋白集合：", "Protein set: "),
    "top_means_panel_left_prefix": (
        "左面板 Group Mean Abundance：",
        "Left panel, Group Mean Abundance: ",
    ),
    "top_means_panel_left_mid": (
        "在各 ",
        "; the panel shows their mean row Z-score in each ",
    ),
    "top_means_panel_left_suffix": (
        " 分组的平均 row Z-score（颜色 -2.5 至 2.5）。",
        " group (colour scale -2.5 to 2.5).",
    ),
    "top_means_panel_right": (
        "右面板 Detection Rate：同组蛋白在各分组的检出率（0-100%）。",
        "Right panel, Detection Rate: detection rate of the same proteins in each group (0-100%).",
    ),
    "top_means_reading_a": (
        "左图颜色越红表示该蛋白在该分组平均丰度越高；右图检出率越低，左图均值受缺失模式的影响越大，",
        "The redder the colour in the left panel, the higher the mean abundance of that protein "
        "in that group; the lower the detection rate in the right panel, the more the left-panel "
        "mean is influenced by the missingness pattern, so ",
    ),
    "top_means_reading_b": ("两个面板应同时读取。", "the two panels should be read together."),
    "top_means_statistics": (
        "展示层为按蛋白行标准化的组均值 row Z-score 与检出率，不执行新的组间检验；组间差异方向以各对比火山图为准。",
        "The display layer is the group-mean row Z-score standardised within each protein row "
        "together with the detection rate, and no new between-group test is performed; the "
        "direction of between-group differences follows the volcano plot of each contrast.",
    ),
    "top_means_interp_prefix": (
        "候选蛋白按 ",
        "Direction of the mean abundance of the candidate proteins across the ",
    ),
    "top_means_interp_mid": (
        " 分组的平均丰度方向与检出可靠性；",
        " grouping and the reliability of their detection; ",
    ),

    # ---- mechanism enrichment dot plot ----
    "db_fallback": (
        "GO/KEGG/Reactome（以 figure_manifest 记录的输入为准）",
        "GO/KEGG/Reactome (as recorded in the figure_manifest inputs)",
    ),
    "dotplot_scale_prefix": ("每个 namespace×对比×方向取 p.adjust 最小的 top ", "the top "),
    "dotplot_scale_suffix": (" 条目", " terms by smallest p.adjust per namespace × contrast × direction"),
    "dotplot_scale_all": (
        "每个 namespace×对比×方向取 p.adjust 最小的 top 条目",
        "the terms with the smallest p.adjust per namespace × contrast × direction",
    ),
    "term_scale_label": ("条目规模：", "Term selection: "),
    "input_db_label": ("输入数据库：", "Input databases: "),
    "dotplot_panel_points": (
        "点图：横轴为对比×方向（Contrast and direction，Up/Down 分列），纵轴为通路条目（namespace: Description）。",
        "Dot plot: contrast × direction on the x axis (Contrast and direction, with Up and "
        "Down in separate columns) and pathway terms on the y axis (namespace: Description).",
    ),
    "dotplot_panel_size": (
        "点大小：Count，即该通路内检出的差异数。",
        "Point size: Count, the number of differential proteins detected in that term.",
    ),
    "dotplot_panel_color": (
        "颜色通道：-log10(FDR)（p.adjust），值越大颜色越亮、校正后越显著。",
        "Colour channel: -log10(FDR) (p.adjust); a larger value is brighter and more "
        "significant after correction.",
    ),
    "dotplot_reading_a": (
        "同一通路条目在 Up 与 Down 列同时出现时，可比较其在两个方向的富集强度；",
        "When the same term appears in both the Up and the Down column, its enrichment "
        "strength in the two directions can be compared; ",
    ),
    "dotplot_reading_b": (
        "优先阅读 Count 较大且 -log10(FDR) 较高的条目，避免只按单一指标排序。",
        "prioritise terms with a larger Count and a higher -log10(FDR) rather than ranking on "
        "a single measure.",
    ),
    "dotplot_statistics": (
        "超几何富集检验（over-representation analysis），BH 校正后报告 p.adjust；点大小为通路内检出差异数（Count）。",
        "Hypergeometric over-representation analysis with p.adjust reported after BH "
        "correction; the point size is the number of differential proteins detected in the "
        "term (Count).",
    ),
    "dotplot_interp_a": (
        "汇总当前矩阵差异蛋白的离线富集证据；条目方向应与对应对比的火山图方向一致阅读，",
        "Summary of the offline enrichment evidence for the differential proteins of the "
        "current matrix; the direction of a term should be read together with the direction "
        "of the matching contrast in its volcano plot, and ",
    ),
    "dotplot_interp_b": (
        "富集结果不改变矩阵内差异蛋白本身的方向与计数。",
        "enrichment does not change the direction or the count of the differential proteins "
        "in the matrix itself.",
    ),

    # ---- enrichment bubble plot ----
    "bubble_scale_prefix": ("每个对比方向取 p.adjust 最小的 top ", "the top "),
    "bubble_scale_suffix": (" 条目", " terms by smallest p.adjust per contrast and direction"),
    "bubble_scale_all": (
        "每个对比方向取 p.adjust 最小的 top 条目",
        "the terms with the smallest p.adjust per contrast and direction",
    ),
    "source_file_label": ("来源文件：", "Source file: "),
    "source_file_missing": (
        "来源文件：figure_manifest 未记录",
        "Source file: not recorded in figure_manifest",
    ),
    "enrich_bubble_panel_axes": (
        "气泡图：横轴 Rich Factor（通路内检出基因数 / 通路注释基因总数），纵轴为通路条目。",
        "Bubble plot: Rich Factor on the x axis (genes detected in the term / genes annotated "
        "to the term) and pathway terms on the y axis.",
    ),
    "enrich_bubble_panel_size": (
        "点大小：Count（通路内检出的差异数）。",
        "Point size: Count (the number of differential proteins detected in the term).",
    ),
    "enrich_bubble_panel_color": (
        "颜色通道：-log10(adjusted p-value)。",
        "Colour channel: -log10(adjusted p-value).",
    ),
    "enrich_bubble_reading_a": (
        "越靠右表示该通路在差异蛋白中的占比越高；点越大表示检出差异数越多；",
        "Further right means a larger share of that pathway among the differential proteins "
        "and a larger point more differential proteins detected; ",
    ),
    "enrich_bubble_reading_b": (
        "Rich Factor 与 Count 应同时读取，避免高占比但检出数极少的条目被过度解读。",
        "Rich Factor and Count should be read together so that a term with a high share but "
        "very few detected proteins is not over-interpreted.",
    ),
    "enrich_bubble_statistics": (
        "超几何富集检验（over-representation analysis），BH 校正后报告 p.adjust。",
        "Hypergeometric over-representation analysis with p.adjust reported after BH correction.",
    ),
    "enrich_bubble_interp_prefix": ("来自 ", "Differential protein set from "),
    "enrich_bubble_interp_mid": (" 的 ", " for "),
    "enrich_bubble_interp_tail": ("差异蛋白集；", " differential protein set; "),
    "enrich_bubble_interp_tail2": (
        "解释机制方向时应与对应火山图的上下调方向一致阅读。",
        "the mechanistic direction should be read together with the up/down direction of the "
        "matching volcano plot.",
    ),

    # ---- pathway gene heatmap ----
    "pathway_panel_prefix": ("热图：行为通路「", "Heatmap: rows are the genes of the pathway “"),
    "pathway_panel_mid": (
        "」匹配到当前矩阵的基因（Matched pathway genes），列为 ",
        "” that map to the current matrix (Matched pathway genes); the columns are the ",
    ),
    "pathway_panel_suffix": (" 分组。", " groups."),
    "pathway_panel_color": (
        "颜色通道：组均值 row Z-score（-2.5 至 2.5），红色为该组平均丰度较高、蓝色较低。",
        "Colour channel: group-mean row Z-score (-2.5 to 2.5), red for a higher mean "
        "abundance in that group and blue for a lower one.",
    ),
    "pathway_reading_a": (
        "逐行读取基因在不同分组间的平均丰度方向；同一通路内方向一致的基因越多，",
        "Read each row for the direction of a gene's mean abundance across the groups; the "
        "more genes within one pathway agree in direction, ",
    ),
    "pathway_reading_b": (
        "该通路的整体方向越可信。",
        "the more reliable the overall direction of that pathway.",
    ),
    "pathway_set_prefix": ("蛋白集合：通路「", "Protein set: the portion of the “"),
    "pathway_set_suffix": (
        "」注释基因中可匹配到当前矩阵的部分（匹配不到的基因不进入热图）。",
        "” annotated genes that maps to the current matrix (genes that do not map are "
        "excluded from the heatmap).",
    ),
    "pathway_statistics": (
        "展示层为按蛋白行标准化的组均值 row Z-score，不执行组间检验；通路归属来自离线富集条目。",
        "The display layer is the group-mean row Z-score standardised within each protein row "
        "and no between-group test is performed; pathway membership comes from the offline "
        "enrichment term.",
    ),
    "pathway_interp_prefix": ("通路「", "Abundance structure of the genes in the pathway “"),
    "pathway_interp_mid": ("」内基因在", "” across "),
    "pathway_interp_all_contrasts": ("各对比", "all contrasts"),
    "pathway_interp_suffix": ("方向下的平均丰度结构，", ", "),
    "pathway_interp_tail": (
        "与对应富集条目配套阅读。",
        "to be read together with the matching enrichment term.",
    ),

    # ---- overlapping differential proteins (UpSet) ----
    "upset_panel_summary": (
        "UpSet 式汇总：底部条形为各对比的差异数，主区条形为交集/独有蛋白数。",
        "UpSet-style summary: the bars at the bottom are the differential counts of each "
        "contrast and the bars of the main area the sizes of the intersections and unique sets.",
    ),
    "upset_panel_labels": (
        "数字标注：各集合的确切蛋白数。",
        "Numeric labels: the exact protein count of each set.",
    ),
    "upset_reading": (
        "先读底部各对比总量，再读主区：独有集合说明对比特异性，交集说明跨对比共享的差异机制。",
        "Read the totals per contrast at the bottom first and the main area second: unique sets "
        "indicate contrast specificity and intersections indicate differential mechanisms "
        "shared across contrasts.",
    ),
    "upset_statistics_prefix": (
        "基于各对比 limma 双侧检验 + ",
        "Based on the differential protein sets of each contrast from the two-sided limma "
        "test plus ",
    ),
    "upset_statistics_suffix": (
        " 校正后的差异蛋白集合，再取交集/差集。",
        " correction, the intersections and differences are then taken.",
    ),
    "upset_interp_a": (
        "衡量候选机制在各对比间的共享与特异程度；交集大说明机制跨对比稳定，",
        "Measures how far candidate mechanisms are shared across contrasts or specific to one; "
        "a large intersection indicates a mechanism that is stable across contrasts and ",
    ),
    "upset_interp_b": (
        "独有集大说明该对比携带独特信号。",
        "a large unique set that a contrast carries a distinctive signal.",
    ),

    # ---- single-contrast integrated evidence panel ----
    "ce_panel_enrichment_terms": ("富集条目子面板", "enrichment term subpanel"),
    "ce_panel_group_means": ("组均值热图子面板", "group-mean heatmap subpanel"),
    "ce_panel_volcano": ("火山证据子面板", "volcano evidence subpanel"),
    "ce_panel_top_proteins": ("核心差异蛋白子面板", "core differential protein subpanel"),
    "ce_panel_prefix": ("单对比综合证据面板（", "Integrated evidence panel for a single contrast ("),
    "ce_panel_suffix": (
        "）：并排展示该对比的差异蛋白方向、组均值与富集要点。",
        "): side-by-side display of the differential protein direction, the group means and "
        "the enrichment highlights of that contrast.",
    ),
    "ce_panel_consistency": (
        "各子面板的坐标与颜色含义与对应单图一致（火山/均值/富集）。",
        "Axes and colour meanings follow the corresponding single-contrast figures "
        "(volcano / group means / enrichment).",
    ),
    "panel_omitted_suffix_ce": (
        " 因该对比无匹配数据未绘制，不使用灰字占位图。",
        " were not drawn because this contrast has no matching data; no grey placeholder "
        "panel is used.",
    ),
    "ce_reading_a": (
        "把同一对比的差异证据与富集解释放进一张图，先读差异方向，再读富集条目；",
        "The differential evidence and the enrichment interpretation of one contrast are "
        "combined in a single figure: read the differential direction first and the "
        "enrichment terms second; ",
    ),
    "ce_reading_b": (
        "两列证据方向冲突时应回到对应单图核对。",
        "if the two columns of evidence conflict in direction, return to the corresponding "
        "single-contrast figure to check.",
    ),
    "ce_statistics_prefix": ("子图沿用 limma 双侧检验（", "The subpanels reuse the two-sided limma test ("),
    "ce_statistics_suffix": (
        " 校正）与超几何富集检验结果，不引入新的检验。",
        " correction) and the hypergeometric enrichment results; no new test is introduced.",
    ),
    "ce_interp_suffix": (
        " 的差异与机制证据整合视图；与该对比火山图和富集气泡图结论应一致。",
        ": integrated view of the differential and mechanistic evidence of this contrast; it "
        "should agree with the conclusions of the matching volcano plot and enrichment bubble plot.",
    ),

    # ---- generic figure ----
    "generic_interp": (
        "本图的面板结构以图内轴标签与图例为准；数字口径见上方阈值行，并与主报告对应任务章节阅读。",
        "The panel structure of this figure follows its axis labels and legends; the numeric "
        "definitions are given in the threshold rows above, and the figure should be read "
        "together with the matching task section of the main report.",
    ),
    "manifest_caption_label": ("figure_manifest 记录：", "figure_manifest records: "),
    "generic_panel_prefix": (
        "面板结构以图内轴标签与图例为准（figure_manifest 类别：",
        "The panel structure follows the axis labels and legends in the figure "
        "(figure_manifest category: ",
    ),
    "generic_reading": (
        "按图内轴标签与图例读取；数值通道与显著性含义见阈值与统计口径行。",
        "Read from the axis labels and legends in the figure; the meaning of the numeric "
        "channels and of significance is given in the threshold and statistical rows.",
    ),
    "generic_statistics": (
        "统计口径以生成该图的 plot 函数记录为准（见 figure_manifest.json 该条目）。",
        "The statistical basis is the record of the plot function that produced the figure "
        "(see the corresponding entry in figure_manifest.json).",
    ),

    # ---- figures_captions.md skeleton ----
    "captions_doc_title_suffix": (" 发表级图注", " publication-grade figure captions"),
    "captions_doc_intro": (
        "本文档为每张关键图提供发表级中文图注，由 run 内 `figure_manifest.json`、`analysis_design` 与 `evaluation_evidence` 证据表自动生成；蛋白、通路与统计术语保留英文原文。",
        "This document gives a publication-grade caption for every key figure, generated "
        "automatically from `figure_manifest.json`, `analysis_design` and the "
        "`evaluation_evidence` tables of the run; protein, pathway and statistical terms keep "
        "their English form.",
    ),
    "heading_thresholds_overview": ("## 阈值与统计总览", "## Thresholds and statistics overview"),
    "heading_boundary_legend": ("## 证据边界图例", "## Evidence boundary legend"),
    "heading_caption_index": ("## 图注目录", "## Caption index"),
    "index_header": ("| 图号 | 标题 | 图片文件 |", "| Figure | Title | Image file |"),
    "label_image_file": ("- 图片文件：", "- Image file: "),
    "label_panels": ("- 面板说明：", "- Panel description: "),
    "label_reading": ("- 如何阅读：", "- How to read: "),
    "label_thresholds": ("- 阈值与样本量：", "- Thresholds and sample size: "),
    "label_statistics": ("- 统计口径：", "- Statistical basis: "),
    "label_interpretation": ("- 一句话解读：", "- One-sentence interpretation: "),
    "label_boundary": ("- 证据边界：", "- Evidence boundary: "),
    "label_paper_ref": ("- 对应原论文图：", "- Corresponding original-paper figures: "),
    "brief_title_suffix_default": ("关键图表嵌入版", "key figures, embedded"),
    "brief_intro_default_a": (
        "每张图只保留图号、标题与嵌图；逐图的面板说明、阈值、统计口径、解读与证据边界见 ",
        "Each figure keeps only its number, title and embedded image; the panel description, "
        "thresholds, statistical basis, interpretation and evidence boundary of each figure "
        "are in the ",
    ),
    "brief_intro_default_b": (
        "`figures_captions.md` 对应小节。",
        " section of `figures_captions.md`.",
    ),
    "brief_pointer_prefix": ("图注详见 ", "Full caption in "),
    "local_title_suffix": ("关键图表本机预览版", "key figures, local preview"),
    "local_intro_a": (
        "本文件使用本机绝对路径，主要用于解决某些 Markdown 预览器不能解析相对图片路径的问题；",
        "This file uses absolute local paths, mainly to work around Markdown previewers that "
        "cannot resolve relative image paths; ",
    ),
    "local_intro_b": (
        "对外发送时优先使用 `figures.md`。逐图图注见 `figures_captions.md`。",
        "prefer `figures.md` when the document is sent out. Per-figure captions are in "
        "`figures_captions.md`.",
    ),
    # ---- thresholds-and-statistics overview block ----
    "overview_default_prefix": (
        "差异分析默认口径：adj.P.Val < ",
        "Default differential-analysis thresholds: adj.P.Val < ",
    ),
    "overview_default_suffix": (
        " 校正，limma 双侧检验）。",
        " correction, two-sided limma test).",
    ),
    "overview_partial_prefix": (
        "差异分析默认口径（部分记录）：adj.P.Val < ",
        "Default differential-analysis thresholds (partially recorded): adj.P.Val < ",
    ),
    "overview_partial_suffix": (" 校正）。", " correction)."),
    "overview_missing": (
        "差异分析默认口径：analysis_design 未记录阈值，逐图阈值见各图注的阈值行。",
        "Default differential-analysis thresholds: analysis_design records no threshold; the "
        "per-figure thresholds are given in the threshold row of each caption.",
    ),
    "main_group_col_prefix": ("主分组列：", "Primary grouping column: "),
    "main_group_col_suffix": ("（来自 analysis_design）。", " (from analysis_design)."),
    "numbers_source": (
        "数字来源：figure_manifest.json（含逐图 caption_metadata/params）与 evaluation_evidence 证据表",
        "Source of the numbers: figure_manifest.json (with the per-figure "
        "caption_metadata/params) and the evaluation_evidence tables",
    ),
    "manifest_time_prefix": ("；manifest 生成时间 ", "; manifest generated at "),

}
register("captions", _CAPTIONS)


def _cap(key: str, **fmt: Any) -> str:
    """Active-language text for a key of this module's caption registry."""
    return t("captions." + key, **fmt)


def boundary_matrix() -> str:
    """Evidence boundary for numbers computed from the current matrix."""
    return _cap("boundary_matrix")


def boundary_enrichment() -> str:
    """Evidence boundary for offline pathway enrichment of matrix proteins."""
    return _cap("boundary_enrichment_a") + _cap("boundary_enrichment_b")


def boundary_mixed() -> str:
    """Evidence boundary for figures that mix matrix values and enrichment sets."""
    return _cap("boundary_mixed_a") + _cap("boundary_mixed_b")


def paper_ref_fallback() -> str:
    """Shown when the original-paper figure map holds no panel for a figure."""
    return _cap("paper_ref_fallback")


# The released module exposed the four texts above as strings under these names; the
# names stay importable but now resolve through the active report language.
BOUNDARY_MATRIX = boundary_matrix
BOUNDARY_ENRICHMENT = boundary_enrichment
BOUNDARY_MIXED = boundary_mixed
PAPER_REF_FALLBACK = paper_ref_fallback

# plot types that have a caption title, mapped to their registry key (the released
# PLOT_TYPE_TITLES held the zh titles themselves; the text now resolves per language)
PLOT_TYPE_TITLES = {
    "qc_sample_overview": "title_qc_sample_overview",
    "pca": "title_pca",
    "umap": "title_umap",
    "heatmap": "title_heatmap",
    "differential_summary_barplot": "title_differential_summary_barplot",
    "key_protein_overview": "title_key_protein_overview",
    "protein_contrast_bubble": "title_protein_contrast_bubble",
    "mechanism_top_protein_group_means": "title_mechanism_top_protein_group_means",
    "mechanism_enrichment_dotplot": "title_mechanism_enrichment_dotplot",
    "mechanism_dep_overlap_upset": "title_mechanism_dep_overlap_upset",
    "mechanism_contrast_evidence": "title_mechanism_contrast_evidence",
    "enrichment_bubble": "title_enrichment_bubble",
    "mechanism_pathway_gene_heatmap": "title_mechanism_pathway_gene_heatmap",
}


def _plot_title(plot_type: str) -> str:
    """Caption title of a plot type in the active language ("" when unknown)."""
    key = PLOT_TYPE_TITLES.get(str(plot_type))
    return _cap(key) if key else ""


# ---------------------------------------------------------------------------
# small generic helpers
# ---------------------------------------------------------------------------


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except Exception:
        return ""


def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _read_csv_dicts(path: str, limit: int = 800) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cleaned: Dict[str, Any] = {}
                for key, value in row.items():
                    if key is None:
                        # extra fields beyond the header (ragged row): keep them
                        # out of the cleaned dict instead of crashing on a list
                        continue
                    if isinstance(value, list):
                        value = ";".join(str(item) for item in value)
                    cleaned[str(key).strip()] = ("" if value is None else str(value)).strip()
                rows.append(cleaned)
                if len(rows) >= limit:
                    break
    except Exception:
        return []
    return rows


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return int(round(number))


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _fmt_num(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def _fmt_pct(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    return f"{number * 100:.1f}%"


def _human_contrast(contrast: str) -> str:
    text = str(contrast or "").strip()
    if not text:
        return ""
    text = text.replace("_vs_", " vs ")
    text = re.sub(r"Cluster[_ ]+(\d+)", r"Cluster \1", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_contrast(contrast: str) -> Tuple[str, str]:
    """``Cluster_1_vs_Cluster_2`` -> (``Cluster 1``, ``Cluster 2``)."""
    text = str(contrast or "").strip()
    if "_vs_" in text:
        left, right = text.split("_vs_", 1)
    elif " vs " in text:
        left, right = text.split(" vs ", 1)
    else:
        return "", ""
    return _human_contrast(left), _human_contrast(right)


def _extract_top_genes(text: Any, k: int = 3) -> List[str]:
    """``HBA2 logFC=2.86, FDR=7.67e-08; ORC1 logFC=...`` -> ``[HBA2, ORC1, ...]``."""
    matches = re.findall(r"([A-Za-z][A-Za-z0-9_.-]*)\s*logFC=", str(text or ""))
    genes: List[str] = []
    for gene in matches:
        if gene not in genes:
            genes.append(gene)
        if len(genes) >= k:
            break
    return genes


def _clean_cell(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_figure_manifest(run_folder: str) -> Dict[str, Any]:
    for rel in (
        os.path.join("visualize_results", "figure_manifest.json"),
        os.path.join("figures", "figure_manifest.json"),
        "figure_manifest.json",
    ):
        path = os.path.join(run_folder, rel)
        data = _load_json(path)
        if isinstance(data, dict) and isinstance(data.get("figures"), list):
            return data
    return {"generated_at": "", "visualize_params": {}, "figures": []}


def _parse_design_regex(text: str) -> Dict[str, Any]:
    design: Dict[str, Any] = {"group_col": "", "differential": {}}
    group_col = re.search(r"(?m)^\s*group_col:\s*(\S+)", text)
    if group_col and group_col.group(1).lower() not in {"null", "~", "none"}:
        design["group_col"] = group_col.group(1).strip("'\"")
    diff = design["differential"]
    match = re.search(r"(?m)^\s*p_thresh:\s*([0-9.eE+-]+)", text)
    if match:
        diff["p_thresh"] = float(match.group(1))
    match = re.search(r"(?m)^\s*logfc_thresh:\s*([0-9.eE+-]+)", text)
    if match:
        diff["logfc_thresh"] = float(match.group(1))
    match = re.search(r"(?m)^\s*fdr_method:\s*(\S+)", text)
    if match:
        diff["fdr_method"] = match.group(1).strip("'\"")
    return design


def load_analysis_design(run_folder: str) -> Dict[str, Any]:
    for name in ("analysis_design.used.yaml", "analysis_design.inferred.yaml"):
        path = os.path.join(run_folder, name)
        if not os.path.exists(path):
            continue
        text = _read_text(path)
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return _parse_design_regex(text)
    return {}


def load_story_outline(run_folder: str) -> Dict[str, Any]:
    data = _load_json(os.path.join(run_folder, "evaluation_evidence", "report_story_outline.json"))
    return data if isinstance(data, dict) else {}


def load_sample_summary(run_folder: str) -> Dict[str, Any]:
    rows = _read_csv_dicts(os.path.join(run_folder, "evaluation_evidence", "group_composition_qc.csv"))
    groups: List[Dict[str, Any]] = []
    group_col = ""
    detected_values = set()
    missing_values = set()
    for row in rows:
        table = _clean_cell(row.get("table"))
        if table and table != "primary_group":
            continue
        n_samples = _to_int(row.get("n_samples"))
        if n_samples is None:
            continue
        group_name = _clean_cell(row.get("group")) or _cap("unnamed_group")
        groups.append({"group": group_name, "n": n_samples})
        group_col = group_col or _clean_cell(row.get("group_col"))
        detected = _to_float(row.get("mean_detected_proteins"))
        if detected is not None:
            detected_values.add(round(detected, 1))
        missing = _to_float(row.get("mean_missing_rate"))
        if missing is not None:
            missing_values.add(round(missing, 6))
    return {
        "group_col": group_col,
        "groups": groups,
        "n_total": sum(g["n"] for g in groups),
        "mean_detected": sorted(detected_values)[0] if len(detected_values) == 1 else None,
        "mean_missing": sorted(missing_values)[0] if len(missing_values) == 1 else None,
    }


def load_contrast_stats(run_folder: str) -> Dict[str, Dict[str, Any]]:
    rows = _read_csv_dicts(os.path.join(run_folder, "evaluation_evidence", "core_story_evidence.csv"))
    stats: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if _clean_cell(row.get("row_type")) != "contrast":
            continue
        display = _clean_cell(row.get("display_contrast"))
        source = _clean_cell(row.get("source_contrast"))
        contrast = display or source
        if not contrast:
            continue
        entry = {
            "contrast": contrast,
            "group_a": _clean_cell(row.get("group_a")) or _split_contrast(contrast)[0],
            "group_b": _clean_cell(row.get("group_b")) or _split_contrast(contrast)[1],
            "n_sig": _to_int(row.get("n_sig")),
            "n_up": _to_int(row.get("n_up_display_group_a")),
            "n_down": _to_int(row.get("n_down_display_group_a")),
            "top_up": _clean_cell(row.get("top_up_display_group_a")),
            "top_down": _clean_cell(row.get("top_down_display_group_a")),
            "interpretation": _clean_cell(row.get("interpretation")),
            "boundary": _clean_cell(row.get("boundary")),
            "confidence": _clean_cell(row.get("confidence")),
        }
        for key in {contrast, display, source}:
            if key:
                stats[key] = entry
    return stats


def load_module_summary(run_folder: str) -> Dict[str, Any]:
    rows = _read_csv_dicts(os.path.join(run_folder, "evaluation_evidence", "curated_module_group_summary.csv"))
    modules: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        name = _clean_cell(row.get("module"))
        if not name:
            continue
        if name not in modules:
            modules[name] = {
                "genes": _clean_cell(row.get("matched_genes")),
                "n_genes": _to_int(row.get("n_matched_genes")),
                "group_col": _clean_cell(row.get("group_col")),
                "groups": [],
            }
            order.append(name)
        group_name = _clean_cell(row.get("group"))
        mean_score = _to_float(row.get("mean_score"))
        if group_name:
            modules[name]["groups"].append({"group": group_name, "mean_score": mean_score})
    return {"modules": modules, "order": order}


def load_original_figure_map(project_root: Optional[str] = None) -> Dict[str, Any]:
    root = project_root or PROJECT_ROOT
    data = _load_json(os.path.join(root, ORIGINAL_FIGURE_MAP_REL))
    return data if isinstance(data, dict) else {}


def build_paper_figure_index(original_map: Dict[str, Any], dataset: str) -> Dict[str, List[Tuple[str, str]]]:
    """Invert ``main_figures[].matched_agent_plots`` into {figure_file: [(Fig. no, qualifier)]}."""
    index: Dict[str, List[Tuple[str, str]]] = {}
    for entry in original_map.get("datasets", []) or []:
        if str(entry.get("dataset", "")).strip() != dataset:
            continue
        for fig in entry.get("main_figures", []) or []:
            fig_no = str(fig.get("figure_no", "") or "").strip()
            raw = fig.get("matched_agent_plots")
            if isinstance(raw, (list, tuple)):
                matched = ", ".join(str(item) for item in raw)
            else:
                matched = str(raw or "")
            matched = matched.strip()
            if not fig_no or not matched:
                continue
            qualifier = ""
            main_part = matched
            if " - " in matched:
                main_part, qualifier = matched.split(" - ", 1)
            for item in main_part.split(", "):
                item = item.strip()
                if not item:
                    continue
                head = item.split(" (", 1)[0]
                for token in head.split(" / "):
                    token = token.strip().strip("[]'\"").strip()
                    if token.lower().endswith(".png"):
                        hits = index.setdefault(token, [])
                        pair = (fig_no, qualifier.strip())
                        if pair not in hits:
                            hits.append(pair)
        break
    return index


def infer_dataset_name(run_folder: str, original_map: Optional[Dict[str, Any]] = None) -> str:
    folder = os.path.basename(os.path.normpath(str(run_folder)))
    if original_map:
        for entry in original_map.get("datasets", []) or []:
            name = str(entry.get("dataset", "")).strip()
            if name and folder.startswith(name):
                return name
    return re.sub(r"_[^_]+_\d{8}$", "", folder)


def load_run_context(run_folder: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Collect every caption input; tolerate missing pieces so captions never crash."""
    run_folder = os.path.normpath(str(run_folder))
    original_map = load_original_figure_map(project_root)
    dataset = infer_dataset_name(run_folder, original_map)
    manifest = load_figure_manifest(run_folder)
    figures_by_name: Dict[str, Dict[str, Any]] = {}
    for record in manifest.get("figures", []) or []:
        if isinstance(record, dict) and record.get("file_name"):
            figures_by_name[str(record["file_name"])] = record
    design = load_analysis_design(run_folder)
    differential = design.get("differential") if isinstance(design.get("differential"), dict) else {}
    return {
        "run_folder": run_folder,
        "project_root": project_root or PROJECT_ROOT,
        "dataset": dataset,
        "manifest": manifest,
        "generated_at": str(manifest.get("generated_at", "") or ""),
        "visualize_params": manifest.get("visualize_params") if isinstance(manifest.get("visualize_params"), dict) else {},
        "figures_by_name": figures_by_name,
        "design": design,
        "differential": differential,
        "group_col": str(design.get("group_col", "") or ""),
        "sample_summary": load_sample_summary(run_folder),
        "contrast_stats": load_contrast_stats(run_folder),
        "module_summary": load_module_summary(run_folder),
        "outline": load_story_outline(run_folder),
        "paper_index": build_paper_figure_index(original_map, dataset),
    }


# ---------------------------------------------------------------------------
# threshold / count resolution (caption_metadata -> params -> design fallback)
# ---------------------------------------------------------------------------


def _resolve_thresholds(record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("caption_metadata") if isinstance(record.get("caption_metadata"), dict) else {}
    th = meta.get("thresholds") if isinstance(meta.get("thresholds"), dict) else {}
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    vp = context.get("visualize_params") or {}
    diff = context.get("differential") or {}

    def pick(*values: Any) -> Optional[float]:
        for value in values:
            number = _to_float(value)
            if number is not None:
                return number
        return None

    return {
        "pval": pick(
            meta.get("pval_cutoff"), th.get("adj_pval"), params.get("pval_cut"),
            vp.get("pval_cut"), diff.get("p_thresh"),
        ),
        "logfc": pick(
            meta.get("logfc_cutoff"), th.get("abs_logfc"), params.get("logfc_cut"),
            vp.get("logfc_cut"), diff.get("logfc_thresh"),
        ),
        "top_proteins": pick(th.get("top_n_proteins"), params.get("top_n_proteins"), vp.get("top_n_proteins")),
        "top_terms": pick(params.get("top_n_terms"), vp.get("top_n_terms")),
        "max_labels": pick(th.get("top_n_labels"), th.get("max_labels"), params.get("max_labels"), vp.get("max_labels")),
        "top_per_contrast": pick(meta.get("top_n_per_contrast"), th.get("top_n_per_contrast"), params.get("top_n_per_contrast")),
        "max_proteins": pick(meta.get("max_proteins"), th.get("max_proteins"), params.get("max_proteins")),
        "fdr_method": str(diff.get("fdr_method") or "BH"),
        "filter_note": _clean_cell(meta.get("filter_note")),
    }


def _contrast_counts(record: Dict[str, Any], context: Dict[str, Any], contrast: str) -> Dict[str, Optional[int]]:
    meta = record.get("caption_metadata") if isinstance(record.get("caption_metadata"), dict) else {}
    counts = {
        "n_sig": _to_int(meta.get("n_sig")),
        "n_up": _to_int(meta.get("n_up")),
        "n_down": _to_int(meta.get("n_down")),
        "source": "figure_manifest.caption_metadata",
    }
    if counts["n_sig"] is None and counts["n_up"] is None and counts["n_down"] is None:
        row = (context.get("contrast_stats") or {}).get(contrast)
        if row:
            counts = {
                "n_sig": row.get("n_sig"),
                "n_up": row.get("n_up"),
                "n_down": row.get("n_down"),
                "source": "evaluation_evidence/core_story_evidence.csv",
            }
    if counts["n_sig"] is None and counts["n_up"] is not None and counts["n_down"] is not None:
        counts["n_sig"] = counts["n_up"] + counts["n_down"]
    return counts


def _sample_line(context: Dict[str, Any]) -> str:
    ss = context.get("sample_summary") or {}
    groups = ss.get("groups") or []
    if not groups:
        return _cap("sample_missing")
    detail = _cap("punct_enum").join(f"{g['group']}: {g['n']}" for g in groups)
    parts = [
        _cap("sample_size_prefix")
        + str(ss.get("n_total"))
        + _cap("punct_paren_open")
        + detail
        + _cap("punct_paren_close")
    ]
    if ss.get("group_col"):
        parts.append(_cap("sample_group_col") + str(ss["group_col"]))
    if ss.get("mean_detected") is not None:
        parts.append(
            _cap("mean_detected_prefix")
            + _fmt_num(ss["mean_detected"])
            + _cap("protein_count_suffix")
        )
    if ss.get("mean_missing") is not None:
        parts.append(_cap("mean_missing_prefix") + _fmt_pct(ss["mean_missing"]))
    return _cap("punct_semicolon").join(parts) + _cap("punct_period")


def _threshold_core(th: Dict[str, Any]) -> str:
    """Core threshold text without a leading label, e.g. ``adj.P.Val < 0.05；|log2FC| ≥ 0.25（BH 校正）``."""
    bits = []
    if th.get("pval") is not None:
        bits.append(f"adj.P.Val < {_fmt_num(th['pval'])}")
    if th.get("logfc") is not None:
        bits.append(f"|log2FC| ≥ {_fmt_num(th['logfc'])}")
    if not bits:
        return ""
    return (
        _cap("punct_semicolon").join(bits)
        + _cap("punct_paren_open")
        + str(th.get("fdr_method", "BH"))
        + _cap("fdr_correction_suffix")
    )


def _threshold_sentence(th: Dict[str, Any]) -> str:
    core = _threshold_core(th)
    if not core:
        return _cap("threshold_not_recorded")
    return _cap("threshold_label") + core + _cap("punct_period")


def _paper_ref_line(file_name: str, context: Dict[str, Any]) -> str:
    hits = (context.get("paper_index") or {}).get(file_name, [])
    if not hits:
        return paper_ref_fallback()
    parts = []
    for fig_no, qualifier in hits:
        parts.append(
            fig_no + _cap("punct_paren_open") + qualifier + _cap("punct_paren_close")
            if qualifier
            else fig_no
        )
    return _cap("paper_ref_prefix") + _cap("punct_enum").join(parts) + _cap("punct_period")


def _databases_from_inputs(record: Dict[str, Any]) -> List[str]:
    databases: List[str] = []
    for src in record.get("inputs", []) or []:
        base = os.path.basename(str(src))
        db = base.split("_", 1)[0].upper()
        if db not in {"GO", "KEGG", "REACTOME"}:
            continue
        display = "Reactome" if db == "REACTOME" else db
        if display not in databases:
            databases.append(display)
    return databases


def _drop_hollow_purpose_parts(parts: Iterable[str]) -> List[str]:
    """Drop sourced sentences that merely restate the figure's purpose.

    Run evidence tables sometimes carry "用于…" style purpose phrases; the
    caption discipline bans them (acceptance: 无"用于…"式空洞描述).
    """
    kept: List[str] = []
    for part in parts:
        text = str(part or "").strip().rstrip("。")
        if not text:
            continue
        for marker in ("；用于", "，用于"):
            if marker in text:
                text = text.split(marker, 1)[0].rstrip("。，")
        if text.startswith("用于") or not text:
            continue
        kept.append(text)
    return kept


# ---------------------------------------------------------------------------
# per-plot-type caption builders
# ---------------------------------------------------------------------------


# Step 04: QC panel keys emitted by tools.plot_qc_sample_overview
# (caption_metadata.omitted_panels), mapped to their registry key
QC_PANEL_TITLES = {
    "detected_proteins": "qc_panel_detected_proteins",
    "missing_rate": "qc_panel_missing_rate",
    "sample_counts": "qc_panel_sample_counts",
    "sample_correlation": "qc_panel_sample_correlation",
}


def _qc_panel_title(panel_key: str) -> str:
    """Panel name of an omitted QC panel (falls back to the raw manifest key)."""
    key = QC_PANEL_TITLES.get(str(panel_key))
    return _cap(key) if key else str(panel_key)


def _caption_qc(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    ss = context.get("sample_summary") or {}
    meta = record.get("caption_metadata") or {}
    n_unannotated = _to_int(meta.get("n_unannotated")) or 0
    large_n = bool(meta.get("large_n_mode"))
    omitted = [str(x) for x in (meta.get("omitted_panels") or [])]
    groups = ss.get("groups") or []
    interp_bits = [_cap("qc_interp_base")]
    if groups:
        detail = _cap("punct_enum").join(f"{g['group']}: {g['n']}" for g in groups)
        interp_bits.append(
            _cap("qc_total_prefix")
            + str(ss.get("n_total"))
            + _cap("qc_total_suffix")
            + detail
            + _cap("punct_paren_close")
        )
    if n_unannotated > 0:
        interp_bits.append(
            _cap("qc_unannotated_prefix")
            + str(n_unannotated)
            + _cap("qc_unannotated_suffix")
        )
    if ss.get("mean_detected") is not None:
        interp_bits.append(
            _cap("mean_detected_prefix")
            + _fmt_num(ss["mean_detected"])
            + _cap("protein_count_suffix")
        )
    if ss.get("mean_missing") is not None:
        interp_bits.append(
            _cap("mean_missing_prefix") + _fmt_pct(ss["mean_missing"])
        )
    if large_n:
        interp_bits.append(_cap("qc_large_n_note"))
    panels: List[str] = []
    if "detected_proteins" not in omitted:
        if large_n:
            panels.append(_cap("qc_panel_detected_line_large"))
        else:
            panels.append(_cap("qc_panel_detected_line"))
    if "missing_rate" not in omitted:
        if large_n:
            panels.append(_cap("qc_panel_missing_line_large"))
        else:
            panels.append(_cap("qc_panel_missing_line"))
    if "sample_counts" not in omitted:
        panels.append(_cap("qc_panel_counts_line"))
    if "sample_correlation" not in omitted:
        panels.append(_cap("qc_panel_correlation_line"))
    if omitted:
        omitted_names = _cap("punct_enum").join(_qc_panel_title(key) for key in omitted)
        panels.append(
            _cap("panel_omitted_prefix") + omitted_names + _cap("panel_omitted_suffix_qc")
        )
    return {
        "title": title or _plot_title("qc_sample_overview"),
        "panels": panels,
        "reading": _cap("qc_reading_a") + _cap("qc_reading_b"),
        "thresholds": [
            _cap("qc_threshold_note"),
            _sample_line(context),
        ],
        "statistics": _cap("qc_statistics"),
        "interpretation": _cap("punct_semicolon").join(interp_bits) + _cap("punct_period"),
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_pca(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    group_col = context.get("group_col") or _cap("group_col_default")
    outline = context.get("outline") or {}
    claim = _clean_cell(outline.get("executive_claim"))
    interp = _cap("pca_interp_prefix") + str(group_col) + _cap("pca_interp_suffix")
    if claim:
        interp += _cap("claim_prefix") + claim + _cap("claim_suffix")
    return {
        "title": title or _plot_title("pca"),
        "panels": [
            _cap("pca_panel_scatter"),
            _cap("pca_panel_groups"),
        ],
        "reading": (
            _cap("pca_reading_a") + str(group_col) + _cap("pca_reading_b")
        ),
        "thresholds": [
            _cap("pca_threshold_note"),
            _sample_line(context),
        ],
        "statistics": _cap("pca_statistics"),
        "interpretation": interp,
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_umap(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    group_col = context.get("group_col") or _cap("group_col_default")
    outline = context.get("outline") or {}
    claim = _clean_cell(outline.get("executive_claim"))
    interp = _cap("umap_interp_prefix") + str(group_col) + _cap("umap_interp_suffix")
    if claim:
        interp += _cap("claim_prefix") + claim + _cap("claim_suffix")
    return {
        "title": title or _plot_title("umap"),
        "panels": [
            _cap("umap_panel_scatter"),
        ],
        "reading": (
            _cap("umap_reading_a") + str(group_col) + _cap("umap_reading_b")
        ),
        "thresholds": [
            _cap("umap_threshold_note"),
            _sample_line(context),
        ],
        "statistics": _cap("umap_statistics"),
        "interpretation": interp,
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_heatmap(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    group_col = context.get("group_col") or _cap("group_col_default")
    top = th.get("top_proteins")
    scale = (
        _cap("heatmap_scale_top_prefix") + _fmt_num(top) + _cap("heatmap_scale_top_suffix")
        if top is not None
        else _cap("heatmap_scale_all")
    )
    return {
        "title": title or _plot_title("heatmap"),
        "panels": [
            _cap("heatmap_panel_prefix")
            + scale
            + _cap("heatmap_panel_mid")
            + str(group_col)
            + _cap("heatmap_panel_suffix"),
            _cap("heatmap_panel_colorbar"),
        ],
        "reading": _cap("heatmap_reading_a") + _cap("heatmap_reading_b"),
        "thresholds": [
            _threshold_sentence(th),
            _sample_line(context),
        ],
        "statistics": _cap("heatmap_statistics"),
        "interpretation": (
            _cap("heatmap_interp_prefix")
            + str(group_col)
            + _cap("heatmap_interp_mid")
            + _cap("heatmap_interp_tail")
        ),
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _volcano_contrast(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    return re.sub(r"_volcano_plot$", "", stem)


def _caption_volcano(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    contrast = _volcano_contrast(file_name)
    group_a, group_b = _split_contrast(contrast)
    counts = _contrast_counts(record, context, contrast)
    stats_row = (context.get("contrast_stats") or {}).get(contrast, {})
    if not group_a:
        group_a, group_b = (
            stats_row.get("group_a") or _cap("group_a_default")
        ), (stats_row.get("group_b") or _cap("group_b_default"))

    count_line = _cap("volcano_count_missing")
    if counts.get("n_sig") is not None or (counts.get("n_up") is not None and counts.get("n_down") is not None):
        count_line = (
            _cap("count_sig_prefix")
            + str(counts.get("n_sig"))
            + _cap("count_sig_paren")
            + _clean_cell(counts.get("source"))
            + _cap("count_source_close")
            + _cap("count_up_label")
            + str(counts.get("n_up"))
            + _cap("count_down_label")
            + str(counts.get("n_down"))
            + _cap("count_direction_prefix")
            + str(group_a)
            + _cap("relative_to")
            + str(group_b)
            + _cap("count_direction_close")
        )

    interp_parts: List[str] = []
    if counts.get("n_sig") is not None:
        interp_parts.append(
            str(group_a)
            + _cap("relative_to")
            + str(group_b)
            + _cap("interp_total_prefix")
            + str(counts.get("n_sig"))
            + _cap("interp_count_open")
            + str(counts.get("n_up"))
            + _cap("interp_count_down")
            + str(counts.get("n_down"))
            + _cap("interp_count_close")
        )
    top_up = _extract_top_genes(stats_row.get("top_up"))
    top_down = _extract_top_genes(stats_row.get("top_down"))
    if top_up:
        interp_parts.append(
            _cap("interp_up_genes_prefix") + _cap("punct_enum").join(top_up)
        )
    if top_down:
        interp_parts.append(
            _cap("interp_down_genes_prefix") + _cap("punct_enum").join(top_down)
        )
    row_interp = _clean_cell(stats_row.get("interpretation"))
    if row_interp:
        interp_parts.append(row_interp)
    interp_parts = _drop_hollow_purpose_parts(interp_parts)
    if not interp_parts:
        interp_parts = [_cap("interp_direction_fallback")]
    interpretation = _cap("punct_semicolon").join(interp_parts) + _cap("punct_period")

    return {
        "title": title or _human_contrast(contrast) + _cap("volcano_title_suffix"),
        "panels": [
            _cap("volcano_panel_points_a")
            + str(group_a)
            + _cap("relative_to")
            + str(group_b)
            + _cap("volcano_panel_points_b"),
            _cap("volcano_panel_thresholds"),
            _cap("volcano_panel_colors"),
            _cap("volcano_panel_labels"),
        ],
        "reading": (
            _cap("volcano_reading_a")
            + str(group_a)
            + _cap("volcano_reading_b")
            + str(group_a)
            + _cap("volcano_reading_c")
            + _cap("volcano_reading_d")
        ),
        "thresholds": [
            _threshold_sentence(th),
            count_line,
            _sample_line(context),
        ],
        "statistics": (
            _cap("volcano_statistics_prefix")
            + str(th.get("fdr_method", "BH"))
            + _cap("volcano_statistics_suffix")
        ),
        "interpretation": interpretation,
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_barplot(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    stats = context.get("contrast_stats") or {}
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for row in stats.values():
        contrast = row.get("contrast") or ""
        if contrast and contrast not in seen:
            seen.add(contrast)
            ordered.append(row)
    count_bits: List[str] = []
    for row in ordered:
        if row.get("n_sig") is None and row.get("n_up") is None:
            continue
        count_bits.append(
            _human_contrast(row["contrast"])
            + _cap("bar_count_up")
            + str(row.get("n_up"))
            + _cap("bar_count_down")
            + str(row.get("n_down"))
        )
    count_line = (
        _cap("bar_count_label")
        + _cap("punct_semicolon").join(count_bits)
        + _cap("punct_period")
        if count_bits
        else _cap("bar_count_missing")
    )
    biggest = None
    for row in ordered:
        n_sig = row.get("n_sig")
        if n_sig is None:
            continue
        if biggest is None or (biggest.get("n_sig") or 0) < n_sig:
            biggest = row
    interp = _cap("bar_interp_base")
    if biggest is not None:
        interp += (
            _cap("bar_interp_biggest_prefix")
            + _human_contrast(biggest["contrast"])
            + _cap("bar_interp_biggest_mid")
            + str(biggest.get("n_sig"))
            + _cap("bar_interp_biggest_suffix")
        )
    interp += _cap("bar_interp_tail")
    return {
        "title": title or _plot_title("differential_summary_barplot"),
        "panels": [
            _cap("bar_panel_bars"),
            _cap("bar_panel_labels"),
        ],
        "reading": _cap("bar_reading_a") + _cap("bar_reading_b"),
        "thresholds": [
            _cap("bar_threshold_prefix")
            + _fmt_num(th.get("pval"))
            + _cap("bar_threshold_mid")
            + _fmt_num(th.get("logfc"))
            + _cap("punct_paren_open")
            + _th_fdr(th)
            + _cap("fdr_correction_suffix")
            + _cap("punct_period")
            if th.get("pval") is not None or th.get("logfc") is not None
            else _cap("bar_threshold_missing"),
            count_line,
            _sample_line(context),
        ],
        "statistics": (
            _cap("bar_statistics_prefix")
            + _th_fdr(th)
            + _cap("bar_statistics_suffix")
        ),
        "interpretation": interp,
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _focus_genes_line(context: Dict[str, Any], k: int = 6) -> str:
    outline = context.get("outline") or {}
    genes = [str(g).strip() for g in (outline.get("focus_genes") or []) if str(g).strip()]
    if not genes:
        return ""
    shown = genes[:k]
    suffix = _cap("focus_genes_more") if len(genes) > k else ""
    return (
        _cap("focus_genes_prefix")
        + _cap("punct_enum").join(shown)
        + suffix
        + _cap("focus_genes_suffix")
    )


def _caption_key_protein_overview(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    per_contrast = th.get("top_per_contrast")
    select_line = _cap("select_label") + (
        _threshold_core(th) or _cap("no_threshold_recorded")
    )
    if per_contrast is not None:
        select_line += (
            _cap("per_contrast_top_prefix")
            + _fmt_num(per_contrast)
            + _cap("per_contrast_top_suffix")
        )
    select_line += _cap("punct_period")
    focus = _focus_genes_line(context)
    interp = _cap("key_protein_interp")
    if focus:
        interp += focus
    return {
        "title": title or _plot_title("key_protein_overview"),
        "panels": [
            _cap("key_panel_points"),
            _cap("key_panel_size"),
            _cap("key_panel_labels"),
        ],
        "reading": _cap("key_reading_a") + _cap("key_reading_b"),
        "thresholds": [
            select_line,
            _sample_line(context),
        ],
        "statistics": (
            _cap("limma_stats_prefix")
            + _th_fdr(th)
            + _cap("fdr_correction_suffix")
            + _cap("punct_period")
        ),
        "interpretation": interp,
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_protein_bubble(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    per_contrast = th.get("top_per_contrast")
    max_proteins = th.get("max_proteins")
    select_bits: List[str] = []
    if per_contrast is not None:
        select_bits.append(
            _cap("per_contrast_take_prefix")
            + _fmt_num(per_contrast)
            + _cap("per_contrast_take_suffix")
        )
    if max_proteins is not None:
        select_bits.append(
            _cap("max_proteins_prefix")
            + _fmt_num(max_proteins)
            + _cap("max_proteins_suffix")
        )
    select_line = _cap("select_label") + (
        _threshold_core(th) or _cap("no_threshold_recorded")
    )
    if select_bits:
        select_line += _cap("punct_semicolon") + _cap("punct_semicolon").join(select_bits)
    select_line += _cap("punct_period")
    focus = _focus_genes_line(context)
    interp = _cap("bubble_interp_a") + _cap("bubble_interp_b")
    if focus:
        interp += focus
    return {
        "title": title or _plot_title("protein_contrast_bubble"),
        "panels": [
            _cap("bubble_panel_axes"),
            _cap("bubble_panel_color"),
            _cap("bubble_panel_size"),
        ],
        "reading": _cap("bubble_reading_a") + _cap("bubble_reading_b"),
        "thresholds": [
            select_line,
            _sample_line(context),
        ],
        "statistics": (
            _cap("limma_stats_prefix")
            + _th_fdr(th)
            + _cap("fdr_correction_suffix")
            + _cap("punct_period")
        ),
        "interpretation": interp,
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_top_means(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    group_col = context.get("group_col") or _cap("group_col_default")
    top = th.get("top_proteins")
    module_summary = context.get("module_summary") or {}
    module_order = module_summary.get("order") or []
    module_bits = ""
    if module_order:
        name = module_order[0]
        info = (module_summary.get("modules") or {}).get(name, {})
        n_genes = info.get("n_genes")
        module_bits = (
            _cap("module_cross_prefix")
            + str(name)
            + _cap("punct_comma")
            + _fmt_num(n_genes)
            + _cap("module_cross_mid")
            + _cap("module_cross_tail")
        )
    else:
        module_bits = _cap("module_cross_plain")
    scale = (
        _cap("top_means_scale_a")
        + _fmt_num(th.get("pval"))
        + _cap("top_means_scale_b")
        + _fmt_num(top)
        + _cap("top_means_scale_c")
        if top is not None
        else _cap("top_means_scale_all")
    )
    protein_set_line = (
        _cap("protein_set_label")
        + scale
        + _cap("punct_paren_open")
        + _threshold_core(th)
        + _cap("punct_paren_close")
        + _cap("punct_period")
        if top is not None and _threshold_core(th)
        else _cap("protein_set_label") + scale + _cap("punct_period")
    )
    return {
        "title": title or _plot_title("mechanism_top_protein_group_means"),
        "panels": [
            _cap("top_means_panel_left_prefix")
            + scale
            + _cap("top_means_panel_left_mid")
            + str(group_col)
            + _cap("top_means_panel_left_suffix"),
            _cap("top_means_panel_right"),
        ],
        "reading": _cap("top_means_reading_a") + _cap("top_means_reading_b"),
        "thresholds": [
            protein_set_line,
            _sample_line(context),
        ],
        "statistics": _cap("top_means_statistics"),
        "interpretation": (
            _cap("top_means_interp_prefix")
            + str(group_col)
            + _cap("top_means_interp_mid")
            + module_bits
            + _cap("punct_period")
        ),
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_enrichment_dotplot(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    top_terms = th.get("top_terms")
    databases = _databases_from_inputs(record)
    db_line = (
        _cap("punct_enum").join(databases) if databases else _cap("db_fallback")
    )
    scale = (
        _cap("dotplot_scale_prefix")
        + _fmt_num(top_terms)
        + _cap("dotplot_scale_suffix")
        if top_terms is not None
        else _cap("dotplot_scale_all")
    )
    return {
        "title": title or _plot_title("mechanism_enrichment_dotplot"),
        "panels": [
            _cap("dotplot_panel_points"),
            _cap("dotplot_panel_size"),
            _cap("dotplot_panel_color"),
        ],
        "reading": _cap("dotplot_reading_a") + _cap("dotplot_reading_b"),
        "thresholds": [
            _cap("term_scale_label") + scale + _cap("punct_period"),
            _cap("input_db_label") + db_line + _cap("punct_period"),
            _sample_line(context),
        ],
        "statistics": _cap("dotplot_statistics"),
        "interpretation": _cap("dotplot_interp_a") + _cap("dotplot_interp_b"),
        "boundary": boundary_enrichment(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _parse_enrichment_bubble_name(file_name: str) -> Dict[str, str]:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    info: Dict[str, str] = {}
    match = re.match(r"([A-Za-z]+)_(.+?)_(upregulated|downregulated)_proteins_enrichment_bubble", stem, re.IGNORECASE)
    if match:
        info["database"] = match.group(1)
        info["contrast"] = match.group(2)
        info["direction"] = match.group(3).lower()
    return info


def _caption_enrichment_bubble(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    info = _parse_enrichment_bubble_name(file_name)
    database = info.get("database", "").upper() or "GO/KEGG/Reactome"
    contrast_human = _human_contrast(info.get("contrast", "")) or _cap("contrast_default")
    direction = info.get("direction", "")
    direction_label = {
        "upregulated": _cap("direction_upregulated"),
        "downregulated": _cap("direction_downregulated"),
    }.get(direction, direction or _cap("direction_default"))
    top_n = th.get("top_terms")
    source_line = (
        _cap("source_file_label")
        + os.path.basename(str((record.get("inputs") or [""])[0]))
        if record.get("inputs")
        else _cap("source_file_missing")
    )
    scale = (
        _cap("bubble_scale_prefix") + _fmt_num(top_n) + _cap("bubble_scale_suffix")
        if top_n is not None
        else _cap("bubble_scale_all")
    )
    return {
        "title": title or (
            database
            + " "
            + contrast_human
            + " "
            + direction_label
            + _cap("enrichment_bubble_title_suffix")
        ),
        "panels": [
            _cap("enrich_bubble_panel_axes"),
            _cap("enrich_bubble_panel_size"),
            _cap("enrich_bubble_panel_color"),
        ],
        "reading": _cap("enrich_bubble_reading_a") + _cap("enrich_bubble_reading_b"),
        "thresholds": [
            _cap("term_scale_label") + scale + _cap("punct_period"),
            source_line + _cap("punct_period"),
            _sample_line(context),
        ],
        "statistics": _cap("enrich_bubble_statistics"),
        "interpretation": (
            _cap("enrich_bubble_interp_prefix")
            + database
            + _cap("enrich_bubble_interp_mid")
            + contrast_human
            + " "
            + direction_label
            + _cap("enrich_bubble_interp_tail")
            + _cap("enrich_bubble_interp_tail2")
        ),
        "boundary": boundary_enrichment(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_pathway_heatmap(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    caption = _clean_cell(record.get("caption"))
    term_match = re.search(r"`([^`]+)`", caption)
    term = term_match.group(1) if term_match else os.path.splitext(os.path.basename(file_name))[0]
    info = _parse_enrichment_bubble_name(file_name)
    contrast_human = _human_contrast(info.get("contrast", ""))
    group_col = context.get("group_col") or _cap("group_col_default")
    source_line = (
        _cap("source_file_label")
        + os.path.basename(str((record.get("inputs") or [""])[0]))
        if record.get("inputs")
        else _cap("source_file_missing")
    )
    return {
        "title": title or _cap("pathway_heatmap_title_prefix") + str(term),
        "panels": [
            _cap("pathway_panel_prefix")
            + str(term)
            + _cap("pathway_panel_mid")
            + str(group_col)
            + _cap("pathway_panel_suffix"),
            _cap("pathway_panel_color"),
        ],
        "reading": _cap("pathway_reading_a") + _cap("pathway_reading_b"),
        "thresholds": [
            _cap("pathway_set_prefix") + str(term) + _cap("pathway_set_suffix"),
            source_line + _cap("punct_period"),
            _sample_line(context),
        ],
        "statistics": _cap("pathway_statistics"),
        "interpretation": (
            _cap("pathway_interp_prefix")
            + str(term)
            + _cap("pathway_interp_mid")
            + (
                " " + contrast_human + " "
                if contrast_human
                else _cap("pathway_interp_all_contrasts")
            )
            + _cap("pathway_interp_suffix")
            + _cap("pathway_interp_tail")
        ),
        "boundary": boundary_mixed(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_upset(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    return {
        "title": title or _plot_title("mechanism_dep_overlap_upset"),
        "panels": [
            _cap("upset_panel_summary"),
            _cap("upset_panel_labels"),
        ],
        "reading": _cap("upset_reading"),
        "thresholds": [
            _threshold_sentence(th),
            _sample_line(context),
        ],
        "statistics": (
            _cap("upset_statistics_prefix")
            + _th_fdr(th)
            + _cap("upset_statistics_suffix")
        ),
        "interpretation": _cap("upset_interp_a") + _cap("upset_interp_b"),
        "boundary": boundary_matrix(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_contrast_evidence(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    contrast = _human_contrast(os.path.splitext(os.path.basename(file_name))[0].replace("mechanism_contrast_evidence_", ""))
    th = _resolve_thresholds(record, context)
    meta = record.get("caption_metadata") or {}
    omitted = [str(x) for x in (meta.get("omitted_panels") or [])]
    omitted_names = {
        "enrichment_terms": "ce_panel_enrichment_terms",
        "group_means": "ce_panel_group_means",
        "volcano_evidence": "ce_panel_volcano",
        "top_proteins": "ce_panel_top_proteins",
    }
    panels = [
        _cap("ce_panel_prefix") + str(contrast) + _cap("ce_panel_suffix"),
        _cap("ce_panel_consistency"),
    ]
    if omitted:
        names = _cap("punct_enum").join(
            _cap(omitted_names[key]) if key in omitted_names else key for key in omitted
        )
        panels.append(
            _cap("panel_omitted_prefix") + names + _cap("panel_omitted_suffix_ce")
        )
    return {
        "title": title or str(contrast) + _cap("ce_title_suffix"),
        "panels": panels,
        "reading": _cap("ce_reading_a") + _cap("ce_reading_b"),
        "thresholds": [
            _threshold_sentence(th),
            _sample_line(context),
        ],
        "statistics": (
            _cap("ce_statistics_prefix")
            + _th_fdr(th)
            + _cap("ce_statistics_suffix")
        ),
        "interpretation": str(contrast) + _cap("ce_interp_suffix"),
        "boundary": boundary_mixed(),
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_generic(
    file_name: str,
    title: str,
    record: Dict[str, Any],
    context: Dict[str, Any],
    error: str = "",
) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    category = _clean_cell(record.get("category"))
    manifest_caption = _clean_cell(record.get("caption"))
    boundary = boundary_enrichment() if category.startswith("Enrichment") else boundary_matrix()
    interp = _cap("generic_interp")
    if manifest_caption:
        interp += _cap("manifest_caption_label") + manifest_caption
    return {
        "title": title or _clean_cell(record.get("plot_type")) or file_name,
        "panels": [
            _cap("generic_panel_prefix")
            + (category or _cap("not_recorded"))
            + _cap("punct_paren_close")
            + _cap("punct_period"),
        ],
        "reading": _cap("generic_reading"),
        "thresholds": [
            _threshold_sentence(th),
            _sample_line(context),
        ],
        "statistics": _cap("generic_statistics"),
        "interpretation": interp,
        "boundary": boundary,
        "paper_ref": _paper_ref_line(file_name, context),
    }


_PLOT_BUILDERS = {
    "qc_sample_overview": _caption_qc,
    "pca": _caption_pca,
    "umap": _caption_umap,
    "heatmap": _caption_heatmap,
    "volcano": _caption_volcano,
    "differential_summary_barplot": _caption_barplot,
    "key_protein_overview": _caption_key_protein_overview,
    "protein_contrast_bubble": _caption_protein_bubble,
    "mechanism_top_protein_group_means": _caption_top_means,
    "mechanism_enrichment_dotplot": _caption_enrichment_dotplot,
    "enrichment_bubble": _caption_enrichment_bubble,
    "mechanism_pathway_gene_heatmap": _caption_pathway_heatmap,
    "mechanism_dep_overlap_upset": _caption_upset,
    "mechanism_contrast_evidence": _caption_contrast_evidence,
}


def _th_fdr(th: Dict[str, Any]) -> str:
    return str(th.get("fdr_method") or "BH")


# ---------------------------------------------------------------------------
# record building + rendering
# ---------------------------------------------------------------------------


def _match_record(figures_by_name: Dict[str, Dict[str, Any]], file_name: str) -> Optional[Dict[str, Any]]:
    record = figures_by_name.get(file_name)
    if record:
        return record
    stem = os.path.splitext(file_name)[0]
    for name, candidate in figures_by_name.items():
        if name.startswith(stem + "_") or name == file_name:
            return candidate
    return None


def _fallback_title(record: Optional[Dict[str, Any]], file_name: str) -> str:
    if record:
        plot_type = _clean_cell(record.get("plot_type"))
        if plot_type in PLOT_TYPE_TITLES:
            return _plot_title(plot_type)
        if plot_type == "volcano":
            return _human_contrast(_volcano_contrast(file_name)) + _cap("volcano_title_suffix")
        if plot_type == "enrichment_bubble":
            info = _parse_enrichment_bubble_name(file_name)
            return (
                info.get("database", "").upper()
                + " "
                + info.get("contrast", "")
                + _cap("enrichment_bubble_title_fallback_suffix")
            )
        if plot_type == "mechanism_pathway_gene_heatmap":
            term_match = re.search(r"`([^`]+)`", _clean_cell(record.get("caption")))
            if term_match:
                return _cap("pathway_heatmap_title_prefix") + str(term_match.group(1))
    return os.path.splitext(os.path.basename(file_name))[0]


def build_caption_records(context: Dict[str, Any], key_figures: Sequence[Any]) -> List[Dict[str, Any]]:
    """Build one caption record per key figure, in the given order.

    ``key_figures`` items are ``(file_name, title)`` tuples, ``(file_name, title, caption)``
    triples (legacy), or dicts with ``file_name``/``title`` keys.
    """
    figures_by_name = context.get("figures_by_name") or {}
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(key_figures, start=1):
        if isinstance(item, dict):
            file_name = str(item.get("file_name") or item.get("filename") or "")
            title = _clean_cell(item.get("title"))
        elif isinstance(item, (tuple, list)):
            file_name = str(item[0] if len(item) >= 1 else "")
            title = _clean_cell(item[1] if len(item) >= 2 else "")
        else:
            file_name, title = str(item), ""
        record = _match_record(figures_by_name, file_name) or {}
        plot_type = _clean_cell(record.get("plot_type"))
        builder = _PLOT_BUILDERS.get(plot_type, _caption_generic)
        try:
            cap = builder(file_name, title, record, context)
        except Exception:
            cap = _caption_generic(file_name, title, record, context)
        if not _clean_cell(cap.get("title")):
            cap["title"] = _fallback_title(record, file_name)
        cap["no"] = _cap("figure_no_prefix") + str(idx)
        cap["file_name"] = os.path.basename(file_name)
        cap["plot_type"] = plot_type
        # empty-cell discipline: every narrative field must carry text
        for field in ("panels", "reading", "thresholds", "statistics", "interpretation", "boundary", "paper_ref"):
            value = cap.get(field)
            if isinstance(value, list):
                cap[field] = [_clean_cell(v) or _cap("not_recorded_info") for v in value]
                if not cap[field]:
                    cap[field] = [_cap("not_recorded_info")]
            elif not _clean_cell(value):
                cap[field] = _cap("not_recorded_info")
        records.append(cap)
    return records


def _overview_lines(context: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    diff = context.get("differential") or {}
    pval = _to_float(diff.get("p_thresh"))
    logfc = _to_float(diff.get("logfc_thresh"))
    fdr = _clean_cell(diff.get("fdr_method")) or "BH"
    if pval is not None and logfc is not None:
        lines.append(
            _cap("overview_default_prefix")
            + _fmt_num(pval)
            + _cap("punct_semicolon")
            + "|log2FC| ≥ "
            + _fmt_num(logfc)
            + _cap("punct_paren_open")
            + fdr
            + _cap("overview_default_suffix")
        )
    elif pval is not None or logfc is not None:
        lines.append(
            _cap("overview_partial_prefix")
            + (_fmt_num(pval) if pval is not None else _cap("not_recorded"))
            + _cap("punct_semicolon")
            + "|log2FC| ≥ "
            + (_fmt_num(logfc) if logfc is not None else _cap("not_recorded"))
            + _cap("punct_paren_open")
            + fdr
            + _cap("overview_partial_suffix")
        )
    else:
        lines.append(_cap("overview_missing"))
    lines.append(_sample_line(context))
    group_col = context.get("group_col")
    if group_col:
        lines.append(
            _cap("main_group_col_prefix")
            + str(group_col)
            + _cap("main_group_col_suffix")
        )
    generated_at = context.get("generated_at")
    lines.append(
        _cap("numbers_source")
        + (
            _cap("manifest_time_prefix") + str(generated_at) + _cap("punct_period")
            if generated_at
            else _cap("punct_period")
        )
    )
    return lines


def render_figures_captions_md(dataset: str, records: Sequence[Dict[str, Any]], context: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.extend([
        f"# {dataset}" + _cap("captions_doc_title_suffix"),
        "",
        _cap("captions_doc_intro"),
        "",
        _cap("heading_thresholds_overview"),
        "",
    ])
    for line in _overview_lines(context):
        lines.append(f"- {line}")
    lines.extend([
        "",
        _cap("heading_boundary_legend"),
        "",
        f"- {boundary_matrix()}",
        f"- {boundary_enrichment()}",
        _cap("boundary_extension"),
        "",
        _cap("heading_caption_index"),
        "",
        _cap("index_header"),
        "|---|---|---|",
    ])
    for record in records:
        lines.append(f"| {record['no']} | {record['title']} | `{record['file_name']}` |")
    lines.append("")
    for record in records:
        lines.extend([
            f"## {record['no']} {record['title']}",
            "",
            _cap("label_image_file") + f"`{record['file_name']}`",
            _cap("label_panels"),
        ])
        for panel in record["panels"]:
            lines.append(f"  - {panel}")
        lines.extend([
            _cap("label_reading") + f"{record['reading']}",
            _cap("label_thresholds"),
        ])
        for threshold in record["thresholds"]:
            lines.append(f"  - {threshold}")
        lines.extend([
            _cap("label_statistics") + f"{record['statistics']}",
            _cap("label_interpretation") + f"{record['interpretation']}",
            _cap("label_boundary") + f"{record['boundary']}",
            _cap("label_paper_ref") + f"{record['paper_ref']}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def render_brief_figures_md(
    dataset: str,
    records: Sequence[Dict[str, Any]],
    links: Sequence[str],
    title_suffix: str = "",
    intro: str = "",
) -> str:
    lines: List[str] = [
        f"# {dataset} " + (title_suffix or _cap("brief_title_suffix_default")),
        "",
    ]
    lines.append(
        intro or (_cap("brief_intro_default_a") + _cap("brief_intro_default_b"))
    )
    lines.append("")
    for record, link in zip(records, links):
        heading = f"{record['no']} {record['title']}"
        lines.extend([
            f"## {heading}",
            "",
            f"![{heading}]({link})",
            "",
            _cap("brief_pointer_prefix")
            + f"`figures_captions.md` §{record['no']}"
            + _cap("punct_period"),
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# offline document writer (CLI + main_agent helper)
# ---------------------------------------------------------------------------


def default_key_figures(context: Dict[str, Any]) -> List[Tuple[str, str]]:
    """All manifest figures ordered by canonical category, with derived titles."""
    figures = context.get("figures_by_name") or {}
    category_rank = {name: rank for rank, name in enumerate(CATEGORY_ORDER)}

    def sort_key(item: Tuple[str, Dict[str, Any]]):
        record = item[1]
        category = _clean_cell(record.get("category"))
        return (category_rank.get(category, len(CATEGORY_ORDER)), item[0])

    result: List[Tuple[str, str]] = []
    for file_name, record in sorted(figures.items(), key=sort_key):
        result.append((file_name, _fallback_title(record, file_name)))
    return result


def write_run_caption_assets(
    run_folder: str,
    out_dir: Optional[str] = None,
    key_figures: Optional[Sequence[Any]] = None,
    project_root: Optional[str] = None,
    copy_figures: bool = True,
) -> Dict[str, Any]:
    """Generate figures_captions.md + slimmed figures.md / figures_preview_local.md.

    Writes into ``out_dir`` (default ``<run_folder>/caption_preview``) and never
    touches the run folder itself; referenced PNGs are copied to ``<out_dir>/figures``
    so all links resolve.
    """
    context = load_run_context(run_folder, project_root=project_root)
    dataset = context.get("dataset") or infer_dataset_name(run_folder)
    if not key_figures:
        key_figures = default_key_figures(context)
    records = build_caption_records(context, list(key_figures))
    out_dir = os.path.normpath(out_dir or os.path.join(run_folder, "caption_preview"))
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    links: List[str] = []
    abs_links: List[str] = []
    for record in records:
        source = os.path.join(run_folder, "visualize_results", record["file_name"])
        if not os.path.exists(source):
            source = os.path.join(run_folder, "figures", record["file_name"])
        base = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(record["file_name"])[0]).strip("_").lower()[:48] or "figure"
        target_name = f"{base}.png"
        target = os.path.join(figures_dir, target_name)
        if copy_figures and os.path.exists(source):
            try:
                shutil.copy2(source, target)
            except Exception:
                pass
        links.append(f"figures/{target_name}")
        abs_links.append(os.path.abspath(target).replace("\\", "/"))

    captions_md = render_figures_captions_md(dataset, records, context)
    figures_md = render_brief_figures_md(dataset, records, links)
    local_md = render_brief_figures_md(
        dataset,
        records,
        abs_links,
        title_suffix=_cap("local_title_suffix"),
        intro=_cap("local_intro_a") + _cap("local_intro_b"),
    )

    written: Dict[str, str] = {
        "figures_captions": os.path.join(out_dir, CAPTIONS_FILENAME),
        "figures_md": os.path.join(out_dir, FIGURES_MD_FILENAME),
        "figures_local_md": os.path.join(out_dir, FIGURES_LOCAL_MD_FILENAME),
    }
    for path, content in (
        (written["figures_captions"], captions_md),
        (written["figures_md"], figures_md),
        (written["figures_local_md"], local_md),
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
    return {
        "dataset": dataset,
        "records": records,
        "context": context,
        "captions_md": captions_md,
        "figures_md": figures_md,
        "figures_local_md": local_md,
        "written": written,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate figures_captions.md and slimmed figure albums for a run.")
    parser.add_argument("--run-folder", required=True, help="Run directory containing visualize_results/figure_manifest.json")
    parser.add_argument("--figures", default="", help="Comma-separated visualize_results figure file names; default = all manifest figures")
    parser.add_argument("--out-dir", default="", help="Output directory (default <run_folder>/caption_preview)")
    parser.add_argument("--project-root", default="", help="Project root containing docs/figure_upgrade_20260904 (default: module dir)")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy referenced PNGs into the output directory")
    args = parser.parse_args(argv)

    key_figures: Optional[List[Tuple[str, str]]] = None
    if args.figures.strip():
        key_figures = []
        for token in args.figures.split(","):
            name = token.strip()
            if not name:
                continue
            stem = os.path.splitext(name)[0]
            title = _plot_title(stem)
            if stem.endswith("_volcano_plot"):
                title = _human_contrast(
                    stem[: -len("_volcano_plot")]
                ) + _cap("volcano_title_suffix")
            key_figures.append((name, title or ""))

    result = write_run_caption_assets(
        args.run_folder,
        out_dir=args.out_dir or None,
        key_figures=key_figures,
        project_root=args.project_root or None,
        copy_figures=not args.no_copy,
    )
    sys.stdout.write(
        "dataset={dataset} figures={n}\n".format(dataset=result["dataset"], n=len(result["records"]))
        + "\n".join(f"{key}: {path}" for key, path in result["written"].items())
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
