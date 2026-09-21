# -*- coding: utf-8 -*-
"""Hybrid report assembly v2 (2026-09-13): model explains, deterministic code owns the facts.

Division of labour
------------------
* model  : research question, result organisation, mechanism explanation (kept verbatim)
* code   : contrast direction, values, thresholds, sample sizes, candidate/module tables,
           task-section assembly, complete appendix, confirmed-contradiction repair stays local

v2 fixes the four defects found in review
-----------------------------------------
1. composite identity keys: a candidate is keyed by (contrast, full protein group or gene) and a
   module by (contrast, module) - name-only keys silently overwrote records across contrasts.
2. thresholds / effect scale come from the run artifacts; a missing record is written as missing
   (never filled with a default that then reads as a fact), and the effect column is only called
   logFC when a log2 scale is actually recorded. Table cells escape | so the markdown grid holds.
3. content ledger: every block of the source text is registered before parsing and afterwards
   marked preserved / replaced-by-deterministic-table / unbound-section / empty, with a reason.
   Nothing is dropped silently; the candidate and module appendix lists every record, not a prefix.
4. explicit task mapping by claim_id / display contrast / source contrast only. No substring
   matching, no empty-key matching, no reuse of one section by several contrasts; an unbound model
   section keeps its own heading and is registered as unbound instead of being relabelled.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import report_evidence as evidence
from report_language import get_language, register, t, t_in

# --------------------------------------------------------------------------- reader-facing wording
# Every reader-facing string this module emits is registered here as key -> (zh, en). The zh
# value of a key is byte-identical to the literal it replaces, so a zh run renders the released
# report text unchanged; the en value is a translation into report English. When the active
# language has no template, report_language.t returns its MISSING_EN_TEMPLATE marker and records
# the key, so an English report can never silently fall back to Chinese.
_STRINGS = {
    'not_recorded': ('未记录', 'not recorded'),
    'status_computed': ('已计算', 'computed'),
    'status_recorded': ('已记录', 'recorded'),
    'status_not_computed': ('未计算', 'not computed'),
    'arm_higher': ('%s 较高', '%s higher'),
    'severity_hint': ('提示', 'hint'),
    'status_not_executed': ('未执行', 'not executed'),
    'arm_first': ('第一臂', 'arm 1'),
    'evidence_strength_low_to_moderate': ('证据强度：较低至中等', 'Evidence strength: low to moderate'),
    'method_not_recorded': ('方法未记录', 'method not recorded'),
    'group_a': ('A 组', 'Group A'),
    'group_b': ('B 组', 'Group B'),
    'analysis_design': ('分析设计', 'analysis design'),
    'topic_dose_trend': ('剂量趋势', 'dose trend'),
    'not_detected_in_matrix': ('在可用矩阵中未检出', 'not detected in the available matrix'),
    'external_annotation': ('外部注释', 'external annotation'),
    'topic_contrast_concordance': ('对比一致性', 'contrast concordance'),
    'current_matrix': ('当前矩阵', 'current matrix'),
    'source_dataset_provided': ('数据集提供', 'provided with the dataset'),
    'source_current_matrix': ('本次分析的数据矩阵', 'the data matrix of this analysis'),
    'threshold_adj_p_and_effect': ('校正 P≤%s 且 |效应量|≥%s', 'adjusted P≤%s and |effect size|≥%s'),
    'topic_sample_count': ('样本数', 'sample count'),
    'source_offline_enrichment': ('离线富集结果', 'offline enrichment result'),
    'arm_second': ('第二臂', 'arm 2'),
    'topic_missing_rate': ('缺失率', 'missing rate'),
    'evidence_strength_moderate': ('证据强度：中等', 'Evidence strength: moderate'),
    'evidence_strength_low': ('证据强度：较低', 'Evidence strength: low'),
    'keyword_boundary': ('边界', 'boundary'),
    'item_matrix_stages': ('运行链路上的矩阵阶段', 'matrix stages along the run chain'),
    'arm_higher_suffix': (' 较高', ' higher'),
    'heading_stratified_summary': ('### 分层对比摘要', '### Stratified Contrast Summary'),
    'heading_contrast_concordance': ('### 对比一致性', '### Contrast Concordance'),
    'heading_ora_hypergeometric': ('### 正式富集检验（超几何）', '### Formal Enrichment Test (hypergeometric)'),
    'arm_higher_in': ('%s 中较高', 'higher in %s'),
    'protein_filtering_text': ('%s；保留蛋白 %s 个', '%s; %s proteins retained'),
    'skipped_test_line': ('- 未执行的检验：%s', '- Test not run: %s'),
    'direction_upregulated': ('上调', 'upregulated'),
    'enrichment_up_direction': ('上调方向', 'upregulated direction'),
    'direction_downregulated': ('下调', 'downregulated'),
    'enrichment_down_direction': ('下调方向', 'downregulated direction'),
    'connector_and': ('且', 'and'),
    'asset_candidate_exclusion': ('候选排除记录', 'candidate exclusion record'),
    'topic_stratified': ('分层对比', 'stratified contrast'),
    'field_group_and_id_cols': ('分组与标识列', 'grouping and identifier columns'),
    'field_covariates': ('协变量', 'covariates'),
    'source_external_literature': ('外部文献', 'external literature'),
    'field_fdr_method': ('多重校正方法', 'multiple-testing method'),
    'form_not_recorded': ('形式未记录', 'form not recorded'),
    'connector_or': ('或', 'or'),
    'field_effect_size': ('效应量', 'effect size'),
    'direction_undetermined': ('方向未判定', 'direction not determined'),
    'direction_not_recorded': ('方向未记录', 'direction not recorded'),
    'none_value': ('无', 'none'),
    'dose_no_monotonic_trend': ('无单调趋势', 'no monotonic trend'),
    'kind_significant_set': ('显著集', 'significant set'),
    'enrichment_status_valid_empty': ('有效空结果', 'valid empty result'),
    'not_entered_statistical_test': ('未进入统计检验', 'did not enter statistical testing'),
    'field_tested_proteins': ('检验蛋白数', 'proteins tested'),
    'topic_formal_enrichment': ('正式富集检验', 'formal enrichment test'),
    'note_drug_stratum': ('注：%s / %s %s', 'Note: %s / %s %s'),
    'source_auto_inferred': ('系统自动推断', 'automatically inferred by the system'),
    'source_drug_database': ('药物数据库', 'drug database'),
    'field_design_source': ('设计来源', 'design source'),
    'field_design_threshold': ('设计阈值', 'design threshold'),
    'evidence_strength_low_and_moderate': ('证据强度：分别为较低与中等', 'Evidence strength: low and moderate, respectively'),
    'lower_to_moderate': ('较低至中等', 'low to moderate'),
    'topic_pathway_terms': ('通路富集条目', 'pathway enrichment entries'),
    'status_partially_recorded': ('部分记录', 'partially recorded'),
    'topic_dimensionality_and_figures': ('降维与图件', 'dimensionality reduction and figures'),
    'dose_rising': ('随剂量上升', 'rising with dose'),
    'dose_falling': ('随剂量下降', 'falling with dose'),
    'evidence_this_run_diff': ('（证据：本轮差异分析）', '(evidence: differential analysis of this run)'),
    'imputed_values_suffix': ('，填补 %s 个取值', ', %s values imputed'),
    'no_further_repeat': ('，此处不再重复。', '; not repeated here.'),
    'unit_effect_summary_line': ('  - %s：已检验 %s 个蛋白，效应量范围 %s 至 %s（中位 %s）；其中 %s 个蛋白的 95%% 置信区间不跨 0（逐蛋白、未经多重校正）。', '  - %s: %s proteins tested, effect sizes from %s to %s (median %s); for %s of them the 95%% confidence interval does not cross 0 (per protein, without multiple-testing correction).'),
    'ora_class_significant_line': ('  - %s：规则 %s；查询集 n=%s；涉及对比 %s', '  - %s: rule %s; query set n=%s; contrasts involved %s'),
    'ora_class_fallback_line': ('  - %s：该对比显著集为 %s 个蛋白；兜底查询集 n=%s；涉及对比 %s', '  - %s: the significant set of this contrast holds %s proteins; fallback query set n=%s; contrasts involved %s'),
    'volcano_row_with_arms': ('  - %s：通过筛选 %s 个（%s 较高 %s、%s 较高 %s；按该差异表自身符号）', '  - %s: %s passed screening (%s higher %s, %s higher %s; per the sign convention of that differential table)'),
    'volcano_row_plain': ('  - %s：通过筛选 %s 个（上调 %s、下调 %s）', '  - %s: %s passed screening (upregulated %s, downregulated %s)'),
    'title_prefix': ('# scProteoAgent 分析报告', '# scProteoAgent Analysis Report'),
    'heading_task': ('## %d. 任务%d：%s', '## %d. Task %d: %s'),
    'heading_assets': ('## %d. 可复核资产清单', '## %d. Reproducible Asset List'),
    'heading_figures': ('## %d. 图表与文字结论', '## %d. Figures and Text Conclusions'),
    'heading_summary': ('## %d. 总结与启发式推测', '## %d. Summary and Heuristic Inference'),
    'heading_boundary': ('## %d. 结论边界与方法局限', '## %d. Conclusion Boundaries and Method Limitations'),
    'heading_evidence_tables': ('## %d. 证据表（逐条）', '## %d. Evidence Tables (record by record)'),
    'heading_run_parameters': ('## %d. 运行参数与证据绑定', '## %d. Run Parameters and Evidence Binding'),
    'heading_key_conclusions': ('## 0. 关键结论速览', '## 0. Key Conclusions at a Glance'),
    'heading_data': ('## 1. 数据与预处理', '## 1. Data and Preprocessing'),
    'request_index_title': ('## 本轮可用事实清单（由运行工件直接生成，与最终报告表格同源）', '## Facts available for this run (generated directly from the run artifacts, from the same source as the final report tables)'),
    'heading_exclusion_trace': ('### 候选排除轨迹', '### Candidate Exclusion Trace'),
    'heading_candidate_stats': ('### 候选蛋白逐对比统计', '### Candidate Proteins by Contrast'),
    'heading_extra_unbound': ('### 其他说明（未能绑定到任务小节的内容）', '### Other Notes (content that could not be bound to a task section)'),
    'heading_stratified_computed': ('### 分层对比摘要（已计算：%d 行；分层变量：%s）', '### Stratified Contrast Summary (computed: %d rows; stratifiers: %s)'),
    'heading_dose_trend': ('### 剂量趋势检验（Spearman）', '### Dose-Trend Test (Spearman)'),
    'heading_dose_trend_computed': ('### 剂量趋势（Spearman，已计算）', '### Dose Trend (Spearman, computed)'),
    'heading_consistency_note': ('### 口径与一致性说明', '### Scope and Consistency Notes'),
    'heading_task_ids': ('### 可用 task_id（每个对比一个；必须逐字使用）', '### Available task_id values (one per contrast; use verbatim)'),
    'heading_figures_plain': ('### 图册', '### Figure Index'),
    'heading_figure_index': ('### 图册（报告按「图号 + 标题 + 文字结论」引用）', '### Figure Index (the report cites each figure as number + title + text conclusion)'),
    'heading_unit_layer': ('### 实验单位层（个体结构敏感性分析）', '### Experimental-Unit Layer (unit-structure sensitivity analysis)'),
    'heading_unit_layer_computed': ('### 实验单位层（个体结构敏感性分析，已计算）', '### Experimental-Unit Layer (unit-structure sensitivity analysis, computed)'),
    'heading_enrichment_hint': ('### 富集方向提示（探索性）', '### Enrichment Direction Hints (exploratory)'),
    'heading_enrichment_boundary': ('### 富集边界说明', '### Enrichment Boundary Note'),
    'heading_concordance_and_exclusion': ('### 对比一致性与候选排除', '### Contrast Concordance and Candidate Exclusion'),
    'heading_contrast_evidence': ('### 对比级证据摘要', '### Contrast-Level Evidence Summary'),
    'heading_data_preprocessing': ('### 数据与预处理', '### Data and Preprocessing'),
    'heading_not_computed': ('### 本轮未计算的证据（不得写成已有结果）', '### Evidence Not Computed in This Run (must not be written as an existing result)'),
    'heading_core_contrasts': ('### 核心对比（task_id 与本节一致）', '### Core Contrasts (task_id matches this section)'),
    'heading_module_scores': ('### 模块分组分数', '### Module Group Scores'),
    'heading_ora_background': ('### 正式富集检验（超几何，背景=本次检测到的蛋白）', '### Formal Enrichment Test (hypergeometric, background = proteins detected in this run)'),
    'heading_overlap_plain': ('### 预定义基因集的重叠富集', '### Overlap Enrichment of Predefined Gene Sets'),
    'heading_overlap_exploratory': ('### 预定义基因集的重叠富集（探索性，无 FDR/q）', '### Overlap Enrichment of Predefined Gene Sets (exploratory, no FDR/q)'),
    'range_two_values': ('%.4f 至 %.4f', '%.4f to %.4f'),
    'samples_two_arms': ('%s / %s（两臂样本数；按运行记录中第一个对比）', '%s / %s (sample counts of the two arms; first contrast in the run record)'),
    'figure_bar_conclusion': ('%s 个对比共 %s 个蛋白通过筛选（上调 %s、下调 %s，阈值 %s）。%s', '%s contrasts with %s proteins passing screening in total (upregulated %s, downregulated %s, threshold %s). %s'),
    'proteins_per_contrast': ('%s 个蛋白（每个对比）', '%s proteins (per contrast)'),
    'unparsable_diff_table': ('%s 的差异表不可解析（%s），筛选与计数无法核对', 'the differential table of %s cannot be parsed (%s), so the screening rule and the counts cannot be checked'),
    'range_generic': ('%s 至 %s', '%s to %s'),
    'title_suffix_analysis_report': ('%s分析报告', '%s Analysis Report'),
    'title_suffix_report': ('%s报告', '%s Report'),
    'title_core_join': ('%s的%s%s', '%s %s%s'),
    'fallback_contrast_label': ('%s（%s 较高 / %s 较高）', '%s (%s higher / %s higher)'),
    'quote_note': ('%s（相关原文：%s）', '%s (source text: %s)'),
    'n_significant_split': ('%s（较高 %s / 较低 %s）', '%s (%s higher / %s lower)'),
    'matrix_stage_shape': ('%s：%d 蛋白 × %d 样本', '%s: %d proteins × %d samples'),
    'fallback_contrast_item': ('%s：该对比显著集 %s 个蛋白，兜底查询集 n=%s', '%s: this contrast holds %s proteins in its significant set and a fallback query set of n=%s'),
    'figure_line_parse_re': ('(图\\d+)\\s*([^：:]*)', '(Figure\\s*\\d+)\\s*([^:：]*)'),
    'unit_sensitivity_completed': ('**实验单位敏感性分析**：以「%s」为分析单位（%s；估计目标：%s%s）。本对比%s；进入检验的蛋白组 %s 个%s，按 FDR 口径 %s 条通过，%s，最小校正 P 值 %s。', '**Experimental-unit sensitivity analysis**: the analysis unit is "%s" (%s; estimand: %s%s). For this contrast, %s; %s protein groups entered testing%s, %s passed under the FDR criterion, %s, smallest adjusted P %s.'),
    'unit_sensitivity_blocked': ('**实验单位敏感性分析**：本对比不做个体级检验（%s）：%s', '**Experimental-unit sensitivity analysis**: no unit-level test is run for this contrast (%s): %s'),
    'task_id_line': ('- %s：对比 %s；正值表示 %s 较高；源差异表 %s%s', '- %s: contrast %s; a positive value means %s is higher; source differential table %s%s'),
    'dose_module_cap': ('- 下表列出模块级前 %d 行；模块级共 %s 行（完整结果见本轮工件的 dose_trend.csv）。', "- The table below lists the first %d module-level rows; there are %s module-level rows in all (full results in this run's dose_trend.csv)."),
    'ora_best_cap': ('- 下表按 q(BH) 升序列出前 %d 条；被检验条目共 %s 条，其余条目未列出（未列出不等于没有结果）。', '- The table below lists the first %d rows in ascending q(BH); %s entries were tested and the rest are not listed (not listed does not mean there is no result).'),
    'query_scope_warning': ('- 不得把某一对比的显著蛋白数写成整个数据集的数字；两类查询集的 q 值不可直接比较。', '- The number of significant proteins of one contrast must not be written as a figure for the whole dataset; the q values of the two query-set classes are not directly comparable.'),
    'boundary_matrix_only': ('- 以上结论只对当前分析矩阵成立；模块分数不替代单蛋白 FDR 或正式富集检验。', '- The conclusions above hold only for the current analysis matrix; module scores do not replace per-protein FDR or a formal enrichment test.'),
    'exclusion_trace_line': ('- 候选排除轨迹：%s；被记录为未检出的候选：%s', '- Candidate exclusion trace: %s; candidates recorded as not detected: %s'),
    'asset_candidate_records': ('- 候选蛋白记录：%d 条，逐条数值见「候选蛋白逐对比统计」表。', '- Candidate protein records: %d; per-record values are in the table "Candidate Proteins by Contrast".'),
    'data_group_cols': ('- 分组列：%s；样本 ID 列：%s；批次列：%s', '- Group column: %s; sample ID column: %s; batch column: %s'),
    'more_hints': ('- 另有 %d 条同类提示未在正文展开。', '- A further %d hints of the same kind are not expanded in the body.'),
    'figure_line': ('- 图%d %s：%s', '- Figure %d %s: %s'),
    'volcano_range_one': ('- 图%d 逐对比火山图（阈值 %s）：', '- Figure %d per-contrast volcano plot (threshold %s):'),
    'volcano_range_many': ('- 图%d–%d 逐对比火山图（阈值 %s）：', '- Figures %d–%d per-contrast volcano plots (threshold %s):'),
    'figure_caption_count': ('- 图注 %s 条%s；visualize_results 下 PNG %s 个%s：%s', '- %s figure captions%s; %s PNG files under visualize_results%s: %s'),
    'figure_number_provenance': ('- 图题中的数字由本运行的报告流水线从本运行工件计算或读取（其中 PCA 解释率来自对 processed_proteins/ProteinQuant_ComBat.csv 的复算：列均值中心化、缺失以蛋白均值填补）。与源表数值冲突时以源表为准，冲突处须标注需核验，不得直接写成已验证事实。', "- The numbers in the figure titles are computed or read from this run's artifacts by this run's report pipeline (the PCA explained variance comes from a recomputation on processed_proteins/ProteinQuant_ComBat.csv: column-mean centring, missing values filled with the protein mean). Where a number conflicts with a source table the source table wins, and the conflict must be flagged as needing verification instead of being written as an established fact."),
    'boundary_external_annotation': ('- 外部注释只作背景说明，不作为当前矩阵的直接证据。', '- External annotation is background only and is not direct evidence from the current matrix.'),
    'concordance_line': ('- 对比一致性：%s（%s 对比较）', '- Contrast concordance: %s (%s pairs)'),
    'asset_contrast_records': ('- 对比级证据：%d 条，逐条数值见「对比级证据摘要」表。', '- Contrast-level evidence: %d records; per-record values are in the table "Contrast-Level Evidence Summary".'),
    'data_threshold': ('- 差异分析阈值：%s', '- Differential-analysis threshold: %s'),
    'units_excluded_bullet': ('- 已单列并排除在独立个体之外的实验单位：%s（按标签语义事先判定，不按检验结果挑选）。', '- Experimental units listed separately and excluded from independent individuals: %s (decided in advance from label semantics, not selected by test result).'),
    'unit_citation_rules': ('- 引用规则：观察级与单位层各自的通过数只能写进各自小节；单位层不显著不等于「没有差异」，不可估计的对比只能写「无法在本设计下识别该估计目标」。', '- Citation rule: the counts of the observation level and of the unit level each belong only in their own section; a unit-level result that is not significant does not mean "no difference", and an inestimable contrast may only be written as "this estimand cannot be identified under this design".'),
    'unit_effect_column_note': ('- 效应量列：%s（尺度：%s）；每个已检验蛋白的效应量、标准误与 95%% 置信区间在该对比的单位层表中逐行给出，未通过最小样本条件的行标为未检验且不带 P 值。', '- Effect column: %s (scale: %s); for every tested protein the effect size, standard error and 95%% confidence interval are given row by row in the unit-level table of that contrast, and rows below the minimum-sample condition are marked as not tested and carry no P value.'),
    'blocked_contrast_line': ('- 未做个体级检验的对比：%s —— %s', '- Contrasts with no unit-level test: %s - %s'),
    'boundary_enrichment_states': ('- 未提供富集结果与未执行富集分析是两种不同状态，报告按实际情况分别写明。', '- Enrichment results not provided and enrichment analysis not run are two different states; the report states which one applies.'),
    'core_contrast_count': ('- 本轮共 %d 个核心对比，证据文件见文末清单。', '- This run has %d core contrasts; the evidence files are listed at the end of the report.'),
    'run_contrasts': ('- 本轮对比：%s', '- Contrasts in this run: %s'),
    'query_definition_line': ('- 查询集定义：%s', '- Query-set definition: %s'),
    'query_classes_intro': ('- 查询集按对比分两类，引用 q 值时必须写明属于哪一类：', '- The query set falls into two classes by contrast; when citing a q value, state which class it belongs to:'),
    'tested_matrix_and_missing': ('- 检验所用矩阵与缺失口径：统计检验在%s上完成；原始交付矩阵的平均缺失率 %s，本轮分析矩阵 %s%s。', "- Matrix used for testing and missing-value scope: statistical testing was carried out on %s; the mean missing rate of the original delivered matrix is %s and that of this run's analysis matrix is %s%s."),
    'tested_matrix_plain': ('- 检验所用矩阵：%s；缺失口径%s（运行记录中没有分阶段缺失统计）。', '- Matrix used for testing: %s; missing-value scope %s (the run record holds no per-stage missing-value statistics).'),
    'dose_summary_line': ('- 模块级 %s 行、蛋白级 %s 行；蛋白级在 q≤0.05 下呈单调趋势 %s 个；最小 q=%s', '- %s module-level rows and %s protein-level rows; %s proteins show a monotonic trend at q≤0.05; smallest q=%s'),
    'dose_module_significant': ('- 模块级通过 q≤0.05 的条目：%s 条（已计算；0 表示已计算且无模块级显著结果）', '- Module-level entries passing q≤0.05: %s (computed; 0 means the test ran and no module-level result was significant)'),
    'asset_module_records': ('- 模块记录：%d 条，逐条数值见「模块分组分数」表。', '- Module records: %d; per-record values are in the table "Module Group Scores".'),
    'data_matrix_state': ('- 矩阵状态：%s；效应尺度：%s（%s）', '- Matrix state: %s; effect scale: %s (%s)'),
    'boundary_offline_enrichment': ('- 离线富集结果只在 FDR 显著时作为显著富集，其余只能作为探索性提示。', '- Offline enrichment results count as significant enrichment only when FDR-significant; otherwise they are exploratory hints only.'),
    'method_summary_pointer': ('- 统计模型、多重校正、筛选条件与各参数的记录位置见「运行参数与证据绑定」；本节只给摘要，完整声明集中在该小节，此处不重复。', '- The statistical model, multiple-testing method, screening conditions and each parameter are recorded in "Run Parameters and Evidence Binding"; this section gives the summary only, so the full statement is not repeated here.'),
    'ora_background_sizes': ('- 背景集大小 %s；统计检验所用矩阵蛋白数 %s（两个数字口径不同，引用时写明指哪一个）', '- Background-set size %s; protein count of the matrix used for testing %s (the two figures have different scopes; state which one is meant when citing)'),
    'boundary_table_values': ('- 表内数值来自本轮分析的确定性表格；正文解释与表格同源，任何数值以表格为准。', "- Values in the tables come from this run's deterministic tables; the explanatory text and the tables share one source, and the tables decide any value."),
    'ora_tested_counts': ('- 被检验条目 %s 条，未检验 %s 条', '- %s entries tested, %s not tested'),
    'data_evidence_paths': ('- 证据表与图件路径见文末清单；数值以确定性表格为准。', '- Evidence tables and figure paths are listed at the end of the report; the deterministic tables decide any value.'),
    'ora_significant_count': ('- 通过 q(BH)≤0.05 的条目：%s 条（已计算；0 表示检验已执行但没有条目通过，不得写成未执行）', '- Entries passing q(BH)≤0.05: %s (computed; 0 means the test ran and no entry passed, which must not be written as not run)'),
    'field_n_a_key': ('A 组样本数=', 'Group A sample count='),
    'field_n_up_a': ('A 组较高数', 'count higher in Group A'),
    'field_n_b_key': ('B 组样本数=', 'Group B sample count='),
    'field_n_down_a': ('B 组较高数', 'count higher in Group B'),
    'fdr_significant': ('FDR 显著', 'FDR significant'),
    'figure_pca_conclusion': ('PC1 解释 %.1f%%、PC2 解释 %.1f%% 的方差；PC1 分组均值从最低的 %s（%.2f）到最高的 %s（%.2f），说明分组在 PC1 方向存在整体位移。', 'PC1 explains %.1f%% and PC2 explains %.1f%% of the variance; the PC1 group mean runs from the lowest %s (%.2f) to the highest %s (%.2f), showing a systematic shift of the groups along PC1.'),
    'figure_qc_title': ('PCA 样本结构', 'Sample and missing-rate QC'),
    'figure_umap_title': ('UMAP 样本结构', 'UMAP sample structure'),
    'threshold_text': ('adj.P<%s 且 \\|logFC\\|>%s（多重校正：%s）', 'adj.P<%s and \\|logFC\\|>%s (multiple testing: %s)'),
    'figure_threshold': ('adj.P<0.05 且 |logFC|>0.25（BH 校正）', 'adj.P<0.05 and |logFC|>0.25 (BH correction)'),
    'threshold_design': ('adj.P≤%s 且 |logFC|≥%s', 'adjusted P≤%s and |effect size|≥%s'),
    'scale_log2_inferred': ('log2（推断，未记录变换）', 'log2 (inferred; no transform recorded)'),
    'note_needs_log_transform': ('parameters.json 的 needs_log_transform 是对输入的推断，两者不是同一件事', 'needs_log_transform in parameters.json is an inference about the input, and the two are not the same thing'),
    'transform_record_derived_source': ('run_events.jsonl 的 combat_calibration 工具结果（派生，不是该步骤直接写出）', 'combat_calibration tool result in run_events.jsonl (derived, not written directly by that step)'),
    'header_task_id_contrast': ('| task_id | 对比 | 两臂 | 通过筛选 | 第一臂较高 | 第二臂较高 | 方向写法 |', '| task_id | Contrast | Two arms | Passed screening | Higher in arm 1 | Higher in arm 2 | Direction convention |'),
    'header_unit_layer': ('| task_id | 本轮任务 | 状态 | 分析单位 | 单位语义来源 | 估计目标 | A/ B/ 共同/ 仅A/ 仅B | 检验蛋白/未检验 | FDR口径通过 | 联合口径通过 | 最小校正P |', '| task_id | Task of this run | Status | Analysis unit | Unit-semantics source | Estimand | A/ B/ shared/ A only/ B only | Tested/not tested | Passed under FDR | Passed under the joint rule | Smallest adjusted P |'),
    'header_contrast_evidence': ('| 任务 | 对比 | 第一臂 | 第二臂 | 通过筛选蛋白数 | 第一臂中较高 | 第二臂中较高 | 证据来源 | 置信度 |', '| Task | Contrast | Arm 1 | Arm 2 | Proteins passing screening | Higher in arm 1 | Higher in arm 2 | Evidence source | Confidence |'),
    'header_candidate_stats': ('| 候选 | 匹配蛋白 | 对比 | 方向 | logFC | P 值 | 校正 P 值 | 缺失率 | 证据来源 |', '| Candidate | Matched protein | Contrast | Direction | logFC | P value | Adjusted P value | Missing rate | Evidence source |'),
    'header_candidate_block': ('| 候选 | 蛋白组 | 显示方向 %s | 源表 %s | adj.P.Val | 方向 | 差异表 |', '| Candidate | Protein group | Display direction %s | Source table %s | adj.P.Val | Direction | Differential table |'),
    'header_exclusion_trace': ('| 候选 | 记录原因 | 记录数 |', '| Candidate | Recorded reason | Records |'),
    'row_missing_rate_raw': ('| 原始交付矩阵的平均缺失率 | %s | %s |', '| Mean missing rate of the original delivered matrix | %s | %s |'),
    'row_transform_record_source': ('| 变换记录的来源 | %s | 元数据 |', '| Source of the transform record | %s | metadata |'),
    'row_fdr': ('| 多重校正 | %s | %s |', '| Multiple testing | %s | %s |'),
    'row_effective_threshold': ('| 实际生效阈值 | %s | %s |', '| Threshold actually in force | %s | %s |'),
    'row_log2_transform': ('| 对数变换（按本轮实际执行记录） | %s | %s |', "| Log transform (per this run's execution record) | %s | %s |"),
    'header_concordance': ('| 对比 A | 对比 B | 共享蛋白 | 效应相关 | 显著重叠 | Fisher p |', '| Contrast A | Contrast B | Shared proteins | Effect correlation | Significant overlap | Fisher p |'),
    'header_stratified': ('| 对比 | 分层变量 | 层 | 第一臂 n | 第二臂 n | 检验蛋白 | 通过筛选 | 结论 |', '| Contrast | Stratifier | Stratum | Arm 1 n | Arm 2 n | Proteins tested | Passed screening | Conclusion |'),
    'header_stratified_request': ('| 对比 | 层 | 第一臂 n | 第二臂 n | 检验蛋白 | 通过筛选 | 分层判定 | 备注 |', '| Contrast | Stratum | Arm 1 n | Arm 2 n | Proteins tested | Passed screening | Stratified verdict | Remark |'),
    'header_ora': ('| 对比 | 方向 | 库 | 通路 | 命中 | 通路在背景中 | 查询集 | p | q(BH) |', '| Contrast | Direction | Database | Pathway | Hits | Pathway in background | Query set | p | q(BH) |'),
    'header_overlap': ('| 对比 | 状态 | 条目数 | 文件中的对比键（臂序） | 各库条目数 |', '| Contrast | Status | Entries | Contrast key in the file (arm order) | Entries per database |'),
    'header_core_contrast': ('| 对比 | 通过筛选的蛋白数 | %s 较高 | %s 较高 | 统计口径 |', '| Contrast | Proteins passing screening | %s higher | %s higher | Statistical scope |'),
    'header_overlap_picks': ('| 对比内的条目 | 库 | 方向 | 通路/语义条目 | 该条种子基因数 | 代表成员 |', '| Entry within contrast | Database | Direction | Pathway/semantic term | Seed genes in that entry | Representative members |'),
    'header_ora_best': ('| 库 | 通路 | 命中 | 查询集 | p | q(BH) |', '| Database | Pathway | Hits | Query set | p | q(BH) |'),
    'header_enrichment_hint': ('| 数据库 | 方向 | 通路/语义条目 | 该条种子基因数 | 代表成员 |', '| Database | Direction | Pathway/semantic term | Seed genes in that entry | Representative members |'),
    'row_missing_rate_analysed': ('| 本轮分析矩阵的平均缺失率 | %s | %s |', "| Mean missing rate of this run's analysis matrix | %s | %s |"),
    'header_module_block': ('| 模块 | A 组均值 | B 组均值 | Δ（A 组均值 − B 组均值） | 匹配蛋白数 |', '| Module | Group A mean | Group B mean | Δ (Group A mean − Group B mean) | Matched proteins |'),
    'header_module_scores': ('| 模块 | 分组或差值 | 样本数 | 平均分 | 中位数 | 匹配基因数 | 成员基因（匹配） |', '| Module | Group or difference | Samples | Mean score | Median | Matched genes | Member genes (matched) |'),
    'row_samples_per_group': ('| 每组样本数 | %s | %s |', '| Samples per group | %s | %s |'),
    'row_matrix_size': ('| 统计检验所用矩阵规模 | %s 个蛋白 | %s |', '| Size of the matrix used for testing | %s proteins | %s |'),
    'row_statistical_model': ('| 统计模型 | %s | %s |', '| Statistical model | %s | %s |'),
    'row_imputation': ('| 缺失值填补（按本轮实际执行记录） | %s | %s |', "| Missing-value imputation (per this run's execution record) | %s | %s |"),
    'header_dose_module': ('| 药物 | 分层 | 模块 | 样本数 | ρ | p | q(BH) |', '| Drug | Stratum | Module | Samples | ρ | p | q(BH) |'),
    'header_dose_module_steps': ('| 药物 | 分层 | 模块 | 样本数 | 剂量步 | ρ | p | q(BH) | 方向 |', '| Drug | Stratum | Module | Samples | Dose steps | ρ | p | q(BH) | Direction |'),
    'header_dose_protein': ('| 蛋白 | 药物 | 分层 | 样本数 | ρ | p | q(BH) |', '| Protein | Drug | Stratum | Samples | ρ | p | q(BH) |'),
    'row_protein_filtering': ('| 蛋白过滤（按本轮实际执行记录） | %s | %s |', "| Protein filtering (per this run's execution record) | %s | %s |"),
    'header_asset_pointer': ('| 资产 | 在何处复核 |', '| Asset | Where to verify it |'),
    'row_matrix_stage': ('| 运行链路上的矩阵阶段 | %s | %s |', '| Matrix stages along the run chain | %s | %s |'),
    'row_dimred_coordinates': ('| 逐样本降维坐标 | %s | %s |', '| Per-sample dimensionality-reduction coordinates | %s | %s |'),
    'header_evidence_binding': ('| 项目 | 取值 | 记录位置 | 状态 |', '| Item | Value | Record location | Status |'),
    'header_request_item': ('| 项目 | 数值或状态 | 状态 |', '| Item | Value or status | Status |'),
    'enrichment_up_label': ('上调方向（%s 较高）', 'upregulated direction (%s higher)'),
    'query_scope_two_classes': ('下列 q 值只描述各自查询集的富集情况，不等同于差异蛋白的富集显著性；某一对比的显著蛋白数不适用于整个数据集。', 'The q values below describe the enrichment of each query set respectively and are not equivalent to the enrichment significance of the differential proteins; the number of significant proteins of one contrast does not apply to the whole dataset.'),
    'query_scope_single': ('下列 q 值描述的是该查询集的富集情况，不等同于差异蛋白的富集显著性。', 'The q values below describe the enrichment of that query set and are not equivalent to the enrichment significance of the differential proteins.'),
    'exclusion_trace_intro': ('下列候选在进入统计前被排除，原因按来源表逐条记录；未检出不等于样本中不存在。', 'Candidates listed below were excluded before entering the statistics, and the reason is recorded entry by entry from the source table; not detected does not mean absent from the sample.'),
    'evidence_binding_intro': ('下表逐项列出本节数值所依赖的运行参数及其记录位置；运行记录未写入的参数标记为「未记录」，不按报告措辞补写，也不选择更有利的读法。', 'The table below lists, item by item, the run parameters the values in this section depend on and where each one is recorded; a parameter the run record never wrote is marked "not recorded" and is neither filled in from the report wording nor read in whichever way flatters the text.'),
    'enrichment_down_label': ('下调方向（%s 较高）', 'downregulated direction (%s higher)'),
    'unit_status_blocked': ('不可估计，未做个体级检验', 'inestimable, no unit-level test performed'),
    'absence_note_excludes_presence_only': ('不含未进入统计检验的候选（这类候选只在附录单独列出）', 'and does not contain records that did not enter statistical testing (such candidates are listed separately in the appendix)'),
    'figure_umap_conclusion': ('与 PCA 同一输入的二维投影，用于交叉检查分组分离与离群样本。', 'A two-dimensional projection of the same input as the PCA, used to cross-check group separation and outlier samples.'),
    'note_imputation_versus_missing': ('与下一行的插补状态是两个不同问题：这里问统计检验如何对待缺失观测，那里问分析矩阵是否被填补', 'This is a different question from the imputation state in the next row: here the question is how statistical testing treats missing observations, there it is whether the analysis matrix was imputed.'),
    'arm_order_matches': ('与本对比臂序一致', 'the arm order matches this contrast'),
    'arm_order_reversed': ('与本对比臂序相反', 'the arm order is reversed relative to this contrast'),
    'field_pair_col': ('个体/实验单位列=%s', 'pair/experimental-unit column=%s'),
    'estimand_within_unit_mean_diff': ('个体内均值差（同一实验单位在两臂的均值差）', 'within-unit mean difference (difference of the means of the same experimental unit across the two arms)'),
    'estimand_within_unit_observed': ('个体内均值差（同一实验单位在两臂都有观测）', 'within-unit mean difference (the same experimental unit observed in both arms)'),
    'reason_matched_by_name_only': ('个体列仅按名称匹配，未在分析设计中声明', 'the unit column is matched by name only and is not declared in the analysis design'),
    'item_unit_sensitivity': ('个体结构敏感性分析', 'individual-structure sensitivity analysis'),
    'estimand_between_unit_pooled': ('个体间均值差（每条臂只使用只在该臂观测的实验单位）', 'between-unit mean difference (each arm uses only the experimental units observed in that arm)'),
    'estimand_between_unit_independent': ('个体间均值差（每条臂只使用只在该臂观测的独立个体）', 'between-unit mean difference (each arm uses only the independent individuals observed in that arm)'),
    'confidence_moderate': ('中', 'moderate'),
    'moderate_slash_low': ('中等/较低', 'moderate/low'),
    'stage_delivered_matrix': ('交付矩阵（未经步骤命名）', 'delivered matrix (no step name)'),
    'scale_log2_range_only': ('仅按数值范围推断，未记录变换', 'log2 (inferred; no transform recorded)'),
    'scale_log2_range_only_long': ('仅按数值范围推断；运行记录只记录了输入的判断字段，未记录实际变换', 'inferred from the value range only; the run record holds only the input judgement fields and nothing about the transform actually applied'),
    'figure_group_means_conclusion': ('代表蛋白组的分组均值；逐条数值与差值见「模块分组分数」表，模块分数无 FDR，只作方向性证据。', 'Group means of representative protein groups; per-record values and differences are in the table "Module Group Scores", and module scores carry no FDR, so they are directional evidence only.'),
    'unit_layer_digest': ('以%s为分析单位（来源：%s）：%d 个对比中 %d 个可做个体级检验（两臂共同单位 %s 个）；按 FDR 口径合计 %d 条通过，按联合口径（FDR 且 |效应| > %s）合计 %d 条通过；%d 个蛋白行因有效单位不足标为未检验。', 'using %s as the analysis unit (source: %s): of %d contrasts, %d support a unit-level test (%s shared units across the two arms); %d entries passed under the FDR criterion and %d passed under the joint criterion (FDR and |effect| > %s); %d protein rows are marked as not tested for lack of valid units.'),
    'file_scoring_coverage': ('任务覆盖检查表', 'task coverage checklist'),
    'confidence_low': ('低', 'low'),
    'unit_design_paired': ('使用两臂都有观测的 %s 个独立单位做配对 t 检验；只在一个臂出现的单位不参与配对', 'paired t-test on the %s independent units observed in both arms; units appearing in only one arm do not enter the pairing'),
    'asset_candidate_exclusion_trace': ('候选排除轨迹', 'candidate exclusion trace'),
    'figure_bubble_conclusion': ('候选蛋白在各对比中的方向与效应量；逐条数值见「候选蛋白逐对比统计」表。', 'Direction and effect size of the candidate proteins across contrasts; per-record values are in the table "Candidate Proteins by Contrast".'),
    'figure_bubble_title': ('候选蛋白对比气泡图', 'Candidate protein contrast bubble plot'),
    'asset_candidate_index': ('候选蛋白索引共 %d 个「对比 × 蛋白组」组合：%d 个已在上文对应任务中列出%s', 'Candidate protein index covers %d "contrast x protein group" combinations: %d are listed in the corresponding task above%s'),
    'file_candidate_protein': ('候选蛋白表', 'candidate protein table'),
    'candidate_caption_all': ('候选蛋白（共 %d 条）', 'Candidate proteins (%d in total)'),
    'candidate_caption_capped': ('候选蛋白（共 %d 条，本节列前 %d 条，完整清单见「证据表（逐条）」）', 'Candidate proteins (%d in total; this section lists the first %d, and the full list is in "Evidence Tables (record by record)")'),
    'fallback_query_scope': ('兜底查询集用于该对比显著集过小的情形：%s；本节所列兜底条目查询集 n=%s。', 'The fallback query set covers contrasts whose significant set is too small: %s; the fallback entries listed in this section have a query set of n=%s.'),
    'kind_fallback_set': ('兜底集（该对比显著集过小时按 |logFC| 取前 10%）', "fallback set (the first 10% by |logFC| when a contrast's significant set is too small)"),
    'unit_design_between': ('共享单位不足以配对，改用只在单臂观测的独立个体（A 臂 %s 个、B 臂 %s 个）做个体间 Welch 检验，共享单位被排除在外、不重复计数', 'the shared units are too few to pair, so independent individuals observed in only one arm (arm A %s, arm B %s) are used in a between-unit Welch test, with shared units excluded and never counted twice'),
    'figure_key_protein_title': ('关键蛋白总览', 'Key protein overview'),
    'merged_flipped_note': ('其中 %s 的臂序与本节坐标相反，其句内正负号与本节表格相反；引用这些句子的数值时以本节表格的方向为准。', 'the arm order of %s is opposite to the coordinates of this section, so the signs inside those sentences are opposite to the tables of this section; when citing numbers from those sentences, follow the direction of the tables in this section.'),
    'backend_internal_fallback': ('内置回退实现（运行环境缺少 R/rpy2，标准流程未启用）', 'built-in fallback implementation (this run environment lacks R/rpy2, so the standard pipeline was not enabled)'),
    'asset_stratified_summary': ('分层对比摘要表', 'stratified contrast summary table'),
    'not_computed_stratified': ('分层对比摘要（源表没有带分层的行）', 'stratified contrast summary (the source table has no stratified rows)'),
    'file_stratified_summary': ('分层对比汇总', 'stratified contrast summary'),
    'stratified_scope_note': ('分层结果按单个分层变量分别分组，不等于同时控制多个变量；主结论仍以主模型为准。', 'Stratified results group by one stratifier at a time and are not equivalent to controlling several variables at once; the main conclusion still follows the main model.'),
    'stratified_scope_note_short': ('分层结果按单个变量分别分组，不等于同时控制多个变量；主结论仍以主模型为准。', 'Stratified results are grouped by one variable at a time and are not equivalent to controlling several variables at once; the main conclusion still follows the main model.'),
    'field_analysed_missing': ('分析后缺失率=', 'analysed missing rate='),
    'field_analysed_missing_rate': ('分析矩阵平均缺失率', 'mean missing rate of the analysis matrix'),
    'transform_basis': ('分析矩阵最大值 %s，过滤后原始矩阵最大值 %s（log2(x+1) 后的上界约 %s）', 'analysis matrix maximum %s, filtered raw matrix maximum %s (upper bound after log2(x+1) about %s)'),
    'note_imputation_flag_scope': ('分析矩阵缺失率为 0 时按此字段判读，不等于全部蛋白被检出', 'When the missing rate of the analysis matrix is 0, this field decides how to read it, and that does not mean every protein was detected'),
    'field_analysed_proteins': ('分析蛋白数=', 'proteins analysed='),
    'item_contrasts_in_design': ('分析设计中的对比（%d 个）', 'contrasts in the analysis design (%d)'),
    'unit_source_design_declared': ('分析设计声明的实验单位列', 'the experimental-unit column declared in the analysis design'),
    'file_analysis_design': ('分析设计记录', 'analysis design record'),
    'file_group_cross_table': ('分组交叉表', 'group cross table'),
    'field_group_col': ('分组列=%s', 'group column=%s'),
    'file_group_composition_qc': ('分组构成与质量控制表', 'group composition and quality-control table'),
    'source_qc_analysed_missing': ('分组组成与质控表 → analysed_mean_missing_rate', 'group composition and QC table -> analysed_mean_missing_rate'),
    'source_qc_imputation': ('分组组成与质控表 → imputation_applied', 'group composition and QC table -> imputation_applied'),
    'source_qc_input_missing': ('分组组成与质控表 → input_mean_missing_rate', 'group composition and QC table -> input_mean_missing_rate'),
    'verdict_line': ('判定：%s 中通过筛选口径的蛋白 %s 个（口径：%s），其中 %s 较高 %s 个、%s 较高 %s 个。', 'Verdict: %s has %s proteins passing the screening scope (scope: %s), of which %s are higher in %s and %s are higher in %s.'),
    'dose_scope_note': ('剂量趋势以样本为观测单位（对照、低剂量、高剂量三步，分池与单细胞分层各自检验）；模块分数只作方向性佐证，不能替代蛋白水平的趋势检验。', 'The dose trend uses samples as the observation unit (control, low dose and high dose; the pooled and the single-cell strata are each tested separately); module scores are directional corroboration only and do not replace a protein-level trend test.'),
    'not_computed_dose_trend': ('剂量趋势检验（Spearman）', 'dose-trend test (Spearman)'),
    'translate_raw_p': ('原始 P<0.05 = ', 'raw P<0.05 =  '),
    'item_missing_rate_raw_by_group': ('原始交付矩阵的平均缺失率（分组）', 'mean missing rate of the original delivered matrix (by group)'),
    'field_input_mean_missing': ('原始矩阵平均缺失率', 'mean missing rate of the raw matrix'),
    'figure_heatmap_conclusion': ('取信息量最高的前 %s 个特征（阈值 %s）展示样本间分布。', 'The %s most informative features (threshold %s) are shown to display the distribution across samples.'),
    'file_confounding_estimability': ('可估计性检查', 'estimability check'),
    'translate_log2_likely': ('可能为 log2 尺度（未记录变换）', 'possibly log2 scale (no transform recorded)'),
    'figure_key_protein_conclusion': ('各对比通过筛选的蛋白及其效应量；逐条数值见「对比级证据摘要」表。', 'Proteins passing screening in each contrast together with their effect sizes; per-record values are in the table "Contrast-Level Evidence Summary".'),
    'merged_note_aligned': ('各小节的对比标识臂序与本节坐标一致。', 'the contrast labels of each subsection state the same arm order as the coordinates of this section.'),
    'field_n_per_group': ('各组样本数', 'samples per group'),
    'no_background_contrast': ('否（背景对比）', 'no (background contrast)'),
    'ora_fallback_reason_count': ('因本数据集通过显著阈值的蛋白仅 %s 个、不足以做标准的过表征分析。', 'because this dataset has only %s proteins passing the significance threshold, too few for a standard over-representation analysis.'),
    'ora_fallback_reason_too_few': ('因本数据集通过显著阈值的蛋白过少，不足以做标准的过表征分析。', 'because this dataset has too few proteins passing the significance threshold for a standard over-representation analysis.'),
    'figures_narrative_note': ('图件以「图号 + 标题 + 文字结论」在正文中给出，图不可见时信息仍可读。', 'Figures are given in the body as "number + title + text conclusion", so the information stays readable when a figure cannot be displayed.'),
    'not_computed_figures': ('图册与图注（visualize_results 未产出可引用图件）', 'figure index and captions (visualize_results produced no citable figure)'),
    'file_figure_index': ('图册索引', 'figure index'),
    'translate_down_in_group_a': ('在 A 组中较低', 'lower in Group A'),
    'translate_up_in_group_a': ('在 A 组中较高', 'higher in Group A'),
    'translate_down_in_capture': ('在\\1中较低', 'lower in \\1'),
    'translate_up_in_capture': ('在\\1中较高', 'higher in \\1'),
    'translate_zero_p': ('在原差异表中记为 0', 'recorded as 0 in the source differential table'),
    'translate_batch_corrected_claimed': ('声称已批次校正', 'reported as batch-corrected'),
    'field_effective_threshold': ('实际生效阈值', 'threshold actually in force'),
    'enrichment_hint_heading': ('富集方向提示', 'Enrichment Direction Hints'),
    'translate_log_transform_form_fallback': ('对数变换', 'logarithmic transform'),
    'field_contrast_vif': ('对比 VIF=', 'contrast VIF='),
    'asset_contrast_concordance': ('对比一致性表', 'contrast concordance table'),
    'not_computed_concordance': ('对比一致性（源分析未产出可比较的对比对）', 'contrast concordance (the source analysis produced no comparable contrast pairs)'),
    'field_contrast_estimable': ('对比可估计性=', 'contrast estimable='),
    'field_contrasts': ('对比清单', 'contrast list'),
    'prob_less_than': ('小于 1e-300', 'less than 1e-300'),
    'item_scale_and_transform': ('尺度与变换（记录字段）', 'scale and transform (recorded fields)'),
    'field_scale_flags': ('尺度判断字段', 'scale judgement fields'),
    'keyword_limitation': ('局限', 'limitation'),
    'record_limma': ('差异分析记录', 'differential-analysis record'),
    'record_limma_dot': ('差异分析记录 · ', 'differential-analysis record ·  '),
    'record_limma_sanity': ('差异分析记录 · 记录字段 · ', 'differential-analysis record · recorded fields ·  '),
    'file_core_story': ('差异检验与故事线摘要', 'differential testing and story-line summary'),
    'file_differential': ('差异检验摘要', 'differential test summary'),
    'asset_differential': ('差异检验结果表（%s）', 'differential test result table (%s)'),
    'figure_differential_bar_title': ('差异蛋白数量概览', 'Overview of differential protein counts'),
    'figure_heatmap_title': ('差异蛋白热图', 'Differential protein heatmap'),
    'translate_already_processed': ('已处理输入', 'already-processed input'),
    'unit_status_completed': ('已完成个体级敏感性分析', 'unit-level sensitivity analysis completed'),
    'executed_parens_two': ('已执行（%s%s）', 'executed (%s%s)'),
    'executed_parens_one': ('已执行（%s）', 'executed (%s)'),
    'imputation_conflict_note': ('已执行（%s）；分组组成与质控表中的 imputation_applied=%s 与执行记录不同，以执行记录为准并登记该冲突', 'executed (%s); imputation_applied=%s in the group composition and QC table differs from the execution record, so the execution record decides here and the conflict is logged'),
    'executed_colon': ('已执行：%s', 'executed: %s'),
    'executed_colon_two': ('已执行：%s%s', 'executed: %s%s'),
    'unit_all_blocked': ('已检查实验单位结构，%d 个对比均不满足个体级检验条件。', 'the experimental-unit structure was checked, and none of the %d contrasts meets the conditions for a unit-level test.'),
    'scale_note_log2_recorded': ('已记录 log2 变换', 'log2 transform recorded'),
    'figure_qc_conclusion': ('平均每样本检出 %s 个蛋白，未注释 %s 个；分组规模与缺失率见「数据与预处理」。', 'an average of %s proteins detected per sample, %s of them unannotated; group sizes and missing rates are in "Data and Preprocessing".'),
    'absence_note_includes_presence_only': ('并包含未进入统计检验的记录', 'and contains records that did not enter statistical testing'),
    'file_normalized_matrix': ('归一化定量矩阵', 'normalized quantitative matrix'),
    'matrix_alias_total_abundance': ('总丰度矩阵', 'total abundance matrix'),
    'keyword_summary': ('总结', 'summary'),
    'field_eta2_batch_pc': ('批次主成分效应量=', 'batch principal-component effect size='),
    'field_batch_col': ('批次列=%s', 'batch column=%s'),
    'file_combat_matrix': ('批次标定后定量矩阵', 'batch-calibrated quantitative matrix'),
    'stage_combat_matrix': ('批次标定后矩阵', 'batch-calibrated matrix'),
    'overlap_pick_rule': ('报告将按同一规则各取一条（每个数据库每个方向种子基因数最多的一条）：', 'The report takes one entry per rule (the entry with the most seed genes for each database and direction):'),
    'translate_display_direction': ('报告方向', 'report direction'),
    'request_direction_by': ('按 %s', 'per %s'),
    'request_direction_by_inverted': ('按 %s（源差异表 %s 符号相反）', 'per %s (the source differential table %s has the opposite sign)'),
    'unit_source_column_name_unconfirmed': ('按列名匹配推断，未在分析设计中确认', 'inferred by column-name matching and not confirmed in the analysis design'),
    'finding_joint_mismatch_detailed': ('按报告自己声明的联合条件（%s 连接，列 %s / %s）在差异表上复算得到 %d 行，与本对比汇总的 %d 不一致；其中仅满足校正 P 的有 %d 行、仅满足效应阈值的有 %d 行、最小校正 P=%s', 'Recounting the joint condition this report declares (%s connector, columns %s / %s) on the differential table gives %d rows, which disagrees with the %d of this contrast summary; of those, %d rows satisfy only the adjusted P and %d satisfy only the effect threshold, and the smallest adjusted P is %s'),
    'scale_note_from_execution_record': ('按本轮执行记录：%s（依据：%s；记录来源：%s）', "per this run's execution record: %s (basis: %s; record source: %s)"),
    'unit_joint_pass': ('按联合口径（校正 P 值 < 0.05 且 |效应量| > %s）%s 条通过', 'under the joint scope (adjusted P < 0.05 and |effect size| > %s), %s entries passed'),
    'exclusion_count_note': ('排除记录数统计的是记录条数，不是蛋白数量。', 'The exclusion count counts records, not proteins.'),
    'enrichment_hint_note_short': ('探索性方向提示：集合重叠结果，无 FDR/q，不作显著性证据（口径见「结论边界与方法局限」）。各节按同一规则各取一条。', 'Exploratory direction hints: set-overlap results with no FDR/q, not significance evidence (see "Conclusion Boundaries and Method Limitations"). Each section takes one entry per rule.'),
    'translate_exploratory': ('探索性，未达到 FDR 显著', 'exploratory, not FDR-significant'),
    'translate_inferred': ('推断', 'inferred'),
    'keyword_inference': ('推测', 'inference'),
    'checks_not_a_metric': ('提示条数不得用作性能计数、验收总分或改进指标；只用于定位。', 'The number of hints must not be used as a performance count, an acceptance total or an improvement metric; it serves to locate items only.'),
    'checks_mode': ('提示（不阻断，不改变结论）', 'hint (non-blocking, never changes a conclusion)'),
    'effect_column_unconfirmed': ('效应量（尺度未确认）', 'effect size (scale not confirmed)'),
    'field_numeric_range': ('数值范围', 'value range'),
    'field_looks_logged': ('数值范围形似对数尺度=%s', 'value range resembles a log scale=%s'),
    'file_dataset_recipe': ('数据集分析配置', 'dataset analysis configuration'),
    'overlap_state_unavailable': ('文件不可用', 'file unavailable'),
    'overlap_state_empty': ('文件存在但是空对象（已执行且无条目）', 'the file exists but is an empty object (the analysis ran and produced no entry)'),
    'overlap_state_unparsable': ('文件存在但解析失败（读取状态，不是未产出）', 'the file exists but does not parse (a read state, not an absent result)'),
    'finding_direction_mismatch': ('方向句写出的较高/较低组与差异表记录的方向不同：表中记为 %s', 'the direction sentence names a different higher/lower group than the differential table records: the table records %s'),
    'candidate_direction_short': ('方向读法同前：显示方向列按「%s」，源表列为反向写法「%s」。', 'Direction convention as above: the display-direction column follows "%s" and the source-table column follows the reverse writing "%s".'),
    'candidate_direction_long': ('方向读法：显示方向列按「%s」（正值为 %s 较高），源表列按反向写法「%s」（正值为 %s 较高）；两列数值符号相反，但指的是同一个对比。', 'Direction convention: the display-direction column follows "%s" (a positive value means %s is higher) and the source-table column follows the reverse writing "%s" (a positive value means %s is higher); the two columns have opposite signs but describe the same contrast.'),
    'dose_flat_zero_rho': ('无单调趋势（ρ=0）', 'no monotonic trend (rho=0)'),
    'estimand_none': ('无可识别的个体内或个体间估计目标', 'no identifiable within-unit or between-unit estimand'),
    'enrichment_status_no_entry': ('无本对比条目', 'no entry for this contrast'),
    'status_unusable': ('无法使用', 'unusable'),
    'exclusion_none': ('无（记录项为分组与实验设计术语，不是蛋白候选）', 'none (the recorded items are grouping and experimental-design terms, not protein candidates)'),
    'yes_value': ('是', 'yes'),
    'item_imputation_executed': ('是否执行插补', 'was imputation performed'),
    'field_imputation_applied': ('是否插补', 'imputation applied'),
    'translate_display_lower': ('显示方向较低', 'display direction lower'),
    'translate_display_higher': ('显示方向较高', 'display direction higher'),
    'enrichment_status_valid_empty_contrast': ('有效空结果（该对比无条目）', 'valid empty result (no entry for this contrast)'),
    'unit_blocked_reasons': ('未做个体级检验的原因：%s。', 'reason why no unit-level test was run: %s.'),
    'pca_ungrouped': ('未分组', 'ungrouped'),
    'unbound_section_default': ('未命名小节', 'unnamed section'),
    'reason_no_unit_column': ('未声明实验单位列，元数据中也没有可识别的个体列', 'no experimental-unit column is declared and the metadata holds no identifiable individual column'),
    'unit_no_joint_threshold': ('未套用联合效应阈值', 'no joint effect threshold applied'),
    'translate_symbol_not_mapped': ('未建立符号映射', 'symbol mapping not established'),
    'evidence_symbol_not_mapped': ('未建立符号映射（不等同于未检出）', 'symbol mapping not established (which is not the same as not detected)'),
    'translate_symbol_not_matched': ('未建立符号映射（该符号未匹配到矩阵标识，不等同于未检出）', 'symbol mapping not established (the symbol matched no matrix identifier, which is not the same as not detected)'),
    'scale_note_unknown': ('未记录尺度', 'no scale recorded'),
    'not_recorded_rule': ('未记录（记录中的规则：%s）', 'not recorded (the rule in the record: %s)'),
    'ledger_dropped': ('未进入成品（需人工确认）', 'did not reach the finished report (needs manual confirmation)'),
    'ora_mixed_scope_lead': ('本小节所列条目来自两类查询集，须按对比分别理解。第一类使用显著集（%s），本节所列条目的查询集 n=%s。', 'The entries listed in this section come from two classes of query set and must be read per contrast. The first class uses the significant set (%s), and the entries listed here have a query set of n=%s.'),
    'ora_mixed_scope_all_fallback': ('本小节的查询集不是一个统一口径，须按对比分别理解：本节所列条目均来自兜底查询集。', 'The query set of this section is not a single scope and must be read per contrast: every entry listed here comes from the fallback query set.'),
    'ora_scope_significant': ('本小节的查询集为各对比中通过显著阈值的蛋白（%s）：%s。', 'The query set of this section is the set of proteins passing the significance threshold in each contrast (%s): %s.'),
    'ora_scope_fallback_lead': ('本小节的查询集为按 |logFC| 取前 10%% 的蛋白（本小节所列条目的查询集 n=%s，全部被检验的查询集 n=%s），', 'The query set of this section is the top 10%% of proteins by |logFC| (the entries listed here have a query set of n=%s and the full set of tested entries has n=%s),'),
    'enrichment_boundary_empty': ('本数据集在当前富集配置下未产出可用条目（预定义基因集的重叠结果为空），故不提供通路层面结论；这是分析能力边界，不是结果缺失，也不引入替代性通路推测。', 'This dataset produced no usable entry under the current enrichment configuration (the overlap result of the predefined gene sets is empty), so no pathway-level conclusion is given; this is an analysis-capability boundary, not an absent result, and no substitute pathway is inferred.'),
    'concordance_absent': ('本数据集未产出对比一致性表：源分析中没有满足条件的可比较对比对，因此不提供跨对比重叠或一致性结论。', 'This dataset produced no contrast concordance table: the source analysis holds no contrast pair meeting the conditions, so no cross-contrast overlap or concordance conclusion is offered.'),
    'enrichment_boundary_missing': ('本数据集未产出预定义基因集的重叠富集文件，因此本报告不提供通路层面结论；这是该项分析未执行，不是分析执行后没有结果。', 'This dataset produced no overlap-enrichment file for the predefined gene sets, so this report gives no pathway-level conclusion; this analysis was not run, which is not the same as running it and finding nothing.'),
    'enrichment_boundary_other_contrasts': ('本数据集的重叠富集文件含有条目，但没有条目能对应到本报告的核心对比，因此本节不给出通路层面结论；这是对比覆盖边界，不是结果缺失。', 'The overlap-enrichment file of this dataset holds entries, but none of them corresponds to a core contrast of this report, so this section gives no pathway-level conclusion; this is a contrast-coverage boundary, not an absent result.'),
    'enrichment_boundary_unparsable': ('本数据集的重叠富集文件存在但无法解析，因此本报告不使用其内容，也不据此写通路结论；这是读取状态，不是结果缺失。', 'The overlap-enrichment file of this dataset exists but cannot be parsed, so this report does not use its content and writes no pathway conclusion from it; this is a read state, not an absent result.'),
    'overlap_state_missing': ('本次运行没有该文件，该项分析未执行', 'this run has no such file, so that analysis was not run'),
    'request_index_intro': ('本清单只列本次运行实际产出的证据。状态含义：已计算=可以直接引用并给出数值；未计算=本轮没有产出，不得据此写结论；无法使用=产出为空或不可读，必须按边界说明处理。报告引用数值时必须与本清单一致：标注「已计算」的内容不得写成「未提供 / 未包含 / 未产出」，标注「未计算」的内容不得写成已完成的分析结果。', 'This list covers only the evidence this run actually produced. Status values: computed = may be cited directly with its numbers; not computed = not produced in this run and no conclusion may rest on it; unusable = the output is empty or unreadable and must be handled as a boundary statement. Any number the report cites must agree with this list: content marked "computed" must not be written as "not provided / not included / not produced", and content marked "not computed" must not be written as a completed analysis result.'),
    'enrichment_hint_note_long': ('本节为探索性方向提示：所列通路来自预定义基因集的集合重叠结果，未经多重检验校正，无 FDR/q 值，不得作为显著性证据，仅用于假设生成。每个数据库每个方向取种子基因数最多的一条。', 'This section is an exploratory direction hint: the pathways listed come from set-overlap results on predefined gene sets, without multiple-testing correction and without FDR/q values, so they are not significance evidence and serve hypothesis generation only. Each database and direction contributes the single entry with the most seed genes.'),
    'consistency_note_intro': ('本节列出运行记录之间、以及正文与表格之间需要核对的口径差异，共 %d 条；提示只指出位置与依据，不修改任何数值或结论。', 'This section lists %d scope differences that need checking, between run records and between the body text and the tables; a hint points to a location and its basis and never changes a value or a conclusion.'),
    'unit_layer_intro': ('本节数值来自正式分析路径的单位层敏感性分析，覆盖本轮 %s 个对比（可做个体级检验 %s 个、不可估计 %s 个）。它以个体均值计权，与以单个观测计权的观察级检验估计目标不同，两套数值不可直接互换，也不可互相替代。', 'The values in this section come from the unit-level sensitivity analysis of the formal analysis path and cover the %s contrasts of this run (%s allow a unit-level test and %s are inestimable). It weights by unit means, whereas the observation-level test weights single observations, so the two estimands differ and the two sets of values can neither be interchanged nor substituted for one another.'),
    'enrichment_hint_reversed': ('本节的通路条目取自富集结果文件（对比键「%s」，该键下正值为 %s 较高）；本任务正文的方向基准为「%s」。两者臂序相反，因此下表方向列逐一写出该方向较高的一组，不沿用正文基准。', 'The pathway entries in this section come from the enrichment result file (contrast key "%s", where a positive value means %s is higher); the direction baseline of this task\'s body text is "%s". The two arm orders are opposite, so the direction column of the table below writes out the group that is higher for each direction instead of following the body baseline.'),
    'concordance_coverage_all': ('本表列出源表中的全部 %d 对比较。', 'This table lists all %d contrast pairs of the source table.'),
    'candidate_other_contrasts_short': ('本表另含任务小节之外的对比（%s），符号按各行「对比」列。', 'This table also contains contrasts outside the task sections (%s), with signs taken from the "Contrast" column of each row.'),
    'candidate_fold_note': ('本表按候选、对比、方向与统计量完全相同的行折叠后共 %d 行（源表 %d 行，重复 %d 行）；这里的计数单位是行，不是蛋白个数。', 'Rows identical in candidate, contrast, direction and statistics are folded, giving %d rows in total (source table %d rows, %d duplicates); the counting unit here is rows, not proteins.'),
    'concordance_coverage_capped': ('本表按源表顺序列出前 %d 对（共 %d 对）。', 'This table lists the first %d pairs in source-table order (of %d pairs in total).'),
    'candidate_source_direction_short': ('本表符号方向按各行「对比」列（与正文反向写法「%s」同义）。', 'The sign convention of this table follows the "Contrast" column of each row (synonymous with the reverse writing "%s" in the body).'),
    'candidate_other_contrasts_long': ('本表还包含报告任务小节之外的对比（%s）：这些行按各自差异表自身的符号给出，正值表示该行「对比」列中前一组较高。', 'This table also contains contrasts outside the report\'s task sections (%s): those rows follow the sign convention of their own differential table, and a positive value means the first group of that row\'s "Contrast" column is higher.'),
    'candidate_source_direction_long': ('本表逐行按该行「对比」列的方向给符号：以 %s 行为例，正值表示 %s 较高；同一对比在正文中按反向写法「%s」叙述时符号相反，两者是同一个对比。', 'This table gives the sign of each row from that row\'s "Contrast" column: for the row %s, a positive value means %s is higher; when the body text describes the same contrast by the reverse writing "%s" the signs are opposite, and the two are one and the same contrast.'),
    'finding_qc_stage': ('本轮分析矩阵的平均缺失率为 %.4f 且运行记录未执行插补（imputation_applied=%s），与原始交付矩阵的平均缺失率 %.4f 至 %.4f 不同一阶段；引用缺失/检出数值时应写明阶段', "The mean missing rate of this run's analysis matrix is %.4f and the run record shows no imputation (imputation_applied=%s), a different stage from the mean missing rate %.4f to %.4f of the original delivered matrix; when citing missing or detection values, state the stage"),
    'item_missing_rate_analysed_by_group': ('本轮分析矩阵的平均缺失率（分组）', "mean missing rate of this run's analysis matrix (by group)"),
    'item_log_transform_executed': ('本轮实际执行的对数变换', 'log transform actually executed in this run'),
    'item_matrix_transform_executed': ('本轮实际执行的矩阵变换', 'matrix transform actually executed in this run'),
    'item_imputation_executed_actual': ('本轮实际执行的缺失填补', 'missing-value imputation actually executed in this run'),
    'item_protein_filtering_executed': ('本轮实际执行的蛋白过滤', 'protein filtering actually executed in this run'),
    'not_computed_stratified_line': ('本轮未产出分层对比（状态：未计算）。', 'not produced in this run (status: not computed).'),
    'not_computed_figures_line': ('本轮未产出可引用图件（状态：未计算）。', 'no citable figure was produced in this run (status: not computed).'),
    'not_computed_line': ('本轮未产出（状态：未计算）。', 'not produced in this run (status: not computed).'),
    'unit_layer_absent': ('本轮未记录单位层结果（状态：未记录）；不得据此写成「已确认个体间无差异」。', 'this run recorded no unit-level result (status: not recorded); it must not be written as "no difference between individuals has been confirmed".'),
    'exclusion_trace_no_protein_level': ('本轮记录中不含蛋白级排除：被记录的未解析符号为分组与实验设计术语（例如分组名、样本表列名），不是蛋白候选，因此本节不提供候选层面的排除结论。', "This run's records hold no protein-level exclusion: the unresolved symbols recorded are grouping and experimental-design terms (for example group names or sample-table column names) and not protein candidates, so this section offers no candidate-level exclusion conclusion."),
    'figure_enrichment_dotplot_title': ('机制富集 dotplot', 'Mechanism enrichment dotplot'),
    'overlap_source_note': ('来源：本次运行 enrichment_results 中的预定义基因集重叠结果；已读取并解析。', 'Source: the predefined-gene-set overlap result in enrichment_results of this run; it was read and parsed.'),
    'reason_fewer_than': ('某一臂或两臂共同个体数不足', 'one arm, or the two arms taken together, has too few individuals'),
    'unit_background_rows_note': ('标注为「否（背景对比）」的行是本轮分析已算出的其它目标对比；它们只作背景，不得为它们另写任务小节，也不得把它们的通过数并入本节任务。', 'Rows marked "no (background contrast)" are other target contrasts this run already computed; they serve as background only, no task section may be written for them, and their passing counts may not be merged into the tasks of this section.'),
    'field_sample_id_col': ('样本 ID 列=%s', 'sample ID column=%s'),
    'figure_qc_sample_title': ('样本与缺失率 QC', 'Sample and missing-rate QC'),
    'file_sample_info': ('样本信息表', 'sample information table'),
    'field_n_samples': ('样本数=', 'sample count='),
    'file_module_sample_scores': ('样本级模块评分', 'sample-level module scores'),
    'finding_joint_mismatch': ('核对提示：按本报告声明的联合筛选条件（%s 连接）在差异表上复算得到 %d 行通过，与本对比汇总的 %s 不一致（只满足校正 P 的 %d 行、只满足效应阈值的 %d 行；最小校正 P=%s）；请核对筛选口径与数据阶段。', 'Recounting the joint screening condition this report declares (%s connector) on the differential table gives %d passing rows, which disagrees with the %s of this contrast summary (%d rows satisfy only the adjusted P, %d satisfy only the effect threshold; smallest adjusted P=%s); please check the screening scope and the data stage.'),
    'finding_unavailable_detail': ('核对提示：正文称「%s」类证据未提供或未执行，但本次运行已产出该类证据（数值见对应表格与「运行参数与证据绑定」）；此处只标记位置，不修改原句。', 'Check hint: the body text says evidence of class "%s" was not provided or not run, but this run produced such evidence (the values are in the corresponding tables and in "Run Parameters and Evidence Binding"); this marks the location only and does not change the sentence.'),
    'core_contrast_caption': ('核心对比（本表方向：正值表示 %s 较高，按「%s」）', 'Core contrast (direction in this table: a positive value means %s is higher, per "%s")'),
    'figure_group_means_title': ('核心蛋白组均值图', 'Representative protein group means'),
    'finding_stage_unclear': ('检出/缺失陈述未写明数据阶段，无法对应到运行记录中的具体矩阵', 'a detection/absence statement does not name its data stage, so it cannot be tied to a specific matrix in the run record'),
    'directive_note': ('模块分数与样本级评分仅作为方向性证据，不等同于显著性检验。', 'Module scores and sample-level scores serve as directional evidence only and are not equivalent to a significance test.'),
    'module_no_fdr_short': ('模块分数无 FDR，只作方向性证据（口径见「结论边界与方法局限」）。', 'Module scores carry no FDR and are directional evidence only (see "Conclusion Boundaries and Method Limitations").'),
    'module_no_fdr_long': ('模块分数是成员蛋白评分的分组汇总，没有 FDR，只能作为方向性证据，不能当作显著富集。', 'A module score is a group summary of the scores of its member proteins; it carries no FDR, so it can only be directional evidence and must not be written as significant enrichment.'),
    'file_module_group_summary': ('模块分组摘要', 'module group summary'),
    'field_module_mean_score': ('模块平均分=', 'module mean score='),
    'dose_flip_note_long': ('模块方向翻转的解释：%s 的 %s 模块在剂量轴上并不单调（%s），两个分层的方向相反且都未达 q≤0.05，因此高低剂量差值的符号相反应登记为未解释的观察，而不是单一机制的结论。', 'Explaining the module direction flip: the module %s of %s is not monotonic along the dose axis (%s), the two strata point in opposite directions and neither reaches q≤0.05, so the opposite sign of the high- versus low-dose difference is logged as an unexplained observation rather than as a conclusion about a single mechanism.'),
    'dose_flip_note_short': ('模块方向翻转：%s 的 %s 模块在分层间符号相反（%s），均未达 q≤0.05；按未解释的观察登记，不解释为单一机制。', 'Module direction flip: the module %s of %s has opposite signs between strata (%s), none of which reaches q≤0.05; it is logged as an unexplained observation and not interpreted as a single mechanism.'),
    'asset_module_index': ('模块记录共 %d 条：%d 条已在上文对应任务中列出%s', 'Module records: %d in total; %d are listed in the corresponding task above%s'),
    'module_caption_capped': ('模块证据（共 %d 条，本节列前 %d 条；Δ 方向同本任务组名）', 'Module evidence (%d in total; this section lists the first %d; the Δ direction follows the group names of this task)'),
    'module_caption_all': ('模块证据（共 %d 条；Δ 方向同本任务组名）', 'Module evidence (%d in total; the Δ direction follows the group names of this task)'),
    'ledger_table_all_found': ('模型表格数值均可在成品中找到', 'every value of the model table can be found in the finished report'),
    'ledger_table_missing': ('模型表格由确定性表替代，但以下数值未在成品中出现，需人工确认：%s', 'the model table was replaced by a deterministic table, but the following values do not appear in the finished report and need manual confirmation: %s'),
    'not_computed_ora': ('正式富集检验（超几何 ORA）', 'formal enrichment test (hypergeometric ORA)'),
    'candidate_rows_note': ('每一行是一个候选蛋白在一个对比中的统计结果；未列出的候选不在候选清单内。', 'Each row is the statistical result of one candidate protein in one contrast; a candidate not listed here is not in the candidate list.'),
    'item_samples_per_group': ('每组样本数（%s）', 'samples per group (%s)'),
    'note_prefix': ('注：%s', 'Note: %s'),
    'candidate_below_threshold_note': ('注：%s 的效应量绝对值未超过当前筛选口径的 0.25 效应阈值，列出供完整性参考，不计入通过筛选的蛋白集合（口径见上表「统计口径」列）。', 'Note: the absolute effect size of %s does not exceed the 0.25 effect threshold of the current screening scope, so it is listed for completeness and is not part of the set of proteins passing screening (see the "Statistical scope" column of the table above).'),
    'module_note_long': ('注：Δ 为 A 组均值减 B 组均值，方向与上表同一组名；模块分数是成员蛋白评分的汇总，没有 FDR，因此只能作为方向性证据，不能写成显著富集。', 'Note: Δ is the Group A mean minus the Group B mean, in the same direction as the group names of the table above; a module score is a summary of the scores of its member proteins and carries no FDR, so it can only be directional evidence and must not be written as significant enrichment.'),
    'module_note_short': ('注：Δ 方向同组名；模块分数无 FDR，只作方向性证据（口径见「结论边界与方法局限」）。', 'Note: the Δ direction follows the group names; module scores carry no FDR and are directional evidence only (see "Conclusion Boundaries and Method Limitations").'),
    'unit_column_unconfirmed_note': ('注：单位列不是研究者确认的字段，而是由自动生成的分析设计按列名匹配得到；“每个标签对应一个独立生物学个体”在本轮是标签层假定，本节数值按标签层探索性分析解读，不作为个体层确认性结论。', 'Note: the unit column is not a field the researcher confirmed but one obtained by column-name matching from an automatically generated analysis design; "each label corresponds to one independent biological individual" is a label-level assumption in this run, so the values in this section are read as a label-level exploratory analysis and not as a confirmatory individual-level conclusion.'),
    'merged_alias_note': ('注：模型原文把同一对比写成 %d 个平级小节（%s）；本节按对比合并，内容按原顺序全部保留（未删减）。本报告统一按「%s」坐标读数', 'Note: the model\'s own text wrote the same contrast as %d headings of equal rank (%s); this section merges them by contrast and keeps all content in its original order (nothing removed). This report reads everything in the "%s" coordinates'),
    'unit_excluded_note_long': ('混合来源或未确认来源的单位已单列并排除在独立个体之外：%s。「是否为独立生物学个体」按标签语义事先判定，不按检验结果挑选。', 'Units of mixed or unconfirmed origin have been listed separately and excluded from the independent individuals: %s. Whether a unit is an independent biological individual is decided in advance from label semantics and not selected by test result.'),
    'unit_excluded_note_digest': ('混合来源／未确认来源单位单列并排除在独立个体之外：%s（按标签语义事先判定）。', 'Units of mixed or unconfirmed origin are listed separately and excluded from the independent individuals: %s (decided in advance from label semantics).'),
    'unit_excluded_note_short': ('混合来源／未确认来源单位同前，已排除在独立个体之外。', 'Units of mixed or unconfirmed origin are as above and have been excluded from the independent individuals.'),
    'file_confounding_adjusted': ('混杂校正效应', 'confounding-adjusted effects'),
    'keyword_index': ('清单', 'index'),
    'overlap_status_line': ('状态：%s；不得据本项写通路结论，按边界说明处理。', 'Status: %s; no pathway conclusion may rest on this item, which is handled as a boundary statement.'),
    'field_threshold_used': ('生效阈值', 'threshold in force'),
    'source_user_provided': ('用户提供', 'provided by the user'),
    'concordance_correlation_note': ('相关性描述两个对比的效应分布关系，不构成显著性检验。', 'The correlation describes how the effect distributions of the two contrasts relate and does not constitute a significance test.'),
    'field_matrix_state': ('矩阵状态', 'matrix state'),
    'scale_note_matrix_state': ('矩阵状态记录为 %s', 'matrix state recorded as %s'),
    'figure_enrichment_dotplot_conclusion': ('离线富集条目按方向提示展示，未经多重检验校正；条目数见「富集方向提示」小节。', 'Offline enrichment entries are shown as direction hints without multiple-testing correction; the entry count is in the "Enrichment Direction Hints" section.'),
    'sign_convention_long': ('符号约定：本表按「%s」显示，底层差异表为 %s，其中 logFC 符号与本表相反；正文数值一律按「%s」方向给出（正值为 %s 较高、负值为 %s 较高），仅当句子明确引用源文件 %s 时按该文件的原始符号读取。', 'Sign convention: this table follows "%s" while the underlying differential table is %s, where the sign of logFC is opposite to this table; values in the body text are always given in the "%s" direction (a positive value means %s is higher, a negative value means %s is higher), and they follow the original sign of the source file %s only where a sentence explicitly cites that file.'),
    'sign_convention_short': ('符号约定：本表方向同上（按「%s」；源文件 %s 的符号相反）。', 'Sign convention: the direction of this table is as above (per "%s"; the sign of the source file %s is opposite).'),
    'samples_two_arms_named': ('第一臂 %s vs 第二臂 %s', 'arm 1 %s vs arm 2 %s'),
    'candidate_screening_scope_note': ('筛选口径只使用校正后 P 值（adj.P.Val）；原始 P 值列出仅供核对，不参与筛选。', 'The screening scope uses the adjusted P value only (adj.P.Val); the raw P values are listed for checking and take no part in screening.'),
    'figure_pca_conclusion_missing': ('线性降维用于检查分组分离与离群样本（解释率未记录）。', 'A linear projection used to check group separation and outlier samples (no explained variance recorded).'),
    'keyword_conclusion': ('结论', 'conclusion'),
    'boundary_pointer_phrase': ('结论边界', 'conclusion boundary'),
    'field_statistical_backend': ('统计后端', 'statistical backend'),
    'field_statistical_backend_and_reason': ('统计后端与回退原因', 'statistical backend and fallback reason'),
    'field_statistical_method': ('统计方法', 'statistical method'),
    'item_matrix_size_tested': ('统计检验所用矩阵规模', 'size of the matrix used for statistical testing'),
    'item_statistical_model_raw': ('统计模型（记录原文）', 'statistical model (recorded verbatim)'),
    'translate_synthesis': ('综合评述', 'synthesis'),
    'item_missing_value_handling': ('缺失值处理（统计口径）', 'missing-value handling (statistical scope)'),
    'note_transform_record_absent': ('缺少该记录只说明执行动作未记录，不能推断为未执行', 'the absence of that record only means the executed action was not recorded and cannot be read as not executed'),
    'ora_background_note': ('背景集为本次分析实际检测到的蛋白的基因符号（去重后 N=%s）：p 为超几何尾概率，q 为同一次查询内全部被检验条目的 BH 校正。', 'The background set is the gene symbols of the proteins this analysis actually detected (N=%s after de-duplication): p is the hypergeometric tail probability and q is the BH correction over all tested entries within the same query.'),
    'unit_source_design_auto': ('自动生成的分析设计中的单位列（按列名匹配，未由研究者确认）', 'the unit column of an automatically generated analysis design (matched by column name, not confirmed by the researcher)'),
    'file_protein_quant': ('蛋白定量矩阵', 'protein quantitative matrix'),
    'dose_protein_summary': ('蛋白水平：每个药物 × 分层单独检验并各自做 BH 校正，共检验 %d 个蛋白，其中 %d 个在 q≤0.05 下呈单调剂量趋势。', 'Protein level: each drug × stratum is tested separately with its own BH correction, testing %d proteins in total, of which %d show a monotonic dose trend at q≤0.05.'),
    'stratified_coverage_note': ('覆盖范围（这是本清单的显示元数据，不是分析结论）：本清单为该源表的节选，列出前 %d 行 / 共 %d 行，其余 %d 行未在本清单列出；节选只影响本清单的展示，不代表分析范围，也不要把「未列出的其余行」写成科学结论。', 'Coverage (this is display metadata of this list, not an analysis conclusion): this list is an excerpt of the source table and shows the first %d of %d rows, with the remaining %d rows not listed here; the excerpt affects only what this list displays, does not represent the analysis scope, and the "remaining rows not listed" must not be written as a scientific conclusion.'),
    'pointer_see_differential': ('见差异表', 'see the differential table'),
    'pointer_exclusion_trace': ('见本报告「候选排除轨迹」', 'see "Candidate Exclusion Trace" in this report'),
    'pointer_stratified': ('见本报告「分层对比摘要」', 'see "Stratified Contrast Summary" in this report'),
    'pointer_concordance': ('见本报告「对比一致性」', 'see "Contrast Concordance" in this report'),
    'pointer_evidence_tables': ('见本报告「证据表（逐条）」', 'see "Evidence Tables (record by record)" in this report'),
    'pointer_figures': ('见随报告交付的图册与图注', 'see the figure index and captions delivered with the report'),
    'unit_reference_note_short': ('观察级检验仍是参照估计，个体级与观察级的估计目标不同。', 'The observation-level test remains the reference estimate; the unit level and the observation level have different estimands.'),
    'unit_reference_note_long': ('观察级检验仍是参照估计：两者以不同的单位计权（个体均值 vs 单细胞观测），估计目标不同，数值不可直接互换。', 'The observation-level test remains the reference estimate: the two weight by different units (unit means vs single-cell observations), so the estimands differ and the values cannot be interchanged.'),
    'enrichment_status_unparsable': ('解析失败', 'parsing failed'),
    'keyword_discussion': ('讨论', 'discussion'),
    'translate_looks_logged_false': ('记录为未确认已对数化', 'recorded as not confirmed to be log-transformed'),
    'translate_needs_log_transform': ('记录为需要 log 变换', 'recorded as needing a log transform'),
    'note_scale_flags_only': ('记录的是对输入的判断字段；实际执行了哪一步变换、检验空间属于哪种尺度都未记录，因此与「数据与预处理」的尺度推断不矛盾也不互相支持', 'what is recorded are judgement fields about the input; neither the transform actually applied nor the scale of the testing space is recorded, so this neither contradicts nor supports the scale inference under "Data and Preprocessing"'),
    'reason_no_residual_df': ('设计没有剩余自由度，处理效应与个体效应无法分离', 'the design has no residual degrees of freedom, so the treatment effect cannot be separated from the individual effect'),
    'evidence_prefix': ('证据 ', 'Evidence  '),
    'evidence_ref_capture': ('证据 \\1', 'Evidence \\1'),
    'evidence_strength_moderate_to_low': ('证据强度：中等至较低', 'Evidence strength: moderate to low'),
    'evidence_strength_moderate_and_low': ('证据强度：分别为中等与较低', 'Evidence strength: moderate and low, respectively'),
    'evidence_strength_high': ('证据强度：较高', 'Evidence strength: high'),
    'translate_evidence_file': ('证据文件', 'evidence file'),
    'asset_evidence_entry_points': ('证据条目的复核入口：', 'Where to check the evidence records:'),
    'file_evidence_ledger': ('证据来源记录', 'evidence source ledger'),
    'translate_evidence_ids': ('证据清单', 'evidence list'),
    'figure_bar_direction_note_plural': ('该图按各对比源差异表的符号绘制，与正文显示方向的对应关系见各任务小节表注。', "This figure is drawn with the sign convention of each contrast's source differential table; the mapping to the display direction of the body text is given in the table notes of each task section."),
    'figure_bar_direction_note_single': ('该图按源差异表 %s 的符号绘制：上调＝%s 较高，下调＝%s 较高；正文按「%s」叙述时符号相反。', 'This figure is drawn with the sign convention of the source differential table %s: upregulated = %s is higher, downregulated = %s is higher; the body text narrates "%s", where the sign is opposite.'),
    'reason_no_unit_level_test': ('该对比不支持个体层检验', 'no unit-level test is defensible for this contrast'),
    'reader_note_no_testable_entry': ('该对比没有可检验的条目。', 'this contrast has no entry that can be tested.'),
    'matrix_alias_this_data': ('该数据', 'this data'),
    'reader_note_too_few': ('该方向进入检验的蛋白只有 %s 个，少于最少 %s 个的要求，因此本轮未执行该检验（保留未执行状态，未用替代方法补算）。', 'only %s proteins of this direction entered the test, fewer than the required minimum of %s, so the test was not run in this run (the not-run state is kept and no substitute method was used to fill it in).'),
    'note_non_missing_fraction_stage': ('该比例在原始交付矩阵上统计，与过滤后矩阵不同一阶段', 'this fraction is computed on the original delivered matrix and belongs to a different stage from the filtered matrix'),
    'overlap_exploratory_note': ('该结果为集合重叠，未经多重检验校正、没有 FDR/q 值，只能作为探索性方向提示；不得写成显著富集。', 'This result is a set overlap without multiple-testing correction and without FDR/q values, so it can only be an exploratory direction hint and must not be written as significant enrichment.'),
    'concordance_gap_note': ('该缺口是分析设计的边界，不是结果缺失。', 'This gap is a boundary of the analysis design, not an absent result.'),
    'ora_background_vs_matrix_note': ('该背景集按本次实际检测到的蛋白对应的基因符号去重后统计（同一基因的多行折叠为一条），因此与统计检验所用矩阵的蛋白行数 %s 不同；两个数字口径不同，引用时写明指哪一个。', 'That background set is counted over the de-duplicated gene symbols of the proteins this run actually detected (multiple rows of one gene fold into one), so it differs from the %s protein rows of the matrix used for statistical testing; the two figures have different scopes and citing them must state which one is meant.'),
    'transform_record_derived_note': ('该记录由本次运行的执行记录派生（%s），不是该步骤直接写出的独立文件。', "this record is derived from this run's execution record (%s) and is not a stand-alone file written directly by that step."),
    'contrast_pointer_note': ('说明：本小节讨论的对比与「任务%d：%s」相同，标题沿用模型给出的对比标识；两节数值来自同一张差异表，表格方向见各表标题后的方向标注，正文按模型原句保留。', 'Note: this subsection discusses the same contrast as "Task %d: %s", and its heading keeps the contrast identifier the model gave; both sections draw on one differential table, the direction of each table is given in the annotation after its caption, and the body text keeps the model\'s own sentences.'),
    'keyword_asset': ('资产', 'asset'),
    'task_alias_note': ('输入兼容：同一对比的其它写法（%s）会被识别为同一个任务，它们不是额外任务，不得为它们另写小节。', 'Input compatibility: other spellings of the same contrast (%s) are recognised as the same task; they are not extra tasks and no separate section may be written for them.'),
    'field_already_processed': ('输入已处理=%s', 'input already processed=%s'),
    'item_input_numeric_range': ('输入矩阵数值范围', 'value range of the input matrix'),
    'item_input_matrix_state': ('输入矩阵状态', 'input matrix state'),
    'item_non_missing_fraction': ('输入矩阵非缺失比例', 'non-missing fraction of the input matrix'),
    'evidence_boundary_line': ('边界：%s', 'Boundary: %s'),
    'stage_marker_filtered': ('过滤后', 'filtered'),
    'file_filtered_matrix': ('过滤后定量矩阵', 'filtered quantitative matrix'),
    'file_filtered_sample_info': ('过滤后样本信息表', 'filtered sample information table'),
    'stage_filtered_matrix': ('过滤后矩阵（统计检验所用）', 'filtered matrix (used for statistical testing)'),
    'source_matrix_shapes': ('运行目录中的矩阵文件行数与列数（行=蛋白，列=样本）', 'row and column counts of the matrix files in the run folder (rows = proteins, columns = samples)'),
    'source_run_record': ('运行记录 %s', 'run record %s'),
    'source_run_records_design': ('运行记录 · 分析设计与差异分析记录', 'run record · analysis design and differential-analysis records'),
    'binding_note': ('运行记录与主张绑定的机器可读副本；正文只呈现其中面向读者的部分', 'machine-readable copy of the binding between run records and claims; the body presents only the reader-facing part of it'),
    'item_n_significant_from_run': ('运行记录中的通过筛选数（%s）', 'passing count in the run record (%s)'),
    'scale_note_unknown_long': ('运行记录只记录了输入的判断字段，未记录实际变换，无法确认检验空间的尺度', 'the run record holds only the input judgement fields and nothing about the transform actually applied, so the scale of the testing space cannot be confirmed'),
    'unit_reason_unclassified': ('运行记录未给出可归类的阻断原因', 'the run record gives no classifiable blocking reason'),
    'note_transform_record_absent_request': ('运行记录没有矩阵变换的完整记录：不能把「未记录」写成「未执行」，也不能写成已经执行', 'The run record holds no complete record of the matrix transform: "not recorded" must not be written as "not executed", nor as having been executed'),
    'record_parameters': ('运行配置', 'run configuration'),
    'record_parameters_dot': ('运行配置 · ', 'run configuration ·  '),
    'record_parameters_design': ('运行配置 · 分析设计 · ', 'run configuration · analysis design ·  '),
    'record_parameters_design_differential': ('运行配置 · 分析设计 · 差异分析 · ', 'run configuration · analysis design · differential analysis ·  '),
    'record_parameters_design_matrix': ('运行配置 · 分析设计 · 矩阵 · ', 'run configuration · analysis design · matrix ·  '),
    'ledger_preserved': ('逐字保留', 'kept verbatim'),
    'dimred_note': ('逐样本 %s 坐标已在 processed_proteins/qc_metrics.csv 给出（%d 行）；UMAP 二维坐标为未产出坐标，不要写成缺少全部降维坐标。', 'per-sample %s coordinates are given in processed_proteins/qc_metrics.csv (%d rows); UMAP two-dimensional coordinates count as not produced, so do not write that all dimensionality-reduction coordinates are missing.'),
    'field_pass_count': ('通过筛选数', 'passing count'),
    'field_proteins_passing_screening': ('通过筛选的蛋白数', 'proteins passing screening'),
    'translate_n_sig': ('通过筛选的蛋白数=', 'proteins passing screening='),
    'keyword_overview': ('速览', 'overview'),
    'threshold_not_recorded': ('阈值未记录（parameters.json 缺 differential 阈值）', 'threshold not recorded (parameters.json has no differential thresholds)'),
    'keyword_appendix': ('附录', 'appendix'),
    'file_unmatched_table': ('附随结果表', 'accompanying result table'),
    'dose_flat_with_dose': ('随剂量不单调', 'non-monotonic with dose'),
    'field_needs_log_transform': ('需要对数变换=%s', 'needs log transform=%s'),
    'field_non_missing_fraction_short': ('非缺失比例', 'non-missing fraction'),
    'filtering_rule': ('非缺失比例 >= %s（%s）', 'non-missing fraction >= %s (%s)'),
    'not_computed_overlap_empty': ('预定义基因集的重叠富集（文件存在但是空对象）', 'overlap enrichment of predefined gene sets (the file exists but is an empty object)'),
    'not_computed_overlap_unparsable': ('预定义基因集的重叠富集（文件存在但解析失败——这是读取状态，不是未产出）', 'overlap enrichment of predefined gene sets (the file exists but does not parse - this is a read state, not an absent result)'),
    'not_computed_overlap_missing': ('预定义基因集的重叠富集（本次运行没有该文件：分析未执行）', 'overlap enrichment of predefined gene sets (this run has no such file: the analysis was not run)'),
    'confidence_high': ('高', 'high'),
    'figure_caption_merged': ('（一个图注条目可能覆盖同一编号段的多张图）', '(one caption entry may cover several figures of the same number range)'),
    'unit_all_testable': ('（全部可检验）', '(all testable)'),
    'unit_some_not_tested': ('（另有 %d 个蛋白行因有效单位不足标记为未检验）', '(%d further protein rows are marked as not tested for lack of valid units)'),
    'figure_file_fallback': ('（图册文件）%s', '(figure file) %s'),
    'candidate_none': ('（本对比没有候选蛋白记录）', '(no candidate protein record for this contrast)'),
    'task_section_unmatched': ('（本轮标准对比未匹配到模型小节：该对比按确定性事实呈现。）', "(this run's standard contrasts matched no model subsection: the contrast is presented from deterministic facts.)"),
    'evidence_source_note': ('（来源：%s）', '(source: %s)'),
    'key_summary_absent': ('（模型未提供速览，以下为确定性事实）', '(the model provided no key summary; the deterministic facts follow)'),
    'positive_means': ('（正值表示 %s 较高）', '(a positive value means %s is higher)'),
    'sign_opposite': ('（符号与此相反）', '(the sign is opposite)'),
    'boundary_pointer': ('（结论边界见本报告「结论边界与方法局限」）', '(for conclusion boundaries see "Conclusion Boundaries and Method Limitations" in this report)'),
    'asset_count_scope_note': ('（该计数按对比与蛋白组去重；「证据表（逐条）」的候选蛋白表另按方向与统计量去重，%s，因此两者行数不同，不是同一口径。）', '(this count is de-duplicated by contrast and protein group; the candidate protein table of "Evidence Tables (record by record)" is additionally de-duplicated by direction and statistics, %s, so the two row counts differ and do not share one scope.)'),
    'unit_direction_reversed_suffix': ('，与本节标题的写法相反，标题方向的正负号需取反', ', which is opposite to the wording of this section heading, so the signs of the heading direction must be flipped'),
    'more_contrasts_suffix': ('；其余 %d 个对比见运行配置中的对比清单', '; the remaining %d contrasts are in the contrast list of the run configuration'),
    'more_records_suffix': ('；其余 %d 个见下表。', '; the remaining %d are in the table below.'),
    'more_module_records_suffix': ('；其余 %d 条见下表。', '; the remaining %d are in the table below.'),
    'imputation_done_suffix': ('；分析矩阵按执行记录做过缺失填补（%s）', '; the analysis matrix was imputed according to the execution record (%s)'),
    'imputation_not_done_suffix': ('；执行记录显示未做缺失填补', '; the execution record shows no missing-value imputation'),
    'unit_direction_suffix': ('；本段数值按「%s − %s」方向给出（正值为 %s 侧较高）', '; the values in this paragraph are given in the "%s − %s" direction (a positive value means the %s side is higher)'),
    'figure_caption_capped_suffix': ('；此处列出前 %d 条，其余 %d 条未列出', '; the first %d are listed here and the remaining %d are not listed'),
    'imputation_unknown_suffix': ('；缺失填补状态未记录，不能写成未执行', '; the imputation state is not recorded and must not be written as not executed'),
}
register("assembly", _STRINGS)

# --------------------------------------------------------------------------- reader-facing wording
# Internal provenance / field vocabulary that must not reach a finished report. The model answers
# are written for an internal reader, so the assembler translates their scaffolding into plain
# Chinese *before* the text becomes part of the deliverable. Science is untouched: values, protein
# ids, evidence ids and directions are preserved verbatim.
#
# These tables are accessors rather than module constants on purpose. The report language is chosen
# at run time, so a table bound at import time would freeze whichever language was active then; each
# accessor resolves its templates for the language in force.


def _title_prefix() -> str:
    """Default report H1, used when a run exposes no descriptive title of its own."""
    return t("assembly.title_prefix")


def _source_labels() -> Dict[str, str]:
    """Provenance token -> reader label for the model's own source annotations."""
    return {
        "current_matrix": t("assembly.current_matrix"),
        "offline_enrichment": t("assembly.source_offline_enrichment"),
        "external_annotation": t("assembly.external_annotation"),
        "external_literature": t("assembly.source_external_literature"),
        "drug_database": t("assembly.source_drug_database"),
        "dataset_provided": t("assembly.source_dataset_provided"),
    }

PROVENANCE_TOKENS = ("current_matrix", "offline_enrichment", "external_annotation",
                     "external_literature", "drug_database", "dataset_provided")


def _field_tokens() -> Tuple[str, ...]:
    """Internal annotation fields: a parenthesised chunk naming one of these is scaffolding."""
    return ("置信度", "置信度：", "证据来源", "证据：", "证据", t("assembly.keyword_boundary"),
            "证据ID", "证据 ID")


def _translate_tokens(text: str) -> str:
    """Reader-facing prose: translate internal scaffolding, never touch numbers or ids.

    This is an English-to-Chinese rewriting layer: the internal vocabulary is already English and
    this step turns it into the Chinese the released report reads in. An English report needs no
    translation, so for every other language the layer is a pass-through and the internal tokens
    (provenance names, field names, direction enums) stay exactly as they are.
    """
    if get_language() != "zh":
        return text
    out = text
    # a source table that records an exact zero usually means underflow, not a true zero p-value
    out = re.sub(r"(?:FDR|adj\.P(?:\.Val)?|P\.Value|P)\s*[=＝]\s*0(?![.\d])",
                 t("assembly.translate_zero_p"), out)
    # confidence vocabulary (internal English level -> readable Chinese level)
    # a CJK range word right after the English level defeats \b, so the 至/到 form needs its own rule
    out = re.sub(r"置信度\s*[:：]?\s*low\s*(?:至|到|~|-|–|—|/)\s*moderate",
                 t("assembly.evidence_strength_low_to_moderate"), out)
    out = re.sub(r"置信度\s*[:：]?\s*moderate\s*(?:至|到|~|-|–|—|/)\s*low",
                 t("assembly.evidence_strength_moderate_to_low"), out)
    out = re.sub(r"置信度\s*[:：]?\s*(?:low[-–/]?moderate|moderate[-–/]?low)\b",
                 t("assembly.evidence_strength_low_to_moderate"), out)
    out = re.sub(r"置信度\s*[:：]?\s*(moderate|medium)\b", t("assembly.evidence_strength_moderate"), out)
    out = re.sub(r"置信度\s*[:：]?\s*low\b", t("assembly.evidence_strength_low"), out)
    out = re.sub(r"置信度\s*[:：]?\s*high\b", t("assembly.evidence_strength_high"), out)
    out = out.replace("置信度 moderate/low", t("assembly.evidence_strength_low_to_moderate"))
    out = out.replace("置信度 low–moderate", t("assembly.evidence_strength_low_to_moderate"))
    out = out.replace("置信度 low-moderate", t("assembly.evidence_strength_low_to_moderate"))
    out = out.replace("置信度 moderate", t("assembly.evidence_strength_moderate"))
    out = out.replace("置信度 low", t("assembly.evidence_strength_low"))
    out = out.replace("moderate/low", t("assembly.moderate_slash_low"))
    out = out.replace("low–moderate", t("assembly.lower_to_moderate")).replace("low-moderate", t("assembly.lower_to_moderate"))
    out = re.sub(r"置信度\s*[:：]?\s*分别为\s*low\s*/\s*moderate",
                 t("assembly.evidence_strength_low_and_moderate"), out)
    out = re.sub(r"置信度\s*[:：]?\s*分别为\s*moderate\s*/\s*low",
                 t("assembly.evidence_strength_moderate_and_low"), out)
    out = re.sub(r"置信度\s*[:：]?\s*分别为\s*low\s*与\s*moderate",
                 t("assembly.evidence_strength_low_and_moderate"), out)
    # provenance vocabulary
    out = out.replace("current_matrix", t("assembly.current_matrix"))
    out = out.replace("offline_enrichment", t("assembly.source_offline_enrichment"))
    out = out.replace("external_annotation", t("assembly.external_annotation"))
    out = out.replace("external_literature", t("assembly.source_external_literature"))
    out = out.replace("drug_database", t("assembly.source_drug_database"))
    out = out.replace("dataset_provided", t("assembly.source_dataset_provided"))
    # remaining internal field names / values
    out = re.sub(r"needs_log_transform\s*[=＝]\s*(?:true|True|1)", t("assembly.translate_needs_log_transform"), out)
    out = out.replace("looks_logged=false", t("assembly.translate_looks_logged_false"))
    out = out.replace("already_processed", t("assembly.translate_already_processed"))
    out = out.replace("batch_corrected_claimed", t("assembly.translate_batch_corrected_claimed"))
    out = out.replace("adj.significant", t("assembly.fdr_significant"))
    out = out.replace("analysis_design", t("assembly.analysis_design"))
    out = out.replace("n_sig=", t("assembly.translate_n_sig"))
    out = out.replace("n_sig", t("assembly.field_proteins_passing_screening"))
    out = out.replace("raw P<0.05 = ", t("assembly.translate_raw_p"))
    out = out.replace("log2_likely", t("assembly.translate_log2_likely"))
    out = out.replace("（type=synthesis）", "").replace("(type=synthesis)", "")
    out = out.replace("type=synthesis", t("assembly.translate_synthesis"))
    out = out.replace("exploratory_not_fdr_significant", t("assembly.translate_exploratory"))
    out = out.replace("inferred", t("assembly.translate_inferred"))
    out = out.replace("tested proteins", t("assembly.field_tested_proteins"))
    out = out.replace("analysed_n_proteins=", t("assembly.field_analysed_proteins"))
    out = out.replace("analysed_mean_missing_rate=", t("assembly.field_analysed_missing"))
    out = out.replace("n_samples=", t("assembly.field_n_samples"))
    out = out.replace("mean_score=", t("assembly.field_module_mean_score"))
    out = out.replace("n_up_display_group_a", t("assembly.field_n_up_a"))
    out = out.replace("n_down_display_group_a", t("assembly.field_n_down_a"))
    out = out.replace("n_a=", t("assembly.field_n_a_key")).replace("n_b=", t("assembly.field_n_b_key"))
    out = out.replace("contrast_estimable=", t("assembly.field_contrast_estimable"))
    out = out.replace("eta2_batch_pc_descriptive=", t("assembly.field_eta2_batch_pc"))
    out = out.replace("contrast_vif=", t("assembly.field_contrast_vif"))
    out = out.replace("confidence=moderate", t("assembly.evidence_strength_moderate"))
    out = out.replace("confidence=low", t("assembly.evidence_strength_low"))
    out = out.replace("up_in_group_a", t("assembly.translate_up_in_group_a"))
    out = out.replace("down_in_group_a", t("assembly.translate_down_in_group_a"))
    out = out.replace("up_in_display", t("assembly.translate_display_higher"))
    out = out.replace("down_in_display", t("assembly.translate_display_lower"))
    # R17: presence state enums are machine values; the claim they carry is kept word for word
    out = out.replace("not_detected_in_available_matrix", t("assembly.not_detected_in_matrix"))
    # R29: an unmapped symbol is a different state from an absent protein and must read differently.
    out = out.replace("symbol_not_matched_in_available_matrix",
                      t("assembly.translate_symbol_not_matched"))
    out = out.replace("symbol_not_mapped", t("assembly.translate_symbol_not_mapped"))
    out = out.replace("not detected in available matrix", t("assembly.not_detected_in_matrix"))
    out = out.replace("matrix_presence_only", t("assembly.not_entered_statistical_test"))
    # remaining English direction vocabulary inside Chinese sentences
    out = re.sub(r"\b(?:up|higher) in ([A-Za-z0-9_.\-]+)", t("assembly.translate_up_in_capture"), out)
    out = re.sub(r"\b(?:down|lower) in ([A-Za-z0-9_.\-]+)", t("assembly.translate_down_in_capture"), out)
    out = out.replace("display 方向", t("assembly.translate_display_direction"))
    out = re.sub(r"当前矩阵\s*\+\s*[A-Za-z0-9_.\-]+\s*[:：]", "", out)
    # provenance label followed by an evidence id: keep the id, drop the scaffolding
    out = re.sub(r"(?:%s)\s*/\s*(E-[A-Z])" % "|".join(PROVENANCE_TOKENS), t("assembly.evidence_ref_capture"), out)
    out = re.sub(r"(?:%s)|matrix_state=|scale_note" % "|".join(PROVENANCE_TOKENS), "", out)
    out = re.sub(r"(?m)^\s*当前矩阵\s*[:：]\s*", "", out)
    out = re.sub(r"\s*,\s*([；。，、])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out)
    # no space between two CJK characters: the translation itself must not re-introduce air
    out = re.sub(r"([\u3000-\u303f\u4e00-\u9fff])\s+(?=[\u3000-\u303f\u4e00-\u9fff])", r"\1", out)
    out = re.sub(r"（\s*）", "", out)
    return out


def _plain_boundary_pointer(text: str) -> str:
    """R16: the model tags sentences with the bare internal field name `boundary`.

    The report already has a section that holds that discussion, so the reader gets a pointer to it
    instead of an English field name. Applied after the annotation cleanup so the pointer is not
    mistaken for scaffolding and rewritten again.
    """
    pointer = t("assembly.boundary_pointer")
    out = re.sub(r"[（(]\s*(?:证据)?边界\s*(?:参见|见)?\s*boundary\s*[）)]", pointer, text, flags=re.I)
    out = re.sub(r"[（(]\s*boundary\s*[）)]", pointer, out, flags=re.I)
    return re.sub(r"\bboundary\b", t("assembly.boundary_pointer_phrase"), out, flags=re.I)


def _clean_parens(text: str) -> str:
    """Compact the model's internal source annotations into one reader-facing 证据 marker."""
    def repl(match):
        inner = match.group(1)
        if not any(token in inner for token in PROVENANCE_TOKENS + _field_tokens()):
            return match.group(0)
        # keep the evidence ids; keep the report-author numbers and provenance labels, drop only
        # the internal confidence vocabulary so no author data is lost.
        parts = []
        for chunk in re.split(r"[；;]", inner):
            chunk = chunk.strip()
            if not chunk:
                continue
            ids = re.findall(r"E-[A-Z][A-Za-z0-9_.-]*", chunk)
            if ids:
                # an evidence id is the load-bearing part of the annotation; drop the scaffolding
                for item in ids:
                    if item not in parts:
                        parts.append(item)
                continue
            chunk = re.sub(r"^\s*(?:置信度|confidence)\s*[:：]?\s*"
                           r"(?:low[-–/]?moderate|moderate[-–/]?low|low|moderate|medium|high)\s*$",
                           "", chunk, flags=re.I)
            # drop the field label only when the annotation carries more than the file name
            _label = re.match(r"^\s*(证据来源|证据|证据ID|evidence|边界)\s*[:：]\s*(.+)$", chunk, re.I)
            if _label and _label.group(2).strip() not in ("", _label.group(1)):
                chunk = _label.group(2).strip()
            chunk = chunk.strip()
            if chunk and chunk not in parts and not all(token in PROVENANCE_TOKENS
                                                       for token in re.split(r"[/、,，\s]+", chunk) if token):
                parts.append(chunk)
        body = t("assembly.evidence_prefix") + "；".join(parts) if parts else ""
        return "（%s）" % body if body else ""
    return re.sub(r"（([^（）]{0,400})）", repl, text)


def polish_for_reader(text: str) -> str:
    """Strip internal prefixes, file names and directive records from reader-facing prose."""
    out = text
    for pattern, label in _file_labels():
        out = re.sub(r"[A-Za-z0-9_/\.-]*" + pattern + r"(?:\.csv|\.json|\.md|\.yaml|\.jsonl)?", label, out)
    for pattern, replacement in _evidence_repairs():
        out = re.sub(pattern, replacement, out)
    for prefix in INTERNAL_PREFIXES:
        out = out.replace(prefix, "")
    out = re.sub(r"(证据)\1", r"\1", out)
    for pattern in DIRECTIVE_PATTERNS:
        if re.search(pattern, out):
            out = re.sub(pattern, "", out)
            out = out.rstrip(" 。；") + "。" + t("assembly.directive_note")
    out = re.sub(r"（\s*）", "", out)
    out = re.sub(r"[ 	]{2,}", " ", out)
    out = re.sub(r"\s+([，。；：])", r"", out)
    out = re.sub(r"([，。；：])\s*", r"", out)
    return out.strip(" ，；：")


def _file_labels() -> Tuple[Tuple[str, str], ...]:
    """Internal file-stem pattern -> reader-facing label; the specific patterns come first."""
    return (
        (r"ProteinQuant_Filtered", t("assembly.file_filtered_matrix")),
        (r"ProteinQuant_ComBat", t("assembly.file_combat_matrix")),
        (r"ProQuant_Normalized", t("assembly.file_normalized_matrix")),
        (r"ProteinQuant", t("assembly.file_protein_quant")),
        (r"SampleInfo_Filtered", t("assembly.file_filtered_sample_info")),
        (r"SampleInfo", t("assembly.file_sample_info")),
        (r"core_story_evidence", t("assembly.file_core_story")),
        (r"candidate_protein_evidence", t("assembly.file_candidate_protein")),
        (r"curated_module_group_summary", t("assembly.file_module_group_summary")),
        (r"curated_module_sample_scores", t("assembly.file_module_sample_scores")),
        (r"group_composition_qc", t("assembly.file_group_composition_qc")),
        (r"group_cross_table_[A-Za-z0-9_]+", t("assembly.file_group_cross_table")),
        (r"dataset_recipe_evidence", t("assembly.file_dataset_recipe")),
        (r"scoring_standard_coverage", t("assembly.file_scoring_coverage")),
        (r"candidate_exclusion_trace", t("assembly.asset_candidate_exclusion")),
        (r"confounding_adjusted_effects", t("assembly.file_confounding_adjusted")),
        (r"confounding_estimability", t("assembly.file_confounding_estimability")),
        (r"stratified_contrast_summary", t("assembly.file_stratified_summary")),
        (r"differential_[A-Za-z0-9_]+", t("assembly.file_differential")),
        (r"analysis_design\.used", t("assembly.file_analysis_design")),
        (r"evidence_ledger", t("assembly.file_evidence_ledger")),
        (r"figure_index", t("assembly.file_figure_index")),
    )


def _evidence_repairs() -> Tuple[Tuple[str, str], ...]:
    """Source-annotation patterns rewritten before the prose reaches a reader."""
    return (
        (r'\uff08\s*\u8bc1\u636e\u6765\u6e90\uff1a?\s*\u5f53\u524d\u77e9\u9635\s*\uff09', t("assembly.evidence_this_run_diff")),
        (r'\uff08\s*\u8bc1\u636e\u5f53\u524d\u77e9\u9635\s*\uff09', t("assembly.evidence_this_run_diff")),
        (r'\u5f53\u524d\u77e9\u9635\s*[:\uff1a]', ''),
        # R16: the catch-all must not claim a specific role for a file it cannot name; run after
        # the file-label table so every known table keeps its own reader label instead of collapsing into one.
        (r'(?<![A-Za-z0-9_])[A-Za-z0-9_-]+\.(?:csv|json|jsonl|yaml|md)', t("assembly.file_unmatched_table")),
    )
INTERNAL_PREFIXES = (
    "当前矩阵：", "当前矩阵:",
    "（证据当前矩阵：",
)
DIRECTIVE_PATTERNS = (
    "\u6309\s*R[123]\s*\u8981\u6c42[^\u3002\uff1b]*[\u3002\uff1b]?",
    "\u9075\u5faa\s*R[123][^\u3002\uff1b]*[\u3002\uff1b]?",
    "\uff08?\s*R[123][^\u3002\uff09]*\uff09?",
    "R[123]\s*\u7ea6\u675f",
)


def _evidence_source_note(match) -> str:
    """Turn an internal `evidence:` annotation into one reader-facing source note, deduplicated."""
    parts: List[str] = []
    for chunk in re.split(r"[;；]", match.group(1)):
        chunk = chunk.strip()
        if chunk and chunk not in parts:
            parts.append(chunk)
    return t("assembly.evidence_source_note") % "、".join(parts) if parts else ""


def rewrite_for_reader(text: str) -> str:
    """One place where model scaffolding becomes finished-report prose."""
    out = text.strip()
    # R2: an evidence marker that carries no readable source is worse than no marker
    out = re.sub(r"\uff08?\s*evidence\s*[:\uff1a]\s*\u5f53\u524d\u77e9\u9635\s*\uff09?", "", out)
    out = re.sub(r"\uff08\s*\u8bc1\u636e\uff1a\u5f53\u524d\u77e9\u9635\s*\uff09", "", out)
    out = re.sub(r"\uff08\s*\u8bc1\u636e\s*\uff1a\s*\uff09", "", out)
    # R16: the model writes the annotation with an ASCII colon as often as a full-width one, and the
    # internal field name can appear outside a parenthesis. Both are the same scaffolding defect.
    out = re.sub(r"evidence[_\s]?ids", t("assembly.translate_evidence_ids"), out, flags=re.I)
    out = re.sub(r"evidence\s*文件", t("assembly.translate_evidence_file"), out, flags=re.I)
    out = re.sub(r"\uff08\s*evidence\s*[:\uff1a]\s*([^\uff09]{0,200})\uff09", _evidence_source_note, out)
    out = out.replace("(evidence: current_matrix)", "")
    out = re.sub(r"\s{2,}", " ", out)
    # model answers sometimes come back with their own heading marker: never emit a heading here
    out = re.sub(r"^#{1,6}\s*", "", out)
    out = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", out)
    out = re.sub(r"[ \t]+#{1,6}[ \t]+", " ", out)
    out = _clean_parens(out)
    out = _translate_tokens(out)
    out = _plain_boundary_pointer(out)
    out = polish_for_reader(out)
    # direction word glued to a protein id, e.g. "PSMA1 与 HSPA1A方向为上调"
    out = re.sub(r"([A-Za-z0-9\-\)])(方向|较低|较高|上调|下调)", r"\1 \2", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([。；，、])", r"\1", out)
    out = re.sub(r"（\s*）", "", out)
    return out.strip()

REPORT_HASH_NORMALISATION = "utf-8, CRLF/CR -> LF"
REPORT_BINDING_SCHEMA = "report-binding-1"


def normalise_report_text(text: str) -> str:
    """The text form the report hash is defined on (CRLF and CR are folded to LF)."""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def report_hash_record(path) -> Dict[str, Any]:
    """Full sha256 of a report file under the declared normalisation (short hash is display only)."""
    raw = Path(path).read_bytes()
    text = normalise_report_text(raw.decode("utf-8", errors="replace"))
    payload = text.encode("utf-8")
    return {"file": str(path), "name": Path(path).name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sha256_raw": hashlib.sha256(raw).hexdigest(),
            "bytes": len(payload), "chars": len(text),
            "normalisation": REPORT_HASH_NORMALISATION}


def write_report_binding(run_folder, report_paths, mode="", extra=None) -> Dict[str, Any]:
    """External binding record, written after the report files exist (never self-referencing)."""
    import datetime
    files = []
    for item in report_paths:
        path = Path(item)
        if path.exists():
            files.append(report_hash_record(path))
    record = {"schema": REPORT_BINDING_SCHEMA,
              "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "hash_object": "final report text as written to disk",
              "normalisation": REPORT_HASH_NORMALISATION,
              "assembly_mode": str(mode or ""),
              "reports": files}
    if extra:
        record["extra"] = extra
    try:
        (Path(run_folder) / "report_binding.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return record


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def read_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return t("assembly.not_recorded")
    if isinstance(value, float):
        return "%.4g" % value
    return str(value)


def _cell(value: Any) -> str:
    """Markdown-safe cell text (escape the pipe that would otherwise break the table grid)."""
    return str(value if value is not None else "").replace("|", "\\|")

# --------------------------------------------------------------------------- conditions


def _unit_reason_pairs() -> Tuple[Tuple[str, str], ...]:
    """Legacy English reason needle -> the reader-facing one-liner that replaces it."""
    return (
        ("no experimental-unit column", t("assembly.reason_no_unit_column")),
        ("the design has no residual degrees of freedom", t("assembly.reason_no_residual_df")),
        ("matched by name only", t("assembly.reason_matched_by_name_only")),
        ("no unit-level test is defensible", t("assembly.reason_no_unit_level_test")),
        ("fewer than", t("assembly.reason_fewer_than")),
    )


def _reader_language_text(text: str) -> bool:
    """True when a recorded string carries text the report language can show as written.

    The unit-level reasons are written into the run records by the analysis path, not by this
    module, so the wording that arrives here depends on which language that path produced. The
    released Chinese wording is accepted always; Latin wording is accepted once the report is no
    longer Chinese, so an English report states the recorded reason instead of the boilerplate
    "unclassifiable" line.
    """
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    return get_language() != "zh" and any(ch.isascii() and ch.isalpha() for ch in text)


def _unit_reason_zh(record: Dict[str, Any]) -> str:
    """Reader-facing one-liner for a blocked unit-level analysis."""
    blob = " | ".join(str(item) for item in (record.get("reasons") or [])).lower()
    for needle, zh in _unit_reason_pairs():
        if needle.lower() in blob:
            return zh
    # R25: the unit-level reasons are written for the reader directly; when no legacy English
    # needle matches, prefer a reason that names a blocking condition, then the first reason that
    # already contains Chinese, instead of a boilerplate "unclassifiable" line.
    for item in record.get("reasons") or []:
        text = str(item).strip()
        if text.startswith("已排除"):
            continue  # the exclusion notice is information, not the blocking reason
        if text and any(key in text for key in ("门槛", "不足", "无法", "不能", "no unit-level test")):
            if _reader_language_text(text):
                return text.split("（")[0].strip()
    for item in record.get("reasons") or []:
        text = str(item).strip()
        if text and not text.startswith("已排除") and _reader_language_text(text):
            return text.split("（")[0].strip()
    return t("assembly.unit_reason_unclassified")


def _unit_source_labels() -> Dict[str, str]:
    """Unit-column provenance token -> the wording a reader can act on."""
    return {
        "analysis_design.pair_col_auto_inferred": t("assembly.unit_source_design_auto"),
        "analysis_design.pair_col": t("assembly.unit_source_design_declared"),
        "column_name_match_unconfirmed": t("assembly.unit_source_column_name_unconfirmed"),
        "absent": t("assembly.not_recorded"),
    }


def _unit_source_label(value: Any) -> str:
    text = str(value or "")
    return _unit_source_labels().get(text, text or t("assembly.not_recorded"))


def read_conditions(run: Path) -> Dict[str, Any]:
    """Recorded thresholds / scale / matrix state; missing stays missing."""
    params = read_json(Path(run) / "parameters.json", {}) or {}
    design = params.get("analysis_design") or {}
    diff = design.get("differential") or {}
    matrix = design.get("matrix") or {}
    p_thresh, logfc_thresh = diff.get("p_thresh"), diff.get("logfc_thresh")
    # R28: the design record only carries inferred flags, so an absent one used to read as "scale
    # unknown" even when the run had recorded the transform it executed. The execution record is
    # consulted second, and the resulting statement names the evidence it rests on.
    transform = read_matrix_transform_record(Path(run))
    t_log = (transform.get("log2_transform") or {}) if isinstance(transform, dict) else {}
    if matrix.get("log2_transform_applied") is True:
        scale, scale_note = "log2", t("assembly.scale_note_log2_recorded")
    elif t_log.get("applied") is True:
        scale = "log2"
        scale_note = (t("assembly.scale_note_from_execution_record")
                      % (t_log.get("form") or t("assembly.translate_log_transform_form_fallback"),
                         t_log.get("applied_basis") or t("assembly.not_recorded"),
                         (transform.get("record_source") if isinstance(transform, dict) else "")
                         or t("assembly.not_recorded")))
    elif str(matrix.get("matrix_state") or "").lower() in {"logged", "log2"}:
        scale, scale_note = "log2", t("assembly.scale_note_matrix_state") % matrix.get("matrix_state")
    elif matrix.get("looks_logged") is True:
        scale, scale_note = "log2_likely", t("assembly.scale_log2_range_only")
    else:
        scale, scale_note = "unknown", t("assembly.scale_note_unknown")
    return {"p_thresh": p_thresh, "logfc_thresh": logfc_thresh,
            "fdr_method": diff.get("fdr_method"), "contrasts": diff.get("contrasts") or [],
            "matrix_state": matrix.get("matrix_state"), "scale": scale, "scale_note": scale_note,
            "group_col": design.get("group_col"), "sample_id_col": design.get("sample_id_col"),
            "batch_col": design.get("batch_col"), "pair_col": design.get("pair_col")}


def threshold_text(cond: Dict[str, Any]) -> str:
    p, logfc = cond.get("p_thresh"), cond.get("logfc_thresh")
    fdr = cond.get("fdr_method") or "BH"
    if p is None or logfc is None:
        return t("assembly.threshold_not_recorded")
    return t("assembly.threshold_text") % (_fmt(p), _fmt(logfc), _fmt(fdr))


def effect_column(cond: Dict[str, Any]) -> str:
    return "logFC" if cond.get("scale") == "log2" else t("assembly.effect_column_unconfirmed")

# --------------------------------------------------------------------------- evidence index


def build_evidence_index(run: Path) -> Dict[str, Any]:
    """Composite keys: candidates by (contrast, protein group / gene), modules by (contrast, module)."""
    facts = evidence.build_facts(run)
    cond = read_conditions(run)
    index: Dict[str, Any] = {"run": str(run), "contrasts": [], "candidates": {}, "modules": {},
                             "conditions": cond, "duplicates": []}
    for order, rec in enumerate(facts["story"]["contrasts"], start=1):
        display = rec.get("display_contrast") or rec.get("source_contrast")
        index["contrasts"].append({
            "id": "E-CTR%d" % order, "display_contrast": display,
            "source_contrast": rec.get("source_contrast"),
            "groups": [rec.get("group_a"), rec.get("group_b")],
            "inverted": bool(rec.get("inverted_from_source")), "n_sig": rec.get("n_sig"),
            "n_up_display_a": rec.get("n_up_display_group_a"),
            "n_down_display_a": rec.get("n_down_display_group_a"),
            "claim_title": rec.get("claim_title"), "claim_id": rec.get("claim_id"),
            "source_file": rec.get("source_file")})
    for rec in facts["story"]["candidates"]:
        contrast = rec.get("display_contrast") or rec.get("source_contrast") or ""
        identity = rec.get("protein_group") or rec.get("candidate") or rec.get("matched_gene") or ""
        key = (contrast, identity)
        entry = {"id": "E-CAND-%s-%s" % (sha12(str(contrast))[:4], identity),
                 "display_contrast": rec.get("display_contrast"),
                 "source_contrast": rec.get("source_contrast"),
                 "candidate": rec.get("candidate"), "matched_gene": rec.get("matched_gene"),
                 "protein_group": rec.get("protein_group"),
                 "logFC_display": rec.get("logFC_display"), "logFC_source": rec.get("logFC_source"),
                 "adj.P.Val": rec.get("adj.P.Val"), "P.Value": rec.get("P.Value"),
                 "direction_display": rec.get("direction_display"),
                 "source_file": rec.get("source_file")}
        if key in index["candidates"]:
            index["duplicates"].append({"kind": "candidate", "key": str(key)})
        index["candidates"][key] = entry
    for rec in facts["story"]["modules"]:
        contrast = rec.get("display_contrast") or rec.get("source_contrast") or ""
        members = sha12(str(rec.get("matched_genes") or ""))[:4]
        key = (contrast, str(rec.get("module")))
        entry = {"id": "E-MOD-%s-%s-%s" % (sha12(str(contrast))[:4], rec.get("module"), members),
                 "display_contrast": rec.get("display_contrast"),
                 "groups": [rec.get("group_a"), rec.get("group_b")], "delta": rec.get("delta"),
                 "n_matched": rec.get("n_matched_genes"), "mean_a": rec.get("group_a_mean_score"),
                 "mean_b": rec.get("group_b_mean_score"), "members_sha4": members,
                 "matched_genes": rec.get("matched_genes")}
        if key in index["modules"]:
            index["duplicates"].append({"kind": "module", "key": str(key)})
        index["modules"][key] = entry
    index["thresholds"] = facts["thresholds"]
    return index


def plan_report(run: Path, sections: Dict[str, Any]) -> Dict[str, Any]:
    index = build_evidence_index(run)
    if not index["contrasts"]:
        return {"mode": "fallback_minimal", "index": index, "tasks": [], "sections": sections}
    tasks = []
    for order, rec in enumerate(index["contrasts"], start=1):
        tasks.append({"order": order, "contrast": rec,
                      "candidates": [k for k, v in index["candidates"].items()
                                     if v["display_contrast"] == rec["display_contrast"]],
                      "modules": [k for k, v in index["modules"].items()
                                  if v["display_contrast"] == rec["display_contrast"]]})
    return {"mode": "hybrid", "index": index, "tasks": tasks, "sections": sections}

# --------------------------------------------------------------------------- blocks / task mapping


def register_blocks(text: str) -> List[Dict[str, Any]]:
    """Register every block of the source text before any parsing touches it."""
    blocks: List[Dict[str, Any]] = []
    heading = ""
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            kind = "table" if any(line.strip().startswith("|") for line in buffer) else "prose"
            if all(line.strip().startswith(("-", "*")) for line in buffer if line.strip()):
                kind = "bullets"
            blocks.append({"index": len(blocks), "heading": heading, "kind": kind,
                           "sha12": sha12(body), "chars": len(body), "text": body})
        buffer.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
            continue
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return blocks


def _match_task_sections(sections: Dict[str, Any],
                         contrast: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """Every source section that names this contrast, in source order.

    R21/F4: one scientific comparison carries several equivalent ids (the canonical one plus the
    aliases the request used to list, such as hsc_gmp / HSC_vs_GMP / GMP_vs_HSC). Matching only the
    first one left the others as unbound sections and printed the same comparison three times. All
    matches are returned so the report can merge them into one task without dropping text that only
    one of the sections holds.
    """
    wanted = {str(contrast.get("claim_id") or ""), str(contrast.get("display_contrast") or ""),
              str(contrast.get("source_contrast") or ""), str(contrast.get("id") or "")}
    wanted = {w for w in wanted if w}
    found = []
    for task in sections.get("tasks") or []:
        key = str(task.get("match") or "").strip()
        if key and key in wanted:
            found.append(task)
    if not found:
        return [], "unbound"
    return found, "exact" if len(found) == 1 else "exact_merged"


def _match_task_section(sections: Dict[str, Any], contrast: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Back-compatible single-section view of the plural matcher added in R21."""
    found, route = _match_task_sections(sections, contrast)
    return (found[0] if found else None), route

# --------------------------------------------------------------------------- rendering
# R17: repeated method sentences are emitted once and then referred to by a short pointer, so the
# reader sees each caveat in full but the sections stop repeating whole paragraphs.
NOTE_SEEN: Dict[str, bool] = {}


def reset_notes() -> None:
    NOTE_SEEN.clear()


def _note_once(key: str, full: str, short: str = "") -> List[str]:
    if NOTE_SEEN.get(key):
        return [short] if short else []
    NOTE_SEEN[key] = True
    return [full]


def _contrast_block(task: Dict[str, Any], cond: Dict[str, Any]) -> List[str]:
    rec = task["contrast"]
    groups = rec.get("groups") or [t("assembly.group_a"), t("assembly.group_b")]
    lines = [t("assembly.core_contrast_caption")
             % (_cell(groups[0]), _cell(rec.get("display_contrast"))),
             t("assembly.header_core_contrast") % (_cell(groups[0]), _cell(groups[1])),
             "|---|---:|---:|---:|---|",
             "| %s | %s | %s | %s | %s |" % (_cell(rec.get("display_contrast")), _fmt(rec.get("n_sig")),
                                             _fmt(rec.get("n_up_display_a")), _fmt(rec.get("n_down_display_a")),
                                             threshold_text(cond))]
    if rec.get("inverted"):
        lines += _note_once(
            "inverted_%s" % _cell(rec.get("source_contrast")),
            t("assembly.sign_convention_long")
            % (_cell(rec.get("display_contrast")), _cell(rec.get("source_contrast")),
               _cell(rec.get("display_contrast")), _cell(groups[0]), _cell(groups[1]),
               _cell(rec.get("source_contrast"))),
            t("assembly.sign_convention_short")
            % (_cell(rec.get("display_contrast")), _cell(rec.get("source_contrast"))))
    return lines


def _candidate_block(task: Dict[str, Any], index: Dict[str, Any], cond: Dict[str, Any],
                     limit: Optional[int] = None, total: Optional[int] = None) -> List[str]:
    keys = task["candidates"] if limit is None else task["candidates"][:limit]
    if not keys:
        return [t("assembly.candidate_none")]
    effect = effect_column(cond)
    # R17: the caption says how many rows exist instead of promising more than the table shows
    shown_total = total if total is not None else len(task["candidates"])
    caption = (t("assembly.candidate_caption_all") % shown_total if shown_total <= len(keys)
               else t("assembly.candidate_caption_capped")
                    % (shown_total, len(keys)))
    lines = [caption,
             t("assembly.header_candidate_block") % (_cell(effect), _cell(effect)),
             "|---|---|---:|---:|---|---|---|"]
    for key in keys:
        rec = index["candidates"][key]
        contrast = task.get("contrast")
        if not contrast:
            contrast = next((item for item in index["contrasts"]
                             if item.get("display_contrast") == rec.get("display_contrast")), {})
        lines.append("| %s | %s | %s | %s | %s | %s | %s |"
                     % (_cell(rec.get("candidate")), _cell(rec.get("protein_group")),
                        _fmt(rec.get("logFC_display")), _fmt(rec.get("logFC_source")),
                        _fmt(rec.get("adj.P.Val")), _cell(_direction_label(rec, contrast.get("groups"))),
                        _cell(_reader_asset_label(Path(str(rec.get("source_file") or "")).name))))
    inverted_keys = {(_cell(rec.get("display_contrast")), _cell(rec.get("source_contrast")))
                     for rec in (index["candidates"][key] for key in keys)
                     if rec.get("source_contrast") and rec.get("display_contrast")
                     and rec.get("source_contrast") != rec.get("display_contrast")}
    for display, source in sorted(inverted_keys):
        groups = next((item.get("groups") for item in index["contrasts"]
                       if item.get("display_contrast") == display), None) or [t("assembly.arm_first"), t("assembly.arm_second")]
        lines += _note_once(
            "cand_direction_%s" % source,
            t("assembly.candidate_direction_long") % (display, _cell(groups[0]), source, _cell(groups[1])),
            t("assembly.candidate_direction_short") % (display, source))
    below = []
    for key in keys:
        record = index["candidates"][key]
        try:
            value = abs(float(record.get("logFC_display")))
        except Exception:
            continue
        if value <= 0.25:
            below.append(_cell(record.get("candidate")))
    if below:
        lines.append("")
        lines.append(t("assembly.candidate_below_threshold_note") % "、".join(below))
    return lines


def _module_block(task: Dict[str, Any], index: Dict[str, Any], limit: Optional[int] = None,
                  total: Optional[int] = None) -> List[str]:
    keys = task["modules"] if limit is None else task["modules"][:limit]
    if not keys:
        return []
    # R17: the member hash is machine metadata (it stays in the manifest), not reader content.
    shown_total = total if total is not None else len(task["modules"])
    caption = (t("assembly.module_caption_all") % shown_total if shown_total <= len(keys)
               else t("assembly.module_caption_capped")
                    % (shown_total, len(keys)))
    lines = [caption,
             t("assembly.header_module_block"),
             "|---|---:|---:|---:|---:|"]
    for key in keys:
        rec = index["modules"][key]
        lines.append("| %s | %s | %s | %s | %s |"
                     % (_cell(key[1]), _fmt(rec.get("mean_a")), _fmt(rec.get("mean_b")),
                        _fmt(rec.get("delta")), _fmt(rec.get("n_matched"))))
    if keys:
        lines += _note_once(
            "module_no_fdr",
            t("assembly.module_note_long"),
            t("assembly.module_note_short"))
    return lines


def _verdict_line(task: Dict[str, Any], cond: Dict[str, Any]) -> str:
    rec = task["contrast"]
    groups = rec.get("groups") or [t("assembly.group_a"), t("assembly.group_b")]
    return (t("assembly.verdict_line")
            % (_cell(rec.get("display_contrast")), _fmt(rec.get("n_sig")), threshold_text(cond),
               _cell(groups[0]), _fmt(rec.get("n_up_display_a")), _cell(groups[1]),
               _fmt(rec.get("n_down_display_a"))))


def _unit_sensitivity_lines(run: Path, rec: Dict[str, Any], index: Dict[str, Any]) -> List[str]:
    """Reader-facing unit-level interpretation for one task.

    R25: the unit-level result is not only a row in the run-parameter table. Each task states its
    own experimental unit, which estimand was used, how many units entered, what passed each screen
    and why the analysis was blocked when it was. The numbers come from the same record the tables
    are built from; nothing is recomputed here.
    """
    limma = read_json(Path(run) / "processed_proteins" / "limma_summary.json", {})
    if not isinstance(limma, dict):
        return []
    key, _ = _match_contrast_key(limma, [rec.get("display_contrast"), rec.get("source_contrast"),
                                        rec.get("claim_id")])
    unit = ((limma.get(key) or {}).get("unit_sensitivity") if key else None)
    if not isinstance(unit, dict) or not unit:
        return []
    status = str(unit.get("status") or "")
    column = str(unit.get("analysis_unit") or t("assembly.not_recorded"))
    source = _unit_source_label(unit.get("unit_column_source"))
    effect_name = str(unit.get("effect_column") or t("assembly.field_effect_size"))
    thr = unit.get("logfc_threshold")
    fdr = unit.get("n_passing_fdr_only")
    if fdr is None:
        fdr = unit.get("n_passing_fdr")
    joint = unit.get("n_passing_joint")
    exclusion = unit.get("unit_exclusion") or {}
    excluded = sorted(set(exclusion.get("pooled_source") or []) | set(exclusion.get("unconfirmed_source") or []))
    lines: List[str] = []
    if status == "completed":
        direction = str(unit.get("direction") or "").replace(" - ", " − ")
        arms = [part.strip() for part in str(unit.get("direction") or "").split(" - ")]
        direction_text = ""
        if len(arms) == 2 and arms[0] and arms[1]:
            direction_text = t("assembly.unit_direction_suffix") % (arms[0], arms[1], arms[0])
            if _reversed_arms(key or "", rec.get("display_contrast")):
                direction_text += t("assembly.unit_direction_reversed_suffix")
        route_key = str(unit.get("analysis_route"))
        estimand_zh = (t("assembly.estimand_within_unit_mean_diff") if route_key == "within_unit_paired"
                       else t("assembly.estimand_between_unit_pooled"))
        tested = _fmt(unit.get("n_proteins_tested"))
        not_tested = int(unit.get("n_proteins_not_tested") or 0)
        joint_text = (t("assembly.unit_joint_pass") % (_fmt(thr), _fmt(joint))
                      if joint is not None else t("assembly.unit_no_joint_threshold"))
        if route_key == "within_unit_paired":
            design_text = (t("assembly.unit_design_paired")
                           % _fmt(unit.get("n_pairs") or unit.get("n_units_shared")))
        else:
            design_text = (t("assembly.unit_design_between")
                           % (_fmt(unit.get("n_units_a_only")), _fmt(unit.get("n_units_b_only"))))
        lines.append(t("assembly.unit_sensitivity_completed")
                     % (_cell(column), source, estimand_zh, direction_text, design_text, tested,
                        (t("assembly.unit_some_not_tested") % not_tested) if not_tested else t("assembly.unit_all_testable"),
                        _fmt(fdr), joint_text, _fmt(unit.get("min_adj_p_value"))))
        lines.append(t("assembly.unit_reference_note_long"))
    else:
        lines.append(t("assembly.unit_sensitivity_blocked")
                     % (source, _unit_reason_zh(unit)))
    if str(unit.get("unit_column_source") or "") in ("column_name_match_unconfirmed",
                                                      "analysis_design.pair_col_auto_inferred"):
        lines.append(t("assembly.unit_column_unconfirmed_note"))
    if excluded:
        lines += _note_once(
            "unit_excluded_units",
            t("assembly.unit_excluded_note_long") % "、".join(_cell(x) for x in excluded),
            t("assembly.unit_excluded_note_short"))
    reason_needed = [str(item) for item in (unit.get("reasons") or [])
                     if "分布不均" in str(item) or "矩阵匹配后" in str(item)]
    for item in reason_needed[:2]:
        lines.append(t("assembly.note_prefix") % item)
    return lines


def _data_section(run: Path, cond: Dict[str, Any],
                  run_evidence: Optional[Dict[str, Any]] = None) -> List[str]:
    contrasts = "; ".join("%s（%s vs %s）" % (c.get("name"), c.get("group_a"), c.get("group_b"))
                          for c in (cond.get("contrasts") or [])[:8])
    state = _fmt(cond.get("matrix_state"))
    scale = cond.get("scale")
    if scale == "log2_likely":
        scale = t("assembly.scale_log2_inferred")
        scale_note = t("assembly.scale_log2_range_only_long")
    elif scale == "unknown":
        scale = t("assembly.not_recorded")
        scale_note = t("assembly.scale_note_unknown_long")
    else:
        scale_note = cond.get("scale_note") or ""
    lines = [t("assembly.data_matrix_state") % (state, _fmt(scale), scale_note),
             t("assembly.data_group_cols") % (_fmt(cond.get("group_col")),
                                                         _fmt(cond.get("sample_id_col")),
                                                         _fmt(cond.get("batch_col"))),
             t("assembly.data_threshold") % threshold_text(cond),
             t("assembly.run_contrasts") % (contrasts or t("assembly.pointer_see_differential"))]
    if run_evidence:
        lines += _run_method_summary_lines(run, cond, run_evidence)
    else:
        lines.append(t("assembly.data_evidence_paths"))
    return lines

REPORT_TITLE: Dict[str, str] = {}


def set_report_title(title: str) -> None:
    """Set a dataset-specific descriptive title used as the report H1.

    R22: the override is single-use, so a title set for one run cannot leak into the next run
    assembled in the same process.
    """
    REPORT_TITLE["value"] = title
    REPORT_TITLE["pending"] = True


def _dataset_task_text(run: Path) -> str:
    """The dataset's own user task text, or "" when this run does not expose it."""
    try:
        params = json.loads((Path(run) / "parameters.json").read_text(encoding="utf-8-sig"))
    except Exception:
        return ""
    for key in ("user_input_path", "input_folder"):
        raw = params.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_dir():
            path = path / "user_input.txt"
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8-sig")
        except Exception:
            continue
    return ""


def _descriptive_title_core(task_text: str) -> str:
    """Verbatim noun phrase from the task's own background line; "" when it cannot be taken."""
    match = re.search(r"【背景信息】\s*(.+?)(?:\n\s*\n|\n【|$)", task_text or "", re.S)
    background = (match.group(1).strip().replace("\n", " ") if match else "")
    sentence = background.split("。")[0]
    match = re.search(r"(?:来自|源于|取自)\s*(.+?)\s*(?:，|。|；|$)", sentence)
    if not match:
        return ""
    core = match.group(1).strip().replace("\x60", "").replace("*", "")
    if "的" in core:
        head, _, raw_tail = core.partition("的")
        gap = " " if raw_tail[:1] == " " else ""
        head, tail = head.strip(), raw_tail.strip()
        core = (t("assembly.title_core_join") % (head, gap, tail)) if (head and tail) else (head or core)
    core = re.sub(r"\s+", " ", core).strip(" ，,。；;：:")
    core = re.sub(r"(矩阵|数据集|数据)$", "", core).strip()
    lowered = core.lower()
    if (not core or len(core) > 40 or core.startswith("/") or "\\" in core or ":/" in core
            or ".csv" in lowered or ".txt" in lowered):
        return ""
    return core


def derive_report_title(run: Path) -> str:
    """Descriptive, brand-free H1 built from this run's own dataset task text.

    R22: production reports carried the generic default title while the descriptive titles lived in
    docs/tools - and one of those named a tissue the dataset does not contain. The title is now taken
    verbatim from the dataset's own background line, and the neutral default is kept whenever that
    line is unavailable, so no species or tissue is ever guessed.
    """
    core = _descriptive_title_core(_dataset_task_text(run))
    if not core:
        return _title_prefix()
    return (t("assembly.title_suffix_report") % core) if core.endswith("分析") else (t("assembly.title_suffix_analysis_report") % core)


def _resolved_report_title(run: Path) -> str:
    """Explicit single-use override first (kept for the offline replay tools), else derived title."""
    if REPORT_TITLE.get("pending"):
        REPORT_TITLE["pending"] = False
        title = str(REPORT_TITLE.get("value") or "").strip()
        if title:
            return title
    return derive_report_title(run)

ENRICHMENT_DATABASES = ("GO", "KEGG", "Reactome")

DIRECTION_KEYS = ("upregulated", "downregulated")
# the overlap file is read once per run: ten task sections share one 97 KB payload
_OVERLAP_CACHE: Dict[str, Any] = {}


def _overlap_state(run: Path) -> Tuple[str, Any]:
    """Which fact the overlap step actually produced, read once per run.

    R21/F2: an unreadable file, an analysis that never ran, an executed analysis with no entry and a
    file that carries entries for other contrasts are four different facts. They used to collapse
    into one empty dict, and the report then told the reader the dataset "produced nothing" even
    when the file held hundreds of entries under another top-level key.
    """
    path = Path(run) / "enrichment_results" / "go_kegg_reactome_results.json"
    try:
        stat = path.stat()
    except Exception:  # noqa: BLE001
        return "missing", None
    key = "%s|%d|%d" % (path, stat.st_mtime_ns, stat.st_size)
    if _OVERLAP_CACHE.get("key") != key:
        try:
            # R21/F2: a strict read on purpose. read_json(path, {}) swallows a parse error and hands
            # back an empty object, which made "the file does not parse" indistinguishable from "the
            # analysis ran and produced nothing" - the one thing this state is here to keep apart.
            _OVERLAP_CACHE["value"] = json.loads(path.read_text(encoding="utf-8-sig"))
            _OVERLAP_CACHE["error"] = ""
        except Exception as exc:  # noqa: BLE001
            _OVERLAP_CACHE["value"] = None
            _OVERLAP_CACHE["error"] = str(exc)
        _OVERLAP_CACHE["key"] = key
    if _OVERLAP_CACHE.get("error"):
        return "unparsable", None
    payload = _OVERLAP_CACHE.get("value")
    if not isinstance(payload, dict) or not payload:
        return "empty", payload if isinstance(payload, dict) else None
    return "ok", payload


def _as_names(names: Any) -> List[str]:
    """One contrast can be named several ways; every reader accepts any of them."""
    if isinstance(names, (list, tuple, set)):
        return [str(item).strip() for item in names if str(item or "").strip()]
    text = str(names or "").strip()
    return [text] if text else []


def _name_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").lower())


def _contrast_order(value: Any) -> List[str]:
    """The arms of a contrast label in the order the label itself states them."""
    text = str(value or "").strip()
    if not text:
        return []
    return [part for part in (_name_key(item) for item in re.split(r"(?i)_?\s*vs\s*_?", text)) if part]


def _contrast_tokens(value: Any) -> Optional[frozenset]:
    """Order-insensitive identity of an `A_vs_B` label: GMP_vs_HSC and HSC_vs_GMP both give {gmp,hsc}."""
    parts = _contrast_order(value)
    return frozenset(parts) if parts else None


def _reversed_arms(file_key: Any, display: Any) -> bool:
    """True when the file and the report name the same two arms in the opposite order."""
    first, second = _contrast_order(file_key), _contrast_order(display)
    if not first or not second or len(first) != len(second):
        return False
    return first != second and sorted(first) == sorted(second)


def _match_contrast_key(container: Any, names: List[str]) -> Tuple[Optional[str], bool]:
    """The container key that denotes one of `names`; the flag says only the token set matched.

    R21/F2: the overlap file is keyed by the name the enrichment step used (GMP_vs_HSC) while the
    report prints the opposite orientation (HSC_vs_GMP). Name matching alone finds nothing there, and
    "upregulated" must then stay bound to the arm order of the file, never to the display order.
    """
    if not isinstance(container, dict):
        return None, False
    exact = {str(key).strip().lower(): key for key in container}
    for name in names:
        key = exact.get(str(name or "").strip().lower())
        if key is not None:
            return key, False
    wanted = {tokens for tokens in (_contrast_tokens(name) for name in names) if tokens}
    for key in container:
        tokens = _contrast_tokens(key)
        if tokens and tokens in wanted:
            return key, True
    return None, False


def _direction_node(node: Any) -> Dict[str, Any]:
    """A direction node maps a direction to its terms; anything else is a level above it."""
    if not isinstance(node, dict):
        return {}
    return {str(key): value for key, value in node.items() if isinstance(value, dict)}


def _direction_layer(db_node: Any, names: List[str]) -> Tuple[Optional[str], bool, Dict[str, Any]]:
    """One database node -> (contrast key used by the file, reversed, direction buckets).

    Shapes seen in staged runs: db -> direction, db -> contrast -> direction, and
    db -> contrast -> contrast -> direction (the file names the contrast at both levels).
    """
    node = _direction_node(db_node)
    if not node:
        return None, False, {}
    if any(key in DIRECTION_KEYS for key in node):
        return None, False, node
    key, reversed_name = _match_contrast_key(node, names)
    if key is None and len(node) == 1:
        key, reversed_name = next(iter(node)), False
    inner = node.get(key) if key is not None else None
    if isinstance(inner, dict):
        inner_node = _direction_node(inner)
        if not inner_node:
            return key, reversed_name, {}
        if any(item in DIRECTION_KEYS for item in inner_node):
            return key, reversed_name, inner_node
        inner_key, inner_reversed = _match_contrast_key(inner_node, names)
        if inner_key is None and len(inner_node) == 1:
            inner_key, inner_reversed = next(iter(inner_node)), False
        deeper = inner_node.get(inner_key) if inner_key is not None else None
        if isinstance(deeper, dict):
            return (inner_key, reversed_name or inner_reversed, _direction_node(deeper))
    return key, reversed_name, {}


def _enrichment_view(data: Dict[str, Any], names: Any) -> Dict[str, Any]:
    """The one resolver for every overlap-file shape, shared by the request index and the report.

    R21/F2: the request fact list and the report derive their numbers here together, so the model
    cannot be told "no entries" while the assembler prints entries for the same file (or the other
    way round). Returns the buckets plus how the file was read, including whether the key that
    matched carries the opposite arm order to the report's display contrast.
    """
    wanted = _as_names(names)
    view: Dict[str, Any] = {"buckets": {db: {} for db in ENRICHMENT_DATABASES},
                            "status": t("assembly.status_not_executed"), "shape": "", "matched_key": "",
                            "reversed": False, "terms": 0}
    if not isinstance(data, dict) or not data:
        return view
    layers = []
    if any(isinstance(data.get(db), dict) for db in ENRICHMENT_DATABASES):
        layers.append(("", data))
    for key, value in data.items():
        if isinstance(value, dict) and any(isinstance(value.get(db), dict) for db in ENRICHMENT_DATABASES):
            layers.append((str(key), value))
    if not layers:
        view["status"] = t("assembly.enrichment_status_unparsable")
        return view
    chosen = None
    for label, layer in layers:
        for database in ENRICHMENT_DATABASES:
            key, reversed_name, node = _direction_layer(layer.get(database), wanted)
            if node:
                chosen = (label, layer, key, reversed_name)
                break
        if chosen:
            break
    if chosen is None:
        label, layer = layers[0]
        view["shape"] = label or "db-first"
        for database in ENRICHMENT_DATABASES:
            view["buckets"][database] = _direction_layer(layer.get(database), wanted)[2]
        view["status"] = t("assembly.enrichment_status_no_entry") if any(view["buckets"].values()) else t("assembly.enrichment_status_valid_empty")
        return view
    label, layer, key, reversed_name = chosen
    for database in ENRICHMENT_DATABASES:
        view["buckets"][database] = _direction_layer(layer.get(database), wanted)[2]
    view["terms"] = sum(len(node.get(direction) or {})
                        for node in view["buckets"].values() for direction in node)
    # R21/F2: "reversed" is a property of the two labels, not of the matching path - the block can
    # also be found through its exact alias while still stating the arms the other way round.
    matched_tokens = _contrast_tokens(key or "")
    display = ""
    for name in wanted:
        if matched_tokens and _contrast_tokens(name) == matched_tokens:
            display = name
            break
    if not display:
        display = wanted[0] if wanted else ""
    view.update({"shape": label or "db-first", "matched_key": key or "",
                 "display_name": display,
                 "reversed": bool(reversed_name) or _reversed_arms(key or "", display),
                 "status": t("assembly.status_computed") if view["terms"] else t("assembly.enrichment_status_valid_empty")})
    return view


def _enrichment_buckets(data: Dict[str, Any], contrast: Any) -> Dict[str, Any]:
    """Back-compatible per-database buckets for one contrast (R21/F2)."""
    return _enrichment_view(data, contrast).get("buckets") or {db: {} for db in ENRICHMENT_DATABASES}


def _enrichment_hint_lines(run: Path, contrast: Any, limit: int = 6,
                           index: Optional[Dict[str, Any]] = None) -> List[str]:
    """Exploratory pathway hint: set overlap only, no FDR/q, therefore never a significance claim."""
    state, data = _overlap_state(run)
    if state != "ok":
        # R21/F2: nothing is printed here for the other three states; the boundary section states
        # which of them applies instead of the report claiming the analysis produced no entry.
        return []
    view = _enrichment_view(data, contrast)
    buckets = view.get("buckets") or {}
    # R17/R21: the overlap file is keyed by whichever name the enrichment step used, and that name
    # states which arm "upregulated" means, so the label follows the file instead of guessing.
    names = _as_names(contrast)
    key_name = view.get("matched_key") or (names[0] if names else "")
    arms = _arm_labels_for(key_name, index)
    up_label = t("assembly.enrichment_up_label") % arms[0] if len(arms) == 2 else t("assembly.enrichment_up_direction")
    down_label = t("assembly.enrichment_down_label") % arms[1] if len(arms) == 2 else t("assembly.enrichment_down_direction")
    rows = []
    for database in ENRICHMENT_DATABASES:
        node = buckets.get(database) or {}
        for direction, label in (("upregulated", up_label), ("downregulated", down_label)):
            terms = [(str(term), str(genes).split("/"))
                     for genes, term in (node.get(direction) or {}).items()]
            # deterministic pick: widest seed overlap first, ties broken by term name
            terms.sort(key=lambda item: (-len(item[1]), item[0]))
            for term, members in terms[:1]:
                rows.append((database, label, term, len(members), members))
    if not rows:
        return []
    lines = [t("assembly.heading_enrichment_hint")]
    lines += _note_once(
        "enrichment_hint",
        t("assembly.enrichment_hint_note_long"),
        t("assembly.enrichment_hint_note_short"))
    if view.get("reversed"):
        # R21/F2: same comparison, opposite arm order between the overlap file and the report table.
        # The labels above follow the file, so the reader has to be told which order they follow.
        lines += [t("assembly.enrichment_hint_reversed") % (_cell(key_name), _cell(arms[0] if arms else ""), _cell(names[0] if names else ""))]
    lines += ["",
              t("assembly.header_enrichment_hint"),
              "|---|---|---|---:|---|"]
    for database, label, term, count, members in rows[:limit]:
        lines.append("| %s | %s | %s | %d | %s |" % (_cell(database), _cell(label), _cell(term),
                                                               count, _cell(", ".join(members[:6]))))
    lines.append("")
    return lines

DENSITY_CAP = 4


def _matrix_aliases() -> Tuple[str, ...]:
    """Natural alternatives used when one term would otherwise repeat too often."""
    return (
        (t("assembly.source_current_matrix"), t("assembly.matrix_alias_total_abundance"), t("assembly.matrix_alias_this_data"))
    )


def cap_term_density(text: str, term: str, cap: int = DENSITY_CAP, state: Optional[Dict[str, int]] = None) -> str:
    """Keep the first `cap` uses of a legitimate term; rephrase the rest with natural alternatives."""
    counter = state if state is not None else {"n": 0}
    out, pos = [], 0
    for match in re.finditer(re.escape(term), text):
        out.append(text[pos:match.start()])
        counter["n"] = counter.get("n", 0) + 1
        if counter["n"] <= cap:
            out.append(term)
        else:
            out.append(_matrix_aliases()[(counter["n"] - cap - 1) % len(_matrix_aliases())])
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def _reader_asset_label(name: str) -> str:
    """Reader-facing name for an evidence asset, derived from the existing label map."""
    differential = re.match(r"differential_(.+)\.csv$", name)
    if differential:
        return t("assembly.asset_differential") % differential.group(1)
    if name.startswith("candidate_exclusion_trace"):
        return t("assembly.asset_candidate_exclusion")
    if name.startswith("contrast_concordance"):
        return t("assembly.asset_contrast_concordance")
    if name.startswith("stratified_contrast_summary"):
        return t("assembly.asset_stratified_summary")
    for pattern, label in _file_labels():
        if re.search(pattern, name):
            return label
    return name.replace(".csv", "")


def _stratified_lines(run):
    """R3: stratified contrast summary straight from evaluation_evidence_ext."""
    import csv as _csv
    out = []
    path = Path(run) / "evaluation_evidence_ext" / "stratified_contrast_summary.csv"
    if not path.exists():
        return out
    try:
        with path.open(encoding="utf-8-sig") as fh:
            rows = list(_csv.DictReader(fh))
    except Exception:
        return out
    named = [r for r in rows if str(r.get("level") or "").strip()]
    if not named:
        return out
    out += [t("assembly.heading_stratified_summary"), ""]
    out.append(t("assembly.header_stratified"))
    out.append("|---|---|---|---:|---:|---:|---:|---|")
    for row in named[:10]:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            _cell(row.get("contrast")), _cell(row.get("stratifier")), _table_cell(row.get("level"), 120),
            _fmt(row.get("n_a")), _fmt(row.get("n_b")), _fmt(row.get("tested_proteins")),
            _fmt(row.get("n_sig")), _table_cell(row.get("verdict"), 80)))
    out += ["", t("assembly.stratified_scope_note"), ""]
    return out

DESIGN_TOKENS = {"sampleinfo", "filename", "batch", "cluster", "label", "type1", "type2",
                 "intact", "permeable", "fresh", "frozen", "pg", "hela", "bortezomib",
                 "cycloheximide", "control", "dmso", "lps"}


def _design_tokens(run: Path) -> set:
    """Design and metadata labels that are never protein candidates.

    Harvested from the run's own evidence-table headers and analysis design, plus a frozen list of
    the group labels used by the three staged datasets; the trace file records exactly these as
    unresolved symbols rather than as excluded proteins.
    """
    import csv as _csv
    tokens = set(DESIGN_TOKENS)
    try:
        parameters = read_json(Path(run) / "parameters.json", {}) or {}
    except Exception:  # noqa: BLE001
        parameters = {}

    def harvest(node):
        if isinstance(node, dict):
            for value in node.values():
                harvest(value)
        elif isinstance(node, list):
            for value in node:
                harvest(value)
        elif isinstance(node, str):
            name = node.strip().lower()
            if name and len(name) <= 24 and "\\" not in name and "/" not in name:
                tokens.add(name)

    harvest(parameters.get("analysis_design") or {})
    for pattern in ("evaluation_evidence/*.csv", "evaluation_evidence_ext/*.csv"):
        for path in sorted(Path(run).glob(pattern)):
            try:
                with path.open(encoding="utf-8-sig") as handle:
                    header = next(_csv.reader(handle), [])
            except Exception:  # noqa: BLE001
                continue
            for cell in header:
                name = str(cell).strip().lower()
                if name:
                    tokens.add(name)
    return tokens


def _exclusion_lines(run, limit=6, seen=None):
    """Candidate exclusion trace: only rows the source table records as absent at a stage."""
    import csv as _csv
    from collections import Counter
    if seen is not None and seen.get('done'):
        # the trace has no contrast column: it is report-level evidence and is shown once
        return []
    path = Path(run) / "evaluation_evidence_ext" / "candidate_exclusion_trace.csv"
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(_csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        return []
    tokens = _design_tokens(Path(run))
    counted = Counter()
    for row in rows:
        name = str(row.get("candidate") or "").strip()
        status = str(row.get("status") or "").strip().lower()
        if not name or status != "absent" or name.lower() in tokens:
            continue
        counted[(name, str(row.get("reason") or "").strip()[:60])] += 1
    out = [t("assembly.heading_exclusion_trace"), "",
           t("assembly.exclusion_trace_intro"), ""]
    if seen is not None:
        seen['done'] = True
    if not counted:
        out += [t("assembly.exclusion_trace_no_protein_level"), ""]
        return out
    out += [t("assembly.header_exclusion_trace"), "|---|---|---:|"]
    for (name, reason), count in counted.most_common(limit):
        out.append("| %s | %s | %d |" % (_cell(name), _cell(reason), count))
    out += ["", t("assembly.exclusion_count_note"), ""]
    return out


def _ext_csv_rows(run, name):
    """Rows of one evaluation_evidence_ext CSV (empty list when the file is absent)."""
    import csv as _csv
    path = Path(run) / "evaluation_evidence_ext" / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(_csv.DictReader(handle))


def _reader_prob(value):
    try:
        number = float(value)
    except Exception:
        return _cell(value)
    if number == 0:
        return t("assembly.prob_less_than")
    return "%.3g" % number


def _passes_cutoff(value, cutoff: float = 0.05) -> bool:
    """True only when the recorded statistic parses and is at or below the cutoff.

    A missing or non-numeric value means the row cannot support the claim, so it is not
    counted; an unparsable value must never raise and drop the whole request block.
    """
    try:
        return float(str(value).strip()) <= cutoff
    except (TypeError, ValueError):
        return False


def _reader_note(raw: Any) -> str:
    """Reader-facing wording for a machine-generated run note; the outcome is never softened.

    The pipeline writes short English diagnostics into the evidence CSVs (for example a skipped
    test). They are translated here instead of being printed verbatim, and a skipped test stays a
    skipped test.
    """
    text = str(raw or "").strip()
    match = re.fullmatch(r"query size (\d+) < (\d+): no test performed", text)
    if match:
        return (t("assembly.reader_note_too_few") % (match.group(1), match.group(2)))
    if re.fullmatch(r"q<=0\.05", text):
        return t("assembly.reader_note_no_testable_entry")
    return _cell(text)


def _dose_direction_labels() -> Dict[str, str]:
    """Recorded dose-direction token -> reader wording."""
    return (
        {"falling with dose": t("assembly.dose_falling"), "rising with dose": t("assembly.dose_rising"),
        "flat": t("assembly.dose_no_monotonic_trend"), "no trend": t("assembly.dose_no_monotonic_trend"), "flat with dose": t("assembly.dose_flat_with_dose")}
    )


def _dose_direction_text(raw: Any, rho: Any = None) -> str:
    """Dose direction; the sign of rho decides it, so a zero rho cannot read as a trend.

    R17: the source table labels rho=0 as a falling trend; the reader text follows the statistic.
    """
    try:
        value = float(rho)
    except (TypeError, ValueError):
        value = None
    if value is not None:
        if value > 0:
            return t("assembly.dose_rising")
        if value < 0:
            return t("assembly.dose_falling")
        return t("assembly.dose_flat_zero_rho")
    text = str(raw or "").strip()
    return _dose_direction_labels().get(text.lower(), text or t("assembly.direction_not_recorded"))


def _dose_step_text(raw):
    text = str(raw or "").strip()
    if not text.startswith("{"):
        return text
    try:
        import ast as _ast
        mapping = _ast.literal_eval(text)
    except Exception:
        return text
    return "；".join("%s µM n=%s" % (key, value) for key, value in sorted(mapping.items()))


def _dose_trend_lines(run):
    """Real dose-gradient statistics; skipped strata are named as insufficient, never blurred."""
    rows = _ext_csv_rows(run, "dose_trend.csv")
    if not rows:
        return []
    modules = [row for row in rows if str(row.get("level")) == "module"]
    proteins = [row for row in rows if str(row.get("level")) == "protein"]
    skipped = [row for row in rows if str(row.get("note") or "").strip()]
    out = [t("assembly.heading_dose_trend"), ""]
    if modules:
        out += [t("assembly.header_dose_module_steps"),
                "|---|---|---|---:|---|---:|---:|---:|---|"]
        for row in modules:
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _cell(row.get("drug")), _cell(row.get("stratum")), _cell(row.get("name")),
                _fmt(row.get("n_samples")), _cell(_dose_step_text(row.get("dose_steps"))),
                _fmt(row.get("rho")), _reader_prob(row.get("p_value")), _reader_prob(row.get("q_value")),
                _cell(_dose_direction_text(row.get("direction"), row.get("rho")))))
        out.append("")
        flipped = {}
        for row in modules:
            try:
                rho = float(row.get("rho"))
            except Exception:
                continue
            key = str(row.get("drug")), str(row.get("name"))
            flipped.setdefault(key, {})[str(row.get("stratum"))] = (rho, row)
        for (drug, name), per_stratum in flipped.items():
            signs = {1 if value[0] > 0 else -1 for value in per_stratum.values() if value[0] != 0}
            if len(signs) > 1:
                detail = "；".join("%s ρ=%s, p=%s, n=%s" % (stratum, _fmt(value[1].get("rho")),
                                                          _reader_prob(value[1].get("p_value")),
                                                          _fmt(value[1].get("n_samples")))
                                  for stratum, value in sorted(per_stratum.items()))
                out += _note_once(
                    "dose_flip",
                    t("assembly.dose_flip_note_long")
                    % (drug, name, detail),
                    t("assembly.dose_flip_note_short") % (drug, name, detail))
                out.append("")
    if proteins:
        significant = [row for row in proteins if str(row.get("significant")) == "yes"]
        usable = [row for row in proteins if str(row.get("rho") or "").strip()]
        out.append(t("assembly.dose_protein_summary") % (len(proteins), len(significant)))
        top = sorted(usable, key=lambda row: (float(row.get("q_value") or 1),
                                              -abs(float(row.get("rho") or 0))))[:6]
        if top:
            out.append("")
            out += [t("assembly.header_dose_protein"), "|---|---|---|---:|---:|---:|---:|"]
            for row in top:
                out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                    _cell(row.get("name")), _cell(row.get("drug")), _cell(row.get("stratum")),
                    _fmt(row.get("n_samples")), _fmt(row.get("rho")), _reader_prob(row.get("p_value")),
                    _reader_prob(row.get("q_value"))))
        out.append("")
    for row in skipped:
        out.append(t("assembly.note_drug_stratum") % (_cell(row.get("drug")), _cell(row.get("stratum")),
                                      _reader_note(row.get("note"))))
        out.append("")
    out.append(t("assembly.dose_scope_note"))
    out.append("")
    return out


def _size_range(values):
    """Compact n=... range over integer-like values; empty string when nothing is numeric."""
    numbers = []
    for value in values:
        try:
            numbers.append(int(float(value)))
        except Exception:  # noqa: BLE001
            continue
    if not numbers:
        return ""
    low, high = min(numbers), max(numbers)
    return "%d" % low if low == high else "%d\u2013%d" % (low, high)


def _fallback_significant_sizes(rows):
    """The significant-set size recorded inside each fallback query definition."""
    found = []
    for row in rows:
        match = re.search(r"significant set had (\d+) genes", str(row.get("query_definition") or ""))
        if match:
            found.append(match.group(1))
    return found


def _query_rule_text(definition):
    """Reader-facing form of the measured screening rule behind a significant-set query."""
    match = re.search(r"adj\.P<=\s*([0-9.]+).*?\|logFC\|>=\s*([0-9.]+)", str(definition or ""))
    if match:
        return t("assembly.threshold_design") % (match.group(1), match.group(2))
    return _cell(definition)


def _direction_labels() -> Dict[str, str]:
    """Internal direction token -> reader wording for the enrichment tables."""
    return (
        {"upregulated": t("assembly.direction_upregulated"), "downregulated": t("assembly.direction_downregulated")}
    )


def _is_fallback_query(row: Dict[str, Any]) -> bool:
    """The fallback query (top |logFC| share) exists only where the significant set was too small."""
    return str(row.get("query_definition") or "").startswith("fallback")


def _fallback_contrast_text(rows, index) -> str:
    """Name every contrast that used the fallback query, with its own significant-set size."""
    parts = []
    for name in sorted({str(row.get("contrast") or "") for row in rows}):
        subset = [row for row in rows if str(row.get("contrast") or "") == name]
        sizes = _size_range(_fallback_significant_sizes(subset)) or t("assembly.not_recorded")
        query_sizes = _size_range([row.get("query_size") for row in subset]) or t("assembly.not_recorded")
        arms = _arm_labels_for(name, index)
        label = name
        if len(arms) == 2:
            label = t("assembly.fallback_contrast_label") % (name, arms[0], arms[1])
        parts.append(t("assembly.fallback_contrast_item") % (label, sizes, query_sizes))
    return "；".join(parts)


def _ora_query_scope_lines(tested, shown, index) -> List[str]:
    """R21/F1: the query-set note states the scope the printed table actually came from.

    The fallback query belongs to single contrasts (one whose significant set was too small), so
    turning one contrast's "0 significant proteins" into a dataset-wide fact contradicted the same
    report's own screening counts. When both kinds of query are in play they are described
    separately, and the note never speaks for a scope the table does not cover.
    """
    fallback_rows = [row for row in tested if _is_fallback_query(row)]
    shown_fallback = [row for row in shown if _is_fallback_query(row)]
    shown_significant = [row for row in shown if not _is_fallback_query(row)]
    rule = next((_query_rule_text(row.get("query_definition")) for row in tested
                 if not _is_fallback_query(row)), "")
    if not fallback_rows:
        per_direction = {}
        for row in tested:
            arms = _arm_labels_for(row.get("contrast"), index)
            label = _direction_labels().get(str(row.get("direction")), _cell(row.get("direction")))
            if len(arms) == 2:
                label = t("assembly.arm_higher") % (arms[0] if str(row.get("direction")) == "upregulated"
                                    else arms[1])
            per_direction.setdefault(label, set()).add(str(row.get("query_size")))
        sizes = "、".join("%s n=%s" % (label, "、".join(sorted(values)))
                          for label, values in sorted(per_direction.items()))
        return [t("assembly.ora_scope_significant") % (rule, sizes)]
    if len(fallback_rows) == len(tested):
        # every contrast in this file fell back, so the dataset-wide wording is the accurate one
        lead = (t("assembly.ora_scope_fallback_lead")
                % (_size_range([row.get("query_size") for row in shown]),
                   _size_range([row.get("query_size") for row in tested])))
        significant_sizes = _size_range(_fallback_significant_sizes(fallback_rows))
        if significant_sizes:
            lead += t("assembly.ora_fallback_reason_count") % significant_sizes
        else:
            lead += t("assembly.ora_fallback_reason_too_few")
        lead += t("assembly.query_scope_single")
        return [lead]
    # both kinds of query are in play: describe each scope instead of generalising one of them
    lines = []
    if shown_significant:
        lines.append(t("assembly.ora_mixed_scope_lead")
                     % (rule, _size_range([row.get("query_size") for row in shown_significant])
                        or t("assembly.not_recorded")))
    else:
        lines.append(t("assembly.ora_mixed_scope_all_fallback"))
    scope_rows = shown_fallback or fallback_rows
    lines.append(t("assembly.fallback_query_scope")
                 % (_fallback_contrast_text(scope_rows, index),
                    _size_range([row.get("query_size") for row in scope_rows]) or t("assembly.not_recorded")))
    lines.append(t("assembly.query_scope_two_classes"))
    return lines


def _enrichment_ora_lines(run, index: Optional[Dict[str, Any]] = None):
    """Hypergeometric ORA with the detected-protein background; the query definition leads the section."""
    rows = _ext_csv_rows(run, "enrichment_ora.csv")
    if not rows:
        return []
    tested = [row for row in rows if str(row.get("p_value") or "").strip()]
    skipped = [row for row in rows if not str(row.get("p_value") or "").strip()]
    out = [t("assembly.heading_ora_background"), ""]
    if tested:
        shown = sorted(tested, key=lambda item: float(item.get("p_value") or 1))[:12]
        out += _ora_query_scope_lines(tested, shown, index)
        out.append("")
        out += [t("assembly.header_ora"),
                "|---|---|---|---|---:|---:|---:|---:|---:|"]
        for row in shown:
            # R17: name the arm the direction belongs to instead of a bare up/down label
            arms = _arm_labels_for(row.get("contrast"), index)
            direction_label = _direction_labels().get(str(row.get("direction")), row.get("direction"))
            if len(arms) == 2:
                direction_label = t("assembly.arm_higher") % (arms[0] if str(row.get("direction")) == "upregulated"
                                              else arms[1])
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _cell(row.get("contrast")),
                _cell(direction_label),
                _cell(row.get("namespace")),
                _cell(row.get("term")), _fmt(row.get("hits")), _fmt(row.get("term_size_in_background")),
                _fmt(row.get("query_size")), _reader_prob(row.get("p_value")), _reader_prob(row.get("q_value"))))
        background = _fmt(tested[0].get("background_size"))
        note = (t("assembly.ora_background_note") % background)
        # R17: the background set and the tested matrix are different counts, and the report says so
        # instead of leaving two numbers that look like a contradiction.
        matrix_proteins = _first_record(read_json(Path(run) / "processed_proteins" / "limma_summary.json",
                                                  {}) or {}).get("n_proteins")
        if matrix_proteins is not None and str(_fmt(matrix_proteins)) != str(background):
            note += (t("assembly.ora_background_vs_matrix_note") % _fmt(matrix_proteins))
        out += ["", note, ""]
    for row in skipped[:3]:
        arms = _arm_labels_for(row.get("contrast"), index)
        direction_label = _direction_labels().get(str(row.get("direction")), row.get("direction"))
        if len(arms) == 2:
            direction_label = t("assembly.arm_higher") % (arms[0] if str(row.get("direction")) == "upregulated"
                                          else arms[1])
        out.append(t("assembly.note_drug_stratum") % (_cell(row.get("contrast")), _cell(direction_label),
                                      _reader_note(row.get("note"))))
        out.append("")
    return out


def _concordance_lines(run, limit=None):
    """R3: agreement between contrasts."""
    import csv as _csv
    path = Path(run) / "evaluation_evidence_ext" / "contrast_concordance.csv"
    if not path.exists():
        return [t("assembly.heading_contrast_concordance"), "",
                t("assembly.concordance_absent"),
                t("assembly.concordance_gap_note"), ""]
    try:
        with path.open(encoding="utf-8-sig") as fh:
            rows = list(_csv.DictReader(fh))
    except Exception:
        return []
    if not rows:
        return []
    out = [t("assembly.heading_contrast_concordance"), "",
           t("assembly.header_concordance"),
           "|---|---|---:|---:|---:|---:|"]
    shown = rows if limit is None else rows[:limit]
    for row in shown:
        out.append("| %s | %s | %s | %s | %s | %s |" % (
            _cell(row.get("contrast_a")), _cell(row.get("contrast_b")), _cell(row.get("shared_proteins")),
            _fmt(row.get("spearman_logfc")), _fmt(row.get("overlap_sig")), _fmt(row.get("fisher_p"))))
    coverage = (t("assembly.concordance_coverage_all") % len(rows)) if len(shown) == len(rows) else (
        t("assembly.concordance_coverage_capped") % (len(shown), len(rows)))
    out += ["", coverage, t("assembly.concordance_correlation_note"), ""]
    return out


def _enrichment_boundary_lines(run, has_terms):
    """R3/R21: the reader-facing status of the overlap step, kept apart per state.

    R21/F2: an unparsable file, a file that is not there, an executed analysis with no entry above
    the configured floor and a file whose entries belong to other contrasts are four different
    facts. Printing the last three as "no usable entry produced" told the reader the analysis ran
    and came back empty when the file in fact held entries for other contrasts.
    """
    if has_terms:
        return []
    state, _payload = _overlap_state(run)
    if state == "missing":
        body = (t("assembly.enrichment_boundary_missing"))
    elif state == "unparsable":
        body = (t("assembly.enrichment_boundary_unparsable"))
    elif state == "empty":
        body = (t("assembly.enrichment_boundary_empty"))
    else:
        body = (t("assembly.enrichment_boundary_other_contrasts"))
    return [t("assembly.heading_enrichment_boundary"), "",
            body, ""]


def _confidence_labels() -> Dict[str, str]:
    """Recorded confidence level -> reader wording."""
    return (
        {"high": t("assembly.confidence_high"), "moderate": t("assembly.confidence_moderate"), "low": t("assembly.confidence_low")}
    )


def _evidence_source_labels() -> Dict[str, str]:
    """Recorded evidence-source token -> reader wording."""
    return (
        {"current_matrix": t("assembly.source_current_matrix"), "analyzed_matrix": t("assembly.source_current_matrix"),
        "dataset_provided": t("assembly.source_dataset_provided"), "user_provided": t("assembly.source_user_provided"),
        "auto_inferred": t("assembly.source_auto_inferred"), "inferred": t("assembly.source_auto_inferred"),
        "offline_enrichment": t("assembly.source_offline_enrichment"), "external_annotation": t("assembly.external_annotation"),
        "analysis_design": t("assembly.analysis_design")}
    )


def _evidence_source_label(value: Any) -> str:
    text = str(value or "").strip()
    return _evidence_source_labels().get(text, text or t("assembly.not_recorded"))


def _confidence_label(value: Any) -> str:
    text = str(value or "").strip()
    return _confidence_labels().get(text.lower(), text or t("assembly.not_recorded"))


def _direction_phrase(direction: Any, groups) -> str:
    """Translate the internal direction token into the reader-facing group names."""
    text = str(direction or "").strip()
    first = _cell(groups[0]) if groups and groups[0] else t("assembly.arm_first")
    second = _cell(groups[1]) if groups and len(groups) > 1 and groups[1] else t("assembly.arm_second")
    if text.startswith("up_in_group_a"):
        return t("assembly.arm_higher_in") % first
    if text.startswith("down_in_group_a"):
        return t("assembly.arm_higher_in") % second
    return text or t("assembly.direction_not_recorded")


def _arm_labels_for(name: Any, index: Optional[Dict[str, Any]] = None) -> List[str]:
    """The two arms of a table, read from the table's own name when the name carries them.

    R17: a source-table name such as Intact_vs_Permeable states its own direction; keeping that
    label makes the counts correct even when the report body shows the same comparison the other
    way round. Falls back to the frozen evidence index, then to nothing.
    """
    text = str(name or "").strip()
    if "_vs_" in text:
        first, second = text.split("_vs_", 1)
        if first and second:
            return [_cell(first), _cell(second)]
    for rec in (index or {}).get("contrasts") or []:
        if text in (str(rec.get("display_contrast")), str(rec.get("source_contrast")),
                    str(rec.get("claim_id"))):
            groups = rec.get("groups") or []
            if len(groups) >= 2 and groups[0] and groups[1]:
                return [_cell(groups[0]), _cell(groups[1])]
    return []


def _read_evidence_rows(path: Path) -> List[Dict[str, Any]]:
    import csv as _csv
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return list(_csv.DictReader(handle))
    except Exception:  # noqa: BLE001
        return []


def _figure_title_map() -> Dict[str, str]:
    """Plot type -> the title the report prints for that figure."""
    return {
        "qc_sample_overview": t("assembly.figure_qc_sample_title"),
        "pca": t("assembly.figure_qc_title"),
        "umap": t("assembly.figure_umap_title"),
        "heatmap": t("assembly.figure_heatmap_title"),
        "differential_summary_barplot": t("assembly.figure_differential_bar_title"),
        "mechanism_enrichment_dotplot": t("assembly.figure_enrichment_dotplot_title"),
        "mechanism_top_protein_group_means": t("assembly.figure_group_means_title"),
        "protein_contrast_bubble": t("assembly.figure_bubble_title"),
        "key_protein_overview": t("assembly.figure_key_protein_title"),
    }


def _pca_numbers(run: Path):
    """PC1/PC2 explained variance and per-group means, recomputed from the matrix the figure uses."""
    try:
        import numpy as _np
    except Exception:  # noqa: BLE001
        return None
    try:
        matrix = Path(run) / "processed_proteins" / "ProteinQuant_ComBat.csv"
        info = Path(run) / "processed_proteins" / "SampleInfo_Filtered.csv"
        if not matrix.exists():
            return None
        import csv as _csv
        with matrix.open(encoding="utf-8-sig") as handle:
            reader = _csv.reader(handle)
            header = next(reader)
            rows = [row for row in reader if len(row) >= 3]
        samples = header[2:]
        values = []
        for row in rows:
            line = []
            for cell in row[2:]:
                try:
                    line.append(float(cell))
                except Exception:  # noqa: BLE001
                    line.append(_np.nan)
            values.append(line)
        data = _np.array(values, dtype=float)
        if data.size == 0 or data.shape[1] != len(samples):
            return None
        protein_mean = _np.nanmean(data, axis=1, keepdims=True)
        data = _np.where(_np.isnan(data), protein_mean, data)
        data = data - data.mean(axis=0, keepdims=True)
        u, s, _vt = _np.linalg.svd(data.T, full_matrices=False)
        variance = (s ** 2) / float((s ** 2).sum())
        scores = u * s
        groups = {}
        if info.exists():
            with info.open(encoding="utf-8-sig") as handle:
                for row in _csv.DictReader(handle):
                    groups[str(row.get("FileName") or "")] = str(row.get("Cluster") or "")
        per_group = {}
        for index, name in enumerate(samples):
            key = groups.get(name) or groups.get(name.replace(".raw", "")) or t("assembly.pca_ungrouped")
            per_group.setdefault(key, []).append(float(scores[index, 0]))
        return {"pc1": float(variance[0]) * 100.0, "pc2": float(variance[1]) * 100.0,
                "group_pc1": {key: float(_np.mean(val)) for key, val in per_group.items()}}
    except Exception:  # noqa: BLE001
        return None


def _figure_reference_lines(run: Path, index: Dict[str, Any]) -> List[str]:
    """Figure captions inside the narrative: number, title and a numeric conclusion per figure."""
    manifest = read_json(Path(run) / "visualize_results" / "figure_manifest.json", {}) or {}
    records = manifest.get("figures") if isinstance(manifest, dict) else None
    if not records:
        return []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(str(record.get("plot_type") or ""), []).append(record)
    params = (records[0].get('params') or {}) if records else {}
    thresholds = t("assembly.figure_threshold")
    top_n = _fmt(params.get('top_n_proteins'))
    lines: List[str] = []
    number = 0
    order = ["qc_sample_overview", "pca", "umap", "heatmap",
             "differential_summary_barplot", "mechanism_enrichment_dotplot",
             "mechanism_top_protein_group_means", "protein_contrast_bubble",
             "key_protein_overview"]
    for plot_type in order:
        grouped = by_type.get(plot_type) or []
        if not grouped:
            continue
        number += 1
        meta = grouped[0].get("caption_metadata") or {}
        title = _figure_title_map().get(plot_type, plot_type)
        if plot_type == "qc_sample_overview":
            conclusion = (t("assembly.figure_qc_conclusion")
                          % (_fmt(meta.get("n_proteins")), _fmt(meta.get("n_unannotated"))))
        elif plot_type == "pca":
            pca = _pca_numbers(Path(run))
            if pca:
                ranked = sorted(pca["group_pc1"].items(), key=lambda item: item[1])
                conclusion = (t("assembly.figure_pca_conclusion")
                              % (pca["pc1"], pca["pc2"], ranked[0][0], ranked[0][1],
                                 ranked[-1][0], ranked[-1][1]))
            else:
                conclusion = t("assembly.figure_pca_conclusion_missing")
        elif plot_type == "umap":
            conclusion = t("assembly.figure_umap_conclusion")
        elif plot_type == "heatmap":
            conclusion = t("assembly.figure_heatmap_conclusion") % (top_n, thresholds)
        elif plot_type == "differential_summary_barplot":
            # R17: the bar counts follow each source table's own sign, so the figure says which arm
            # "up" means; without this the figure and the body read as a contradiction.
            inverted = [rec for rec in index["contrasts"] if rec.get("inverted")]
            direction_note = ""
            if len(inverted) == 1:
                arms = _arm_labels_for(inverted[0].get("source_contrast"), index)
                if len(arms) == 2:
                    direction_note = (t("assembly.figure_bar_direction_note_single")
                                      % (inverted[0].get("source_contrast"), arms[0], arms[1],
                                         inverted[0].get("display_contrast")))
            elif inverted:
                direction_note = t("assembly.figure_bar_direction_note_plural")
            conclusion = (t("assembly.figure_bar_conclusion")
                          % (_fmt(meta.get("n_contrasts")), _fmt(meta.get("n_sig")),
                             _fmt(meta.get("n_up")), _fmt(meta.get("n_down")), thresholds,
                             direction_note))
        elif plot_type == "mechanism_enrichment_dotplot":
            conclusion = t("assembly.figure_enrichment_dotplot_conclusion")
        elif plot_type == "mechanism_top_protein_group_means":
            conclusion = t("assembly.figure_group_means_conclusion")
        elif plot_type == "protein_contrast_bubble":
            conclusion = t("assembly.figure_bubble_conclusion")
        else:
            conclusion = t("assembly.figure_key_protein_conclusion")
        lines.append(t("assembly.figure_line") % (number, title, conclusion))
    volcanos = by_type.get("volcano") or []
    if volcanos:
        if len(volcanos) == 1:
            lines.append(t("assembly.volcano_range_one") % (number + 1, thresholds))
        else:
            lines.append(t("assembly.volcano_range_many") % (number + 1, number + len(volcanos), thresholds))
        for record in volcanos:
            meta = record.get("caption_metadata") or {}
            contrast = str(record.get("file_name") or "").replace("_volcano_plot.png", "")
            arms = _arm_labels_for(contrast, index)
            if len(arms) == 2:
                lines.append(t("assembly.volcano_row_with_arms")
                             % (contrast, _fmt(meta.get("n_sig")), arms[0], _fmt(meta.get("n_up")),
                                arms[1], _fmt(meta.get("n_down"))))
            else:
                lines.append(t("assembly.volcano_row_plain")
                             % (contrast, _fmt(meta.get("n_sig")), _fmt(meta.get("n_up")),
                                _fmt(meta.get("n_down"))))
    return lines


def _evidence_table_lines(run: Path, index: Dict[str, Any]) -> List[str]:
    """Reader-facing tables built from the deterministic evidence CSVs (no CSV dump)."""
    folder = Path(run) / "evaluation_evidence"
    groups_of = {str(rec.get("display_contrast")): [rec.get("group_a"), rec.get("group_b")]
                 for rec in index["contrasts"]}

    def groups_for(contrast: Any):
        """Two arms of a contrast: from the frozen index when known, else from the name itself."""
        text = str(contrast or "").strip()
        if "_vs_" in text:
            first, second = text.split("_vs_", 1)
            if first and second:
                return [first, second]
        if text in groups_of:
            return [arm for arm in groups_of[text] if arm]
        return []

    out: List[str] = []

    contrast_rows = [row for row in _read_evidence_rows(folder / "core_story_evidence.csv")
                     if str(row.get("row_type") or "").strip() == "contrast"]
    if contrast_rows:
        out += [t("assembly.heading_contrast_evidence"), "",
                t("assembly.header_contrast_evidence"),
                "|---|---|---|---|---:|---:|---:|---|---|"]
        boundary = ""
        for row in contrast_rows:
            groups = groups_for(row.get("display_contrast"))
            first = _cell(row.get("group_a") or (groups[0] if groups else t("assembly.arm_first")))
            second = _cell(row.get("group_b") or (groups[1] if len(groups) > 1 else t("assembly.arm_second")))
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _cell(row.get("claim_title")), _cell(row.get("display_contrast")), first, second,
                _fmt(row.get("n_sig")), _fmt(row.get("n_up_display_group_a")),
                _fmt(row.get("n_down_display_group_a")),
                _cell(_evidence_source_label(row.get("evidence_source"))),
                _cell(_confidence_label(row.get("confidence")))))
            if not boundary and str(row.get("boundary") or "").strip():
                boundary = str(row.get("boundary")).strip()
        if boundary:
            out += ["", t("assembly.evidence_boundary_line") % boundary]
        out.append("")

    candidate_rows = _read_evidence_rows(folder / "candidate_protein_evidence.csv")
    if candidate_rows:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for row in candidate_rows:
            key = tuple(str(row.get(field) or "")
                        for field in ("candidate", "matched_protein", "contrast", "direction",
                                      "logFC", "P.Value", "adj.P.Val", "missing_rate"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        out += [t("assembly.heading_candidate_stats"), "",
                t("assembly.header_candidate_stats"),
                "|---|---|---|---|---:|---:|---:|---:|---|"]
        source_contrasts = set()
        for row in unique:
            groups = groups_for(row.get("contrast"))
            contrast_cell = _cell(row.get("contrast"))
            direction_cell = _cell(_direction_phrase(row.get("direction"), groups))
            # R17: internal presence enums become reader wording; the state itself is unchanged.
            if str(row.get("contrast") or "").strip() == "matrix_presence_only":
                contrast_cell = t("assembly.not_entered_statistical_test")
            if str(row.get("direction") or "").strip() == "not_detected_in_available_matrix":
                direction_cell = t("assembly.not_detected_in_matrix")
            if str(row.get("direction") or "").strip() == "symbol_not_matched_in_available_matrix":
                direction_cell = t("assembly.evidence_symbol_not_mapped")
            if "_vs_" in str(row.get("contrast") or ""):
                source_contrasts.add(str(row.get("contrast")))
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _cell(row.get("candidate")), _cell(row.get("matched_protein")),
                contrast_cell, direction_cell,
                _fmt(row.get("logFC")), _fmt(row.get("P.Value")), _fmt(row.get("adj.P.Val")),
                _fmt(row.get("missing_rate")),
                _cell(_evidence_source_label(row.get("evidence_source")))))
        note = t("assembly.candidate_rows_note")
        if len(unique) != len(candidate_rows):
            note += (t("assembly.candidate_fold_note")
                     % (len(unique), len(candidate_rows), len(candidate_rows) - len(unique)))
        note += t("assembly.candidate_screening_scope_note")
        out += ["", note, ""]
        display_names = {str(rec.get("display_contrast")) for rec in index["contrasts"]}
        source_map = {str(rec.get("source_contrast")): str(rec.get("display_contrast"))
                      for rec in index["contrasts"] if rec.get("source_contrast")}
        extras = sorted(name for name in source_contrasts if name not in display_names)
        flipped = [name for name in extras if name in source_map]
        others = [name for name in extras if name not in source_map]
        # R17 fix: only a contrast whose flip is printed elsewhere is a "reverse writing"; the other
        # extra contrasts are simply not part of the task list, and the note says which is which
        # instead of leaving a placeholder when the flip cannot be resolved.
        for source in flipped:
            arms = _arm_labels_for(source, index)
            out += _note_once(
                "source_direction_%s" % source,
                t("assembly.candidate_source_direction_long")
                % (source, arms[0] if arms else t("assembly.arm_first"), source_map[source]),
                t("assembly.candidate_source_direction_short") % source_map[source])
            out.append("")
        if others:
            out += _note_once(
                "other_contrasts_%s" % "_".join(others[:2]),
                t("assembly.candidate_other_contrasts_long") % "、".join(others),
                t("assembly.candidate_other_contrasts_short") % "、".join(others))
            out.append("")

    module_rows = _read_evidence_rows(folder / "curated_module_group_summary.csv")
    if module_rows:
        out += [t("assembly.heading_module_scores"), "",
                t("assembly.header_module_scores"),
                "|---|---|---:|---:|---:|---:|---|"]
        for row in module_rows:
            label = str(row.get("group") or "").replace("_minus_", " − ")
            members = str(row.get("matched_genes") or "")
            out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                _cell(row.get("module")), _cell(label), _fmt(row.get("n_samples")),
                _fmt(row.get("mean_score")), _fmt(row.get("median_score")),
                _fmt(row.get("n_matched_genes")), _cell(members[:60])))
        out += [""]
        out += _note_once(
            "module_no_fdr",
            t("assembly.module_no_fdr_long"),
            t("assembly.module_no_fdr_short"))
        out.append("")
    return out


def _asset_lines(run: Path, index: Dict[str, Any]) -> List[str]:
    # Reader-facing index: labels plus where to check the numbers. Internal file names stay out of
    # the reader-facing document; the delivered evidence bundle keeps them for reproduction.
    lines = [t("assembly.header_asset_pointer"), "|---|---|"]
    def _pointer_for(name: str) -> str:
        if "candidate_exclusion_trace" in name:
            return t("assembly.pointer_exclusion_trace")
        if "contrast_concordance" in name:
            return t("assembly.pointer_concordance")
        if "stratified" in name:
            return t("assembly.pointer_stratified")
        if name.endswith(".csv") or name.endswith(".json"):
            return t("assembly.pointer_evidence_tables")
        return t("assembly.pointer_figures")

    index_pointer = (("evaluation_evidence/*.csv", None),
                     ("evaluation_evidence_ext/*.csv", None),
                     ("processed_proteins/differential_*.csv", None),
                     ("visualize_results/figure_index.md", None))
    for pattern, _unused in index_pointer:
        for path in sorted(Path(run).glob(pattern)):
            lines.append("| %s | %s |" % (_cell(_reader_asset_label(path.name)), _pointer_for(path.name)))
    # A table already printed inside its task section is not repeated here: the asset list is an
    # index plus the records that no task section displayed (unbound / non-standing contrasts).
    shown_candidates = {str(c) for task in index["tasks"] for c in task["candidates"][:8]}
    shown_modules = {str(m) for task in index["tasks"] for m in task["modules"][:6]}
    rest_candidates = [key for key in index["candidates"] if str(key) not in shown_candidates]
    rest_modules = [key for key in index["modules"] if str(key) not in shown_modules]
    n_cand, n_mod = len(index["candidates"]), len(index["modules"])
    # R17 fix: the note must state what that other table actually contains; the presence-only clause
    # is only true for datasets that record such rows.
    presence_only = any(str(row.get("contrast") or "").strip() == "matrix_presence_only"
                        for row in _read_evidence_rows(Path(run) / "evaluation_evidence"
                                                       / "candidate_protein_evidence.csv"))
    other_note = (t("assembly.absence_note_includes_presence_only") if presence_only
                  else t("assembly.absence_note_excludes_presence_only"))
    lines += ["", t("assembly.asset_candidate_index")
              % (n_cand, n_cand - len(rest_candidates),
                 t("assembly.more_records_suffix") % len(rest_candidates) if rest_candidates else t("assembly.no_further_repeat")),
              t("assembly.asset_count_scope_note") % other_note]
    if rest_candidates:
        lines += [""] + _candidate_block({"candidates": rest_candidates}, index, index["conditions"], limit=None)
    lines += ["", t("assembly.asset_module_index")
              % (n_mod, n_mod - len(rest_modules),
                 t("assembly.more_module_records_suffix") % len(rest_modules) if rest_modules else t("assembly.no_further_repeat"))]
    if rest_modules:
        lines += [""] + _module_block({"modules": rest_modules}, index, limit=None)
    # Internal evidence IDs and file names stay out of the reader-facing document; each line points
    # at the section that carries the same records in reader-readable form.
    lines += ["", t("assembly.asset_evidence_entry_points")]
    lines.append(t("assembly.asset_contrast_records") % len(index["contrasts"]))
    lines.append(t("assembly.asset_candidate_records") % len(index["candidates"]))
    lines.append(t("assembly.asset_module_records") % len(index["modules"]))
    return lines


def _serialisable_index(index: Dict[str, Any]) -> Dict[str, Any]:
    """Tuple composite keys cannot be JSON-serialised; expose them as explicit key lists."""
    return {"contrasts": index["contrasts"],
            "candidates": [{"key": list(key), **value} for key, value in index["candidates"].items()],
            "modules": [{"key": list(key), **value} for key, value in index["modules"].items()],
            "duplicates": index["duplicates"]}

# --------------------------------------------------------------------------- run evidence binding
# Every quantitative claim depends on parameters the run either recorded or did not. This block reads
# them from the run artifacts only. A parameter the run never wrote down stays 未记录; the assembler
# does not re-read the report prose to fill it, and never keeps whichever reading flatters the text.


def _matrix_stage_files() -> Tuple[Tuple[str, str], ...]:
    """Reader label and relative path of each matrix stage on the run chain."""
    return (
        (t("assembly.stage_delivered_matrix"), "processed_proteins/ProQuant_Normalized.csv"),
        (t("assembly.stage_combat_matrix"), "processed_proteins/ProteinQuant_ComBat.csv"),
        (t("assembly.stage_filtered_matrix"), "processed_proteins/ProteinQuant_Filtered.csv"),
    )

# Key tokens that describe a handling *policy*. "non_missing_fraction" is a level, not a policy, and
# must never be presented as the missing-value treatment.
MISSING_KEYS = ("imput", "missing_policy", "missing_handling", "missing_value", "missing_code",
                "fillna", "nan_policy", "detection_policy", "missing_treatment")

ANNOTATION_PREFIXES = ("pg.", "unnamed", "index", "id", "protein", "gene", "uniprot", "accession")


def _sample_columns(header: List[str]) -> int:
    """Number of sample columns in a matrix CSV.

    R17: the previous count subtracted a single annotation column, so every matrix was reported with
    one sample too many (turnover 41 instead of 40, carr 162 instead of 161, leakage 2296 instead of
    2295). Sample columns follow the run's own naming convention (raw-file names); the remaining
    columns are the identifier columns that precede them.
    """
    names = [str(cell).strip() for cell in header if str(cell).strip()]
    if not names:
        return 0
    sample_like = [name for name in names if re.search(r"\.(raw|mzml|d|wiff)$", name, re.I)]
    if sample_like:
        return len(sample_like)
    annotation = [name for name in names if name.lower().startswith(ANNOTATION_PREFIXES)]
    return max(len(names) - len(annotation), 0)


def _csv_shape(path: Path) -> Optional[Dict[str, int]]:
    """(rows, columns) of a matrix file; None when the file is absent or unreadable."""
    import csv as _csv
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8-sig", errors="replace") as handle:
            reader = _csv.reader(handle)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        return {"rows": rows, "cols": _sample_columns(header)}
    except Exception:  # noqa: BLE001
        return None


def _first_record(limma: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(limma, dict):
        return {}
    for value in limma.values():
        if isinstance(value, dict):
            return value
    return {}


def _missing_declaration(*nodes) -> Tuple[str, str]:
    """Search the run records for an explicit statement of how missing values are handled."""
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if any(token in str(key).lower() for token in MISSING_KEYS):
                return _fmt(value), t("assembly.source_run_record") % key
    return t("assembly.not_recorded"), ""


def _qc_stage_rows(run: Path) -> List[Dict[str, Any]]:
    """Primary-group rows of the stage-aware QC table (input stage vs analysed stage)."""
    rows = _read_evidence_rows(Path(run) / "evaluation_evidence" / "group_composition_qc.csv")
    return [row for row in rows if str(row.get("table") or "") == "primary_group"]


def _num(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def read_run_evidence(run: Path) -> Dict[str, Any]:
    """Declare the run parameters behind the quantitative claims, each with the record it came from."""
    run = Path(run)
    params = read_json(run / "parameters.json", {}) or {}
    design = params.get("analysis_design") or {}
    matrix = design.get("matrix") or {}
    diff = design.get("differential") or {}
    limma = read_json(run / "processed_proteins" / "limma_summary.json", {}) or {}
    first = _first_record(limma)
    sanity = first.get("sanity_checks") or {}
    thresholds_used = sanity.get("thresholds_used") or {}
    cond = read_conditions(run)
    items: List[Dict[str, Any]] = []

    def add(item, value, source, status=t("assembly.status_recorded"), note=""):
        items.append({"item": item, "value": _fmt(value), "source": source, "status": status,
                      "note": note})

    add(t("assembly.item_input_matrix_state"), matrix.get("matrix_state"),
        "parameters.json → analysis_design.matrix.matrix_state")
    if matrix.get("non_missing_fraction") is not None:
        add(t("assembly.item_non_missing_fraction"), "%.4f" % float(matrix["non_missing_fraction"]),
            "parameters.json → analysis_design.matrix.non_missing_fraction",
            note=t("assembly.note_non_missing_fraction_stage"))
    stages = []
    for label, rel in _matrix_stage_files():
        shape = _csv_shape(run / rel)
        if shape:
            stages.append(t("assembly.matrix_stage_shape") % (label, shape["rows"], shape["cols"]))
    add(t("assembly.item_matrix_stages"), "；".join(stages) or t("assembly.not_recorded"),
        t("assembly.source_matrix_shapes"),
        status=t("assembly.status_recorded") if stages else t("assembly.not_recorded"))
    if first.get("n_proteins") is not None:
        add(t("assembly.item_matrix_size_tested"), t("assembly.proteins_per_contrast") % _fmt(first.get("n_proteins")),
            "limma_summary.json → n_proteins")

    scale_bits = []
    if matrix.get("looks_logged") is not None:
        scale_bits.append(t("assembly.field_looks_logged") % matrix.get("looks_logged"))
    if matrix.get("needs_log_transform") is not None:
        scale_bits.append(t("assembly.field_needs_log_transform") % matrix.get("needs_log_transform"))
    if matrix.get("already_processed") is not None:
        scale_bits.append(t("assembly.field_already_processed") % matrix.get("already_processed"))
    add(t("assembly.item_scale_and_transform"), "；".join(scale_bits) or t("assembly.not_recorded"),
        "parameters.json → analysis_design.matrix.looks_logged / needs_log_transform / already_processed",
        status=t("assembly.status_partially_recorded") if scale_bits else t("assembly.not_recorded"),
        note=t("assembly.note_scale_flags_only"))

    # R28: the design metadata holds only inferred flags. The actions the pipeline actually executed
    # are written by the matrix-building step, so the two are reported as separate items and an absent
    # action record is stated as not-recorded rather than as "not executed".
    transform = read_matrix_transform_record(run)
    if isinstance(transform, dict) and transform:
        imp = transform.get("imputation") or {}
        logt = transform.get("log2_transform") or {}
        filt = transform.get("protein_filtering") or {}
        transform_source = str(transform.get("record_source")
                               or "processed_proteins/matrix_transform_record.json")
        derived_note = (t("assembly.transform_record_derived_note")
                        % transform_source) if transform.get("derived") else ""
        if "applied" in imp:
            add(t("assembly.item_imputation_executed_actual"),
                (t("assembly.executed_colon_two") % (imp.get("method") or t("assembly.method_not_recorded"),
                                  (t("assembly.imputed_values_suffix") % imp.get("n_values_imputed"))
                                  if imp.get("n_values_imputed") not in (None, "") else ""))
                if imp.get("applied") else t("assembly.status_not_executed"),
                "%s → imputation" % transform_source, note=derived_note)
        if "applied" in logt:
            add(t("assembly.item_log_transform_executed"),
                (t("assembly.executed_colon") % (logt.get("form") or t("assembly.form_not_recorded"))) if logt.get("applied")
                else (t("assembly.not_recorded_rule") % (logt.get("recorded_rule") or t("assembly.not_recorded"))
                      if logt.get("applied") is None else t("assembly.status_not_executed")),
                "%s → log2_transform" % transform_source,
                note=("；".join(part for part in (
                    logt.get("applied_basis") or "",
                    t("assembly.note_needs_log_transform"),
                    derived_note) if part)))
        if filt:
            add(t("assembly.item_protein_filtering_executed"),
                t("assembly.protein_filtering_text") % (filt.get("rule") or t("assembly.not_recorded"),
                                      _fmt(filt.get("n_proteins_retained"))),
                "%s → protein_filtering" % transform_source)
    else:
        add(t("assembly.item_matrix_transform_executed"), t("assembly.not_recorded"),
            "processed_proteins/matrix_transform_record.json",
            status=t("assembly.not_recorded"),
            note=t("assembly.note_transform_record_absent"))
    if matrix.get("numeric_min") is not None and matrix.get("numeric_max") is not None:
        add(t("assembly.item_input_numeric_range"), t("assembly.range_generic") % (_fmt(matrix.get("numeric_min")),
                                          _fmt(matrix.get("numeric_max"))),
            "parameters.json → analysis_design.matrix.numeric_min / numeric_max")

    missing_value, missing_source = _missing_declaration(matrix, diff, design, first)
    add(t("assembly.item_missing_value_handling"), missing_value, missing_source or t("assembly.source_run_records_design"),
        status=t("assembly.status_recorded") if missing_source else t("assembly.not_recorded"),
        note=t("assembly.note_imputation_versus_missing"))

    add(t("assembly.field_design_source"), _evidence_source_label(design.get("source")),
        "parameters.json → analysis_design.source")
    add(t("assembly.field_group_and_id_cols"), "；".join(
        part for part in (t("assembly.field_group_col") % _fmt(cond.get("group_col")),
                          t("assembly.field_sample_id_col") % _fmt(cond.get("sample_id_col")),
                          t("assembly.field_batch_col") % _fmt(cond.get("batch_col")),
                          t("assembly.field_pair_col") % _fmt(cond.get("pair_col"))) if part),
        "parameters.json → analysis_design.group_col / sample_id_col / batch_col / pair_col")
    contrasts = cond.get("contrasts") or []
    contrast_names = [str(c.get("name")) for c in contrasts if c.get("name")]
    if contrasts:
        shown = "；".join(contrast_names[:3]) + ((t("assembly.more_contrasts_suffix") % (len(contrast_names) - 3)) if len(contrast_names) > 3 else "")
        add(t("assembly.item_contrasts_in_design") % len(contrasts), shown or t("assembly.not_recorded"),
            "parameters.json → analysis_design.differential.contrasts")
    covariates = design.get("covariates")
    if covariates is not None:
        add(t("assembly.field_covariates"), "；".join(str(c) for c in covariates) if covariates else t("assembly.none_value"),
            "parameters.json → analysis_design.covariates")

    add(t("assembly.item_statistical_model_raw"), first.get("method"),
        "limma_summary.json → method", status=t("assembly.status_recorded") if first.get("method") else t("assembly.not_recorded"))
    if first.get("statistical_backend"):
        # the raw backend name and the library error string are developer-facing; the reader only
        # needs to know that the standard pipeline was unavailable and which implementation ran
        backend_raw = str(first.get("statistical_backend"))
        backend = (t("assembly.backend_internal_fallback")
                   if "fallback" in backend_raw.lower() else backend_raw)
        add(t("assembly.field_statistical_backend"), backend, "limma_summary.json → statistical_backend / fallback_reason")
    unit_records = [rec.get("unit_sensitivity") for rec in limma.values()
                    if isinstance(rec, dict) and isinstance(rec.get("unit_sensitivity"), dict)]
    if unit_records:
        unit_done = [u for u in unit_records if str(u.get("status")) == "completed"]
        unit_names = sorted({str(u.get("analysis_unit")) for u in unit_done if u.get("analysis_unit")})
        unit_shared = [u.get("n_units_shared") for u in unit_done if isinstance(u.get("n_units_shared"), int)]
        unit_sources = sorted({_unit_source_label(u.get("unit_column_source")) for u in unit_done})
        unit_pass_fdr = sum(int(u.get("n_passing_fdr_only") if u.get("n_passing_fdr_only") is not None
                                else (u.get("n_passing_fdr") or 0)) for u in unit_done)
        unit_pass_joint = sum(int(u.get("n_passing_joint") or 0) for u in unit_done)
        unit_not_tested = sum(int(u.get("n_proteins_not_tested") or 0) for u in unit_done)
        unit_thresholds = sorted({str(u.get("logfc_threshold")) for u in unit_done
                                  if u.get("logfc_threshold") is not None})
        unit_excluded = sorted({str(label) for u in unit_records
                                for label in ((u.get("unit_exclusion") or {}).get("pooled_source") or [])
                                + ((u.get("unit_exclusion") or {}).get("unconfirmed_source") or [])})
        if unit_done:
            span = ("%d–%d" % (min(unit_shared), max(unit_shared))) if unit_shared else t("assembly.not_recorded")
            unit_value = (t("assembly.unit_layer_digest")
                          % ("／".join(unit_names) or t("assembly.not_recorded"), "、".join(unit_sources) or t("assembly.not_recorded"),
                             len(unit_records), len(unit_done), span, unit_pass_fdr,
                             "／".join(unit_thresholds) or t("assembly.not_recorded"), unit_pass_joint, unit_not_tested))
            unit_status = t("assembly.status_recorded")
        else:
            unit_value = t("assembly.unit_all_blocked") % len(unit_records)
            unit_status = t("assembly.status_partially_recorded")
        unit_blocked = [u for u in unit_records if str(u.get("status")) != "completed" and u.get("reasons")]
        unit_notes = []
        if unit_excluded:
            unit_notes.append(t("assembly.unit_excluded_note_digest")
                              % "、".join(_cell(x) for x in unit_excluded))
        if unit_blocked:
            unique_reasons = []
            for record in unit_blocked:
                text = _unit_reason_zh(record)
                if text not in unique_reasons:
                    unique_reasons.append(text)
            unit_notes.append(t("assembly.unit_blocked_reasons") % "；".join(unique_reasons[:2]))
        unit_notes.append(t("assembly.unit_reference_note_short"))
        add(t("assembly.item_unit_sensitivity"), unit_value, "limma_summary.json → unit_sensitivity",
            status=unit_status, note="".join(unit_notes))
    if first.get("n_samples_group_a") is not None and first.get("n_samples_group_b") is not None:
        first_name = contrast_names[0] if contrast_names else t("assembly.not_recorded")
        add(t("assembly.item_samples_per_group") % first_name,
            t("assembly.samples_two_arms_named") % (_fmt(first.get("n_samples_group_a")),
                                    _fmt(first.get("n_samples_group_b"))),
            "limma_summary.json → n_samples_group_a / n_samples_group_b")

    add(t("assembly.field_fdr_method"), diff.get("fdr_method"),
        "parameters.json → analysis_design.differential.fdr_method",
        status=t("assembly.status_recorded") if diff.get("fdr_method") else t("assembly.not_recorded"))
    if thresholds_used:
        add(t("assembly.field_effective_threshold"),
            t("assembly.threshold_adj_p_and_effect") % (_fmt(thresholds_used.get("adj_p")),
                                      _fmt(thresholds_used.get("abs_logFC"))),
            "limma_summary.json → sanity_checks.thresholds_used")
    elif diff.get("p_thresh") is not None and diff.get("logfc_thresh") is not None:
        add(t("assembly.field_design_threshold"),
            t("assembly.threshold_adj_p_and_effect") % (_fmt(diff.get("p_thresh")), _fmt(diff.get("logfc_thresh"))),
            "parameters.json → analysis_design.differential.p_thresh / logfc_thresh")
    if sanity.get("n_significant") is not None:
        first_name = contrast_names[0] if contrast_names else t("assembly.not_recorded")
        add(t("assembly.item_n_significant_from_run") % first_name,
            t("assembly.n_significant_split") % (_fmt(sanity.get("n_significant")), _fmt(sanity.get("n_up")),
                                     _fmt(sanity.get("n_down"))),
            "limma_summary.json → sanity_checks.n_significant / n_up / n_down")

    qc_rows = _qc_stage_rows(run)
    input_missing = [x for x in (_num(row.get("input_mean_missing_rate")) for row in qc_rows)
                     if x is not None]
    analysed_missing = [x for x in (_num(row.get("analysed_mean_missing_rate")) for row in qc_rows)
                        if x is not None]
    imputation = sorted({str(row.get("imputation_applied")) for row in qc_rows
                         if row.get("imputation_applied") not in (None, "")})
    if input_missing:
        add(t("assembly.item_missing_rate_raw_by_group"),
            _range_text([min(input_missing), max(input_missing)]),
            t("assembly.source_qc_input_missing"))
    if analysed_missing:
        add(t("assembly.item_missing_rate_analysed_by_group"),
            _range_text([min(analysed_missing), max(analysed_missing)]),
            t("assembly.source_qc_analysed_missing"))
    _transform_imp = read_matrix_transform_record(run)
    _imp = (_transform_imp.get("imputation") or {}) if isinstance(_transform_imp, dict) else {}
    if imputation or _imp.get("applied") is not None:
        # R28: the QC flag is written before the matrix is imputed; when the execution record has an
        # answer it wins, and a conflict between the two is stated instead of silently resolved.
        qc_text = "；".join(imputation) or t("assembly.not_recorded")
        if _imp.get("applied") is True:
            value = (t("assembly.imputation_conflict_note") % (_imp.get("method") or t("assembly.method_not_recorded"), qc_text))
        elif _imp.get("applied") is False:
            value = t("assembly.status_not_executed")
        else:
            value = qc_text
        add(t("assembly.item_imputation_executed"), value,
            "processed_proteins/matrix_transform_record.json → imputation"
            if _imp.get("applied") is not None else t("assembly.source_qc_imputation"),
            note=t("assembly.note_imputation_flag_scope"))
    return {"items": items, "conditions": cond, "limma_first": first, "design_source": design.get("source"),
            "input_missing_rate": [min(input_missing), max(input_missing)] if input_missing else None,
            "analysed_missing_rate": [min(analysed_missing), max(analysed_missing)] if analysed_missing else None,
            "imputation": imputation, "qc_rows": len(qc_rows)}


def _run_method_summary_lines(run: Path, cond: Dict[str, Any], evidence_data: Dict[str, Any]) -> List[str]:
    """Three reader-facing lines: which matrix was tested, which model ran, which numbers are bound."""
    lines: List[str] = []
    stage_text = next((x["value"] for x in evidence_data["items"]
                       if x["item"] == t("assembly.item_matrix_stages")), "")
    tested = next((part.strip() for part in str(stage_text).split("；") if t("assembly.stage_marker_filtered") in part),
                  t("assembly.not_recorded"))
    analysed = evidence_data.get("analysed_missing_rate")
    raw = evidence_data.get("input_missing_rate")
    if raw and analysed:
        # R28: the QC table's imputation flag is written before the calibration step runs, so it cannot
        # be used to say that no imputation happened. The execution record decides this sentence.
        _transform = read_matrix_transform_record(run)
        _imp = (_transform.get("imputation") or {}) if isinstance(_transform, dict) else {}
        if _imp.get("applied") is True:
            imputation_text = t("assembly.imputation_done_suffix") % (_imp.get("method") or t("assembly.method_not_recorded"))
        elif _imp.get("applied") is False:
            imputation_text = t("assembly.imputation_not_done_suffix")
        else:
            imputation_text = t("assembly.imputation_unknown_suffix")
        lines.append(t("assembly.tested_matrix_and_missing") % (tested, _range_text(raw), _range_text(analysed),
                                          imputation_text))
    else:
        lines.append(t("assembly.tested_matrix_plain")
                     % (tested, t("assembly.not_recorded")))
    lines.append(t("assembly.method_summary_pointer"))
    return lines


def _range_text(bounds) -> str:
    low, high = bounds[0], bounds[1]
    return "%.4f" % low if low == high else t("assembly.range_two_values") % (low, high)

# Record locators are rendered in reader language; the machine keys stay in the JSON sidecar so a
# reviewer can still find the exact field. Longest keys first, so a specific rule wins over a general one.


def _record_labels() -> Tuple[Tuple[str, str], ...]:
    """Machine record locator -> the reader wording for that location, longest key first."""
    return (
        ("parameters.json → analysis_design.matrix.", t("assembly.record_parameters_design_matrix")),
        ("parameters.json → analysis_design.differential.", t("assembly.record_parameters_design_differential")),
        ("parameters.json → analysis_design.", t("assembly.record_parameters_design")),
        ("parameters.json → ", t("assembly.record_parameters_dot")),
        ("limma_summary.json → sanity_checks.", t("assembly.record_limma_sanity")),
        ("limma_summary.json → ", t("assembly.record_limma_dot")),
        ("analysis_design", t("assembly.analysis_design")),
        ("limma_summary", t("assembly.record_limma")),
        ("parameters.json", t("assembly.record_parameters")),
    )


def _field_labels() -> Tuple[Tuple[str, str], ...]:
    """Machine field name -> the reader wording for that field."""
    return (
        ("looks_logged / needs_log_transform / already_processed", t("assembly.field_scale_flags")),
        ("statistical_backend / fallback_reason", t("assembly.field_statistical_backend_and_reason")),
        ("n_samples_group_a / n_samples_group_b", t("assembly.field_n_per_group")),
        ("n_significant / n_up / n_down", t("assembly.field_pass_count")),
        ("group_col / sample_id_col / batch_col", t("assembly.field_group_and_id_cols")),
        ("p_thresh / logfc_thresh", t("assembly.field_design_threshold")),
        ("numeric_min / numeric_max", t("assembly.field_numeric_range")),
        ("non_missing_fraction", t("assembly.field_non_missing_fraction_short")),
        ("thresholds_used", t("assembly.field_threshold_used")),
        ("input_mean_missing_rate", t("assembly.field_input_mean_missing")),
        ("analysed_mean_missing_rate", t("assembly.field_analysed_missing_rate")),
        ("imputation_applied", t("assembly.field_imputation_applied")),
        ("matrix_state", t("assembly.field_matrix_state")),
        ("fdr_method", t("assembly.field_fdr_method")),
        ("n_proteins", t("assembly.field_tested_proteins")),
        ("contrasts", t("assembly.field_contrasts")),
        ("covariates", t("assembly.field_covariates")),
        ("method", t("assembly.field_statistical_method")),
        ("source", t("assembly.field_design_source")),
    )


def _reader_source_label(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return t("assembly.not_recorded")
    for key, label in _field_labels():
        text = text.replace(key, label)
    for key, label in _record_labels():
        text = text.replace(key, label)
    return text.replace(" → ", " · ")


def _run_evidence_binding_lines(evidence_data: Dict[str, Any]) -> List[str]:
    lines = [t("assembly.evidence_binding_intro"), "",
             t("assembly.header_evidence_binding"), "|---|---|---|---|"]
    for item in evidence_data["items"]:
        note = ("；%s" % item["note"]) if item.get("note") else ""
        lines.append("| %s | %s | %s | %s%s |" % (_cell(item["item"]), _cell(item["value"]),
                                                 _cell(_reader_source_label(item["source"])),
                                                 item["status"], _cell(note)))
    return lines


def _differential_table(run: Path, rec: Dict[str, Any]) -> Optional[Path]:
    for name in (rec.get("source_contrast"), rec.get("display_contrast")):
        if not name:
            continue
        candidate = Path(run) / "processed_proteins" / ("differential_%s.csv" % name)
        if candidate.exists():
            return candidate
    return None


def _joint_filter_findings(run: Path, index: Dict[str, Any],
                           cond: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Recount the declared joint condition on the run's own differential table, per contrast.

    The connector is read back from the report's own threshold sentence, so an OR condition is never
    silently checked as an AND. A table whose columns cannot be parsed yields a hint, not a verdict.
    """
    import csv as _csv
    declared = threshold_text(cond)
    connector = t("assembly.connector_or") if t("assembly.connector_or") in declared else t("assembly.connector_and")
    p_thresh, effect_thresh = cond.get("p_thresh"), cond.get("logfc_thresh")
    out: Dict[str, Dict[str, Any]] = {}
    if p_thresh is None or effect_thresh is None:
        return out
    for rec in index["contrasts"]:
        key = str(rec.get("display_contrast"))
        path = _differential_table(run, rec)
        if path is None:
            out[key] = {"status": "no_table", "connector": connector}
            continue
        p_col, effect_col = None, None
        n_p = n_e = n_joint = 0
        min_p = None
        try:
            with path.open(encoding="utf-8-sig", errors="replace") as handle:
                reader = _csv.DictReader(handle)
                header = reader.fieldnames or []
                for name in ("adj.P.Val", "padj", "FDR", "adj_p"):
                    if name in header:
                        p_col = name
                        break
                for name in ("logFC", "LogFC", "log2FC", "effect"):
                    if name in header and name not in ("adj.P.Val",):
                        effect_col = name
                        break
                if not p_col or not effect_col:
                    out[key] = {"status": "unparseable_columns", "connector": connector,
                                "columns": header}
                    continue
                for row in reader:
                    p_val = _num(row.get(p_col))
                    eff = _num(row.get(effect_col))
                    if p_val is None:
                        continue
                    min_p = p_val if min_p is None else min(min_p, p_val)
                    ok_p = p_val <= float(p_thresh)
                    ok_e = eff is not None and abs(eff) >= float(effect_thresh)
                    n_p += 1 if ok_p else 0
                    n_e += 1 if ok_e else 0
                    n_joint += 1 if (ok_p and ok_e if connector == t("assembly.connector_and") else (ok_p or ok_e)) else 0
        except Exception:  # noqa: BLE001
            out[key] = {"status": "unreadable", "connector": connector}
            continue
        out[key] = {"status": "ok", "connector": connector, "p_col": p_col, "effect_col": effect_col,
                    "n_joint": n_joint, "n_pass_p": n_p, "n_pass_effect": n_e, "min_adj_p": min_p,
                    "declared_n_sig": rec.get("n_sig"), "table": path.name}
    return out

HIGH_WORDS = ("较高", "更高", "偏高", t("assembly.direction_upregulated"), "升高", "较高表达")
LOW_WORDS = ("较低", "更低", "偏低", t("assembly.direction_downregulated"), "降低", "较低表达")
DIRECTION_WORDS = HIGH_WORDS + LOW_WORDS
NEGATION_WORDS = ("不", "未", t("assembly.none_value"), "非", "尚未", "没有")
# naming *which* matrix the statement is about is the binding the check asks for; a sentence that
# names neither a matrix nor a stage is the only case worth a hint.
STAGE_MARKERS = ("原始", "过滤", "插补", "矩阵", "质控", "本次", "本报告")
DETECTION_WORDS = ("未检出", "未检测", "完全不存在", "未被检出", "未鉴定")
# A sentence that states the absence of a number is itself a boundary statement, not an unbound claim.
DETECTION_EXCLUSIONS = ("未给出", "未提供", t("assembly.not_recorded"), "未纳入", "没有给出", "不给出", "未在当前",
                        "无法确认", "不适用")
UNAVAILABLE_PAT = re.compile(r"未[^。；]{0,16}(给出|提供|记录|纳入|统计|报告|包含|覆盖|获得)")


def _arm_variants(group: str) -> List[str]:
    """Both the stored label and the reader spelling (underscore replaced by a space)."""
    text = str(group or "").strip()
    if not text:
        return []
    variants = [text]
    if "_" in text:
        variants.append(text.replace("_", " "))
    return variants


def _sentence_with(text: str, token: str) -> str:
    for sentence in re.split(r"(?<=[。！？；;])", text):
        if token in sentence:
            return sentence.strip()
    return ""


def _model_claim_findings(index: Dict[str, Any], model_texts: List[Dict[str, Any]]
                          ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Bounded checks on what the model wrote: direction subject, stage binding, table disagreement.

    Scope is deliberately narrow: direction is only compared when the sentence names the entity, one
    of the two arms, and a direction word, and when the paragraph is bound to exactly one contrast.
    Everything else is counted as uncovered instead of guessed at. Negated sentences are skipped, so
    "未达显著" cannot be read as a direction claim.
    """
    findings: List[Dict[str, Any]] = []
    coverage = {"paragraphs": len(model_texts), "contrast_scoped": 0, "direction_checked": 0,
                "stage_checked": 0, "uncovered_no_contrast": 0}
    by_id = {rec.get("id"): rec for rec in index["contrasts"]}
    single = index["contrasts"][0] if len(index["contrasts"]) == 1 else None
    for para in model_texts:
        text = str(para.get("text") or "")
        if not text:
            continue
        contrast = None
        for eid in para.get("evidence_ids") or []:
            if eid in by_id:
                contrast = by_id[eid]
                break
        if contrast is None and single is not None:
            contrast = single
        if contrast is not None:
            coverage["contrast_scoped"] += 1
        else:
            coverage["uncovered_no_contrast"] += 1
        groups = [g for g in (contrast or {}).get("groups") or [] if g]
        for token in DETECTION_WORDS:
            sentence = _sentence_with(text, token)
            if sentence and not any(marker in sentence for marker in STAGE_MARKERS):
                if any(word in sentence for word in DETECTION_EXCLUSIONS):
                    continue
                if UNAVAILABLE_PAT.search(sentence):
                    continue
                coverage["stage_checked"] += 1
                findings.append({"rule": "STAGE", "severity": t("assembly.severity_hint"),
                                 "detail": t("assembly.finding_stage_unclear"),
                                 "quote": sentence[:120]})
                break
        if contrast is None or len(groups) < 2:
            continue
        group_variants = [(group, variant) for group in groups
                          for variant in _arm_variants(group)]
        for (key, cand) in index["candidates"].items():
            if (cand.get("display_contrast") or cand.get("source_contrast")) != contrast.get("display_contrast"):
                continue
            symbols = [s for s in (cand.get("candidate"), cand.get("matched_gene")) if s]
            symbols += [s for s in str(cand.get("protein_group") or "").split(";") if s]
            for symbol in symbols:
                if not symbol or symbol not in text:
                    continue
                sentence = _sentence_with(text, symbol)
                if not sentence or any(word in sentence for word in NEGATION_WORDS):
                    continue
                hits = [(sentence.find(variant), group) for group, variant in group_variants
                        if variant in sentence]
                # a comparative sentence names both arms; only a single-arm statement is unambiguous
                if len({group for _, group in hits}) != 1:
                    continue
                words = [(sentence.find(w), w) for w in DIRECTION_WORDS if w in sentence]
                if not words:
                    continue
                arm_pos, arm = hits[0]
                distance, word = min(((abs(arm_pos - pos), w) for pos, w in words), key=lambda x: x[0])
                if distance > 20:
                    continue
                phrase = _direction_label(cand, groups)
                if phrase == t("assembly.direction_undetermined") or t("assembly.arm_higher_suffix") not in phrase:
                    continue
                high_arm = phrase.split(t("assembly.arm_higher_suffix"))[0]
                low_arm = next((g for g in groups if g != high_arm), None)
                if low_arm is None:
                    continue
                expected = high_arm if word in HIGH_WORDS else low_arm
                coverage["direction_checked"] += 1
                if arm != expected:
                    findings.append({"rule": "DIR", "severity": t("assembly.severity_hint"),
                                     "detail": t("assembly.finding_direction_mismatch")
                                               % phrase,
                                     "quote": sentence[:160], "entity": symbol,
                                     "contrast": contrast.get("display_contrast")})
                break
    return findings, coverage

# R17: a sentence that denies evidence the run actually produced is a locator, never a rewrite.
UNAVAILABLE_TOPICS = (
    (t("assembly.topic_sample_count"), (t("assembly.topic_sample_count"), "观测数", "每组样本", "样本量")),
    (t("assembly.topic_missing_rate"), (t("assembly.topic_missing_rate"), "缺失比例", "缺失统计")),
    (t("assembly.topic_stratified"), ("分层", "亚组", "分层统计")),
    (t("assembly.topic_formal_enrichment"), ("正式富集", "富集检验", "超几何", "通路富集", "ORA")),
    (t("assembly.topic_pathway_terms"), ("富集", "通路")),
    (t("assembly.topic_dose_trend"), (t("assembly.topic_dose_trend"), "趋势检验", "spearman")),
    (t("assembly.topic_contrast_concordance"), (t("assembly.topic_contrast_concordance"), "跨对比一致性")),
    (t("assembly.topic_dimensionality_and_figures"), ("降维", "pca", "umap", "图册", "火山图", "降维坐标")),
)


def available_evidence_topics(run: Path, index: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Which evidence classes this run actually produced, read from the artifacts (cheap checks)."""
    run = Path(run)
    evidence = (index or {}).get("run_evidence") or read_run_evidence(run)
    first = evidence.get("limma_first") or {}
    topics = {
        t("assembly.topic_sample_count"): t("assembly.status_computed") if first.get("n_samples_group_a") is not None
                  or first.get("n_samples_group_b") is not None else t("assembly.not_recorded"),
        t("assembly.topic_missing_rate"): t("assembly.status_computed") if evidence.get("input_missing_rate")
                  or evidence.get("analysed_missing_rate") else t("assembly.not_recorded"),
    }
    stratified = _ext_csv_rows(run, "stratified_contrast_summary.csv")
    topics[t("assembly.topic_stratified")] = t("assembly.status_computed") if any(str(row.get("level") or "").strip() for row in stratified) else t("assembly.status_not_computed")
    ora = _ext_csv_rows(run, "enrichment_ora.csv")
    topics[t("assembly.topic_formal_enrichment")] = t("assembly.status_computed") if any(str(row.get("p_value") or "").strip() for row in ora) else t("assembly.status_not_computed")
    overlap = run / "enrichment_results" / "go_kegg_reactome_results.json"
    topics[t("assembly.topic_pathway_terms")] = t("assembly.status_computed") if overlap.exists() else t("assembly.status_not_computed")
    topics[t("assembly.topic_dose_trend")] = t("assembly.status_computed") if _ext_csv_rows(run, "dose_trend.csv") else t("assembly.status_not_computed")
    topics[t("assembly.topic_contrast_concordance")] = t("assembly.status_computed") if _ext_csv_rows(run, "contrast_concordance.csv") else t("assembly.status_not_computed")
    try:
        figures = sorted(path.name for path in (run / "visualize_results").glob("*.png"))
    except Exception:  # noqa: BLE001
        figures = []
    topics[t("assembly.topic_dimensionality_and_figures")] = t("assembly.status_computed") if figures else t("assembly.status_not_computed")
    return topics


def _unavailable_claim_findings(run: Path, index: Dict[str, Any], model_texts: List[Dict[str, Any]],
                                limit: int = 4) -> Tuple[List[Dict[str, Any]], int]:
    """Flag a model sentence that says evidence was not provided although this run has it.

    The sentence is quoted, never edited: the finding says which artifact contradicts it so the
    reader (or the next generation) can act on it.
    """
    topics = available_evidence_topics(run, index)
    findings: List[Dict[str, Any]] = []
    total = 0
    for item in model_texts or []:
        sentence = str(item.get("text") or "")
        if not sentence or not UNAVAILABLE_PAT.search(sentence):
            continue
        lowered = sentence.lower()
        for topic, patterns in UNAVAILABLE_TOPICS:
            if topics.get(topic) != t("assembly.status_computed"):
                continue
            if any(pattern in lowered for pattern in patterns):
                total += 1
                if len(findings) < limit:
                    findings.append({
                        "rule": "G7", "severity": t("assembly.severity_hint"), "topic": topic,
                        "owner": item.get("owner"),
                        # reader-facing and short: the note locates the conflict without repeating the
                        # sentence and without naming any internal request artefact
                        "detail": t("assembly.finding_unavailable_detail")
                                  % topic})
                break
    return findings, total


def _qc_binding_finding(evidence_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = evidence_data.get("input_missing_rate")
    analysed = evidence_data.get("analysed_missing_rate")
    if not raw or not analysed:
        return None
    if analysed[1] > 0 or evidence_data.get("imputation") == ["True"]:
        return None
    return {"rule": "QC", "severity": t("assembly.severity_hint"),
            "detail": t("assembly.finding_qc_stage")
                      % (analysed[0], "；".join(evidence_data.get("imputation") or [t("assembly.not_recorded")]),
                         raw[0], raw[1])}


def declared_mismatch(info: Dict[str, Any]) -> bool:
    """True when the contrast's own summary count differs from the recount of its declared rule."""
    if info.get("status") != "ok":
        return False
    try:
        return int(float(info.get("declared_n_sig"))) != int(info.get("n_joint"))
    except (TypeError, ValueError):
        return False


def _joint_note_line(rec: Dict[str, Any], info: Dict[str, Any]) -> str:
    return (t("assembly.finding_joint_mismatch")
            % (info.get("connector"), info.get("n_joint"), _fmt(info.get("declared_n_sig")),
               info.get("n_pass_p"), info.get("n_pass_effect"), _fmt(info.get("min_adj_p"))))


def _consistency_note_lines(checks: Dict[str, Any], limit: int = 6) -> List[str]:
    """Reader-facing rendering of the findings: location and evidence only, never a rewrite."""
    findings = checks.get("findings") or []
    if not findings:
        return []
    lines = [t("assembly.consistency_note_intro") % len(findings), ""]
    for item in findings[:limit]:
        detail = str(item.get("detail") or "")
        quote = item.get("quote")
        if quote:
            detail = t("assembly.quote_note") % (detail, quote)
        lines.append("- %s" % detail)
    if len(findings) > limit:
        lines.append(t("assembly.more_hints") % (len(findings) - limit))
    return lines


def run_consistency_checks(run: Path, index: Dict[str, Any], cond: Dict[str, Any],
                           model_texts: List[Dict[str, Any]],
                           joint: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Same-scope consistency checks. Findings are hints with the triggering evidence; none of them
    rewrites a number or a scientific conclusion."""
    joint = joint if joint is not None else _joint_filter_findings(run, index, cond)
    findings: List[Dict[str, Any]] = []
    for rec in index["contrasts"]:
        key = str(rec.get("display_contrast"))
        info = joint.get(key) or {}
        if info.get("status") != "ok":
            findings.append({"rule": "G1", "severity": t("assembly.severity_hint"),
                             "detail": t("assembly.unparsable_diff_table")
                                       % (key, info.get("status")),
                             "contrast": key})
            continue
        declared_n = info.get("declared_n_sig")
        if declared_n is None:
            continue
        try:
            declared_n = int(float(declared_n))
        except (TypeError, ValueError):
            continue
        if declared_n != info["n_joint"]:
            findings.append({"rule": "G1", "severity": t("assembly.severity_hint"), "contrast": key,
                             "detail": t("assembly.finding_joint_mismatch_detailed")
                                       % (info["connector"], info["p_col"], info["effect_col"],
                                          info["n_joint"], declared_n, info["n_pass_p"],
                                          info["n_pass_effect"], _fmt(info.get("min_adj_p")))})
    model_findings, coverage = _model_claim_findings(index, model_texts)
    findings += model_findings
    unavailable, n_unavailable = _unavailable_claim_findings(run, index, model_texts)
    findings += unavailable
    if isinstance(coverage, dict):
        coverage["unavailable_claims"] = n_unavailable
    qc = _qc_binding_finding(index.get("run_evidence") or {})
    if qc:
        findings.append(qc)
    deduped, seen = [], set()
    for item in findings:
        signature = (item.get("rule"), item.get("quote") or item.get("detail"), item.get("entity"))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    findings = deduped
    summary: Dict[str, int] = {}
    for item in findings:
        summary[item["rule"]] = summary.get(item["rule"], 0) + 1
    return {"joint_filter": joint, "findings": findings, "summary": summary, "coverage": coverage,
            "n_findings": len(findings), "mode": t("assembly.checks_mode"),
            # R21: these findings mark places worth looking at. They are not a quality score and are
            # not part of any acceptance total, and they never rewrite a scientific conclusion -
            # G7 was measured to raise false positives on sentences that state a real limitation.
            "diagnostic_only": True,
            "not_a_metric": t("assembly.checks_not_a_metric")}


def _contrast_pointer(title: str, index: Dict[str, Any]) -> str:
    """Label a model section that only reuses another name of a contrast already printed.

    The model may answer once under the claim id and once under the contrast name; both are its own
    text and stay in the report, but a reader should not have to guess that the two sections are the
    same comparison.
    """
    text = str(title or "").strip()
    if not text:
        return ""
    for order, rec in enumerate(index["contrasts"], start=1):
        names = {str(rec.get("claim_id") or ""), str(rec.get("display_contrast") or ""),
                 str(rec.get("source_contrast") or "")}
        if text in names:
            return (t("assembly.contrast_pointer_note")
                    % (order, rec.get("claim_title") or rec.get("display_contrast")))
    return ""


def assemble(run: Path, sections: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, idempotent assembly; returns the report, the manifest and the content ledger."""
    run = Path(run)
    reset_notes()  # a repeated method note is emitted once per assembled report, not once per process
    plan = plan_report(run, sections)
    index = plan["index"]
    index["tasks"] = plan["tasks"]  # the asset list needs to know which tables the task sections printed
    cond = index["conditions"]
    run_evidence = read_run_evidence(run)
    index["run_evidence"] = run_evidence
    joint_filter = _joint_filter_findings(run, index, cond)
    bindings = check_bindings(sections, index)
    source_text = sections.get("source_text") or ""
    blocks = register_blocks(source_text) if source_text else []

    model_paragraphs: List[Dict[str, Any]] = []
    model_texts: List[Dict[str, Any]] = []
    body: List[str] = [_resolved_report_title(run), ""]
    # R2: the report is a document, so the first line must stay an H1 even when a descriptive title
    # replaced the default one
    body[0] = body[0] if body[0].lstrip().startswith("#") else "# " + body[0].lstrip("# ")
    density_state: Dict[str, int] = {"n": 0}

    def add_model_lines(items, owner, evidence_ids=None):
        for item in items or []:
            text = rewrite_for_reader(str(item))
            if not text:
                continue
            if any(line.strip().startswith("|") for line in text.splitlines()):
                continue  # model tables are replaced by deterministic tables; recorded in the ledger
            text = cap_term_density(text, t("assembly.current_matrix"), DENSITY_CAP, density_state)
            body.append(text)
            body.append("")
            model_paragraphs.append({"owner": owner, "sha12": sha12(text), "chars": len(text),
                                     "evidence_ids": evidence_ids or []})
            model_texts.append({"owner": owner, "text": text, "evidence_ids": evidence_ids or []})

    body.append(t("assembly.heading_key_conclusions"))
    body.append("")
    add_model_lines(sections.get("key_summary") or [], "key_summary")
    if not sections.get("key_summary"):
        body += [t("assembly.key_summary_absent"),
                 t("assembly.core_contrast_count") % len(plan["tasks"]), ""]

    body += _stratified_lines(run)
    body.append(t("assembly.heading_data"))
    body.append("")
    body += _data_section(run, cond, run_evidence)
    body.append("")

    heading = 2
    matched: set = set()
    exclusion_state: Dict[str, bool] = {}
    merged_alias_sections: List[Dict[str, Any]] = []
    for task in plan["tasks"]:
        rec = task["contrast"]
        body.append(t("assembly.heading_task") % (heading, task["order"], rec.get("claim_title") or rec.get("display_contrast")))
        body.append("")
        model_tasks, route = _match_task_sections(sections, rec)
        if model_tasks:
            for item in model_tasks:
                matched.add(id(item))
                add_model_lines(item.get("question") or [], "task_question", [rec.get("id")])
            if len(model_tasks) > 1:
                merged_alias_sections.append({
                    "canonical": str(rec.get("claim_id") or rec.get("display_contrast") or ""),
                    "display_contrast": str(rec.get("display_contrast") or ""),
                    "sections": [str(item.get("match") or "").strip() for item in model_tasks],
                    "route": route})
        body += _contrast_block(task, cond)
        body.append("")
        unit_lines = _unit_sensitivity_lines(run, rec, index)
        if unit_lines:
            body += unit_lines
            body.append("")
        body += _candidate_block(task, index, cond, limit=8, total=len(task["candidates"]))
        body.append("")
        body += _module_block(task, index, limit=6, total=len(task["modules"]))
        if task["modules"]:
            body.append("")
        # R21/F2: the overlap file may be keyed by the source contrast name while the report shows
        # the opposite orientation, so every name this contrast is known by goes to the resolver.
        hint = _enrichment_hint_lines(run, [rec.get("display_contrast"), rec.get("source_contrast"),
                                            rec.get("claim_id")], index=index)
        if hint:
            body += hint
            body.append("")
        if model_tasks:
            for item in model_tasks:
                add_model_lines(item.get("findings") or [], "task_findings", [rec.get("id")])
                add_model_lines(item.get("explanation") or [], "task_explanation", [rec.get("id")])
            if len(model_tasks) > 1:
                # R21/F4: merging by name alone would hide that the model wrote the same comparison
                # in two sign coordinates; the shared direction statement keeps the numbers readable.
                labels = [str(item.get("match") or "").strip() for item in model_tasks]
                display = str(rec.get("display_contrast") or "")
                arms = rec.get("groups") or []
                flipped = [label for label in labels if _reversed_arms(label, display)]
                note = (t("assembly.merged_alias_note")
                        % (len(labels), "、".join(_cell(label) for label in labels), _cell(display)))
                if len(arms) >= 2 and arms[0]:
                    note += t("assembly.positive_means") % _cell(arms[0])
                note += "。"
                if flipped:
                    note += (t("assembly.merged_flipped_note")
                             % "、".join(_cell(label) for label in flipped))
                else:
                    note += t("assembly.merged_note_aligned")
                body.append(note)
                body.append("")
        else:
            body += [t("assembly.task_section_unmatched"), ""]
        body += _exclusion_lines(run, seen=exclusion_state)
        body.append(_verdict_line(task, cond))
        body.append("")
        info = joint_filter.get(str(rec.get("display_contrast"))) or {}
        if info.get("status") == "ok" and declared_mismatch(info):
            body.append(_joint_note_line(rec, info))
            body.append("")
        heading += 1

    unbound: List[Dict[str, Any]] = []
    for task in sections.get("tasks") or []:
        if id(task) in matched:
            continue
        title = str(task.get("match") or "").strip() or t("assembly.unbound_section_default")
        body.append("## %d. %s" % (heading, title))
        body.append("")
        pointer = _contrast_pointer(title, index)
        if pointer:
            body.append(pointer)
            body.append("")
        unbound.append({"heading": title, "route": "unbound", "contrast_pointer": bool(pointer)})
        add_model_lines((task.get("question") or []) + (task.get("findings") or [])
                        + (task.get("explanation") or []), "unbound_section")
        heading += 1

    body += _concordance_lines(run)
    body += _dose_trend_lines(run)
    body += _enrichment_ora_lines(run, index=index)
    body += _enrichment_boundary_lines(run, any(t("assembly.enrichment_hint_heading") in line for line in body))
    figure_lines = _figure_reference_lines(run, index)
    if figure_lines:
        body.append(t("assembly.heading_figures") % heading)
        body.append("")
        body.append(t("assembly.figures_narrative_note"))
        body.append("")
        body += figure_lines
        body.append("")
        heading += 1
    evidence_tables = _evidence_table_lines(run, index)
    if evidence_tables:
        body.append(t("assembly.heading_evidence_tables") % heading)
        body.append("")
        body += evidence_tables
        heading += 1
    checks = run_consistency_checks(run, index, cond, model_texts, joint=joint_filter)
    body.append(t("assembly.heading_run_parameters") % heading)
    body.append("")
    body += _run_evidence_binding_lines(run_evidence)
    body.append("")
    note_lines = _consistency_note_lines(checks)
    if note_lines:
        body.append(t("assembly.heading_consistency_note"))
        body.append("")
        body += note_lines
        body.append("")
    heading += 1
    body.append(t("assembly.heading_summary") % heading)
    body.append("")
    add_model_lines(sections.get("summary") or [], "summary")
    heading += 1
    body.append(t("assembly.heading_assets") % heading)
    body.append("")
    body += _asset_lines(run, index)
    heading += 1
    body.append(t("assembly.heading_boundary") % heading)
    body.append("")
    add_model_lines(sections.get("boundary") or [], "boundary_model")
    if sections.get("extra"):
        # content the assembler could not bind to a section must still be visible to a reader
        body.append(t("assembly.heading_extra_unbound"))
        body.append("")
        add_model_lines(sections.get("extra") or [], "extra_unbound")
    body += [t("assembly.boundary_matrix_only"),
             t("assembly.boundary_offline_enrichment"),
             t("assembly.boundary_external_annotation"),
             t("assembly.boundary_enrichment_states"),
             t("assembly.boundary_table_values"), ""]

    report = "\n".join(body).rstrip() + "\n"
    ledger = _content_ledger_v3(blocks, report, model_paragraphs, index)
    try:
        binding = {"run": str(run), "report_sha12": sha12(report), "run_evidence": run_evidence,
                   "report_sha256": hashlib.sha256(normalise_report_text(report).encode("utf-8")).hexdigest(),
                   "hash_normalisation": REPORT_HASH_NORMALISATION,
                   "hash_stage": "assembly (re-bind after any later revision)",
                   "joint_filter": joint_filter, "consistency_checks": checks,
                   "note": t("assembly.binding_note")}
        (Path(run) / "report_evidence_binding.json").write_text(
            json.dumps(binding, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    manifest = {
        "mode": plan["mode"], "run": str(run), "report_sha12": sha12(report), "chars": len(report),
        "report_sha256": hashlib.sha256(normalise_report_text(report).encode("utf-8")).hexdigest(),
        "hash_normalisation": REPORT_HASH_NORMALISATION,
        "model_chars": sum(p["chars"] for p in model_paragraphs),
        "deterministic_chars": len(report) - sum(p["chars"] for p in model_paragraphs),
        "model_paragraphs": model_paragraphs, "unbound_sections": unbound,
        "merged_alias_sections": merged_alias_sections,
        "conditions": cond, "threshold_text": threshold_text(cond), "effect_column": effect_column(cond),
        "evidence_index_sha12": sha12(json.dumps(_serialisable_index(index), ensure_ascii=False,
                                                 sort_keys=True, default=str)),
        "headings": [line for line in report.splitlines() if line.startswith("## ")],
        "content_ledger": ledger,
        "binding_report": bindings, "summary_type": sections.get("summary_type", "free_text"),
        "counts": {"contrasts": len(index["contrasts"]), "candidates": len(index["candidates"]),
                   "modules": len(index["modules"]), "duplicate_keys": len(index["duplicates"])},
        "run_evidence": run_evidence["items"],
        "run_evidence_status": {"n_items": len(run_evidence["items"]),
                                "not_recorded": sum(1 for x in run_evidence["items"]
                                                    if x["status"] != t("assembly.status_recorded")),
                                "imputation": run_evidence["imputation"],
                                "input_missing_rate": run_evidence["input_missing_rate"],
                                "analysed_missing_rate": run_evidence["analysed_missing_rate"]},
        "consistency_checks": checks,
    }
    return {"report": report, "manifest": manifest, "plan": plan}


def _content_ledger(blocks: List[Dict[str, Any]], report: str, model_paragraphs: List[Dict[str, Any]],
                    index: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What happened to every source block: preserved, replaced by a table, kept as unbound, empty."""
    kept = {p["sha12"] for p in model_paragraphs}
    ledger = []
    for block in blocks:
        if block["kind"] == "table":
            # a table may only be called "replaced" when its distinctive content is really present
            # in the finished report; otherwise the unmatched tokens are listed for manual review
            tokens = [t for t in re.findall(r"\d+(?:\.\d+)?", block["text"]) if len(t) >= 2]
            unmatched = sorted({t for t in tokens if t not in report})[:10]
            if unmatched:
                disposition = "replaced_with_unmatched_content"
                reason = t("assembly.ledger_table_missing") % ", ".join(unmatched)
            else:
                disposition, reason = "replaced_by_deterministic_table", t("assembly.ledger_table_all_found")
            ledger.append({"index": block["index"], "heading": block["heading"], "kind": block["kind"],
                           "sha12": block["sha12"], "chars": block["chars"],
                           "disposition": disposition, "reason": reason,
                           "unmatched_tokens": unmatched})
            continue
        elif block["sha12"] in kept:
            disposition, reason = "preserved", t("assembly.ledger_preserved")
        else:
            disposition, reason = "dropped", t("assembly.ledger_dropped")
        ledger.append({"index": block["index"], "heading": block["heading"], "kind": block["kind"],
                       "sha12": block["sha12"], "chars": block["chars"],
                       "disposition": disposition, "reason": reason})
    return ledger

# --------------------------------------------------- request-side evidence index (R17, production)


def _request_index_title() -> str:
    """Heading of the request-side fact list."""
    return (
        t("assembly.request_index_title")
    )


def _request_index_intro() -> str:
    """How the request-side fact list is meant to be read."""
    return (
        t("assembly.request_index_intro")
    )


def _table_cell(value: Any, limit: int = 120) -> str:
    text = _fmt(value).replace("|", "/").replace(chr(10), " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _figure_titles(run: Path, index: Dict[str, Any], limit: int = 60) -> List[str]:
    """Figure number, title and the numeric conclusion, exactly as the report prints them.

    The conclusion text is included so the model can state the same measured numbers the figure
    section of the report carries instead of reporting that no coordinates were provided.
    """
    titles: List[str] = []
    for line in _figure_reference_lines(run, index):
        text = str(line).strip().lstrip("-").strip()
        match = re.match(t("assembly.figure_line_parse_re"), text)
        if match:
            conclusion = text[match.end():].lstrip("：: ").strip()
            titles.append(("%s %s：%s" % (match.group(1), match.group(2).strip(), conclusion))[:220])
    if not titles:  # figures exist on disk but the manifest could not be read
        try:
            pngs = sorted(p.name for p in (Path(run) / "visualize_results").glob("*.png"))
        except Exception:  # noqa: BLE001
            pngs = []
        titles = [t("assembly.figure_file_fallback") % name for name in pngs[:limit]]
    return titles[:limit]


def canonical_task_ids(run: Path) -> List[str]:
    """One task id per scientific task: the display contrast, which names both arms in report order.

    R21/F4: the structured request used to offer every alias of a comparison (the run's claim id,
    the display name and the source-table name) as if each were a task of its own, and the model
    answered each one - the same comparison then appeared three times in one report. Aliases stay
    valid on input (the binder matches them all) but are no longer offered as tasks.
    """
    ids: List[str] = []
    for rec in build_evidence_index(Path(run))["contrasts"]:
        canonical = str(rec.get("display_contrast") or rec.get("claim_id") or "").strip()
        if canonical and canonical not in ids:
            ids.append(canonical)
    return ids


def _matrix_shape_of(path: Path) -> Dict[str, float]:
    """Min and max of a long protein x sample matrix CSV; empty when the file is absent."""
    import csv as _csv
    path = Path(path)
    if not path.exists():
        return {}
    low = None
    high = None
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = _csv.reader(handle)
            header = next(reader)
            start = 2 if len(header) > 2 else 1
            for row in reader:
                for cell in row[start:]:
                    try:
                        value = float(cell)
                    except (TypeError, ValueError):
                        continue
                    if value != value:
                        continue
                    low = value if low is None else min(low, value)
                    high = value if high is None else max(high, value)
    except Exception:  # noqa: BLE001
        return {}
    if low is None:
        return {}
    return {"min": low, "max": high}

_TRANSFORM_RECORD_CACHE: Dict[str, Dict[str, Any]] = {}


def read_matrix_transform_record(run: Path) -> Dict[str, Any]:
    """What the matrix-building step did to this run's matrix, from the run's own records.

    R28: a reader-facing preprocessing statement has to come from what executed, not from an inferred
    flag. The step writes processed_proteins/matrix_transform_record.json; runs made before that file
    existed still carry the calibration tool result in run_events.jsonl, which is used as a named
    fallback. Neither being present means not recorded, which is not the same as not executed.
    """
    run = Path(run)
    cache_key = str(run)
    if cache_key in _TRANSFORM_RECORD_CACHE:
        return _TRANSFORM_RECORD_CACHE[cache_key]
    direct = run / "processed_proteins" / "matrix_transform_record.json"
    data = read_json(direct, {})
    if isinstance(data, dict) and data:
        data = dict(data)
        data.setdefault("record_source", "processed_proteins/matrix_transform_record.json")
        data.setdefault("derived", False)
        _TRANSFORM_RECORD_CACHE[cache_key] = data
        return data
    events = run / "run_events.jsonl"
    if not events.exists():
        return {}
    summary = None
    try:
        with events.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "protein_quant_combat_calibration" not in line:
                    continue
                try:
                    node = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                candidate = node.get("result_summary") if isinstance(node, dict) else None
                if (isinstance(candidate, dict)
                        and candidate.get("analysis_type") == "protein_quant_combat_calibration"):
                    summary = candidate
    except Exception:  # noqa: BLE001
        return {}
    if not summary:
        _TRANSFORM_RECORD_CACHE[cache_key] = {}
        return {}
    norm = summary.get("normalization") or {}
    method = str(norm.get("method") or "")
    filt = summary.get("protein_filtering") or {}
    batch = summary.get("batch_correction") or {}
    inputs = summary.get("input") or {}
    outputs = (summary.get("outputs") or {}).get("files") or {}
    raw = _matrix_shape_of(run / "processed_proteins" / "ProteinQuant_Filtered.csv")
    analysed = _matrix_shape_of(run / "processed_proteins" / "ProQuant_Normalized.csv")
    log2_applied = None
    basis = ""
    if raw and analysed:
        expected_ceiling = 0.0
        try:
            import math as _math
            expected_ceiling = _math.log2(raw["max"] + 1.0) * 1.05 + 0.5
        except Exception:  # noqa: BLE001
            expected_ceiling = 0.0
        log2_applied = bool(expected_ceiling and analysed["max"] <= expected_ceiling
                            and analysed["max"] < raw["max"])
        basis = (t("assembly.transform_basis")
                 % (_fmt(analysed["max"]), _fmt(raw["max"]), _fmt(expected_ceiling)))
    built = {
        "record_type": "matrix_transform_record",
        "record_source": t("assembly.transform_record_derived_source"),
        "derived": True,
        "stage": "protein_quant_combat_calibration",
        "input": {"protein_quant_file": Path(str(inputs.get("protein_quant_file") or "")).name,
                  "protein_rows": inputs.get("initial_protein_rows"),
                  "samples": inputs.get("initial_samples")},
        "protein_filtering": {"rule": t("assembly.filtering_rule") % (_fmt(filt.get("reproducibility_cutoff")),
                                                                 filt.get("filtering_strategy") or t("assembly.not_recorded")),
                              "reproducibility_cutoff": filt.get("reproducibility_cutoff"),
                              "strategy": filt.get("filtering_strategy"),
                              "n_proteins_retained": filt.get("number_of_proteins_after_filtering")},
        "imputation": {"applied": bool("half-min" in method.lower()),
                       "method": method or t("assembly.not_recorded")},
        "log2_transform": {"applied": log2_applied,
                           "form": "log2(x+1)",
                           "recorded_rule": method or t("assembly.not_recorded"),
                           "applied_basis": basis},
        "batch_correction": {"combat_success": bool(batch.get("combat_success")),
                             "new_batch_correction_applied": bool(batch.get("new_batch_correction_applied")),
                             "batch_column": batch.get("batch_column"),
                             "effective_matrix_status": batch.get("effective_matrix_status"),
                             "reason": batch.get("fallback_reason")},
        "analysed_matrix": outputs.get("normalized_matrix") or "ProQuant_Normalized.csv",
        "compatibility_matrix": outputs.get("compatibility_matrix") or "ProteinQuant_ComBat.csv",
    }
    _TRANSFORM_RECORD_CACHE[cache_key] = built
    return built


def _unit_effect_summary(path: Path) -> Dict[str, Any]:
    """Range of the per-protein unit-level effects and how many 95% intervals exclude zero.

    Read from the unit-level table of one contrast. This is a summary of rows that were actually
    tested; nothing is recomputed and rows without a test carry no effect.
    """
    try:
        rows = _read_evidence_rows(Path(path))
    except Exception:  # noqa: BLE001
        return {}
    effects: List[float] = []
    scale_states = set()
    excludes = 0
    for row in rows:
        if str(row.get("tested") or "").strip().lower() not in ("true", "1"):
            continue
        # R29: the same table is written twice (processed_proteins/ and evaluation_evidence_ext/).
        # The column name follows the resolved effect scale, so a legacy copy may still carry
        # effect_units; the scale state each row declares is collected, so a copy whose scale
        # contradicts the run-resolved scale is reported as a conflict instead of being read silently.
        raw_effect = row.get("log2FC_units")
        if raw_effect in (None, ""):
            raw_effect = row.get("effect_units")
        _row_scale = str(row.get("effect_scale_state") or "").strip()
        if _row_scale:
            scale_states.add(_row_scale)
        try:
            effect = float(raw_effect)
            low = float(row.get("ci_low"))
            high = float(row.get("ci_high"))
        except (TypeError, ValueError):
            continue
        if effect != effect or low != low or high != high:  # NaN guard
            continue
        effects.append(effect)
        if (low > 0 and high > 0) or (low < 0 and high < 0):
            excludes += 1
    if not effects:
        return {}
    ordered = sorted(effects)
    return {"n": len(effects), "min": min(effects), "max": max(effects),
            "median": ordered[len(ordered) // 2], "ci_excludes_zero": excludes,
            "scale_states": sorted(scale_states)}


def _unit_layer_index(run: Path, index: Dict[str, Any],
                      limma: Dict[str, Any]) -> Dict[str, Any]:
    """The unit-level results as request evidence, one entry per contrast.

    R28: the unit-level sensitivity analysis ran on every contrast of this run, but its results only
    reached the finished report when the hybrid assembler happened to add them; the request that the
    model answers did not mention them at all. This digest carries the estimand, the unit semantics,
    the effective unit counts, the effect column together with its per-protein intervals, the screen
    and the reason a contrast could not be tested, so a report can interpret observation-level and
    unit-level results as what they are.
    """
    run = Path(run)
    layer: Dict[str, Any] = {"contrasts": [], "n_contrasts": 0, "n_completed": 0, "n_blocked": 0,
                             "effect_column": "", "effect_scale": "", "summary_file": ""}
    if not isinstance(limma, dict):
        return layer
    summary_rows = _ext_csv_rows(run, "unit_sensitivity_summary.csv")
    by_contrast = {str(row.get("contrast") or "").strip(): row for row in summary_rows}
    unit_record = None
    # R28: every contrast the analysis computed carries a unit-level record, including the ones whose
    # unit structure cannot support a test. Listing only the round's task contrasts would hide exactly
    # those reasons, so the round's tasks come first and the remaining contrasts follow as background.
    key_to_task: Dict[str, str] = {}
    task_keys: List[str] = []
    for rec in index.get("contrasts") or []:
        names = [rec.get("display_contrast"), rec.get("source_contrast"), rec.get("claim_id")]
        matched_key, _ = _match_contrast_key(limma, names)
        if matched_key:
            key_to_task[matched_key] = str(rec.get("display_contrast") or rec.get("claim_id") or "")
            if matched_key not in task_keys:
                task_keys.append(matched_key)
    ordered_keys = task_keys + [name for name in limma.keys() if name not in task_keys]
    for key in ordered_keys:
        node = limma.get(key) if isinstance(limma.get(key), dict) else {}
        unit = node.get("unit_sensitivity")
        if not isinstance(unit, dict) or not unit:
            continue
        unit_record = unit
        in_task_list = key in key_to_task
        canonical = key_to_task.get(key) or str(key)
        status = str(unit.get("status") or "")
        route = str(unit.get("analysis_route") or "")
        blocked_reason = ""
        if status != "completed":
            blocked_reason = _unit_reason_zh(unit)
        estimand = ""
        if route == "within_unit_paired":
            estimand = t("assembly.estimand_within_unit_observed")
        elif route == "between_unit_independent":
            estimand = t("assembly.estimand_between_unit_independent")
        elif status != "completed":
            estimand = t("assembly.estimand_none")
        exclusion = unit.get("unit_exclusion") or {}
        excluded = sorted(set(exclusion.get("pooled_source") or [])
                          | set(exclusion.get("unconfirmed_source") or []))
        summary_row = by_contrast.get(key or "") or by_contrast.get(canonical) or {}
        effect_column = str(unit.get("effect_column") or "")
        effect_scale = unit.get("effect_scale") or {}
        detail_name = "unit_sensitivity_%s.csv" % (key or canonical)
        detail_path = run / "processed_proteins" / detail_name
        if not detail_path.exists():
            detail_path = run / "evaluation_evidence_ext" / detail_name
        effect_summary = _unit_effect_summary(detail_path) if status == "completed" else {}
        min_p = unit.get("min_p_value")
        if min_p is None:
            min_p = summary_row.get("min_p_value")
        min_adj = unit.get("min_adj_p_value")
        if min_adj is None:
            min_adj = summary_row.get("min_adj_p_value")
        fdr_only = unit.get("n_passing_fdr_only")
        if fdr_only is None:
            fdr_only = unit.get("n_passing_fdr")
        layer["contrasts"].append({
            "task_id": canonical,
            "display_contrast": canonical,
            "limma_key": str(key),
            "in_task_list": in_task_list,
            "direction": unit.get("direction"),
            "status": status,
            "status_zh": t("assembly.unit_status_completed") if status == "completed" else t("assembly.unit_status_blocked"),
            "estimand": estimand,
            "analysis_unit": unit.get("analysis_unit"),
            "unit_source": _unit_source_label(unit.get("unit_column_source")),
            "design_class": unit.get("design_class"),
            "n_units_a": unit.get("n_units_a"), "n_units_b": unit.get("n_units_b"),
            "n_units_shared": unit.get("n_units_shared"),
            "n_units_a_only": unit.get("n_units_a_only"),
            "n_units_b_only": unit.get("n_units_b_only"),
            "n_pairs": unit.get("n_pairs"),
            "excluded_units": excluded,
            "n_proteins_tested": unit.get("n_proteins_tested"),
            "n_proteins_not_tested": unit.get("n_proteins_not_tested"),
            "n_passing_fdr_only": fdr_only,
            "n_passing_joint": unit.get("n_passing_joint"),
            "logfc_threshold": unit.get("logfc_threshold"),
            "min_p_value": min_p, "min_adj_p_value": min_adj,
            "effect_column": effect_column,
            "effect_scale_state": (effect_scale or {}).get("state"),
            "effect_scale_source": (effect_scale or {}).get("source"),
            "effect_summary": effect_summary,
            "blocked_reason": blocked_reason,
            "reasons": [str(item) for item in (unit.get("reasons") or [])][:3],
        })
    contrasts = layer["contrasts"]
    layer["n_contrasts"] = len(contrasts)
    layer["n_completed"] = sum(1 for item in contrasts if item["status"] == "completed")
    layer["n_blocked"] = sum(1 for item in contrasts if item["status"] != "completed")
    if unit_record:
        layer["effect_column"] = str(unit_record.get("effect_column") or "")
        layer["effect_scale"] = str((unit_record.get("effect_scale") or {}).get("label")
                                    or (unit_record.get("effect_scale") or {}).get("state") or "")
        layer["unit_semantics_note"] = unit_record.get("correction_scope") or ""
    if summary_rows:
        layer["summary_file"] = "evaluation_evidence_ext/unit_sensitivity_summary.csv"
    return layer


def build_request_evidence_index(run: Path) -> Dict[str, Any]:
    """The evidence index shared by the model request and the deterministic assembler.

    Every entry is read from this run's artifacts through the same readers the assembler uses, so
    what the model is told it has cannot drift away from what the finished report prints. Anything
    absent from the run is recorded as not-computed; nothing is filled in from another source.
    """
    run = Path(run)
    index = build_evidence_index(run)
    evidence = read_run_evidence(run)
    first = evidence.get("limma_first") or {}
    sanity = first.get("sanity_checks") or {}
    thresholds = sanity.get("thresholds_used") or {}
    design = (read_json(run / "parameters.json", {}) or {}).get("analysis_design", {}) or {}
    data: Dict[str, Any] = {"run": str(run), "contrasts": [], "stratified": {}, "overlap": {},
                            "ora": {}, "dose": {}, "concordance": {}, "exclusion": {},
                            "figures": {}, "task_ids": [], "task_aliases": {},
                            "overlap_state": "", "not_computed": []}

    params = {"matrix_stage": "", "tested_proteins": _fmt(first.get("n_proteins")),
              "model": _table_cell(first.get("method"), 80),
              "fdr": _fmt((design.get("differential") or {}).get("fdr_method")),
              "threshold": "", "samples": "", "missing_raw": "", "missing_analysed": "",
              "imputation": "；".join(evidence.get("imputation") or []) or t("assembly.not_recorded")}
    stage_text = next((x["value"] for x in evidence["items"] if x["item"] == t("assembly.item_matrix_stages")), "")
    params["matrix_stage"] = _table_cell(stage_text, 200)
    if thresholds:
        params["threshold"] = t("assembly.threshold_adj_p_and_effect") % (_fmt(thresholds.get("adj_p")),
                                                          _fmt(thresholds.get("abs_logFC")))
    if first.get("n_samples_group_a") is not None and first.get("n_samples_group_b") is not None:
        params["samples"] = t("assembly.samples_two_arms") % (
            _fmt(first.get("n_samples_group_a")), _fmt(first.get("n_samples_group_b")))
    if evidence.get("input_missing_rate"):
        params["missing_raw"] = _range_text(evidence["input_missing_rate"])
    if evidence.get("analysed_missing_rate"):
        params["missing_analysed"] = _range_text(evidence["analysed_missing_rate"])
    # R18: the report says the linear projection is missing unless the request says where the
    # coordinates live; PCA coordinates are in the QC table, UMAP coordinates are not produced.
    dimred = ""
    try:
        import csv as _csv_dim
        qc_path = run / "processed_proteins" / "qc_metrics.csv"
        if qc_path.exists():
            with qc_path.open(encoding="utf-8-sig") as handle:
                reader = _csv_dim.reader(handle)
                head = [str(cell).strip() for cell in next(reader)]
                rows_n = sum(1 for _ in reader)
            have = [name for name in ("PC1", "PC2", "PC3") if name in head]
            if have:
                dimred = (t("assembly.dimred_note")
                          % ("、".join(have), rows_n))
    except Exception:  # noqa: BLE001
        dimred = ""
    params["dimred"] = _table_cell(dimred, 200)
    # R28: the executed preprocessing is part of what the model has to be told; otherwise a report can
    # conclude "imputation was not run" from a record that merely did not mention it. These fields come
    # from the run's own execution record (see read_matrix_transform_record).
    transform = read_matrix_transform_record(run)
    if transform:
        _imp = transform.get("imputation") or {}
        _log = transform.get("log2_transform") or {}
        _filt = transform.get("protein_filtering") or {}
        if _imp.get("applied") is not None:
            if _imp.get("applied"):
                params["imputation"] = t("assembly.executed_parens_two") % (
                    _imp.get("method") or t("assembly.method_not_recorded"),
                    (t("assembly.imputed_values_suffix") % _imp.get("n_values_imputed"))
                    if _imp.get("n_values_imputed") not in (None, "") else "")
            else:
                params["imputation"] = t("assembly.status_not_executed")
        if _log.get("applied") is not None:
            params["log2_transform"] = (t("assembly.executed_parens_one") % (_log.get("form") or t("assembly.form_not_recorded"))
                                        if _log.get("applied") else t("assembly.status_not_executed"))
        if _filt:
            params["protein_filtering"] = t("assembly.protein_filtering_text") % (
                _filt.get("rule") or t("assembly.not_recorded"), _fmt(_filt.get("n_proteins_retained")))
        params["transform_record_source"] = _table_cell(str(transform.get("record_source") or ""), 140)
    else:
        params["imputation"] = t("assembly.not_recorded")
        params["log2_transform"] = t("assembly.not_recorded")
        params["transform_record_source"] = (t("assembly.note_transform_record_absent_request"))
    data["parameters"] = params

    for rec in index["contrasts"]:
        arms = rec.get("groups") or ["", ""]
        # R21/F4: one canonical task_id per scientific task. The display contrast states the two
        # arms in the order the report prints them, so it can carry the direction; the run's own
        # claim id and the source-table name stay valid input aliases but are no longer offered as
        # extra tasks (that is what made the model write the same comparison three times).
        canonical = str(rec.get("display_contrast") or rec.get("claim_id") or "").strip()
        data["contrasts"].append({
            "task_id": canonical,
            "display_contrast": rec.get("display_contrast"), "source_contrast": rec.get("source_contrast"),
            "arms": "%s / %s" % (_fmt(arms[0]), _fmt(arms[1] if len(arms) > 1 else "")),
            "positive_means": _fmt(arms[0]),
            "n_sig": _fmt(rec.get("n_sig")), "n_up": _fmt(rec.get("n_up_display_a")),
            "n_down": _fmt(rec.get("n_down_display_a")), "inverted": bool(rec.get("inverted"))})
        if canonical:
            if canonical not in data["task_ids"]:
                data["task_ids"].append(canonical)
            data["task_aliases"][canonical] = [str(name) for name in
                                              (rec.get("claim_id"), rec.get("source_contrast"))
                                              if name and str(name) != canonical]

    # R28: the unit-level sensitivity analysis is part of this run's evidence. It is read here so
    # the request the model answers and the finished report describe the same unit-level results.
    limma_all = read_json(run / "processed_proteins" / "limma_summary.json", {}) or {}
    data["unit_layer"] = _unit_layer_index(run, index, limma_all)

    rows = _ext_csv_rows(run, "stratified_contrast_summary.csv")
    named = [row for row in rows if str(row.get("level") or "").strip()]
    shown_stratified = named[:8]
    data["stratified"] = {"rows": len(named),
                          "stratifiers": sorted({str(row.get("stratifier") or "") for row in named
                                                 if row.get("stratifier")}),
                          "shown": len(shown_stratified),
                          "truncated": len(named) > len(shown_stratified),
                          "examples": [{"contrast": _table_cell(row.get("contrast"), 60),
                                        "level": _table_cell(row.get("level"), 40),
                                        "n_a": _fmt(row.get("n_a")), "n_b": _fmt(row.get("n_b")),
                                        "tested": _fmt(row.get("tested_proteins")),
                                        "n_sig": _fmt(row.get("n_sig")),
                                        "verdict": _table_cell(row.get("verdict"), 40),
                                        "note": _table_cell(row.get("note"), 60)}
                                       for row in shown_stratified]}
    if not named:
        data["not_computed"].append(t("assembly.not_computed_stratified"))

    overlap: Dict[str, Any] = {}
    # R21/F2: the request fact list and the report read the same file through the same resolver, and
    # the four outcomes stay apart instead of collapsing into "no usable entry produced".
    overlap_state, blob = _overlap_state(run)
    data["overlap_state"] = overlap_state
    if overlap_state == "ok":
        for rec in index["contrasts"]:
            names = [rec.get("display_contrast"), rec.get("source_contrast"), rec.get("claim_id")]
            contrast = str(rec.get("display_contrast") or "")
            view = _enrichment_view(blob, names)
            buckets = view.get("buckets") or {}
            arms = _arm_labels_for(view.get("matched_key") or contrast, index)
            per_db = {}
            picks = []
            for database in ENRICHMENT_DATABASES:
                node = buckets.get(database) or {}
                per_db[database] = {direction: len(node.get(direction) or {})
                                    for direction in ("upregulated", "downregulated")}
                for direction, label in (("upregulated", t("assembly.enrichment_up_direction")), ("downregulated", t("assembly.enrichment_down_direction"))):
                    if len(arms) == 2:
                        label = t("assembly.arm_higher") % (arms[0] if direction == "upregulated" else arms[1])
                    terms = [(str(term), str(genes).split("/"))
                             for genes, term in (node.get(direction) or {}).items()]
                    terms.sort(key=lambda item: (-len(item[1]), item[0]))
                    for term, members in terms[:1]:
                        picks.append({"database": database, "direction": label,
                                      "term": _table_cell(term, 80), "seed": len(members),
                                      "members": _table_cell(", ".join(members[:6]), 120)})
            total = sum(sum(counts.values()) for counts in per_db.values())
            overlap[contrast] = {"status": t("assembly.status_computed") if total else t("assembly.enrichment_status_valid_empty_contrast"),
                                 "terms": total, "file_contrast": view.get("matched_key") or "",
                                 "reversed": bool(view.get("reversed")),
                                 "per_database": per_db, "top_terms": picks}
    data["overlap"] = overlap
    if overlap_state == "unparsable":
        data["not_computed"].append(t("assembly.not_computed_overlap_unparsable"))
    elif overlap_state == "empty":
        data["not_computed"].append(t("assembly.not_computed_overlap_empty"))
    elif overlap_state == "missing":
        data["not_computed"].append(t("assembly.not_computed_overlap_missing"))

    ora_rows = _ext_csv_rows(run, "enrichment_ora.csv")
    tested = [row for row in ora_rows if str(row.get("p_value") or "").strip()]
    # R19: an executed test with nothing passing must be distinguishable from a test that was
    # never run, so the request block states the passing count explicitly.
    ora_significant = [row for row in tested if _passes_cutoff(row.get("q_value"))]
    if ora_rows:
        best = sorted(tested, key=lambda row: float(row.get("q_value") or 1))[:3] if tested else []
        data["ora"] = {"tested": len(tested), "skipped": len(ora_rows) - len(tested),
                       "significant": len(ora_significant),
                       "background": _fmt(tested[0].get("background_size")) if tested else "",
                       "matrix_proteins": _fmt(first.get("n_proteins")),
                       "query_rule": _table_cell(tested[0].get("query_definition"), 160) if tested else "",
                       "best": [{"namespace": _table_cell(row.get("namespace"), 20),
                                 "term": _table_cell(row.get("term"), 60),
                                 "hits": _fmt(row.get("hits")), "query_size": _fmt(row.get("query_size")),
                                 "p": _reader_prob(row.get("p_value")),
                                 "q": _reader_prob(row.get("q_value"))} for row in best],
                       "skipped_notes": sorted({_reader_note(row.get("note")) for row in ora_rows
                                                if str(row.get("note") or "").strip()
                                                and not str(row.get("p_value") or "").strip()})[:4]}
        # R21/F1: the request states the same query-set scope the report prints, contrast by
        # contrast, instead of one rule that only describes whichever row happened to be first.
        fallback_rows = [row for row in tested if _is_fallback_query(row)]
        significant_rows = [row for row in tested if not _is_fallback_query(row)]
        classes = []
        if significant_rows:
            classes.append({"kind": t("assembly.kind_significant_set"),
                            "rule": _table_cell(significant_rows[0].get("query_definition"), 160),
                            "rows": len(significant_rows),
                            "query_sizes": _size_range([row.get("query_size")
                                                        for row in significant_rows]),
                            "contrasts": sorted({str(row.get("contrast") or "")
                                                 for row in significant_rows})[:8]})
        if fallback_rows:
            classes.append({"kind": t("assembly.kind_fallback_set"),
                            "rows": len(fallback_rows),
                            "significant_set_sizes": _size_range(_fallback_significant_sizes(fallback_rows)),
                            "query_sizes": _size_range([row.get("query_size") for row in fallback_rows]),
                            "contrasts": sorted({str(row.get("contrast") or "")
                                                 for row in fallback_rows})[:8]})
        data["ora"]["query_classes"] = classes
    else:
        data["not_computed"].append(t("assembly.not_computed_ora"))

    dose_rows = _ext_csv_rows(run, "dose_trend.csv")
    if dose_rows:
        modules = [row for row in dose_rows if str(row.get("level")) == "module"]
        proteins = [row for row in dose_rows if str(row.get("level")) == "protein"]
        significant = [row for row in proteins if str(row.get("significant")) == "yes"]
        module_significant = [row for row in modules if _passes_cutoff(row.get("q_value"))]
        usable_q = [float(row.get("q_value")) for row in proteins
                    if str(row.get("q_value") or "").strip()]
        data["dose"] = {"module_rows": len(modules), "protein_rows": len(proteins),
                        "protein_significant": len(significant),
                        "module_significant": len(module_significant),
                        "best_q": _reader_prob(min(usable_q)) if usable_q else "",
                        "modules": [{"drug": _table_cell(row.get("drug"), 40),
                                     "stratum": _table_cell(row.get("stratum"), 60),
                                     "name": _table_cell(row.get("name"), 40),
                                     "n": _fmt(row.get("n_samples")), "rho": _fmt(row.get("rho")),
                                     "p": _reader_prob(row.get("p_value")),
                                     "q": _reader_prob(row.get("q_value")),
                                     "direction": _table_cell(row.get("direction"), 40)}
                                    for row in modules[:12]],
                        "skipped_notes": sorted({_reader_note(row.get("note"))
                                                 for row in dose_rows
                                                 if str(row.get("note") or "").strip()})[:4]}
    else:
        data["not_computed"].append(t("assembly.not_computed_dose_trend"))

    concordance_rows = _ext_csv_rows(run, "contrast_concordance.csv")
    data["concordance"] = {"pairs": len(concordance_rows),
                           "status": t("assembly.status_computed") if concordance_rows else t("assembly.status_unusable")}
    if not concordance_rows:
        data["not_computed"].append(t("assembly.not_computed_concordance"))

    trace_rows = _ext_csv_rows(run, "candidate_exclusion_trace.csv")
    if trace_rows:
        tokens = _design_tokens(run)
        excluded = sorted({str(row.get("candidate")).strip() for row in trace_rows
                           if str(row.get("status") or "").strip().lower() == "absent"
                           and str(row.get("candidate") or "").strip()
                           and str(row.get("candidate")).strip().lower() not in tokens})
    else:
        excluded = []
        data["not_computed"].append(t("assembly.asset_candidate_exclusion_trace"))
    data["exclusion"] = {"excluded": excluded, "status": t("assembly.status_computed") if trace_rows else t("assembly.status_not_computed")}

    titles = _figure_titles(run, index)
    data["figures"] = {"count": len(titles), "titles": titles[:24]}
    if not titles:
        data["not_computed"].append(t("assembly.not_computed_figures"))
    return data


def format_request_evidence_index(data: Dict[str, Any]) -> str:
    """Render the shared evidence index as the request-side markdown block."""
    params = data.get("parameters") or {}

    def status(value: Any, ok: str = t("assembly.status_computed")) -> str:
        return ok if str(value or "").strip() else t("assembly.status_not_computed")

    lines = [_request_index_title(), "", _request_index_intro(), "", t("assembly.heading_data_preprocessing"), "",
             t("assembly.header_request_item"), "|---|---|---|",
             t("assembly.row_matrix_stage") % (_table_cell(params.get("matrix_stage"), 200),
                                                  status(params.get("matrix_stage"))),
             t("assembly.row_matrix_size") % (params.get("tested_proteins"),
                                                        status(params.get("tested_proteins"))),
             t("assembly.row_samples_per_group") % (_table_cell(params.get("samples"), 120),
                                         status(params.get("samples"))),
             t("assembly.row_missing_rate_raw") % (params.get("missing_raw"),
                                                    status(params.get("missing_raw"))),
             t("assembly.row_missing_rate_analysed") % (params.get("missing_analysed"),
                                                    status(params.get("missing_analysed"))),
             t("assembly.row_imputation") % (
                 params.get("imputation") or t("assembly.not_recorded"), status(params.get("imputation"), t("assembly.status_recorded"))),
             t("assembly.row_log2_transform") % (
                 params.get("log2_transform") or t("assembly.not_recorded"), status(params.get("log2_transform"), t("assembly.status_recorded"))),
             t("assembly.row_protein_filtering") % (
                 params.get("protein_filtering") or t("assembly.not_recorded"),
                 status(params.get("protein_filtering"), t("assembly.status_recorded"))),
             t("assembly.row_transform_record_source") % _table_cell(
                 params.get("transform_record_source") or t("assembly.not_recorded"), 140),
             t("assembly.row_statistical_model") % (params.get("model"), status(params.get("model"), t("assembly.status_recorded"))),
             t("assembly.row_fdr") % (params.get("fdr"), status(params.get("fdr"), t("assembly.status_recorded"))),
             t("assembly.row_effective_threshold") % (params.get("threshold"),
                                          status(params.get("threshold"), t("assembly.status_recorded"))),
             t("assembly.row_dimred_coordinates") % (params.get("dimred") or t("assembly.not_recorded"),
                                            status(params.get("dimred"), t("assembly.status_computed"))),
             ""]
    if data.get("contrasts"):
        lines += [t("assembly.heading_core_contrasts"), "",
                  t("assembly.header_task_id_contrast"),
                  "|---|---|---|---:|---:|---:|---|"]
        for rec in data["contrasts"]:
            direction = t("assembly.request_direction_by") % rec.get("display_contrast")
            if rec.get("inverted"):
                direction = t("assembly.request_direction_by_inverted") % (rec.get("display_contrast"),
                                                           rec.get("source_contrast"))
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                rec.get("task_id"), rec.get("display_contrast"), rec.get("arms"),
                rec.get("n_sig"), rec.get("n_up"), rec.get("n_down"), direction))
        lines.append("")
    unit_layer = data.get("unit_layer") or {}
    unit_items = list(unit_layer.get("contrasts") or [])
    if unit_items:
        lines += [t("assembly.heading_unit_layer_computed"), "",
                  t("assembly.unit_layer_intro")
                  % (unit_layer.get("n_contrasts"), unit_layer.get("n_completed"),
                     unit_layer.get("n_blocked")), "",
                  t("assembly.header_unit_layer"),
                  "|---|---|---|---|---|---|---|---|---:|---:|---:|"]
        for item in unit_items:
            lines.append("| %s | %s | %s | %s | %s | %s | %s/%s/%s/%s/%s | %s/%s | %s | %s | %s |" % (
                item.get("task_id"), t("assembly.yes_value") if item.get("in_task_list") else t("assembly.no_background_contrast"),
                item.get("status_zh"), item.get("analysis_unit"),
                item.get("unit_source"), item.get("estimand"),
                _fmt(item.get("n_units_a")), _fmt(item.get("n_units_b")),
                _fmt(item.get("n_units_shared")), _fmt(item.get("n_units_a_only")),
                _fmt(item.get("n_units_b_only")),
                _fmt(item.get("n_proteins_tested")), _fmt(item.get("n_proteins_not_tested")),
                _fmt(item.get("n_passing_fdr_only")), _fmt(item.get("n_passing_joint")),
                _fmt(item.get("min_adj_p_value"))))
        if any(not item.get("in_task_list") for item in unit_items):
            lines += ["", t("assembly.unit_background_rows_note")]
        lines += ["", t("assembly.unit_effect_column_note")
                      % (unit_layer.get("effect_column") or t("assembly.not_recorded"),
                         unit_layer.get("effect_scale") or t("assembly.not_recorded"))]
        for item in unit_items:
            summary = item.get("effect_summary") or {}
            if not summary:
                continue
            lines.append(t("assembly.unit_effect_summary_line")
                         % (item.get("task_id"), _fmt(summary.get("n")),
                            _fmt(summary.get("min")), _fmt(summary.get("max")),
                            _fmt(summary.get("median")), _fmt(summary.get("ci_excludes_zero"))))
        units_excluded = sorted({str(label) for item in unit_items
                                 for label in (item.get("excluded_units") or [])})
        if units_excluded:
            lines.append(t("assembly.units_excluded_bullet")
                         % "、".join(_cell(x) for x in units_excluded))
        for item in unit_items:
            if item.get("blocked_reason"):
                lines.append(t("assembly.blocked_contrast_line") % (item.get("task_id"),
                                                                    item.get("blocked_reason")))
        lines += [t("assembly.unit_citation_rules"), ""]
    else:
        lines += [t("assembly.heading_unit_layer"), "",
                  t("assembly.unit_layer_absent"), ""]
    strat = data.get("stratified") or {}
    if strat.get("rows"):
        lines += [t("assembly.heading_stratified_computed")
                  % (strat["rows"], "、".join(strat.get("stratifiers") or [t("assembly.not_recorded")])),
                  "", t("assembly.header_stratified_request"),
                  "|---|---|---:|---:|---:|---:|---|---|"]
        for row in strat.get("examples") or []:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row.get("contrast"), row.get("level"), row.get("n_a"), row.get("n_b"),
                row.get("tested"), row.get("n_sig"), row.get("verdict") or t("assembly.not_recorded"),
                row.get("note") or ""))
        shown = len(strat.get("examples") or [])
        if strat.get("truncated") or strat["rows"] > shown:
            # R21: the listing cap is display metadata. A model read the old wording as a finding
            # and wrote "分层变量包括 Cluster、Type2 及未列出的其余一行" into a reader-facing sentence.
            lines += ["", t("assembly.stratified_coverage_note")
                          % (shown, strat["rows"], max(0, strat["rows"] - shown))]
        lines += ["", t("assembly.stratified_scope_note_short"), ""]
    else:
        lines += [t("assembly.heading_stratified_summary"), "", t("assembly.not_computed_stratified_line"), ""]
    overlap = data.get("overlap") or {}
    if overlap:
        lines += [t("assembly.heading_overlap_exploratory"), "",
                  t("assembly.overlap_source_note"), "",
                  t("assembly.header_overlap"),
                  "|---|---|---:|---|---|"]
        for contrast, info in overlap.items():
            detail = "；".join(
                "%s %s" % (db, "/".join("%s %d" % (key, value) for key, value in counts.items()))
                for db, counts in (info.get("per_database") or {}).items())
            lines.append("| %s | %s | %d | %s（%s） | %s |" % (
                contrast, info.get("status"), info.get("terms"),
                info.get("file_contrast") or t("assembly.not_recorded"),
                t("assembly.arm_order_reversed") if info.get("reversed") else t("assembly.arm_order_matches"), detail))
        picks = [pick for info in overlap.values() for pick in (info.get("top_terms") or [])]
        if picks:
            lines += ["", t("assembly.overlap_pick_rule"), "",
                      t("assembly.header_overlap_picks"),
                      "|---|---|---|---|---:|---|"]
            for contrast, info in overlap.items():
                for pick in info.get("top_terms") or []:
                    lines.append("| %s | %s | %s | %s | %s | %s |" % (
                        contrast, pick.get("database"), pick.get("direction"), pick.get("term"),
                        pick.get("seed"), pick.get("members")))
            lines.append("")
        lines += ["", t("assembly.overlap_exploratory_note"), ""]
    else:
        state_note = {"missing": t("assembly.overlap_state_missing"),
                      "unparsable": t("assembly.overlap_state_unparsable"),
                      "empty": t("assembly.overlap_state_empty")}.get(
            str(data.get("overlap_state") or ""), t("assembly.overlap_state_unavailable"))
        lines += [t("assembly.heading_overlap_plain"), "",
                  t("assembly.overlap_status_line") % state_note, ""]
    ora = data.get("ora") or {}
    if ora:
        lines += [t("assembly.heading_ora_hypergeometric"), "",
                  t("assembly.ora_tested_counts") % (ora.get("tested"), ora.get("skipped")),
                  t("assembly.ora_background_sizes")
                  % (ora.get("background") or t("assembly.not_recorded"), ora.get("matrix_proteins") or t("assembly.not_recorded")),
                  t("assembly.ora_significant_count")
                  % ora.get("significant"),
                  ""]
        classes = ora.get("query_classes") or []
        if classes:
            lines += [t("assembly.query_classes_intro")]
            for item in classes:
                if item.get("kind", "").startswith(t("assembly.kind_significant_set")):
                    lines.append(t("assembly.ora_class_significant_line")
                                 % (item.get("kind"), item.get("rule") or t("assembly.not_recorded"),
                                    item.get("query_sizes") or t("assembly.not_recorded"),
                                    "、".join(item.get("contrasts") or []) or t("assembly.not_recorded")))
                else:
                    lines.append(t("assembly.ora_class_fallback_line")
                                 % (item.get("kind"), item.get("significant_set_sizes") or t("assembly.not_recorded"),
                                    item.get("query_sizes") or t("assembly.not_recorded"),
                                    "、".join(item.get("contrasts") or []) or t("assembly.not_recorded")))
            lines += ["", t("assembly.query_scope_warning"),
                      ""]
        elif ora.get("query_rule"):
            lines += [t("assembly.query_definition_line") % ora.get("query_rule"), ""]
        if ora.get("best"):
            lines += [t("assembly.ora_best_cap") % (len(ora["best"]), ora.get("tested")), ""]
            lines += [t("assembly.header_ora_best"), "|---|---|---:|---:|---:|---:|"]
            for row in ora["best"]:
                lines.append("| %s | %s | %s | %s | %s | %s |" % (
                    row.get("namespace"), row.get("term"), row.get("hits"), row.get("query_size"),
                    row.get("p"), row.get("q")))
            lines.append("")
        for note in ora.get("skipped_notes") or []:
            lines.append(t("assembly.skipped_test_line") % note)
        lines.append("")
    else:
        lines += [t("assembly.heading_ora_hypergeometric"), "", t("assembly.not_computed_line"), ""]
    dose = data.get("dose") or {}
    if dose:
        lines += [t("assembly.heading_dose_trend_computed"), "",
                  t("assembly.dose_summary_line")
                  % (dose.get("module_rows"), dose.get("protein_rows"),
                     dose.get("protein_significant"), dose.get("best_q") or t("assembly.not_recorded")),
                  t("assembly.dose_module_significant")
                  % dose.get("module_significant"),
                  ""]
        for note in dose.get("skipped_notes") or []:
            lines.append(t("assembly.skipped_test_line") % note)
        lines.append("")
        if dose.get("modules"):
            lines += [t("assembly.dose_module_cap")
                      % (len(dose["modules"]), dose.get("module_rows")), ""]
            lines += [t("assembly.header_dose_module"), "|---|---|---|---:|---:|---:|---:|"]
            for row in dose["modules"]:
                lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                    row.get("drug"), row.get("stratum"), row.get("name"), row.get("n"),
                    row.get("rho"), row.get("p"), row.get("q")))
            lines.append("")
    conc = data.get("concordance") or {}
    excl = data.get("exclusion") or {}
    lines += [t("assembly.heading_concordance_and_exclusion"), "",
              t("assembly.concordance_line") % (conc.get("status"), conc.get("pairs")),
              t("assembly.exclusion_trace_line")
              % (excl.get("status"), "、".join(excl.get("excluded") or [])
                 or t("assembly.exclusion_none")),
              ""]
    figures = data.get("figures") or {}
    # R18: the caption list is a merged, numbered view, so state how many caption entries and how many
    # PNG files exist instead of reporting one number for both; state the provenance of the numbers and
    # disclose the listing cap so the model does not read a truncated list as the whole figure set.
    png_count = 0
    try:
        png_count = len(list((Path(data.get("run") or ".") / "visualize_results").glob("*.png")))
    except Exception:  # noqa: BLE001
        png_count = 0
    if figures.get("count"):
        listed = list(figures.get("titles") or [])
        merged = t("assembly.figure_caption_merged") if png_count > figures["count"] else ""
        capped = ""
        if figures["count"] > len(listed):
            capped = t("assembly.figure_caption_capped_suffix") % (len(listed),
                                                          figures["count"] - len(listed))
        lines += [t("assembly.heading_figure_index"), "",
                  t("assembly.figure_caption_count")
                  % (figures["count"], merged, png_count, capped, "；".join(listed)),
                  t("assembly.figure_number_provenance"),
                  ""]
    else:
        lines += [t("assembly.heading_figures_plain"), "", t("assembly.not_computed_figures_line"), ""]
    if data.get("not_computed"):
        lines += [t("assembly.heading_not_computed"), ""]
        lines += ["- %s" % item for item in data["not_computed"]]
        lines.append("")
    if data.get("task_ids"):
        # R21/F4: the request hands over exactly one task id per scientific task, together with the
        # direction that id stands for; aliases stay valid input, but they are not extra tasks.
        lines += [t("assembly.heading_task_ids"), ""]
        for rec in data.get("contrasts") or []:
            if not rec.get("task_id"):
                continue
            lines.append(t("assembly.task_id_line")
                         % (rec.get("task_id"), rec.get("display_contrast"),
                            rec.get("positive_means") or t("assembly.arm_first"), rec.get("source_contrast") or t("assembly.not_recorded"),
                            t("assembly.sign_opposite") if rec.get("inverted") else ""))
        aliases = {task: names for task, names in (data.get("task_aliases") or {}).items() if names}
        if aliases:
            lines += ["", t("assembly.task_alias_note")
                      % "；".join("%s ≡ %s" % (task, "、".join(names))
                                  for task, names in aliases.items())]
        lines.append("")
    return chr(10).join(lines).rstrip() + chr(10)


def request_evidence_block(run: Path) -> str:
    """The block the production request appends so the model reads the same evidence as the report."""
    return format_request_evidence_index(build_request_evidence_index(Path(run)))


def request_evidence_index_sha(run: Path) -> str:
    data = build_request_evidence_index(Path(run))
    return sha12(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str))

# --------------------------------------------------------------------------- free-text conversion
STRUCTURED_SCHEMA = ("key_summary: [str]; tasks: [{task_id, question, findings, explanation, "
                     "evidence_ids}]; synthesis: {type: 'synthesis', text: [str]}; boundary: [str]")


def parse_structured(text: str) -> Dict[str, Any]:
    """Parse a structured model answer; every validation problem is reported, never dropped.

    Rule: a result task section must carry a valid task id; a synthesis section must be typed as
    synthesis and needs no contrast binding; an unknown or duplicated id is an error, not a match.
    """
    payload = None
    fenced = re.search(r"[" + chr(96) * 3 + r"]+(?:json)?\s*(\{.*?\})\s*[" + chr(96) * 3 + r"]+", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    brace = re.search(r"\{.*\}", text, re.S)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except Exception:
            payload = None
    sections: Dict[str, Any] = {"key_summary": [], "tasks": [], "summary": [], "boundary": [],
                                "extra": [], "source_text": text, "summary_type": "synthesis"}
    report = {"mode": "structured", "errors": [], "task_ids": [], "unknown_ids": [],
              "duplicate_ids": [], "missing_ids": []}
    if not isinstance(payload, dict):
        report["mode"] = "unstructured"
        report["errors"].append("no JSON payload found; the whole text was kept as an unbound block")
        sections["extra"].append(text)
        sections["binding_report"] = report
        return {"sections": sections, "binding_report": report}
    sections["key_summary"] = _as_list(payload.get("key_summary"))
    sections["boundary"] = _as_list(payload.get("boundary"))
    synthesis = payload.get("synthesis") or {}
    if isinstance(synthesis, dict):
        declared = str(synthesis.get("type") or "").strip()
        if declared != "synthesis":
            report["errors"].append("synthesis.type must be 'synthesis' (got %r)" % declared)
        sections["summary_type"] = "synthesis" if declared == "synthesis" else "unverified_synthesis"
        sections["summary"] = _as_list(synthesis.get("text"))
    elif synthesis:
        report["errors"].append("synthesis must be an object with type and text; kept as text")
        sections["summary_type"] = "unverified_synthesis"
        sections["summary"] = _as_list(synthesis)
    seen: Dict[str, int] = {}
    for task in payload.get("tasks") or []:
        if not isinstance(task, dict):
            report["errors"].append("task entry is not an object; kept as unbound text")
            sections["tasks"].append({"match": "", "question": [], "explanation": [],
                                      "findings": [json.dumps(task, ensure_ascii=False)[:2000]]})
            continue
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            report["errors"].append("task without a task_id (kept unbound with every field)")
            sections["tasks"].append({"match": "", "question": _as_list(task.get("question")),
                                      "findings": _as_list(task.get("findings")),
                                      "explanation": _as_list(task.get("explanation")),
                                      "evidence_ids": _as_list(task.get("evidence_ids"))})
            continue
        seen[task_id] = seen.get(task_id, 0) + 1
        if seen[task_id] > 1:
            report["duplicate_ids"].append(task_id)
        evidence_ids = _as_list(task.get("evidence_ids"))
        sections["tasks"].append({
            "match": task_id,
            "question": _as_list(task.get("question")),
            "findings": _as_list(task.get("findings")),
            "explanation": _as_list(task.get("explanation")),
            "evidence_ids": evidence_ids,
            "cross_contrast": bool(task.get("cross_contrast"))})
        report["task_ids"].append(task_id)
    sections["binding_report"] = report
    return {"sections": sections, "binding_report": report}


def check_bindings(sections: Dict[str, Any], index: Dict[str, Any]) -> Dict[str, Any]:
    """Unknown ids, missing result tasks and citations of non-existent evidence ids."""
    report = dict(sections.get("binding_report") or {"mode": "free_text", "errors": []})
    valid_ids = {str(c.get("claim_id") or "") for c in index["contrasts"]} | \
                {str(c.get("display_contrast") or "") for c in index["contrasts"]} | \
                {str(c.get("source_contrast") or "") for c in index["contrasts"]}
    valid_ids = {item for item in valid_ids if item}
    known_evidence = {v["id"] for v in index["candidates"].values()} | \
                     {v["id"] for v in index["modules"].values()} | {c["id"] for c in index["contrasts"]}
    unknown = [t for t in (sections.get("tasks") or [])
               if str(t.get("match") or "").strip() and str(t.get("match")).strip() not in valid_ids]
    valid_ids = valid_ids | {str(contrast.get("id") or "") for contrast in index["contrasts"]}
    unknown = [task for task in unknown if str(task.get("match")).strip() not in valid_ids]
    report["unknown_ids"] = sorted({str(t.get("match")) for t in unknown})
    canon = {}
    for contrast in index["contrasts"]:
        for alias in (contrast.get("claim_id"), contrast.get("display_contrast"),
                      contrast.get("source_contrast"), contrast.get("id")):
            if alias:
                canon[str(alias)] = contrast["id"]
    used = {str(t.get("match")).strip() for t in (sections.get("tasks") or [])}
    used_canon = [canon[item] for item in used if item in canon]
    canonical_duplicates = sorted({cid for cid in used_canon if used_canon.count(cid) > 1})
    report["duplicate_ids"] = sorted(set(report.get("duplicate_ids") or []) | set(canonical_duplicates))
    report["missing_ids"] = sorted(contrast["id"] for contrast in index["contrasts"]
                                   if contrast["id"] not in set(used_canon))
    evidence_contrast = {value["id"]: canon.get(str(value.get("display_contrast")), "")
                         for value in index["candidates"].values()}
    evidence_contrast.update({value["id"]: canon.get(str(value.get("display_contrast")), "")
                              for value in index["modules"].values()})
    evidence_contrast.update({contrast["id"]: contrast["id"] for contrast in index["contrasts"]})
    out_of_scope = []
    for task in sections.get("tasks") or []:
        task_cid = canon.get(str(task.get("match")).strip())
        if not task_cid:
            continue
        for eid in task.get("evidence_ids") or []:
            owner = evidence_contrast.get(eid)
            if owner and owner != task_cid and not task.get("cross_contrast"):
                out_of_scope.append({"task": task_cid, "evidence_id": eid, "belongs_to": owner})
    report["evidence_out_of_scope"] = out_of_scope
    cited = [eid for task in (sections.get("tasks") or []) for eid in (task.get("evidence_ids") or [])]
    report["unknown_evidence_ids"] = sorted({eid for eid in cited if eid not in known_evidence})
    return report


def sections_from_text(text: str) -> Dict[str, Any]:
    """Convert a free-form model report into the section schema; the raw text travels with it."""
    blocks: List[Dict[str, Any]] = []
    current = {"heading": "", "lines": []}
    for line in text.splitlines():
        if line.startswith("## "):
            if current["lines"] or current["heading"]:
                blocks.append(current)
            current = {"heading": line[3:].strip(), "lines": []}
        else:
            current["lines"].append(line)
    blocks.append(current)

    sections: Dict[str, Any] = {"key_summary": [], "tasks": [], "summary": [], "boundary": [],
                                "extra": [], "source_text": text}
    for block in blocks:
        heading = block["heading"]
        # Case-folded copy used only for keyword matching, so one rule recognises both the Chinese
        # and the English heading; lower() leaves a Chinese heading unchanged.
        heading_key = heading.lower()
        body = "\n".join(block["lines"]).strip()
        if not body and not heading:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if not heading:
            sections["extra"].extend(paragraphs)
        elif (t("assembly.keyword_overview") in heading_key) or (t("assembly.keyword_conclusion") in heading_key and t("assembly.keyword_boundary") not in heading_key):
            sections["key_summary"].extend(paragraphs)
        elif t("assembly.keyword_boundary") in heading_key or t("assembly.keyword_limitation") in heading_key:
            sections["boundary"].extend(paragraphs)
        elif t("assembly.keyword_summary") in heading_key or t("assembly.keyword_inference") in heading_key or t("assembly.keyword_discussion") in heading_key:
            sections["summary"].extend(paragraphs)
        elif t("assembly.keyword_asset") in heading_key or t("assembly.keyword_appendix") in heading_key or t("assembly.keyword_index") in heading_key:
            sections["extra"].extend(paragraphs)
        else:
            sections["tasks"].append({"match": heading, "question": [], "findings": paragraphs,
                                      "explanation": []})
    return sections


def verify_preservation(report: str, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Model paragraphs that did not survive verbatim."""
    missing = []
    for item in manifest.get("model_paragraphs") or []:
        if not hash_text_present(report, item.get("sha12")):
            missing.append({"sha12": item.get("sha12"), "owner": item.get("owner"),
                            "chars": item.get("chars")})
    return missing


def hash_text_present(report: str, target_sha: str) -> bool:
    if not target_sha:
        return False
    for chunk in re.split(r"\n\s*\n", report):
        if sha12(chunk.strip()) == target_sha:
            return True
    return False

# --------------------------------------------------------------------------- content ledger v2
NUM_TOKEN = re.compile(r"(?<![A-Za-z0-9.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![0-9])")
SUBJECT_TOKEN = re.compile(r"\b(?:[A-Z][0-9][A-Z0-9]{4,9}|[OPQ][0-9][A-Z0-9]{3}[0-9])\b")
DISPLAY_KEYS_A = ("显示", "display")


def _direction_label(record: Dict[str, Any], groups) -> str:
    """Internal direction tokens must never reach the report."""
    token = str(record.get("direction_display") or "").strip().lower()
    first, second = (list(groups or []) + [t("assembly.group_a"), t("assembly.group_b")])[:2]
    if token.startswith("up") or "group_a_higher" in token or token in {"higher_in_group_a"}:
        return t("assembly.arm_higher") % first
    if token.startswith("down") or "group_b_higher" in token or token in {"lower_in_group_a"}:
        return t("assembly.arm_higher") % second
    display = record.get("logFC_display")
    if isinstance(display, (int, float)):
        return t("assembly.arm_higher") % (first if display >= 0 else second)
    return t("assembly.direction_undetermined")
DISPLAY_KEYS = ("显示", "display")
SOURCE_KEYS = ("源表", "源差异表", "source")
ADJP_KEYS = ("adj.p", "adj.p.val", "fdr", "q值")
EFFECT_KEYS = ("logfc", "log2fc", t("assembly.field_effect_size"), "δ", "delta")


def _contrast_id_of(value, index):
    for contrast in index["contrasts"]:
        if value.get("display_contrast") == contrast.get("display_contrast"):
            return contrast["id"]
    return ""


def _expected_by_header(header, record):
    """The record field a column header asks for; None when the header names no known statistic."""
    head = str(header or "").lower()
    if any(key in head for key in ADJP_KEYS):
        return record.get("adj.P.Val")
    if any(key in head for key in DISPLAY_KEYS):
        return record.get("logFC_display")
    if any(key in head for key in SOURCE_KEYS):
        return record.get("logFC_source")
    if any(key in head for key in EFFECT_KEYS):
        candidates = [record.get("logFC_display"), record.get("logFC_source")]
        return candidates[0] if candidates[0] is not None else candidates[1]
    return None


def _model_table_carried(block_text, index):
    """Conservative rule: only an explicitly linked, field-bearing table may count as carried over."""
    rows = _table_rows(block_text)
    if len(rows) < 2:
        return False, "table has no data row", []
    linked = [contrast["id"] for contrast in index["contrasts"]
              if contrast["id"] in block_text or str(contrast.get("display_contrast")) in block_text
              or str(contrast.get("source_contrast") or "") in block_text]
    if len(set(linked)) != 1:
        return False, "the table does not name exactly one canonical contrast", []
    contrast_id = linked[0]
    records = [value for value in index["candidates"].values()
               if _contrast_id_of(value, index) == contrast_id]
    records += [value for value in index["modules"].values()
                if _contrast_id_of(value, index) == contrast_id]
    if not records:
        return False, "no deterministic record for that contrast", []
    header = rows[0]
    if not any(any(key in str(cell).lower() for key in DISPLAY_KEYS + SOURCE_KEYS + ADJP_KEYS + EFFECT_KEYS)
               for cell in header):
        return False, "the header names no statistic or orientation field", []
    unmatched = []
    for cells in rows[1:]:
        record = None
        for cell in cells:
            for candidate in records:
                for token in filter(None, (candidate.get("candidate"), candidate.get("matched_gene"),
                                           candidate.get("protein_group"), candidate.get("module"))):
                    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(str(token)) + r"(?![A-Za-z0-9_])", cell):
                        record = candidate
                        break
                if record:
                    break
            if record:
                break
        if record is None:
            unmatched.append({"row": " | ".join(cells)[:120], "reason": "no object identity in the evidence index"})
            continue
        for col, cell in enumerate(cells):
            for token in NUM_TOKEN.findall(cell):
                expected = _expected_by_header(header[col] if col < len(header) else "", record)
                if expected is None:
                    unmatched.append({"row": " | ".join(cells)[:120],
                                      "reason": "column %d has no orientation/statistic header" % col})
                    continue
                if token != _fmt(expected) and token != str(expected):
                    unmatched.append({"row": " | ".join(cells)[:120],
                                      "reason": "value %s does not match %s for this field" % (token, _fmt(expected))})
    if unmatched:
        return False, "%d cell(s) could not be matched field by field" % len(unmatched), unmatched
    return True, "every row matched a deterministic record by contrast, object and field", []


def _content_ledger_v3(blocks, report, model_paragraphs, index):
    """Content ledger: only explicit, field-bearing tables are auto-replaced; the rest is kept."""
    kept = {item["sha12"] for item in model_paragraphs}
    ledger = []
    for block in blocks:
        if block["kind"] == "table":
            carried, reason, unmatched = _model_table_carried(block["text"], index)
            ledger.append({"index": block["index"], "heading": block["heading"], "kind": block["kind"],
                           "sha12": block["sha12"], "chars": block["chars"],
                           "disposition": "replaced_by_deterministic_table" if carried else "needs_manual_review",
                           "reason": reason, "unmatched_rows": unmatched})
            continue
        block_text = str(block.get("text") or "")
        # R29: a structured model answer is registered as a single block, so its whole-answer hash can
        # never equal a per-paragraph hash and every run reported "not in the finished report" for it.
        # The registry (model_paragraphs) is the thing that actually answers "did the model content
        # survive", so a structured answer block points at that registry instead of claiming a loss.
        _structured = False
        try:
            _payload = json.loads(block_text)
            _structured = isinstance(_payload, dict) and any(
                key in _payload for key in ('key_summary', 'tasks', 'boundary', 'synthesis'))
        except Exception:
            _structured = False
        _registered = len(model_paragraphs)
        _reg_reason = ("the structured answer was registered as %d model paragraphs; "
                       "verify the delivered text against model_paragraphs") % _registered
        if block["sha12"] in kept:
            _disposition, _reason = "preserved", "kept verbatim"
        elif _structured and _registered:
            _disposition = "consumed_by_structured_answer"
            _reason = _reg_reason
        else:
            _disposition, _reason = "needs_manual_review", "not in the finished report"
        ledger.append({"index": block["index"], "heading": block["heading"], "kind": block["kind"],
                       "sha12": block["sha12"], "chars": block["chars"],
                       "disposition": _disposition,
                       "reason": (_reason if isinstance(_reason, str) else "".join(_reason))})
    return ledger


def _content_ledger_v2(blocks, report, model_paragraphs, index):
    """(superseded by _content_ledger_v3) row-level subject-and-value matching."""


def _as_list(value):
    """Never iterate a string character by character."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _table_rows(text: str):
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= set("|-: "):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def _row_carried(cells, report_lines, index):
    """A row counts as carried only when one deterministic row holds the same subject and the same
    signed values (scientific notation and signs preserved)."""
    subjects = set()
    for cell in cells:
        if SUBJECT_TOKEN.fullmatch(cell.strip()):
            subjects.add(cell.strip())
        for key in index["candidates"]:
            identity = str(key[1])
            if identity and identity in cell:
                subjects.add(identity)
        for value in index["candidates"].values():
            gene = str(value.get("candidate") or "")
            if gene and re.search(r"\b" + re.escape(gene) + r"\b", cell):
                subjects.add(gene)
        for key in index["modules"]:
            if str(key[1]) and str(key[1]) in cell:
                subjects.add(str(key[1]))
    numbers = [token for cell in cells for token in NUM_TOKEN.findall(cell)]
    if not subjects:
        return False, "no subject in this row matches a protein group, gene or module in the evidence index"
    if not numbers:
        return False, "no signed numeric value to check (free-text table stays unverified)"
    for line in report_lines:
        if not all(subject in line for subject in subjects):
            continue
        # token equality, never substring containment: a flipped sign must not match
        line_tokens = set(NUM_TOKEN.findall(line))
        if all(number in line_tokens for number in numbers):
            return True, ""
    return False, "no deterministic row carries this subject with the identical signed values"


def _content_ledger_v2(blocks, report, model_paragraphs, index):
    """Narrowed rule: only a row-level, subject-and-value match counts as carried over."""
    kept = {item["sha12"] for item in model_paragraphs}
    report_lines = report.splitlines()
    ledger = []
    for block in blocks:
        if block["kind"] == "table":
            rows = _table_rows(block["text"])
            if not rows:
                disposition = "needs_manual_review"
                reason = "table could not be parsed into data rows; kept for manual review"
                unmatched = []
            else:
                unmatched = []
                # the first markdown row is the column header, not data
                data_rows = rows[1:] if len(rows) > 1 else rows
                for cells in data_rows:
                    ok, why = _row_carried(cells, report_lines, index)
                    if not ok:
                        unmatched.append({"row": " | ".join(cells)[:160], "reason": why})
                if unmatched:
                    disposition = "needs_manual_review"
                    reason = "%d row(s) were not carried over by the deterministic tables" % len(unmatched)
                else:
                    disposition = "replaced_by_deterministic_table"
                    reason = "every row matched a deterministic row by subject and signed values"
            ledger.append({"index": block["index"], "heading": block["heading"], "kind": block["kind"],
                           "sha12": block["sha12"], "chars": block["chars"],
                           "disposition": disposition, "reason": reason,
                           "unmatched_rows": unmatched})
            continue
        if block["sha12"] in kept:
            disposition, reason = "preserved", "kept verbatim"
        else:
            disposition, reason = "needs_manual_review", "not present in the finished report; kept for review"
        ledger.append({"index": block["index"], "heading": block["heading"], "kind": block["kind"],
                       "sha12": block["sha12"], "chars": block["chars"],
                       "disposition": disposition, "reason": reason})
    return ledger
