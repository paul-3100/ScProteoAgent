# -*- coding: utf-8 -*-
"""Generic analysis extensions (2026-09-11).

Four reusable analyses that answer scoring questions the report previously could not:

1. trace_candidate_exclusion  - where and why a candidate fell out of the pipeline
2. estimate_confounding       - can treatment and batch be estimated separately
3. stratified_contrasts       - within-stratum effects when both arms have support
4. contrast_concordance       - overlap / sign agreement between two contrasts

Pure computation on run artifacts. No network, no rubric, no ground truth.
Usage:  python analysis_extensions.py --run-folder <run dir> [--min-n 5]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import report_language  # same directory as this file; the shared zh/en registry

# --------------------------------------------------------------------------- language
# Reader-facing text of the extension emitters (candidate-exclusion trace, treatment/batch
# confounding verdicts, stratified contrasts, contrast concordance, unit-layer diagnostics).
# The zh value of every entry is the released literal copied byte for byte, so a zh run writes
# exactly the released text; the en value is the same statement in English. Placeholders stay
# in %-form so the call site is one expression for both languages.
#
# Strings that were already English in the released module (the unit-level filter rule,
# correction scope and row note) keep that English text as their zh value: the released text is
# the contract for a zh run, and the entry simply makes the en path explicit and reachable.
EXTENSION_TEXT = {
    # --- shared
    "list_separator": ("、", ", "),
    "unknown": ("未知", "unknown"),
    # --- trace_candidate_exclusion
    "trace_no_protein_in_stage": (
        "该阶段矩阵中无此蛋白组",
        "this protein group is absent from the matrix at this stage",
    ),
    "trace_not_detected_in_stage": (
        "该阶段全样本未检出",
        "not detected in any sample at this stage",
    ),
    "trace_not_in_diff_table": (
        "未进入该对比的差异表",
        "did not enter this contrast's differential table",
    ),
    "trace_verdict_matched_diff": (
        "已匹配：进入分析矩阵与差异表",
        "matched: present in the analysis matrix and in the differential table",
    ),
    "trace_verdict_matched_matrix": (
        "已匹配：进入分析矩阵",
        "matched: present in the analysis matrix",
    ),
    "trace_verdict_absent_matrix": (
        "未匹配：符号可识别，但各阶段矩阵中不存在该蛋白组",
        "not matched: the symbol is resolvable, but this protein group is absent from the "
        "matrices at every stage",
    ),
    "trace_verdict_name_unresolved": (
        "名称待解析：未在任何矩阵、候选表或差异表中识别该符号",
        "name unresolved: the symbol was not recognised in any matrix, candidate table or "
        "differential table",
    ),
    "trace_verdict_not_detected_input": (
        "未检出：原始矩阵中全样本未检出",
        "not detected: absent from every sample of the input matrix",
    ),
    "trace_stage_filtered_matrix": ("过滤后矩阵", "the filtered matrix"),
    "trace_stage_preprocessing": ("预处理", "preprocessing"),
    "trace_verdict_filtered_out": (
        "被过滤移除：输入矩阵检出 %s/%s（%s），过滤后矩阵已不含该蛋白组（该步保留 %s 个蛋白）",
        "removed by filtering: detected in %s/%s of the input matrix (%s), and the filtered "
        "matrix no longer contains this protein group (that step kept %s proteins)",
    ),
    "trace_verdict_not_computed": (
        "未计算：进入分析矩阵但未进入任何已计算对比的差异表",
        "not computed: present in the analysis matrix but in no computed contrast's "
        "differential table",
    ),
    # --- estimate_confounding
    "conf_no_matrix_yet": (
        "本轮尚未生成分析矩阵：分析阶段尚未完成（processed_proteins 目录尚无产物），"
        "因此未做处理与批次的可比性判断；该判词不是数据缺失的结论。",
        "No analysis matrix has been produced for this run yet: the analysis stage has no "
        "output in the processed_proteins directory, so treatment-batch comparability was not "
        "assessed; this verdict is not a finding that data are missing.",
    ),
    "conf_no_matrix_or_sampleinfo": (
        "找不到可用的分析矩阵或样本信息，无法评估处理与批次的可比性。",
        "No usable analysis matrix or sample metadata could be resolved, so treatment-batch "
        "comparability cannot be assessed.",
    ),
    "conf_no_group_column": (
        "该数据集没有可用的分组列，处理与批次的可比性不适用。",
        "This dataset has no usable group column, so treatment-batch comparability does not "
        "apply.",
    ),
    "conf_no_batch_column": (
        "元数据中没有声明批次列，本轮不做处理与批次的可比性判断。",
        "No batch column is declared in the metadata, so treatment-batch comparability was not "
        "assessed in this run.",
    ),
    "conf_too_few_samples": ("可用样本过少，无法评估可比性。", "Too few usable samples to assess comparability."),
    "conf_too_few_proteins": (
        "分析矩阵中完整蛋白过少，无法评估可比性。",
        "Too few complete proteins in the analysis matrix to assess comparability.",
    ),
    "conf_note_same_complete_set": (
        "效应估计与置换使用同一完整集合",
        "the effect estimate and the permutation use the same complete set",
    ),
    "conf_note_projection_residual": (
        "对比在 row space(X) 上的投影残差 %.3g",
        "residual of the contrast projected onto the row space of X: %.3g",
    ),
    "conf_note_variance_inflation": (
        "对比估计相对无批次设计的方差膨胀",
        "variance inflation of the contrast estimate relative to the design without batch",
    ),
    "conf_note_weak_identification": (
        "VIF > 50 时判为弱识别",
        "a VIF above 50 is read as weak identification",
    ),
    "conf_note_empty_cell": ("0 表示存在空交叉格", "0 means an empty treatment-by-batch cell exists"),
    "conf_note_adjusted_median": ("批次校正后 %s", "%s after batch adjustment"),
    "conf_note_permutation": (
        "%d 次批次内置换；观测与置换使用同一固定集合（按蛋白方差预选 %d/%d 个，与处理标签无关）",
        "%d within-batch permutations; the observed statistic and the permutations use the same "
        "fixed set (%d/%d preselected on protein variance, independent of the treatment labels)",
    ),
    "conf_note_eta2_treatment": (
        "仅描述性：主成分上与处理的单因素关联",
        "descriptive only: single-factor association with treatment on the principal components",
    ),
    "conf_note_eta2_batch": (
        "仅描述性：主成分上与批次的单因素关联",
        "descriptive only: single-factor association with batch on the principal components",
    ),
    "conf_note_empty_cell_estimable": (
        "；存在空交叉格（最小格 0 个观测）：加性模型下该对比仍可估计，但不能估计该格的处理×批次交互",
        "; an empty treatment-by-batch cell exists (smallest cell 0 observations): the contrast "
        "is still estimable under the additive model, but the treatment-by-batch interaction in "
        "that cell cannot be estimated",
    ),
    "conf_nearly_collinear": ("处理与批次接近共线", "treatment and batch are nearly collinear"),
    "conf_verdict_not_estimable": (
        "目标对比 %s 在联合设计中不可估计（对比向量几乎完全落在批次空间内，残差范数 %.4g）："
        "不报告批次校正后的效应，也不做处理与批次效应的强弱比较；结果按处理+批次合并效应报告。",
        "The target contrast %s is not estimable in the joint design (the contrast vector lies "
        "almost entirely inside the batch space, residual norm %.4g): no batch-adjusted effect "
        "is reported and treatment and batch effects are not compared in strength; results are "
        "reported as a combined treatment-plus-batch effect.",
    ),
    "conf_verdict_weakly_identified": (
        "目标对比 %s 可估计但只是弱识别（相对无批次设计的方差膨胀 VIF=%.1f；%s）："
        "批次校正后的效应不稳定，只能作为参考，不用于与批次效应比较强弱；"
        "处理×批次覆盖情况必须与结果一同引用。",
        "The target contrast %s is estimable but only weakly identified (variance inflation "
        "VIF=%.1f relative to the design without batch; %s): the batch-adjusted effect is "
        "unstable and is given as a reference only, not for comparing strength against the "
        "batch effect; the treatment-by-batch coverage must be cited with the result.",
    ),
    "conf_verdict_estimable": (
        "目标对比 %s 在联合设计中可估计（VIF=%.2f，处理×批次最小格 %d 个观测%s）："
        "批次校正后对比效应中位数 %.3f、四分位距 %.3f，与未校正均值差的 Spearman 相关 %.3f、"
        "符号一致 %d/%d；批次内置换检验 p=%.4f，观测与置换使用同一固定集合（按蛋白方差预选 %d/%d 个，"
        "与处理标签无关）。主成分上的处理/批次单因素关联（%.3f / %.3f）只作描述性指标。",
        "The target contrast %s is estimable in the joint design (VIF=%.2f, smallest "
        "treatment-by-batch cell %d observations%s): after batch adjustment the contrast effect "
        "has median %.3f and interquartile range %.3f, with Spearman correlation %.3f and sign "
        "agreement %d/%d against the unadjusted mean difference; the within-batch permutation "
        "test gives p=%.4f, and the observed statistic and the permutations use the same fixed "
        "set (%d/%d preselected on protein variance, independent of the treatment labels). The "
        "single-factor treatment/batch associations on the principal components (%.3f / %.3f) "
        "are descriptive only.",
    ),
    # --- _effect_scale / effect_scale_record
    "scale_log2_from_record": (
        "log2FC（依据本次运行写出的矩阵变换执行记录）",
        "log2FC (from the matrix-transform record written by this run)",
    ),
    "scale_log2_from_design": (
        "log2FC（分析设计明确记录已执行对数变换）",
        "log2FC (the analysis design records that a log transform was applied)",
    ),
    "scale_linear_recorded": (
        "均值差（设计记录未做对数变换，数值范围与线性强度一致；单位为矩阵强度）",
        "mean difference (the design records no log transform and the value range matches "
        "linear intensities; units are matrix intensity)",
    ),
    "scale_unknown_heuristic": (
        "均值差（仅有尺度启发式提示，未记录对数变换；尺度未知，不能称为 log2FC）",
        "mean difference (only a scale heuristic is available and no log transform is "
        "recorded; the scale is unknown and this is not a log2FC)",
    ),
    "scale_unknown": (
        "均值差（未记录对数变换，尺度未知；不能称为 log2FC）",
        "mean difference (no log transform is recorded and the scale is unknown; this is not "
        "a log2FC)",
    ),
    "scale_source_run_record": (
        "processed_proteins/matrix_transform_record.json（本次运行写出的变换执行记录）",
        "processed_proteins/matrix_transform_record.json (the transform record written by "
        "this run)",
    ),
    # --- stratified_contrasts
    "strat_threshold_joint": (
        "|logFC| > %.2f 且 adj.P < %.2f",
        "|logFC| > %.2f and adj.P < %.2f",
    ),
    "strat_threshold_fdr_only": (
        "仅 adj.P < %.2f（尺度未确认为 log2，未套用 logFC 阈值）",
        "adj.P < %.2f only (the scale is not confirmed as log2, so no logFC threshold is applied)",
    ),
    "strat_verdict_below_min_n": (
        "未分层（低于最低计算门槛）",
        "not stratified (below the minimum computation threshold)",
    ),
    "strat_note_min_n": (
        "两臂各需 ≥%d，实际 %d vs %d；该门槛只是计算下限，不代表设计上可分层",
        "each arm needs at least %d, actual %d vs %d; this threshold is only a computation "
        "floor and does not mean the design supports stratification",
    ),
    "strat_verdict_stratified": ("已分层", "stratified"),
    "strat_note_alignment": (
        "按完整蛋白组标识与合并结果对齐；同向 %d/%d、异向 %d；逐蛋白结果见 %s.csv",
        "aligned with the pooled result by the full protein-group identifier; same sign %d/%d, "
        "opposite sign %d; per-protein results in %s.csv",
    ),
    "strat_verdict_convention_note": ("口径说明", "convention note"),
    "strat_note_separate_not_joint": (
        "分别按各变量分层不等于同时控制这些变量；需要联合控制时应把两个因素放进同一模型。",
        "Stratifying by each variable separately is not the same as controlling both variables "
        "at once; joint control requires both factors in one model.",
    ),
    # --- contrast_concordance
    "conc_not_applicable": ("不适用", "not applicable"),
    "conc_insufficient_shared": ("共享蛋白不足，无法比较", "too few shared proteins to compare"),
    "conc_comparable": (
        "可比较（两对比均有显著蛋白）",
        "comparable (both contrasts have significant proteins)",
    ),
    "conc_no_shared_significant": ("无共同显著蛋白", "no shared significant proteins"),
    "conc_zero_overlap": (
        "显著集合重叠为零（%s 显著 %d 个，%s 显著 %d 个）；方向一致率不适用。"
        "整体足迹仍可比：见效应量分布与共享蛋白相关性。",
        "the significant sets do not overlap at all (%s: %d significant, %s: %d significant), "
        "so the sign-agreement rate is not applicable. The overall footprints remain "
        "comparable: see the effect-size distributions and the shared-protein correlation.",
    ),
    "conc_note_fdr_scope": (
        "显著集合用 adj.P（FDR）判定；原始 P 单列保留。Spearman 基于全部共享蛋白，含近零效应。",
        "the significant sets are defined by adj.P (FDR) and raw P is kept as its own column. "
        "Spearman is computed over all shared proteins, including near-zero effects.",
    ),
    # --- unit_structure_diagnostics / unit_aware_sensitivity
    "unit_exclusion_rule": (
        "实验单位的独立生物学个体判定规则：标签中出现 mixed/mix/pool/combined/"
        "composite 的样本按混合来源处理，标签为 unknown/unassigned/none/null/NA 的按"
        "未确认来源处理；两者都不作为独立生物学个体进入个体层检验，也不参与个体间比较。",
        "Rule for treating an experimental-unit label as an independent biological individual: "
        "samples whose label contains mixed/mix/pool/combined/composite are treated as pooled "
        "sources, and labels unknown/unassigned/none/null/NA as unconfirmed sources; neither "
        "enters the unit-level test as an independent individual and neither takes part in "
        "between-unit comparisons.",
    ),
    "unit_reason_group_col_missing": (
        "分组列不在样本元数据中（group column is missing from the sample metadata）",
        "the group column is missing from the sample metadata",
    ),
    "unit_reason_no_unit_column": (
        "分析设计未声明实验单位列，元数据中也未推断出可用列"
        "（no experimental-unit column is declared and none could be inferred）",
        "no experimental-unit column is declared in the analysis design and none could be "
        "inferred from the metadata",
    ),
    "unit_reason_unit_col_empty": (
        "实验单位列在参与分析的样本上全部为空（the unit column is empty for every analysed sample）",
        "the unit column is empty for every analysed sample",
    ),
    "unit_reason_no_unit_left": (
        "排除混合来源与未确认来源后没有剩余实验单位"
        "（no experimental unit remains after excluding pooled and unconfirmed sources）",
        "no experimental unit remains after excluding pooled and unconfirmed sources",
    ),
    "unit_structure_nested": (
        "每个个体只出现在一个臂上：处理与个体身份在联合模型中不可分离。"
        "个体间比较仍然可用，但它比较的是不同个体，不能排除个体来源差异。",
        "each individual appears in one arm only: treatment and individual identity are not "
        "separable in a joint model. A between-unit comparison is still available, but it "
        "compares different individuals and cannot rule out differences between them.",
    ),
    "unit_structure_partially_crossed": (
        "部分个体同时出现在两臂：个体内（配对）估计目标可识别；"
        "只在一个臂出现的个体为个体间估计目标提供独立观测。",
        "some individuals appear in both arms: the within-unit (paired) estimand is "
        "identifiable, and individuals appearing in one arm only provide independent "
        "observations for the between-unit estimand.",
    ),
    "unit_structure_crossed": (
        "每个个体在两臂都有观测：两种估计目标都基于同一批个体。",
        "every individual is observed in both arms: both estimands rest on the same set of "
        "individuals.",
    ),
    "unit_note_rank": (
        "该秩与残余自由度只描述「处理与个体身份能否放进同一个固定效应模型」，"
        "是诊断，不用来一概否决个体间比较。",
        "this rank and the residual degrees of freedom only describe whether treatment and "
        "individual identity fit in one fixed-effects model; it is a diagnostic and does not "
        "by itself rule out a between-unit comparison.",
    ),
    "unit_estimand_within": (
        "个体内（同一实验单位在两臂都有观测）的均值差",
        "within-unit mean difference (the same experimental unit is observed in both arms)",
    ),
    "unit_reason_within_insufficient": (
        "同时出现在两臂的独立个体只有 %d 个（配对分析门槛 %d 个）：共享个体属于两臂，"
        "把它同时当成两臂的独立样本会重复计数，因此不进入个体间检验。",
        "only %d independent individuals appear in both arms (paired-analysis threshold %d): "
        "a shared individual belongs to both arms, so treating it as an independent sample of "
        "each arm would double-count it, and it is kept out of the between-unit test.",
    ),
    "unit_estimand_between": (
        "个体间（每条臂只使用只在该臂观测的独立个体）的均值差",
        "between-unit mean difference (each arm uses only the independent individuals observed "
        "in that arm)",
    ),
    "unit_reason_between_insufficient": (
        "只在一个臂上观测的独立个体不足（A 臂 %d 个、B 臂 %d 个，门槛各 %d 个）："
        "共享个体不能同时充当两臂的独立样本。",
        "too few independent individuals are observed in a single arm (arm A %d, arm B %d, "
        "threshold %d each): a shared individual cannot serve as an independent sample of both "
        "arms.",
    ),
    "unit_reason_matched_by_name": (
        "单位列是按列名匹配推断的，未在分析设计中确认"
        "（the unit column was matched by name only and is not declared in the analysis design）。",
        "the unit column was matched by name only and is not declared in the analysis design.",
    ),
    "unit_reason_auto_design": (
        "单位列来自自动生成的分析设计（按列名匹配，未由研究者确认）：“标签对应独立生物学个体”"
        "是标签层假定，单位层结果按标签层探索性分析解读"
        "（the unit column comes from the auto-inferred design; label-level independence is assumed, not confirmed）。",
        "the unit column comes from the auto-inferred design (matched by column name, not "
        "confirmed by the researcher): label-level independence is an assumption, so the "
        "unit-level result is read as an exploratory analysis at the label level.",
    ),
    "unit_reason_excluded_labels": (
        "已排除 %d 个混合来源或未确认来源标签（%s）：混合池与未知标签不能包装成独立生物学个体。",
        "%d pooled or unconfirmed-source labels were excluded (%s): pooled and unknown labels "
        "cannot be packaged as independent biological individuals.",
    ),
    "unit_reason_switched_between": (
        "本节改用个体间估计目标：只使用单臂专属个体（A 臂 %d 个、B 臂 %d 个）。",
        "this section switches to the between-unit estimand and uses single-arm individuals "
        "only (arm A %d, arm B %d).",
    ),
    "unit_reason_unbalanced_structure": (
        "结构变量在两臂分布不均：%d 个个体只在其中一个臂出现（共 %d 个个体）。",
        "the structural variable is unevenly distributed across arms: %d individuals appear "
        "in one arm only (of %d individuals in total).",
    ),
    "unit_reason_matrix_matched_units_unresolved": (
        "矩阵匹配后有已声明的单位找不到可分析的样本列：已声明个体（A %s / B %s / 共同 %s）→ "
        "实际参与（A %d / B %d / 共同 %d）。",
        "after matrix matching, some declared units have no analysable sample column: declared "
        "individuals (A %s / B %s / shared %s) versus those actually included "
        "(A %d / B %d / shared %d).",
    ),
    "unit_reference_analysis": (
        "observation-level Welch t-test (differential_%s.csv)",
        "observation-level Welch t-test (differential_%s.csv)",
    ),
    "unit_filter_rule_fdr_only": (
        "FDR scope only: adjusted P < %s (BH over the %d protein groups of this contrast that "
        "pass the minimum-sample condition)",
        "FDR scope only: adjusted P < %s (BH over the %d protein groups of this contrast that "
        "pass the minimum-sample condition)",
    ),
    "unit_filter_rule_joint": (
        "joint threshold: adjusted P < %s and |%s| > %s (thresholds inherited from the "
        "observation-level design; effect scale state: %s)",
        "joint threshold: adjusted P < %s and |%s| > %s (thresholds inherited from the "
        "observation-level design; effect scale state: %s)",
    ),
    "unit_correction_scope": (
        "BH correction scope: the %d protein groups of this contrast that pass the "
        "minimum-sample condition; not pooled across contrasts or datasets.",
        "BH correction scope: the %d protein groups of this contrast that pass the "
        "minimum-sample condition; not pooled across contrasts or datasets.",
    ),
    "unit_row_level_note": (
        "Per protein the table reports the effective number of units (or pairs), the degrees "
        "of freedom, the standard error and the 95% confidence interval; rows below the "
        "minimum sample condition are marked as not tested and carry no P value.",
        "Per protein the table reports the effective number of units (or pairs), the degrees "
        "of freedom, the standard error and the 95% confidence interval; rows below the "
        "minimum sample condition are marked as not tested and carry no P value.",
    ),
}
report_language.register("extensions", EXTENSION_TEXT)


def t(key: str, *args) -> str:
    """Reader-facing extension text for the active language, optionally %-formatted."""
    value = report_language.t("extensions." + key)
    return value % args if args else value


SEED = 20260911
GENE_COL_CANDIDATES = ["PG.Genes", "gene", "Gene", "Genes", "gene_name"]
PROTEIN_COL_CANDIDATES = ["PG.ProteinGroups", "protein", "Protein", "protein_group", "ProteinGroups"]


def _resolve(path_value: str, run_folder: Path) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(str(path_value))
    if p.exists():
        return p
    # parameters.json often stores paths relative to the agent project root
    if len(p.parts) > 1:
        candidate = Path(run_folder.parents[1]) / Path(*p.parts[1:]) if p.drive else None
        if candidate and candidate.exists():
            return candidate
        parts = list(p.parts)
        for idx in range(len(parts)):
            cand = Path(*parts[idx:])
            if cand.exists():
                return cand
    return None


def _col(df: pd.DataFrame, names) -> str:
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name in df.columns:
            return name
    return ""


def _params(run_folder: Path) -> Dict[str, Any]:
    path = run_folder / "parameters.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(";").str[0].str.strip()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _matrix_files(run_folder: Path, params: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    files = (params.get("combat_calibration") or {}).get("files", {}) if isinstance(params.get("combat_calibration"), dict) else {}
    proc = run_folder / "processed_proteins"
    out = {
        "input": _resolve(params.get("protein_quant_path", ""), run_folder),
        "filtered": _resolve(files.get("filtered_protein_quant", ""), run_folder) or (proc / "ProteinQuant_Filtered.csv" if (proc / "ProteinQuant_Filtered.csv").exists() else None),
        "analysed": None,
    }
    for name in ("ProteinQuant_ComBat.csv", "ProQuant_Normalized.csv", "ProteinQuant_Filtered.csv"):
        if (proc / name).exists():
            out["analysed"] = proc / name
            break
    return out


def _detection(df: pd.DataFrame, sample_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    numeric = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    detected = (numeric.notna() & (numeric != 0)).sum(axis=1)
    missing = (numeric.isna() | (numeric == 0)).mean(axis=1)
    return detected, missing


def _sample_cols(df: pd.DataFrame, sampleinfo: pd.DataFrame, id_col: str) -> List[str]:
    if not id_col or id_col not in sampleinfo.columns:
        return []
    wanted = set(sampleinfo[id_col].astype(str))
    return [c for c in df.columns if str(c) in wanted]



def bh_adjust(p_values) -> np.ndarray:
    """Benjamini-Hochberg with the mandatory reverse cumulative minimum.

    NaN entries stay NaN and do not count towards m. Ties are handled by the
    reverse cumulative minimum, which is exactly what the library implementations do.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = int(valid.sum())
    if m == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * m / (np.arange(m) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    out[valid] = adjusted
    return out

def _welch_table(matrix: pd.DataFrame, sample_cols_a: List[str], sample_cols_b: List[str],
                 gene_col: str, protein_col: str, logfc_thresh: float, p_thresh: float) -> pd.DataFrame:
    a = matrix[sample_cols_a].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    b = matrix[sample_cols_b].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mean_a, mean_b = np.nanmean(a, axis=1), np.nanmean(b, axis=1)
    logfc = mean_a - mean_b
    t_stat, p_val = stats.ttest_ind(a, b, axis=1, equal_var=False, nan_policy="omit")
    out = pd.DataFrame({
        "protein_group": matrix[protein_col].astype(str) if protein_col else "",
        "gene": _symbol(matrix[gene_col]) if gene_col else "",
        "logFC": logfc,
        "P.Value": p_val,
    })
    p = out["P.Value"].to_numpy(dtype=float)
    out["adj.P.Val"] = bh_adjust(p)
    out.loc[out["P.Value"].isna(), "adj.P.Val"] = np.nan
    out["passes"] = (out["adj.P.Val"] < p_thresh) & (out["logFC"].abs() > logfc_thresh)
    return out



TASK_TOKEN_STOP = {
    "CSV", "TSV", "QC", "PCA", "UMAP", "MS", "GO", "KEGG", "FDR", "PBS", "DMSO", "DNA", "RNA",
    "ATP", "PH", "ID", "AI", "LLM", "HGNC", "SP", "THP", "NF", "KB", "IL", "TNF", "IFN", "CD",
    "GSEA", "ORA", "FC", "CV", "SD", "SE", "CI", "IQR", "HSC", "MPP", "LMPP", "GMP", "MEP",
    "EB", "IPSC", "IPS", "RG", "ORG", "EN", "IPC", "LPS", "EDTA", "LC", "MS2", "HLA",
}


def task_named_tokens(run_folder: Path, limit: Optional[int] = None) -> List[str]:
    """Every gene-like symbol the task text names. Data presence must not filter this list."""
    params = _params(run_folder)
    path = _resolve(params.get("user_input_path", ""), run_folder)
    if not path or not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    import re as _re
    tokens = _re.findall(r"\b[A-Z][A-Za-z0-9]{1,9}\b", text)
    out: List[str] = []
    for token in tokens:
        if token.upper() in TASK_TOKEN_STOP or len(token) < 2:
            continue
        if token not in out:
            out.append(token)
    return out[:limit] if limit else out

def _focus_candidates(run_folder: Path, limit: int = 25) -> List[str]:
    params = _params(run_folder)
    genes: List[str] = []
    task = _resolve(params.get("user_input_path", ""), run_folder)
    if task and task.exists():
        text = task.read_text(encoding="utf-8", errors="replace")
        import re as _re
        genes += _re.findall(r"\b[A-Z][A-Za-z0-9]{1,9}\b", text)
    ev = run_folder / "evaluation_evidence" / "candidate_protein_evidence.csv"
    known = set()
    if ev.exists():
        rows = pd.read_csv(ev)
        for column in ("matched_gene", "candidate"):
            if column in rows.columns:
                known |= {str(v).strip() for v in rows[column].dropna()}
    diff_dir = run_folder / "processed_proteins"
    for path in sorted(diff_dir.glob("differential_*.csv"))[:1]:
        df = pd.read_csv(path)
        gene_col = _col(df, GENE_COL_CANDIDATES)
        if gene_col:
            known |= {g for g in _symbol(df[gene_col]).unique() if isinstance(g, str) and g.strip() and g.lower() != "nan"}
    ordered = [g for g in genes if g in known]
    preset: List[str] = []
    ev = run_folder / "evaluation_evidence" / "candidate_protein_evidence.csv"
    if ev.exists():
        rows = pd.read_csv(ev)
        for column in ("matched_gene", "candidate"):
            if column in rows.columns:
                for value in rows[column].dropna():
                    symbol = str(value).strip()
                    if symbol and symbol not in preset:
                        preset.append(symbol)
    known = {g for g in known if isinstance(g, str)}
    for symbol in preset:
        if symbol not in ordered:
            ordered.append(symbol)
    for g in sorted(known):
        if g not in ordered:
            ordered.append(g)
    return ordered[:limit]


def trace_candidate_exclusion(run_folder: Path, candidates: Optional[List[str]] = None) -> pd.DataFrame:
    params = _params(run_folder)
    files = _matrix_files(run_folder, params)
    sampleinfo = None
    si_path = _resolve(params.get("sampleinfo_path", ""), run_folder)
    if si_path and si_path.exists():
        sampleinfo = pd.read_csv(si_path)
    id_col = str((params.get("analysis_design") or {}).get("sample_id_col") or "")
    if candidates is not None:
        cands = list(candidates)
    else:
        named = task_named_tokens(run_folder)
        preset = [g for g in _focus_candidates(run_folder, limit=60) if g not in named]
        cands = named + preset
    preset_genes = set(_focus_candidates(run_folder, limit=60))
    rows: List[Dict[str, Any]] = []
    stages: Dict[str, Optional[pd.DataFrame]] = {}
    for stage, path in files.items():
        if path and path.exists():
            df = pd.read_csv(path)
            sample_cols = _sample_cols(df, sampleinfo, id_col) if sampleinfo is not None else [c for c in df.columns if c not in {_col(df, GENE_COL_CANDIDATES), _col(df, PROTEIN_COL_CANDIDATES)}]
            stages[stage] = (df, sample_cols)
    diff_tables = {}
    for path in sorted((run_folder / "processed_proteins").glob("differential_*.csv")):
        diff_tables[path.stem.replace("differential_", "")] = pd.read_csv(path)

    for gene in cands:
        positions: Dict[str, str] = {}
        for stage, payload in stages.items():
            if payload is None:
                continue
            df, sample_cols = payload
            gene_col = _col(df, GENE_COL_CANDIDATES)
            prot_col = _col(df, PROTEIN_COL_CANDIDATES)
            if not gene_col:
                continue
            hit = _symbol(df[gene_col]) == gene
            if not hit.any():
                positions[stage] = "absent"
                rows.append({"candidate": gene, "stage": stage, "protein_group": "", "detected_samples": 0,
                             "total_samples": len(sample_cols), "detected_fraction": 0.0,
                             "status": "absent", "reason": t("trace_no_protein_in_stage")})
                continue
            sub = df[hit].iloc[0]
            detected, _ = _detection(df[hit], sample_cols)
            n_det = int(detected.iloc[0])
            frac = round(n_det / max(1, len(sample_cols)), 4)
            positions[stage] = "present" if n_det > 0 else "undetected"
            rows.append({"candidate": gene, "stage": stage,
                         "protein_group": str(sub[prot_col]) if prot_col else "",
                         "detected_samples": n_det, "total_samples": len(sample_cols),
                         "detected_fraction": frac,
                         "status": "present" if n_det > 0 else "undetected",
                         "reason": "" if n_det > 0 else t("trace_not_detected_in_stage")})
        for contrast, df in diff_tables.items():
            gene_col = _col(df, GENE_COL_CANDIDATES)
            if not gene_col:
                continue
            hit = _symbol(df[gene_col]) == gene
            rows.append({"candidate": gene, "stage": "differential:" + contrast, "protein_group": "",
                         "detected_samples": int(hit.sum()), "total_samples": int(df.shape[0]),
                         "detected_fraction": round(float(hit.sum()) / max(1, df.shape[0]), 6),
                         "status": "present" if hit.any() else "absent",
                         "reason": "" if hit.any() else t("trace_not_in_diff_table")})
        trace = [r for r in rows if r["candidate"] == gene]
        input_row = next((r for r in trace if r["stage"] == "input"), None)
        analysis_row = next((r for r in trace if r["stage"] == "analysed"), None)
        diff_rows = [r for r in trace if str(r["stage"]).startswith("differential:")]
        in_diff = any(r["status"] == "present" for r in diff_rows)
        in_preset = gene in preset_genes
        verdict = (t("trace_verdict_matched_diff")
                   if (analysis_row and analysis_row["status"] == "present" and in_diff)
                   else t("trace_verdict_matched_matrix"))
        identity_known = in_preset or in_diff or any(
            r["status"] != "absent" for r in trace
            if str(r["stage"]) in {"input", "filtered", "analysed"})
        if input_row and input_row["status"] == "absent" and (analysis_row is None or analysis_row["status"] == "absent"):
            if identity_known:
                verdict = t("trace_verdict_absent_matrix")
            else:
                verdict = t("trace_verdict_name_unresolved")
        elif input_row and input_row["status"] == "undetected":
            verdict = t("trace_verdict_not_detected_input")
        elif analysis_row and analysis_row["status"] == "absent":
            filtered_row = next((r for r in trace if r["stage"] == "filtered"), None)
            stage_name = (t("trace_stage_filtered_matrix")
                          if (filtered_row and filtered_row["status"] == "absent")
                          else t("trace_stage_preprocessing"))
            verdict = t("trace_verdict_filtered_out",
                        (input_row or {}).get("detected_samples", ""),
                        (input_row or {}).get("total_samples", ""),
                        (input_row or {}).get("detected_fraction", ""),
                        int(stages.get("analysed", (None, []))[0].shape[0])
                        if stages.get("analysed") else t("unknown"))
        elif analysis_row and analysis_row["status"] == "present" and not in_diff:
            verdict = t("trace_verdict_not_computed")
        rows.append({"candidate": gene, "stage": "verdict", "protein_group": "", "detected_samples": 0,
                     "total_samples": 0, "detected_fraction": 0.0, "status": "summary", "reason": verdict})
    return pd.DataFrame(rows)


def estimate_confounding(run_folder: Path, n_perm: int = 500, out_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """Is the declared contrast estimable in the joint (treatment + batch) design?

    Estimability is decided by the contrast vector itself: c is estimable iff its projection on
    the batch space leaves a non-zero residual. A treatment level confined to one batch is not
    disqualifying as long as the contrast can still be formed. The adjusted effect is the
    least-squares coefficient of that same contrast with batch in the model, and the p-value
    comes from permuting treatment labels *within batch strata*, computing the statistic on the
    SAME full protein set every time (no selection on observed effects).
    """
    params = _params(run_folder)
    design = params.get("analysis_design") or {}
    files = _matrix_files(run_folder, params)
    matrix_path = files["analysed"] or files["filtered"]
    si_path = _resolve(params.get("sampleinfo_path", ""), run_folder)
    empty = pd.DataFrame()
    if matrix_path is None or si_path is None or not si_path.exists():
        # R27: separate "the analysis has not produced a matrix yet" from "a matrix that
        # should exist cannot be resolved". This verdict is shown to the model, so it must
        # not assert a data defect that has not happened.
        produced = sorted(q.name for q in (run_folder / "processed_proteins").glob("*.csv")) if (run_folder / "processed_proteins").exists() else []
        if not produced:
            return empty, t("conf_no_matrix_yet")
        return empty, t("conf_no_matrix_or_sampleinfo")
    matrix, sampleinfo = pd.read_csv(matrix_path), pd.read_csv(si_path)
    group_col = str(design.get("group_col") or "")
    batch_col = str(design.get("batch_col") or "")
    id_col = str(design.get("sample_id_col") or "")
    contrasts = (design.get("differential") or {}).get("contrasts") or []
    if not group_col or group_col not in sampleinfo.columns:
        return empty, t("conf_no_group_column")
    if not batch_col or batch_col not in sampleinfo.columns:
        return empty, t("conf_no_batch_column")
    sample_cols = _sample_cols(matrix, sampleinfo, id_col)
    if len(sample_cols) < 6:
        return empty, t("conf_too_few_samples")
    prot_col = _col(matrix, PROTEIN_COL_CANDIDATES)
    df = matrix.set_index(matrix[prot_col].astype(str) if prot_col else matrix.index)[sample_cols].apply(pd.to_numeric, errors="coerce")
    df = df.loc[df.notna().all(axis=1)]
    if df.shape[0] < 10:
        return empty, t("conf_too_few_proteins")
    meta = sampleinfo.set_index(sampleinfo[id_col].astype(str)).reindex([str(c) for c in sample_cols])
    treat = np.asarray([str(v) for v in meta[group_col].tolist()])
    batch = np.asarray([str(v) for v in meta[batch_col].tolist()])
    levels = sorted(set(treat))
    if len(contrasts) and str(contrasts[0].get("group_a", "")) in set(treat) and str(contrasts[0].get("group_b", "")) in set(treat):
        arm_a, arm_b = str(contrasts[0]["group_a"]), str(contrasts[0]["group_b"])
        contrast_name = str(contrasts[0].get("name", "%s_vs_%s" % (arm_a, arm_b)))
    else:
        arm_a, arm_b = levels[0], levels[-1]
        contrast_name = "%s_vs_%s" % (arm_a, arm_b)
    # full one-hot parameterisation (reference = first level) so the reported effect IS the
    # difference between the two arms; a +1/-1 coding would return half of it.
    treat_levels = sorted(set(treat))
    ref = treat_levels[0]
    treat_cols = [lv for lv in treat_levels if lv != ref]
    T = np.column_stack([(treat == lv).astype(float) for lv in treat_cols])
    w = np.array([1.0 if lv == arm_a else (-1.0 if lv == arm_b else 0.0) for lv in treat_cols])
    b_levels = sorted(set(batch))
    B = np.column_stack([(batch == lv).astype(float) for lv in b_levels[1:]]) if len(b_levels) > 1 else np.zeros((len(batch), 0))
    X = np.column_stack([np.ones(len(batch)), T, B]) if B.size else np.column_stack([np.ones(len(batch)), T])
    c_coef = np.concatenate([[0.0], w, np.zeros(B.shape[1])])
    XtX_pinv = np.linalg.pinv(X.T @ X)
    # c is estimable iff it lies in range(X'X) = row space of X: project and look at the residual
    c_res = c_coef - ((XtX_pinv @ (X.T @ X)) @ c_coef)
    resid_norm = float(c_res @ c_res)
    scale_ref = float(c_coef @ c_coef) or 1.0
    estimable = resid_norm <= 1e-8 * scale_ref
    # variance inflation versus the treatment-only design separates "identified" from "barely"
    X_u = np.column_stack([np.ones(len(treat)), T]) if T.size else np.ones((len(treat), 1))
    c_coef_u = np.concatenate([[0.0], w]) if w.size else np.array([0.0])
    Xu_pinv = np.linalg.pinv(X_u.T @ X_u)
    var_adj = float(c_coef @ (XtX_pinv @ c_coef))
    var_unadj = float(c_coef_u @ (Xu_pinv @ c_coef_u))
    vif = (var_adj / var_unadj) if var_unadj > 0 else float("inf")
    weakly_identified = estimable and vif > 50

    y = df.to_numpy(dtype=float)
    Y = y.T
    adjusted = (c_coef @ (XtX_pinv @ (X.T @ Y))) if estimable else np.full(y.shape[0], np.nan)
    unadjusted = y[:, treat == arm_a].mean(axis=1) - y[:, treat == arm_b].mean(axis=1)
    ok = np.isfinite(adjusted) & np.isfinite(unadjusted)
    adj_ok, unadj_ok = adjusted[ok], unadjusted[ok]
    corr = float(stats.spearmanr(adj_ok, unadj_ok).statistic) if adj_ok.size > 2 else float("nan")
    sign_agree = int(np.sum(np.sign(adj_ok) == np.sign(unadj_ok)))

    p_adj = float("nan")
    if estimable:
        rng = np.random.default_rng(SEED)
        # permutation set is fixed by protein variance only (label-independent), identical for
        # the observed statistic and every permutation.
        var = Y.var(axis=0)
        keep = np.argsort(-var)[: min(800, Y.shape[1])]
        Y_perm = Y[:, keep]
        obs_stat = float(np.mean(np.abs(adjusted[keep])))
        null = np.empty(n_perm)
        for i in range(n_perm):
            perm = treat.copy()
            for level in b_levels:
                idx_b = np.where(batch == level)[0]
                perm[idx_b] = rng.permutation(perm[idx_b])
            perm = np.asarray([str(v) for v in perm])
            Tk = np.column_stack([(perm == lv).astype(float) for lv in treat_cols])
            Xk = np.column_stack([np.ones(len(batch)), Tk, B]) if B.size else np.column_stack([np.ones(len(batch)), Tk])
            Xk_pinv = np.linalg.pinv(Xk.T @ Xk)
            ck = Xk @ (Xk_pinv @ c_coef)
            den = float(ck @ ck)
            if den <= 1e-10:
                null[i] = np.nan
                continue
            null[i] = float(np.mean(np.abs(c_coef @ (Xk_pinv @ (Xk.T @ Y_perm)))))
        valid_null = null[~np.isnan(null)]
        p_adj = float((np.sum(valid_null >= obs_stat) + 1) / (len(valid_null) + 1)) if len(valid_null) else float("nan")

    x = df.to_numpy(dtype=float).T
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
    k = min(5, min(x.shape) - 1)
    u, sv, _ = np.linalg.svd(x, full_matrices=False)
    pcs = u[:, :k] * sv[:k]

    def r2_by_factor(scores, factor):
        factor = np.asarray([str(v) for v in np.asarray(factor).ravel()])
        out = []
        for j in range(scores.shape[1]):
            yv = scores[:, j]
            grand = yv.mean()
            ss_tot = ((yv - grand) ** 2).sum()
            ss_between = 0.0
            for level in np.unique(factor):
                m = factor == level
                ss_between += m.sum() * (yv[m].mean() - grand) ** 2
            out.append(float(ss_between / ss_tot) if ss_tot > 0 else 0.0)
        return out

    eta_t = float(np.mean(r2_by_factor(pcs, treat)))
    eta_b = float(np.mean(r2_by_factor(pcs, batch)))
    cross = pd.crosstab(pd.Series(treat, name=group_col), pd.Series(batch, name=batch_col))
    min_cell = int(cross.to_numpy().min())

    rows = [
        {"metric": "contrast", "value": contrast_name, "note": "%s - %s" % (arm_a, arm_b)},
        {"metric": "n_samples", "value": len(sample_cols), "note": ""},
        {"metric": "n_proteins_complete", "value": int(df.shape[0]),
         "note": t("conf_note_same_complete_set")},
        {"metric": "contrast_estimable", "value": int(estimable),
         "note": t("conf_note_projection_residual", resid_norm)},
        {"metric": "contrast_vif", "value": round(vif, 3) if np.isfinite(vif) else None,
         "note": t("conf_note_variance_inflation")},
        {"metric": "contrast_weakly_identified", "value": int(weakly_identified),
         "note": t("conf_note_weak_identification")},
        {"metric": "treatment_levels", "value": len(levels),
         "note": t("list_separator").join(levels)},
        {"metric": "batch_levels", "value": len(b_levels),
         "note": t("list_separator").join(b_levels)},
        {"metric": "min_cell_size_treatment_x_batch", "value": min_cell,
         "note": t("conf_note_empty_cell")},
        {"metric": "adjusted_contrast_median", "value": round(float(np.median(adj_ok)), 4) if adj_ok.size else None,
         "note": t("conf_note_adjusted_median", contrast_name)},
        {"metric": "adjusted_contrast_iqr", "value": round(float(np.percentile(adj_ok, 75) - np.percentile(adj_ok, 25)), 4) if adj_ok.size else None, "note": ""},
        {"metric": "unadjusted_vs_adjusted_spearman", "value": round(corr, 4) if corr == corr else None, "note": ""},
        {"metric": "unadjusted_vs_adjusted_sign_agreement", "value": "%d/%d" % (sign_agree, adj_ok.size), "note": ""},
        {"metric": "within_batch_permutation_p", "value": round(p_adj, 4) if p_adj == p_adj else None,
         "note": t("conf_note_permutation", n_perm, int(min(800, Y.shape[1])), int(Y.shape[1]))},
        {"metric": "eta2_treatment_pc_descriptive", "value": round(eta_t, 4),
         "note": t("conf_note_eta2_treatment")},
        {"metric": "eta2_batch_pc_descriptive", "value": round(eta_b, 4),
         "note": t("conf_note_eta2_batch")},
    ]
    cell_note = t("conf_note_empty_cell_estimable") if min_cell == 0 else ""
    if not estimable:
        verdict = t("conf_verdict_not_estimable", contrast_name, resid_norm)
    elif weakly_identified:
        verdict = t("conf_verdict_weakly_identified", contrast_name, vif,
                    cell_note or t("conf_nearly_collinear"))
    else:
        verdict = t("conf_verdict_estimable", contrast_name, vif, min_cell, cell_note,
                    float(np.median(adj_ok)) if adj_ok.size else float("nan"),
                    float(np.percentile(adj_ok, 75) - np.percentile(adj_ok, 25)) if adj_ok.size else float("nan"),
                    corr, sign_agree, adj_ok.size, p_adj,
                    int(min(800, Y.shape[1])), int(Y.shape[1]), eta_t, eta_b)
    try:
        target_dir = Path(out_dir) if out_dir else (run_folder / "evaluation_evidence_ext")
        target_dir.mkdir(parents=True, exist_ok=True)
        proteins = list(df.index)
        pd.DataFrame({
            "protein_group": proteins,
            "adjusted_contrast_coef": adjusted,
            "unadjusted_mean_difference": unadjusted,
        }).to_csv(target_dir / "confounding_adjusted_effects.csv", index=False)
    except Exception:
        pass
    return pd.DataFrame(rows), verdict


def _effect_scale(design: Dict[str, Any], transform_record: Optional[Dict[str, Any]] = None) -> Tuple[str, bool, str]:
    """(label, is_confirmed_log2, state) with state in {log2, linear, unknown}.

    A transform RECORD is required to claim log2; a value range can only flag anomalies and can
    never prove the transform or its base. Heuristics such as "looks_logged" stay in the unknown
    bucket and must not license a log2 effect threshold.
    """
    matrix = design.get("matrix", {}) if isinstance(design.get("matrix"), dict) else {}
    record = matrix.get("log2_transform_applied")
    if record is None and isinstance(transform_record, dict):
        _applied = (transform_record.get('log2_transform') or {}).get('applied')
        if _applied is True:
            return (t('scale_log2_from_record'), True, 'log2')
        if _applied is False:
            record = False
    mn, mx = matrix.get("numeric_min"), matrix.get("numeric_max")
    range_large_positive = None
    try:
        if mn is not None and mx is not None:
            range_large_positive = (float(mn) > 0) and (float(mx) > 1000)
    except Exception:
        range_large_positive = None
    if record is True:
        return t("scale_log2_from_design"), True, "log2"
    if record is False and range_large_positive:
        return t("scale_linear_recorded"), False, "linear"
    if bool(matrix.get("looks_logged")):
        return t("scale_unknown_heuristic"), False, "unknown"
    return t("scale_unknown"), False, "unknown"




def _transform_record_for(run_folder) -> Dict[str, Any]:
    # R29 single authority for the effect scale: the record written by the step that actually
    # transformed the matrix.  Both unit-sensitivity writers read it, so the two copies of the
    # same table can no longer disagree about whether the effect is on a log2 scale.
    if run_folder is None:
        return {}
    run_folder = Path(run_folder)
    candidates = [run_folder / 'processed_proteins' / 'matrix_transform_record.json',
                  run_folder / 'matrix_transform_record.json']
    for path in candidates:
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(payload, dict):
                    return payload
        except Exception:
            continue
    events = run_folder / 'run_events.jsonl'
    try:
        if events.exists():
            for line in events.read_text(encoding='utf-8', errors='replace').splitlines():
                if 'log2_transform' not in line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                blob = json.dumps(item, ensure_ascii=False)
                if 'combat_calibration' in blob and 'log2_transform' in blob:
                    return {'record_type': 'matrix_transform_record_derived_from_run_events',
                            'log2_transform': {'applied': ('"applied": true' in blob
                                                          or '"applied":true' in blob)}}
    except Exception:
        pass
    return {}

def _effect_scale_label(design: Dict[str, Any]) -> str:
    return _effect_scale(design)[0]


def stratified_contrasts(run_folder: Path, min_n: int = 5, contrast_name: Optional[str] = None,
                         stratifiers: Optional[List[str]] = None, out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Within-stratum effects for one explicitly chosen contrast.

    Records the contrast direction, the effect scale actually used, the effective observation
    counts, and writes the complete per-stratum result table so any candidate can be checked.
    Separate stratification is not joint control: a stratum-adjusted analysis needs both factors
    in one model and is not produced here.
    """
    params = _params(run_folder)
    design = params.get("analysis_design") or {}
    files = _matrix_files(run_folder, params)
    matrix_path = files["analysed"] or files["filtered"]
    si_path = _resolve(params.get("sampleinfo_path", ""), run_folder)
    rows: List[Dict[str, Any]] = []
    if matrix_path is None or si_path is None or not si_path.exists():
        return pd.DataFrame(rows)
    matrix, sampleinfo = pd.read_csv(matrix_path), pd.read_csv(si_path)
    group_col = str(design.get("group_col") or "")
    id_col = str(design.get("sample_id_col") or "")
    contrasts = (design.get("differential") or {}).get("contrasts") or []
    if not group_col or not contrasts:
        return pd.DataFrame(rows)
    chosen = None
    if contrast_name:
        chosen = next((c for c in contrasts if str(c.get("name")) == contrast_name), None)
    chosen = chosen or contrasts[0]
    arm_a, arm_b = str(chosen.get("group_a", "")), str(chosen.get("group_b", ""))
    contrast = str(chosen.get("name", ""))
    logfc_thresh = float((design.get("differential") or {}).get("logfc_thresh") or 0.25)
    p_thresh = float((design.get("differential") or {}).get("p_thresh") or 0.05)
    gene_col = _col(matrix, GENE_COL_CANDIDATES)
    prot_col = _col(matrix, PROTEIN_COL_CANDIDATES)
    scale_label, log2_confirmed, scale_state = _effect_scale(design, _transform_record_for(run_folder))
    sampleinfo = sampleinfo.copy()
    sampleinfo["_sid"] = sampleinfo[id_col].astype(str)
    pooled_path = run_folder / "processed_proteins" / ("differential_%s.csv" % contrast)
    pooled = pd.read_csv(pooled_path) if pooled_path.exists() else None
    pooled_key = ""
    if pooled is not None:
        pooled_prot = _col(pooled, PROTEIN_COL_CANDIDATES)
        pooled_gene = _col(pooled, GENE_COL_CANDIDATES)
        pooled_key = pooled_prot or pooled_gene
    source_vars = stratifiers or ([str(v) for v in (design.get("covariates") or [])] + [str(design.get("batch_col") or "")])
    source_vars = [c for c in dict.fromkeys(source_vars) if c and c in sampleinfo.columns]
    replicate_like = [c for c in sampleinfo.columns
                      if any(k in c.lower() for k in ("donor", "subject", "patient", "mouse", "replicate", "animal"))]
    stratum_dir = (Path(out_dir) / "stratified") if out_dir else (run_folder / "evaluation_evidence_ext" / "stratified")
    for strat in source_vars:
        levels = [lvl for lvl in pd.unique(sampleinfo[strat]) if str(lvl) not in {"nan", "None", ""}]
        for level in levels:
            mask = sampleinfo[strat].astype(str) == str(level)
            sub_meta = sampleinfo.loc[mask]
            ids_a = set(sub_meta.loc[sub_meta[group_col].astype(str) == arm_a, "_sid"].astype(str))
            ids_b = set(sub_meta.loc[sub_meta[group_col].astype(str) == arm_b, "_sid"].astype(str))
            cols_a = [c for c in matrix.columns if str(c) in ids_a]
            cols_b = [c for c in matrix.columns if str(c) in ids_b]
            threshold_text = (t("strat_threshold_joint", logfc_thresh, p_thresh)
                              if log2_confirmed else
                              t("strat_threshold_fdr_only", p_thresh))
            base = {"stratifier": strat, "level": str(level), "contrast": contrast,
                    "direction": "%s - %s" % (arm_a, arm_b), "effect_scale": scale_label,
                    "scale_state": scale_state, "threshold_used": threshold_text,
                    "n_a": len(cols_a), "n_b": len(cols_b), "min_n_rule": min_n}
            for replicate in replicate_like[:2]:
                base["n_%s_a" % replicate] = int(sub_meta.loc[sub_meta[group_col].astype(str) == arm_a, replicate].nunique())
                base["n_%s_b" % replicate] = int(sub_meta.loc[sub_meta[group_col].astype(str) == arm_b, replicate].nunique())
            if len(cols_a) < min_n or len(cols_b) < min_n:
                rows.append({**base, "tested_proteins": 0, "n_sig": 0,
                             "verdict": t("strat_verdict_below_min_n"),
                             "note": t("strat_note_min_n", min_n, len(cols_a), len(cols_b))})
                continue
            effective_logfc = logfc_thresh if log2_confirmed else 0.0
            table = _welch_table(matrix, cols_a, cols_b, gene_col, prot_col, effective_logfc, p_thresh)
            table["threshold_used"] = (t("strat_threshold_joint", logfc_thresh, p_thresh)
                                       if log2_confirmed else
                                       t("strat_threshold_fdr_only", p_thresh))
            table.insert(0, "contrast", contrast)
            table.insert(1, "stratifier", strat)
            table.insert(2, "level", str(level))
            stratum_dir.mkdir(parents=True, exist_ok=True)
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in "%s__%s__%s" % (contrast, strat, level))
            table.to_csv(stratum_dir / ("%s.csv" % safe_name), index=False)
            n_sig = int(table["passes"].sum())
            agree = total = 0
            if pooled is not None and pooled_key:
                key_stratum = (table["protein_group"].astype(str) if "protein_group" in table.columns
                               else table["gene"].astype(str))
                merged = pd.DataFrame({"key": key_stratum.astype(str), "logFC_stratum": table["logFC"]})
                pooled_small = pd.DataFrame({
                    "key": pooled[pooled_key].astype(str),
                    "logFC_pooled": pd.to_numeric(pooled["logFC"], errors="coerce"),
                }).drop_duplicates(subset=["key"])
                joined = merged.merge(pooled_small, on="key", how="inner").dropna()
                total = int(joined.shape[0])
                agree = int((np.sign(joined["logFC_stratum"]) == np.sign(joined["logFC_pooled"])).sum())
            rows.append({**base, "tested_proteins": int(table.shape[0]), "n_sig": n_sig,
                         "verdict": t("strat_verdict_stratified"),
                         "note": t("strat_note_alignment", agree, total, total - agree, safe_name)})
    if rows:
        rows.append({"stratifier": "—", "level": "—", "contrast": contrast, "direction": "%s - %s" % (arm_a, arm_b),
                     "effect_scale": scale_label, "n_a": "", "n_b": "", "min_n_rule": min_n,
                     "tested_proteins": "", "n_sig": "",
                     "verdict": t("strat_verdict_convention_note"),
                     "note": t("strat_note_separate_not_joint")})
    return pd.DataFrame(rows)


def contrast_concordance(run_folder: Path, p_thresh: float = 0.05, logfc_thresh: float = 0.25) -> pd.DataFrame:
    """Overlap and sign agreement between contrasts, with raw P and FDR kept explicit.

    If a table ships no adjusted p-value, BH is recomputed inside the table and recorded as such;
    raw P is never silently treated as FDR. A zero significant set is reported as zero overlap
    with the direction-agreement statistic marked not applicable - not as "the contrasts cannot be
    compared": effect-size distributions remain informative.
    """
    tables: Dict[str, pd.DataFrame] = {}
    fdr_source: Dict[str, str] = {}
    for path in sorted((run_folder / "processed_proteins").glob("differential_*.csv")):
        df = pd.read_csv(path)
        gene_col = _col(df, GENE_COL_CANDIDATES)
        prot_col = _col(df, PROTEIN_COL_CANDIDATES)
        if not gene_col or "logFC" not in df.columns:
            continue
        name = path.stem.replace("differential_", "")
        p_raw = pd.to_numeric(df.get("P.Value"), errors="coerce")
        if "adj.P.Val" in df.columns and pd.to_numeric(df["adj.P.Val"], errors="coerce").notna().any():
            adj = pd.to_numeric(df["adj.P.Val"], errors="coerce")
            fdr_source[name] = "table.adj.P.Val"
        else:
            adj = pd.Series(bh_adjust(p_raw.to_numpy(dtype=float)), index=df.index)
            fdr_source[name] = "BH recomputed from P.Value (no adj.P.Val in table)"
        sub = pd.DataFrame({
            "key": (df[prot_col].astype(str) if prot_col else _symbol(df[gene_col])),
            "gene": _symbol(df[gene_col]),
            "logFC": pd.to_numeric(df["logFC"], errors="coerce"),
            "p_raw": p_raw,
            "adj": adj,
        }).dropna(subset=["logFC"])
        sub = sub[sub["key"].astype(str).str.strip().ne("")].drop_duplicates(subset=["key"])
        tables[name] = sub.set_index("key")
    rows: List[Dict[str, Any]] = []
    names = sorted(tables)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = tables[names[i]], tables[names[j]]
            shared = a.index.intersection(b.index)
            if len(shared) < 10:
                rows.append({"contrast_a": names[i], "contrast_b": names[j], "shared_proteins": len(shared),
                             "spearman_logfc": None, "median_abs_logfc_a": None, "median_abs_logfc_b": None,
                             "n_sig_a": 0, "n_sig_b": 0, "overlap_sig": None, "fisher_p": None,
                             "sign_agreement": t("conc_not_applicable"),
                             "fdr_source_a": fdr_source.get(names[i], ""),
                             "fdr_source_b": fdr_source.get(names[j], ""),
                             "verdict": t("conc_insufficient_shared")})
                continue
            sa = a.reindex(shared)["logFC"].astype(float)
            sb = b.reindex(shared)["logFC"].astype(float)
            try:
                rho = float(stats.spearmanr(sa.to_numpy(), sb.to_numpy()).statistic)
            except Exception:
                rho = float("nan")
            sig_a = (a["adj"] < p_thresh) & (a["logFC"].abs() > logfc_thresh)
            sig_b = (b["adj"] < p_thresh) & (b["logFC"].abs() > logfc_thresh)
            set_a, set_b = set(a.index[sig_a]), set(b.index[sig_b])
            both = list(set_a & set_b)
            fisher_p = None
            if set_a and set_b:
                only_a = len((set_a - set_b) & set(shared))
                only_b = len((set_b - set_a) & set(shared))
                neither = len(set(shared) - set_a - set_b)
                _, fisher_p = stats.fisher_exact([[len(both), max(0, only_a)], [max(0, only_b), max(0, neither)]])
            if set_a and set_b:
                verdict = t("conc_comparable")
                sign_text = "%d/%d" % (sum(1 for k in both if np.sign(a.loc[k, "logFC"]) == np.sign(b.loc[k, "logFC"])), len(both)) if both else t("conc_no_shared_significant")
            else:
                verdict = t("conc_zero_overlap", names[i], len(set_a), names[j], len(set_b))
                sign_text = t("conc_not_applicable")
            rows.append({"contrast_a": names[i], "contrast_b": names[j], "shared_proteins": len(shared),
                         "spearman_logfc": round(rho, 4) if rho == rho else None,
                         "median_abs_logfc_a": round(float(sa.abs().median()), 4),
                         "median_abs_logfc_b": round(float(sb.abs().median()), 4),
                         "n_sig_a": len(set_a), "n_sig_b": len(set_b),
                         "overlap_sig": len(both) if (set_a and set_b) else 0,
                         "fisher_p": round(float(fisher_p), 4) if fisher_p is not None else None,
                         "sign_agreement": sign_text,
                         "fdr_source_a": fdr_source.get(names[i], ""), "fdr_source_b": fdr_source.get(names[j], ""),
                         "note": t("conc_note_fdr_scope"),
                         "verdict": verdict})
    return pd.DataFrame(rows)


DOSE_DRUGS = ("Bortezomib", "Cycloheximide")


def _dose_design(sampleinfo: pd.DataFrame) -> pd.DataFrame:
    """Parse (stratum, drug, dose) from the sample labels; a control sample gets dose 0."""
    rows = []
    for _, record in sampleinfo.iterrows():
        label = str(record.get("Label") or "")
        stratum = "single_cell" if "single_cell" in label else ("pool" if "10cell_pool" in label else "unknown")
        drug, dose = "Control", 0.0
        for name in DOSE_DRUGS:
            if name.lower() in label.lower():
                drug = name
                match = re.search(r"%s\s+([0-9.]+)\s*uM" % name, label)
                dose = float(match.group(1)) if match else float("nan")
                break
        rows.append({"sample": str(record.get("FileName") or record.get("_sample_id") or ""),
                     "label": label, "stratum": stratum, "drug": drug, "dose": dose})
    return pd.DataFrame(rows)


def dose_trends(run_folder: Path, out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Spearman dose-gradient statistics per drug, stratified by sample type. No network.

    Protein level: rho between dose and abundance across the control/low/high samples of one
    stratum; module level: rho between dose and the curated module score of the same samples.
    Every row carries rho, the two-sided p, the BH q and the sample count. A stratum with fewer
    than three samples in any dose step is reported as insufficient instead of being tested;
    module scores never substitute for a protein-level trend test.
    """
    run_folder = Path(run_folder)
    params = _params(run_folder)
    files = _matrix_files(run_folder, params)
    sampleinfo_path = run_folder / "processed_proteins" / "SampleInfo_Filtered.csv"
    matrix_path = files.get("analysed")
    if not sampleinfo_path.exists() or not matrix_path:
        return pd.DataFrame()
    sampleinfo = _read(sampleinfo_path)
    design = _dose_design(sampleinfo)
    matrix = _read(matrix_path)
    gene_col = _col(matrix, GENE_COL_CANDIDATES)
    sample_cols = _sample_cols(matrix, sampleinfo, "FileName")
    if not sample_cols or not gene_col:
        return pd.DataFrame()
    module_path = run_folder / "evaluation_evidence" / "curated_module_sample_scores.csv"
    module = _read(module_path) if module_path.exists() else pd.DataFrame()
    module_cols = [c for c in module.columns
                   if c not in ("FileName", "Cluster", "Label", "_sample_id")] if not module.empty else []
    rows = []
    for drug in DOSE_DRUGS:
        for stratum in sorted({s for s in design["stratum"] if s and s != "unknown"}):
            subset = design[(design["stratum"] == stratum) & (design["drug"].isin([drug, "Control"]))]
            subset = subset[subset["dose"].notna()]
            if subset["drug"].nunique() < 2:
                continue
            steps: Dict[int, int] = {}
            for dose_value in subset["dose"].tolist():
                steps[int(dose_value)] = steps.get(int(dose_value), 0) + 1
            label = "%s / %s" % (drug, stratum)
            if min(steps.values()) < 3:
                rows.append({"level": "protein", "drug": drug, "stratum": stratum, "name": "(whole matrix)",
                             "n_samples": int(len(subset)), "dose_steps": str(steps), "rho": np.nan,
                             "p_value": np.nan, "q_value": np.nan, "direction": "", "significant": "",
                             "note": "n=%d per dose step is insufficient for a trend test" % min(steps.values())})
                continue
            sample_list = [s for s in subset["sample"].tolist() if s in set(sample_cols)]
            doses = subset.set_index("sample").loc[sample_list, "dose"].astype(float).tolist()
            block = matrix.set_index(gene_col)[sample_list].apply(pd.to_numeric, errors="coerce")
            stats_rows = []
            for gene, values in block.iterrows():
                pair = pd.DataFrame({"dose": doses, "value": values.tolist()}).dropna()
                if len(pair) < 6 or pair["value"].nunique() < 2 or pair["dose"].nunique() < 2:
                    continue
                rho, p_value = stats.spearmanr(pair["dose"], pair["value"])
                stats_rows.append((str(gene), float(rho), float(p_value), int(len(pair))))
            if stats_rows:
                q_values = bh_adjust([item[2] for item in stats_rows])
                for (gene, rho, p_value, n_used), q_value in zip(stats_rows, q_values):
                    rows.append({"level": "protein", "drug": drug, "stratum": stratum, "name": gene,
                                 "n_samples": n_used, "dose_steps": str(steps), "rho": round(rho, 4),
                                 "p_value": float(p_value), "q_value": float(q_value),
                                 "direction": "rising with dose" if rho > 0 else "falling with dose",
                                 "significant": "yes" if q_value <= 0.05 else "no", "note": ""})
            if module_cols:
                module_lookup = module.set_index("FileName") if "FileName" in module.columns else module
                module_stats = []
                for name in module_cols:
                    if name not in module_lookup.columns:
                        continue
                    values = pd.to_numeric(module_lookup[name], errors="coerce")
                    pair = pd.DataFrame({"dose": doses,
                                         "value": [values.get(s, np.nan) for s in sample_list]}).dropna()
                    if len(pair) < 6 or pair["value"].nunique() < 2:
                        continue
                    rho, p_value = stats.spearmanr(pair["dose"], pair["value"])
                    module_stats.append((name, float(rho), float(p_value), int(len(pair))))
                if module_stats:
                    q_values = bh_adjust([item[2] for item in module_stats])
                    for (name, rho, p_value, n_used), q_value in zip(module_stats, q_values):
                        rows.append({"level": "module", "drug": drug, "stratum": stratum, "name": name,
                                     "n_samples": n_used, "dose_steps": str(steps), "rho": round(rho, 4),
                                     "p_value": float(p_value), "q_value": float(q_value),
                                     "direction": "rising with dose" if rho > 0 else "falling with dose",
                                     "significant": "yes" if q_value <= 0.05 else "no", "note": ""})
    frame = pd.DataFrame(rows)
    if out_dir is not None and not frame.empty:
        frame.to_csv(Path(out_dir) / "dose_trend.csv", index=False)
    return frame


def enrichment_ora(run_folder: Path, out_dir: Optional[Path] = None,
                   namespaces: Tuple[str, ...] = ("KEGG", "Reactome", "GO"),
                   p_thresh: float = 0.05, logfc_thresh: float = 0.25,
                   min_query: int = 5, top_per_bucket: int = 8) -> pd.DataFrame:
    """Hypergeometric over-representation with the experiment's own detected proteins as background.

    Query = genes of the proteins passing the report's screening criterion for one contrast and
    direction; background = genes of every protein present in the analysed matrix. The p value comes
    from the production hypergeometric tail and q is BH over the terms actually tested. A query
    smaller than min_query is reported as insufficient rather than tested, and no FDR is invented
    when the background cannot be built.
    """
    run_folder = Path(run_folder)
    params = _params(run_folder)
    files = _matrix_files(run_folder, params)
    matrix_path = files.get("analysed")
    if not matrix_path:
        return pd.DataFrame()
    project = Path(__file__).resolve().parent
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    try:
        import offline_enrichment as offline
    except Exception:
        return pd.DataFrame()
    species = str(params.get("species") or "human")
    if "mus" in species.lower() or "mouse" in species.lower():
        species = "mouse"
    else:
        species = "human"
    matrix = _read(matrix_path)
    gene_col = _col(matrix, GENE_COL_CANDIDATES)
    if not gene_col:
        return pd.DataFrame()
    background = sorted({g for g in _symbol(matrix[gene_col]).dropna().tolist() if g and g != "nan"})
    if len(background) < 50:
        return pd.DataFrame()
    background_set = set(background)
    rows = []
    proc = run_folder / "processed_proteins"
    for path in sorted(proc.glob("differential_*.csv")):
        contrast = path.stem.replace("differential_", "")
        table = _read(path)
        gene_table_col = _col(table, GENE_COL_CANDIDATES)
        if not gene_table_col or "adj.P.Val" not in table.columns or "logFC" not in table.columns:
            continue
        table = table.assign(_gene=_symbol(table[gene_table_col]).astype(str).str.strip(),
                             _adj=pd.to_numeric(table["adj.P.Val"], errors="coerce"),
                             _logfc=pd.to_numeric(table["logFC"], errors="coerce"))
        selected = table[(table["_adj"] <= p_thresh) & (table["_logfc"].abs() >= logfc_thresh)]
        for direction, mask in (("upregulated", selected["_logfc"] > 0), ("downregulated", selected["_logfc"] < 0)):
            query = sorted({str(g) for g in selected.loc[mask, "_gene"].tolist() if str(g) not in ("", "nan", "None")})
            definition = "significant set: adj.P<=%s and |logFC|>=%s" % (p_thresh, logfc_thresh)
            significant_size = len(query)
            if len(query) < min_query:
                # a documented fallback keeps a real test possible when the significant set is tiny:
                # the top-ranked fraction by |logFC|, never presented as a significance result
                ranked = table.assign(_abs=table["_logfc"].abs()).sort_values("_abs", ascending=False)
                size = max(min_query, min(len(ranked), int(round(0.10 * len(ranked))) or min_query))
                fallback = ranked.head(size)
                fallback_mask = fallback["_logfc"] > 0 if direction == "upregulated" else fallback["_logfc"] < 0
                query = sorted({str(g) for g in fallback.loc[fallback_mask, "_gene"].tolist()
                                if str(g) not in ("", "nan", "None")})
                definition = ("fallback query: top %d ranked by |logFC| (the significant set had %d genes); "
                              "descriptive, not a significance result" % (size, significant_size))
            if len(query) < min_query:
                rows.append({"contrast": contrast, "direction": direction, "namespace": "", "term": "",
                             "hits": "", "term_size_in_background": "", "background_size": len(background),
                             "query_size": len(query), "p_value": np.nan, "q_value": np.nan, "hit_genes": "",
                             "query_definition": definition,
                             "note": "query size %d < %d: no test performed" % (len(query), min_query)})
                continue
            query_set = set(query)
            for namespace in namespaces:
                try:
                    gene_sets = offline.load_gene_sets(namespace, species)
                except Exception:
                    continue
                tested = []
                for term_id, entry in gene_sets.items():
                    term_genes = set(offline.normalize_gene_list(entry.get("genes") or [], species)) & background_set
                    if not term_genes:
                        continue
                    overlap = query_set & term_genes
                    if not overlap:
                        continue
                    p_value = offline.hypergeom_sf(len(overlap), len(background_set), len(term_genes), len(query_set))
                    tested.append({"term": offline.clean_term_description(term_id), "hits": len(overlap),
                                   "term_size_in_background": len(term_genes), "p_value": float(p_value),
                                   "hit_genes": "/".join(sorted(overlap))[:200]})
                if not tested:
                    continue
                q_values = offline.bh_adjust([item["p_value"] for item in tested])
                for item, q_value in zip(tested, q_values):
                    item["q_value"] = float(q_value)
                tested.sort(key=lambda item: (item["p_value"], -item["hits"]))
                for item in tested[:top_per_bucket]:
                    rows.append({"contrast": contrast, "direction": direction, "namespace": namespace,
                                 "term": item["term"], "hits": item["hits"],
                                 "term_size_in_background": item["term_size_in_background"],
                                 "background_size": len(background), "query_size": len(query),
                                 "p_value": round(item["p_value"], 6), "q_value": round(item["q_value"], 6),
                                 "hit_genes": item["hit_genes"],
                                 "query_definition": definition,
                                 "note": "q<=0.05" if item["q_value"] <= 0.05 else ""})
    frame = pd.DataFrame(rows)
    if out_dir is not None and not frame.empty:
        frame.to_csv(Path(out_dir) / "enrichment_ora.csv", index=False)
    return frame




# ---------------------------------------------------------------------------
# Unit-aware sensitivity analysis (2026-09-14)
#
# analysis_design may declare an individual / experimental-unit column
# (pair_col) together with further covariates. Until now nothing consumed
# pair_col, so a contrast was always evaluated at the observation level even
# when many observations shared one biological unit. This block makes the unit
# structure explicit, checks whether the contrast is estimable at unit level,
# and - when it is - produces a documented sensitivity analysis that
# aggregates observations to the unit before testing.
#
# Rules kept deliberately conservative:
#   * the declared design wins; a column-name match is recorded as unconfirmed;
#   * a column that happens to be called Donor/Replicate is never promoted to
#     "independent biological replicate" on the strength of its name alone;
#   * the observation-level result stays the reference analysis; the unit-level
#     result is a sensitivity analysis with a different estimand;
#   * when the structure is not estimable we return the reason, not a number;
#   * a simpler unit-level test is preferred over an ill-conditioned model.
# ---------------------------------------------------------------------------

UNIT_MIN_UNITS_PER_ARM = 3
UNIT_MIN_PAIRED_UNITS = 3
UNIT_COLUMN_HINTS = ("donor", "subject", "patient", "mouse", "animal", "individual", "biorep", "replicate")
# R25: a unit label that names a mixture, a pool or an unassigned value is not evidence of an
# independent biological individual. Classification is by label token only - never by how the
# resulting test behaves, so the choice of method cannot follow the significance of the result.
UNIT_POOL_TOKENS = frozenset({"mixed", "mix", "pool", "pooled", "combine", "combined",
                              "composite", "concatenated"})
UNIT_UNCONFIRMED_TOKENS = frozenset({"unknown", "unassigned", "unlabelled", "unlabeled",
                                      "none", "null", "na", "tbd", "unclear"})
UNIT_EXCLUSION_RULE = ("实验单位的独立生物学个体判定规则：标签中出现 mixed/mix/pool/combined/"
                       "composite 的样本按混合来源处理，标签为 unknown/unassigned/none/null/NA 的按"
                       "未确认来源处理；两者都不作为独立生物学个体进入个体层检验，也不参与个体间比较。")


def unit_exclusion_rule() -> str:
    """The released zh rule for a zh run; the registered bilingual entry for an en run."""
    if report_language.get_language() != "en":
        return UNIT_EXCLUSION_RULE
    return t("unit_exclusion_rule")


def _unit_label_tokens(label: Any) -> set:
    """Token set of a unit label, split on case boundaries so OldMixed -> {old, mixed}."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(label).strip())
    return {tok for tok in re.split(r"[^0-9A-Za-z]+", text.lower()) if tok}


def _unit_class(label: Any) -> str:
    tokens = _unit_label_tokens(label)
    if tokens & UNIT_POOL_TOKENS:
        return "pooled_source"
    if tokens & UNIT_UNCONFIRMED_TOKENS:
        return "unconfirmed_source"
    return "individual_like"


def _unit_column(sampleinfo: pd.DataFrame, design: Dict[str, Any]) -> Tuple[str, str]:
    """(column, source). A declared pair_col wins; name hints stay unconfirmed."""
    declared = str(design.get("pair_col") or "").strip()
    if declared and declared in sampleinfo.columns:
        # R26: an auto-inferred design is generated by this pipeline, not confirmed by the
        # researcher; the pair column keeps that provenance instead of posing as declared.
        if str(design.get("source") or "").strip().lower() == "auto_inferred":
            return declared, "analysis_design.pair_col_auto_inferred"
        return declared, "analysis_design.pair_col"
    for col in sampleinfo.columns:
        low = str(col).lower()
        if any(hint in low for hint in UNIT_COLUMN_HINTS):
            values = sampleinfo[col].dropna().astype(str)
            values = values[values != ""]
            if 1 < values.nunique() < len(values):
                return str(col), "column_name_match_unconfirmed"
    return "", "absent"

def _unit_design_rank(sub: pd.DataFrame, group_col: str, unit_col: str, arm_a: str) -> int:
    """Rank of [intercept, arm indicator, unit dummies] over the analysed rows.

    This is a property of one joint fixed-effects model. It is reported as a diagnostic of
    whether treatment and unit identity can be separated inside that single model; the coverage
    counts below, not this rank, decide which estimand the data can support.
    """
    n = len(sub)
    if n == 0:
        return 0
    cols = [np.ones(n), (sub[group_col].astype(str) == arm_a).to_numpy(dtype=float)]
    dummies = pd.get_dummies(sub[unit_col].astype(str), drop_first=True)
    for name in dummies.columns:
        cols.append(dummies[name].to_numpy(dtype=float))
    design = np.column_stack(cols)
    return int(np.linalg.matrix_rank(design))


def _arm_vif(sub: pd.DataFrame, group_col: str, unit_col: str, arm_a: str) -> Dict[str, Any]:
    """How much of the arm indicator is already explained by the unit identities.

    Diagnostic only: a high value means the unit structure carries the same information as the
    contrast, which makes a joint model ill conditioned. It is never on its own a verdict that
    the contrast cannot be estimated - that decision stays with the coverage counts.
    """
    y = (sub[group_col].astype(str) == arm_a).to_numpy(dtype=float)
    dummies = pd.get_dummies(sub[unit_col].astype(str), drop_first=True).to_numpy(dtype=float)
    if dummies.shape[1] == 0:
        return {"vif_arm": 1.0, "r2_arm_from_units": 0.0}
    X = np.column_stack([np.ones(len(y)), dummies])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    r2 = max(0.0, min(1.0, r2))
    vif = float("inf") if r2 >= 1.0 - 1e-9 else float(1.0 / (1.0 - r2))
    return {"vif_arm": vif, "r2_arm_from_units": round(r2, 6)}


def effect_scale_record(design: Dict[str, Any], run_folder=None) -> Dict[str, Any]:
    """Machine-readable effect scale so an effect name matches the scale that was recorded."""
    transform_record = _transform_record_for(run_folder)
    label, confirmed_log2, state = _effect_scale(design, transform_record)
    declared = (design.get('matrix') or {}).get('log2_transform_applied') if isinstance(design.get('matrix'), dict) else None
    if state == 'log2' and declared is None and transform_record:
        source = str(transform_record.get('record_source') or
                     t('scale_source_run_record'))
    else:
        source = 'parameters.json -> analysis_design.matrix'
    return {"state": state, "label": label, "confirmed_log2": bool(confirmed_log2),
            "source": source}


def _effect_name(scale: Dict[str, Any]) -> str:
    """Column name for the unit-level effect that does not imply an unconfirmed log scale."""
    return "log2FC_units" if scale.get("state") == "log2" else "effect_units"


def unit_structure_diagnostics(sampleinfo: pd.DataFrame, design: Dict[str, Any], group_col: str,
                               arm_a: str, arm_b: str) -> Dict[str, Any]:
    """Design-level facts behind one contrast, separated by estimand.

    Nothing here is a test result. The block reports which unit labels may stand for independent
    biological individuals, how many of them each arm carries, which units are observed twice,
    and the diagnostics of the joint fixed-effects model. A unit observed in both arms is never
    counted as two independent samples: it either enters the within-unit (paired) estimand or it
    is left out of the between-unit comparison.
    """
    out: Dict[str, Any] = {
        "direction": "%s - %s" % (arm_a, arm_b),
        "group_column": group_col,
        "unit_column": "",
        "unit_column_source": "absent",
        "reasons": [],
    }
    if not group_col or group_col not in sampleinfo.columns:
        out["estimability"] = "not_estimable"
        out["analysis_route"] = "none"
        out["reasons"] = [t("unit_reason_group_col_missing")]
        return out
    sub = sampleinfo.loc[sampleinfo[group_col].astype(str).isin([arm_a, arm_b])].copy()
    out["n_observations"] = {
        "group_a": int((sub[group_col].astype(str) == arm_a).sum()),
        "group_b": int((sub[group_col].astype(str) == arm_b).sum()),
    }
    unit_col, unit_source = _unit_column(sub, design)
    out["unit_column"] = unit_col
    out["unit_column_source"] = unit_source
    if not unit_col:
        out["estimability"] = "not_estimable"
        out["analysis_route"] = "none"
        out["reasons"] = [t("unit_reason_no_unit_column")]
        return out
    sub = sub[sub[unit_col].notna()].copy()
    sub[unit_col] = sub[unit_col].astype(str).str.strip()
    sub = sub[~sub[unit_col].isin(["", "nan", "None", "NaN"])]
    if sub.empty:
        out["estimability"] = "not_estimable"
        out["analysis_route"] = "none"
        out["reasons"] = [t("unit_reason_unit_col_empty")]
        return out
    classes: Dict[str, List[str]] = {"individual_like": [], "pooled_source": [], "unconfirmed_source": []}
    for value in sorted(set(sub[unit_col])):
        classes[_unit_class(value)].append(str(value))
    excluded = set(classes["pooled_source"]) | set(classes["unconfirmed_source"])
    out["unit_classes"] = classes
    out["unit_exclusion"] = {
        "n_excluded": len(excluded),
        "pooled_source": classes["pooled_source"],
        "unconfirmed_source": classes["unconfirmed_source"],
        "rule": unit_exclusion_rule(),
    }
    all_units_a = set(sub.loc[sub[group_col].astype(str) == arm_a, unit_col])
    all_units_b = set(sub.loc[sub[group_col].astype(str) == arm_b, unit_col])
    out["n_units_all"] = {"group_a": len(all_units_a), "group_b": len(all_units_b),
                          "shared": len(all_units_a & all_units_b)}
    included = sub[~sub[unit_col].isin(excluded)].copy()
    if included.empty:
        out["estimability"] = "not_estimable"
        out["analysis_route"] = "none"
        out["reasons"] = [t("unit_reason_no_unit_left")]
        return out
    units_a = set(included.loc[included[group_col].astype(str) == arm_a, unit_col])
    units_b = set(included.loc[included[group_col].astype(str) == arm_b, unit_col])
    shared = units_a & units_b
    a_only = units_a - units_b
    b_only = units_b - units_a
    out["n_units"] = {"group_a": len(units_a), "group_b": len(units_b), "shared": len(shared),
                      "group_a_only": len(a_only), "group_b_only": len(b_only)}
    out["unit_coverage"] = {"group_a_only": len(a_only), "group_b_only": len(b_only), "both": len(shared)}
    out["design_rank"] = _unit_design_rank(included, group_col, unit_col, arm_a)
    out["n_observations_used"] = int(len(included))
    out["residual_df"] = int(len(included) - out["design_rank"])
    counts = included.groupby([unit_col, group_col]).size()
    all_units = sorted(set(included[unit_col]))
    full_index = pd.MultiIndex.from_product([all_units, [arm_a, arm_b]])
    counts = counts.reindex(full_index, fill_value=0)
    out["unit_by_arm_cells"] = {"total": int(len(counts)), "empty": int((counts == 0).sum()),
                                "single": int((counts == 1).sum())}
    out["unit_by_arm_cells"]["empty_fraction"] = round(float((counts == 0).mean()), 4)
    out["unit_arm_imbalance"] = {
        "units_in_both_arms": len(shared),
        "units_in_one_arm": len(a_only) + len(b_only),
        "fraction_single_arm": round((len(a_only) + len(b_only)) / max(1, len(units_a | units_b)), 4),
    }
    vif = _arm_vif(included, group_col, unit_col, arm_a)
    if not shared:
        design_class = "nested_within_arm"
    elif a_only or b_only:
        design_class = "partially_crossed"
    else:
        design_class = "crossed"
    out["design_class"] = design_class
    class_notes = {
        "nested_within_arm": t("unit_structure_nested"),
        "partially_crossed": t("unit_structure_partially_crossed"),
        "crossed": t("unit_structure_crossed"),
    }
    out["joint_model"] = {
        "model": "arm indicator + unit dummies (fixed effects)",
        "design_rank": out["design_rank"],
        "residual_df": out["residual_df"],
        "vif_arm": vif.get("vif_arm"),
        "r2_arm_from_units": vif.get("r2_arm_from_units"),
        "note": t("unit_note_rank"),
    }
    out["confounding"] = {
        "design_class": design_class,
        "treatment_and_unit_separable_in_joint_model": bool(out["residual_df"] > 0),
        "note": class_notes[design_class],
    }
    within = {
        "estimable": bool(len(shared) >= UNIT_MIN_PAIRED_UNITS),
        "n_shared_units": len(shared),
        "min_shared_units": int(UNIT_MIN_PAIRED_UNITS),
        "estimand": t("unit_estimand_within"),
        "reason": "",
    }
    if not within["estimable"]:
        within["reason"] = t("unit_reason_within_insufficient", len(shared), UNIT_MIN_PAIRED_UNITS)
    between = {
        "estimable": bool(len(a_only) >= UNIT_MIN_UNITS_PER_ARM and len(b_only) >= UNIT_MIN_UNITS_PER_ARM),
        "n_units_arm_a_only": len(a_only),
        "n_units_arm_b_only": len(b_only),
        "min_units_per_arm": int(UNIT_MIN_UNITS_PER_ARM),
        "excludes_shared_units": True,
        "estimand": t("unit_estimand_between"),
        "reason": "",
    }
    if not between["estimable"]:
        between["reason"] = t("unit_reason_between_insufficient", len(a_only), len(b_only),
                              UNIT_MIN_UNITS_PER_ARM)
    out["within_unit"] = within
    out["between_unit"] = between
    reasons: List[str] = list(out["reasons"])
    if unit_source == "column_name_match_unconfirmed":
        reasons.append(t("unit_reason_matched_by_name"))
    if unit_source == "analysis_design.pair_col_auto_inferred":
        reasons.append(t("unit_reason_auto_design"))
    if excluded:
        reasons.append(t("unit_reason_excluded_labels", len(excluded),
                         t("list_separator").join(sorted(excluded))))
    if within["estimable"]:
        out["estimability"] = "estimable_paired_within_unit"
        out["analysis_route"] = "within_unit_paired"
        out["estimand"] = within["estimand"]
        out["estimand_alternatives"] = {} if between["estimable"] else {"between_unit": between["reason"]}
    elif between["estimable"]:
        out["estimability"] = "estimable_between_unit"
        out["analysis_route"] = "between_unit_unpaired"
        out["estimand"] = between["estimand"]
        reasons.append(within["reason"] + t("unit_reason_switched_between", len(a_only), len(b_only)))
        out["estimand_alternatives"] = {}
    else:
        out["estimability"] = "not_estimable"
        out["analysis_route"] = "none"
        out["estimand"] = ""
        reasons.append(within["reason"])
        reasons.append(between["reason"])
        out["estimand_alternatives"] = {"within_unit": within["reason"], "between_unit": between["reason"]}
    if out["estimability"].startswith("estimable") and out["unit_arm_imbalance"]["units_in_one_arm"]:
        reasons.append(t("unit_reason_unbalanced_structure",
                         out["unit_arm_imbalance"]["units_in_one_arm"], len(units_a | units_b)))
    out["reasons"] = reasons
    return out

def _t_p_value(t_stat, df):
    t_stat = np.asarray(t_stat, dtype=float)
    df = np.asarray(df, dtype=float)
    p = np.full(t_stat.shape, np.nan)
    ok = np.isfinite(t_stat) & np.isfinite(df) & (df > 0)
    if ok.any():
        p[ok] = 2.0 * stats.t.sf(np.abs(t_stat[ok]), df[ok])
    return p


def _t_crit_975(df):
    df = np.asarray(df, dtype=float)
    out = np.full(df.shape, np.nan)
    ok = np.isfinite(df) & (df > 0)
    if ok.any():
        out[ok] = stats.t.ppf(0.975, df[ok])
    return out


def _paired_effect_rows(values: pd.DataFrame, min_pairs: int = UNIT_MIN_PAIRED_UNITS) -> Dict[str, np.ndarray]:
    """One-sample t test per protein over the per-unit differences.

    Reports the quantities a reader needs to check the row: number of effective pairs, degrees of
    freedom, standard error and the 95% interval. Rows below the minimum number of effective
    pairs are marked as not tested instead of receiving a P value.
    """
    data = values.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.sum(np.isfinite(data), axis=1).astype(float)
        mean = np.nanmean(data, axis=1)
        var = np.nanvar(data, axis=1, ddof=1)
        se = np.sqrt(np.divide(var, n, out=np.full_like(var, np.nan), where=n > 0))
        df = n - 1.0
        t_stat = np.divide(mean, se, out=np.full_like(mean, np.nan), where=se > 0)
        p_val = _t_p_value(t_stat, df)
        crit = _t_crit_975(df)
        ci_low = mean - crit * se
        ci_high = mean + crit * se
        tested = np.isfinite(p_val) & (n >= float(min_pairs))
        zero_spread = np.isfinite(se) & (se == 0)
        flat = zero_spread & np.isfinite(mean) & (np.abs(mean) < 1e-12)
        p_val = np.where(flat, 1.0, p_val)
        t_stat = np.where(flat, 0.0, t_stat)
        ci_low = np.where(flat, mean, ci_low)
        ci_high = np.where(flat, mean, ci_high)
        tested = np.where(flat & (n >= float(min_pairs)), True, tested)
    return {"effect": np.asarray(mean, dtype=float), "t": t_stat, "p": p_val, "n": n, "df": df,
            "se": se, "ci_low": ci_low, "ci_high": ci_high, "tested": tested,
            "degenerate": zero_spread & ~flat}


def _welch_effect_rows(a: pd.DataFrame, b: pd.DataFrame,
                       min_units: int = UNIT_MIN_UNITS_PER_ARM) -> Dict[str, np.ndarray]:
    """Welch t test per protein between two frames of per-unit means.

    The two frames must carry distinct units: a unit observed in both arms belongs to the
    within-unit estimand and is never passed in here.
    """
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        na = np.sum(np.isfinite(av), axis=1).astype(float)
        nb = np.sum(np.isfinite(bv), axis=1).astype(float)
        mean_a = np.nanmean(av, axis=1)
        mean_b = np.nanmean(bv, axis=1)
        var_a = np.nanvar(av, axis=1, ddof=1)
        var_b = np.nanvar(bv, axis=1, ddof=1)
        se2 = (np.divide(var_a, na, out=np.full_like(var_a, np.nan), where=na > 0)
               + np.divide(var_b, nb, out=np.full_like(var_b, np.nan), where=nb > 0))
        se = np.sqrt(se2)
        effect = mean_a - mean_b
        t_stat = np.divide(effect, se, out=np.full_like(effect, np.nan), where=se > 0)
        denom = (np.divide((var_a / na) ** 2, na - 1.0, out=np.full_like(var_a, np.nan), where=na > 1.0)
                 + np.divide((var_b / nb) ** 2, nb - 1.0, out=np.full_like(var_b, np.nan), where=nb > 1.0))
        df = np.divide(se2 ** 2, denom, out=np.full_like(se2, np.nan), where=denom > 0)
        p_val = _t_p_value(t_stat, df)
        crit = _t_crit_975(df)
        ci_low = effect - crit * se
        ci_high = effect + crit * se
        tested = np.isfinite(p_val) & (na >= float(min_units)) & (nb >= float(min_units))
        zero_spread = np.isfinite(se) & (se == 0)
        flat = zero_spread & np.isfinite(effect) & (np.abs(effect) < 1e-12)
        p_val = np.where(flat, 1.0, p_val)
        t_stat = np.where(flat, 0.0, t_stat)
        tested = np.where(flat & (na >= float(min_units)) & (nb >= float(min_units)), True, tested)
    return {"effect": np.asarray(effect, dtype=float), "t": t_stat, "p": p_val, "n_a": na, "n_b": nb,
            "df": df, "se": se, "ci_low": ci_low, "ci_high": ci_high, "tested": tested,
            "degenerate": zero_spread & ~flat}


def _unit_arm_mean_matrix(values: pd.DataFrame, column_keys: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """protein x (unit, arm) matrix of per-unit means."""
    renamed = values.copy()
    renamed.columns = pd.MultiIndex.from_tuples([column_keys[c] for c in renamed.columns], names=["unit", "arm"])
    return renamed.T.groupby(level=["unit", "arm"]).mean().T


def unit_aware_sensitivity(matrix: pd.DataFrame, sampleinfo: pd.DataFrame, design: Dict[str, Any],
                           group_col: str, arm_a: str, arm_b: str, contrast: str,
                           sample_col: str = "", protein_col: str = "PG.ProteinGroups",
                           gene_map: Optional[Dict[str, str]] = None, p_thresh: float = 0.05,
                           logfc_thresh: Optional[float] = None,
                           effect_scale: Optional[Dict[str, Any]] = None,
                           out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Aggregate to the experimental unit before testing, then report what changed.

    The observation-level contrast in differential_<contrast>.csv stays the reference analysis.
    This function answers a different question - is the arm difference still present when every
    independent biological unit contributes one value - keeps its own estimand, uncertainty and
    multiple-testing scope, and returns the blocking reason when the design cannot support it.
    No adjusted number is invented, and unit labels that cannot stand for independent individuals
    are excluded by an a priori label rule rather than by how the test turns out.
    """
    scale = dict(effect_scale) if isinstance(effect_scale, dict) else effect_scale_record(design)
    diagnostics = unit_structure_diagnostics(sampleinfo, design, group_col, arm_a, arm_b)
    result: Dict[str, Any] = {
        "contrast": contrast,
        "direction": diagnostics.get("direction", ""),
        "role": "sensitivity_analysis",
        "reference_analysis": t("unit_reference_analysis", contrast),
        "reference_result_file": "processed_proteins/differential_%s.csv" % contrast,
        "backend": "python_fallback_scipy",
        "status": "not_estimable",
        "estimability": diagnostics.get("estimability", "not_estimable"),
        "analysis_route": diagnostics.get("analysis_route", "none"),
        "reasons": list(diagnostics.get("reasons", [])),
        "unit_column": diagnostics.get("unit_column", ""),
        "unit_column_source": diagnostics.get("unit_column_source", "absent"),
        "unit_classes": diagnostics.get("unit_classes", {}),
        "unit_exclusion": diagnostics.get("unit_exclusion", {}),
        "n_observations": diagnostics.get("n_observations", {}),
        "n_units": diagnostics.get("n_units", {}),
        "n_units_all": diagnostics.get("n_units_all", {}),
        "unit_coverage": diagnostics.get("unit_coverage", {}),
        "design_class": diagnostics.get("design_class", ""),
        "design_rank": diagnostics.get("design_rank"),
        "residual_df": diagnostics.get("residual_df"),
        "joint_model": diagnostics.get("joint_model", {}),
        "confounding": diagnostics.get("confounding", {}),
        "within_unit": diagnostics.get("within_unit", {}),
        "between_unit": diagnostics.get("between_unit", {}),
        "unit_by_arm_cells": diagnostics.get("unit_by_arm_cells", {}),
        "unit_arm_imbalance": diagnostics.get("unit_arm_imbalance", {}),
        "test": "",
        "estimand": "",
        "n_units_tested": {},
        "n_proteins_tested": 0,
        "n_proteins_not_tested": 0,
        "n_passing_fdr": 0,
        "n_passing_fdr_only": 0,
        "n_passing_joint": None,
        "fdr_threshold": float(p_thresh),
        "logfc_threshold": None if logfc_thresh is None else float(logfc_thresh),
        "correction_scope": "",
        "filter_rule": "",
        "effect_scale": scale,
        "effect_column": _effect_name(scale),
        "results_file": None,
        "limitations": [
            "Unit-level aggregation changes the estimand; it is reported as a sensitivity analysis next to the observation-level result, not as a replacement.",
            "Few units per arm keep this analysis low powered; a non-significant unit-level result is not evidence that the observation-level effect is absent.",
            "Labels that name a mixture, a pool or an unassigned value are excluded from the unit-level estimand; they are not independent biological individuals.",
        ],
    }
    if str(result["estimability"]) == "not_estimable":
        return result
    if not sample_col or sample_col not in sampleinfo.columns:
        sample_col = "FileName" if "FileName" in sampleinfo.columns else ""
    if not sample_col or protein_col not in matrix.columns:
        result["status"] = "skipped"
        result["reasons"] = list(result["reasons"]) + ["sample or protein identifier column is unavailable"]
        return result
    unit_col = str(result["unit_column"])
    exclusion = result.get("unit_exclusion") or {}
    excluded = set(exclusion.get("pooled_source") or []) | set(exclusion.get("unconfirmed_source") or [])
    sub = sampleinfo.loc[sampleinfo[group_col].astype(str).isin([arm_a, arm_b])].copy()
    sub = sub[sub[sample_col].notna()].copy()
    sub["_sid"] = sub[sample_col].astype(str)
    sub[unit_col] = sub[unit_col].astype(str).str.strip()
    matrix_cols = {str(c) for c in matrix.columns}
    sub = sub[sub["_sid"].isin(matrix_cols)]
    sub = sub[~sub[unit_col].isin(["", "nan", "None", "NaN"])]
    sub = sub[~sub[unit_col].isin(excluded)]
    if sub.empty:
        result["status"] = "skipped"
        result["reasons"] = list(result["reasons"]) + ["no analysed matrix column matched an independent experimental unit"]
        return result
    keep_cols = [c for c in matrix.columns if str(c) in set(sub["_sid"])]
    values = matrix.loc[:, [protein_col] + keep_cols].copy()
    values[keep_cols] = values[keep_cols].apply(pd.to_numeric, errors="coerce")
    column_keys: Dict[str, Tuple[str, str]] = {}
    for _, row in sub.drop_duplicates("_sid").iterrows():
        column_keys[str(row["_sid"])] = (str(row[unit_col]), str(row[group_col]).strip())
    values = values.loc[:, [protein_col] + [c for c in keep_cols if str(c) in column_keys]]
    if values.shape[1] < 2:
        result["status"] = "skipped"
        result["reasons"] = list(result["reasons"]) + ["fewer than two analysed columns carry an experimental unit"]
        return result
    per_unit = _unit_arm_mean_matrix(values.set_index(protein_col), column_keys)
    key_set = set(per_unit.columns)
    # R25: the effective number of experimental units has to be re-checked after the matrix
    # matching and after the per-protein filtering, not only on the declared metadata.
    eff_a = {u for u, arm in key_set if arm == arm_a}
    eff_b = {u for u, arm in key_set if arm == arm_b}
    eff_shared = eff_a & eff_b
    declared = dict(result.get("n_units") or {})
    result["n_units_declared"] = declared
    result["estimability_declared"] = result["estimability"]
    result["n_units"] = {"group_a": len(eff_a), "group_b": len(eff_b), "shared": len(eff_shared),
                         "group_a_only": len(eff_a - eff_b), "group_b_only": len(eff_b - eff_a)}
    result["unit_coverage"] = {"group_a_only": len(eff_a - eff_b), "group_b_only": len(eff_b - eff_a),
                               "both": len(eff_shared)}
    result["units_effective"] = {"group_a": sorted(eff_a), "group_b": sorted(eff_b),
                                 "shared": sorted(eff_shared)}
    if declared and (declared.get("shared") != len(eff_shared)
                     or declared.get("group_a") != len(eff_a)
                     or declared.get("group_b") != len(eff_b)):
        result["reasons"] = list(result["reasons"]) + [
            t("unit_reason_matrix_matched_units_unresolved",
              declared.get("group_a"), declared.get("group_b"), declared.get("shared"),
              len(eff_a), len(eff_b), len(eff_shared))]
    units_a = sorted({u for u, arm in key_set if arm == arm_a})
    units_b = sorted({u for u, arm in key_set if arm == arm_b})
    shared = [u for u in units_a if u in set(units_b)]
    route = str(result["analysis_route"])
    a_only = [u for u in units_a if u not in set(units_b)]
    b_only = [u for u in units_b if u not in set(units_a)]
    n_rows = len(per_unit.index)
    not_tested_reason = np.full(n_rows, "", dtype=object)
    if route == "within_unit_paired":
        keep = [u for u in shared if (u, arm_a) in key_set and (u, arm_b) in key_set]
        if len(keep) < UNIT_MIN_PAIRED_UNITS:
            result["status"] = "not_estimable"
            result["estimability"] = "not_estimable"
            result["analysis_route"] = "none"
            result["reasons"] = list(result["reasons"]) + [
                "fewer than %d units keep a value in both arms after missing-value filtering" % UNIT_MIN_PAIRED_UNITS]
            return result
        a_frame = pd.DataFrame(per_unit.loc[:, [(u, arm_a) for u in keep]].to_numpy(dtype=float), index=per_unit.index)
        b_frame = pd.DataFrame(per_unit.loc[:, [(u, arm_b) for u in keep]].to_numpy(dtype=float), index=per_unit.index)
        rows = _paired_effect_rows(a_frame - b_frame, min_pairs=UNIT_MIN_PAIRED_UNITS)
        test_label = "paired t-test on per-unit means"
        estimand = "mean within-unit difference (%s - %s)" % (arm_a, arm_b)
        result["n_pairs"] = len(keep)
        result["units_used"] = {"paired": list(keep)}
        n_a = np.full(n_rows, float(len(keep)))
        n_b = np.full(n_rows, float(len(keep)))
        not_tested_reason = np.where(
            ~np.asarray(rows["tested"], dtype=bool),
            np.where(np.asarray(rows["degenerate"], dtype=bool),
                     "all paired differences identical and non-zero: the paired t-test is undefined",
                     "fewer than %d effective pairs after per-protein missing filtering" % UNIT_MIN_PAIRED_UNITS),
            "")
    elif route == "between_unit_unpaired":
        a_cols = [(u, arm_a) for u in a_only]
        b_cols = [(u, arm_b) for u in b_only]
        if min(len(a_cols), len(b_cols)) < UNIT_MIN_UNITS_PER_ARM:
            result["status"] = "not_estimable"
            result["estimability"] = "not_estimable"
            result["analysis_route"] = "none"
            result["reasons"] = list(result["reasons"]) + [
                "fewer than %d arm-exclusive units per arm after missing-value filtering" % UNIT_MIN_UNITS_PER_ARM]
            return result
        a_frame = pd.DataFrame(per_unit.loc[:, a_cols].to_numpy(dtype=float), index=per_unit.index)
        b_frame = pd.DataFrame(per_unit.loc[:, b_cols].to_numpy(dtype=float), index=per_unit.index)
        rows = _welch_effect_rows(a_frame, b_frame, min_units=UNIT_MIN_UNITS_PER_ARM)
        test_label = "Welch t-test on per-unit means (arm-exclusive independent units)"
        estimand = "difference in per-unit means (%s - %s)" % (arm_a, arm_b)
        result["n_pairs"] = 0
        result["units_used"] = {"group_a_only": list(a_only), "group_b_only": list(b_only),
                                "shared_units_excluded": list(shared)}
        n_a = np.asarray(rows["n_a"], dtype=float)
        n_b = np.asarray(rows["n_b"], dtype=float)
        not_tested_reason = np.where(
            ~np.asarray(rows["tested"], dtype=bool),
            np.where(np.asarray(rows["degenerate"], dtype=bool),
                     "all per-unit means identical and non-zero: the Welch test is undefined",
                     "fewer than %d effective units per arm after per-protein missing filtering" % UNIT_MIN_UNITS_PER_ARM),
            "")
    else:  # pragma: no cover - guarded by the estimability check above
        result["status"] = "not_estimable"
        return result
    tested_mask = np.asarray(rows["tested"], dtype=bool)
    p_all = np.asarray(rows["p"], dtype=float)
    adj = np.full(p_all.shape, np.nan)
    if tested_mask.any():
        adj[tested_mask] = bh_adjust(p_all[tested_mask])
    effect_column = _effect_name(scale)
    protein_ids = [str(v) for v in per_unit.index]
    lookup = gene_map or {}
    table = pd.DataFrame({
        "protein_group": protein_ids,
        "gene": [str(lookup.get(pid, "")) for pid in protein_ids],
        "contrast": contrast,
        "direction": "%s - %s" % (arm_a, arm_b),
        "analysis_unit": unit_col,
        "unit_scope": route,
        "test": test_label,
        "estimand": estimand,
        effect_column: np.asarray(rows["effect"], dtype=float),
        "effect_scale_state": scale.get("state"),
        "effect_scale_label": scale.get("label"),
        "t": np.asarray(rows["t"], dtype=float),
        "df": np.asarray(rows["df"], dtype=float),
        "se": np.asarray(rows["se"], dtype=float),
        "ci_low": np.asarray(rows["ci_low"], dtype=float),
        "ci_high": np.asarray(rows["ci_high"], dtype=float),
        "P.Value": p_all,
        "adj.P.Val": adj,
        "n_units_used": (np.asarray(rows["n"], dtype=float) if route == "within_unit_paired" else n_a + n_b),
        "n_units_group_a": n_a,
        "n_units_group_b": n_b,
        "tested": tested_mask,
        "not_tested_reason": list(not_tested_reason),
    })
    tested = table.loc[table["tested"]]
    result["n_proteins_tested"] = int(tested.shape[0])
    result["n_proteins_not_tested"] = int(table.shape[0] - tested.shape[0])
    fdr_only = int((tested["adj.P.Val"] < float(p_thresh)).sum())
    result["n_passing_fdr"] = fdr_only
    result["n_passing_fdr_only"] = fdr_only
    if logfc_thresh is None:
        result["n_passing_joint"] = None
        result["filter_rule"] = t("unit_filter_rule_fdr_only", p_thresh,
                                  result["n_proteins_tested"])
    else:
        joint = int(((tested["adj.P.Val"] < float(p_thresh))
                     & (np.abs(tested[effect_column]) > float(logfc_thresh))).sum())
        result["n_passing_joint"] = joint
        result["filter_rule"] = t("unit_filter_rule_joint", p_thresh, effect_column, logfc_thresh,
                                  scale.get("state"))
    result["correction_scope"] = t("unit_correction_scope", result["n_proteins_tested"])
    result["n_units_tested"] = {"group_a": len(units_a), "group_b": len(units_b),
                                "shared": len(shared),
                                "group_a_only": len(a_only), "group_b_only": len(b_only),
                                "paired": int(result.get("n_pairs") or 0)}
    result["test"] = test_label
    result["estimand"] = estimand
    result["status"] = "completed" if result["n_proteins_tested"] else "skipped"
    result["min_p_value"] = float(tested["P.Value"].min()) if result["n_proteins_tested"] else None
    result["min_adj_p_value"] = float(tested["adj.P.Val"].min()) if result["n_proteins_tested"] else None
    result["row_level_note"] = t("unit_row_level_note")
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(contrast))
        path = out_dir / ("unit_sensitivity_%s.csv" % safe)
        table.to_csv(path, index=False)
        result["results_file"] = str(path)
    return result


def unit_sensitivity_for_run(run_folder: Path, out_dir: Optional[Path] = None) -> Dict[str, str]:
    """Unit-level sensitivity analysis for every declared contrast of one run."""
    run_folder = Path(run_folder)
    params = _params(run_folder)
    design = params.get("analysis_design", {}) if isinstance(params.get("analysis_design"), dict) else {}
    if not design:
        fallback = run_folder / "analysis_design.used.yaml"
        if fallback.exists():
            try:
                import yaml
                loaded = yaml.safe_load(fallback.read_text(encoding="utf-8"))
                design = loaded if isinstance(loaded, dict) else {}
            except Exception:
                design = {}
    sampleinfo_path = _resolve(params.get("sampleinfo_path", ""), run_folder)
    files = _matrix_files(run_folder, params)
    matrix_path = files["analysed"] or files["filtered"]
    contrasts = (design.get("differential") or {}).get("contrasts") or []
    if not design or sampleinfo_path is None or matrix_path is None or not contrasts:
        return {}
    group_col = str(design.get("group_col") or "")
    if not group_col:
        return {}
    matrix = _read(matrix_path)
    sampleinfo = _read(sampleinfo_path)
    protein_col = _col(matrix, PROTEIN_COL_CANDIDATES)
    gene_col = _col(matrix, GENE_COL_CANDIDATES)
    if not protein_col:
        return {}
    sample_col = str(design.get("sample_id_col") or "")
    if not sample_col or sample_col not in sampleinfo.columns:
        sample_col = "FileName" if "FileName" in sampleinfo.columns else _col(sampleinfo, ["Sample", "SampleID", "sample"])
    gene_map: Dict[str, str] = {}
    if gene_col:
        gene_map = dict(zip(matrix[protein_col].astype(str), _symbol(matrix[gene_col]).astype(str)))
    diff_cfg = design.get("differential") or {}
    p_thresh = float(diff_cfg.get("p_thresh") or 0.05)
    logfc_thresh = diff_cfg.get("logfc_thresh")
    logfc_thresh = None if logfc_thresh is None else float(logfc_thresh)
    scale = effect_scale_record(design, run_folder)
    target = Path(out_dir) if out_dir else (run_folder / "evaluation_evidence_ext")
    target.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for spec in contrasts:
        if not isinstance(spec, dict):
            continue
        arm_a = str(spec.get("group_a") or "").strip()
        arm_b = str(spec.get("group_b") or "").strip()
        contrast = str(spec.get("name") or "%s_vs_%s" % (arm_a, arm_b))
        if not arm_a or not arm_b:
            continue
        records.append(unit_aware_sensitivity(
            matrix, sampleinfo, design, group_col, arm_a, arm_b, contrast,
            sample_col=sample_col, protein_col=protein_col, gene_map=gene_map,
            p_thresh=p_thresh, logfc_thresh=logfc_thresh, effect_scale=scale, out_dir=target))
    if not records:
        return {}
    flat_rows = []
    for rec in records:
        flat = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
        flat["n_units_a"] = (rec.get("n_units") or {}).get("group_a")
        flat["n_units_b"] = (rec.get("n_units") or {}).get("group_b")
        flat["n_units_shared"] = (rec.get("n_units") or {}).get("shared")
        flat["n_units_a_only"] = (rec.get("n_units") or {}).get("group_a_only")
        flat["n_units_b_only"] = (rec.get("n_units") or {}).get("group_b_only")
        exclusion = rec.get("unit_exclusion") or {}
        flat["n_units_excluded"] = exclusion.get("n_excluded")
        flat["excluded_unit_labels"] = "|".join(sorted(set(exclusion.get("pooled_source") or [])
                                                       | set(exclusion.get("unconfirmed_source") or [])))
        flat_rows.append(flat)
    summary = pd.DataFrame(flat_rows)
    summary_path = target / "unit_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    json_path = target / "unit_structure.json"
    json_path.write_text(json.dumps({r.get("contrast", ""): r for r in records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"unit_sensitivity_summary": str(summary_path), "unit_structure": str(json_path)}

def run_all(run_folder: Path, min_n: int = 5, out_dir: Optional[Path] = None) -> Dict[str, str]:
    warnings.filterwarnings("ignore")
    run_folder = Path(run_folder)
    out = Path(out_dir) if out_dir else (run_folder / "evaluation_evidence_ext")
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    # R27: record what was written and, when something was not written, why. The earlier
    # version skipped the batch-level tables silently, so a table that had not been produced
    # yet was indistinguishable from one that was missing, empty, not executed or failed.
    processed = run_folder / "processed_proteins"
    params = _params(run_folder)
    design = params.get("analysis_design") or {}
    files = _matrix_files(run_folder, params)
    status: Dict[str, object] = {
        "kind": "analysis_extensions_status",
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "inputs": {
            "processed_proteins_exists": processed.exists(),
            "processed_proteins_csv_count": len(list(processed.glob("*.csv"))) if processed.exists() else 0,
            "differential_tables": len(list(processed.glob("differential_*.csv"))) if processed.exists() else 0,
            "analysis_matrix_resolved": bool(files.get("analysed") or files.get("filtered")),
            "sampleinfo_resolved": bool(_resolve(params.get("sampleinfo_path", ""), run_folder)),
            "analysis_design_present": bool(design),
            "group_col_declared": bool(str(design.get("group_col") or "")),
            "batch_col_declared": bool(str(design.get("batch_col") or "")),
        },
        "artifacts": {},
        "errors": [],
    }

    def _mark(name: str, state: str, reason: str = "") -> None:
        status["artifacts"][name] = {"status": state, "reason": reason[:400], "path": str(out / name)}
        if state == "error":
            status["errors"].append({"artifact": name, "status": state, "reason": reason[:400]})

    def _matrix_ready() -> bool:
        return bool(files.get("analysed") or files.get("filtered"))

    try:
        trace = trace_candidate_exclusion(run_folder)
        if not trace.empty:
            path = out / "candidate_exclusion_trace.csv"
            trace.to_csv(path, index=False)
            written["candidate_exclusion_trace"] = str(path)
            _mark("candidate_exclusion_trace.csv", "written", "%d rows" % trace.shape[0])
        else:
            _mark("candidate_exclusion_trace.csv", "empty_result", "no candidate rows in this run")
    except Exception as exc:  # noqa: BLE001
        _mark("candidate_exclusion_trace.csv", "error", str(exc))

    try:
        conf, verdict = estimate_confounding(run_folder, out_dir=out)
        if not conf.empty:
            path = out / "confounding_estimability.csv"
            conf.to_csv(path, index=False)
            written["confounding_estimability"] = str(path)
            _mark("confounding_estimability.csv", "written", "%d rows" % conf.shape[0])
        else:
            _mark("confounding_estimability.csv", "empty_result", verdict)
        path = out / "estimability_verdict.txt"
        path.write_text(verdict + chr(10), encoding="utf-8")
        written["estimability_verdict"] = str(path)
        _mark("estimability_verdict.txt", "written", verdict)
    except Exception as exc:  # noqa: BLE001
        _mark("estimability_verdict.txt", "error", str(exc))

    try:
        strat = stratified_contrasts(run_folder, min_n=min_n, out_dir=out)
        if not strat.empty:
            path = out / "stratified_contrast_summary.csv"
            strat.to_csv(path, index=False)
            written["stratified_contrast_summary"] = str(path)
            _mark("stratified_contrast_summary.csv", "written", "%d rows" % strat.shape[0])
        else:
            if not _matrix_ready():
                _mark("stratified_contrast_summary.csv", "not_generated_yet",
                      "analysis matrix not resolved yet; re-run after the analysis phase")
            else:
                _mark("stratified_contrast_summary.csv", "empty_result", "no stratified rows for the requested design")
    except Exception as exc:  # noqa: BLE001
        _mark("stratified_contrast_summary.csv", "error", str(exc))

    try:
        conc = contrast_concordance(run_folder)
        if not conc.empty:
            path = out / "contrast_concordance.csv"
            conc.to_csv(path, index=False)
            written["contrast_concordance"] = str(path)
            _mark("contrast_concordance.csv", "written", "%d rows" % conc.shape[0])
        else:
            _mark("contrast_concordance.csv", "empty_result", "no differential tables to reconcile")
    except Exception as exc:  # noqa: BLE001
        _mark("contrast_concordance.csv", "error", str(exc))

    try:
        units = unit_sensitivity_for_run(run_folder, out_dir=out)
        written.update(units)
        _mark("unit_sensitivity_summary.csv", "written", "%s" % units.get("unit_sensitivity_summary", ""))
    except Exception as exc:  # noqa: BLE001
        _mark("unit_sensitivity_summary.csv", "error", str(exc))

    try:
        dose = dose_trends(run_folder, out_dir=out)
        if not dose.empty:
            written["dose_trend"] = str(out / "dose_trend.csv")
            _mark("dose_trend.csv", "written", "%d rows" % dose.shape[0])
        else:
            _mark("dose_trend.csv", "not_applicable", "no dose gradient in this design")
    except Exception as exc:  # noqa: BLE001
        _mark("dose_trend.csv", "error", str(exc))

    try:
        ora = enrichment_ora(run_folder, out_dir=out)
        if not ora.empty:
            written["enrichment_ora"] = str(out / "enrichment_ora.csv")
            _mark("enrichment_ora.csv", "written", "%d rows" % ora.shape[0])
        else:
            _mark("enrichment_ora.csv", "empty_result", "no over-representation rows")
    except Exception as exc:  # noqa: BLE001
        _mark("enrichment_ora.csv", "error", str(exc))

    try:
        status_path = out / "analysis_extensions_status.json"
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        written["analysis_extensions_status"] = str(status_path)
    except Exception:  # noqa: BLE001
        pass
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    written = run_all(Path(args.run_folder), min_n=args.min_n, out_dir=Path(args.out_dir) if args.out_dir else None)
    for key, value in written.items():
        print("%-30s %s" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
