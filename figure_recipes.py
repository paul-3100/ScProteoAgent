# -*- coding: utf-8 -*-
r"""Per-dataset figure recipe declarations (figure_upgrade_20260904 Step 05).

This module upgrades plotting from "one generic figure set for every dataset"
to "generic figure set + per-dataset must-have recipe figures". Each recipe
declares one dataset-specific figure that either reproduces a core analytical
form of the original paper (``original_equivalent``) or extends it inside the
current matrix boundary (``extension``). The declaration layer is aligned with
``evaluation_evidence.CORE_STORY_RECIPES`` (same 11 main datasets, same
primary contrasts / focus genes / curated modules) without changing any
evaluation semantics.

Design contract
---------------
- ``plot_function`` names a function that Step 06 implements. The dispatcher in
  ``tools.py`` (``_render_dataset_figure_recipes``) resolves the name first from
  ``tools._FIGURE_RECIPE_FUNCTION_REGISTRY`` (register via
  ``tools.register_figure_recipe_function``), then from the ``tools`` module
  globals. Missing functions are recorded as ``pending_function`` in the recipe
  manifest and never break a run (backward compatible).
- ``input_tables`` use run-artifact naming conventions relative to the run
  folder (see ``INPUT_TABLE_PATTERNS``). ``SampleInfo.csv`` is special-cased by
  the dispatcher and resolved through ``resolve_path('sampleinfo_path')``.
  Missing tables are reported per recipe; a recipe whose declared inputs are
  all absent is skipped as ``missing_inputs`` (no placeholder image, Step 04
  rule).
- Plot-function calling contract (Step 06)::

      info = plot_function(
          this_run_folder,
          data_BC=data_BC, sampleinfo=sampleinfo,
          protein_gene_map=protein_gene_map,
          input_paths={table: abs_path_or_empty, ...},
          params={...recipe params...},
      )

  ``info`` follows the standard plot-function return dict (``file_path`` plus
  optional caption metadata); an empty ``file_path`` marks an omitted figure.

- ``title`` and ``description`` are not literals here: they resolve from the shared
  report-language registry (``recipes`` namespace) at access time, so a zh run
  renders the released wording and an en run the English wording.

This module is intentionally standard-library only so that ``tools.py`` can
import it without extra dependencies.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple

from report_language import has_template, register, t

SCHEMA_VERSION = "figure_recipes/1"

#: Legal figure kinds.
FIGURE_KINDS: Tuple[str, ...] = ("original_equivalent", "extension")

#: The 11 main benchmark datasets (must stay aligned with the V3 main set).
MAIN_DATASETS: Tuple[str, ...] = (
    "Nat_Commun_PiSPA_2024",
    "Nat_Methods_iPSC_2025",
    "Nat_Biotech_Brain_2026",
    "Nat_Commun_Carr_2024",
    "Nat_Methods_pSCoPE_2023",
    "Nat_Methods_DVP_2023",
    "Nat_Commun_SCPro_2024",
    "Nat_Commun_Nociceptor_2026",
    "Science_BloodCell_2025",
    "Cell_TurnoverDynamics_2025",
    "Nat_Commun_ProteinLeakage_2025",
)

# ---------------------------------------------------------------------------
# Run artifact naming conventions for recipe input tables
# ---------------------------------------------------------------------------

#: (kind, regex) pairs; a recipe input table must match exactly one pattern.
INPUT_TABLE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # Raw dataset sample sheet (dispatcher resolves via sampleinfo_path).
    ("sampleinfo", r"^SampleInfo\.csv$"),
    # processed_proteins/
    ("processed", r"^processed_proteins/(ProteinQuant_ComBat|ProteinQuant_Filtered|ProQuant_Normalized|Protein_Gene_Map|SampleInfo_Filtered|sampleinfo_with_clusters)\.csv$"),
    ("processed", r"^processed_proteins/limma_summary\.json$"),
    ("differential", r"^processed_proteins/differential_[A-Za-z0-9._-]+_vs_[A-Za-z0-9._-]+\.csv$"),
    ("differential", r"^processed_proteins/[A-Za-z0-9._-]+_vs_[A-Za-z0-9._-]+/[A-Za-z0-9._-]+_(up|down)\.csv$"),
    # enrichment_results/
    ("enrichment", r"^enrichment_results/(GO|KEGG|Reactome)_[A-Za-z0-9._-]+_vs_[A-Za-z0-9._-]+_(upregulated|downregulated)_proteins\.csv$"),
    ("enrichment", r"^enrichment_results/go_kegg_reactome_results\.json$"),
    # evaluation_evidence/
    ("evidence", r"^evaluation_evidence/group_composition_qc\.(csv|json)$"),
    ("evidence", r"^evaluation_evidence/group_cross_table_[A-Za-z0-9._-]+_by_[A-Za-z0-9._-]+\.csv$"),
    ("evidence", r"^evaluation_evidence/curated_module_(group_summary|sample_scores)\.csv$"),
    ("evidence", r"^evaluation_evidence/curated_module_scores\.json$"),
    ("evidence", r"^evaluation_evidence/candidate_protein_evidence\.(csv|json)$"),
    ("evidence", r"^evaluation_evidence/core_story_evidence\.(csv|json)$"),
    ("evidence", r"^evaluation_evidence/report_story_outline\.json$"),
    # proteins_leakage/
    ("leakage", r"^proteins_leakage/leakage_(summary\.json|module_summary\.csv)$"),
)

_INPUT_TABLE_COMPILED: Tuple[Tuple[str, Any], ...] = tuple(
    (kind, re.compile(pattern)) for kind, pattern in INPUT_TABLE_PATTERNS
)


def match_input_table_kind(table: str) -> str | None:
    """Return the artifact kind of ``table``, or ``None`` if it breaks convention."""
    text = str(table or "").strip().replace("\\", "/")
    for kind, pattern in _INPUT_TABLE_COMPILED:
        if pattern.match(text):
            return kind
    return None


def input_table_matches_convention(table: str) -> bool:
    return match_input_table_kind(table) is not None


# ---------------------------------------------------------------------------

# Reader-facing recipe text (zh released verbatim, en added)

# ---------------------------------------------------------------------------



# Titles and descriptions are keyed by recipe id: ``<recipe_id>_title`` and

# ``<recipe_id>_description``. The zh column repeats the released literals character

# for character; ``FigureRecipe.title`` / ``.description`` resolve the text at access

# time, so a run renders the language that is active when the manifest is written.

_RECIPE_TEXT = {

    "pispa_cluster_type_composition_title": (
        "Cluster×迁移组成富集图",
        "Cluster by migration-state composition",
    ),
    "pispa_cluster_type_composition_description": (
        "按 Cluster 1/2/3 展示迁移/对照细胞组成与迁移占比，使 Cluster 1 迁移细胞富集可视化。",
        "Composition of migrated and control cells and the migrated fraction across Clusters 1/2/3, making the enrichment of migrated cells in Cluster 1 visible.",
    ),
    "pispa_rho_gtpase_candidate_boxplot_title": (
        "Rho GTPase-细胞骨架-黏附候选蛋白分组箱线图",
        "Grouped boxplots of Rho GTPase, cytoskeleton and adhesion candidate proteins",
    ),
    "pispa_rho_gtpase_candidate_boxplot_description": (
        "Cdc42/Rac1/RhoA 及 ERM/黏附候选在 Cluster 1/2/3 的丰度箱线图，标注 logFC/FDR 与迁移亚步骤。",
        "Abundance boxplots of Cdc42/Rac1/RhoA and ERM/adhesion candidates across Clusters 1/2/3, annotated with logFC/FDR and the migration sub-step.",
    ),
    "pispa_migration_module_gradient_title": (
        "迁移模块分数梯度图",
        "Gradient of migration module scores",
    ),
    "pispa_migration_module_gradient_description": (
        "迁移相关 curated 模块分数在 Cluster 1/2/3 的组均值与样本分布梯度，扩展 Cluster×迁移机制链证据。",
        "Group means and sample distributions of migration-related curated module scores across Clusters 1/2/3, extending the Cluster x migration mechanism chain.",
    ),
    "brain_state_axis_module_gradient_title": (
        "RG→oRG→IPC-EN→EN 状态轴 marker/模块梯度图",
        "Marker and module gradients along the RG to oRG to IPC-EN to EN state axis",
    ),
    "brain_state_axis_module_gradient_description": (
        "沿发育状态轴展示 stage marker 与神经成熟/突触/染色质模块分数梯度；ASD/NDD 保持 external_annotation 边界。",
        "Stage markers and neuronal maturation, synapse and chromatin module scores along the developmental state axis; ASD/NDD annotations stay external_annotation.",
    ),
    "brain_marker_cluster_heatmap_title": (
        "细胞类型典型 marker×cluster 平均丰度热图",
        "Mean abundance of canonical cell-type markers by cluster",
    ),
    "brain_marker_cluster_heatmap_description": (
        "oRG/IPC-EN/EN/OPC/小胶质/血管等典型 marker 的 cluster 均值丰度热图，补足运行缺少的 marker 导向丰度图。",
        "Heatmap of cluster-mean abundance for canonical markers of oRG, IPC-EN, EN, OPC, microglia and vasculature, supplying the marker-oriented abundance view the run was missing.",
    ),
    "dvp_zonation_zone_axis_heatmap_title": (
        "Portal-Midlobular-Central 分区 z-score 热图与 marker 曲线",
        "Portal to Midlobular to Central zone z-score heatmap with marker curves",
    ),
    "dvp_zonation_zone_axis_heatmap_description": (
        "在可用的三区序数轴上重建 zonation：top 分区蛋白 z-score 热图、urea cycle/xenobiotic marker 梯度曲线。",
        "Zonation rebuilt on the available three-zone ordinal axis: a z-score heatmap of the top zone-specific proteins plus gradient curves for urea-cycle and xenobiotic markers.",
    ),
    "dvp_zonation_module_curves_title": (
        "门静脉-中央模块方向曲线图",
        "Directional module curves along the portal-to-central axis",
    ),
    "dvp_zonation_module_curves_description": (
        "periportal urea 与 central xenobiotic 模块分数沿区轴的组均值曲线，支持 OXPHOS/代谢分区的方向判读。",
        "Group-mean curves of periportal urea and central xenobiotic module scores along the zone axis, supporting directional reading of OXPHOS and metabolic zonation.",
    ),
    "bloodcell_hspc_state_hierarchy_dotplot_title": (
        "HSPC 状态层级 marker dotplot",
        "Marker dot plot across the HSPC state hierarchy",
    ),
    "bloodcell_hspc_state_hierarchy_dotplot_description": (
        "HSC→MPP→LMPP→GMP→MEP 状态层级下的干性/代谢与粒细胞颗粒 marker 平均丰度点图，统一 HSC_vs_GMP 方向。",
        "Mean abundance dot plot of stemness, metabolic and granulocyte-granule markers across the HSC to MPP to LMPP to GMP to MEP hierarchy, keeping the HSC_vs_GMP direction consistent.",
    ),
    "bloodcell_cluster_label_agreement_title": (
        "蛋白簇×FACS 标签一致性热图",
        "Agreement between protein clusters and FACS labels",
    ),
    "bloodcell_cluster_label_agreement_description": (
        "蛋白聚类与 FACS 标签的交叉表热图，暴露 MEP/GMP/LMPP 一致与错配，支撑状态解释边界。",
        "Cross-tabulation heatmap of protein clusters against FACS labels, exposing where MEP/GMP/LMPP agree and where they mismatch, which bounds the state interpretation.",
    ),
    "turnover_drug_dose_footprint_title": (
        "双药剂量丰度足迹对比图",
        "Abundance footprint of two drugs across doses",
    ),
    "turnover_drug_dose_footprint_description": (
        "Bortezomib 与 Cycloheximide 相对 Control 的总丰度与低/高剂量 log2FC 散点；只描述 abundance footprint，不推断真实 turnover rate。",
        "Total abundance and low/high-dose log2FC scatter for Bortezomib and Cycloheximide relative to Control; this describes the abundance footprint only and infers no true turnover rate.",
    ),
    "turnover_drug_module_direction_title": (
        "双药机制模块方向合成图",
        "Shared and opposing module directions under two drugs",
    ),
    "turnover_drug_module_direction_description": (
        "Bortezomib（蛋白酶体/蛋白稳态）与 Cycloheximide（翻译/核糖体）富集与模块方向的共有/相反变化合成视图。",
        "Combined view of shared and opposing changes in enrichment and module direction for Bortezomib (proteasome/protein homeostasis) and Cycloheximide (translation/ribosome).",
    ),
    "turnover_sc_lowinput_stratification_title": (
        "单细胞 vs 低输入分层结构图",
        "Stratification of single-cell versus low-input samples",
    ),
    "turnover_sc_lowinput_stratification_description": (
        "按处理着色并叠加 single_cell/10cell_pool 分层的 PCA/UMAP 面板，回答单细胞与低输入是否需分层解读。",
        "PCA/UMAP panels coloured by treatment with the single_cell/10cell_pool strata overlaid, addressing whether single-cell and low-input samples need separate interpretation.",
    ),
    "pscope_polarization_gradient_title": (
        "极化强度梯度功能集投影图",
        "Functional-set projection of the polarisation gradient",
    ),
    "pscope_polarization_gradient_description": (
        "以 IFN/吞噬体/V-ATPase 功能集中位丰度 z-score 重新着色 Untreated→LPS_low→LPS_high 的 PCA，展示极化强度梯度。",
        "PCA of Untreated to LPS_low to LPS_high recoloured by the median-abundance z-score of the IFN, phagosome and V-ATPase functional sets, showing a gradient of polarisation strength.",
    ),
    "pscope_vatpase_covariation_title": (
        "V-ATPase/吞噬体蛋白共变散点图",
        "Co-variation of V-ATPase and phagosome proteins",
    ),
    "pscope_vatpase_covariation_description": (
        "分条件展示 V-ATPase 与溶酶体模块蛋白的蛋白-蛋白共变散点，补足单细胞内功能模块共变证据。",
        "Protein-protein co-variation scatter of V-ATPase and lysosomal module proteins per condition, supplying within-cell evidence for functional module co-variation.",
    ),
    "proteinleakage_compartment_leakage_title": (
        "区室标记泄漏对比箱线图",
        "Compartment-marker leakage compared across conditions",
    ),
    "proteinleakage_compartment_leakage_description": (
        "cytosol/nucleus 与 mito/membrane 区室标记在 Intact vs Permeable 的 log2FC 箱线对比，量化区室选择性泄漏。",
        "log2FC boxplot comparison of cytosol/nucleus and mito/membrane markers between Intact and Permeable samples, quantifying compartment-selective leakage.",
    ),
    "proteinleakage_status_confounder_dotplot_title": (
        "细胞类型×保存状态泄漏占比点图",
        "Permeable fraction by cell type and preservation state",
    ),
    "proteinleakage_status_confounder_dotplot_description": (
        "按细胞类型与 Fresh/Frozen 保存状态展示 permeable 占比，暴露保存方式与细胞类型混杂边界。",
        "Permeable fraction shown by cell type and Fresh/Frozen preservation state, exposing where preservation method and cell type are confounded.",
    ),
    "proteinleakage_leakage_correlation_heatmap_title": (
        "跨细胞类型泄漏 fold-change 相关热图",
        "Cross-cell-type correlation of leakage fold changes",
    ),
    "proteinleakage_leakage_correlation_heatmap_description": (
        "复用泄漏分析的跨细胞类型 fold-change 相关矩阵绘制热图，支持可泛化的无染色泄漏 QC 证据。",
        "Heatmap of the cross-cell-type fold-change correlation matrix from the leakage analysis, supporting generalisable stain-free leakage QC evidence.",
    ),
    "scpro_klrg1_context_matrix_title": (
        "Treg/CD4/CD8 KLRG1 背景对比矩阵图",
        "KLRG1 contrast matrix across the Treg/CD4/CD8 contexts",
    ),
    "scpro_klrg1_context_matrix_description": (
        "KLRG1+ vs KLRG1- 在 Treg/CD4/CD8 三个背景下的方向矩阵，标出共享与发散方向，避免跨背景外推。",
        "Direction matrix for KLRG1+ versus KLRG1- across the Treg, CD4 and CD8 contexts, marking shared and divergent directions so that no context is extrapolated to another.",
    ),
    "scpro_treg_klrg1_enrichment_title": (
        "Treg KLRG1 对比富集气泡图",
        "Enrichment bubble plot for the Treg KLRG1 contrast",
    ),
    "scpro_treg_klrg1_enrichment_description": (
        "为最高权重的 Treg KLRG1+/- 对比补充 GO/KEGG/Reactome 富集读出（当前运行缺失该对比富集）。",
        "GO/KEGG/Reactome enrichment readout added for the highest-weight Treg KLRG1+/- contrast, which the run itself leaves without enrichment.",
    ),
    "scpro_treg_effector_group_means_title": (
        "Treg KLRG1+ 免疫抑制程序组均值图",
        "Group means of the immunosuppressive programme in Treg KLRG1+ cells",
    ),
    "scpro_treg_effector_group_means_description": (
        "免疫抑制/效应候选与 treg_klrg1_immune 模块在 KLRG1+/KLRG1- 的组均值+检出率，附方向语境。",
        "Group means and detection rates of immunosuppressive/effector candidates and of the treg_klrg1_immune module in KLRG1+ versus KLRG1- cells, with directional context.",
    ),
    "nociceptor_subtype_treatment_matrix_title": (
        "亚型×处理效应矩阵图",
        "Treatment effect matrix across subtypes",
    ),
    "nociceptor_subtype_treatment_matrix_description": (
        "TrkA/IB4/Mechano 亚型内 Inflamed vs Control 的候选蛋白组均值+检出率矩阵，标注 logFC/FDR 与模块 delta；弱信号时只陈述方向与模块支持。",
        "Group-mean and detection-rate matrix of candidate proteins for Inflamed versus Control within the TrkA, IB4 and Mechano subtypes, annotated with logFC/FDR and module deltas; where signal is weak it states direction and module support only.",
    ),
    "nociceptor_subtype_marker_heatmap_title": (
        "三基线亚型 z-score marker 热图",
        "Marker z-score heatmap of the three baseline subtypes",
    ),
    "nociceptor_subtype_marker_heatmap_description": (
        "TrkA/IB4/Mechano 基线亚型的蛋白 z-score 层次聚类热图，突出亚型 marker 与感觉通道分离。",
        "Hierarchically clustered z-score heatmap of proteins in the TrkA, IB4 and Mechano baseline subtypes, highlighting subtype markers and the separation of sensory channels.",
    ),
    "nociceptor_baseline_signature_compare_title": (
        "基线亚型富集签名对比图",
        "Enrichment signatures of the baseline subtypes compared",
    ),
    "nociceptor_baseline_signature_compare_description": (
        "三个基线亚型对比的上调富集签名并列对比，核对 TrkA 代谢/生物合成与 IB4/机械感受 C-fiber 方向。",
        "Side-by-side comparison of the up-regulated enrichment signatures of the three baseline subtype contrasts, cross-checking the metabolic/biosynthetic direction in TrkA and the mechanosensory C-fiber direction in IB4.",
    ),
    "carr_lps_reactome_signature_title": (
        "LPS Reactome NES 签名条形图",
        "Reactome NES signature of the LPS response",
    ),
    "carr_lps_reactome_signature_description": (
        "按 signed -log10(p) 排序的 Reactome NES 水平条形图，突出干扰素/白介素/感染疾病炎症签名方向。",
        "Horizontal Reactome NES bar chart sorted by signed -log10(p), highlighting the direction of interferon, interleukin and infectious-disease inflammatory signatures.",
    ),
    "carr_batch_treatment_boundary_title": (
        "LPS 效应与批次边界图",
        "LPS effect against the batch boundary",
    ),
    "carr_batch_treatment_boundary_description": (
        "量化批次效应与处理效应的相对大小（校正前后方差/PCA），支撑 LPS 效应大于批次边界的定量结论。",
        "Quantifies the relative size of batch and treatment effects (variance and PCA before and after correction), supporting the quantitative conclusion that the LPS effect exceeds the batch boundary.",
    ),
    "carr_per_cell_depth_bar_title": (
        "单样本蛋白检出深度条形图",
        "Per-sample protein detection depth",
    ),
    "carr_per_cell_depth_bar_description": (
        "按处理组分组的单样本蛋白组检出数条形图（median+MAD+样本点），补足聚合 QC 缺少的逐样本深度视图。",
        "Bar chart of per-sample detected-protein counts grouped by treatment (median, MAD and sample points), supplying the per-sample depth view that the aggregated QC lacks.",
    ),
    "ipsc_pluripotency_gradient_title": (
        "iPSC→EB 多能性/谱系梯度图",
        "Pluripotency and lineage gradient from iPSC to EB",
    ),
    "ipsc_pluripotency_gradient_description": (
        "多能性 marker 着色的 PCA 与 OCT4/SOX2 组间箱线（t 检验 P），连接多能性下降与谱系/ECM 上升。",
        "PCA coloured by pluripotency markers together with OCT4/SOX2 between-group boxplots (t-test P), linking the decline in pluripotency to the rise in lineage and ECM proteins.",
    ),
    "ipsc_eb_heterogeneity_clusters_title": (
        "EB 内部异质性聚类与功能注释图",
        "Within-EB heterogeneity clusters with functional annotation",
    ),
    "ipsc_eb_heterogeneity_clusters_description": (
        "EB 内部显著调控蛋白的无监督聚类与逐簇功能注释，展示 EB 内异质性而非单一分化终点。",
        "Unsupervised clustering of significantly regulated proteins within EBs with per-cluster functional annotation, showing intra-EB heterogeneity rather than a single differentiation endpoint.",
    ),
}

register("recipes", _RECIPE_TEXT)


# ---------------------------------------------------------------------------
# Recipe data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureRecipe:
    """One dataset-specific figure recipe.

    ``recipe_id`` is globally unique and snake_case; ``caption_slug`` defaults
    to ``recipe_id`` and names the output file (``recipe_<slug>.png``) and the
    figure-manifest ``plot_type`` (``recipe_<slug>``).
    """

    recipe_id: str
    dataset: str
    plot_function: str
    input_tables: Tuple[str, ...]
    figure_kind: str = "extension"
    paper_figure: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    caption_slug: str = ""

    # The manifest and caption text lives in the ``recipes`` namespace of the
    # shared report-language registry. These are properties rather than fields so
    # that a run renders the language active when the manifest is written.
    @property
    def title(self) -> str:
        """Manifest/caption title in the active report language."""
        return t("recipes." + self.recipe_id + "_title")

    @property
    def description(self) -> str:
        """Manifest/caption description in the active report language."""
        return t("recipes." + self.recipe_id + "_description")

    def __post_init__(self) -> None:
        if not self.caption_slug:
            object.__setattr__(self, "caption_slug", self.recipe_id)
        object.__setattr__(self, "input_tables", tuple(str(t) for t in self.input_tables))
        object.__setattr__(self, "params", dict(self.params or {}))

    @property
    def out_file(self) -> str:
        return f"recipe_{self.caption_slug}.png"

    @property
    def plot_type(self) -> str:
        return f"recipe_{self.caption_slug}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "dataset": self.dataset,
            "plot_function": self.plot_function,
            "input_tables": list(self.input_tables),
            "params": dict(self.params),
            "paper_figure": self.paper_figure,
            "figure_kind": self.figure_kind,
            "caption_slug": self.caption_slug,
            "title": self.title,
            "description": self.description,
            "out_file": self.out_file,
            "plot_type": self.plot_type,
        }


def _r(
    recipe_id: str,
    dataset: str,
    plot_function: str,
    input_tables: Tuple[str, ...],
    *,
    figure_kind: str,
    paper_figure: str = "",
    params: Dict[str, Any] | None = None,
) -> FigureRecipe:
    """Build one recipe; its title and description come from the registry."""
    return FigureRecipe(
        recipe_id=recipe_id,
        dataset=dataset,
        plot_function=plot_function,
        input_tables=input_tables,
        figure_kind=figure_kind,
        paper_figure=paper_figure,
        params=params or {},
    )


# ---------------------------------------------------------------------------
# Per-dataset ordered recipes (aligned with Step 01 original_figure_map.json
# extendable gaps and AGENTS.md dataset recipe notes; plot functions land in
# Step 06).
# ---------------------------------------------------------------------------

DATASET_FIGURE_RECIPES: Dict[str, Tuple[FigureRecipe, ...]] = {
    # ----------------------------------------------------------------- PiSPA
    "Nat_Commun_PiSPA_2024": (
        _r(
            "pispa_cluster_type_composition",
            "Nat_Commun_PiSPA_2024",
            "plot_recipe_pispa_cluster_type_composition",
            (
                "evaluation_evidence/group_cross_table_Cluster_by_Type.csv",
                "evaluation_evidence/group_composition_qc.csv",
                "SampleInfo.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 5b",
            params={
                "cluster_order": ["Cluster 1", "Cluster 2", "Cluster 3"],
                "context_col": "Type",
                "migrated_label": "Migrated",
            },
        ),
        _r(
            "pispa_rho_gtpase_candidate_boxplot",
            "Nat_Commun_PiSPA_2024",
            "plot_recipe_pispa_rho_gtpase_candidate_boxplot",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/SampleInfo_Filtered.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "processed_proteins/differential_Cluster_1_vs_Cluster_2.csv",
                "processed_proteins/differential_Cluster_1_vs_Cluster_3.csv",
                "processed_proteins/differential_Cluster_2_vs_Cluster_3.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 5f",
            params={
                "genes": ["CDC42", "RAC1", "RHOA", "EZR", "MSN", "MYL9", "TLN1", "VCL", "ITGB6", "FLNA", "ACTN1"],
                "group_order": ["Cluster 1", "Cluster 2", "Cluster 3"],
            },
        ),
        _r(
            "pispa_migration_module_gradient",
            "Nat_Commun_PiSPA_2024",
            "plot_recipe_pispa_migration_module_gradient",
            (
                "evaluation_evidence/curated_module_group_summary.csv",
                "evaluation_evidence/curated_module_sample_scores.csv",
            ),
            figure_kind="extension",
            paper_figure="",
            params={
                "modules": [
                    "migration_signature",
                    "rho_gtpase_migration",
                    "erm_membrane_cortex",
                    "myosin_contractility",
                    "talin_vinculin_focal_adhesion",
                ],
                "group_order": ["Cluster 1", "Cluster 2", "Cluster 3"],
            },
        ),
    ),
    # ----------------------------------------------------------------- Brain
    "Nat_Biotech_Brain_2026": (
        _r(
            "brain_state_axis_module_gradient",
            "Nat_Biotech_Brain_2026",
            "plot_recipe_brain_state_axis_module_gradient",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/SampleInfo_Filtered.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "evaluation_evidence/curated_module_group_summary.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 6a,b",
            params={
                "state_order": ["RG", "oRG", "IPC-EN", "EN"],
                "state_markers": ["HOPX", "TNC", "EOMES", "TBR1", "BCL11B", "NEUROD2", "MAP2"],
                "modules": ["brain_ipc_en_transition", "brain_en_maturation", "brain_synapse_neurite", "brain_chromatin_baf"],
            },
        ),
        _r(
            "brain_marker_cluster_heatmap",
            "Nat_Biotech_Brain_2026",
            "plot_recipe_brain_marker_cluster_heatmap",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "processed_proteins/SampleInfo_Filtered.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 2g",
            params={
                "markers": ["HOPX", "TNC", "EOMES", "TBR1", "SCGN", "CALB2", "S100B", "SIRT2", "P2RY12", "PDGFRB", "PCNA", "MKI67"],
            },
        ),
    ),
    # ------------------------------------------------------------------- DVP
    "Nat_Methods_DVP_2023": (
        _r(
            "dvp_zonation_zone_axis_heatmap",
            "Nat_Methods_DVP_2023",
            "plot_recipe_dvp_zonation_zone_axis_heatmap",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/SampleInfo_Filtered.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "processed_proteins/differential_Central_vs_Portal.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 3c,d,e",
            params={
                "zone_order": ["Portal", "Midlobular", "Central"],
                "markers": ["ARG1", "ASS1", "ASL", "CPS1", "GLUL", "CYP2E1", "CYP1A2"],
            },
        ),
        _r(
            "dvp_zonation_module_curves",
            "Nat_Methods_DVP_2023",
            "plot_recipe_dvp_zonation_module_curves",
            (
                "evaluation_evidence/curated_module_group_summary.csv",
                "evaluation_evidence/curated_module_scores.json",
                "processed_proteins/SampleInfo_Filtered.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 3h,i",
            params={
                "modules": ["liver_periportal_urea", "liver_central_xenobiotic"],
                "zone_order": ["Portal", "Midlobular", "Central"],
            },
        ),
    ),
    # -------------------------------------------------------------- BloodCell
    "Science_BloodCell_2025": (
        _r(
            "bloodcell_hspc_state_hierarchy_dotplot",
            "Science_BloodCell_2025",
            "plot_recipe_bloodcell_hspc_state_hierarchy_dotplot",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "processed_proteins/differential_GMP_vs_HSC.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 2f,g",
            params={
                "state_order": ["HSC", "MPP", "LMPP", "GMP", "MEP"],
                "markers": ["TALDO1", "H1F0", "MPO", "ELANE"],
            },
        ),
        _r(
            "bloodcell_cluster_label_agreement",
            "Science_BloodCell_2025",
            "plot_recipe_bloodcell_cluster_label_agreement",
            (
                "evaluation_evidence/group_cross_table_Cluster_by_Label.csv",
                "processed_proteins/SampleInfo_Filtered.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 2d",
            params={},
        ),
    ),
    # -------------------------------------------------------------- Turnover
    "Cell_TurnoverDynamics_2025": (
        _r(
            "turnover_drug_dose_footprint",
            "Cell_TurnoverDynamics_2025",
            "plot_recipe_turnover_drug_dose_footprint",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "SampleInfo.csv",
                "processed_proteins/differential_Bortezomib_High_vs_Control.csv",
                "processed_proteins/differential_Bortezomib_Low_vs_Control.csv",
                "processed_proteins/differential_Cycloheximide_High_vs_Control.csv",
                "processed_proteins/differential_Cycloheximide_Low_vs_Control.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 2b,d,f",
            params={
                "drugs": {
                    "Bortezomib": ["Low", "High"],
                    "Cycloheximide": ["Low", "High"],
                },
                "control_group": "Control",
            },
        ),
        _r(
            "turnover_drug_module_direction",
            "Cell_TurnoverDynamics_2025",
            "plot_recipe_turnover_drug_module_direction",
            (
                "enrichment_results/go_kegg_reactome_results.json",
                "enrichment_results/GO_Bortezomib_High_vs_Control_upregulated_proteins.csv",
                "enrichment_results/GO_Bortezomib_High_vs_Control_downregulated_proteins.csv",
                "enrichment_results/GO_Cycloheximide_High_vs_Control_upregulated_proteins.csv",
                "enrichment_results/GO_Cycloheximide_High_vs_Control_downregulated_proteins.csv",
                "evaluation_evidence/curated_module_group_summary.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 2e,g",
            params={
                "drug_modules": {
                    "Bortezomib": ["proteasome_proteostasis"],
                    "Cycloheximide": ["ribosome_translation"],
                },
            },
        ),
        _r(
            "turnover_sc_lowinput_stratification",
            "Cell_TurnoverDynamics_2025",
            "plot_recipe_turnover_sc_lowinput_stratification",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "SampleInfo.csv",
                "evaluation_evidence/group_cross_table_Cluster_by_Label.csv",
            ),
            figure_kind="extension",
            paper_figure="Supplementary Fig. S3",
            params={"tier_col": "Label"},
        ),
    ),
    # --------------------------------------------------------------- pSCoPE
    "Nat_Methods_pSCoPE_2023": (
        _r(
            "pscope_polarization_gradient",
            "Nat_Methods_pSCoPE_2023",
            "plot_recipe_pscope_polarization_gradient",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "enrichment_results/GO_LPS_high_vs_Untreated_upregulated_proteins.csv",
                "processed_proteins/differential_LPS_high_vs_Untreated.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 4a",
            params={
                "polarization_order": ["Untreated", "LPS_low", "LPS_high"],
                "signature_modules": ["phagosome_vatpase_lysosome", "inflammation_interferon"],
                "signature_genes": ["ATP6V0A1", "ATP6V1A", "ATP6V1B2", "LAMP1", "LAMP2", "CTSB", "CTSD"],
            },
        ),
        _r(
            "pscope_vatpase_covariation",
            "Nat_Methods_pSCoPE_2023",
            "plot_recipe_pscope_vatpase_covariation",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "processed_proteins/differential_LPS_high_vs_LPS_low.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 4d",
            params={
                "gene_sets": {
                    "vatpase_proton_transport": ["ATP6V0A1", "ATP6V1A", "ATP6V1B2", "ATP6V1E1"],
                    "phagosome_lysosome": ["LAMP1", "LAMP2", "CTSB", "CTSD"],
                },
                "condition_order": ["Untreated", "LPS_low", "LPS_high"],
            },
        ),
    ),
    # -------------------------------------------------------- ProteinLeakage
    "Nat_Commun_ProteinLeakage_2025": (
        _r(
            "proteinleakage_compartment_leakage",
            "Nat_Commun_ProteinLeakage_2025",
            "plot_recipe_proteinleakage_compartment_leakage",
            (
                "processed_proteins/differential_Intact_vs_Permeable.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "proteins_leakage/leakage_module_summary.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 1e",
            params={
                "compartments": {
                    "cytosol_nucleus_depleted": ["GAPDH", "LDHA", "TUBA1B", "LMNB1"],
                    "mito_membrane_retained": ["VDAC1", "ATP5F1A"],
                },
            },
        ),
        _r(
            "proteinleakage_status_confounder_dotplot",
            "Nat_Commun_ProteinLeakage_2025",
            "plot_recipe_proteinleakage_status_confounder_dotplot",
            (
                "processed_proteins/SampleInfo_Filtered.csv",
                "evaluation_evidence/group_cross_table_Type1_by_Type2.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 1c",
            params={"cluster_col": "Cluster", "type1_col": "Type1", "type2_col": "Type2"},
        ),
        _r(
            "proteinleakage_leakage_correlation_heatmap",
            "Nat_Commun_ProteinLeakage_2025",
            "plot_recipe_proteinleakage_leakage_correlation_heatmap",
            (
                "proteins_leakage/leakage_summary.json",
                "processed_proteins/SampleInfo_Filtered.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 2a",
            params={},
        ),
    ),
    # ----------------------------------------------------------------- SCPro
    "Nat_Commun_SCPro_2024": (
        _r(
            "scpro_klrg1_context_matrix",
            "Nat_Commun_SCPro_2024",
            "plot_recipe_scpro_klrg1_context_matrix",
            (
                "processed_proteins/differential_CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg.csv",
                "processed_proteins/differential_CD4_Klrg1pos_vs_CD4_Klrg1neg.csv",
                "processed_proteins/differential_CD8_Klrg1pos_vs_CD8_Klrg1neg.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 6f",
            params={
                "contexts": [
                    {"label": "Treg", "contrast": "CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg"},
                    {"label": "CD4", "contrast": "CD4_Klrg1pos_vs_CD4_Klrg1neg"},
                    {"label": "CD8", "contrast": "CD8_Klrg1pos_vs_CD8_Klrg1neg"},
                ],
                "focus_genes": ["KLRG1", "FOXP3", "IL2RA", "CTLA4", "IKZF2", "LAG3", "TIGIT"],
            },
        ),
        _r(
            "scpro_treg_klrg1_enrichment",
            "Nat_Commun_SCPro_2024",
            "plot_recipe_scpro_treg_klrg1_enrichment",
            (
                "processed_proteins/differential_CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg.csv",
                "enrichment_results/GO_CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg_upregulated_proteins.csv",
                "enrichment_results/GO_CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg_downregulated_proteins.csv",
                "enrichment_results/KEGG_CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg_upregulated_proteins.csv",
                "enrichment_results/Reactome_CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg_upregulated_proteins.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 6g",
            params={"contrast": "CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg"},
        ),
        _r(
            "scpro_treg_effector_group_means",
            "Nat_Commun_SCPro_2024",
            "plot_recipe_scpro_treg_effector_group_means",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "evaluation_evidence/curated_module_group_summary.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 6h",
            params={
                "genes": ["CASP1", "KDELR2", "MAP4K1", "CTLA4", "IKZF2", "TIGIT"],
                "module": "treg_klrg1_immune",
                "group_order": ["CD4_CD25pos_Klrg1pos", "CD4_CD25pos_Klrg1neg"],
            },
        ),
    ),
    # ----------------------------------------------------------- Nociceptor
    "Nat_Commun_Nociceptor_2026": (
        _r(
            "nociceptor_subtype_treatment_matrix",
            "Nat_Commun_Nociceptor_2026",
            "plot_recipe_nociceptor_subtype_treatment_matrix",
            (
                "processed_proteins/differential_TrkA_Inflamed_vs_TrkA_Control.csv",
                "processed_proteins/differential_IB4_Inflamed_vs_IB4_Control.csv",
                "processed_proteins/differential_Mechano_Inflamed_vs_Mechano_Control.csv",
                "processed_proteins/ProteinQuant_ComBat.csv",
                "evaluation_evidence/curated_module_group_summary.csv",
                "evaluation_evidence/curated_module_sample_scores.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 4e-g",
            params={
                "subtypes": ["TrkA", "IB4", "Mechano"],
                "focus_genes": ["B3GNT2", "RRAD", "NTRK1"],
                "modules": ["nociceptor_inflammation_response", "membrane_glycosylation_traffic"],
            },
        ),
        _r(
            "nociceptor_subtype_marker_heatmap",
            "Nat_Commun_Nociceptor_2026",
            "plot_recipe_nociceptor_subtype_marker_heatmap",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "processed_proteins/limma_summary.json",
                "processed_proteins/differential_TrkA_Control_vs_IB4_Control.csv",
                "processed_proteins/differential_TrkA_Control_vs_Mechano_Control.csv",
            ),
            figure_kind="extension",
            paper_figure="Supplementary Fig. 1c",
            params={},
        ),
        _r(
            "nociceptor_baseline_signature_compare",
            "Nat_Commun_Nociceptor_2026",
            "plot_recipe_nociceptor_baseline_signature_compare",
            (
                "enrichment_results/GO_TrkA_Control_vs_IB4_Control_upregulated_proteins.csv",
                "enrichment_results/GO_IB4_Control_vs_Mechano_Control_upregulated_proteins.csv",
                "enrichment_results/GO_TrkA_Control_vs_Mechano_Control_upregulated_proteins.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 1k",
            params={},
        ),
    ),
    # ----------------------------------------------------------------- Carr
    "Nat_Commun_Carr_2024": (
        _r(
            "carr_lps_reactome_signature",
            "Nat_Commun_Carr_2024",
            "plot_recipe_carr_lps_reactome_signature",
            (
                "processed_proteins/differential_LPS_vs_DMSO.csv",
                "enrichment_results/Reactome_LPS_vs_DMSO_upregulated_proteins.csv",
                "enrichment_results/go_kegg_reactome_results.json",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 3e",
            params={"contrast": "LPS_vs_DMSO", "top_n_terms": 12},
        ),
        _r(
            "carr_batch_treatment_boundary",
            "Nat_Commun_Carr_2024",
            "plot_recipe_carr_batch_treatment_boundary",
            (
                "processed_proteins/ProteinQuant_Filtered.csv",
                "processed_proteins/ProQuant_Normalized.csv",
                "processed_proteins/ProteinQuant_ComBat.csv",
                "SampleInfo.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Supplementary Fig. 5",
            params={"treatment_col": "Cluster", "batch_col": "Batch"},
        ),
        _r(
            "carr_per_cell_depth_bar",
            "Nat_Commun_Carr_2024",
            "plot_recipe_carr_per_cell_depth_bar",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "SampleInfo.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 3b",
            params={},
        ),
    ),
    # ----------------------------------------------------------------- iPSC
    "Nat_Methods_iPSC_2025": (
        _r(
            "ipsc_pluripotency_gradient",
            "Nat_Methods_iPSC_2025",
            "plot_recipe_ipsc_pluripotency_gradient",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "processed_proteins/SampleInfo_Filtered.csv",
                "processed_proteins/differential_EB_vs_iPSCs.csv",
            ),
            figure_kind="original_equivalent",
            paper_figure="Fig. 6b,c",
            params={
                "pluripotency_markers": ["POU5F1", "SOX2", "NANOG", "LIN28A"],
                "lineage_markers": ["GATA4", "HAND1", "MAP2", "FN1", "COL1A1"],
                "group_order": ["iPSCs", "EB"],
            },
        ),
        _r(
            "ipsc_eb_heterogeneity_clusters",
            "Nat_Methods_iPSC_2025",
            "plot_recipe_ipsc_eb_heterogeneity_clusters",
            (
                "processed_proteins/ProteinQuant_ComBat.csv",
                "processed_proteins/Protein_Gene_Map.csv",
                "processed_proteins/SampleInfo_Filtered.csv",
                "processed_proteins/differential_EB_vs_iPSCs.csv",
                "enrichment_results/GO_EB_vs_iPSCs_upregulated_proteins.csv",
                "enrichment_results/GO_EB_vs_iPSCs_downregulated_proteins.csv",
            ),
            figure_kind="extension",
            paper_figure="Fig. 6d",
            params={},
        ),
    ),
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def known_dataset_names() -> Tuple[str, ...]:
    """Known dataset names, longest first (safe prefix matching for run folders)."""
    return tuple(sorted(DATASET_FIGURE_RECIPES, key=len, reverse=True))


def has_recipes_for_dataset(dataset: str) -> bool:
    return str(dataset or "").strip() in DATASET_FIGURE_RECIPES


def get_recipes_for_dataset(dataset: str) -> Tuple[FigureRecipe, ...]:
    return DATASET_FIGURE_RECIPES.get(str(dataset or "").strip(), ())


def iter_all_recipes() -> Iterator[FigureRecipe]:
    for recipes in DATASET_FIGURE_RECIPES.values():
        yield from recipes


def find_recipe(recipe_id: str) -> FigureRecipe | None:
    for recipe in iter_all_recipes():
        if recipe.recipe_id == str(recipe_id or "").strip():
            return recipe
    return None


def match_dataset_from_folder(name: str) -> str:
    """Match a run/input folder name (or design input_folder basename) to a dataset."""
    text = str(name or "").strip()
    if not text:
        return ""
    for candidate in known_dataset_names():
        if text == candidate or text.startswith(candidate + "_") or text.startswith(candidate + "-"):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Schema self-check
# ---------------------------------------------------------------------------

_RECIPE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")
_CAPTION_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")


def validate_figure_recipes() -> Dict[str, Any]:
    """Schema self-check over every declared recipe.

    Checks: recipe ids unique, datasets known and 11/11 covered, plot_function
    is a valid identifier, figure_kind legal, caption_slug valid + unique,
    input tables follow run-artifact naming conventions, params is a dict,
    title/description present. Returns a JSON-able report; ``ok`` is True when
    no problems were found.
    """
    problems: List[str] = []
    seen_ids: Dict[str, str] = {}
    seen_slugs: Dict[str, str] = {}
    n_by_kind: Dict[str, int] = {kind: 0 for kind in FIGURE_KINDS}

    declared_datasets = set(DATASET_FIGURE_RECIPES.keys())
    for dataset in MAIN_DATASETS:
        if dataset not in declared_datasets:
            problems.append(f"dataset coverage missing: {dataset}")
    for dataset in sorted(declared_datasets - set(MAIN_DATASETS)):
        problems.append(f"unknown dataset declared: {dataset}")

    recipes_report: List[Dict[str, Any]] = []
    for recipe in iter_all_recipes():
        rid = recipe.recipe_id
        if rid in seen_ids:
            problems.append(f"duplicate recipe_id: {rid} ({seen_ids[rid]} vs {recipe.dataset})")
        seen_ids.setdefault(rid, recipe.dataset)
        if not _RECIPE_ID_RE.match(rid):
            problems.append(f"{rid}: recipe_id must be snake_case")
        if recipe.dataset not in MAIN_DATASETS:
            problems.append(f"{rid}: unknown dataset {recipe.dataset}")
        if not recipe.plot_function or not recipe.plot_function.isidentifier():
            problems.append(f"{rid}: plot_function is not a valid identifier: {recipe.plot_function!r}")
        if recipe.figure_kind not in FIGURE_KINDS:
            problems.append(f"{rid}: illegal figure_kind {recipe.figure_kind!r}")
        else:
            n_by_kind[recipe.figure_kind] += 1
        if recipe.caption_slug in seen_slugs:
            problems.append(f"{rid}: duplicate caption_slug {recipe.caption_slug} ({seen_slugs[recipe.caption_slug]})")
        seen_slugs.setdefault(recipe.caption_slug, rid)
        if not _CAPTION_SLUG_RE.match(recipe.caption_slug):
            problems.append(f"{rid}: caption_slug must be snake_case: {recipe.caption_slug!r}")
        if not recipe.input_tables:
            problems.append(f"{rid}: input_tables is empty")
        for table in recipe.input_tables:
            if not input_table_matches_convention(table):
                problems.append(f"{rid}: input table breaks run-artifact naming convention: {table}")
        if not isinstance(recipe.params, dict):
            problems.append(f"{rid}: params must be a dict")
        if not str(recipe.title or "").strip():
            problems.append(f"{rid}: title is empty")
        if not str(recipe.description or "").strip():
            problems.append(f"{rid}: description is empty")
        for suffix in ("title", "description"):
            text_key = "recipes." + rid + "_" + suffix
            if not has_template("zh", text_key):
                problems.append(f"{rid}: {suffix} text is not registered for zh")
            elif not has_template("en", text_key):
                problems.append(f"{rid}: {suffix} text has no en template")
        recipes_report.append({
            "recipe_id": recipe.recipe_id,
            "dataset": recipe.dataset,
            "figure_kind": recipe.figure_kind,
            "paper_figure": recipe.paper_figure,
            "plot_function": recipe.plot_function,
            "caption_slug": recipe.caption_slug,
            "n_input_tables": len(recipe.input_tables),
        })

    return {
        "ok": not problems,
        "schema_version": SCHEMA_VERSION,
        "n_datasets": len(declared_datasets),
        "n_recipes": len(recipes_report),
        "n_by_kind": n_by_kind,
        "problems": problems,
        "recipes": recipes_report,
    }


if __name__ == "__main__":
    report = validate_figure_recipes()
    print(f"figure_recipes schema self-check: ok={report['ok']} "
          f"datasets={report['n_datasets']} recipes={report['n_recipes']} kinds={report['n_by_kind']}")
    for problem in report["problems"]:
        print(f"  PROBLEM: {problem}")
    sys.exit(0 if report["ok"] else 1)
