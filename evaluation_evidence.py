import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from evidence_utils import append_evidence
from report_language import register, t


GENE_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,12}\b")

TOKEN_STOPWORDS = {
    "AGENT", "ALL", "ARI", "ASD", "BH", "CD", "CSV", "DIA", "DNA", "EB", "ECM",
    "EN", "FACS", "FDR", "GO", "GSEA", "HSC", "HSPC", "IB4", "IPC", "KEGG",
    "LLM", "LPS", "MS", "NADPH", "NDD", "NF", "PCA", "PDF", "PDAC", "PG",
    "PMA", "QC", "RNA", "RG", "UMAP", "DMSO", "DRG", "NGF", "PBS", "SILAC",
}

CURATED_CANDIDATES: Dict[str, List[str]] = {
    "Nat_Commun_Carr_2024": ["CD44", "DDX21", "STAT1", "IRF1", "NFKB1", "RELA"],
    "Nat_Commun_PiSPA_2024": ["EZR", "MSN", "MYL9", "CDC42", "RAC1", "RHOA", "VCL", "TLN1", "FLNA", "ACTN1", "TPM4", "ITGB1", "ITGAV", "ITGB6", "PXN", "RDX"],
    "Nat_Methods_DVP_2023": ["ARG1", "ASS1", "CPS1", "GLUL", "CYP2E1", "CYP1A2", "UGT1A1", "GSTM1", "ALDH1A1"],
    "Nat_Commun_Nociceptor_2026": ["B3GNT2", "RRAD", "NTRK1", "PIEZO2", "SCN10A", "TRPV1"],
    "Science_BloodCell_2025": ["TALDO1", "H1F0", "H1-0", "HIST1H1A", "MPO", "ELANE"],
    "Nat_Methods_pSCoPE_2023": ["ATP6V0A1", "ATP6V1A", "LAMP1", "LAMP2", "CTSB", "CTSD", "CTSL"],
    "Nat_Methods_iPSC_2025": ["POU5F1", "SOX2", "LIN28A", "NANOG", "DPPA4", "DNMT3B", "GATA4", "HAND1", "MAP2", "VIM", "FN1", "COL1A1", "COL3A1"],
    "Nat_Commun_ProteinLeakage_2025": ["GAPDH", "LDHA", "TUBA1B", "LMNB1", "HIST1H1A", "VDAC1", "ATP5F1A"],
    "Nat_Commun_Slavov_2025": ["GAPDH", "LDHA", "TUBA1B", "LMNB1", "HIST1H1A", "VDAC1", "ATP5F1A"],
    "Nat_Commun_SCPro_2024": ["KLRG1", "FOXP3", "IL2RA", "CTLA4", "IKZF2", "LAG3", "TIGIT"],
    "Nat_Biotech_Brain_2026": ["SOX2", "PAX6", "EOMES", "TBR1", "BCL11B", "NEUROD2", "DCX", "TUBB3", "MAP2", "SYN1", "ADNP", "ARID1A", "ARID1B", "CHD3", "RUVBL2", "SMARCC2", "CTBP1", "SMARCA4", "STXBP1", "SYNGAP1", "SCN2A"],
    "Nat_Commun_DIA_2025": ["HSP90AA1", "ACTB", "ENO1", "PGK1", "TDH3", "ENO2"],
    "Cell_TurnoverDynamics_2025": ["PSMA1", "PSMB1", "PSMD1", "RPLP0", "RPS3", "EIF4A1", "HSPA1A"],
}

CURATED_MODULES: Dict[str, Dict[str, List[str]]] = {
    "inflammation_interferon": ["STAT1", "STAT2", "IRF1", "IRF7", "ISG15", "MX1", "OAS1", "NFKB1", "RELA", "IL1B", "TNF"],
    "migration_adhesion_cytoskeleton": ["EZR", "MSN", "MYL9", "CDC42", "RAC1", "RHOA", "VCL", "TLN1", "FLNA", "ACTN1", "TPM4", "ITGB1", "PXN"],
    "liver_periportal_urea": ["ARG1", "ASS1", "ASL", "CPS1", "OTC", "SDS", "HAL", "GLDC"],
    "liver_central_xenobiotic": ["GLUL", "CYP1A2", "CYP2E1", "CYP3A11", "UGT1A1", "GSTM1", "GSTM3", "POR", "ALDH1A1", "FMO1"],
    "protein_leakage_cytosol_nucleus": ["GAPDH", "LDHA", "ENO1", "TUBA1B", "ACTB", "LMNB1", "HIST1H1A", "HNRNPA1"],
    "protein_leakage_mito_membrane": ["VDAC1", "VDAC2", "ATP5F1A", "ATP5F1B", "COX4I1", "TOMM20", "SLC25A3"],
    "treg_klrg1_immune": ["KLRG1", "FOXP3", "IL2RA", "CTLA4", "IKZF2", "LAG3", "TIGIT", "ENTPD1", "STAT5A"],
    "hsc_maintenance_ppp_chromatin": ["TALDO1", "H1F0", "H1-0", "G6PD", "PGD", "TKT", "HMGB1", "HIST1H1A"],
    "granulocyte_granule": ["MPO", "ELANE", "CTSG", "AZU1", "PRTN3", "LTF", "S100A8", "S100A9"],
    "phagosome_vatpase_lysosome": ["ATP6V0A1", "ATP6V1A", "ATP6V1B2", "LAMP1", "LAMP2", "CTSB", "CTSD", "CTSL", "RAB7A"],
    "pluripotency": ["POU5F1", "NANOG", "SOX2", "LIN28A", "SALL4", "DNMT3B"],
    "pluripotency_core": ["POU5F1", "SOX2", "NANOG", "LIN28A", "SALL4", "DPPA4", "DNMT3B"],
    "lineage_endoderm": ["GATA4", "GATA6", "AFP", "ALB", "SOX17", "FOXA2"],
    "lineage_mesoderm": ["HAND1", "ACTA2", "TAGLN", "DES", "FN1", "COL1A1", "COL3A1"],
    "lineage_ectoderm": ["MAP2", "TUBB3", "NCAM1", "NES", "NEFL", "NEFM"],
    "cell_cycle_replication": ["MKI67", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "PCNA", "POLA1", "RFC4"],
    "chromatin_open_remodeling": ["SMARCA4", "SMARCC1", "SMARCC2", "ARID1A", "ARID1B", "EP300", "KAT2A", "CHD3"],
    "ecm_adhesion": ["FN1", "COL1A1", "COL3A1", "COL4A1", "ITGA5", "ITGB1", "VCL", "TLN1"],
    "lineage_adhesion_emt": ["VIM", "FN1", "COL1A1", "COL3A1", "ITGA5", "ITGB1", "TAGLN"],
    "migration_signature": ["EZR", "MSN", "MYL9", "CDC42", "RAC1", "RHOA", "VCL", "TLN1", "FLNA", "ACTN1", "TPM4", "ITGB1", "PXN"],
    "rho_gtpase_migration": ["CDC42", "RAC1", "RHOA", "RHOC", "ARPC2", "ARPC4", "ARPC5", "WASL", "CYFIP1"],
    "erm_membrane_cortex": ["EZR", "MSN", "RDX", "SPTBN1", "ACTN1", "FLNA"],
    "myosin_contractility": ["MYL9", "MYL6", "MYL12A", "MYH9", "MYH10", "TPM3", "TPM4"],
    "talin_vinculin_focal_adhesion": ["TLN1", "TLN2", "VCL", "PXN", "ITGB1", "ITGAV", "ITGB6", "PTK2", "ZYX"],
    "ribosome_translation": ["RPLP0", "RPL3", "RPL7", "RPS3", "RPS6", "RPS8", "EEF1A1", "EEF2", "EIF4A1"],
    "proteasome_proteostasis": ["PSMA1", "PSMA3", "PSMB1", "PSMB5", "PSMD1", "PSMD2", "PSMD3", "UBB", "HSPA1A", "HSP90AA1"],
    "brain_neuronal_maturation": ["SOX2", "PAX6", "EOMES", "DCX", "TUBB3", "MAP2", "SYN1", "STMN2"],
    "brain_rg_org_progenitor": ["SOX2", "PAX6", "HOPX", "VIM", "NES", "MSI1", "FABP7"],
    "brain_ipc_en_transition": ["EOMES", "MKI67", "TOP2A", "UBE2C", "NUSAP1", "RRM2", "SMC4"],
    "brain_en_maturation": ["TBR1", "BCL11B", "NEUROD2", "MAP2", "TUBB3", "DCX", "STMN2", "SATB2"],
    "brain_synapse_neurite": ["SYN1", "STXBP1", "SYNGAP1", "KIF5C", "NOVA2", "ELAVL3", "SCN2A"],
    "brain_chromatin_baf": ["ADNP", "ARID1A", "ARID1B", "SMARCA4", "SMARCC2", "SMARCE1", "CHD3", "RUVBL2", "CTBP1"],
    "mitochondrial_oxphos": ["NDUFS1", "NDUFS2", "NDUFV1", "UQCRC1", "UQCRC2", "COX5A", "COX5B", "ATP5F1A", "ATP5F1B"],
    "nociceptor_sensory_transduction": ["NTRK1", "PIEZO2", "SCN10A", "SCN11A", "TRPV1", "TRPA1", "TAC1", "CALCA"],
    "nociceptor_inflammation_response": ["B3GNT2", "RRAD", "STAT1", "IRF1", "ISG15", "S100A8", "S100A9", "ANXA1"],
    "membrane_glycosylation_traffic": ["B3GNT2", "B4GALT1", "MGAT1", "STT3A", "COPA", "COPB1", "RAB5A", "RAB7A", "VAMP2"],
}

DATASET_MODULES: Dict[str, List[str]] = {
    "Nat_Commun_Carr_2024": ["inflammation_interferon", "proteasome_proteostasis"],
    "Nat_Commun_PiSPA_2024": ["migration_signature", "rho_gtpase_migration", "erm_membrane_cortex", "myosin_contractility", "talin_vinculin_focal_adhesion", "ribosome_translation"],
    "Nat_Methods_DVP_2023": ["liver_periportal_urea", "liver_central_xenobiotic"],
    "Nat_Commun_Nociceptor_2026": ["nociceptor_sensory_transduction", "nociceptor_inflammation_response", "membrane_glycosylation_traffic"],
    "Science_BloodCell_2025": ["hsc_maintenance_ppp_chromatin", "granulocyte_granule"],
    "Nat_Methods_pSCoPE_2023": ["inflammation_interferon", "phagosome_vatpase_lysosome"],
    "Nat_Methods_iPSC_2025": ["pluripotency_core", "lineage_endoderm", "lineage_mesoderm", "lineage_ectoderm", "cell_cycle_replication", "chromatin_open_remodeling", "ecm_adhesion", "ribosome_translation", "mitochondrial_oxphos"],
    "Nat_Commun_ProteinLeakage_2025": ["protein_leakage_cytosol_nucleus", "protein_leakage_mito_membrane"],
    "Nat_Commun_Slavov_2025": ["protein_leakage_cytosol_nucleus", "protein_leakage_mito_membrane"],
    "Nat_Commun_SCPro_2024": ["treg_klrg1_immune", "ribosome_translation"],
    "Nat_Biotech_Brain_2026": ["brain_rg_org_progenitor", "brain_ipc_en_transition", "brain_en_maturation", "brain_synapse_neurite", "brain_chromatin_baf", "cell_cycle_replication", "mitochondrial_oxphos", "ribosome_translation"],
    "Nat_Commun_DIA_2025": ["ribosome_translation", "proteasome_proteostasis"],
    "Cell_TurnoverDynamics_2025": ["proteasome_proteostasis", "ribosome_translation"],
}

DATASET_RECIPE_REQUIREMENTS: Dict[str, List[Dict[str, str]]] = {
    "Nat_Commun_Carr_2024": [
        {"recipe_item": "LPS vs DMSO core contrast", "expected_evidence": "LPS_vs_DMSO;CD44;DDX21;STAT1;IRF1;inflammation_interferon", "boundary": "LPS direction must be displayed as LPS relative to DMSO even when the computed table used the reverse coding."},
        {"recipe_item": "Inflammation/stress evidence chain", "expected_evidence": "inflammation_interferon;proteasome_proteostasis;offline_enrichment", "boundary": "Pathway language requires current-matrix proteins and/or offline enrichment support."},
        {"recipe_item": "Batch and treatment boundary", "expected_evidence": "group_composition_qc;Batch", "boundary": "Batch balance is QC evidence; it does not by itself prove treatment biology."},
    ],
    "Nat_Methods_iPSC_2025": [
        {"recipe_item": "EB lineage module scores", "expected_evidence": "lineage_endoderm;lineage_mesoderm;lineage_ectoderm", "boundary": "current_matrix module scores; lineage calls are exploratory without orthogonal labels"},
        {"recipe_item": "Pluripotency detection boundary", "expected_evidence": "POU5F1;SOX2;LIN28A;NANOG", "boundary": "absence or low detection should be reported as matrix boundary, not biological absence"},
        {"recipe_item": "Differentiation candidate table", "expected_evidence": "GATA4;HAND1;MAP2;FN1;COL1A1", "boundary": "candidate proteins support direction only when logFC/FDR or detection evidence is present"},
    ],
    "Nat_Commun_PiSPA_2024": [
        {"recipe_item": "Cluster-Type composition", "expected_evidence": "group_composition_qc", "boundary": "cluster/type balance is current metadata evidence"},
        {"recipe_item": "Migration signature score", "expected_evidence": "migration_signature;rho_gtpase_migration;erm_membrane_cortex;myosin_contractility;talin_vinculin_focal_adhesion", "boundary": "module score summarizes current-matrix proteins and does not replace formal pathway enrichment"},
        {"recipe_item": "Cluster 1 vs 2/3 and Cluster 2 vs 3 candidate evidence", "expected_evidence": "EZR;MSN;MYL9;CDC42;RAC1;RHOA;TLN1;VCL", "boundary": "interpret cluster migration axes only with explicit contrast direction"},
    ],
    "Nat_Biotech_Brain_2026": [
        {"recipe_item": "RG/oRG to IPC/EN trajectory module comparison", "expected_evidence": "brain_rg_org_progenitor;brain_ipc_en_transition;brain_en_maturation", "boundary": "trajectory language is a score contrast unless pseudotime was explicitly computed"},
        {"recipe_item": "Neuronal and synaptic candidate table", "expected_evidence": "EOMES;TBR1;BCL11B;NEUROD2;MAP2;SYN1", "boundary": "candidate interpretation requires matrix detection/statistical support"},
        {"recipe_item": "Offline ASD/NDD risk annotation crosswalk", "expected_evidence": "external_annotation_asd_ndd_local", "boundary": "ASD/NDD labels are local external_annotation, not current-matrix disease causality"},
    ],
    "Nat_Commun_Nociceptor_2026": [
        {"recipe_item": "Subtype-internal Inflamed vs Control contrasts", "expected_evidence": "TrkA_Inflamed_vs_TrkA_Control;IB4_Inflamed_vs_IB4_Control;Mechano_Inflamed_vs_Mechano_Control", "boundary": "Inflammation response should be interpreted within subtype before making across-subtype claims."},
        {"recipe_item": "Nociceptor candidate and sensory module table", "expected_evidence": "B3GNT2;RRAD;NTRK1;PIEZO2;SCN10A;TRPV1;nociceptor_sensory_transduction", "boundary": "Candidate proteins are supportive only when detected and tied to logFC/FDR or explicit presence boundary."},
        {"recipe_item": "Inflammation and membrane/glycosylation extension", "expected_evidence": "nociceptor_inflammation_response;membrane_glycosylation_traffic", "boundary": "Glycosylation or channel-sensitization language is an extension unless current-matrix contrasts support it."},
    ],
    "Nat_Commun_SCPro_2024": [
        {"recipe_item": "Treg KLRG1 direct contrast", "expected_evidence": "CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg;KLRG1;FOXP3;IL2RA;CTLA4;treg_klrg1_immune", "boundary": "Treg KLRG1+/− must be prioritized over broad pairwise T-cell contrasts."},
        {"recipe_item": "CD4/CD8 KLRG1 context comparison", "expected_evidence": "CD4_Klrg1pos_vs_CD4_Klrg1neg;CD8_Klrg1pos_vs_CD8_Klrg1neg", "boundary": "Shared KLRG1 programs and Treg-specific programs must be separated."},
        {"recipe_item": "Immune-state module boundary", "expected_evidence": "treg_klrg1_immune;ribosome_translation", "boundary": "Clinical or therapeutic interpretation is an extension, not a current-matrix proof."},
    ],
    "Cell_TurnoverDynamics_2025": [
        {"recipe_item": "Bortezomib abundance footprint", "expected_evidence": "Bortezomib_High_vs_Control;Bortezomib_Low_vs_Control;proteasome_proteostasis", "boundary": "Proteasome-inhibitor effects are total-abundance footprints in this matrix."},
        {"recipe_item": "Cycloheximide abundance footprint", "expected_evidence": "Cycloheximide_High_vs_Control;Cycloheximide_Low_vs_Control;ribosome_translation", "boundary": "Cycloheximide contrasts must be displayed relative to Control and not as Control_vs_Cycloheximide."},
        {"recipe_item": "Turnover-rate boundary", "expected_evidence": "proteasome_proteostasis;ribosome_translation", "boundary": "Without time-resolved L/H turnover measurements, do not claim direct turnover rates."},
    ],
    "Nat_Methods_DVP_2023": [
        {"recipe_item": "Liver zonation core contrasts", "expected_evidence": "Portal_vs_Central;liver_periportal_urea;liver_central_xenobiotic;ARG1;CPS1;GLUL", "boundary": "Zonation is supported by current-matrix marker/module direction, not fixed hidden thresholds."},
        {"recipe_item": "Periportal and central candidate table", "expected_evidence": "ARG1;ASS1;CPS1;GLUL;CYP2E1;CYP1A2", "boundary": "Candidate interpretation should report logFC/FDR or detection boundary."},
    ],
    "Nat_Methods_pSCoPE_2023": [
        {"recipe_item": "LPS polarization and phagosome maturation", "expected_evidence": "LPS_high_vs_Untreated;LPS_low_vs_Untreated;phagosome_vatpase_lysosome;ATP6V0A1;LAMP1;CTSB", "boundary": "High/low LPS should be compared with untreated before cross-dose interpretation."},
        {"recipe_item": "Inflammation module boundary", "expected_evidence": "inflammation_interferon;offline_enrichment", "boundary": "Macrophage activation terms require current-matrix or offline-enrichment evidence."},
    ],
    "Science_BloodCell_2025": [
        {"recipe_item": "HSPC state hierarchy", "expected_evidence": "HSC_vs_MPP;HSC_vs_GMP;hsc_maintenance_ppp_chromatin;granulocyte_granule", "boundary": "State hierarchy language is based on sampled labels and module/candidate contrasts."},
        {"recipe_item": "Blood-cell candidate table", "expected_evidence": "TALDO1;H1F0;MPO;ELANE", "boundary": "Candidate support must retain logFC/FDR or detection boundary."},
    ],
    "Nat_Commun_ProteinLeakage_2025": [
        {"recipe_item": "Permeable vs Intact leakage contrast", "expected_evidence": "Permeable_vs_Intact;protein_leakage_cytosol_nucleus;protein_leakage_mito_membrane;GAPDH;LDHA;LMNB1", "boundary": "Leakage conclusion is a QC/status interpretation and should not be generalized to unrelated disease biology."},
        {"recipe_item": "Fresh/Frozen and cell-type boundary", "expected_evidence": "group_composition_qc;Type2;Cluster", "boundary": "Cell-type and preservation status are confounder/context evidence."},
    ],
}


# ---------------------------------------------------------------------------

# Reader-facing display text (zh released verbatim, en added)

# ---------------------------------------------------------------------------



# CORE_STORY_RECIPES fields and the labels that reach a report-visible evidence

# table cell. The zh column repeats the released literals character for character.

_EVIDENCE_TEXT = {

    "nat_commun_carr_2024_story_title": (
        "LPS 刺激是否在当前矩阵中形成可复核的炎症响应",
        "Does LPS stimulation produce a reproducible inflammatory response in the current matrix?",
    ),
    "nat_commun_carr_2024_executive_claim": (
        "当前报告必须先回答 LPS 相对 DMSO 的处理效应，再把候选蛋白、模块和富集组织成炎症/应激证据链。",
        "The report must first answer the LPS versus DMSO treatment effect and then organise candidate proteins, modules and enrichment into an inflammation/stress evidence chain.",
    ),
    "nat_commun_carr_2024_boundary": (
        "Batch 是 QC 边界；LPS/DMSO 方向以 current_matrix 差异表为准。",
        "Batch is a QC boundary; the LPS/DMSO direction follows the current_matrix differential table.",
    ),
    "contrast_carr_lps_dmso_claim": (
        "LPS 相对 DMSO 的代表差异蛋白",
        "Representative differential proteins of LPS versus DMSO",
    ),
    "contrast_carr_lps_dmso_interpretation": (
        "优先展示 LPS 上调/下调代表蛋白及 CD44、DDX21、STAT/IRF/NF-kB 同模块证据。",
        "Prioritises the representative up- and down-regulated proteins of LPS together with module evidence for CD44, DDX21 and STAT/IRF/NF-kB.",
    ),
    "nat_commun_nociceptor_2026_story_title": (
        "炎症响应是否发生在特定伤害感受神经元亚型内",
        "Does the inflammatory response occur within specific nociceptor subtypes?",
    ),
    "nat_commun_nociceptor_2026_executive_claim": (
        "核心不是亚型之间的基线差异，而是 TrkA、IB4、Mechano 各自内部 Inflamed 相对 Control 的方向和强度。",
        "The core question is not the baseline difference between subtypes but the direction and strength of Inflamed versus Control within TrkA, IB4 and Mechano separately.",
    ),
    "nat_commun_nociceptor_2026_boundary": (
        "亚型内对比来自 current_matrix；糖基化、通道致敏等只作为 extension，除非候选/模块表支持。",
        "Within-subtype contrasts come from the current_matrix; glycosylation and channel sensitisation stay extensions unless the candidate or module tables support them.",
    ),
    "contrast_trka_inflamed_claim": (
        "TrkA 亚型内炎症响应",
        "Inflammatory response within the TrkA subtype",
    ),
    "contrast_trka_inflamed_interpretation": (
        "评估肽能/TrkA+ 伤害感受神经元的炎症足迹。",
        "Assesses the inflammatory footprint of peptidergic/TrkA+ nociceptors.",
    ),
    "contrast_ib4_inflamed_claim": (
        "IB4 亚型内炎症响应",
        "Inflammatory response within the IB4 subtype",
    ),
    "contrast_ib4_inflamed_interpretation": (
        "评估非肽能 IB4+ 神经元的炎症足迹。",
        "Assesses the inflammatory footprint of non-peptidergic IB4+ neurons.",
    ),
    "contrast_mechano_inflamed_claim": (
        "Mechano 亚型内炎症响应",
        "Inflammatory response within the Mechano subtype",
    ),
    "contrast_mechano_inflamed_interpretation": (
        "评估机械感受相关细胞的炎症足迹。",
        "Assesses the inflammatory footprint of mechanosensory cells.",
    ),
    "nat_commun_scpro_2024_story_title": (
        "KLRG1 状态在 Treg 与其他 T 细胞背景中的蛋白程序差异",
        "How the KLRG1 state differs in protein programmes between the Treg and other T-cell contexts",
    ),
    "nat_commun_scpro_2024_executive_claim": (
        "报告必须优先展开 Treg KLRG1+/- 直接对比，再说明 CD4/CD8 背景下哪些方向共享或不同。",
        "The report must open with the direct Treg KLRG1+/- contrast and then state which directions are shared and which differ in the CD4 and CD8 contexts.",
    ),
    "nat_commun_scpro_2024_boundary": (
        "治疗和免疫微环境解释属于 extension，核心证据仍是 current_matrix 的 KLRG1 对比。",
        "Therapeutic and immune-microenvironment interpretations are extensions; the core evidence remains the current_matrix KLRG1 contrast.",
    ),
    "contrast_treg_klrg1_claim": (
        "Treg 内部 KLRG1+ 相对 KLRG1- 的直接差异",
        "Direct KLRG1+ versus KLRG1- difference within Tregs",
    ),
    "contrast_treg_klrg1_interpretation": (
        "这是本数据集评分最高权重的核心对比。",
        "This is the highest-weight core contrast of the dataset.",
    ),
    "contrast_cd4_klrg1_claim": (
        "CD4 背景下 KLRG1 相关程序",
        "KLRG1-associated programme in the CD4 context",
    ),
    "contrast_cd4_klrg1_interpretation": (
        "用于区分 Treg 特异与更广泛 CD4 KLRG1 程序。",
        "Separates Treg-specific from broader CD4 KLRG1 programmes.",
    ),
    "contrast_cd8_klrg1_claim": (
        "CD8 背景下 KLRG1 相关程序",
        "KLRG1-associated programme in the CD8 context",
    ),
    "contrast_cd8_klrg1_interpretation": (
        "用于判断 KLRG1 程序是否跨 T 细胞背景共享。",
        "Tests whether the KLRG1 programme is shared across T-cell contexts.",
    ),
    "cell_turnoverdynamics_2025_story_title": (
        "蛋白酶体抑制与翻译抑制是否呈现不同总丰度足迹",
        "Do proteasome inhibition and translation inhibition show distinct total-abundance footprints?",
    ),
    "cell_turnoverdynamics_2025_executive_claim": (
        "报告必须同时写清 Bortezomib 和 Cycloheximide 相对 Control 的足迹，并把 turnover-rate 边界说清楚。",
        "The report must state the Bortezomib and Cycloheximide footprints relative to Control together and make the turnover-rate boundary explicit.",
    ),
    "cell_turnoverdynamics_2025_boundary": (
        "当前矩阵支持 abundance footprint；无时间序列 L/H 信息时不能直接给出完整 turnover rate。",
        "The current matrix supports an abundance footprint; without time-resolved heavy/light information a full turnover rate cannot be given directly.",
    ),
    "contrast_bort_high_claim": (
        "高剂量 Bortezomib 足迹",
        "High-dose Bortezomib footprint",
    ),
    "contrast_bort_high_interpretation": (
        "重点看蛋白稳态、蛋白酶体和应激相关变化。",
        "Focuses on protein homeostasis, proteasome and stress-related changes.",
    ),
    "contrast_bort_low_claim": (
        "低剂量 Bortezomib 足迹",
        "Low-dose Bortezomib footprint",
    ),
    "contrast_bort_low_interpretation": (
        "用于剂量趋势和稳健性判断。",
        "Supports the dose trend and robustness checks.",
    ),
    "contrast_chx_high_claim": (
        "高剂量 Cycloheximide 足迹",
        "High-dose Cycloheximide footprint",
    ),
    "contrast_chx_high_interpretation": (
        "重点看翻译、核糖体和短寿命/应激相关变化。",
        "Focuses on translation, ribosome and short-lived/stress-related changes.",
    ),
    "contrast_chx_low_claim": (
        "低剂量 Cycloheximide 足迹",
        "Low-dose Cycloheximide footprint",
    ),
    "contrast_chx_low_interpretation": (
        "用于剂量趋势和与 Bortezomib 的共有/相反变化比较。",
        "Supports the dose trend and the comparison of shared or opposing changes with Bortezomib.",
    ),
    "nat_commun_pispa_2024_story_title": (
        "迁移细胞是否形成 Rho-ERM-收缩-黏附机制链",
        "Do migrated cells form a Rho-ERM-contraction-adhesion mechanism chain?",
    ),
    "nat_commun_pispa_2024_executive_claim": (
        "Cluster-Type 结构、迁移候选蛋白和模块分数共同组织成迁移机制链。",
        "Cluster-Type structure, migration candidate proteins and module scores together form the migration mechanism chain.",
    ),
    "nat_commun_pispa_2024_boundary": (
        "迁移机制主要来自 current_matrix 与 curated module，富集无 FDR 时只作探索。",
        "The migration mechanism comes mainly from the current_matrix and curated modules; enrichment without FDR is exploratory only.",
    ),
    "contrast_cluster1_2_claim": (
        "Cluster 1 相对 Cluster 2 的迁移轴",
        "Migration axis of Cluster 1 versus Cluster 2",
    ),
    "contrast_cluster1_2_interpretation": (
        "迁移细胞富集的 Cluster 1 是主轴。",
        "Cluster 1, which is enriched for migrated cells, is the main axis.",
    ),
    "contrast_cluster1_3_claim": (
        "Cluster 1 相对 Cluster 3 的迁移轴补充",
        "Migration axis of Cluster 1 versus Cluster 3, as a complement",
    ),
    "contrast_cluster1_3_interpretation": (
        "用于确认 Cluster 1 是否相对另一对照群体仍保持迁移机制方向。",
        "Confirms whether Cluster 1 keeps its migration mechanism direction against a second control population.",
    ),
    "contrast_cluster2_3_claim": (
        "Cluster 2/3 对照内部异质性",
        "Heterogeneity inside the Cluster 2/3 controls",
    ),
    "contrast_cluster2_3_interpretation": (
        "用于解释非迁移群体内部差异。",
        "Explains the differences within the non-migrated population.",
    ),
    "contrast_migrated_control_claim": (
        "迁移表型相对对照的总体差异",
        "Overall difference of the migrated phenotype versus control",
    ),
    "contrast_migrated_control_interpretation": (
        "用于把 cluster 结果和迁移/对照表型连接起来。",
        "Links the cluster results to the migrated versus control phenotype.",
    ),
    "nat_methods_ipsc_2025_story_title": (
        "iPSC 到 EB 的多能性下降、谱系/ECM 上升和 EB 内异质性",
        "Falling pluripotency, rising lineage/ECM signal and intra-EB heterogeneity from iPSC to EB",
    ),
    "nat_methods_ipsc_2025_executive_claim": (
        "核心故事是 EB 相对 iPSCs 的多能性边界、谱系候选和模块梯度。",
        "The core story is the pluripotency boundary, lineage candidates and module gradients of EB relative to iPSCs.",
    ),
    "nat_methods_ipsc_2025_boundary": (
        "EB 内部梯度是 current_matrix 模块分布，不等同于真实发育轨迹。",
        "The intra-EB gradient is a current_matrix module distribution and is not equivalent to a true developmental trajectory.",
    ),
    "contrast_eb_ipsc_claim": (
        "EB 相对 iPSCs 的差异主轴",
        "Main differential axis of EB versus iPSCs",
    ),
    "contrast_eb_ipsc_interpretation": (
        "连接多能性下降与谱系/ECM 相关候选上升。",
        "Links the decline in pluripotency to rising lineage/ECM candidates.",
    ),
    "nat_biotech_brain_2026_story_title": (
        "RG/oRG 到 IPC-EN/EN 的发育状态轴和本地 ASD/NDD 注释边界",
        "The RG/oRG to IPC-EN/EN developmental state axis and the local ASD/NDD annotation boundary",
    ),
    "nat_biotech_brain_2026_executive_claim": (
        "报告应围绕 RG/oRG -> IPC-EN -> EN 轨迹、候选蛋白和本地外部注释边界组织。",
        "The report should be organised around the RG/oRG -> IPC-EN -> EN trajectory, the candidate proteins and the local external-annotation boundary.",
    ),
    "nat_biotech_brain_2026_boundary": (
        "ASD/NDD 是 external_annotation，不是 current_matrix 疾病因果结论。",
        "ASD/NDD is external_annotation, not a current_matrix causal disease conclusion.",
    ),
    "contrast_ipcen_org_claim": (
        "IPC-EN 相对 oRG 的过渡轴",
        "Transition axis of IPC-EN versus oRG",
    ),
    "contrast_ipcen_org_interpretation": (
        "连接 progenitor 到 IPC/early neuron 程序。",
        "Links progenitor to IPC/early-neuron programmes.",
    ),
    "contrast_en_ipcen_claim": (
        "EN 相对 IPC-EN 的成熟轴",
        "Maturation axis of EN versus IPC-EN",
    ),
    "contrast_en_ipcen_interpretation": (
        "连接 neuronal maturation、synapse/neurite 与 chromatin/BAF。",
        "Links neuronal maturation and synapse/neurite programmes to chromatin/BAF.",
    ),
    "nat_methods_dvp_2023_story_title": (
        "肝小叶 Portal-Central 分区蛋白程序",
        "Portal-to-Central zonation protein programmes in the liver lobule",
    ),
    "nat_methods_dvp_2023_executive_claim": (
        "核心是 periportal urea cycle 与 central xenobiotic/metabolism 模块的方向性。",
        "The core is the directionality of the periportal urea-cycle and central xenobiotic/metabolism modules.",
    ),
    "nat_methods_dvp_2023_boundary": (
        "Zonation 结论来自当前矩阵标签和模块/候选蛋白，不使用隐藏阈值。",
        "The zonation conclusion comes from the current matrix labels and the module/candidate proteins, with no hidden thresholds.",
    ),
    "contrast_portal_central_claim": (
        "Portal 相对 Central 的分区轴",
        "Zonation axis of Portal versus Central",
    ),
    "contrast_portal_central_interpretation": (
        "对照 urea-cycle 与 xenobiotic metabolism 候选/模块方向。",
        "Cross-checks the direction of urea-cycle and xenobiotic-metabolism candidates and modules.",
    ),
    "nat_methods_pscope_2023_story_title": (
        "LPS 高/低极化 macrophage 的吞噬体和炎症模块",
        "Phagosome and inflammation modules in high- and low-LPS-polarised macrophages",
    ),
    "nat_methods_pscope_2023_executive_claim": (
        "核心是 LPS_high/LPS_low 相对 Untreated 的方向，并区分强弱极化。",
        "The core is the direction of LPS_high/LPS_low relative to Untreated, separating strong from weak polarisation.",
    ),
    "nat_methods_pscope_2023_boundary": (
        "Phagosome/lysosome 结论需要候选蛋白、模块或离线富集支撑。",
        "Phagosome/lysosome conclusions need support from candidate proteins, modules or offline enrichment.",
    ),
    "contrast_lpshigh_base_claim": (
        "LPS high 相对 Untreated",
        "LPS high versus Untreated",
    ),
    "contrast_lpshigh_base_interpretation": (
        "检测高极化和 phagosome maturation 程序。",
        "Tests high polarisation and the phagosome maturation programme.",
    ),
    "contrast_lpslow_base_claim": (
        "LPS low 相对 Untreated",
        "LPS low versus Untreated",
    ),
    "contrast_lpslow_base_interpretation": (
        "用于判断低极化响应是否弱于 LPS_high。",
        "Tests whether the low-polarisation response is weaker than LPS_high.",
    ),
    "science_bloodcell_2025_story_title": (
        "HSPC 状态层级中的干性、代谢和粒细胞颗粒程序",
        "Stemness, metabolic and granulocyte-granule programmes across the HSPC state hierarchy",
    ),
    "science_bloodcell_2025_executive_claim": (
        "报告要把 HSC/MPP/MEP/LMPP/GMP 的候选和模块组织成状态层级，而不是只列群体。",
        "The report should organise the candidates and modules of HSC/MPP/MEP/LMPP/GMP into a state hierarchy rather than listing populations.",
    ),
    "science_bloodcell_2025_boundary": (
        "状态解释来自当前标签和蛋白矩阵，不直接外推造血命运因果。",
        "The state interpretation comes from the current labels and the protein matrix and does not extrapolate directly to causal haematopoietic fate.",
    ),
    "contrast_hsc_gmp_claim": (
        "HSC 相对 GMP 的干性/粒细胞轴",
        "Stemness/granulocyte axis of HSC versus GMP",
    ),
    "contrast_hsc_gmp_interpretation": (
        "对照 TALDO1/H1F0 与 MPO/ELANE 方向。",
        "Cross-checks the direction of TALDO1/H1F0 against MPO/ELANE.",
    ),
    "nat_commun_proteinleakage_2025_story_title": (
        "Permeable 相对 Intact 的蛋白泄漏 QC 与细胞类型边界",
        "Protein-leakage QC of Permeable versus Intact cells and the cell-type boundary",
    ),
    "nat_commun_proteinleakage_2025_executive_claim": (
        "核心是 membrane-status 对比中的 cytosol/nucleus 与 mito/membrane 标记足迹，并报告 Fresh/Frozen/Cluster 边界。",
        "The core is the cytosol/nucleus and mito/membrane marker footprint of the membrane-status contrast, reported together with the Fresh/Frozen/Cluster boundary.",
    ),
    "nat_commun_proteinleakage_2025_boundary": (
        "Leakage 是样本状态/QC 解释，不外推为疾病机制。",
        "Leakage is a sample-state/QC interpretation and is not extrapolated to disease mechanisms.",
    ),
    "contrast_permeable_intact_claim": (
        "Permeable 相对 Intact 的泄漏足迹",
        "Leakage footprint of Permeable versus Intact",
    ),
    "contrast_permeable_intact_interpretation": (
        "检查 cytosol/nucleus 和 mito/membrane 标记是否支持膜破损/泄漏。",
        "Checks whether the cytosol/nucleus and mito/membrane markers support membrane damage and leakage.",
    ),
    "story_title_default": (
        "当前矩阵核心对比故事",
        "the core contrast story of the current matrix",
    ),
    "executive_claim_default": (
        "围绕当前矩阵核心对比组织结论。",
        "Conclusions are organised around the core contrasts of the current matrix.",
    ),
    "boundary_default": (
        "所有结论必须标注 current_matrix/offline_enrichment/external_annotation/extension 边界。",
        "Every conclusion must carry its current_matrix/offline_enrichment/external_annotation/extension boundary.",
    ),
    "contrast_claim_suffix": (
        " 核心对比",
        " core contrast",
    ),
    "contrast_interpretation_generic": (
        "通用核心对比，需保守解释。",
        "Generic core contrast; interpret it conservatively.",
    ),
    "candidate_boundary": (
        "候选蛋白只在该对比差异表中检出时用于当前矩阵结论。",
        "Candidate proteins support current-matrix conclusions only when detected in that contrast's differential table.",
    ),
    "story_kind_contrast": (
        "对比",
        "contrast",
    ),
    "story_kind_candidate": (
        "候选蛋白",
        "candidate protein",
    ),
    "story_kind_module": (
        "模块",
        "module",
    ),
    "module_source_stored": (
        "存储行",
        "stored row",
    ),
    "module_source_reverse_prefix": (
        "由 ",
        "derived from ",
    ),
    "module_source_reverse_suffix": (
        " 取负派生",
        " by negation",
    ),
    "module_source_subtract": (
        "由两行分组均值相减派生",
        "derived by subtracting the two group means",
    ),
    "qc_definition_a": (
        "未检出定义=空值或0；input=原始交付矩阵（过滤/填补之前）；analysed=本轮分析矩阵；",
        "Not-detected definition = empty value or 0; input = the originally delivered matrix (before filtering/imputation); analysed = this run's analysis matrix; ",
    ),
    "qc_definition_b": (
        "analysed 缺失率为 0 时按填补状态判读，不等于全部蛋白被检出；",
        "a missing rate of 0 in analysed is read from the imputation state and does not mean that every protein was detected; ",
    ),
    "qc_definition_c": (
        "imputation_applied 的来源=%s",
        "source of imputation_applied = %s",
    ),
    "qc_definition_source_missing": (
        "运行记录中未找到矩阵变换记录",
        "no matrix transform record found in the run",
    ),
    "mapping_note_a": (
        "该符号未在本次矩阵的基因标签、蛋白组标识、蛋白名称与 UniProt GN= 字段中匹配到，",
        "This symbol was matched in none of the gene labels, protein-group identifiers, protein names or UniProt GN= fields of the current matrix, ",
    ),
    "mapping_note_b": (
        "因此未建立映射；未建立映射不等同于该蛋白不存在。",
        "so no mapping was established; an unmapped symbol does not mean the protein is absent.",
    ),
    "evidence_cell_a": (
        "通过筛选 ",
        "passing the filter: ",
    ),
    "evidence_cell_b": (
        " 个；第一臂较高 ",
        "; higher in the first arm: ",
    ),
    "evidence_cell_c": (
        " 个；第二臂较高 ",
        "; higher in the second arm: ",
    ),
    "evidence_cell_d": (
        " 个；最强上调 ",
        "; strongest up-regulated: ",
    ),
}

register("evidence", _EVIDENCE_TEXT)



# Keys handled by the direct ``t()`` calls below; values that are not in this

# set pass through the lazy mapping unchanged (lists, ids, flags).

_STORY_TEXT_KEYS = frozenset(_EVIDENCE_TEXT)





def _story_text(value):

    """Text of a registry key, or the value itself when it is not one."""

    if isinstance(value, str) and value in _STORY_TEXT_KEYS:

        return t("evidence." + value)

    return value





class _StoryText(dict):

    """Recipe metadata whose text resolves in the active report language.



    Stored values are keys of the ``evidence`` namespace; every read through

    ``mapping[key]`` or ``mapping.get(key)`` returns the wording of the language that

    is active when the evidence pack is written, not the language active at import.

    """



    def __getitem__(self, key):

        return _story_text(dict.__getitem__(self, key))



    def get(self, key, default=None):

        return _story_text(dict.get(self, key, default))


CORE_STORY_RECIPES: Dict[str, Dict[str, Any]] = {
    "Nat_Commun_Carr_2024": {
        "story_title": "nat_commun_carr_2024_story_title",
        "executive_claim": "nat_commun_carr_2024_executive_claim",
        "boundary": "nat_commun_carr_2024_boundary",
        "primary_contrasts": [
            {
                "id": "carr_lps_dmso",
                "display": "LPS_vs_DMSO",
                "group_a": "LPS",
                "group_b": "DMSO",
                "aliases": [{"name": "LPS_vs_DMSO"}, {"name": "DMSO_vs_LPS", "invert": True}],
                "claim": "contrast_carr_lps_dmso_claim",
                "interpretation": "contrast_carr_lps_dmso_interpretation",
                "focus_genes": ["CD44", "DDX21", "STAT1", "IRF1", "NFKB1", "RELA"],
                "modules": ["inflammation_interferon", "proteasome_proteostasis"],
            }
        ],
    },
    "Nat_Commun_Nociceptor_2026": {
        "story_title": "nat_commun_nociceptor_2026_story_title",
        "executive_claim": "nat_commun_nociceptor_2026_executive_claim",
        "boundary": "nat_commun_nociceptor_2026_boundary",
        "primary_contrasts": [
            {"id": "trkA_inflamed", "display": "TrkA_Inflamed_vs_TrkA_Control", "group_a": "TrkA_Inflamed", "group_b": "TrkA_Control", "aliases": [{"name": "TrkA_Inflamed_vs_TrkA_Control"}], "claim": "contrast_trka_inflamed_claim", "interpretation": "contrast_trka_inflamed_interpretation", "focus_genes": ["NTRK1", "TRPV1", "SCN10A", "PIEZO2", "B3GNT2", "RRAD"], "modules": ["nociceptor_sensory_transduction", "nociceptor_inflammation_response", "membrane_glycosylation_traffic"]},
            {"id": "ib4_inflamed", "display": "IB4_Inflamed_vs_IB4_Control", "group_a": "IB4_Inflamed", "group_b": "IB4_Control", "aliases": [{"name": "IB4_Inflamed_vs_IB4_Control"}], "claim": "contrast_ib4_inflamed_claim", "interpretation": "contrast_ib4_inflamed_interpretation", "focus_genes": ["B3GNT2", "RRAD", "SCN10A", "PIEZO2"], "modules": ["nociceptor_inflammation_response", "membrane_glycosylation_traffic"]},
            {"id": "mechano_inflamed", "display": "Mechano_Inflamed_vs_Mechano_Control", "group_a": "Mechano_Inflamed", "group_b": "Mechano_Control", "aliases": [{"name": "Mechano_Inflamed_vs_Mechano_Control"}], "claim": "contrast_mechano_inflamed_claim", "interpretation": "contrast_mechano_inflamed_interpretation", "focus_genes": ["PIEZO2", "RRAD", "B3GNT2"], "modules": ["nociceptor_sensory_transduction", "nociceptor_inflammation_response"]},
        ],
    },
    "Nat_Commun_SCPro_2024": {
        "story_title": "nat_commun_scpro_2024_story_title",
        "executive_claim": "nat_commun_scpro_2024_executive_claim",
        "boundary": "nat_commun_scpro_2024_boundary",
        "primary_contrasts": [
            {"id": "treg_klrg1", "display": "CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg", "group_a": "CD4_CD25pos_Klrg1pos", "group_b": "CD4_CD25pos_Klrg1neg", "aliases": [{"name": "CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg"}, {"name": "CD4_CD25pos_Klrg1neg_vs_CD4_CD25pos_Klrg1pos", "invert": True}], "claim": "contrast_treg_klrg1_claim", "interpretation": "contrast_treg_klrg1_interpretation", "focus_genes": ["KLRG1", "FOXP3", "IL2RA", "CTLA4", "IKZF2", "LAG3", "TIGIT"], "modules": ["treg_klrg1_immune", "ribosome_translation"]},
            {"id": "cd4_klrg1", "display": "CD4_Klrg1pos_vs_CD4_Klrg1neg", "group_a": "CD4_Klrg1pos", "group_b": "CD4_Klrg1neg", "aliases": [{"name": "CD4_Klrg1pos_vs_CD4_Klrg1neg"}, {"name": "CD4_Klrg1neg_vs_CD4_Klrg1pos", "invert": True}], "claim": "contrast_cd4_klrg1_claim", "interpretation": "contrast_cd4_klrg1_interpretation", "focus_genes": ["KLRG1", "IL2RA", "CTLA4"], "modules": ["treg_klrg1_immune", "ribosome_translation"]},
            {"id": "cd8_klrg1", "display": "CD8_Klrg1pos_vs_CD8_Klrg1neg", "group_a": "CD8_Klrg1pos", "group_b": "CD8_Klrg1neg", "aliases": [{"name": "CD8_Klrg1pos_vs_CD8_Klrg1neg"}, {"name": "CD8_Klrg1neg_vs_CD8_Klrg1pos", "invert": True}], "claim": "contrast_cd8_klrg1_claim", "interpretation": "contrast_cd8_klrg1_interpretation", "focus_genes": ["KLRG1", "LAG3", "TIGIT"], "modules": ["treg_klrg1_immune", "ribosome_translation"]},
        ],
    },
    "Cell_TurnoverDynamics_2025": {
        "story_title": "cell_turnoverdynamics_2025_story_title",
        "executive_claim": "cell_turnoverdynamics_2025_executive_claim",
        "boundary": "cell_turnoverdynamics_2025_boundary",
        "primary_contrasts": [
            {"id": "bort_high", "display": "Bortezomib_High_vs_Control", "group_a": "Bortezomib_High", "group_b": "Control", "aliases": [{"name": "Bortezomib_High_vs_Control"}, {"name": "Control_vs_Bortezomib_High", "invert": True}], "claim": "contrast_bort_high_claim", "interpretation": "contrast_bort_high_interpretation", "focus_genes": ["PSMA1", "PSMB1", "PSMD1", "HSPA1A", "HSP90AA1"], "modules": ["proteasome_proteostasis"]},
            {"id": "bort_low", "display": "Bortezomib_Low_vs_Control", "group_a": "Bortezomib_Low", "group_b": "Control", "aliases": [{"name": "Bortezomib_Low_vs_Control"}, {"name": "Control_vs_Bortezomib_Low", "invert": True}], "claim": "contrast_bort_low_claim", "interpretation": "contrast_bort_low_interpretation", "focus_genes": ["PSMA1", "PSMB1", "PSMD1"], "modules": ["proteasome_proteostasis"]},
            {"id": "chx_high", "display": "Cycloheximide_High_vs_Control", "group_a": "Cycloheximide_High", "group_b": "Control", "aliases": [{"name": "Cycloheximide_High_vs_Control"}, {"name": "Control_vs_Cycloheximide_High", "invert": True}], "claim": "contrast_chx_high_claim", "interpretation": "contrast_chx_high_interpretation", "focus_genes": ["RPLP0", "RPS3", "EIF4A1", "EEF1A1"], "modules": ["ribosome_translation"]},
            {"id": "chx_low", "display": "Cycloheximide_Low_vs_Control", "group_a": "Cycloheximide_Low", "group_b": "Control", "aliases": [{"name": "Cycloheximide_Low_vs_Control"}, {"name": "Control_vs_Cycloheximide_Low", "invert": True}], "claim": "contrast_chx_low_claim", "interpretation": "contrast_chx_low_interpretation", "focus_genes": ["RPLP0", "RPS3", "EIF4A1"], "modules": ["ribosome_translation"]},
        ],
    },
    "Nat_Commun_PiSPA_2024": {
        "story_title": "nat_commun_pispa_2024_story_title",
        "executive_claim": "nat_commun_pispa_2024_executive_claim",
        "boundary": "nat_commun_pispa_2024_boundary",
        "primary_contrasts": [
            {"id": "cluster1_2", "display": "Cluster_1_vs_Cluster_2", "group_a": "Cluster 1", "group_b": "Cluster 2", "aliases": [{"name": "Cluster_1_vs_Cluster_2"}, {"name": "Cluster 1_vs_Cluster 2"}], "claim": "contrast_cluster1_2_claim", "interpretation": "contrast_cluster1_2_interpretation", "focus_genes": ["EZR", "MSN", "MYL9", "CDC42", "RAC1", "RHOA", "TLN1", "VCL"], "modules": ["migration_signature", "rho_gtpase_migration", "erm_membrane_cortex", "myosin_contractility", "talin_vinculin_focal_adhesion"]},
            {"id": "cluster1_3", "display": "Cluster_1_vs_Cluster_3", "group_a": "Cluster 1", "group_b": "Cluster 3", "aliases": [{"name": "Cluster_1_vs_Cluster_3"}, {"name": "Cluster 1_vs_Cluster 3"}], "claim": "contrast_cluster1_3_claim", "interpretation": "contrast_cluster1_3_interpretation", "focus_genes": ["EZR", "MSN", "MYL9", "CDC42", "RAC1", "RHOA", "TLN1", "VCL", "FLNA", "ACTN1", "ITGB1"], "modules": ["migration_signature", "rho_gtpase_migration", "erm_membrane_cortex", "myosin_contractility", "talin_vinculin_focal_adhesion"]},
            {"id": "cluster2_3", "display": "Cluster_2_vs_Cluster_3", "group_a": "Cluster 2", "group_b": "Cluster 3", "aliases": [{"name": "Cluster_2_vs_Cluster_3"}, {"name": "Cluster 2_vs_Cluster 3"}], "claim": "contrast_cluster2_3_claim", "interpretation": "contrast_cluster2_3_interpretation", "focus_genes": ["EZR", "MSN", "MYL9", "TLN1", "VCL"], "modules": ["migration_signature", "talin_vinculin_focal_adhesion"]},
            {"id": "migrated_control", "display": "Migrated_vs_Control", "group_a": "Migrated", "group_b": "Control", "aliases": [{"name": "Migrated_vs_Control"}, {"name": "Control_vs_Migrated", "invert": True}, {"name": "Migrated_vs_non-migrated"}], "claim": "contrast_migrated_control_claim", "interpretation": "contrast_migrated_control_interpretation", "focus_genes": ["EZR", "MSN", "MYL9", "CDC42", "RAC1", "RHOA", "TLN1", "VCL", "FLNA", "ACTN1", "ITGB1"], "modules": ["migration_signature", "rho_gtpase_migration", "erm_membrane_cortex", "myosin_contractility", "talin_vinculin_focal_adhesion"]},
        ],
    },
    "Nat_Methods_iPSC_2025": {
        "story_title": "nat_methods_ipsc_2025_story_title",
        "executive_claim": "nat_methods_ipsc_2025_executive_claim",
        "boundary": "nat_methods_ipsc_2025_boundary",
        "primary_contrasts": [
            {"id": "eb_ipsc", "display": "EB_vs_iPSCs", "group_a": "EB", "group_b": "iPSCs", "aliases": [{"name": "EB_vs_iPSCs"}, {"name": "iPSCs_vs_EB", "invert": True}], "claim": "contrast_eb_ipsc_claim", "interpretation": "contrast_eb_ipsc_interpretation", "focus_genes": ["POU5F1", "SOX2", "LIN28A", "NANOG", "GATA4", "HAND1", "MAP2", "FN1", "COL1A1"], "modules": ["pluripotency_core", "lineage_endoderm", "lineage_mesoderm", "lineage_ectoderm", "ecm_adhesion"]},
        ],
    },
    "Nat_Biotech_Brain_2026": {
        "story_title": "nat_biotech_brain_2026_story_title",
        "executive_claim": "nat_biotech_brain_2026_executive_claim",
        "boundary": "nat_biotech_brain_2026_boundary",
        "primary_contrasts": [
            {"id": "ipcen_org", "display": "IPC-EN_vs_oRG", "group_a": "IPC-EN", "group_b": "oRG", "aliases": [{"name": "IPC-EN_vs_oRG"}, {"name": "oRG_vs_IPC-EN", "invert": True}], "claim": "contrast_ipcen_org_claim", "interpretation": "contrast_ipcen_org_interpretation", "focus_genes": ["EOMES", "TBR1", "BCL11B", "NEUROD2", "MAP2"], "modules": ["brain_ipc_en_transition", "brain_en_maturation", "brain_chromatin_baf"]},
            {"id": "en_ipcen", "display": "EN_vs_IPC-EN", "group_a": "EN", "group_b": "IPC-EN", "aliases": [{"name": "EN_vs_IPC-EN"}, {"name": "IPC-EN_vs_EN", "invert": True}], "claim": "contrast_en_ipcen_claim", "interpretation": "contrast_en_ipcen_interpretation", "focus_genes": ["TBR1", "BCL11B", "NEUROD2", "MAP2", "SYN1"], "modules": ["brain_en_maturation", "brain_synapse_neurite", "brain_chromatin_baf"]},
        ],
    },
    "Nat_Methods_DVP_2023": {
        "story_title": "nat_methods_dvp_2023_story_title",
        "executive_claim": "nat_methods_dvp_2023_executive_claim",
        "boundary": "nat_methods_dvp_2023_boundary",
        "primary_contrasts": [
            {"id": "portal_central", "display": "Portal_vs_Central", "group_a": "Portal", "group_b": "Central", "aliases": [{"name": "Portal_vs_Central"}, {"name": "Central_vs_Portal", "invert": True}], "claim": "contrast_portal_central_claim", "interpretation": "contrast_portal_central_interpretation", "focus_genes": ["ARG1", "ASS1", "CPS1", "GLUL", "CYP2E1", "CYP1A2"], "modules": ["liver_periportal_urea", "liver_central_xenobiotic"]},
        ],
    },
    "Nat_Methods_pSCoPE_2023": {
        "story_title": "nat_methods_pscope_2023_story_title",
        "executive_claim": "nat_methods_pscope_2023_executive_claim",
        "boundary": "nat_methods_pscope_2023_boundary",
        "primary_contrasts": [
            {"id": "lpshigh_base", "display": "LPS_high_vs_Untreated", "group_a": "LPS_high", "group_b": "Untreated", "aliases": [{"name": "LPS_high_vs_Untreated"}, {"name": "Untreated_vs_LPS_high", "invert": True}], "claim": "contrast_lpshigh_base_claim", "interpretation": "contrast_lpshigh_base_interpretation", "focus_genes": ["ATP6V0A1", "ATP6V1A", "LAMP1", "LAMP2", "CTSB", "CTSD"], "modules": ["phagosome_vatpase_lysosome", "inflammation_interferon"]},
            {"id": "lpslow_base", "display": "LPS_low_vs_Untreated", "group_a": "LPS_low", "group_b": "Untreated", "aliases": [{"name": "LPS_low_vs_Untreated"}, {"name": "Untreated_vs_LPS_low", "invert": True}], "claim": "contrast_lpslow_base_claim", "interpretation": "contrast_lpslow_base_interpretation", "focus_genes": ["ATP6V0A1", "LAMP1", "CTSB"], "modules": ["phagosome_vatpase_lysosome", "inflammation_interferon"]},
        ],
    },
    "Science_BloodCell_2025": {
        "story_title": "science_bloodcell_2025_story_title",
        "executive_claim": "science_bloodcell_2025_executive_claim",
        "boundary": "science_bloodcell_2025_boundary",
        "primary_contrasts": [
            {"id": "hsc_gmp", "display": "HSC_vs_GMP", "group_a": "HSC", "group_b": "GMP", "aliases": [{"name": "HSC_vs_GMP"}, {"name": "GMP_vs_HSC", "invert": True}], "claim": "contrast_hsc_gmp_claim", "interpretation": "contrast_hsc_gmp_interpretation", "focus_genes": ["TALDO1", "H1F0", "MPO", "ELANE"], "modules": ["hsc_maintenance_ppp_chromatin", "granulocyte_granule"]},
        ],
    },
    "Nat_Commun_ProteinLeakage_2025": {
        "story_title": "nat_commun_proteinleakage_2025_story_title",
        "executive_claim": "nat_commun_proteinleakage_2025_executive_claim",
        "boundary": "nat_commun_proteinleakage_2025_boundary",
        "primary_contrasts": [
            {"id": "permeable_intact", "display": "Permeable_vs_Intact", "group_a": "Permeable", "group_b": "Intact", "aliases": [{"name": "Permeable_vs_Intact"}, {"name": "Intact_vs_Permeable", "invert": True}], "claim": "contrast_permeable_intact_claim", "interpretation": "contrast_permeable_intact_interpretation", "focus_genes": ["GAPDH", "LDHA", "TUBA1B", "LMNB1", "VDAC1", "ATP5F1A"], "modules": ["protein_leakage_cytosol_nucleus", "protein_leakage_mito_membrane"]},
        ],
    },
}


# Every text value above is a key of the ``evidence`` registry namespace; these
# wrappers resolve it in the active report language on each read.
CORE_STORY_RECIPES = {
    dataset: _StoryText(recipe) for dataset, recipe in CORE_STORY_RECIPES.items()
}
for _recipe in CORE_STORY_RECIPES.values():
    _recipe["primary_contrasts"] = [
        _StoryText(spec) for spec in _recipe.get("primary_contrasts", [])
    ]


def read_text(path: str | os.PathLike[str]) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def scoring_resource_dir(project_dir: Path) -> Path:
    return project_dir / "kb_source" / "scoring_resources"


def load_local_curated_modules(project_dir: Path) -> Dict[str, List[str]]:
    path = scoring_resource_dir(project_dir) / "curated_modules_v20260518.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    modules = data.get("modules", {}) if isinstance(data, dict) else {}
    cleaned: Dict[str, List[str]] = {}
    for name, genes in modules.items():
        if isinstance(genes, list):
            cleaned[str(name)] = [str(g).strip().upper() for g in genes if str(g).strip()]
    return cleaned


def load_local_asd_ndd_annotations(project_dir: Path) -> pd.DataFrame:
    path = scoring_resource_dir(project_dir) / "asd_ndd_risk_genes.tsv"
    if not path.exists():
        return pd.DataFrame(columns=["gene", "category", "source_note"])
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame(columns=["gene", "category", "source_note"])
    if "gene" not in df.columns:
        return pd.DataFrame(columns=["gene", "category", "source_note"])
    df["gene"] = df["gene"].astype(str).str.upper().str.strip()
    if "category" not in df.columns:
        df["category"] = "ASD_NDD_local_seed"
    if "source_note" not in df.columns:
        df["source_note"] = "offline curated seed for evaluation annotation"
    return df[["gene", "category", "source_note"]].drop_duplicates()


def safe_json_dump(data: Any, path: str | os.PathLike[str]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(long_fs_path(p), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    return str(path)


def safe_csv_write(df: pd.DataFrame, path: str | os.PathLike[str]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(long_fs_path(p), index=False, encoding="utf-8-sig")
    return str(path)


# ---------------------------------------------------------------------------
# F3 统计量精度策略（2026-09-14 修复）
# ---------------------------------------------------------------------------
# 缺陷：本模块曾对统计量使用 round(value, 6) 的“绝对小数位”舍入，把 <5e-7 的 p 值压成 0.0。
# 实测（冻结 run 的源差异表）：Science_BloodCell_2025 的 TALDO1 P.Value=4.830e-09、
# adj.P.Val=4.057e-07 被写成 0.0/0.0；MPO P.Value=1.641e-08、adj.P.Val=9.983e-07 被写成 0.0/1e-06；
# Nat_Commun_ProteinLeakage_2025 的 GAPDH P.Value=3.077e-47 被写成 0.0。报告因此出现统计学上
# 不可能的 “P=0”，并与同一 run 中按 %.3g 输出的对照行自相矛盾。
#
# 现行规则把三种情况分开处理，禁止互相顶替：
#   1) 源值为 0：源差异表里本来就是 0.0 时保持 0；证据层不做“恢复”，不补造 1e-300 之类的小值。
#   2) 计算下溢：上游统计（limma 等）已经下溢成 0.0 时，证据层无法从 0 反推真值，因此同样按 0 记录，
#      由解释层写成“源表即为 0（无法区分真值 0 与上游下溢）”，绝不虚构非零值。
#   3) 展示舍入：只在展示层用有效数字/科学计数法（%.6g），禁止绝对小数位 round()，
#      因此展示层不可能把非零源值显示成 0。
# 落地方式：机器可读 JSON 保存源统计量全精度(float)；CSV 与请求内证据表走 stat_display()。
# 判定/筛选逻辑（例如 confidence 的 fdr <= 0.05）始终使用未格式化的原始数值，不受展示格式影响。
STAT_DISPLAY_SIG = 6
STAT_DISPLAY_COLUMNS = ("logFC", "logFC_display", "P.Value", "adj.P.Val")


def stat_display(value: Any, sig: int = STAT_DISPLAY_SIG) -> Optional[str]:
    """展示层统计量格式化：有效数字/科学计数法；缺失返回 None（写空单元格，绝不写 0）。

    - 源值本身为 0.0（含 -0.0）时返回 "0"：保持 0，也不把下溢伪装成有信息的极小值；
    - None / NaN / 非数值返回 None，由调用方写成空值；
    - 其它值一律用 f"{num:.{sig}g}"：4.83e-09 -> "4.83e-09"，4.057e-07 -> "4.057e-07"，
      0.0123 -> "0.0123"，0.5 -> "0.5"。
    """
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(num):
        return None
    if num == 0.0:
        return "0"
    return f"{num:.{sig}g}"


def stat_display_frame(rows: List[Dict[str, Any]], columns: Iterable[str] = STAT_DISPLAY_COLUMNS) -> pd.DataFrame:
    """CSV 用展示表：只把统计量列换成有效数字文本，其余列原样保留。"""
    frame = pd.DataFrame(rows)
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].map(stat_display)
    return frame


def long_fs_path(path: str | os.PathLike[str]) -> str:
    p = Path(path)
    text = str(p.resolve())
    if os.name == "nt" and not text.startswith("\\\\?\\") and len(text) >= 240:
        return "\\\\?\\" + text
    return str(path)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def normalize_gene(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def split_gene_cell(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    tokens = re.split(r"[;,|/\s]+", str(value))
    return [t.strip() for t in tokens if t.strip()]


def first_existing(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    return None


def resolve_optional_path(path: str | None, project_dir: Optional[Path] = None) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    if not p.is_absolute() and project_dir:
        q = project_dir / p
        if q.exists():
            return str(q)
    return str(p)


def dataset_name_from_config(config: Dict[str, Any]) -> str:
    input_folder = config.get("input_folder") or config.get("input_file_path")
    if input_folder:
        return Path(str(input_folder)).name
    sampleinfo = config.get("sampleinfo_path", "")
    return Path(str(sampleinfo)).parent.name if sampleinfo else "dataset"


def parse_dimensions(grading_text: str) -> List[Dict[str, Any]]:
    dims = []
    current: Optional[Dict[str, Any]] = None
    for line in grading_text.splitlines():
        stripped = line.strip()
        m = re.match(r"^(维度\s*\d+|Dimension\s*\d+)[：:]\s*(.+?)(?:[（(]\s*([0-9.]+)\s*分?\s*[)）])?$", stripped, re.I)
        if m:
            current = {
                "id": re.sub(r"\s+", "", m.group(1)),
                "title": m.group(2).strip(),
                "points": m.group(3),
                "requirements": [],
            }
            dims.append(current)
            continue
        if current and stripped.startswith("-"):
            current["requirements"].append(stripped.lstrip("- ").strip())
    return dims


def extract_candidate_tokens(text: str) -> List[str]:
    found = []
    for token in GENE_TOKEN_RE.findall(text):
        token = token.strip().upper()
        if token in TOKEN_STOPWORDS:
            continue
        if len(token) <= 2 and token not in {"EZR", "MSN"}:
            continue
        found.append(token)
    return found


def infer_required_modules(dataset_name: str, text: str) -> List[str]:
    modules = list(DATASET_MODULES.get(dataset_name, []))
    lowered = text.lower()
    keyword_map = [
        ("interferon", "inflammation_interferon"),
        ("nf-kb", "inflammation_interferon"),
        ("炎症", "inflammation_interferon"),
        ("migration", "migration_adhesion_cytoskeleton"),
        ("adhesion", "migration_adhesion_cytoskeleton"),
        ("迁移", "migration_adhesion_cytoskeleton"),
        ("骨架", "migration_adhesion_cytoskeleton"),
        ("尿素", "liver_periportal_urea"),
        ("drug metabolism", "liver_central_xenobiotic"),
        ("泄漏", "protein_leakage_cytosol_nucleus"),
        ("permeable", "protein_leakage_cytosol_nucleus"),
        ("klrg1", "treg_klrg1_immune"),
        ("hsc", "hsc_maintenance_ppp_chromatin"),
        ("phagosome", "phagosome_vatpase_lysosome"),
        ("v-atpase", "phagosome_vatpase_lysosome"),
        ("ipsc", "pluripotency"),
        ("pluripot", "pluripotency_core"),
        ("lineage", "lineage_mesoderm"),
        ("embryoid", "lineage_endoderm"),
        ("migration signature", "migration_signature"),
        ("rho", "rho_gtpase_migration"),
        ("talin", "talin_vinculin_focal_adhesion"),
        ("ribosome", "ribosome_translation"),
        ("translation", "ribosome_translation"),
        ("proteasome", "proteasome_proteostasis"),
        ("brain", "brain_neuronal_maturation"),
        ("org", "brain_rg_org_progenitor"),
        ("ipc", "brain_ipc_en_transition"),
        ("synapse", "brain_synapse_neurite"),
        ("chromatin", "brain_chromatin_baf"),
        ("asd", "brain_chromatin_baf"),
        ("ndd", "brain_chromatin_baf"),
        ("neur", "brain_neuronal_maturation"),
    ]
    for keyword, module in keyword_map:
        if keyword in lowered and module not in modules:
            modules.append(module)
    return modules


def extract_evaluation_requirements(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = dataset_name_from_config(config)
    project_dir = Path(config.get("project_dir", Path.cwd()))
    grading_text = read_text(config.get("grading_standard_path", ""))
    ground_truth_text = read_text(config.get("ground_truth_path", ""))
    combined = f"{grading_text}\n{ground_truth_text}"
    candidates = list(CURATED_CANDIDATES.get(dataset_name, []))
    for token in extract_candidate_tokens(combined):
        if token not in candidates:
            candidates.append(token)
    modules = infer_required_modules(dataset_name, combined)
    module_sources = dict(CURATED_MODULES)
    module_sources.update(load_local_curated_modules(project_dir))
    return {
        "dataset": dataset_name,
        "source_files": {
            "grading_standard": config.get("grading_standard_path", ""),
            "ground_truth": config.get("ground_truth_path", ""),
        },
        "candidate_proteins": candidates[:80],
        "curated_modules": [
            {"name": module, "genes": module_sources[module]}
            for module in modules
            if module in module_sources
        ],
        "local_scoring_resources": {
            "resource_dir": str(scoring_resource_dir(project_dir)),
            "curated_modules": str(scoring_resource_dir(project_dir) / "curated_modules_v20260518.json"),
            "asd_ndd_risk_genes": str(scoring_resource_dir(project_dir) / "asd_ndd_risk_genes.tsv"),
        },
        "required_tables": [
            "group_composition_qc",
            "core_story_evidence",
            "candidate_protein_evidence",
            "curated_module_scores",
            "dataset_recipe_evidence",
            "scoring_standard_coverage",
        ],
        "scoring_dimensions": parse_dimensions(grading_text),
    }


def extract_user_visible_requirements(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build report-generation requirements without evaluator-only standards.

    This is the publication-safe path. It may use current matrix metadata,
    user_input, analysis design, dataset recipes, and local versioned resources,
    but it must not read ground_truth.txt or grading_standard.txt.
    """
    dataset_name = dataset_name_from_config(config)
    project_dir = Path(config.get("project_dir", Path.cwd()))
    user_input_text = read_text(config.get("user_input_path", ""))
    analysis_design = config.get("analysis_design", {}) or {}
    analysis_design_text = json.dumps(analysis_design, ensure_ascii=False)
    combined_visible_text = f"{dataset_name}\n{user_input_text}\n{analysis_design_text}"
    candidates = list(CURATED_CANDIDATES.get(dataset_name, []))
    modules = infer_required_modules(dataset_name, combined_visible_text)
    module_sources = dict(CURATED_MODULES)
    module_sources.update(load_local_curated_modules(project_dir))
    return {
        "dataset": dataset_name,
        "evidence_scope": "user_visible",
        "source_files": {
            "sampleinfo": config.get("sampleinfo_path", ""),
            "protein_quant": config.get("protein_quant_path", ""),
            "user_input": config.get("user_input_path", ""),
            "analysis_design": config.get("analysis_design_path", ""),
            "analysis_design_inferred": config.get("analysis_design_inferred_path", ""),
        },
        "evaluator_only_sources": [],
        "candidate_proteins": candidates[:80],
        "curated_modules": [
            {"name": module, "genes": module_sources[module]}
            for module in modules
            if module in module_sources
        ],
        "local_scoring_resources": {
            "resource_dir": str(scoring_resource_dir(project_dir)),
            "curated_modules": str(scoring_resource_dir(project_dir) / "curated_modules_v20260518.json"),
            "asd_ndd_risk_genes": str(scoring_resource_dir(project_dir) / "asd_ndd_risk_genes.tsv"),
        },
        "required_tables": [
            "group_composition_qc",
            "core_story_evidence",
            "candidate_protein_evidence",
            "curated_module_scores",
            "dataset_recipe_evidence",
            "scoring_standard_coverage",
        ],
        "scoring_dimensions": [],
    }


def load_sampleinfo(config: Dict[str, Any]) -> pd.DataFrame:
    path = config.get("sampleinfo_path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"SampleInfo.csv not found: {path}")
    return pd.read_csv(path)


def sample_id_col(sampleinfo: pd.DataFrame, design: Dict[str, Any]) -> str:
    design_col = design.get("sample_id_col")
    if design_col in sampleinfo.columns:
        return str(design_col)
    for col in ["FileName", "R.FileName", "Sample", "SampleID", "sample", "sample_id"]:
        if col in sampleinfo.columns:
            return col
    return str(sampleinfo.columns[0])


def group_col(sampleinfo: pd.DataFrame, design: Dict[str, Any]) -> Optional[str]:
    design_col = design.get("group_col")
    if design_col in sampleinfo.columns:
        return str(design_col)
    for col in ["MembraneStatus", "Type1", "Cluster", "Group", "Condition", "Treatment", "Type", "CellType", "State", "Phenotype", "Label"]:
        if col in sampleinfo.columns and 1 < sampleinfo[col].dropna().nunique() < len(sampleinfo):
            return col
    return None


def normalized_column_lookup(cols: Iterable[str]) -> Dict[str, str]:
    return {re.sub(r"[^a-z0-9]", "", str(col).lower()): str(col) for col in cols}


def first_existing_fuzzy(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    direct = first_existing(cols, candidates)
    if direct:
        return direct
    lookup = normalized_column_lookup(cols)
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", str(candidate).lower())
        if key in lookup:
            return lookup[key]
    return None


def protein_gene_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    protein_col = first_existing_fuzzy(
        df.columns,
        ["PG.ProteinGroups", "Protein.Group", "ProteinGroups", "ProteinID", "Protein.Ids", "PG.ProteinNames", "protein"],
    )
    gene_col = first_existing_fuzzy(
        df.columns,
        ["PG.Genes", "Genes", "Gene", "gene", "Gene.Symbol", "GeneSymbol", "Symbol"],
    )
    return protein_col or str(df.columns[0]), gene_col


def load_protein_gene_map(path: str | os.PathLike[str]) -> pd.DataFrame:
    preview = pd.read_csv(path, nrows=0)
    protein_col, gene_col = protein_gene_columns(preview)
    usecols = [col for col in [protein_col, gene_col] if col and col in preview.columns]
    if len(usecols) < 2:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols)


def attach_gene_annotations(df: pd.DataFrame, config: Dict[str, Any], project_dir: Optional[Path] = None) -> pd.DataFrame:
    protein_col, gene_col = protein_gene_columns(df)
    if gene_col or protein_col not in df.columns:
        return df

    candidate_maps: List[Optional[str]] = []
    try:
        params_path = config.get("parameters_path")
        if params_path and os.path.exists(params_path):
            params = json.loads(Path(params_path).read_text(encoding="utf-8", errors="ignore"))
            files = params.get("combat_calibration", {}).get("files", {})
            candidate_maps.append(files.get("protein_gene_map"))
    except Exception:
        pass

    candidate_maps.extend([
        str(Path(config.get("this_run_folder_path", ".")) / "processed_proteins" / "Protein_Gene_Map.csv"),
        config.get("protein_quant_path"),
    ])

    lookup: Dict[str, str] = {}
    for raw_path in candidate_maps:
        path = resolve_optional_path(raw_path, project_dir)
        if not path or not os.path.exists(path):
            continue
        try:
            gene_map = load_protein_gene_map(path)
        except Exception:
            continue
        map_protein_col, map_gene_col = protein_gene_columns(gene_map)
        if map_protein_col not in gene_map.columns or not map_gene_col:
            continue
        tmp = gene_map.loc[:, [map_protein_col, map_gene_col]].copy()
        tmp.columns = ["_protein", "_gene"]
        tmp["_protein"] = tmp["_protein"].astype(str).str.split(";").str[0]
        tmp["_gene"] = tmp["_gene"].astype(str)
        lookup.update(tmp.drop_duplicates("_protein").set_index("_protein")["_gene"].to_dict())
        if lookup:
            break

    if not lookup:
        return df

    out = df.copy()
    canonical = out[protein_col].astype(str).str.split(";").str[0]
    out["PG.Genes"] = canonical.map(lookup).fillna("")
    return out


def load_matrix(config: Dict[str, Any], project_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    paths = []
    try:
        params_path = config.get("parameters_path")
        if params_path and os.path.exists(params_path):
            params = json.loads(Path(params_path).read_text(encoding="utf-8", errors="ignore"))
            files = params.get("combat_calibration", {}).get("files", {})
            paths.extend([files.get("combat_matrix"), files.get("normalized_matrix"), files.get("filtered_protein_quant")])
    except Exception:
        pass
    paths.extend([
        config.get("protein_quant_combat_path"),
        config.get("protein_quant_path"),
    ])
    for raw_path in paths:
        path = resolve_optional_path(raw_path, project_dir)
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            return attach_gene_annotations(df, config, project_dir), path
    raise FileNotFoundError("No protein quantification matrix could be loaded.")


def sample_columns_in_matrix(matrix: pd.DataFrame, sampleinfo: pd.DataFrame, sample_col: str) -> List[str]:
    samples = sampleinfo[sample_col].dropna().astype(str).tolist()
    return [s for s in samples if s in matrix.columns]


def compute_detected_and_missing(matrix_row: pd.Series, sample_cols: List[str]) -> Tuple[bool, float]:
    if not sample_cols:
        return True, float("nan")
    values = pd.to_numeric(matrix_row[sample_cols], errors="coerce")
    missing = values.isna() | (values == 0)
    return bool((~missing).any()), round(float(missing.mean()), 4)


def build_group_composition_qc(config: Dict[str, Any], requirements: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    sampleinfo = load_sampleinfo(config)
    design = config.get("analysis_design", {}) if isinstance(config.get("analysis_design"), dict) else {}
    s_col = sample_id_col(sampleinfo, design)
    g_col = group_col(sampleinfo, design)
    batch_col = design.get("batch_col") if design.get("batch_col") in sampleinfo.columns else first_existing(sampleinfo.columns, ["Batch", "batch", "Date", "Run", "MS_Run", "Plate", "Donor"])
    matrix, matrix_path = load_matrix(config, Path(config.get("project_dir", Path.cwd())))
    sample_cols = sample_columns_in_matrix(matrix, sampleinfo, s_col)
    protein_col, gene_col = protein_gene_columns(matrix)
    numeric = matrix[sample_cols].apply(pd.to_numeric, errors="coerce") if sample_cols else pd.DataFrame()
    detected_per_sample = (numeric.notna() & (numeric != 0)).sum(axis=0) if not numeric.empty else pd.Series(dtype=float)
    missing_per_sample = (numeric.isna() | (numeric == 0)).mean(axis=0) if not numeric.empty else pd.Series(dtype=float)

    per_sample = sampleinfo.copy()
    per_sample["_sample_id"] = per_sample[s_col].astype(str)
    per_sample["detected_proteins"] = per_sample["_sample_id"].map(detected_per_sample.to_dict())
    per_sample["missing_rate"] = per_sample["_sample_id"].map(missing_per_sample.to_dict())

    # input matrix (as delivered) so the report never mixes it up with the analysed matrix
    input_info: Dict[str, Any] = {"path": "", "n_proteins": None, "n_matched_samples": None}
    try:
        input_path = resolve_optional_path(config.get("protein_quant_path"), Path(config.get("project_dir", Path.cwd())))
        if input_path and os.path.exists(input_path):
            in_df = pd.read_csv(input_path)
            in_cols = sample_columns_in_matrix(in_df, sampleinfo, s_col)
            if in_cols:
                in_num = in_df[in_cols].apply(pd.to_numeric, errors="coerce")
                in_detected = (in_num.notna() & (in_num != 0)).sum(axis=0)
                in_missing = (in_num.isna() | (in_num == 0)).mean(axis=0)
                per_sample["input_detected_proteins"] = per_sample["_sample_id"].map(in_detected.to_dict())
                per_sample["input_missing_rate"] = per_sample["_sample_id"].map(in_missing.to_dict())
                input_info = {"path": str(input_path), "n_proteins": int(in_df.shape[0]),
                              "n_matched_samples": int(len(in_cols))}
    except Exception:
        pass

    try:
        params_path = config.get("parameters_path")
        params = json.loads(Path(params_path).read_text(encoding="utf-8", errors="ignore")) if params_path else {}
    except Exception:
        params = {}
    # R28: the analysed matrix is imputed by the calibration step, so this flag has to come from what
    # that step recorded. An absent record leaves the flag unknown rather than False.
    transform_record: Dict[str, Any] = {}
    transform_source = ""
    try:
        params_path = config.get("parameters_path")
        if params_path:
            record_path = Path(params_path).parent / "processed_proteins" / "matrix_transform_record.json"
            if record_path.exists():
                transform_record = json.loads(record_path.read_text(encoding="utf-8", errors="ignore"))
                transform_source = "processed_proteins/matrix_transform_record.json"
    except Exception:
        transform_record = {}
    transform_imputation = transform_record.get("imputation") if isinstance(transform_record, dict) else None
    if isinstance(transform_imputation, dict) and "applied" in transform_imputation:
        imputation_applied = bool(transform_imputation.get("applied"))
    else:
        imputation_applied = bool(params.get("imputation_applied")) or bool(
            (config.get("analysis_design") or {}).get("matrix", {}).get("imputation_applied"))
    qc_definition = (
        t("evidence.qc_definition_a")
        + t("evidence.qc_definition_b")
        + t("evidence.qc_definition_c")
        % (transform_source or t("evidence.qc_definition_source_missing"))
    )

    group_cols = [c for c in [g_col, batch_col, "Type", "Type1", "Type2", "Condition", "Treatment", "Cluster", "Label", "Donor"] if c and c in per_sample.columns]
    group_cols = list(dict.fromkeys(group_cols))
    rows = []
    if g_col:
        for value, sub in per_sample.groupby(g_col, dropna=False):
            rows.append({
                "table": "primary_group",
                "group_col": g_col,
                "group": str(value),
                "n_samples": int(sub.shape[0]),
                "mean_detected_proteins": round(float(sub["detected_proteins"].mean()), 4) if sub["detected_proteins"].notna().any() else None,
                "mean_missing_rate": round(float(sub["missing_rate"].mean()), 4) if sub["missing_rate"].notna().any() else None,
                "input_mean_detected_proteins": round(float(sub["input_detected_proteins"].mean()), 4) if "input_detected_proteins" in sub and sub["input_detected_proteins"].notna().any() else None,
                "input_mean_missing_rate": round(float(sub["input_missing_rate"].mean()), 4) if "input_missing_rate" in sub and sub["input_missing_rate"].notna().any() else None,
                "analysed_mean_detected_proteins": round(float(sub["detected_proteins"].mean()), 4) if sub["detected_proteins"].notna().any() else None,
                "analysed_mean_missing_rate": round(float(sub["missing_rate"].mean()), 4) if sub["missing_rate"].notna().any() else None,
                "input_n_proteins": input_info.get("n_proteins"),
                "analysed_n_proteins": int(matrix.shape[0]),
                "imputation_applied": imputation_applied,
                "qc_definition": qc_definition,
                "batch_col": batch_col or "",
                "n_batches": int(sub[batch_col].nunique()) if batch_col else None,
            })
        for other in group_cols:
            if other == g_col:
                continue
            table = pd.crosstab(per_sample[g_col].astype(str), per_sample[other].astype(str), dropna=False)
            table_path = out_dir / f"group_cross_table_{safe_name(g_col)}_by_{safe_name(other)}.csv"
            safe_csv_write(table.reset_index(), table_path)
            for _, row in table.reset_index().iterrows():
                row_dict = row.to_dict()
                main_group = row_dict.pop(g_col)
                rows.append({
                    "table": "cross_table",
                    "group_col": g_col,
                    "group": str(main_group),
                    "secondary_col": other,
                    "counts": row_dict,
                    "file": str(table_path),
                })

    df = pd.DataFrame(rows)
    csv_path = safe_csv_write(df, out_dir / "group_composition_qc.csv")
    safe_json_dump({
        "matrix_path": matrix_path,
        "sample_id_col": s_col,
        "primary_group_col": g_col,
        "batch_col": batch_col,
        "n_samples": int(sampleinfo.shape[0]),
        "n_matrix_samples_matched": int(len(sample_cols)),
        "protein_col": protein_col,
        "gene_col": gene_col,
        "qc_definition": qc_definition,
        "input_matrix": input_info,
        "analysed_matrix_rows": int(matrix.shape[0]),
        "imputation_applied": imputation_applied,
        "tables": rows,
    }, out_dir / "group_composition_qc.json")
    return {"csv": csv_path, "rows": rows, "primary_group_col": g_col, "batch_col": batch_col}


def differential_files(config: Dict[str, Any], project_dir: Optional[Path]) -> List[Tuple[str, str]]:
    files = []
    params_path = config.get("parameters_path")
    if params_path and os.path.exists(params_path):
        try:
            params = json.loads(Path(params_path).read_text(encoding="utf-8", errors="ignore"))
            for key, value in params.items():
                if not isinstance(value, dict):
                    continue
                diff = resolve_optional_path(value.get("differential_protein_file"), project_dir)
                if diff and os.path.exists(diff):
                    files.append((str(value.get("contrast_type") or key), diff))
        except Exception:
            pass
    run_folder = Path(config.get("this_run_folder_path", "."))
    processed = run_folder / "processed_proteins"
    if processed.exists():
        for path in processed.glob("differential_*.csv"):
            contrast = path.stem.replace("differential_", "")
            if (contrast, str(path)) not in files:
                files.append((contrast, str(path)))
    return files


def matrix_gene_index(matrix: pd.DataFrame) -> Dict[str, List[int]]:
    protein_col, gene_col = protein_gene_columns(matrix)
    index: Dict[str, List[int]] = {}
    for idx, row in matrix.iterrows():
        values = [row.get(protein_col, "")]
        if gene_col:
            values.append(row.get(gene_col, ""))
        for value in values:
            for token in split_gene_cell(value):
                norm = normalize_gene(token)
                if norm:
                    index.setdefault(norm, []).append(idx)
    return index


PROTEIN_NAME_COLUMNS = ('First.Protein.Description', 'Protein.Description', 'Protein.Names',
                        'PG.ProteinNames', 'Fasta.headers')
GN_TOKEN_PATTERN = re.compile(r'(?:^|[^A-Za-z])GN=([^\s;]+)')


def matrix_alias_index(matrix):
    # R29 traceable alias routes: protein-group id, protein name column, UniProt GN= field.
    # A symbol that only differs by naming convention must not be reported as an absent protein.
    index = {}
    if matrix is None or matrix.empty:
        return index
    columns = [c for c in matrix.columns if str(c) in PROTEIN_NAME_COLUMNS]
    if not columns:
        return index
    for idx, row in matrix.iterrows():
        for column in columns:
            value = row.get(column, '')
            if pd.isna(value):
                continue
            text = str(value)
            tokens = list(split_gene_cell(text))
            tokens.extend(GN_TOKEN_PATTERN.findall(text))
            for token in tokens:
                norm = normalize_gene(token)
                if norm:
                    index.setdefault(norm, []).append(idx)
    return index


def match_candidate_rows(df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    norm = normalize_gene(candidate)
    columns = [c for c in ["PG.Genes", "Genes", "Gene", "gene", "PG.ProteinGroups", "Protein.Group", "protein"] if c in df.columns]
    if not columns:
        return df.iloc[0:0]
    mask = pd.Series(False, index=df.index)
    for col in columns:
        mask = mask | df[col].astype(str).map(lambda x: norm in {normalize_gene(t) for t in split_gene_cell(x)})
    return df[mask]


def choose_candidate_row(rows: pd.DataFrame) -> Optional[pd.Series]:
    if rows.empty:
        return None
    rank_col = "adj.P.Val" if "adj.P.Val" in rows.columns else "P.Value" if "P.Value" in rows.columns else None
    if rank_col:
        ranked = rows.copy()
        ranked[rank_col] = pd.to_numeric(ranked[rank_col], errors="coerce")
        ranked["_abs_logFC"] = pd.to_numeric(ranked.get("logFC", 0), errors="coerce").abs()
        ranked = ranked.sort_values([rank_col, "_abs_logFC"], ascending=[True, False])
        return ranked.iloc[0]
    return rows.iloc[0]


def build_candidate_evidence(config: Dict[str, Any], requirements: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    project_dir = Path(config.get("project_dir", Path.cwd()))
    matrix, matrix_path = load_matrix(config, project_dir)
    raw_matrix = pd.DataFrame()
    raw_matrix_path = ""
    raw_idx: Dict[str, List[int]] = {}
    raw_sample_cols: List[str] = []
    raw_protein_col = ""
    raw_gene_col: Optional[str] = None
    sampleinfo = load_sampleinfo(config)
    design = config.get("analysis_design", {}) if isinstance(config.get("analysis_design"), dict) else {}
    s_col = sample_id_col(sampleinfo, design)
    sample_cols = sample_columns_in_matrix(matrix, sampleinfo, s_col)
    protein_col, gene_col = protein_gene_columns(matrix)
    matrix_idx = matrix_gene_index(matrix)
    alias_idx = matrix_alias_index(matrix)
    rows = []
    diff_files = differential_files(config, project_dir)

    raw_candidate_path = resolve_optional_path(config.get("protein_quant_path"), project_dir)
    if raw_candidate_path and os.path.exists(raw_candidate_path) and os.path.abspath(raw_candidate_path) != os.path.abspath(matrix_path):
        try:
            raw_matrix = attach_gene_annotations(pd.read_csv(raw_candidate_path), config, project_dir)
            raw_matrix_path = raw_candidate_path
            raw_idx = matrix_gene_index(raw_matrix)
            raw_sample_cols = sample_columns_in_matrix(raw_matrix, sampleinfo, s_col)
            raw_protein_col, raw_gene_col = protein_gene_columns(raw_matrix)
        except Exception:
            raw_matrix = pd.DataFrame()
            raw_idx = {}

    for candidate in requirements.get("candidate_proteins", []):
        norm = normalize_gene(candidate)
        matrix_rows = matrix_idx.get(norm, [])
        raw_rows = raw_idx.get(norm, [])
        detected = False
        missing_rate = None
        matched_protein = ""
        matched_gene = ""
        presence_source = "analyzed_matrix"
        alias_rows = alias_idx.get(norm, [])
        match_route = ('gene_symbol' if matrix_rows else 'alias_index' if alias_rows
                       else 'raw_matrix' if (raw_rows and not raw_matrix.empty) else '')
        if matrix_rows:
            row = matrix.loc[matrix_rows[0]]
            detected, missing_rate = compute_detected_and_missing(row, sample_cols)
            matched_protein = str(row.get(protein_col, ""))
            matched_gene = str(row.get(gene_col, "")) if gene_col else ""
        elif alias_rows:
            row = matrix.loc[alias_rows[0]]
            detected, missing_rate = compute_detected_and_missing(row, sample_cols)
            matched_protein = str(row.get(protein_col, ''))
            matched_gene = str(row.get(gene_col, '')) if gene_col else ''
            presence_source = 'analyzed_matrix_alias'
        elif raw_rows and not raw_matrix.empty:
            row = raw_matrix.loc[raw_rows[0]]
            detected, missing_rate = compute_detected_and_missing(row, raw_sample_cols)
            matched_protein = str(row.get(raw_protein_col, ""))
            matched_gene = str(row.get(raw_gene_col, "")) if raw_gene_col else ""
            presence_source = "raw_protein_quant_matrix"
        hit_any = False
        for contrast, path in diff_files:
            try:
                diff = pd.read_csv(path)
            except Exception:
                continue
            matched = match_candidate_rows(diff, candidate)
            chosen = choose_candidate_row(matched)
            if chosen is None:
                continue
            hit_any = True
            logfc = pd.to_numeric(pd.Series([chosen.get("logFC")]), errors="coerce").iloc[0]
            pval = pd.to_numeric(pd.Series([chosen.get("P.Value")]), errors="coerce").iloc[0]
            fdr = pd.to_numeric(pd.Series([chosen.get("adj.P.Val")]), errors="coerce").iloc[0]
            rows.append({
                "candidate": candidate,
                "matched_gene": str(chosen.get("PG.Genes", chosen.get("Gene", matched_gene))),
                "matched_protein": str(chosen.get("PG.ProteinGroups", chosen.get("protein", matched_protein))),
                "detected_in_matrix": True,
                "contrast": contrast,
                "direction": "up_in_group_a" if pd.notna(logfc) and logfc > 0 else "down_in_group_a" if pd.notna(logfc) and logfc < 0 else "no_direction",
                # F3: 机器可读层保留源统计量全精度；展示层由 stat_display_frame() 格式化。
                "logFC": float(logfc) if pd.notna(logfc) else None,
                "P.Value": float(pval) if pd.notna(pval) else None,
                "adj.P.Val": float(fdr) if pd.notna(fdr) else None,
                "missing_rate": missing_rate,
                "evidence_source": "current_matrix",
                "presence_source": presence_source,
                "confidence": "moderate" if pd.notna(fdr) and fdr <= 0.05 else "low" if pd.notna(pval) else "low",
                "match_route": match_route,
                "source_file": path,
            })
        if not hit_any:
            source_file = raw_matrix_path if presence_source == "raw_protein_quant_matrix" and raw_matrix_path else matrix_path
            mapped = bool(detected or matrix_rows or raw_rows or alias_rows)
            rows.append({
                "candidate": candidate,
                "matched_gene": matched_gene,
                "matched_protein": matched_protein,
                "detected_in_matrix": bool(mapped),
                "match_route": match_route,
                "mapping_note": "" if mapped else (
                    t("evidence.mapping_note_a") + t("evidence.mapping_note_b")),
                "contrast": "matrix_presence_only",
                "direction": ("detected_without_selected_contrast" if mapped
                              else "symbol_not_matched_in_available_matrix"),
                "logFC": None,
                "P.Value": None,
                "adj.P.Val": None,
                "missing_rate": missing_rate,
                "evidence_source": "current_matrix",
                "presence_source": presence_source if mapped else "symbol_not_mapped",
                "confidence": "low",
                "source_file": source_file,
            })
    # F3: CSV 是展示层（有效数字/科学计数法）；上面的 rows 仍以全精度进入 JSON。
    df = stat_display_frame(rows)
    csv_path = safe_csv_write(df, out_dir / "candidate_protein_evidence.csv")
    safe_json_dump({"rows": rows, "n_candidates": len(requirements.get("candidate_proteins", [])), "n_rows": len(rows)}, out_dir / "candidate_protein_evidence.json")
    return {"csv": csv_path, "rows": rows}


def build_module_scores(config: Dict[str, Any], requirements: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    matrix, matrix_path = load_matrix(config, Path(config.get("project_dir", Path.cwd())))
    sampleinfo = load_sampleinfo(config)
    design = config.get("analysis_design", {}) if isinstance(config.get("analysis_design"), dict) else {}
    s_col = sample_id_col(sampleinfo, design)
    g_col = group_col(sampleinfo, design)
    sample_cols = sample_columns_in_matrix(matrix, sampleinfo, s_col)
    idx = matrix_gene_index(matrix)
    if not sample_cols:
        return {"sample_scores_csv": "", "group_summary_csv": "", "rows": []}

    sample_score_frames = []
    summary_rows = []
    for module in requirements.get("curated_modules", []):
        name = module.get("name")
        genes = module.get("genes", [])
        row_ids = sorted({i for gene in genes for i in idx.get(normalize_gene(gene), [])})
        matched_genes = [gene for gene in genes if normalize_gene(gene) in idx]
        if not row_ids:
            summary_rows.append({
                "module": name,
                "matched_genes": "",
                "n_matched_genes": 0,
                "group_col": g_col or "",
                "group": "",
                "mean_score": None,
                "evidence_source": "current_matrix",
                "confidence": "low",
            })
            continue
        values = matrix.loc[row_ids, sample_cols].apply(pd.to_numeric, errors="coerce")
        gene_means = values.mean(axis=1)
        gene_sds = values.std(axis=1).replace(0, np.nan)
        z_values = values.sub(gene_means, axis=0).div(gene_sds, axis=0)
        scores = z_values.mean(axis=0, skipna=True)
        frame = pd.DataFrame({"_sample_id": scores.index.astype(str), str(name): scores.values})
        sample_score_frames.append(frame)
        annotated = sampleinfo.copy()
        annotated["_sample_id"] = annotated[s_col].astype(str)
        annotated = annotated.merge(frame, on="_sample_id", how="left")
        if g_col and g_col in annotated.columns:
            for group_value, sub in annotated.groupby(g_col, dropna=False):
                summary_rows.append({
                    "module": name,
                    "matched_genes": ";".join(matched_genes),
                    "n_matched_genes": int(len(matched_genes)),
                    "group_col": g_col,
                    "group": str(group_value),
                    "n_samples": int(sub.shape[0]),
                    "mean_score": round(float(sub[str(name)].mean()), 5) if sub[str(name)].notna().any() else None,
                    "median_score": round(float(sub[str(name)].median()), 5) if sub[str(name)].notna().any() else None,
                    "evidence_source": "current_matrix",
                    "confidence": "moderate" if len(matched_genes) >= 3 else "low",
                })
            groups = [x for x in annotated[g_col].dropna().astype(str).unique().tolist()]
            for a, b in combinations(sorted(groups), 2):
                va = annotated.loc[annotated[g_col].astype(str) == a, str(name)]
                vb = annotated.loc[annotated[g_col].astype(str) == b, str(name)]
                summary_rows.append({
                    "module": name,
                    "matched_genes": ";".join(matched_genes),
                    "n_matched_genes": int(len(matched_genes)),
                    "group_col": g_col,
                    "group": f"{a}_minus_{b}",
                    "n_samples": int(va.notna().sum() + vb.notna().sum()),
                    "mean_score": round(float(va.mean() - vb.mean()), 5) if va.notna().any() and vb.notna().any() else None,
                    "median_score": None,
                    "evidence_source": "current_matrix",
                    "confidence": "moderate" if len(matched_genes) >= 3 else "low",
                })

    sample_scores = sampleinfo.copy()
    sample_scores["_sample_id"] = sample_scores[s_col].astype(str)
    for frame in sample_score_frames:
        sample_scores = sample_scores.merge(frame, on="_sample_id", how="left")
    sample_csv = safe_csv_write(sample_scores, out_dir / "curated_module_sample_scores.csv")
    summary_csv = safe_csv_write(pd.DataFrame(summary_rows), out_dir / "curated_module_group_summary.csv")
    safe_json_dump({"matrix_path": matrix_path, "rows": summary_rows}, out_dir / "curated_module_scores.json")
    return {"sample_scores_csv": sample_csv, "group_summary_csv": summary_csv, "rows": summary_rows}


def build_external_annotation_evidence(config: Dict[str, Any], requirements: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    dataset = requirements.get("dataset", "")
    combined = " ".join(
        [dataset]
        + requirements.get("candidate_proteins", [])
        + [m.get("name", "") for m in requirements.get("curated_modules", [])]
        + [d.get("title", "") for d in requirements.get("scoring_dimensions", [])]
    ).lower()
    applicable = dataset == "Nat_Biotech_Brain_2026" or any(k in combined for k in ["asd", "ndd", "neurodevelopment"])
    project_dir = Path(config.get("project_dir", Path.cwd()))
    rows: List[Dict[str, Any]] = []
    csv_path = ""
    if applicable:
        annotations = load_local_asd_ndd_annotations(project_dir)
        try:
            matrix, matrix_path = load_matrix(config, project_dir)
            idx = matrix_gene_index(matrix)
        except Exception:
            matrix_path = ""
            idx = {}
        for _, item in annotations.iterrows():
            gene = str(item.get("gene", "")).upper().strip()
            matched = idx.get(normalize_gene(gene), [])
            rows.append({
                "gene": gene,
                "category": item.get("category", "ASD_NDD_local_seed"),
                "detected_in_matrix": bool(matched),
                "matched_row_count": int(len(matched)),
                "evidence_source": "external_annotation",
                "annotation_source": "kb_source/scoring_resources/asd_ndd_risk_genes.tsv",
                "current_matrix_support_file": matrix_path if matched else "",
                "boundary": "Local ASD/NDD annotation is an external_annotation crosswalk only; current-matrix expression/statistics must support any dataset-specific claim.",
                "source_note": item.get("source_note", ""),
            })
        csv_path = safe_csv_write(pd.DataFrame(rows), out_dir / "external_annotation_asd_ndd_local.csv")
    json_path = safe_json_dump(
        {
            "applicable": applicable,
            "n_rows": len(rows),
            "n_detected": int(sum(1 for row in rows if row.get("detected_in_matrix"))),
            "rows": rows,
        },
        out_dir / "external_annotation_asd_ndd_local.json",
    )
    return {"csv": csv_path, "json": json_path, "rows": rows, "applicable": applicable}


def build_module_score_distribution_plot(requirements: Dict[str, Any], artifacts: Dict[str, Any], out_dir: Path) -> str:
    sample_scores_csv = artifacts.get("module_scores", {}).get("sample_scores_csv", "")
    if not sample_scores_csv or not os.path.exists(sample_scores_csv):
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    try:
        scores = pd.read_csv(sample_scores_csv)
    except Exception:
        return ""
    module_names = [m.get("name", "") for m in requirements.get("curated_modules", [])]
    module_names = [m for m in module_names if m in scores.columns]
    if not module_names:
        return ""
    group_candidates = [
        c for c in ["Cluster", "Type1", "Type", "CellType", "State", "Group", "Condition", "Treatment"]
        if c in scores.columns and scores[c].dropna().nunique() > 1
    ]
    group = group_candidates[0] if group_candidates else None
    shown_modules = module_names[:8]
    try:
        fig, axes = plt.subplots(len(shown_modules), 1, figsize=(8, max(3, 1.7 * len(shown_modules))), squeeze=False)
        for ax, module in zip(axes[:, 0], shown_modules):
            plot_df = scores[[module] + ([group] if group else [])].copy()
            plot_df[module] = pd.to_numeric(plot_df[module], errors="coerce")
            if group:
                labels = [str(x) for x in plot_df[group].dropna().astype(str).unique().tolist()]
                data = [plot_df.loc[plot_df[group].astype(str) == label, module].dropna().values for label in labels]
                ax.boxplot(data, labels=labels, showfliers=False)
                ax.tick_params(axis="x", rotation=30, labelsize=8)
            else:
                ax.hist(plot_df[module].dropna().values, bins=20)
            ax.set_title(module, fontsize=9)
            ax.set_ylabel("z-score", fontsize=8)
        fig.tight_layout()
        path = out_dir / "dataset_recipe_module_score_distribution.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return str(path)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return ""


def build_dataset_recipe_evidence(config: Dict[str, Any], requirements: Dict[str, Any], artifacts: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    dataset = requirements.get("dataset", "")
    recipe_items = DATASET_RECIPE_REQUIREMENTS.get(dataset, [])
    if not recipe_items:
        recipe_items = [{
            "recipe_item": "General scoring evidence",
            "expected_evidence": "group_composition_qc;candidate_protein_evidence;curated_module_scores",
            "boundary": "Dataset-specific recipe is not defined; use generated current-matrix evidence tables conservatively.",
        }]

    candidate_names = {normalize_gene(x) for x in requirements.get("candidate_proteins", [])}
    module_names = {str(m.get("name", "")) for m in requirements.get("curated_modules", [])}
    plot_path = build_module_score_distribution_plot(requirements, artifacts, out_dir)
    rows: List[Dict[str, Any]] = []
    for item in recipe_items:
        expected = item.get("expected_evidence", "")
        expected_tokens = [t.strip() for t in re.split(r"[;,|]+", expected) if t.strip()]
        evidence_files: List[str] = []
        evidence_source = "current_matrix;analysis_design"
        matched_tokens: List[str] = []
        lower_expected = expected.lower()

        if "group_composition_qc" in lower_expected or "cluster" in lower_expected or "type" in lower_expected:
            evidence_files.append(artifacts.get("group_qc", {}).get("csv", ""))
            matched_tokens.append("group_composition_qc")
        if "external_annotation" in lower_expected or "asd" in lower_expected or "ndd" in lower_expected:
            evidence_files.append(artifacts.get("external_annotation", {}).get("csv", ""))
            evidence_source = "current_matrix;external_annotation"
            matched_tokens.append("external_annotation_asd_ndd_local")

        for token in expected_tokens:
            if token in module_names:
                evidence_files.extend([
                    artifacts.get("module_scores", {}).get("group_summary_csv", ""),
                    artifacts.get("module_scores", {}).get("sample_scores_csv", ""),
                ])
                matched_tokens.append(token)
            if normalize_gene(token) in candidate_names:
                evidence_files.append(artifacts.get("candidate_evidence", {}).get("csv", ""))
                matched_tokens.append(token)

        if not evidence_files:
            evidence_files.extend([
                artifacts.get("group_qc", {}).get("csv", ""),
                artifacts.get("candidate_evidence", {}).get("csv", ""),
                artifacts.get("module_scores", {}).get("group_summary_csv", ""),
            ])
        if plot_path:
            evidence_files.append(plot_path)

        clean_files = []
        for path in evidence_files:
            if path and path not in clean_files:
                clean_files.append(path)
        confidence = "moderate" if clean_files else "low"
        rows.append({
            "dataset": dataset,
            "recipe_item": item.get("recipe_item", ""),
            "expected_evidence": expected,
            "matched_recipe_tokens": ";".join(matched_tokens),
            "evidence_files": ";".join(clean_files),
            "evidence_source": evidence_source,
            "confidence": confidence,
            "boundary": item.get("boundary", ""),
        })

    csv_path = safe_csv_write(pd.DataFrame(rows), out_dir / "dataset_recipe_evidence.csv")
    json_path = safe_json_dump({"rows": rows, "plot": plot_path}, out_dir / "dataset_recipe_evidence.json")
    return {"csv": csv_path, "json": json_path, "plot": plot_path, "rows": rows}


def _contrast_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _diff_gene_label(row: pd.Series) -> str:
    for col in ["PG.Genes", "Genes", "Gene", "gene", "protein", "PG.ProteinGroups", "Protein.Group"]:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def _read_diff_for_story(config: Dict[str, Any], contrast_spec: Dict[str, Any], project_dir: Path) -> Tuple[pd.DataFrame, str, bool, str]:
    files = differential_files(config, project_dir)
    by_key = {_contrast_key(name): (name, path) for name, path in files}
    for alias in contrast_spec.get("aliases", []):
        alias_name = alias.get("name", "") if isinstance(alias, dict) else str(alias)
        found = by_key.get(_contrast_key(alias_name))
        if not found:
            continue
        source_name, path = found
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        return df, source_name, bool(isinstance(alias, dict) and alias.get("invert")), path
    return pd.DataFrame(), "", False, ""


def _prepare_story_diff(df: pd.DataFrame, invert: bool) -> pd.DataFrame:
    if df.empty or "logFC" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["logFC_display"] = pd.to_numeric(out["logFC"], errors="coerce")
    if invert:
        out["logFC_display"] = -out["logFC_display"]
    p_col = "adj.P.Val" if "adj.P.Val" in out.columns else "P.Value" if "P.Value" in out.columns else ""
    if p_col:
        out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    out = out.dropna(subset=["logFC_display"])
    return out


def _top_story_items(diff: pd.DataFrame, direction: str, limit: int = 5) -> str:
    if diff.empty:
        return ""
    p_col = "adj.P.Val" if "adj.P.Val" in diff.columns else "P.Value" if "P.Value" in diff.columns else ""
    sub = diff[diff["logFC_display"] > 0].copy() if direction == "up" else diff[diff["logFC_display"] < 0].copy()
    if sub.empty:
        return ""
    if p_col:
        sub["_rank_p"] = pd.to_numeric(sub[p_col], errors="coerce").fillna(999.0)
        sub = sub.sort_values(["_rank_p", "logFC_display"], ascending=[True, direction != "up"])
    else:
        sub = sub.reindex(sub["logFC_display"].abs().sort_values(ascending=False).index)
    bits = []
    for _, row in sub.head(limit).iterrows():
        label = _diff_gene_label(row)
        logfc = row.get("logFC_display")
        adj = row.get("adj.P.Val", row.get("P.Value", ""))
        bits.append(f"{label} logFC={float(logfc):.3g}, FDR={adj:.3g}" if isinstance(adj, (int, float, np.floating)) and pd.notna(adj) else f"{label} logFC={float(logfc):.3g}")
    return "; ".join(bits)


def _story_candidate_rows(diff: pd.DataFrame, genes: List[str], contrast_spec: Dict[str, Any], source_name: str, source_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if diff.empty:
        return rows
    for gene in genes:
        matched = match_candidate_rows(diff, gene)
        chosen = choose_candidate_row(matched)
        if chosen is None:
            continue
        logfc = pd.to_numeric(pd.Series([chosen.get("logFC_display", chosen.get("logFC"))]), errors="coerce").iloc[0]
        pval = pd.to_numeric(pd.Series([chosen.get("P.Value")]), errors="coerce").iloc[0]
        fdr = pd.to_numeric(pd.Series([chosen.get("adj.P.Val")]), errors="coerce").iloc[0]
        rows.append({
            "dataset": "",
            "row_type": "candidate",
            "claim_id": contrast_spec.get("id", ""),
            "claim_title": contrast_spec.get("claim", ""),
            "display_contrast": contrast_spec.get("display", ""),
            "source_contrast": source_name,
            "candidate": gene,
            "matched_gene": str(chosen.get("PG.Genes", chosen.get("Gene", gene))),
            "matched_protein": str(chosen.get("PG.ProteinGroups", chosen.get("protein", ""))),
            # F3: core_story 与 candidate 表共用同一精度规则（JSON 全精度 / CSV 有效数字）。
            "logFC_display": float(logfc) if pd.notna(logfc) else None,
            "P.Value": float(pval) if pd.notna(pval) else None,
            "adj.P.Val": float(fdr) if pd.notna(fdr) else None,
            "direction_display": "up_in_display_group_a" if pd.notna(logfc) and logfc > 0 else "down_in_display_group_a" if pd.notna(logfc) and logfc < 0 else "no_direction",
            "evidence_source": "current_matrix",
            "confidence": "moderate" if pd.notna(fdr) and fdr <= 0.05 else "low" if pd.notna(pval) else "low",
            "boundary": t("evidence.candidate_boundary"),
            "interpretation": contrast_spec.get("interpretation", ""),
            "source_file": source_path,
        })
    return rows


def _module_mean(module_rows: List[Dict[str, Any]], module: str, group: str) -> Optional[float]:
    for row in module_rows:
        if row.get("module") == module and str(row.get("group")) == str(group):
            try:
                return float(row.get("mean_score"))
            except Exception:
                return None
    return None


def _story_module_rows(module_rows: List[Dict[str, Any]], contrast_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    group_a = str(contrast_spec.get("group_a", ""))
    group_b = str(contrast_spec.get("group_b", ""))
    for module in contrast_spec.get("modules", []):
        a = _module_mean(module_rows, module, group_a)
        b = _module_mean(module_rows, module, group_b)
        matched = ""
        n_matched = ""
        for row in module_rows:
            if row.get("module") == module:
                matched = row.get("matched_genes", "")
                n_matched = row.get("n_matched_genes", "")
                break
        rows.append({
            "dataset": "",
            "row_type": "module",
            "claim_id": contrast_spec.get("id", ""),
            "claim_title": contrast_spec.get("claim", ""),
            "display_contrast": contrast_spec.get("display", ""),
            "module": module,
            "group_a": group_a,
            "group_b": group_b,
            "group_a_mean_score": round(float(a), 5) if a is not None else None,
            "group_b_mean_score": round(float(b), 5) if b is not None else None,
            "module_delta_group_a_minus_group_b": round(float(a - b), 5) if a is not None and b is not None else None,
            "matched_genes": matched,
            "n_matched_genes": n_matched,
            "evidence_source": "current_matrix",
            "confidence": "moderate" if a is not None and b is not None else "low",
            "boundary": "Curated module score summarizes matched current-matrix proteins and is not a replacement for formal enrichment.",
            "interpretation": contrast_spec.get("interpretation", ""),
        })
    return rows


def build_core_story_evidence(config: Dict[str, Any], requirements: Dict[str, Any], artifacts: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    dataset = requirements.get("dataset", "")
    recipe = CORE_STORY_RECIPES.get(dataset, {})
    project_dir = Path(config.get("project_dir", Path.cwd()))
    module_rows = artifacts.get("module_scores", {}).get("rows", [])
    rows: List[Dict[str, Any]] = []
    metadata = {
        "dataset": dataset,
        "story_title": recipe.get("story_title", t("evidence.story_title_default")),
        "executive_claim": recipe.get(
            "executive_claim", t("evidence.executive_claim_default")
        ),
        "boundary": recipe.get("boundary", t("evidence.boundary_default")),
        "has_dataset_story_recipe": bool(recipe),
    }
    primary_contrasts = recipe.get("primary_contrasts", []) if recipe else []
    if not primary_contrasts:
        design = config.get("analysis_design", {}) if isinstance(config.get("analysis_design"), dict) else {}
        for contrast in design.get("differential", {}).get("contrasts", [])[:3]:
            name = str(contrast.get("name", ""))
            primary_contrasts.append({
                "id": safe_name(name),
                "display": name,
                "group_a": contrast.get("group_a", ""),
                "group_b": contrast.get("group_b", ""),
                "aliases": [{"name": name}],
                "claim": str(name) + t("evidence.contrast_claim_suffix"),
                "interpretation": t("evidence.contrast_interpretation_generic"),
                "focus_genes": requirements.get("candidate_proteins", [])[:8],
                "modules": [m.get("name", "") for m in requirements.get("curated_modules", [])[:4]],
            })

    for contrast_spec in primary_contrasts:
        raw_diff, source_name, inverted, source_path = _read_diff_for_story(config, contrast_spec, project_dir)
        diff = _prepare_story_diff(raw_diff, inverted)
        p_col = "adj.P.Val" if "adj.P.Val" in diff.columns else "P.Value" if "P.Value" in diff.columns else ""
        sig = pd.DataFrame()
        if not diff.empty and p_col:
            sig = diff[(pd.to_numeric(diff[p_col], errors="coerce") <= 0.05) & (diff["logFC_display"].abs() >= 0.25)]
        n_up = int((sig["logFC_display"] > 0).sum()) if not sig.empty else 0
        n_down = int((sig["logFC_display"] < 0).sum()) if not sig.empty else 0
        rows.append({
            "dataset": dataset,
            "row_type": "contrast",
            "claim_id": contrast_spec.get("id", ""),
            "claim_title": contrast_spec.get("claim", ""),
            "display_contrast": contrast_spec.get("display", ""),
            "source_contrast": source_name,
            "group_a": contrast_spec.get("group_a", ""),
            "group_b": contrast_spec.get("group_b", ""),
            "inverted_from_source": bool(inverted),
            "n_sig": int(sig.shape[0]) if not sig.empty else 0,
            "n_up_display_group_a": n_up,
            "n_down_display_group_a": n_down,
            "top_up_display_group_a": _top_story_items(diff, "up", limit=5),
            "top_down_display_group_a": _top_story_items(diff, "down", limit=5),
            "evidence_source": "current_matrix",
            "confidence": "moderate" if source_name else "low",
            "boundary": recipe.get("boundary", ""),
            "interpretation": contrast_spec.get("interpretation", ""),
            "source_file": source_path,
        })
        for row in _story_candidate_rows(diff, contrast_spec.get("focus_genes", []), contrast_spec, source_name, source_path):
            row["dataset"] = dataset
            rows.append(row)
        for row in _story_module_rows(module_rows, contrast_spec):
            row["dataset"] = dataset
            rows.append(row)

    # F3: core_story 的 CSV 同样走展示层格式化，避免与 candidate 表对同一蛋白给出不同数值。
    csv_path = safe_csv_write(stat_display_frame(rows), out_dir / "core_story_evidence.csv")
    json_path = safe_json_dump({"metadata": metadata, "rows": rows}, out_dir / "core_story_evidence.json")
    outline = {
        "dataset": dataset,
        "rubric_use": "report_story_outline_v1",
        "story_title": metadata.get("story_title", ""),
        "executive_claim": metadata.get("executive_claim", ""),
        "boundary": metadata.get("boundary", ""),
        "required_report_order": [
            "关键结论速览",
            "数据与预处理",
            "分任务结果解读和相关证据",
            "总结与启发式推测",
            "可复核资产清单",
            "结论边界与方法局限",
        ],
        "primary_contrasts": [spec.get("display", spec.get("contrast", "")) for spec in primary_contrasts],
        "required_tables": [
            "core_story_evidence.csv",
            "candidate_protein_evidence.csv",
            "curated_module_group_summary.csv",
            "group_composition_qc.csv",
        ],
        "focus_genes": sorted({gene for spec in primary_contrasts for gene in spec.get("focus_genes", [])}),
        "mechanism_modules": [module.get("name", "") for module in requirements.get("curated_modules", [])],
        "row_counts": {
            "contrast": sum(1 for row in rows if row.get("row_type") == "contrast"),
            "candidate": sum(1 for row in rows if row.get("row_type") == "candidate"),
            "module": sum(1 for row in rows if row.get("row_type") == "module"),
        },
    }
    outline_path = safe_json_dump(outline, out_dir / "report_story_outline.json")
    return {"csv": csv_path, "json": json_path, "outline_json": outline_path, "metadata": metadata, "rows": rows}


def build_scoring_coverage_legacy(requirements: Dict[str, Any], artifacts: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    candidate_rows = artifacts.get("candidate_evidence", {}).get("rows", [])
    module_rows = artifacts.get("module_scores", {}).get("rows", [])
    group_rows = artifacts.get("group_qc", {}).get("rows", [])
    rows = []
    for dim in requirements.get("scoring_dimensions", []):
        req_text = " ".join(dim.get("requirements", []))
        lower = (dim.get("title", "") + " " + req_text).lower()
        evidence_files = [artifacts.get("group_qc", {}).get("csv", "")]
        if artifacts.get("dataset_recipe", {}).get("csv"):
            evidence_files.append(artifacts.get("dataset_recipe", {}).get("csv", ""))
        status = "covered"
        confidence = "moderate"
        if any(k in lower for k in ["候选", "candidate", "protein", "glul", "cd44", "ezr", "taldo", "atp6v"]):
            evidence_files.append(artifacts.get("candidate_evidence", {}).get("csv", ""))
            if not candidate_rows:
                status = "missing_candidate_table"
                confidence = "low"
        if any(k in lower for k in ["module", "模块", "富集", "机制", "score", "signature", "通路"]):
            evidence_files.append(artifacts.get("module_scores", {}).get("group_summary_csv", ""))
            if not module_rows:
                status = "missing_module_score"
                confidence = "low"
        if any(k in lower for k in ["qc", "样本量", "组成", "cluster", "type", "batch", "donor"]):
            if not group_rows:
                status = "missing_group_qc"
                confidence = "low"
        rows.append({
            "dimension_id": dim.get("id", ""),
            "dimension_title": dim.get("title", ""),
            "points": dim.get("points", ""),
            "coverage_status": status,
            "confidence": confidence,
            "evidence_files": ";".join([f for f in evidence_files if f]),
            "requirements": " | ".join(dim.get("requirements", [])),
            "evidence_source": evidence_source,
        })
    if not rows:
        rows.append({
            "dimension_id": "general",
            "dimension_title": "General scoring coverage",
            "points": "",
            "coverage_status": "covered" if group_rows else "missing_group_qc",
            "confidence": "moderate" if group_rows else "low",
            "evidence_files": ";".join([
                artifacts.get("group_qc", {}).get("csv", ""),
                artifacts.get("candidate_evidence", {}).get("csv", ""),
                artifacts.get("module_scores", {}).get("group_summary_csv", ""),
            ]),
            "requirements": "Generated generic current-matrix evidence tables.",
            "evidence_source": "current_matrix;analysis_design",
        })
    coverage_csv = safe_csv_write(pd.DataFrame(rows), out_dir / "scoring_standard_coverage.csv")
    coverage_json = safe_json_dump({"rows": rows}, out_dir / "scoring_standard_coverage.json")
    return {"csv": coverage_csv, "json": coverage_json, "rows": rows}


def evidence_tables_markdown(artifacts: Dict[str, Any]) -> str:
    """Neutral index of the evidence tables this run produced.

    R19: this text is appended to the report request, so it carries no evaluation-side framing
    (no scoring-coverage table, no coverage status, no scoring dimensions) and no absolute paths.
    It states what the run actually produced, by readable name and relative path.
    """
    candidate_rows = artifacts.get("candidate_evidence", {}).get("rows", [])
    module_rows = artifacts.get("module_scores", {}).get("rows", [])
    recipe_rows = artifacts.get("dataset_recipe", {}).get("rows", [])
    external_rows = artifacts.get("external_annotation", {}).get("rows", [])
    external_applicable = bool(artifacts.get("external_annotation", {}).get("applicable"))
    story_rows = artifacts.get("core_story", {}).get("rows", [])

    def _relative(value: Any) -> str:
        """Readable table name; the workspace-absolute prefix is dropped."""
        text = str(value or "").replace("\\", "/")
        index = text.find("evaluation_evidence/")
        if index >= 0:
            return text[index:]
        return text.rsplit("/", 1)[-1] if text else ""

    lines = [
        "## 本次运行的可用证据表",
        "下列表格由本次运行的工件直接生成，是报告数值的来源；引用时写明表格口径并给出具体数值。",
        "",
        "| 证据表 | 相对路径 |",
        "|---|---|",
    ]
    table_rows = [
        ("分组组成与 QC", artifacts.get("group_qc", {}).get("csv", "")),
        ("核心对比证据", artifacts.get("core_story", {}).get("csv", "")),
        ("候选蛋白证据", artifacts.get("candidate_evidence", {}).get("csv", "")),
        ("模块分数（分组汇总）", artifacts.get("module_scores", {}).get("group_summary_csv", "")),
        ("模块分数（逐样本）", artifacts.get("module_scores", {}).get("sample_scores_csv", "")),
        ("数据集 recipe 证据", artifacts.get("dataset_recipe", {}).get("csv", "")),
    ]
    if external_applicable:
        table_rows.append(("外部注释交叉表", artifacts.get("external_annotation", {}).get("csv", "")))
    for label, path in table_rows:
        relative = _relative(path)
        if relative:
            lines.append(f"| {label} | `{relative}` |")
    if recipe_rows:
        lines.extend([
            "",
            "### 数据集 recipe 证据",
            "| recipe 项目 | 期望证据 | 证据来源 | 置信度 | 边界 |",
            "|---|---|---|---|---|",
        ])
        for row in recipe_rows:
            lines.append(
                f"| {row.get('recipe_item', '')} | {row.get('expected_evidence', '')} | "
                f"{row.get('evidence_source', '')} | {row.get('confidence', '')} | {row.get('boundary', '')} |"
            )
    story_kind = {
        "contrast": t("evidence.story_kind_contrast"),
        "candidate": t("evidence.story_kind_candidate"),
        "module": t("evidence.story_kind_module"),
    }
    if story_rows:
        lines.extend([
            "",
            "### 核心对比证据",
            "| 类型 | 主张 | 对比 | 关键证据 | 置信度 | 边界 |",
            "|---|---|---|---|---|---|",
        ])
        for row in story_rows[:35]:
            if row.get("row_type") == "contrast":
                evidence = (
                    t("evidence.evidence_cell_a")
                    + str(row.get("n_sig", ""))
                    + t("evidence.evidence_cell_b")
                    + str(row.get("n_up_display_group_a", ""))
                    + t("evidence.evidence_cell_c")
                    + str(row.get("n_down_display_group_a", ""))
                    + t("evidence.evidence_cell_d")
                    + str(row.get("top_up_display_group_a", ""))
                )
            elif row.get("row_type") == "candidate":
                # F3: 请求内证据表属于展示层，不能把全精度浮点原样塞进提示词。
                evidence = (f"{row.get('candidate', '')} logFC={stat_display(row.get('logFC_display')) or ''}, "
                            f"adj.P.Val={stat_display(row.get('adj.P.Val')) or ''}")
            else:
                ga = str(row.get('group_a') or '')
                gb = str(row.get('group_b') or '')
                if ga and gb:
                    evidence = '%s delta(%s - %s)=%s' % (row.get('module', ''), ga, gb,
                                                        row.get('module_delta_group_a_minus_group_b', ''))
                else:
                    evidence = '%s delta=%s' % (row.get('module', ''),
                                                row.get('module_delta_group_a_minus_group_b', ''))
            kind = story_kind.get(str(row.get("row_type", "")), str(row.get("row_type", "")))
            lines.append(
                f"| {kind} | {row.get('claim_title', '')} | {row.get('display_contrast', '')} | "
                f"{evidence} | {row.get('confidence', '')} | {row.get('boundary', '')} |"
            )
    if candidate_rows:
        lines.extend([
            "",
            "### 候选蛋白证据摘要",
            "| 候选蛋白 | 对比 | 方向 | logFC | P 值 | 校正后 P 值 | 是否检出 | 置信度 |",
            "|---|---|---|---:|---:|---:|---|---|",
        ])
        for row in candidate_rows[:30]:
            lines.append(
                f"| {row.get('candidate', '')} | {row.get('contrast', '')} | {row.get('direction', '')} | "
                f"{stat_display(row.get('logFC')) or ''} | {stat_display(row.get('P.Value')) or ''} | "
                f"{stat_display(row.get('adj.P.Val')) or ''} | "
                f"{row.get('detected_in_matrix', '')} | {row.get('confidence', '')} |"
            )
    if module_rows:
        lines.extend([
            '',
            '### 模块分数摘要',
            '「分组」列写法 `X_minus_Y` 表示 X 组均值减 Y 组均值，正值为 X 较高；核心对比的方向若与存储行相反，本表已补出该方向的行，「来源」列标明该行是存储值还是由反向行取负派生。',
            '| 模块 | 分组 | 匹配基因数 | 平均分 | 置信度 | 来源 |',
            '|---|---|---:|---:|---|---|',
        ])
        by_module = {}
        for row in module_rows:
            if row.get('group'):
                by_module.setdefault(str(row.get('module', '')), {})[str(row.get('group'))] = row
        core_pairs = []
        for row in story_rows:
            if row.get('row_type') != 'contrast':
                continue
            ga = str(row.get('group_a') or '')
            gb = str(row.get('group_b') or '')
            if ga and gb and ga != gb and (ga, gb) not in core_pairs:
                core_pairs.append((ga, gb))
        derived_rows = []
        for module_name, groups in by_module.items():
            for ga, gb in core_pairs:
                wanted = '%s_minus_%s' % (ga, gb)
                if wanted in groups:
                    continue
                reverse = groups.get('%s_minus_%s' % (gb, ga))
                value = None
                reason = ''
                n_genes = ''
                if reverse is not None and reverse.get('mean_score') is not None:
                    value = -float(reverse['mean_score'])
                    reason = (
                        t("evidence.module_source_reverse_prefix")
                        + "%s_minus_%s" % (gb, ga)
                        + t("evidence.module_source_reverse_suffix")
                    )
                    n_genes = reverse.get('n_matched_genes', '')
                elif (groups.get(ga, {}).get('mean_score') is not None
                      and groups.get(gb, {}).get('mean_score') is not None):
                    value = float(groups[ga]['mean_score']) - float(groups[gb]['mean_score'])
                    reason = t("evidence.module_source_subtract")
                    n_genes = groups.get(ga, {}).get('n_matched_genes', '')
                if value is None:
                    continue
                derived_rows.append({'module': module_name, 'group': wanted,
                                     'n_matched_genes': n_genes, 'mean_score': round(value, 5),
                                     'source_note': reason})
        shown = 0
        for row in module_rows:
            if row.get('group') and shown < 30:
                lines.append(
                    '| %s | %s | %s | %s | %s | %s |' % (row.get('module', ''), row.get('group', ''),
                                                         row.get('n_matched_genes', ''), row.get('mean_score', ''),
                                                         row.get('confidence', ''),
                                                         t("evidence.module_source_stored"))
                )
                shown += 1
        for row in derived_rows[:20]:
            lines.append(
                '| %s | %s | %s | %s |  | %s |' % (row['module'], row['group'], row['n_matched_genes'],
                                                     row['mean_score'], row['source_note'])
            )
    if external_rows:
        lines.extend([
            "",
            "### 离线外部注释交叉表",
            "| 基因 | 类别 | 矩阵中检出 | 匹配行数 | 证据来源 | 边界 |",
            "|---|---|---|---:|---|---|",
        ])
        shown = 0
        for row in external_rows:
            if shown >= 30:
                break
            if not row.get("detected_in_matrix") and shown >= 10:
                continue
            lines.append(
                f"| {row.get('gene', '')} | {row.get('category', '')} | {row.get('detected_in_matrix', '')} | "
                f"{row.get('matched_row_count', '')} | {row.get('evidence_source', '')} | {row.get('boundary', '')} |"
            )
            shown += 1
    return "\n".join(lines)


# R19: the legacy name is kept so existing callers keep working.
coverage_markdown = evidence_tables_markdown


def build_scoring_coverage(requirements: Dict[str, Any], artifacts: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    candidate_rows = artifacts.get("candidate_evidence", {}).get("rows", [])
    module_rows = artifacts.get("module_scores", {}).get("rows", [])
    group_rows = artifacts.get("group_qc", {}).get("rows", [])
    story_rows = artifacts.get("core_story", {}).get("rows", [])
    rows = []

    for dim in requirements.get("scoring_dimensions", []):
        req_text = " ".join(dim.get("requirements", []))
        lower = (dim.get("title", "") + " " + req_text).lower()
        evidence_files = [
            artifacts.get("group_qc", {}).get("csv", ""),
            artifacts.get("dataset_recipe", {}).get("csv", ""),
            artifacts.get("core_story", {}).get("csv", ""),
        ]
        status = "covered"
        confidence = "moderate"

        if any(k in lower for k in ["candidate", "protein", "gene", "marker", "glul", "cd44", "ezr", "taldo", "atp6v", "pou5f1", "eomes", "tbr1"]):
            evidence_files.append(artifacts.get("candidate_evidence", {}).get("csv", ""))
            if not candidate_rows:
                status = "missing_candidate_table"
                confidence = "low"
        if any(k in lower for k in ["module", "score", "signature", "pathway", "enrichment", "mechanism", "trajectory", "lineage", "migration", "translation", "chromatin", "synapse"]):
            evidence_files.extend([
                artifacts.get("module_scores", {}).get("group_summary_csv", ""),
                artifacts.get("module_scores", {}).get("sample_scores_csv", ""),
            ])
            if not module_rows:
                status = "missing_module_score"
                confidence = "low"
        if any(k in lower for k in ["qc", "sample", "composition", "cluster", "type", "batch", "donor", "group"]):
            if not group_rows:
                status = "missing_group_qc"
                confidence = "low"
        if any(k in lower for k in ["asd", "ndd", "risk", "disease", "neurodevelopment", "autism"]):
            evidence_files.append(artifacts.get("external_annotation", {}).get("csv", ""))
            if artifacts.get("external_annotation", {}).get("applicable") and not artifacts.get("external_annotation", {}).get("rows", []):
                status = "missing_external_annotation"
                confidence = "low"
        if any(k in lower for k in ["contrast", "vs", "对比", "方向", "效应量", "炎症", "klrg1", "cycloheximide", "bortezomib", "lps"]):
            if not story_rows:
                status = "missing_core_story_evidence"
                confidence = "low"

        clean_files = []
        for path in evidence_files:
            if path and path not in clean_files:
                clean_files.append(path)
        evidence_source = "current_matrix;analysis_design"
        if any("external_annotation" in str(path) for path in clean_files):
            evidence_source += ";external_annotation"
        rows.append({
            "dimension_id": dim.get("id", ""),
            "dimension_title": dim.get("title", ""),
            "points": dim.get("points", ""),
            "coverage_status": status,
            "confidence": confidence,
            "evidence_files": ";".join(clean_files),
            "requirements": " | ".join(dim.get("requirements", [])),
            "evidence_source": evidence_source,
        })

    if not rows:
        evidence_files = [
            artifacts.get("group_qc", {}).get("csv", ""),
            artifacts.get("core_story", {}).get("csv", ""),
            artifacts.get("candidate_evidence", {}).get("csv", ""),
            artifacts.get("module_scores", {}).get("group_summary_csv", ""),
            artifacts.get("dataset_recipe", {}).get("csv", ""),
            artifacts.get("external_annotation", {}).get("csv", ""),
        ]
        rows.append({
            "dimension_id": "general",
            "dimension_title": "General scoring coverage",
            "points": "",
            "coverage_status": "covered" if group_rows else "missing_group_qc",
            "confidence": "moderate" if group_rows else "low",
            "evidence_files": ";".join([path for path in evidence_files if path]),
            "requirements": "Generated generic current-matrix evidence tables.",
            "evidence_source": "current_matrix;analysis_design;external_annotation",
        })

    coverage_csv = safe_csv_write(pd.DataFrame(rows), out_dir / "scoring_standard_coverage.csv")
    coverage_json = safe_json_dump({"rows": rows}, out_dir / "scoring_standard_coverage.json")
    return {"csv": coverage_csv, "json": coverage_json, "rows": rows}


def _prepare_evidence_pack_from_requirements(config: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    run_folder = Path(config.get("this_run_folder_path", "."))
    out_dir = run_folder / "evaluation_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(config.get("project_dir", Path.cwd()))
    config = dict(config)
    config.setdefault("project_dir", str(project_dir))

    evidence_scope = requirements.get("evidence_scope", "user_visible")
    requirements_filename = "user_visible_requirements.used.json" if evidence_scope == "user_visible" else "evaluator_requirements.used.json"
    requirements_path = safe_json_dump(requirements, run_folder / requirements_filename)
    artifacts: Dict[str, Any] = {"requirements_path": requirements_path, "evidence_scope": evidence_scope}
    artifacts["group_qc"] = build_group_composition_qc(config, requirements, out_dir)
    artifacts["candidate_evidence"] = build_candidate_evidence(config, requirements, out_dir)
    artifacts["module_scores"] = build_module_scores(config, requirements, out_dir)
    artifacts["external_annotation"] = build_external_annotation_evidence(config, requirements, out_dir)
    artifacts["core_story"] = build_core_story_evidence(config, requirements, artifacts, out_dir)
    artifacts["dataset_recipe"] = build_dataset_recipe_evidence(config, requirements, artifacts, out_dir)
    artifacts["scoring_coverage"] = build_scoring_coverage(requirements, artifacts, out_dir)
    artifacts["markdown"] = evidence_tables_markdown(artifacts)
    safe_json_dump(artifacts, out_dir / "evaluation_evidence_manifest.json")

    try:
        append_evidence(
            str(run_folder),
            source="current_matrix",
            tool="prepare_user_visible_evidence_pack" if evidence_scope == "user_visible" else "prepare_evaluator_evidence_pack",
            claim="Dataset-specific evidence tables were generated from user-visible current matrix outputs and local versioned resources.",
            files=[
                requirements_path,
                artifacts["group_qc"].get("csv", ""),
                artifacts["core_story"].get("csv", ""),
                artifacts["candidate_evidence"].get("csv", ""),
                artifacts["module_scores"].get("group_summary_csv", ""),
                artifacts["module_scores"].get("sample_scores_csv", ""),
                artifacts["dataset_recipe"].get("csv", ""),
                artifacts["external_annotation"].get("csv", ""),
                artifacts["scoring_coverage"].get("csv", ""),
            ],
            metrics={
                "dataset": requirements.get("dataset"),
                "evidence_scope": evidence_scope,
                "n_candidates": len(requirements.get("candidate_proteins", [])),
                "n_modules": len(requirements.get("curated_modules", [])),
                "n_scoring_dimensions": len(requirements.get("scoring_dimensions", [])),
                "n_core_story_rows": len(artifacts["core_story"].get("rows", [])),
                "n_dataset_recipe_items": len(artifacts["dataset_recipe"].get("rows", [])),
                "n_external_annotation_rows": len(artifacts["external_annotation"].get("rows", [])),
            },
            confidence="moderate",
            limitations=[
                "Curated modules are heuristic analysis aids; they support scoring coverage but do not replace formal pathway enrichment.",
                "External annotations are local offline crosswalks and must not be described as direct current-matrix disease evidence.",
            ],
        )
    except Exception:
        pass
    try:
        import analysis_extensions
        extensions = analysis_extensions.run_all(Path(run_folder))
        if extensions:
            artifacts["analysis_extensions"] = extensions
    except Exception as exc:
        try:
            (Path(run_folder) / "evaluation_evidence" / "analysis_extensions_error.txt").write_text(
                str(exc), encoding="utf-8")
        except Exception:
            pass
    return artifacts


def prepare_user_visible_evidence_pack(config: Dict[str, Any]) -> Dict[str, Any]:
    requirements = extract_user_visible_requirements(config)
    return _prepare_evidence_pack_from_requirements(config, requirements)


def prepare_evaluator_evidence_pack(config: Dict[str, Any], evaluator_root: str | Path | None = None) -> Dict[str, Any]:
    evaluator_config = dict(config)
    if evaluator_root:
        evaluator_root_path = Path(evaluator_root)
        evaluator_config["ground_truth_path"] = str(evaluator_root_path / "ground_truth.txt")
        evaluator_config["grading_standard_path"] = str(evaluator_root_path / "grading_standard.txt")
    requirements = extract_evaluation_requirements(evaluator_config)
    requirements["evidence_scope"] = "evaluator_only"
    return _prepare_evidence_pack_from_requirements(evaluator_config, requirements)


def prepare_evaluation_evidence_pack(config: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible wrapper for generation-time evidence.

    The legacy name is retained for old callers, but the default behavior is now
    publication-safe and does not read evaluator-only standards.
    """
    return prepare_user_visible_evidence_pack(config)
