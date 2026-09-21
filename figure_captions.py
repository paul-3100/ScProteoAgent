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

The module is stdlib-only (csv/json/os/re/argparse) so it stays importable in any
environment, including minimal scoring venvs.

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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTIONS_FILENAME = "figures_captions.md"
FIGURES_MD_FILENAME = "figures.md"
FIGURES_LOCAL_MD_FILENAME = "figures_preview_local.md"
ORIGINAL_FIGURE_MAP_REL = os.path.join("docs", "figure_upgrade_20260904", "original_figure_map.json")

CATEGORY_ORDER = ("QC", "Sample structure", "Differential evidence", "Mechanism evidence", "Enrichment evidence")

BOUNDARY_MATRIX = (
    "current_matrix：图中数字由当前定量矩阵直接计算，可在 run 目录内复现。"
)
BOUNDARY_ENRICHMENT = (
    "offline_enrichment：富集条目来自当前矩阵差异蛋白的离线通路注释，支撑机制解释，"
    "不构成矩阵内的直接测量或因果证据。"
)
BOUNDARY_MIXED = (
    "current_matrix + offline_enrichment：热图数值来自当前矩阵的组均值 row Z-score，"
    "蛋白集合来自离线富集通路条目。"
)
PAPER_REF_FALLBACK = "本图为 Agent 标准分析图，原论文中没有直接对应的面板（映射基线见 original_figure_map 提取记录）。"

# Chinese titles for plot types when no explicit title is supplied (CLI default mode).
PLOT_TYPE_TITLES = {
    "qc_sample_overview": "样本与缺失率 QC",
    "pca": "PCA 样本结构",
    "umap": "UMAP 样本结构",
    "heatmap": "差异蛋白热图",
    "differential_summary_barplot": "差异蛋白数量概览",
    "key_protein_overview": "关键蛋白总览",
    "protein_contrast_bubble": "候选蛋白对比气泡图",
    "mechanism_top_protein_group_means": "核心蛋白组均值图",
    "mechanism_enrichment_dotplot": "机制富集 dotplot",
    "mechanism_dep_overlap_upset": "差异蛋白重叠 UpSet 图",
    "mechanism_contrast_evidence": "单对比综合证据面板",
    "enrichment_bubble": "富集气泡图",
    "mechanism_pathway_gene_heatmap": "通路基因热图",
}


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
        group_name = _clean_cell(row.get("group")) or "未命名分组"
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
        return "样本量：run 内缺少 group_composition_qc.csv，未记录分组样本数（可在 evaluation_evidence 目录核对）。"
    detail = "、".join(f"{g['group']}: {g['n']}" for g in groups)
    parts = [f"样本量：N={ss.get('n_total')}（{detail}）"]
    if ss.get("group_col"):
        parts.append(f"分组列 {ss['group_col']}")
    if ss.get("mean_detected") is not None:
        parts.append(f"平均每样本检出 {_fmt_num(ss['mean_detected'])} 个蛋白")
    if ss.get("mean_missing") is not None:
        parts.append(f"平均缺失率 {_fmt_pct(ss['mean_missing'])}")
    return "；".join(parts) + "。"


def _threshold_core(th: Dict[str, Any]) -> str:
    """Core threshold text without a leading label, e.g. ``adj.P.Val < 0.05；|log2FC| ≥ 0.25（BH 校正）``."""
    bits = []
    if th.get("pval") is not None:
        bits.append(f"adj.P.Val < {_fmt_num(th['pval'])}")
    if th.get("logfc") is not None:
        bits.append(f"|log2FC| ≥ {_fmt_num(th['logfc'])}")
    if not bits:
        return ""
    return f"{'；'.join(bits)}（{th.get('fdr_method', 'BH')} 校正）"


def _threshold_sentence(th: Dict[str, Any]) -> str:
    core = _threshold_core(th)
    if not core:
        return "显著性阈值：figure_manifest 未记录该图阈值（差异分析默认口径见文档总览）。"
    return f"显著性阈值：{core}。"


def _paper_ref_line(file_name: str, context: Dict[str, Any]) -> str:
    hits = (context.get("paper_index") or {}).get(file_name, [])
    if not hits:
        return PAPER_REF_FALLBACK
    parts = []
    for fig_no, qualifier in hits:
        parts.append(f"{fig_no}（{qualifier}）" if qualifier else fig_no)
    return "对应原论文 " + "、".join(parts) + "。"


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
# (caption_metadata.omitted_panels) and their Chinese panel names.
QC_PANEL_TITLES = {
    "detected_proteins": "检出蛋白面板（Detected Proteins Per Sample）",
    "missing_rate": "缺失率面板（Missing Rate By Cluster）",
    "sample_counts": "样本数面板（Sample Count By Cluster）",
    "sample_correlation": "相关性面板（Sample Correlation）",
}


def _caption_qc(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    ss = context.get("sample_summary") or {}
    meta = record.get("caption_metadata") or {}
    n_unannotated = _to_int(meta.get("n_unannotated")) or 0
    large_n = bool(meta.get("large_n_mode"))
    omitted = [str(x) for x in (meta.get("omitted_panels") or [])]
    groups = ss.get("groups") or []
    interp_bits = ["QC 面板给出后续所有差异与模块分析的样本基础"]
    if groups:
        detail = "、".join(f"{g['group']}: {g['n']}" for g in groups)
        interp_bits.append(f"共 {ss.get('n_total')} 个样本（{detail}）")
    if n_unannotated > 0:
        interp_bits.append(f"另有 {n_unannotated} 个样本缺少分组注释，图中统一显示为 Unannotated")
    if ss.get("mean_detected") is not None:
        interp_bits.append(f"平均每样本检出 {_fmt_num(ss['mean_detected'])} 个蛋白")
    if ss.get("mean_missing") is not None:
        interp_bits.append(f"平均缺失率 {_fmt_pct(ss['mean_missing'])}")
    if large_n:
        interp_bits.append("样本量超过 600 时检出蛋白面板自动切换为排序折线（附 Q1/中位数/Q3 分位参考线），缺失率散点按固定随机种子抽样")
    panels: List[str] = []
    if "detected_proteins" not in omitted:
        if large_n:
            panels.append("检出蛋白面板（Detected Proteins Per Sample）：样本按检出蛋白数排序绘制折线，标注 Q1/中位数/Q3 分位参考线与四分位带（样本量较大时替代逐样本柱状图）。")
        else:
            panels.append("检出蛋白面板（Detected Proteins Per Sample）：每个样本一根柱，柱高为该样本检出蛋白数。")
    if "missing_rate" not in omitted:
        if large_n:
            panels.append("缺失率面板（Missing Rate By Cluster）：按分组汇总的缺失率箱线图，叠加样本散点（样本量较大时散点按固定随机种子抽样，控制图形密度）。")
        else:
            panels.append("缺失率面板（Missing Rate By Cluster）：按分组汇总的平均缺失率。")
    if "sample_counts" not in omitted:
        panels.append("样本数面板（Sample Count By Cluster）：各分组样本数。")
    if "sample_correlation" not in omitted:
        panels.append("相关性面板（Sample Correlation）：样本两两 Spearman 相关系数热图，颜色范围 -1 至 1。")
    if omitted:
        omitted_names = "、".join(QC_PANEL_TITLES.get(key, key) for key in omitted)
        panels.append(f"面板省略说明：{omitted_names} 因无匹配数据未绘制，不使用占位图。")
    return {
        "title": title or PLOT_TYPE_TITLES["qc_sample_overview"],
        "panels": panels,
        "reading": (
            "各面板依次回答：每个样本测到多少蛋白、缺失了多少、分组是否均衡、样本间重复性如何；"
            "相关系数越接近 1，两个样本的蛋白图谱越一致。"
        ),
        "thresholds": [
            "统计性质：描述性 QC 汇总，不设显著性阈值。",
            _sample_line(context),
        ],
        "statistics": "描述性统计；相关性面板使用 Spearman 相关系数。",
        "interpretation": "；".join(interp_bits) + "。",
        "boundary": BOUNDARY_MATRIX,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_pca(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    group_col = context.get("group_col") or "分组"
    outline = context.get("outline") or {}
    claim = _clean_cell(outline.get("executive_claim"))
    interp = f"PCA 展示 {group_col} 维度的整体分离与离散程度，为后续各对比差异方向提供矩阵内结构参照。"
    if claim:
        interp += f"结合主结论「{claim}」阅读。"
    return {
        "title": title or PLOT_TYPE_TITLES["pca"],
        "panels": [
            "PCA 散点：每个点为一个样本，横轴 PC1、纵轴 PC2，轴标签附各主成分解释方差百分比。",
            "分组中心与椭圆：同组样本的中心位置与离散范围（按组着色）。",
        ],
        "reading": (
            "点间距离越近表示两个样本的蛋白丰度总模式越相似；颜色对应 "
            f"{group_col} 分组；同组椭圆与异组椭圆重叠越少，组间整体分离越明显。"
        ),
        "thresholds": [
            "降维设置：基于当前矩阵蛋白丰度的方差结构，不设显著性阈值。",
            _sample_line(context),
        ],
        "statistics": "描述性降维（PCA），不涉及假设检验；PC1/PC2 解释方差比例标注于轴标签。",
        "interpretation": interp,
        "boundary": BOUNDARY_MATRIX,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_umap(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    group_col = context.get("group_col") or "分组"
    outline = context.get("outline") or {}
    claim = _clean_cell(outline.get("executive_claim"))
    interp = f"UMAP 展示 {group_col} 维度样本的邻域结构，可与 PCA 互为对照判断分组结构是否稳健。"
    if claim:
        interp += f"结合主结论「{claim}」阅读。"
    return {
        "title": title or PLOT_TYPE_TITLES["umap"],
        "panels": [
            "UMAP 散点：每个点为一个样本，横轴 UMAP 1、纵轴 UMAP 2，按组着色并附图例。",
        ],
        "reading": (
            "相邻点表示蛋白丰度模式相似的样本；颜色对应 "
            f"{group_col} 分组；UMAP 保留局部邻域结构，同组点的聚集比全局距离更可比较。"
        ),
        "thresholds": [
            "降维设置：基于当前矩阵蛋白丰度的邻域结构，不设显著性阈值。",
            _sample_line(context),
        ],
        "statistics": "描述性降维（UMAP），不涉及假设检验。",
        "interpretation": interp,
        "boundary": BOUNDARY_MATRIX,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_heatmap(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    group_col = context.get("group_col") or "分组"
    top = th.get("top_proteins")
    scale = f"行数为按信息量挑选的 top {_fmt_num(top)} 个蛋白" if top is not None else "行数为信息量最高的差异蛋白"
    return {
        "title": title or PLOT_TYPE_TITLES["heatmap"],
        "panels": [
            f"聚类热图：{scale}，列为样本（按 {group_col} 分组并附列颜色条）。",
            "颜色条：Row Z-score（按蛋白行标准化后的丰度）。",
        ],
        "reading": (
            "红色表示该蛋白在对应样本中丰度高于其跨样本平均水平，蓝色表示低于；"
            "列的聚类与颜色条可同时反映分组结构是否在差异蛋白上重现。"
        ),
        "thresholds": [
            _threshold_sentence(th),
            _sample_line(context),
        ],
        "statistics": "展示层为 row Z-score（按蛋白行标准化），不涉及组间假设检验。",
        "interpretation": (
            f"top 差异蛋白在 {group_col} 分组上的丰度结构；行方向（蛋白）与列方向（样本）的聚集"
            "一起说明组间差异是否一致、是否存在混杂亚群。"
        ),
        "boundary": BOUNDARY_MATRIX,
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
        group_a, group_b = (stats_row.get("group_a") or "A 组"), (stats_row.get("group_b") or "B 组")

    count_line = "该对比的上下调计数在 figure_manifest 与 core_story_evidence.csv 中均未记录，以图内图例计数为准。"
    if counts.get("n_sig") is not None or (counts.get("n_up") is not None and counts.get("n_down") is not None):
        count_line = (
            f"显著差异蛋白 {counts.get('n_sig')} 个（{_clean_cell(counts.get('source'))}）："
            f"上调 {counts.get('n_up')}、下调 {counts.get('n_down')}（方向为 {group_a} 相对 {group_b}）。"
        )

    interp_parts: List[str] = []
    if counts.get("n_sig") is not None:
        interp_parts.append(f"{group_a} 相对 {group_b} 共 {counts.get('n_sig')} 个显著差异蛋白（上调 {counts.get('n_up')} / 下调 {counts.get('n_down')}）")
    top_up = _extract_top_genes(stats_row.get("top_up"))
    top_down = _extract_top_genes(stats_row.get("top_down"))
    if top_up:
        interp_parts.append("上调端代表蛋白：" + "、".join(top_up))
    if top_down:
        interp_parts.append("下调端代表蛋白：" + "、".join(top_down))
    row_interp = _clean_cell(stats_row.get("interpretation"))
    if row_interp:
        interp_parts.append(row_interp)
    interp_parts = _drop_hollow_purpose_parts(interp_parts)
    if not interp_parts:
        interp_parts = ["方向与显著性以图内散点和阈值参考线为准"]
    interpretation = "；".join(interp_parts) + "。"

    return {
        "title": title or f"{_human_contrast(contrast)} 火山图",
        "panels": [
            f"散点：每个点为一个蛋白，横轴 log2 Fold Change（{group_a} 相对 {group_b}），纵轴 -log10(adj.P.Val)。",
            "阈值参考线：两条竖虚线为 |log2FC| 阈值，一条横虚线为 adj.P.Val 阈值。",
            "配色：红色 Upregulated、蓝色 Downregulated、灰色 Not significant，图例附各类计数。",
            "基因标注：每侧标注效应量×显著性评分最高的基因名（adjust_text 防重叠）。",
        ],
        "reading": (
            f"右上（红色）为 {group_a} 中升高的蛋白，左上（蓝色）为 {group_a} 中降低的蛋白；"
            "点越靠上校正后越显著，越靠右/左效应量越大；灰色点不进入候选与富集分析。"
        ),
        "thresholds": [
            _threshold_sentence(th),
            count_line,
            _sample_line(context),
        ],
        "statistics": (
            f"limma 线性模型双侧检验；多重校正 {th.get('fdr_method', 'BH')}（adj.P.Val）；效应量为 log2 fold change。"
        ),
        "interpretation": interpretation,
        "boundary": BOUNDARY_MATRIX,
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
            f"{_human_contrast(row['contrast'])} 上调 {row.get('n_up')} / 下调 {row.get('n_down')}"
        )
    count_line = (
        "各对比计数（方向为对比名中 A 组相对 B 组）：" + "；".join(count_bits) + "。"
        if count_bits
        else "各对比计数见图中柱上标注数字（figure_manifest 未记录计数元数据）。"
    )
    biggest = None
    for row in ordered:
        n_sig = row.get("n_sig")
        if n_sig is None:
            continue
        if biggest is None or (biggest.get("n_sig") or 0) < n_sig:
            biggest = row
    interp = "汇总各对比通过阈值的差异数规模"
    if biggest is not None:
        interp += f"；差异规模最大的对比为 {_human_contrast(biggest['contrast'])}（{biggest.get('n_sig')} 个显著差异蛋白）"
    interp += "，规模差异提示主对比与对照间对比的信号强度不同。"
    return {
        "title": title or PLOT_TYPE_TITLES["differential_summary_barplot"],
        "panels": [
            "柱状图：每个对比两根柱，红色 Up（上调蛋白数，向上）、蓝色 Down（下调蛋白数，向下）。",
            "数字标注：柱端标出各方向的确切计数。",
        ],
        "reading": (
            "柱长为该对比同时满足 adj.P.Val 与 |log2FC| 阈值的蛋白数；向上为对比名中 A 组升高、"
            "向下为 A 组降低；上下规模不对称时说明方向性偏移。"
        ),
        "thresholds": [
            f"计数口径：adj.P.Val < {_fmt_num(th.get('pval'))} 且 |log2FC| ≥ {_fmt_num(th.get('logfc'))}（{_th_fdr(th)} 校正）。"
            if th.get("pval") is not None or th.get("logfc") is not None
            else "计数口径：figure_manifest 未记录该图阈值（差异分析默认口径见文档总览）。",
            count_line,
            _sample_line(context),
        ],
        "statistics": (
            f"各对比先经 limma 双侧检验 + {_th_fdr(th)} 校正，再计数；本图本身不执行新的检验。"
        ),
        "interpretation": interp,
        "boundary": BOUNDARY_MATRIX,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _focus_genes_line(context: Dict[str, Any], k: int = 6) -> str:
    outline = context.get("outline") or {}
    genes = [str(g).strip() for g in (outline.get("focus_genes") or []) if str(g).strip()]
    if not genes:
        return ""
    shown = genes[:k]
    suffix = "等" if len(genes) > k else ""
    return "报告 focus 基因（" + "、".join(shown) + suffix + "）可在图中对照追踪。"


def _caption_key_protein_overview(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    per_contrast = th.get("top_per_contrast")
    select_line = "筛选口径：" + (_threshold_core(th) or "figure_manifest 未记录该图阈值")
    if per_contrast is not None:
        select_line += f"；每对比取评分 top {_fmt_num(per_contrast)} 个蛋白"
    select_line += "。"
    focus = _focus_genes_line(context)
    interp = "并排查看各对比显著蛋白的方向与强度：红色 Up、蓝色 Down 两类点分别对应 A 组升高与降低。"
    if focus:
        interp += focus
    return {
        "title": title or PLOT_TYPE_TITLES["key_protein_overview"],
        "panels": [
            "散点：横轴为对比（Contrast，按评分排序），纵轴 log2 Fold Change；红色 Up、蓝色 Down。",
            "点大小：-log10(adj.P.Val)，越显著点越大；深灰横线为该对比 log2FC 的中位数。",
            "基因标注：每对比标注评分最高的若干基因名。",
        ],
        "reading": (
            "同一对比内，点越高表示该蛋白在 A 组升高越多（Down 相反）；不同对比的横轴位置不可横向比较，"
            "应以纵轴 log2FC 数值与点大小为准。"
        ),
        "thresholds": [
            select_line,
            _sample_line(context),
        ],
        "statistics": f"log2FC 与 adj.P.Val 来自 limma 双侧检验（{_th_fdr(th)} 校正）。",
        "interpretation": interp,
        "boundary": BOUNDARY_MATRIX,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_protein_bubble(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    per_contrast = th.get("top_per_contrast")
    max_proteins = th.get("max_proteins")
    select_bits: List[str] = []
    if per_contrast is not None:
        select_bits.append(f"每对比取评分 top {_fmt_num(per_contrast)} 个蛋白")
    if max_proteins is not None:
        select_bits.append(f"最多展示 {_fmt_num(max_proteins)} 个蛋白")
    select_line = "筛选口径：" + (_threshold_core(th) or "figure_manifest 未记录该图阈值")
    if select_bits:
        select_line += "；" + "；".join(select_bits)
    select_line += "。"
    focus = _focus_genes_line(context)
    interp = (
        "跨对比并排比较候选蛋白：颜色一致（同为红/蓝）说明方向在对比间稳定，颜色翻转说明方向相反，"
        "空白表示该蛋白未进入该对比的 top 集合。"
    )
    if focus:
        interp += focus
    return {
        "title": title or PLOT_TYPE_TITLES["protein_contrast_bubble"],
        "panels": [
            "气泡图：横轴为对比（Contrast），纵轴为关键蛋白（Key protein）。",
            "颜色通道：log2 Fold Change，红正蓝负，色标以 0 为中心对称。",
            "大小通道：-log10(adj.P.Val)，越显著气泡越大。",
        ],
        "reading": (
            "横向读一个蛋白跨对比的方向与显著性变化；纵向读一个对比内候选蛋白的组成；"
            "颜色深浅与气泡大小要同时读，避免只按显著性排序。"
        ),
        "thresholds": [
            select_line,
            _sample_line(context),
        ],
        "statistics": f"log2FC 与 adj.P.Val 来自 limma 双侧检验（{_th_fdr(th)} 校正）。",
        "interpretation": interp,
        "boundary": BOUNDARY_MATRIX,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_top_means(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    group_col = context.get("group_col") or "分组"
    top = th.get("top_proteins")
    module_summary = context.get("module_summary") or {}
    module_order = module_summary.get("order") or []
    module_bits = ""
    if module_order:
        name = module_order[0]
        info = (module_summary.get("modules") or {}).get(name, {})
        n_genes = info.get("n_genes")
        module_bits = (
            f"可与 curated module（{name}，{_fmt_num(n_genes)} 个匹配基因，"
            "见 evaluation_evidence/curated_module_group_summary.csv）的分组均分交叉验证"
        )
    else:
        module_bits = "可与 evaluation_evidence/curated_module_group_summary.csv 的分组均分交叉验证"
    scale = f"各对比 adj.P.Val < {_fmt_num(th.get('pval'))} 的 top {_fmt_num(top)} 差异蛋白" if top is not None else "各对比 top 差异蛋白"
    protein_set_line = (
        f"蛋白集合：{scale}（{_threshold_core(th)}）。"
        if top is not None and _threshold_core(th)
        else f"蛋白集合：{scale}。"
    )
    return {
        "title": title or PLOT_TYPE_TITLES["mechanism_top_protein_group_means"],
        "panels": [
            f"左面板 Group Mean Abundance：{scale}在各 {group_col} 分组的平均 row Z-score（颜色 -2.5 至 2.5）。",
            "右面板 Detection Rate：同组蛋白在各分组的检出率（0-100%）。",
        ],
        "reading": (
            "左图颜色越红表示该蛋白在该分组平均丰度越高；右图检出率越低，左图均值受缺失模式的影响越大，"
            "两个面板应同时读取。"
        ),
        "thresholds": [
            protein_set_line,
            _sample_line(context),
        ],
        "statistics": "展示层为按蛋白行标准化的组均值 row Z-score 与检出率，不执行新的组间检验；组间差异方向以各对比火山图为准。",
        "interpretation": f"候选蛋白按 {group_col} 分组的平均丰度方向与检出可靠性；{module_bits}。",
        "boundary": BOUNDARY_MATRIX,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_enrichment_dotplot(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    top_terms = th.get("top_terms")
    databases = _databases_from_inputs(record)
    db_line = "、".join(databases) if databases else "GO/KEGG/Reactome（以 figure_manifest 记录的输入为准）"
    scale = (
        f"每个 namespace×对比×方向取 p.adjust 最小的 top {_fmt_num(top_terms)} 条目"
        if top_terms is not None
        else "每个 namespace×对比×方向取 p.adjust 最小的 top 条目"
    )
    return {
        "title": title or PLOT_TYPE_TITLES["mechanism_enrichment_dotplot"],
        "panels": [
            "点图：横轴为对比×方向（Contrast and direction，Up/Down 分列），纵轴为通路条目（namespace: Description）。",
            "点大小：Count，即该通路内检出的差异数。",
            "颜色通道：-log10(FDR)（p.adjust），值越大颜色越亮、校正后越显著。",
        ],
        "reading": (
            "同一通路条目在 Up 与 Down 列同时出现时，可比较其在两个方向的富集强度；"
            "优先阅读 Count 较大且 -log10(FDR) 较高的条目，避免只按单一指标排序。"
        ),
        "thresholds": [
            f"条目规模：{scale}。",
            f"输入数据库：{db_line}。",
            _sample_line(context),
        ],
        "statistics": "超几何富集检验（over-representation analysis），BH 校正后报告 p.adjust；点大小为通路内检出差异数（Count）。",
        "interpretation": (
            "汇总当前矩阵差异蛋白的离线富集证据；条目方向应与对应对比的火山图方向一致阅读，"
            "富集结果不改变矩阵内差异蛋白本身的方向与计数。"
        ),
        "boundary": BOUNDARY_ENRICHMENT,
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
    contrast_human = _human_contrast(info.get("contrast", "")) or "对应对比"
    direction = info.get("direction", "")
    direction_zh = {"upregulated": "上调", "downregulated": "下调"}.get(direction, direction or "对应方向")
    top_n = th.get("top_terms")
    source_line = "来源文件：" + os.path.basename(str((record.get("inputs") or [""])[0])) if record.get("inputs") else "来源文件：figure_manifest 未记录"
    scale = f"每个对比方向取 p.adjust 最小的 top {_fmt_num(top_n)} 条目" if top_n is not None else "每个对比方向取 p.adjust 最小的 top 条目"
    return {
        "title": title or f"{database} {contrast_human} {direction_zh}富集气泡图",
        "panels": [
            "气泡图：横轴 Rich Factor（通路内检出基因数 / 通路注释基因总数），纵轴为通路条目。",
            "点大小：Count（通路内检出的差异数）。",
            "颜色通道：-log10(adjusted p-value)。",
        ],
        "reading": (
            "越靠右表示该通路在差异蛋白中的占比越高；点越大表示检出差异数越多；"
            "Rich Factor 与 Count 应同时读取，避免高占比但检出数极少的条目被过度解读。"
        ),
        "thresholds": [
            f"条目规模：{scale}。",
            source_line + "。",
            _sample_line(context),
        ],
        "statistics": "超几何富集检验（over-representation analysis），BH 校正后报告 p.adjust。",
        "interpretation": (
            f"来自 {database} 的 {contrast_human} {direction_zh}差异蛋白集；"
            "解释机制方向时应与对应火山图的上下调方向一致阅读。"
        ),
        "boundary": BOUNDARY_ENRICHMENT,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_pathway_heatmap(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    caption = _clean_cell(record.get("caption"))
    term_match = re.search(r"`([^`]+)`", caption)
    term = term_match.group(1) if term_match else os.path.splitext(os.path.basename(file_name))[0]
    info = _parse_enrichment_bubble_name(file_name)
    contrast_human = _human_contrast(info.get("contrast", ""))
    group_col = context.get("group_col") or "分组"
    source_line = "来源文件：" + os.path.basename(str((record.get("inputs") or [""])[0])) if record.get("inputs") else "来源文件：figure_manifest 未记录"
    return {
        "title": title or f"通路基因热图：{term}",
        "panels": [
            f"热图：行为通路「{term}」匹配到当前矩阵的基因（Matched pathway genes），列为 {group_col} 分组。",
            "颜色通道：组均值 row Z-score（-2.5 至 2.5），红色为该组平均丰度较高、蓝色较低。",
        ],
        "reading": (
            "逐行读取基因在不同分组间的平均丰度方向；同一通路内方向一致的基因越多，"
            "该通路的整体方向越可信。"
        ),
        "thresholds": [
            f"蛋白集合：通路「{term}」注释基因中可匹配到当前矩阵的部分（匹配不到的基因不进入热图）。",
            source_line + "。",
            _sample_line(context),
        ],
        "statistics": "展示层为按蛋白行标准化的组均值 row Z-score，不执行组间检验；通路归属来自离线富集条目。",
        "interpretation": (
            f"通路「{term}」内基因在{(' ' + contrast_human + ' ') if contrast_human else '各对比'}方向下的平均丰度结构，"
            "与对应富集条目配套阅读。"
        ),
        "boundary": BOUNDARY_MIXED,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_upset(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    th = _resolve_thresholds(record, context)
    return {
        "title": title or PLOT_TYPE_TITLES["mechanism_dep_overlap_upset"],
        "panels": [
            "UpSet 式汇总：底部条形为各对比的差异数，主区条形为交集/独有蛋白数。",
            "数字标注：各集合的确切蛋白数。",
        ],
        "reading": (
            "先读底部各对比总量，再读主区：独有集合说明对比特异性，交集说明跨对比共享的差异机制。"
        ),
        "thresholds": [
            _threshold_sentence(th),
            _sample_line(context),
        ],
        "statistics": f"基于各对比 limma 双侧检验 + {_th_fdr(th)} 校正后的差异蛋白集合，再取交集/差集。",
        "interpretation": (
            "衡量候选机制在各对比间的共享与特异程度；交集大说明机制跨对比稳定，"
            "独有集大说明该对比携带独特信号。"
        ),
        "boundary": BOUNDARY_MATRIX,
        "paper_ref": _paper_ref_line(file_name, context),
    }


def _caption_contrast_evidence(file_name: str, title: str, record: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    contrast = _human_contrast(os.path.splitext(os.path.basename(file_name))[0].replace("mechanism_contrast_evidence_", ""))
    th = _resolve_thresholds(record, context)
    meta = record.get("caption_metadata") or {}
    omitted = [str(x) for x in (meta.get("omitted_panels") or [])]
    omitted_names = {
        "enrichment_terms": "富集条目子面板",
        "group_means": "组均值热图子面板",
        "volcano_evidence": "火山证据子面板",
        "top_proteins": "核心差异蛋白子面板",
    }
    panels = [
        f"单对比综合证据面板（{contrast}）：并排展示该对比的差异蛋白方向、组均值与富集要点。",
        "各子面板的坐标与颜色含义与对应单图一致（火山/均值/富集）。",
    ]
    if omitted:
        names = "、".join(omitted_names.get(key, key) for key in omitted)
        panels.append(f"面板省略说明：{names} 因该对比无匹配数据未绘制，不使用灰字占位图。")
    return {
        "title": title or f"{contrast} 综合证据面板",
        "panels": panels,
        "reading": (
            "把同一对比的差异证据与富集解释放进一张图，先读差异方向，再读富集条目；"
            "两列证据方向冲突时应回到对应单图核对。"
        ),
        "thresholds": [
            _threshold_sentence(th),
            _sample_line(context),
        ],
        "statistics": f"子图沿用 limma 双侧检验（{_th_fdr(th)} 校正）与超几何富集检验结果，不引入新的检验。",
        "interpretation": f"{contrast} 的差异与机制证据整合视图；与该对比火山图和富集气泡图结论应一致。",
        "boundary": BOUNDARY_MIXED,
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
    boundary = BOUNDARY_ENRICHMENT if category.startswith("Enrichment") else BOUNDARY_MATRIX
    interp = "本图的面板结构以图内轴标签与图例为准；数字口径见上方阈值行，并与主报告对应任务章节阅读。"
    if manifest_caption:
        interp += f"figure_manifest 记录：{manifest_caption}"
    return {
        "title": title or _clean_cell(record.get("plot_type")) or file_name,
        "panels": [
            f"面板结构以图内轴标签与图例为准（figure_manifest 类别：{category or '未记录'}）。"
        ],
        "reading": "按图内轴标签与图例读取；数值通道与显著性含义见阈值与统计口径行。",
        "thresholds": [
            _threshold_sentence(th),
            _sample_line(context),
        ],
        "statistics": "统计口径以生成该图的 plot 函数记录为准（见 figure_manifest.json 该条目）。",
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
            return PLOT_TYPE_TITLES[plot_type]
        if plot_type == "volcano":
            return f"{_human_contrast(_volcano_contrast(file_name))} 火山图"
        if plot_type == "enrichment_bubble":
            info = _parse_enrichment_bubble_name(file_name)
            return f"{info.get('database', '').upper()} {info.get('contrast', '')} 富集气泡图"
        if plot_type == "mechanism_pathway_gene_heatmap":
            term_match = re.search(r"`([^`]+)`", _clean_cell(record.get("caption")))
            if term_match:
                return f"通路基因热图：{term_match.group(1)}"
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
        cap["no"] = f"图{idx}"
        cap["file_name"] = os.path.basename(file_name)
        cap["plot_type"] = plot_type
        # empty-cell discipline: every narrative field must carry text
        for field in ("panels", "reading", "thresholds", "statistics", "interpretation", "boundary", "paper_ref"):
            value = cap.get(field)
            if isinstance(value, list):
                cap[field] = [(_clean_cell(v) or "figure_manifest 未记录该信息。") for v in value]
                if not cap[field]:
                    cap[field] = ["figure_manifest 未记录该信息。"]
            elif not _clean_cell(value):
                cap[field] = "figure_manifest 未记录该信息。"
        records.append(cap)
    return records


def _overview_lines(context: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    diff = context.get("differential") or {}
    pval = _to_float(diff.get("p_thresh"))
    logfc = _to_float(diff.get("logfc_thresh"))
    fdr = _clean_cell(diff.get("fdr_method")) or "BH"
    if pval is not None and logfc is not None:
        lines.append(f"差异分析默认口径：adj.P.Val < {_fmt_num(pval)}；|log2FC| ≥ {_fmt_num(logfc)}（{fdr} 校正，limma 双侧检验）。")
    elif pval is not None or logfc is not None:
        lines.append(f"差异分析默认口径（部分记录）：adj.P.Val < {_fmt_num(pval) if pval is not None else '未记录'}；|log2FC| ≥ {_fmt_num(logfc) if logfc is not None else '未记录'}（{fdr} 校正）。")
    else:
        lines.append("差异分析默认口径：analysis_design 未记录阈值，逐图阈值见各图注的阈值行。")
    lines.append(_sample_line(context))
    group_col = context.get("group_col")
    if group_col:
        lines.append(f"主分组列：{group_col}（来自 analysis_design）。")
    generated_at = context.get("generated_at")
    lines.append(
        "数字来源：figure_manifest.json（含逐图 caption_metadata/params）与 evaluation_evidence 证据表"
        + (f"；manifest 生成时间 {generated_at}。" if generated_at else "。")
    )
    return lines


def render_figures_captions_md(dataset: str, records: Sequence[Dict[str, Any]], context: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.extend([
        f"# {dataset} 发表级图注",
        "",
        "本文档为每张关键图提供发表级中文图注，由 run 内 `figure_manifest.json`、`analysis_design` 与 `evaluation_evidence` 证据表自动生成；蛋白、通路与统计术语保留英文原文。",
        "",
        "## 阈值与统计总览",
        "",
    ])
    for line in _overview_lines(context):
        lines.append(f"- {line}")
    lines.extend([
        "",
        "## 证据边界图例",
        "",
        f"- {BOUNDARY_MATRIX}",
        f"- {BOUNDARY_ENRICHMENT}",
        "- extension：启发式推测和后续验证建议只作为可检验假设，不进入图注数字。",
        "",
        "## 图注目录",
        "",
        "| 图号 | 标题 | 图片文件 |",
        "|---|---|---|",
    ])
    for record in records:
        lines.append(f"| {record['no']} | {record['title']} | `{record['file_name']}` |")
    lines.append("")
    for record in records:
        lines.extend([
            f"## {record['no']} {record['title']}",
            "",
            f"- 图片文件：`{record['file_name']}`",
            "- 面板说明：",
        ])
        for panel in record["panels"]:
            lines.append(f"  - {panel}")
        lines.extend([
            f"- 如何阅读：{record['reading']}",
            "- 阈值与样本量：",
        ])
        for threshold in record["thresholds"]:
            lines.append(f"  - {threshold}")
        lines.extend([
            f"- 统计口径：{record['statistics']}",
            f"- 一句话解读：{record['interpretation']}",
            f"- 证据边界：{record['boundary']}",
            f"- 对应原论文图：{record['paper_ref']}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def render_brief_figures_md(
    dataset: str,
    records: Sequence[Dict[str, Any]],
    links: Sequence[str],
    title_suffix: str = "关键图表嵌入版",
    intro: str = "",
) -> str:
    lines: List[str] = [f"# {dataset} {title_suffix}", ""]
    lines.append(intro or (
        "每张图只保留图号、标题与嵌图；逐图的面板说明、阈值、统计口径、解读与证据边界见 "
        "`figures_captions.md` 对应小节。"
    ))
    lines.append("")
    for record, link in zip(records, links):
        heading = f"{record['no']} {record['title']}"
        lines.extend([
            f"## {heading}",
            "",
            f"![{heading}]({link})",
            "",
            f"图注详见 `figures_captions.md` §{record['no']}。",
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
        title_suffix="关键图表本机预览版",
        intro=(
            "本文件使用本机绝对路径，主要用于解决某些 Markdown 预览器不能解析相对图片路径的问题；"
            "对外发送时优先使用 `figures.md`。逐图图注见 `figures_captions.md`。"
        ),
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
            title = PLOT_TYPE_TITLES.get(stem)
            if stem.endswith("_volcano_plot"):
                title = f"{_human_contrast(stem[: -len('_volcano_plot')])} 火山图"
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
