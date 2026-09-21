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

TITLE_PREFIX = "# scProteoAgent 分析报告"


# --------------------------------------------------------------------------- reader-facing wording
# Internal provenance / field vocabulary that must not reach a finished report. The model answers
# are written for an internal reader, so the assembler translates their scaffolding into plain
# Chinese *before* the text becomes part of the deliverable. Science is untouched: values, protein
# ids, evidence ids and directions are preserved verbatim.
SOURCE_LABELS = {
    "current_matrix": "当前矩阵",
    "offline_enrichment": "离线富集结果",
    "external_annotation": "外部注释",
    "external_literature": "外部文献",
    "drug_database": "药物数据库",
    "dataset_provided": "数据集提供",
}

PROVENANCE_TOKENS = ("current_matrix", "offline_enrichment", "external_annotation",
                     "external_literature", "drug_database", "dataset_provided")

FIELD_TOKENS = ("置信度", "置信度：", "证据来源", "证据：", "证据", "边界", "证据ID", "证据 ID")


def _translate_tokens(text: str) -> str:
    """Reader-facing prose: translate internal scaffolding, never touch numbers or ids."""
    out = text
    # a source table that records an exact zero usually means underflow, not a true zero p-value
    out = re.sub(r"(?:FDR|adj\.P(?:\.Val)?|P\.Value|P)\s*[=＝]\s*0(?![.\d])",
                 "在原差异表中记为 0", out)
    # confidence vocabulary (internal English level -> readable Chinese level)
    # a CJK range word right after the English level defeats \b, so the 至/到 form needs its own rule
    out = re.sub(r"置信度\s*[:：]?\s*low\s*(?:至|到|~|-|–|—|/)\s*moderate",
                 "证据强度：较低至中等", out)
    out = re.sub(r"置信度\s*[:：]?\s*moderate\s*(?:至|到|~|-|–|—|/)\s*low",
                 "证据强度：中等至较低", out)
    out = re.sub(r"置信度\s*[:：]?\s*(?:low[-–/]?moderate|moderate[-–/]?low)\b",
                 "证据强度：较低至中等", out)
    out = re.sub(r"置信度\s*[:：]?\s*(moderate|medium)\b", "证据强度：中等", out)
    out = re.sub(r"置信度\s*[:：]?\s*low\b", "证据强度：较低", out)
    out = re.sub(r"置信度\s*[:：]?\s*high\b", "证据强度：较高", out)
    out = out.replace("置信度 moderate/low", "证据强度：较低至中等")
    out = out.replace("置信度 low–moderate", "证据强度：较低至中等")
    out = out.replace("置信度 low-moderate", "证据强度：较低至中等")
    out = out.replace("置信度 moderate", "证据强度：中等")
    out = out.replace("置信度 low", "证据强度：较低")
    out = out.replace("moderate/low", "中等/较低")
    out = out.replace("low–moderate", "较低至中等").replace("low-moderate", "较低至中等")
    out = re.sub(r"置信度\s*[:：]?\s*分别为\s*low\s*/\s*moderate",
                 "证据强度：分别为较低与中等", out)
    out = re.sub(r"置信度\s*[:：]?\s*分别为\s*moderate\s*/\s*low",
                 "证据强度：分别为中等与较低", out)
    out = re.sub(r"置信度\s*[:：]?\s*分别为\s*low\s*与\s*moderate",
                 "证据强度：分别为较低与中等", out)
    # provenance vocabulary
    out = out.replace("current_matrix", "当前矩阵")
    out = out.replace("offline_enrichment", "离线富集结果")
    out = out.replace("external_annotation", "外部注释")
    out = out.replace("external_literature", "外部文献")
    out = out.replace("drug_database", "药物数据库")
    out = out.replace("dataset_provided", "数据集提供")
    # remaining internal field names / values
    out = re.sub(r"needs_log_transform\s*[=＝]\s*(?:true|True|1)", "记录为需要 log 变换", out)
    out = out.replace("looks_logged=false", "记录为未确认已对数化")
    out = out.replace("already_processed", "已处理输入")
    out = out.replace("batch_corrected_claimed", "声称已批次校正")
    out = out.replace("adj.significant", "FDR 显著")
    out = out.replace("analysis_design", "分析设计")
    out = out.replace("n_sig=", "通过筛选的蛋白数=")
    out = out.replace("n_sig", "通过筛选的蛋白数")
    out = out.replace("raw P<0.05 = ", "原始 P<0.05 = ")
    out = out.replace("log2_likely", "可能为 log2 尺度（未记录变换）")
    out = out.replace("（type=synthesis）", "").replace("(type=synthesis)", "")
    out = out.replace("type=synthesis", "综合评述")
    out = out.replace("exploratory_not_fdr_significant", "探索性，未达到 FDR 显著")
    out = out.replace("inferred", "推断")
    out = out.replace("tested proteins", "检验蛋白数")
    out = out.replace("analysed_n_proteins=", "分析蛋白数=")
    out = out.replace("analysed_mean_missing_rate=", "分析后缺失率=")
    out = out.replace("n_samples=", "样本数=")
    out = out.replace("mean_score=", "模块平均分=")
    out = out.replace("n_up_display_group_a", "A 组较高数")
    out = out.replace("n_down_display_group_a", "B 组较高数")
    out = out.replace("n_a=", "A 组样本数=").replace("n_b=", "B 组样本数=")
    out = out.replace("contrast_estimable=", "对比可估计性=")
    out = out.replace("eta2_batch_pc_descriptive=", "批次主成分效应量=")
    out = out.replace("contrast_vif=", "对比 VIF=")
    out = out.replace("confidence=moderate", "证据强度：中等")
    out = out.replace("confidence=low", "证据强度：较低")
    out = out.replace("up_in_group_a", "在 A 组中较高")
    out = out.replace("down_in_group_a", "在 A 组中较低")
    out = out.replace("up_in_display", "显示方向较高")
    out = out.replace("down_in_display", "显示方向较低")
    # R17: presence state enums are machine values; the claim they carry is kept word for word
    out = out.replace("not_detected_in_available_matrix", "在可用矩阵中未检出")
    # R29: an unmapped symbol is a different state from an absent protein and must read differently.
    out = out.replace("symbol_not_matched_in_available_matrix",
                      "未建立符号映射（该符号未匹配到矩阵标识，不等同于未检出）")
    out = out.replace("symbol_not_mapped", "未建立符号映射")
    out = out.replace("not detected in available matrix", "在可用矩阵中未检出")
    out = out.replace("matrix_presence_only", "未进入统计检验")
    # remaining English direction vocabulary inside Chinese sentences
    out = re.sub(r"\b(?:up|higher) in ([A-Za-z0-9_.\-]+)", r"在\1中较高", out)
    out = re.sub(r"\b(?:down|lower) in ([A-Za-z0-9_.\-]+)", r"在\1中较低", out)
    out = out.replace("display 方向", "报告方向")
    out = re.sub(r"当前矩阵\s*\+\s*[A-Za-z0-9_.\-]+\s*[:：]", "", out)
    # provenance label followed by an evidence id: keep the id, drop the scaffolding
    out = re.sub(r"(?:%s)\s*/\s*(E-[A-Z])" % "|".join(PROVENANCE_TOKENS), r"证据 \1", out)
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
    pointer = "（结论边界见本报告「结论边界与方法局限」）"
    out = re.sub(r"[（(]\s*(?:证据)?边界\s*(?:参见|见)?\s*boundary\s*[）)]", pointer, text, flags=re.I)
    out = re.sub(r"[（(]\s*boundary\s*[）)]", pointer, out, flags=re.I)
    return re.sub(r"\bboundary\b", "结论边界", out, flags=re.I)


def _clean_parens(text: str) -> str:
    """Compact the model's internal source annotations into one reader-facing 证据 marker."""
    def repl(match):
        inner = match.group(1)
        if not any(token in inner for token in PROVENANCE_TOKENS + FIELD_TOKENS):
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
        body = "证据 " + "；".join(parts) if parts else ""
        return "（%s）" % body if body else ""
    return re.sub(r"（([^（）]{0,400})）", repl, text)


def polish_for_reader(text: str) -> str:
    """Strip internal prefixes, file names and directive records from reader-facing prose."""
    out = text
    for pattern, label in FILE_LABELS:
        out = re.sub(r"[A-Za-z0-9_/\.-]*" + pattern + r"(?:\.csv|\.json|\.md|\.yaml|\.jsonl)?", label, out)
    for pattern, replacement in EVIDENCE_REPAIRS:
        out = re.sub(pattern, replacement, out)
    for prefix in INTERNAL_PREFIXES:
        out = out.replace(prefix, "")
    out = re.sub(r"(证据)\1", r"\1", out)
    for pattern in DIRECTIVE_PATTERNS:
        if re.search(pattern, out):
            out = re.sub(pattern, "", out)
            out = out.rstrip(" 。；") + "。" + DIRECTIVE_NOTE
    out = re.sub(r"（\s*）", "", out)
    out = re.sub(r"[ 	]{2,}", " ", out)
    out = re.sub(r"\s+([，。；：])", r"", out)
    out = re.sub(r"([，。；：])\s*", r"", out)
    return out.strip(" ，；：")


FILE_LABELS = (
    (r"ProteinQuant_Filtered", "过滤后定量矩阵"),
    (r"ProteinQuant_ComBat", "批次标定后定量矩阵"),
    (r"ProQuant_Normalized", "归一化定量矩阵"),
    (r"ProteinQuant", "蛋白定量矩阵"),
    (r"SampleInfo_Filtered", "过滤后样本信息表"),
    (r"SampleInfo", "样本信息表"),
    (r"core_story_evidence", "差异检验与故事线摘要"),
    (r"candidate_protein_evidence", "候选蛋白表"),
    (r"curated_module_group_summary", "模块分组摘要"),
    (r"curated_module_sample_scores", "样本级模块评分"),
    (r"group_composition_qc", "分组构成与质量控制表"),
    (r"group_cross_table_[A-Za-z0-9_]+", "分组交叉表"),
    (r"dataset_recipe_evidence", "数据集分析配置"),
    (r"scoring_standard_coverage", "任务覆盖检查表"),
    (r"candidate_exclusion_trace", "候选排除记录"),
    (r"confounding_adjusted_effects", "混杂校正效应"),
    (r"confounding_estimability", "可估计性检查"),
    (r"stratified_contrast_summary", "分层对比汇总"),
    (r"differential_[A-Za-z0-9_]+", "差异检验摘要"),
    (r"analysis_design\.used", "分析设计记录"),
    (r"evidence_ledger", "证据来源记录"),
    (r"figure_index", "图册索引"),
)
EVIDENCE_REPAIRS = (
    (r'\uff08\s*\u8bc1\u636e\u6765\u6e90\uff1a?\s*\u5f53\u524d\u77e9\u9635\s*\uff09', '\uff08\u8bc1\u636e\uff1a\u672c\u8f6e\u5dee\u5f02\u5206\u6790\uff09'),
    (r'\uff08\s*\u8bc1\u636e\u5f53\u524d\u77e9\u9635\s*\uff09', '\uff08\u8bc1\u636e\uff1a\u672c\u8f6e\u5dee\u5f02\u5206\u6790\uff09'),
    (r'\u5f53\u524d\u77e9\u9635\s*[:\uff1a]', ''),
    # R16: the catch-all must not claim a specific role for a file it cannot name; run after
    # FILE_LABELS so every known table keeps its own reader label instead of collapsing into one.
    (r'(?<![A-Za-z0-9_])[A-Za-z0-9_-]+\.(?:csv|json|jsonl|yaml|md)', '\u9644\u968f\u7ed3\u679c\u8868'),
)
INTERNAL_PREFIXES = (
    "当前矩阵：", "当前矩阵:",
    "（证据当前矩阵：",
)
DIRECTIVE_NOTE = "模块分数与样本级评分仅作为方向性证据，不等同于显著性检验。"
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
    return "（来源：%s）" % "、".join(parts) if parts else ""


def rewrite_for_reader(text: str) -> str:
    """One place where model scaffolding becomes finished-report prose."""
    out = text.strip()
    # R2: an evidence marker that carries no readable source is worse than no marker
    out = re.sub(r"\uff08?\s*evidence\s*[:\uff1a]\s*\u5f53\u524d\u77e9\u9635\s*\uff09?", "", out)
    out = re.sub(r"\uff08\s*\u8bc1\u636e\uff1a\u5f53\u524d\u77e9\u9635\s*\uff09", "", out)
    out = re.sub(r"\uff08\s*\u8bc1\u636e\s*\uff1a\s*\uff09", "", out)
    # R16: the model writes the annotation with an ASCII colon as often as a full-width one, and the
    # internal field name can appear outside a parenthesis. Both are the same scaffolding defect.
    out = re.sub(r"evidence[_\s]?ids", "证据清单", out, flags=re.I)
    out = re.sub(r"evidence\s*文件", "证据文件", out, flags=re.I)
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
        return "未记录"
    if isinstance(value, float):
        return "%.4g" % value
    return str(value)


def _cell(value: Any) -> str:
    """Markdown-safe cell text (escape the pipe that would otherwise break the table grid)."""
    return str(value if value is not None else "").replace("|", "\\|")


# --------------------------------------------------------------------------- conditions
UNIT_REASON_ZH = (
    ("no experimental-unit column", "未声明实验单位列，元数据中也没有可识别的个体列"),
    ("the design has no residual degrees of freedom", "设计没有剩余自由度，处理效应与个体效应无法分离"),
    ("matched by name only", "个体列仅按名称匹配，未在分析设计中声明"),
    ("no unit-level test is defensible", "该对比不支持个体层检验"),
    ("fewer than", "某一臂或两臂共同个体数不足"),
)


def _unit_reason_zh(record: Dict[str, Any]) -> str:
    """Reader-facing one-liner for a blocked unit-level analysis."""
    blob = " | ".join(str(item) for item in (record.get("reasons") or [])).lower()
    for needle, zh in UNIT_REASON_ZH:
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
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                return text.split("（")[0].strip()
    for item in record.get("reasons") or []:
        text = str(item).strip()
        if text and not text.startswith("已排除") and any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return text.split("（")[0].strip()
    return "运行记录未给出可归类的阻断原因"


UNIT_SOURCE_ZH = {
    "analysis_design.pair_col_auto_inferred": "自动生成的分析设计中的单位列（按列名匹配，未由研究者确认）",
    "analysis_design.pair_col": "分析设计声明的实验单位列",
    "column_name_match_unconfirmed": "按列名匹配推断，未在分析设计中确认",
    "absent": "未记录",
}


def _unit_source_label(value: Any) -> str:
    text = str(value or "")
    return UNIT_SOURCE_ZH.get(text, text or "未记录")


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
        scale, scale_note = "log2", "已记录 log2 变换"
    elif t_log.get("applied") is True:
        scale = "log2"
        scale_note = ("按本轮执行记录：%s（依据：%s；记录来源：%s）"
                      % (t_log.get("form") or "对数变换",
                         t_log.get("applied_basis") or "未记录",
                         (transform.get("record_source") if isinstance(transform, dict) else "")
                         or "未记录"))
    elif str(matrix.get("matrix_state") or "").lower() in {"logged", "log2"}:
        scale, scale_note = "log2", "矩阵状态记录为 %s" % matrix.get("matrix_state")
    elif matrix.get("looks_logged") is True:
        scale, scale_note = "log2_likely", "仅按数值范围推断，未记录变换"
    else:
        scale, scale_note = "unknown", "未记录尺度"
    return {"p_thresh": p_thresh, "logfc_thresh": logfc_thresh,
            "fdr_method": diff.get("fdr_method"), "contrasts": diff.get("contrasts") or [],
            "matrix_state": matrix.get("matrix_state"), "scale": scale, "scale_note": scale_note,
            "group_col": design.get("group_col"), "sample_id_col": design.get("sample_id_col"),
            "batch_col": design.get("batch_col"), "pair_col": design.get("pair_col")}


def threshold_text(cond: Dict[str, Any]) -> str:
    p, logfc = cond.get("p_thresh"), cond.get("logfc_thresh")
    fdr = cond.get("fdr_method") or "BH"
    if p is None or logfc is None:
        return "阈值未记录（parameters.json 缺 differential 阈值）"
    return "adj.P<%s 且 \\|logFC\\|>%s（多重校正：%s）" % (_fmt(p), _fmt(logfc), _fmt(fdr))


def effect_column(cond: Dict[str, Any]) -> str:
    return "logFC" if cond.get("scale") == "log2" else "效应量（尺度未确认）"


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
    groups = rec.get("groups") or ["A 组", "B 组"]
    lines = ["核心对比（本表方向：正值表示 %s 较高，按「%s」）"
             % (_cell(groups[0]), _cell(rec.get("display_contrast"))),
             "| 对比 | 通过筛选的蛋白数 | %s 较高 | %s 较高 | 统计口径 |" % (_cell(groups[0]), _cell(groups[1])),
             "|---|---:|---:|---:|---|",
             "| %s | %s | %s | %s | %s |" % (_cell(rec.get("display_contrast")), _fmt(rec.get("n_sig")),
                                             _fmt(rec.get("n_up_display_a")), _fmt(rec.get("n_down_display_a")),
                                             threshold_text(cond))]
    if rec.get("inverted"):
        lines += _note_once(
            "inverted_%s" % _cell(rec.get("source_contrast")),
            "符号约定：本表按「%s」显示，底层差异表为 %s，其中 logFC 符号与本表相反；"
            "正文数值一律按「%s」方向给出（正值为 %s 较高、负值为 %s 较高），"
            "仅当句子明确引用源文件 %s 时按该文件的原始符号读取。"
            % (_cell(rec.get("display_contrast")), _cell(rec.get("source_contrast")),
               _cell(rec.get("display_contrast")), _cell(groups[0]), _cell(groups[1]),
               _cell(rec.get("source_contrast"))),
            "符号约定：本表方向同上（按「%s」；源文件 %s 的符号相反）。"
            % (_cell(rec.get("display_contrast")), _cell(rec.get("source_contrast"))))
    return lines


def _candidate_block(task: Dict[str, Any], index: Dict[str, Any], cond: Dict[str, Any],
                     limit: Optional[int] = None, total: Optional[int] = None) -> List[str]:
    keys = task["candidates"] if limit is None else task["candidates"][:limit]
    if not keys:
        return ["（本对比没有候选蛋白记录）"]
    effect = effect_column(cond)
    # R17: the caption says how many rows exist instead of promising more than the table shows
    shown_total = total if total is not None else len(task["candidates"])
    caption = ("候选蛋白（共 %d 条）" % shown_total if shown_total <= len(keys)
               else "候选蛋白（共 %d 条，本节列前 %d 条，完整清单见「证据表（逐条）」）"
                    % (shown_total, len(keys)))
    lines = [caption,
             "| 候选 | 蛋白组 | 显示方向 %s | 源表 %s | adj.P.Val | 方向 | 差异表 |" % (_cell(effect), _cell(effect)),
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
                       if item.get("display_contrast") == display), None) or ["第一臂", "第二臂"]
        lines += _note_once(
            "cand_direction_%s" % source,
            "方向读法：显示方向列按「%s」（正值为 %s 较高），源表列按反向写法「%s」（正值为 %s 较高）；"
            "两列数值符号相反，但指的是同一个对比。" % (display, _cell(groups[0]), source, _cell(groups[1])),
            "方向读法同前：显示方向列按「%s」，源表列为反向写法「%s」。" % (display, source))
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
        lines.append("注：%s 的效应量绝对值未超过当前筛选口径的 0.25 效应阈值，列出供完整性参考，"
                     "不计入通过筛选的蛋白集合（口径见上表「统计口径」列）。" % "、".join(below))
    return lines


def _module_block(task: Dict[str, Any], index: Dict[str, Any], limit: Optional[int] = None,
                  total: Optional[int] = None) -> List[str]:
    keys = task["modules"] if limit is None else task["modules"][:limit]
    if not keys:
        return []
    # R17: the member hash is machine metadata (it stays in the manifest), not reader content.
    shown_total = total if total is not None else len(task["modules"])
    caption = ("模块证据（共 %d 条；Δ 方向同本任务组名）" % shown_total if shown_total <= len(keys)
               else "模块证据（共 %d 条，本节列前 %d 条；Δ 方向同本任务组名）"
                    % (shown_total, len(keys)))
    lines = [caption,
             "| 模块 | A 组均值 | B 组均值 | Δ（A 组均值 − B 组均值） | 匹配蛋白数 |",
             "|---|---:|---:|---:|---:|"]
    for key in keys:
        rec = index["modules"][key]
        lines.append("| %s | %s | %s | %s | %s |"
                     % (_cell(key[1]), _fmt(rec.get("mean_a")), _fmt(rec.get("mean_b")),
                        _fmt(rec.get("delta")), _fmt(rec.get("n_matched"))))
    if keys:
        lines += _note_once(
            "module_no_fdr",
            "注：Δ 为 A 组均值减 B 组均值，方向与上表同一组名；模块分数是成员蛋白评分的汇总，"
            "没有 FDR，因此只能作为方向性证据，不能写成显著富集。",
            "注：Δ 方向同组名；模块分数无 FDR，只作方向性证据（口径见「结论边界与方法局限」）。")
    return lines


def _verdict_line(task: Dict[str, Any], cond: Dict[str, Any]) -> str:
    rec = task["contrast"]
    groups = rec.get("groups") or ["A 组", "B 组"]
    return ("判定：%s 中通过筛选口径的蛋白 %s 个（口径：%s），其中 %s 较高 %s 个、%s 较高 %s 个。"
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
    column = str(unit.get("analysis_unit") or "未记录")
    source = _unit_source_label(unit.get("unit_column_source"))
    effect_name = str(unit.get("effect_column") or "效应量")
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
            direction_text = "；本段数值按「%s − %s」方向给出（正值为 %s 侧较高）" % (arms[0], arms[1], arms[0])
            if _reversed_arms(key or "", rec.get("display_contrast")):
                direction_text += "，与本节标题的写法相反，标题方向的正负号需取反"
        route_key = str(unit.get("analysis_route"))
        estimand_zh = ("个体内均值差（同一实验单位在两臂的均值差）" if route_key == "within_unit_paired"
                       else "个体间均值差（每条臂只使用只在该臂观测的实验单位）")
        tested = _fmt(unit.get("n_proteins_tested"))
        not_tested = int(unit.get("n_proteins_not_tested") or 0)
        joint_text = ("按联合口径（校正 P 值 < 0.05 且 |效应量| > %s）%s 条通过" % (_fmt(thr), _fmt(joint))
                      if joint is not None else "未套用联合效应阈值")
        if route_key == "within_unit_paired":
            design_text = ("使用两臂都有观测的 %s 个独立单位做配对 t 检验；只在一个臂出现的单位不参与配对"
                           % _fmt(unit.get("n_pairs") or unit.get("n_units_shared")))
        else:
            design_text = ("共享单位不足以配对，改用只在单臂观测的独立个体（A 臂 %s 个、B 臂 %s 个）"
                           "做个体间 Welch 检验，共享单位被排除在外、不重复计数"
                           % (_fmt(unit.get("n_units_a_only")), _fmt(unit.get("n_units_b_only"))))
        lines.append("**实验单位敏感性分析**：以「%s」为分析单位（%s；估计目标：%s%s）。本对比%s；"
                     "进入检验的蛋白组 %s 个%s，按 FDR 口径 %s 条通过，%s，最小校正 P 值 %s。"
                     % (_cell(column), source, estimand_zh, direction_text, design_text, tested,
                        ("（另有 %d 个蛋白行因有效单位不足标记为未检验）" % not_tested) if not_tested else "（全部可检验）",
                        _fmt(fdr), joint_text, _fmt(unit.get("min_adj_p_value"))))
        lines.append("观察级检验仍是参照估计：两者以不同的单位计权（个体均值 vs 单细胞观测），"
                     "估计目标不同，数值不可直接互换。")
    else:
        lines.append("**实验单位敏感性分析**：本对比不做个体级检验（%s）：%s"
                     % (source, _unit_reason_zh(unit)))
    if str(unit.get("unit_column_source") or "") in ("column_name_match_unconfirmed",
                                                      "analysis_design.pair_col_auto_inferred"):
        lines.append("注：单位列不是研究者确认的字段，而是由自动生成的分析设计按列名匹配得到；"
                     "“每个标签对应一个独立生物学个体”在本轮是标签层假定，本节数值按标签层探索性分析解读，"
                     "不作为个体层确认性结论。")
    if excluded:
        lines += _note_once(
            "unit_excluded_units",
            "混合来源或未确认来源的单位已单列并排除在独立个体之外：%s。「是否为独立生物学个体」按标签语义事先判定，"
            "不按检验结果挑选。" % "、".join(_cell(x) for x in excluded),
            "混合来源／未确认来源单位同前，已排除在独立个体之外。")
    reason_needed = [str(item) for item in (unit.get("reasons") or [])
                     if "分布不均" in str(item) or "矩阵匹配后" in str(item)]
    for item in reason_needed[:2]:
        lines.append("注：%s" % item)
    return lines


def _data_section(run: Path, cond: Dict[str, Any],
                  run_evidence: Optional[Dict[str, Any]] = None) -> List[str]:
    contrasts = "; ".join("%s（%s vs %s）" % (c.get("name"), c.get("group_a"), c.get("group_b"))
                          for c in (cond.get("contrasts") or [])[:8])
    state = _fmt(cond.get("matrix_state"))
    scale = cond.get("scale")
    if scale == "log2_likely":
        scale = "log2（推断，未记录变换）"
        scale_note = "仅按数值范围推断；运行记录只记录了输入的判断字段，未记录实际变换"
    elif scale == "unknown":
        scale = "未记录"
        scale_note = "运行记录只记录了输入的判断字段，未记录实际变换，无法确认检验空间的尺度"
    else:
        scale_note = cond.get("scale_note") or ""
    lines = ["- 矩阵状态：%s；效应尺度：%s（%s）" % (state, _fmt(scale), scale_note),
             "- 分组列：%s；样本 ID 列：%s；批次列：%s" % (_fmt(cond.get("group_col")),
                                                         _fmt(cond.get("sample_id_col")),
                                                         _fmt(cond.get("batch_col"))),
             "- 差异分析阈值：%s" % threshold_text(cond),
             "- 本轮对比：%s" % (contrasts or "见差异表")]
    if run_evidence:
        lines += _run_method_summary_lines(run, cond, run_evidence)
    else:
        lines.append("- 证据表与图件路径见文末清单；数值以确定性表格为准。")
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
        core = ("%s的%s%s" % (head, gap, tail)) if (head and tail) else (head or core)
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
        return TITLE_PREFIX
    return ("%s报告" % core) if core.endswith("分析") else ("%s分析报告" % core)


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
                            "status": "未执行", "shape": "", "matched_key": "",
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
        view["status"] = "解析失败"
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
        view["status"] = "无本对比条目" if any(view["buckets"].values()) else "有效空结果"
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
                 "status": "已计算" if view["terms"] else "有效空结果"})
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
    up_label = "上调方向（%s 较高）" % arms[0] if len(arms) == 2 else "上调方向"
    down_label = "下调方向（%s 较高）" % arms[1] if len(arms) == 2 else "下调方向"
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
    lines = ["### 富集方向提示（探索性）"]
    lines += _note_once(
        "enrichment_hint",
        "本节为探索性方向提示：所列通路来自预定义基因集的集合重叠结果，未经多重检验校正，无 FDR/q 值，"
        "不得作为显著性证据，仅用于假设生成。每个数据库每个方向取种子基因数最多的一条。",
        "探索性方向提示：集合重叠结果，无 FDR/q，不作显著性证据（口径见「结论边界与方法局限」）。"
        "各节按同一规则各取一条。")
    if view.get("reversed"):
        # R21/F2: same comparison, opposite arm order between the overlap file and the report table.
        # The labels above follow the file, so the reader has to be told which order they follow.
        lines += ["本节的通路条目取自富集结果文件（对比键「%s」，该键下正值为 %s 较高）；本任务正文的方向基准为「%s」。"
                  "两者臂序相反，因此下表方向列逐一写出该方向较高的一组，不沿用正文基准。" % (_cell(key_name), _cell(arms[0] if arms else ""), _cell(names[0] if names else ""))]
    lines += ["",
              "| 数据库 | 方向 | 通路/语义条目 | 该条种子基因数 | 代表成员 |",
              "|---|---|---|---:|---|"]
    for database, label, term, count, members in rows[:limit]:
        lines.append("| %s | %s | %s | %d | %s |" % (_cell(database), _cell(label), _cell(term),
                                                               count, _cell(", ".join(members[:6]))))
    lines.append("")
    return lines


DENSITY_CAP = 4
MATRIX_ALIASES = ("\u672c\u6b21\u5206\u6790\u7684\u6570\u636e\u77e9\u9635", "\u603b\u4e30\u5ea6\u77e9\u9635", "\u8be5\u6570\u636e")


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
            out.append(MATRIX_ALIASES[(counter["n"] - cap - 1) % len(MATRIX_ALIASES)])
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def _reader_asset_label(name: str) -> str:
    """Reader-facing name for an evidence asset, derived from the existing label map."""
    differential = re.match(r"differential_(.+)\.csv$", name)
    if differential:
        return "差异检验结果表（%s）" % differential.group(1)
    if name.startswith("candidate_exclusion_trace"):
        return "候选排除记录"
    if name.startswith("contrast_concordance"):
        return "对比一致性表"
    if name.startswith("stratified_contrast_summary"):
        return "分层对比摘要表"
    for pattern, label in FILE_LABELS:
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
    out += ["### 分层对比摘要", ""]
    out.append("| 对比 | 分层变量 | 层 | 第一臂 n | 第二臂 n | 检验蛋白 | 通过筛选 | 结论 |")
    out.append("|---|---|---|---:|---:|---:|---:|---|")
    for row in named[:10]:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            _cell(row.get("contrast")), _cell(row.get("stratifier")), _table_cell(row.get("level"), 120),
            _fmt(row.get("n_a")), _fmt(row.get("n_b")), _fmt(row.get("tested_proteins")),
            _fmt(row.get("n_sig")), _table_cell(row.get("verdict"), 80)))
    out += ["", "分层结果按单个分层变量分别分组，不等于同时控制多个变量；主结论仍以主模型为准。", ""]
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
    out = ["### 候选排除轨迹", "",
           "下列候选在进入统计前被排除，原因按来源表逐条记录；未检出不等于样本中不存在。", ""]
    if seen is not None:
        seen['done'] = True
    if not counted:
        out += ["本轮记录中不含蛋白级排除：被记录的未解析符号为分组与实验设计术语（例如分组名、样本表列名），"
                "不是蛋白候选，因此本节不提供候选层面的排除结论。", ""]
        return out
    out += ["| 候选 | 记录原因 | 记录数 |", "|---|---|---:|"]
    for (name, reason), count in counted.most_common(limit):
        out.append("| %s | %s | %d |" % (_cell(name), _cell(reason), count))
    out += ["", "排除记录数统计的是记录条数，不是蛋白数量。", ""]
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
        return "小于 1e-300"
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
        return ("该方向进入检验的蛋白只有 %s 个，少于最少 %s 个的要求，因此本轮未执行该检验"
                "（保留未执行状态，未用替代方法补算）。" % (match.group(1), match.group(2)))
    if re.fullmatch(r"q<=0\.05", text):
        return "该对比没有可检验的条目。"
    return _cell(text)


DOSE_DIRECTION_LABELS = {"falling with dose": "随剂量下降", "rising with dose": "随剂量上升",
                         "flat": "无单调趋势", "no trend": "无单调趋势", "flat with dose": "随剂量不单调"}


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
            return "随剂量上升"
        if value < 0:
            return "随剂量下降"
        return "无单调趋势（ρ=0）"
    text = str(raw or "").strip()
    return DOSE_DIRECTION_LABELS.get(text.lower(), text or "方向未记录")


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
    out = ["### 剂量趋势检验（Spearman）", ""]
    if modules:
        out += ["| 药物 | 分层 | 模块 | 样本数 | 剂量步 | ρ | p | q(BH) | 方向 |",
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
                    "模块方向翻转的解释：%s 的 %s 模块在剂量轴上并不单调（%s），两个分层的方向相反且都未达 q≤0.05，"
                    "因此高低剂量差值的符号相反应登记为未解释的观察，而不是单一机制的结论。"
                    % (drug, name, detail),
                    "模块方向翻转：%s 的 %s 模块在分层间符号相反（%s），均未达 q≤0.05；"
                    "按未解释的观察登记，不解释为单一机制。" % (drug, name, detail))
                out.append("")
    if proteins:
        significant = [row for row in proteins if str(row.get("significant")) == "yes"]
        usable = [row for row in proteins if str(row.get("rho") or "").strip()]
        out.append("蛋白水平：每个药物 × 分层单独检验并各自做 BH 校正，共检验 %d 个蛋白，其中 %d 个在 q≤0.05 下"
                   "呈单调剂量趋势。" % (len(proteins), len(significant)))
        top = sorted(usable, key=lambda row: (float(row.get("q_value") or 1),
                                              -abs(float(row.get("rho") or 0))))[:6]
        if top:
            out.append("")
            out += ["| 蛋白 | 药物 | 分层 | 样本数 | ρ | p | q(BH) |", "|---|---|---|---:|---:|---:|---:|"]
            for row in top:
                out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                    _cell(row.get("name")), _cell(row.get("drug")), _cell(row.get("stratum")),
                    _fmt(row.get("n_samples")), _fmt(row.get("rho")), _reader_prob(row.get("p_value")),
                    _reader_prob(row.get("q_value"))))
        out.append("")
    for row in skipped:
        out.append("注：%s / %s %s" % (_cell(row.get("drug")), _cell(row.get("stratum")),
                                      _reader_note(row.get("note"))))
        out.append("")
    out.append("剂量趋势以样本为观测单位（对照、低剂量、高剂量三步，分池与单细胞分层各自检验）；"
               "模块分数只作方向性佐证，不能替代蛋白水平的趋势检验。")
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
        return "adj.P\u2264%s \u4e14 |logFC|\u2265%s" % (match.group(1), match.group(2))
    return _cell(definition)


DIRECTION_LABELS = {"upregulated": "\u4e0a\u8c03", "downregulated": "\u4e0b\u8c03"}


def _is_fallback_query(row: Dict[str, Any]) -> bool:
    """The fallback query (top |logFC| share) exists only where the significant set was too small."""
    return str(row.get("query_definition") or "").startswith("fallback")


def _fallback_contrast_text(rows, index) -> str:
    """Name every contrast that used the fallback query, with its own significant-set size."""
    parts = []
    for name in sorted({str(row.get("contrast") or "") for row in rows}):
        subset = [row for row in rows if str(row.get("contrast") or "") == name]
        sizes = _size_range(_fallback_significant_sizes(subset)) or "未记录"
        query_sizes = _size_range([row.get("query_size") for row in subset]) or "未记录"
        arms = _arm_labels_for(name, index)
        label = name
        if len(arms) == 2:
            label = "%s（%s 较高 / %s 较高）" % (name, arms[0], arms[1])
        parts.append("%s：该对比显著集 %s 个蛋白，兜底查询集 n=%s" % (label, sizes, query_sizes))
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
            label = DIRECTION_LABELS.get(str(row.get("direction")), _cell(row.get("direction")))
            if len(arms) == 2:
                label = "%s 较高" % (arms[0] if str(row.get("direction")) == "upregulated"
                                    else arms[1])
            per_direction.setdefault(label, set()).add(str(row.get("query_size")))
        sizes = "、".join("%s n=%s" % (label, "、".join(sorted(values)))
                          for label, values in sorted(per_direction.items()))
        return ["本小节的查询集为各对比中通过显著阈值的蛋白（%s）：%s。" % (rule, sizes)]
    if len(fallback_rows) == len(tested):
        # every contrast in this file fell back, so the dataset-wide wording is the accurate one
        lead = ("本小节的查询集为按 |logFC| 取前 10%% 的蛋白（本小节所列条目的查询集 n=%s，"
                "全部被检验的查询集 n=%s），"
                % (_size_range([row.get("query_size") for row in shown]),
                   _size_range([row.get("query_size") for row in tested])))
        significant_sizes = _size_range(_fallback_significant_sizes(fallback_rows))
        if significant_sizes:
            lead += "因本数据集通过显著阈值的蛋白仅 %s 个、不足以做标准的过表征分析。" % significant_sizes
        else:
            lead += "因本数据集通过显著阈值的蛋白过少，不足以做标准的过表征分析。"
        lead += "下列 q 值描述的是该查询集的富集情况，不等同于差异蛋白的富集显著性。"
        return [lead]
    # both kinds of query are in play: describe each scope instead of generalising one of them
    lines = []
    if shown_significant:
        lines.append("本小节所列条目来自两类查询集，须按对比分别理解。第一类使用显著集（%s），"
                     "本节所列条目的查询集 n=%s。"
                     % (rule, _size_range([row.get("query_size") for row in shown_significant])
                        or "未记录"))
    else:
        lines.append("本小节的查询集不是一个统一口径，须按对比分别理解：本节所列条目均来自兜底查询集。")
    scope_rows = shown_fallback or fallback_rows
    lines.append("兜底查询集用于该对比显著集过小的情形：%s；本节所列兜底条目查询集 n=%s。"
                 % (_fallback_contrast_text(scope_rows, index),
                    _size_range([row.get("query_size") for row in scope_rows]) or "未记录"))
    lines.append("下列 q 值只描述各自查询集的富集情况，不等同于差异蛋白的富集显著性；"
                 "某一对比的显著蛋白数不适用于整个数据集。")
    return lines


def _enrichment_ora_lines(run, index: Optional[Dict[str, Any]] = None):
    """Hypergeometric ORA with the detected-protein background; the query definition leads the section."""
    rows = _ext_csv_rows(run, "enrichment_ora.csv")
    if not rows:
        return []
    tested = [row for row in rows if str(row.get("p_value") or "").strip()]
    skipped = [row for row in rows if not str(row.get("p_value") or "").strip()]
    out = ["### \u6b63\u5f0f\u5bcc\u96c6\u68c0\u9a8c\uff08\u8d85\u51e0\u4f55\uff0c\u80cc\u666f=\u672c\u6b21\u68c0\u6d4b\u5230\u7684\u86cb\u767d\uff09", ""]
    if tested:
        shown = sorted(tested, key=lambda item: float(item.get("p_value") or 1))[:12]
        out += _ora_query_scope_lines(tested, shown, index)
        out.append("")
        out += ["| \u5bf9\u6bd4 | \u65b9\u5411 | \u5e93 | \u901a\u8def | \u547d\u4e2d | \u901a\u8def\u5728\u80cc\u666f\u4e2d | \u67e5\u8be2\u96c6 | p | q(BH) |",
                "|---|---|---|---|---:|---:|---:|---:|---:|"]
        for row in shown:
            # R17: name the arm the direction belongs to instead of a bare up/down label
            arms = _arm_labels_for(row.get("contrast"), index)
            direction_label = DIRECTION_LABELS.get(str(row.get("direction")), row.get("direction"))
            if len(arms) == 2:
                direction_label = "%s 较高" % (arms[0] if str(row.get("direction")) == "upregulated"
                                              else arms[1])
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _cell(row.get("contrast")),
                _cell(direction_label),
                _cell(row.get("namespace")),
                _cell(row.get("term")), _fmt(row.get("hits")), _fmt(row.get("term_size_in_background")),
                _fmt(row.get("query_size")), _reader_prob(row.get("p_value")), _reader_prob(row.get("q_value"))))
        background = _fmt(tested[0].get("background_size"))
        note = ("\u80cc\u666f\u96c6\u4e3a\u672c\u6b21\u5206\u6790\u5b9e\u9645\u68c0\u6d4b\u5230\u7684\u86cb\u767d\u7684\u57fa\u56e0\u7b26\u53f7\uff08\u53bb\u91cd\u540e N=%s\uff09\uff1ap \u4e3a\u8d85\u51e0\u4f55\u5c3e\u6982\u7387\uff0c"
                "q \u4e3a\u540c\u4e00\u6b21\u67e5\u8be2\u5185\u5168\u90e8\u88ab\u68c0\u9a8c\u6761\u76ee\u7684 BH \u6821\u6b63\u3002" % background)
        # R17: the background set and the tested matrix are different counts, and the report says so
        # instead of leaving two numbers that look like a contradiction.
        matrix_proteins = _first_record(read_json(Path(run) / "processed_proteins" / "limma_summary.json",
                                                  {}) or {}).get("n_proteins")
        if matrix_proteins is not None and str(_fmt(matrix_proteins)) != str(background):
            note += ("\u8be5\u80cc\u666f\u96c6\u6309\u672c\u6b21\u5b9e\u9645\u68c0\u6d4b\u5230\u7684\u86cb\u767d\u5bf9\u5e94\u7684\u57fa\u56e0\u7b26\u53f7\u53bb\u91cd\u540e\u7edf\u8ba1\uff08\u540c\u4e00\u57fa\u56e0\u7684\u591a\u884c\u6298\u53e0\u4e3a\u4e00\u6761\uff09\uff0c"
                     "\u56e0\u6b64\u4e0e\u7edf\u8ba1\u68c0\u9a8c\u6240\u7528\u77e9\u9635\u7684\u86cb\u767d\u884c\u6570 %s \u4e0d\u540c\uff1b\u4e24\u4e2a\u6570\u5b57\u53e3\u5f84\u4e0d\u540c\uff0c\u5f15\u7528\u65f6\u5199\u660e\u6307\u54ea\u4e00\u4e2a\u3002" % _fmt(matrix_proteins))
        out += ["", note, ""]
    for row in skipped[:3]:
        arms = _arm_labels_for(row.get("contrast"), index)
        direction_label = DIRECTION_LABELS.get(str(row.get("direction")), row.get("direction"))
        if len(arms) == 2:
            direction_label = "%s 较高" % (arms[0] if str(row.get("direction")) == "upregulated"
                                          else arms[1])
        out.append("注：%s / %s %s" % (_cell(row.get("contrast")), _cell(direction_label),
                                      _reader_note(row.get("note"))))
        out.append("")
    return out


def _concordance_lines(run, limit=None):
    """R3: agreement between contrasts."""
    import csv as _csv
    path = Path(run) / "evaluation_evidence_ext" / "contrast_concordance.csv"
    if not path.exists():
        return ["### 对比一致性", "",
                "本数据集未产出对比一致性表：源分析中没有满足条件的可比较对比对，因此不提供跨对比重叠或一致性结论。",
                "该缺口是分析设计的边界，不是结果缺失。", ""]
    try:
        with path.open(encoding="utf-8-sig") as fh:
            rows = list(_csv.DictReader(fh))
    except Exception:
        return []
    if not rows:
        return []
    out = ["### 对比一致性", "",
           "| 对比 A | 对比 B | 共享蛋白 | 效应相关 | 显著重叠 | Fisher p |",
           "|---|---|---:|---:|---:|---:|"]
    shown = rows if limit is None else rows[:limit]
    for row in shown:
        out.append("| %s | %s | %s | %s | %s | %s |" % (
            _cell(row.get("contrast_a")), _cell(row.get("contrast_b")), _cell(row.get("shared_proteins")),
            _fmt(row.get("spearman_logfc")), _fmt(row.get("overlap_sig")), _fmt(row.get("fisher_p"))))
    coverage = ("本表列出源表中的全部 %d 对比较。" % len(rows)) if len(shown) == len(rows) else (
        "本表按源表顺序列出前 %d 对（共 %d 对）。" % (len(shown), len(rows)))
    out += ["", coverage, "相关性描述两个对比的效应分布关系，不构成显著性检验。", ""]
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
        body = ("本数据集未产出预定义基因集的重叠富集文件，因此本报告不提供通路层面结论；"
                "这是该项分析未执行，不是分析执行后没有结果。")
    elif state == "unparsable":
        body = ("本数据集的重叠富集文件存在但无法解析，因此本报告不使用其内容，也不据此写通路结论；"
                "这是读取状态，不是结果缺失。")
    elif state == "empty":
        body = ("本数据集在当前富集配置下未产出可用条目（预定义基因集的重叠结果为空），"
                "故不提供通路层面结论；这是分析能力边界，不是结果缺失，也不引入替代性通路推测。")
    else:
        body = ("本数据集的重叠富集文件含有条目，但没有条目能对应到本报告的核心对比，"
                "因此本节不给出通路层面结论；这是对比覆盖边界，不是结果缺失。")
    return ["### 富集边界说明", "",
            body, ""]


CONFIDENCE_LABELS = {"high": "高", "moderate": "中", "low": "低"}
EVIDENCE_SOURCE_LABELS = {"current_matrix": "本次分析的数据矩阵", "analyzed_matrix": "本次分析的数据矩阵",
                          "dataset_provided": "数据集提供", "user_provided": "用户提供",
                          "auto_inferred": "系统自动推断", "inferred": "系统自动推断",
                          "offline_enrichment": "离线富集结果", "external_annotation": "外部注释",
                          "analysis_design": "分析设计"}


def _evidence_source_label(value: Any) -> str:
    text = str(value or "").strip()
    return EVIDENCE_SOURCE_LABELS.get(text, text or "未记录")


def _confidence_label(value: Any) -> str:
    text = str(value or "").strip()
    return CONFIDENCE_LABELS.get(text.lower(), text or "未记录")


def _direction_phrase(direction: Any, groups) -> str:
    """Translate the internal direction token into the reader-facing group names."""
    text = str(direction or "").strip()
    first = _cell(groups[0]) if groups and groups[0] else "第一臂"
    second = _cell(groups[1]) if groups and len(groups) > 1 and groups[1] else "第二臂"
    if text.startswith("up_in_group_a"):
        return "%s 中较高" % first
    if text.startswith("down_in_group_a"):
        return "%s 中较高" % second
    return text or "方向未记录"


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


FIGURE_TITLES = {
    "qc_sample_overview": "样本与缺失率 QC",
    "pca": "PCA 样本结构",
    "umap": "UMAP 样本结构",
    "heatmap": "差异蛋白热图",
    "differential_summary_barplot": "差异蛋白数量概览",
    "mechanism_enrichment_dotplot": "机制富集 dotplot",
    "mechanism_top_protein_group_means": "核心蛋白组均值图",
    "protein_contrast_bubble": "候选蛋白对比气泡图",
    "key_protein_overview": "关键蛋白总览",
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
            key = groups.get(name) or groups.get(name.replace(".raw", "")) or "未分组"
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
    thresholds = "adj.P<0.05 且 |logFC|>0.25（BH 校正）"
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
        title = FIGURE_TITLES.get(plot_type, plot_type)
        if plot_type == "qc_sample_overview":
            conclusion = ("平均每样本检出 %s 个蛋白，未注释 %s 个；分组规模与缺失率见「数据与预处理」。"
                          % (_fmt(meta.get("n_proteins")), _fmt(meta.get("n_unannotated"))))
        elif plot_type == "pca":
            pca = _pca_numbers(Path(run))
            if pca:
                ranked = sorted(pca["group_pc1"].items(), key=lambda item: item[1])
                conclusion = ("PC1 解释 %.1f%%、PC2 解释 %.1f%% 的方差；PC1 分组均值从最低的 %s（%.2f）到最高的 %s（%.2f），说明分组在 PC1 方向存在整体位移。"
                              % (pca["pc1"], pca["pc2"], ranked[0][0], ranked[0][1],
                                 ranked[-1][0], ranked[-1][1]))
            else:
                conclusion = "线性降维用于检查分组分离与离群样本（解释率未记录）。"
        elif plot_type == "umap":
            conclusion = "与 PCA 同一输入的二维投影，用于交叉检查分组分离与离群样本。"
        elif plot_type == "heatmap":
            conclusion = "取信息量最高的前 %s 个特征（阈值 %s）展示样本间分布。" % (top_n, thresholds)
        elif plot_type == "differential_summary_barplot":
            # R17: the bar counts follow each source table's own sign, so the figure says which arm
            # "up" means; without this the figure and the body read as a contradiction.
            inverted = [rec for rec in index["contrasts"] if rec.get("inverted")]
            direction_note = ""
            if len(inverted) == 1:
                arms = _arm_labels_for(inverted[0].get("source_contrast"), index)
                if len(arms) == 2:
                    direction_note = ("该图按源差异表 %s 的符号绘制：上调＝%s 较高，下调＝%s 较高；"
                                      "正文按「%s」叙述时符号相反。"
                                      % (inverted[0].get("source_contrast"), arms[0], arms[1],
                                         inverted[0].get("display_contrast")))
            elif inverted:
                direction_note = "该图按各对比源差异表的符号绘制，与正文显示方向的对应关系见各任务小节表注。"
            conclusion = ("%s 个对比共 %s 个蛋白通过筛选（上调 %s、下调 %s，阈值 %s）。%s"
                          % (_fmt(meta.get("n_contrasts")), _fmt(meta.get("n_sig")),
                             _fmt(meta.get("n_up")), _fmt(meta.get("n_down")), thresholds,
                             direction_note))
        elif plot_type == "mechanism_enrichment_dotplot":
            conclusion = "离线富集条目按方向提示展示，未经多重检验校正；条目数见「富集方向提示」小节。"
        elif plot_type == "mechanism_top_protein_group_means":
            conclusion = "代表蛋白组的分组均值；逐条数值与差值见「模块分组分数」表，模块分数无 FDR，只作方向性证据。"
        elif plot_type == "protein_contrast_bubble":
            conclusion = "候选蛋白在各对比中的方向与效应量；逐条数值见「候选蛋白逐对比统计」表。"
        else:
            conclusion = "各对比通过筛选的蛋白及其效应量；逐条数值见「对比级证据摘要」表。"
        lines.append("- 图%d %s：%s" % (number, title, conclusion))
    volcanos = by_type.get("volcano") or []
    if volcanos:
        if len(volcanos) == 1:
            lines.append("- 图%d 逐对比火山图（阈值 %s）：" % (number + 1, thresholds))
        else:
            lines.append("- 图%d–%d 逐对比火山图（阈值 %s）：" % (number + 1, number + len(volcanos), thresholds))
        for record in volcanos:
            meta = record.get("caption_metadata") or {}
            contrast = str(record.get("file_name") or "").replace("_volcano_plot.png", "")
            arms = _arm_labels_for(contrast, index)
            if len(arms) == 2:
                lines.append("  - %s：通过筛选 %s 个（%s 较高 %s、%s 较高 %s；按该差异表自身符号）"
                             % (contrast, _fmt(meta.get("n_sig")), arms[0], _fmt(meta.get("n_up")),
                                arms[1], _fmt(meta.get("n_down"))))
            else:
                lines.append("  - %s：通过筛选 %s 个（上调 %s、下调 %s）"
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
        out += ["### 对比级证据摘要", "",
                "| 任务 | 对比 | 第一臂 | 第二臂 | 通过筛选蛋白数 | 第一臂中较高 | 第二臂中较高 | 证据来源 | 置信度 |",
                "|---|---|---|---|---:|---:|---:|---|---|"]
        boundary = ""
        for row in contrast_rows:
            groups = groups_for(row.get("display_contrast"))
            first = _cell(row.get("group_a") or (groups[0] if groups else "第一臂"))
            second = _cell(row.get("group_b") or (groups[1] if len(groups) > 1 else "第二臂"))
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _cell(row.get("claim_title")), _cell(row.get("display_contrast")), first, second,
                _fmt(row.get("n_sig")), _fmt(row.get("n_up_display_group_a")),
                _fmt(row.get("n_down_display_group_a")),
                _cell(_evidence_source_label(row.get("evidence_source"))),
                _cell(_confidence_label(row.get("confidence")))))
            if not boundary and str(row.get("boundary") or "").strip():
                boundary = str(row.get("boundary")).strip()
        if boundary:
            out += ["", "边界：%s" % boundary]
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
        out += ["### 候选蛋白逐对比统计", "",
                "| 候选 | 匹配蛋白 | 对比 | 方向 | logFC | P 值 | 校正 P 值 | 缺失率 | 证据来源 |",
                "|---|---|---|---|---:|---:|---:|---:|---|"]
        source_contrasts = set()
        for row in unique:
            groups = groups_for(row.get("contrast"))
            contrast_cell = _cell(row.get("contrast"))
            direction_cell = _cell(_direction_phrase(row.get("direction"), groups))
            # R17: internal presence enums become reader wording; the state itself is unchanged.
            if str(row.get("contrast") or "").strip() == "matrix_presence_only":
                contrast_cell = "未进入统计检验"
            if str(row.get("direction") or "").strip() == "not_detected_in_available_matrix":
                direction_cell = "在可用矩阵中未检出"
            if str(row.get("direction") or "").strip() == "symbol_not_matched_in_available_matrix":
                direction_cell = "未建立符号映射（不等同于未检出）"
            if "_vs_" in str(row.get("contrast") or ""):
                source_contrasts.add(str(row.get("contrast")))
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _cell(row.get("candidate")), _cell(row.get("matched_protein")),
                contrast_cell, direction_cell,
                _fmt(row.get("logFC")), _fmt(row.get("P.Value")), _fmt(row.get("adj.P.Val")),
                _fmt(row.get("missing_rate")),
                _cell(_evidence_source_label(row.get("evidence_source")))))
        note = "每一行是一个候选蛋白在一个对比中的统计结果；未列出的候选不在候选清单内。"
        if len(unique) != len(candidate_rows):
            note += ("本表按候选、对比、方向与统计量完全相同的行折叠后共 %d 行（源表 %d 行，重复 %d 行）；"
                     "这里的计数单位是行，不是蛋白个数。"
                     % (len(unique), len(candidate_rows), len(candidate_rows) - len(unique)))
        note += "筛选口径只使用校正后 P 值（adj.P.Val）；原始 P 值列出仅供核对，不参与筛选。"
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
                "本表逐行按该行「对比」列的方向给符号：以 %s 行为例，正值表示 %s 较高；"
                "同一对比在正文中按反向写法「%s」叙述时符号相反，两者是同一个对比。"
                % (source, arms[0] if arms else "第一臂", source_map[source]),
                "本表符号方向按各行「对比」列（与正文反向写法「%s」同义）。" % source_map[source])
            out.append("")
        if others:
            out += _note_once(
                "other_contrasts_%s" % "_".join(others[:2]),
                "本表还包含报告任务小节之外的对比（%s）：这些行按各自差异表自身的符号给出，"
                "正值表示该行「对比」列中前一组较高。" % "、".join(others),
                "本表另含任务小节之外的对比（%s），符号按各行「对比」列。" % "、".join(others))
            out.append("")

    module_rows = _read_evidence_rows(folder / "curated_module_group_summary.csv")
    if module_rows:
        out += ["### 模块分组分数", "",
                "| 模块 | 分组或差值 | 样本数 | 平均分 | 中位数 | 匹配基因数 | 成员基因（匹配） |",
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
            "模块分数是成员蛋白评分的分组汇总，没有 FDR，只能作为方向性证据，不能当作显著富集。",
            "模块分数无 FDR，只作方向性证据（口径见「结论边界与方法局限」）。")
        out.append("")
    return out
def _asset_lines(run: Path, index: Dict[str, Any]) -> List[str]:
    # Reader-facing index: labels plus where to check the numbers. Internal file names stay out of
    # the reader-facing document; the delivered evidence bundle keeps them for reproduction.
    lines = ["| 资产 | 在何处复核 |", "|---|---|"]
    def _pointer_for(name: str) -> str:
        if "candidate_exclusion_trace" in name:
            return "见本报告「候选排除轨迹」"
        if "contrast_concordance" in name:
            return "见本报告「对比一致性」"
        if "stratified" in name:
            return "见本报告「分层对比摘要」"
        if name.endswith(".csv") or name.endswith(".json"):
            return "见本报告「证据表（逐条）」"
        return "见随报告交付的图册与图注"

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
    other_note = ("并包含未进入统计检验的记录" if presence_only
                  else "不含未进入统计检验的候选（这类候选只在附录单独列出）")
    lines += ["", "候选蛋白索引共 %d 个「对比 × 蛋白组」组合：%d 个已在上文对应任务中列出%s"
              % (n_cand, n_cand - len(rest_candidates),
                 "；其余 %d 个见下表。" % len(rest_candidates) if rest_candidates else "，此处不再重复。"),
              "（该计数按对比与蛋白组去重；「证据表（逐条）」的候选蛋白表另按方向与统计量去重，%s，"
              "因此两者行数不同，不是同一口径。）" % other_note]
    if rest_candidates:
        lines += [""] + _candidate_block({"candidates": rest_candidates}, index, index["conditions"], limit=None)
    lines += ["", "模块记录共 %d 条：%d 条已在上文对应任务中列出%s"
              % (n_mod, n_mod - len(rest_modules),
                 "；其余 %d 条见下表。" % len(rest_modules) if rest_modules else "，此处不再重复。")]
    if rest_modules:
        lines += [""] + _module_block({"modules": rest_modules}, index, limit=None)
    # Internal evidence IDs and file names stay out of the reader-facing document; each line points
    # at the section that carries the same records in reader-readable form.
    lines += ["", "证据条目的复核入口："]
    lines.append("- 对比级证据：%d 条，逐条数值见「对比级证据摘要」表。" % len(index["contrasts"]))
    lines.append("- 候选蛋白记录：%d 条，逐条数值见「候选蛋白逐对比统计」表。" % len(index["candidates"]))
    lines.append("- 模块记录：%d 条，逐条数值见「模块分组分数」表。" % len(index["modules"]))
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
NOT_RECORDED = "未记录"

MATRIX_STAGE_FILES = (
    ("交付矩阵（未经步骤命名）", "processed_proteins/ProQuant_Normalized.csv"),
    ("批次标定后矩阵", "processed_proteins/ProteinQuant_ComBat.csv"),
    ("过滤后矩阵（统计检验所用）", "processed_proteins/ProteinQuant_Filtered.csv"),
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
                return _fmt(value), "运行记录 %s" % key
    return NOT_RECORDED, ""


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

    def add(item, value, source, status="已记录", note=""):
        items.append({"item": item, "value": _fmt(value), "source": source, "status": status,
                      "note": note})

    add("输入矩阵状态", matrix.get("matrix_state"),
        "parameters.json → analysis_design.matrix.matrix_state")
    if matrix.get("non_missing_fraction") is not None:
        add("输入矩阵非缺失比例", "%.4f" % float(matrix["non_missing_fraction"]),
            "parameters.json → analysis_design.matrix.non_missing_fraction",
            note="该比例在原始交付矩阵上统计，与过滤后矩阵不同一阶段")
    stages = []
    for label, rel in MATRIX_STAGE_FILES:
        shape = _csv_shape(run / rel)
        if shape:
            stages.append("%s：%d 蛋白 × %d 样本" % (label, shape["rows"], shape["cols"]))
    add("运行链路上的矩阵阶段", "；".join(stages) or NOT_RECORDED,
        "运行目录中的矩阵文件行数与列数（行=蛋白，列=样本）",
        status="已记录" if stages else NOT_RECORDED)
    if first.get("n_proteins") is not None:
        add("统计检验所用矩阵规模", "%s 个蛋白（每个对比）" % _fmt(first.get("n_proteins")),
            "limma_summary.json → n_proteins")

    scale_bits = []
    if matrix.get("looks_logged") is not None:
        scale_bits.append("数值范围形似对数尺度=%s" % matrix.get("looks_logged"))
    if matrix.get("needs_log_transform") is not None:
        scale_bits.append("需要对数变换=%s" % matrix.get("needs_log_transform"))
    if matrix.get("already_processed") is not None:
        scale_bits.append("输入已处理=%s" % matrix.get("already_processed"))
    add("尺度与变换（记录字段）", "；".join(scale_bits) or NOT_RECORDED,
        "parameters.json → analysis_design.matrix.looks_logged / needs_log_transform / already_processed",
        status="部分记录" if scale_bits else NOT_RECORDED,
        note="记录的是对输入的判断字段；实际执行了哪一步变换、检验空间属于哪种尺度都未记录，"
             "因此与「数据与预处理」的尺度推断不矛盾也不互相支持")

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
        derived_note = ("该记录由本次运行的执行记录派生（%s），不是该步骤直接写出的独立文件。"
                        % transform_source) if transform.get("derived") else ""
        if "applied" in imp:
            add("本轮实际执行的缺失填补",
                ("已执行：%s%s" % (imp.get("method") or "方法未记录",
                                  ("，填补 %s 个取值" % imp.get("n_values_imputed"))
                                  if imp.get("n_values_imputed") not in (None, "") else ""))
                if imp.get("applied") else "未执行",
                "%s → imputation" % transform_source, note=derived_note)
        if "applied" in logt:
            add("本轮实际执行的对数变换",
                ("已执行：%s" % (logt.get("form") or "形式未记录")) if logt.get("applied")
                else ("未记录（记录中的规则：%s）" % (logt.get("recorded_rule") or "未记录")
                      if logt.get("applied") is None else "未执行"),
                "%s → log2_transform" % transform_source,
                note=("；".join(part for part in (
                    logt.get("applied_basis") or "",
                    "parameters.json 的 needs_log_transform 是对输入的推断，两者不是同一件事",
                    derived_note) if part)))
        if filt:
            add("本轮实际执行的蛋白过滤",
                "%s；保留蛋白 %s 个" % (filt.get("rule") or NOT_RECORDED,
                                      _fmt(filt.get("n_proteins_retained"))),
                "%s → protein_filtering" % transform_source)
    else:
        add("本轮实际执行的矩阵变换", NOT_RECORDED,
            "processed_proteins/matrix_transform_record.json",
            status=NOT_RECORDED,
            note="缺少该记录只说明执行动作未记录，不能推断为未执行")
    if matrix.get("numeric_min") is not None and matrix.get("numeric_max") is not None:
        add("输入矩阵数值范围", "%s 至 %s" % (_fmt(matrix.get("numeric_min")),
                                          _fmt(matrix.get("numeric_max"))),
            "parameters.json → analysis_design.matrix.numeric_min / numeric_max")

    missing_value, missing_source = _missing_declaration(matrix, diff, design, first)
    add("缺失值处理（统计口径）", missing_value, missing_source or "运行记录 · 分析设计与差异分析记录",
        status="已记录" if missing_source else NOT_RECORDED,
        note="与下一行的插补状态是两个不同问题：这里问统计检验如何对待缺失观测，那里问分析矩阵是否被填补")

    add("设计来源", _evidence_source_label(design.get("source")),
        "parameters.json → analysis_design.source")
    add("分组与标识列", "；".join(
        part for part in ("分组列=%s" % _fmt(cond.get("group_col")),
                          "样本 ID 列=%s" % _fmt(cond.get("sample_id_col")),
                          "批次列=%s" % _fmt(cond.get("batch_col")),
                          "个体/实验单位列=%s" % _fmt(cond.get("pair_col"))) if part),
        "parameters.json → analysis_design.group_col / sample_id_col / batch_col / pair_col")
    contrasts = cond.get("contrasts") or []
    contrast_names = [str(c.get("name")) for c in contrasts if c.get("name")]
    if contrasts:
        shown = "；".join(contrast_names[:3]) + (("；其余 %d 个对比见运行配置中的对比清单" % (len(contrast_names) - 3)) if len(contrast_names) > 3 else "")
        add("分析设计中的对比（%d 个）" % len(contrasts), shown or NOT_RECORDED,
            "parameters.json → analysis_design.differential.contrasts")
    covariates = design.get("covariates")
    if covariates is not None:
        add("协变量", "；".join(str(c) for c in covariates) if covariates else "无",
            "parameters.json → analysis_design.covariates")

    add("统计模型（记录原文）", first.get("method"),
        "limma_summary.json → method", status="已记录" if first.get("method") else NOT_RECORDED)
    if first.get("statistical_backend"):
        # the raw backend name and the library error string are developer-facing; the reader only
        # needs to know that the standard pipeline was unavailable and which implementation ran
        backend_raw = str(first.get("statistical_backend"))
        backend = ("内置回退实现（运行环境缺少 R/rpy2，标准流程未启用）"
                   if "fallback" in backend_raw.lower() else backend_raw)
        add("统计后端", backend, "limma_summary.json → statistical_backend / fallback_reason")
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
            span = ("%d–%d" % (min(unit_shared), max(unit_shared))) if unit_shared else NOT_RECORDED
            unit_value = ("以%s为分析单位（来源：%s）：%d 个对比中 %d 个可做个体级检验（两臂共同单位 %s 个）；"
                          "按 FDR 口径合计 %d 条通过，按联合口径（FDR 且 |效应| > %s）合计 %d 条通过；"
                          "%d 个蛋白行因有效单位不足标为未检验。"
                          % ("／".join(unit_names) or NOT_RECORDED, "、".join(unit_sources) or NOT_RECORDED,
                             len(unit_records), len(unit_done), span, unit_pass_fdr,
                             "／".join(unit_thresholds) or NOT_RECORDED, unit_pass_joint, unit_not_tested))
            unit_status = "已记录"
        else:
            unit_value = "已检查实验单位结构，%d 个对比均不满足个体级检验条件。" % len(unit_records)
            unit_status = "部分记录"
        unit_blocked = [u for u in unit_records if str(u.get("status")) != "completed" and u.get("reasons")]
        unit_notes = []
        if unit_excluded:
            unit_notes.append("混合来源／未确认来源单位单列并排除在独立个体之外：%s（按标签语义事先判定）。"
                              % "、".join(_cell(x) for x in unit_excluded))
        if unit_blocked:
            unique_reasons = []
            for record in unit_blocked:
                text = _unit_reason_zh(record)
                if text not in unique_reasons:
                    unique_reasons.append(text)
            unit_notes.append("未做个体级检验的原因：%s。" % "；".join(unique_reasons[:2]))
        unit_notes.append("观察级检验仍是参照估计，个体级与观察级的估计目标不同。")
        add("个体结构敏感性分析", unit_value, "limma_summary.json → unit_sensitivity",
            status=unit_status, note="".join(unit_notes))
    if first.get("n_samples_group_a") is not None and first.get("n_samples_group_b") is not None:
        first_name = contrast_names[0] if contrast_names else NOT_RECORDED
        add("每组样本数（%s）" % first_name,
            "第一臂 %s vs 第二臂 %s" % (_fmt(first.get("n_samples_group_a")),
                                    _fmt(first.get("n_samples_group_b"))),
            "limma_summary.json → n_samples_group_a / n_samples_group_b")

    add("多重校正方法", diff.get("fdr_method"),
        "parameters.json → analysis_design.differential.fdr_method",
        status="已记录" if diff.get("fdr_method") else NOT_RECORDED)
    if thresholds_used:
        add("实际生效阈值",
            "校正 P≤%s 且 |效应量|≥%s" % (_fmt(thresholds_used.get("adj_p")),
                                      _fmt(thresholds_used.get("abs_logFC"))),
            "limma_summary.json → sanity_checks.thresholds_used")
    elif diff.get("p_thresh") is not None and diff.get("logfc_thresh") is not None:
        add("设计阈值",
            "校正 P≤%s 且 |效应量|≥%s" % (_fmt(diff.get("p_thresh")), _fmt(diff.get("logfc_thresh"))),
            "parameters.json → analysis_design.differential.p_thresh / logfc_thresh")
    if sanity.get("n_significant") is not None:
        first_name = contrast_names[0] if contrast_names else NOT_RECORDED
        add("运行记录中的通过筛选数（%s）" % first_name,
            "%s（较高 %s / 较低 %s）" % (_fmt(sanity.get("n_significant")), _fmt(sanity.get("n_up")),
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
        add("原始交付矩阵的平均缺失率（分组）",
            _range_text([min(input_missing), max(input_missing)]),
            "分组组成与质控表 → input_mean_missing_rate")
    if analysed_missing:
        add("本轮分析矩阵的平均缺失率（分组）",
            _range_text([min(analysed_missing), max(analysed_missing)]),
            "分组组成与质控表 → analysed_mean_missing_rate")
    _transform_imp = read_matrix_transform_record(run)
    _imp = (_transform_imp.get("imputation") or {}) if isinstance(_transform_imp, dict) else {}
    if imputation or _imp.get("applied") is not None:
        # R28: the QC flag is written before the matrix is imputed; when the execution record has an
        # answer it wins, and a conflict between the two is stated instead of silently resolved.
        qc_text = "；".join(imputation) or "未记录"
        if _imp.get("applied") is True:
            value = ("已执行（%s）；分组组成与质控表中的 imputation_applied=%s 与执行记录不同，"
                     "以执行记录为准并登记该冲突" % (_imp.get("method") or "方法未记录", qc_text))
        elif _imp.get("applied") is False:
            value = "未执行"
        else:
            value = qc_text
        add("是否执行插补", value,
            "processed_proteins/matrix_transform_record.json → imputation"
            if _imp.get("applied") is not None else "分组组成与质控表 → imputation_applied",
            note="分析矩阵缺失率为 0 时按此字段判读，不等于全部蛋白被检出")
    return {"items": items, "conditions": cond, "limma_first": first, "design_source": design.get("source"),
            "input_missing_rate": [min(input_missing), max(input_missing)] if input_missing else None,
            "analysed_missing_rate": [min(analysed_missing), max(analysed_missing)] if analysed_missing else None,
            "imputation": imputation, "qc_rows": len(qc_rows)}


def _run_method_summary_lines(run: Path, cond: Dict[str, Any], evidence_data: Dict[str, Any]) -> List[str]:
    """Three reader-facing lines: which matrix was tested, which model ran, which numbers are bound."""
    lines: List[str] = []
    stage_text = next((x["value"] for x in evidence_data["items"]
                       if x["item"] == "运行链路上的矩阵阶段"), "")
    tested = next((part.strip() for part in str(stage_text).split("；") if "过滤后" in part),
                  NOT_RECORDED)
    analysed = evidence_data.get("analysed_missing_rate")
    raw = evidence_data.get("input_missing_rate")
    if raw and analysed:
        # R28: the QC table's imputation flag is written before the calibration step runs, so it cannot
        # be used to say that no imputation happened. The execution record decides this sentence.
        _transform = read_matrix_transform_record(run)
        _imp = (_transform.get("imputation") or {}) if isinstance(_transform, dict) else {}
        if _imp.get("applied") is True:
            imputation_text = "；分析矩阵按执行记录做过缺失填补（%s）" % (_imp.get("method") or "方法未记录")
        elif _imp.get("applied") is False:
            imputation_text = "；执行记录显示未做缺失填补"
        else:
            imputation_text = "；缺失填补状态未记录，不能写成未执行"
        lines.append("- 检验所用矩阵与缺失口径：统计检验在%s上完成；原始交付矩阵的平均缺失率 %s，"
                     "本轮分析矩阵 %s%s。" % (tested, _range_text(raw), _range_text(analysed),
                                          imputation_text))
    else:
        lines.append("- 检验所用矩阵：%s；缺失口径%s（运行记录中没有分阶段缺失统计）。"
                     % (tested, NOT_RECORDED))
    lines.append("- 统计模型、多重校正、筛选条件与各参数的记录位置见「运行参数与证据绑定」；"
                 "本节只给摘要，完整声明集中在该小节，此处不重复。")
    return lines


def _range_text(bounds) -> str:
    low, high = bounds[0], bounds[1]
    return "%.4f" % low if low == high else "%.4f 至 %.4f" % (low, high)


# Record locators are rendered in reader language; the machine keys stay in the JSON sidecar so a
# reviewer can still find the exact field. Longest keys first, so a specific rule wins over a general one.
RECORD_LABELS = (
    ("parameters.json → analysis_design.matrix.", "运行配置 · 分析设计 · 矩阵 · "),
    ("parameters.json → analysis_design.differential.", "运行配置 · 分析设计 · 差异分析 · "),
    ("parameters.json → analysis_design.", "运行配置 · 分析设计 · "),
    ("parameters.json → ", "运行配置 · "),
    ("limma_summary.json → sanity_checks.", "差异分析记录 · 记录字段 · "),
    ("limma_summary.json → ", "差异分析记录 · "),
    ("analysis_design", "分析设计"),
    ("limma_summary", "差异分析记录"),
    ("parameters.json", "运行配置"),
)
FIELD_LABELS = (
    ("looks_logged / needs_log_transform / already_processed", "尺度判断字段"),
    ("statistical_backend / fallback_reason", "统计后端与回退原因"),
    ("n_samples_group_a / n_samples_group_b", "各组样本数"),
    ("n_significant / n_up / n_down", "通过筛选数"),
    ("group_col / sample_id_col / batch_col", "分组与标识列"),
    ("p_thresh / logfc_thresh", "设计阈值"),
    ("numeric_min / numeric_max", "数值范围"),
    ("non_missing_fraction", "非缺失比例"),
    ("thresholds_used", "生效阈值"),
    ("input_mean_missing_rate", "原始矩阵平均缺失率"),
    ("analysed_mean_missing_rate", "分析矩阵平均缺失率"),
    ("imputation_applied", "是否插补"),
    ("matrix_state", "矩阵状态"),
    ("fdr_method", "多重校正方法"),
    ("n_proteins", "检验蛋白数"),
    ("contrasts", "对比清单"),
    ("covariates", "协变量"),
    ("method", "统计方法"),
    ("source", "设计来源"),
)


def _reader_source_label(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return NOT_RECORDED
    for key, label in FIELD_LABELS:
        text = text.replace(key, label)
    for key, label in RECORD_LABELS:
        text = text.replace(key, label)
    return text.replace(" → ", " · ")


def _run_evidence_binding_lines(evidence_data: Dict[str, Any]) -> List[str]:
    lines = ["下表逐项列出本节数值所依赖的运行参数及其记录位置；" 
             "运行记录未写入的参数标记为「未记录」，不按报告措辞补写，也不选择更有利的读法。", "",
             "| 项目 | 取值 | 记录位置 | 状态 |", "|---|---|---|---|"]
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
    connector = "或" if "或" in declared else "且"
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
                    n_joint += 1 if (ok_p and ok_e if connector == "且" else (ok_p or ok_e)) else 0
        except Exception:  # noqa: BLE001
            out[key] = {"status": "unreadable", "connector": connector}
            continue
        out[key] = {"status": "ok", "connector": connector, "p_col": p_col, "effect_col": effect_col,
                    "n_joint": n_joint, "n_pass_p": n_p, "n_pass_effect": n_e, "min_adj_p": min_p,
                    "declared_n_sig": rec.get("n_sig"), "table": path.name}
    return out


HIGH_WORDS = ("较高", "更高", "偏高", "上调", "升高", "较高表达")
LOW_WORDS = ("较低", "更低", "偏低", "下调", "降低", "较低表达")
DIRECTION_WORDS = HIGH_WORDS + LOW_WORDS
NEGATION_WORDS = ("不", "未", "无", "非", "尚未", "没有")
# naming *which* matrix the statement is about is the binding the check asks for; a sentence that
# names neither a matrix nor a stage is the only case worth a hint.
STAGE_MARKERS = ("原始", "过滤", "插补", "矩阵", "质控", "本次", "本报告")
DETECTION_WORDS = ("未检出", "未检测", "完全不存在", "未被检出", "未鉴定")
# A sentence that states the absence of a number is itself a boundary statement, not an unbound claim.
DETECTION_EXCLUSIONS = ("未给出", "未提供", "未记录", "未纳入", "没有给出", "不给出", "未在当前",
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
                findings.append({"rule": "STAGE", "severity": "提示",
                                 "detail": "检出/缺失陈述未写明数据阶段，无法对应到运行记录中的具体矩阵",
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
                if phrase == "方向未判定" or " 较高" not in phrase:
                    continue
                high_arm = phrase.split(" 较高")[0]
                low_arm = next((g for g in groups if g != high_arm), None)
                if low_arm is None:
                    continue
                expected = high_arm if word in HIGH_WORDS else low_arm
                coverage["direction_checked"] += 1
                if arm != expected:
                    findings.append({"rule": "DIR", "severity": "提示",
                                     "detail": "方向句写出的较高/较低组与差异表记录的方向不同：表中记为 %s"
                                               % phrase,
                                     "quote": sentence[:160], "entity": symbol,
                                     "contrast": contrast.get("display_contrast")})
                break
    return findings, coverage


# R17: a sentence that denies evidence the run actually produced is a locator, never a rewrite.
UNAVAILABLE_TOPICS = (
    ("样本数", ("样本数", "观测数", "每组样本", "样本量")),
    ("缺失率", ("缺失率", "缺失比例", "缺失统计")),
    ("分层对比", ("分层", "亚组", "分层统计")),
    ("正式富集检验", ("正式富集", "富集检验", "超几何", "通路富集", "ORA")),
    ("通路富集条目", ("富集", "通路")),
    ("剂量趋势", ("剂量趋势", "趋势检验", "spearman")),
    ("对比一致性", ("对比一致性", "跨对比一致性")),
    ("降维与图件", ("降维", "pca", "umap", "图册", "火山图", "降维坐标")),
)


def available_evidence_topics(run: Path, index: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Which evidence classes this run actually produced, read from the artifacts (cheap checks)."""
    run = Path(run)
    evidence = (index or {}).get("run_evidence") or read_run_evidence(run)
    first = evidence.get("limma_first") or {}
    topics = {
        "样本数": "已计算" if first.get("n_samples_group_a") is not None
                  or first.get("n_samples_group_b") is not None else "未记录",
        "缺失率": "已计算" if evidence.get("input_missing_rate")
                  or evidence.get("analysed_missing_rate") else "未记录",
    }
    stratified = _ext_csv_rows(run, "stratified_contrast_summary.csv")
    topics["分层对比"] = "已计算" if any(str(row.get("level") or "").strip() for row in stratified) else "未计算"
    ora = _ext_csv_rows(run, "enrichment_ora.csv")
    topics["正式富集检验"] = "已计算" if any(str(row.get("p_value") or "").strip() for row in ora) else "未计算"
    overlap = run / "enrichment_results" / "go_kegg_reactome_results.json"
    topics["通路富集条目"] = "已计算" if overlap.exists() else "未计算"
    topics["剂量趋势"] = "已计算" if _ext_csv_rows(run, "dose_trend.csv") else "未计算"
    topics["对比一致性"] = "已计算" if _ext_csv_rows(run, "contrast_concordance.csv") else "未计算"
    try:
        figures = sorted(path.name for path in (run / "visualize_results").glob("*.png"))
    except Exception:  # noqa: BLE001
        figures = []
    topics["降维与图件"] = "已计算" if figures else "未计算"
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
            if topics.get(topic) != "已计算":
                continue
            if any(pattern in lowered for pattern in patterns):
                total += 1
                if len(findings) < limit:
                    findings.append({
                        "rule": "G7", "severity": "提示", "topic": topic,
                        "owner": item.get("owner"),
                        # reader-facing and short: the note locates the conflict without repeating the
                        # sentence and without naming any internal request artefact
                        "detail": "核对提示：正文称「%s」类证据未提供或未执行，但本次运行已产出该类证据"
                                  "（数值见对应表格与「运行参数与证据绑定」）；此处只标记位置，不修改原句。"
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
    return {"rule": "QC", "severity": "提示",
            "detail": "本轮分析矩阵的平均缺失率为 %.4f 且运行记录未执行插补（imputation_applied=%s），"
                      "与原始交付矩阵的平均缺失率 %.4f 至 %.4f 不同一阶段；引用缺失/检出数值时应写明阶段"
                      % (analysed[0], "；".join(evidence_data.get("imputation") or [NOT_RECORDED]),
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
    return ("核对提示：按本报告声明的联合筛选条件（%s 连接）在差异表上复算得到 %d 行通过，"
            "与本对比汇总的 %s 不一致（只满足校正 P 的 %d 行、只满足效应阈值的 %d 行；"
            "最小校正 P=%s）；请核对筛选口径与数据阶段。"
            % (info.get("connector"), info.get("n_joint"), _fmt(info.get("declared_n_sig")),
               info.get("n_pass_p"), info.get("n_pass_effect"), _fmt(info.get("min_adj_p"))))


def _consistency_note_lines(checks: Dict[str, Any], limit: int = 6) -> List[str]:
    """Reader-facing rendering of the findings: location and evidence only, never a rewrite."""
    findings = checks.get("findings") or []
    if not findings:
        return []
    lines = ["本节列出运行记录之间、以及正文与表格之间需要核对的口径差异，共 %d 条；"
             "提示只指出位置与依据，不修改任何数值或结论。" % len(findings), ""]
    for item in findings[:limit]:
        detail = str(item.get("detail") or "")
        quote = item.get("quote")
        if quote:
            detail = "%s（相关原文：%s）" % (detail, quote)
        lines.append("- %s" % detail)
    if len(findings) > limit:
        lines.append("- 另有 %d 条同类提示未在正文展开。" % (len(findings) - limit))
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
            findings.append({"rule": "G1", "severity": "提示",
                             "detail": "%s 的差异表不可解析（%s），筛选与计数无法核对"
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
            findings.append({"rule": "G1", "severity": "提示", "contrast": key,
                             "detail": "按报告自己声明的联合条件（%s 连接，列 %s / %s）在差异表上复算得到 "
                                       "%d 行，与本对比汇总的 %d 不一致；"
                                       "其中仅满足校正 P 的有 %d 行、仅满足效应阈值的有 %d 行、最小校正 P=%s"
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
            "n_findings": len(findings), "mode": "提示（不阻断，不改变结论）",
            # R21: these findings mark places worth looking at. They are not a quality score and are
            # not part of any acceptance total, and they never rewrite a scientific conclusion -
            # G7 was measured to raise false positives on sentences that state a real limitation.
            "diagnostic_only": True,
            "not_a_metric": "提示条数不得用作性能计数、验收总分或改进指标；只用于定位。"}


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
            return ("说明：本小节讨论的对比与「任务%d：%s」相同，标题沿用模型给出的对比标识；"
                    "两节数值来自同一张差异表，表格方向见各表标题后的方向标注，正文按模型原句保留。"
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
            text = cap_term_density(text, "\u5f53\u524d\u77e9\u9635", DENSITY_CAP, density_state)
            body.append(text)
            body.append("")
            model_paragraphs.append({"owner": owner, "sha12": sha12(text), "chars": len(text),
                                     "evidence_ids": evidence_ids or []})
            model_texts.append({"owner": owner, "text": text, "evidence_ids": evidence_ids or []})

    body.append("## 0. 关键结论速览")
    body.append("")
    add_model_lines(sections.get("key_summary") or [], "key_summary")
    if not sections.get("key_summary"):
        body += ["（模型未提供速览，以下为确定性事实）",
                 "- 本轮共 %d 个核心对比，证据文件见文末清单。" % len(plan["tasks"]), ""]

    body += _stratified_lines(run)
    body.append("## 1. 数据与预处理")
    body.append("")
    body += _data_section(run, cond, run_evidence)
    body.append("")

    heading = 2
    matched: set = set()
    exclusion_state: Dict[str, bool] = {}
    merged_alias_sections: List[Dict[str, Any]] = []
    for task in plan["tasks"]:
        rec = task["contrast"]
        body.append("## %d. 任务%d：%s" % (heading, task["order"], rec.get("claim_title") or rec.get("display_contrast")))
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
                note = ("注：模型原文把同一对比写成 %d 个平级小节（%s）；本节按对比合并，"
                        "内容按原顺序全部保留（未删减）。本报告统一按「%s」坐标读数"
                        % (len(labels), "、".join(_cell(label) for label in labels), _cell(display)))
                if len(arms) >= 2 and arms[0]:
                    note += "（正值表示 %s 较高）" % _cell(arms[0])
                note += "。"
                if flipped:
                    note += ("其中 %s 的臂序与本节坐标相反，其句内正负号与本节表格相反；"
                             "引用这些句子的数值时以本节表格的方向为准。"
                             % "、".join(_cell(label) for label in flipped))
                else:
                    note += "各小节的对比标识臂序与本节坐标一致。"
                body.append(note)
                body.append("")
        else:
            body += ["（本轮标准对比未匹配到模型小节：该对比按确定性事实呈现。）", ""]
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
        title = str(task.get("match") or "").strip() or "未命名小节"
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
    body += _enrichment_boundary_lines(run, any("富集方向提示" in line for line in body))
    figure_lines = _figure_reference_lines(run, index)
    if figure_lines:
        body.append("## %d. 图表与文字结论" % heading)
        body.append("")
        body.append("图件以「图号 + 标题 + 文字结论」在正文中给出，图不可见时信息仍可读。")
        body.append("")
        body += figure_lines
        body.append("")
        heading += 1
    evidence_tables = _evidence_table_lines(run, index)
    if evidence_tables:
        body.append("## %d. 证据表（逐条）" % heading)
        body.append("")
        body += evidence_tables
        heading += 1
    checks = run_consistency_checks(run, index, cond, model_texts, joint=joint_filter)
    body.append("## %d. 运行参数与证据绑定" % heading)
    body.append("")
    body += _run_evidence_binding_lines(run_evidence)
    body.append("")
    note_lines = _consistency_note_lines(checks)
    if note_lines:
        body.append("### 口径与一致性说明")
        body.append("")
        body += note_lines
        body.append("")
    heading += 1
    body.append("## %d. 总结与启发式推测" % heading)
    body.append("")
    add_model_lines(sections.get("summary") or [], "summary")
    heading += 1
    body.append("## %d. 可复核资产清单" % heading)
    body.append("")
    body += _asset_lines(run, index)
    heading += 1
    body.append("## %d. 结论边界与方法局限" % heading)
    body.append("")
    add_model_lines(sections.get("boundary") or [], "boundary_model")
    if sections.get("extra"):
        # content the assembler could not bind to a section must still be visible to a reader
        body.append("### 其他说明（未能绑定到任务小节的内容）")
        body.append("")
        add_model_lines(sections.get("extra") or [], "extra_unbound")
    body += ["- 以上结论只对当前分析矩阵成立；模块分数不替代单蛋白 FDR 或正式富集检验。",
             "- 离线富集结果只在 FDR 显著时作为显著富集，其余只能作为探索性提示。",
             "- 外部注释只作背景说明，不作为当前矩阵的直接证据。",
             "- 未提供富集结果与未执行富集分析是两种不同状态，报告按实际情况分别写明。",
             "- 表内数值来自本轮分析的确定性表格；正文解释与表格同源，任何数值以表格为准。", ""]

    report = "\n".join(body).rstrip() + "\n"
    ledger = _content_ledger_v3(blocks, report, model_paragraphs, index)
    try:
        binding = {"run": str(run), "report_sha12": sha12(report), "run_evidence": run_evidence,
                   "report_sha256": hashlib.sha256(normalise_report_text(report).encode("utf-8")).hexdigest(),
                   "hash_normalisation": REPORT_HASH_NORMALISATION,
                   "hash_stage": "assembly (re-bind after any later revision)",
                   "joint_filter": joint_filter, "consistency_checks": checks,
                   "note": "运行记录与主张绑定的机器可读副本；正文只呈现其中面向读者的部分"}
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
                                                    if x["status"] != "已记录"),
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
                reason = "模型表格由确定性表替代，但以下数值未在成品中出现，需人工确认：%s" % ", ".join(unmatched)
            else:
                disposition, reason = "replaced_by_deterministic_table", "模型表格数值均可在成品中找到"
            ledger.append({"index": block["index"], "heading": block["heading"], "kind": block["kind"],
                           "sha12": block["sha12"], "chars": block["chars"],
                           "disposition": disposition, "reason": reason,
                           "unmatched_tokens": unmatched})
            continue
        elif block["sha12"] in kept:
            disposition, reason = "preserved", "逐字保留"
        else:
            disposition, reason = "dropped", "未进入成品（需人工确认）"
        ledger.append({"index": block["index"], "heading": block["heading"], "kind": block["kind"],
                       "sha12": block["sha12"], "chars": block["chars"],
                       "disposition": disposition, "reason": reason})
    return ledger


# --------------------------------------------------- request-side evidence index (R17, production)
REQUEST_INDEX_TITLE = "## 本轮可用事实清单（由运行工件直接生成，与最终报告表格同源）"
REQUEST_INDEX_INTRO = (
    "本清单只列本次运行实际产出的证据。状态含义：已计算=可以直接引用并给出数值；"
    "未计算=本轮没有产出，不得据此写结论；无法使用=产出为空或不可读，必须按边界说明处理。"
    "报告引用数值时必须与本清单一致：标注「已计算」的内容不得写成「未提供 / 未包含 / 未产出」，"
    "标注「未计算」的内容不得写成已完成的分析结果。")


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
        match = re.match(r"(图\d+)\s*([^：:]*)", text)
        if match:
            conclusion = text[match.end():].lstrip("：: ").strip()
            titles.append(("%s %s：%s" % (match.group(1), match.group(2).strip(), conclusion))[:220])
    if not titles:  # figures exist on disk but the manifest could not be read
        try:
            pngs = sorted(p.name for p in (Path(run) / "visualize_results").glob("*.png"))
        except Exception:  # noqa: BLE001
            pngs = []
        titles = ["（图册文件）%s" % name for name in pngs[:limit]]
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
        basis = ("分析矩阵最大值 %s，过滤后原始矩阵最大值 %s（log2(x+1) 后的上界约 %s）"
                 % (_fmt(analysed["max"]), _fmt(raw["max"]), _fmt(expected_ceiling)))
    built = {
        "record_type": "matrix_transform_record",
        "record_source": "run_events.jsonl 的 combat_calibration 工具结果（派生，不是该步骤直接写出）",
        "derived": True,
        "stage": "protein_quant_combat_calibration",
        "input": {"protein_quant_file": Path(str(inputs.get("protein_quant_file") or "")).name,
                  "protein_rows": inputs.get("initial_protein_rows"),
                  "samples": inputs.get("initial_samples")},
        "protein_filtering": {"rule": "非缺失比例 >= %s（%s）" % (_fmt(filt.get("reproducibility_cutoff")),
                                                                 filt.get("filtering_strategy") or "未记录"),
                              "reproducibility_cutoff": filt.get("reproducibility_cutoff"),
                              "strategy": filt.get("filtering_strategy"),
                              "n_proteins_retained": filt.get("number_of_proteins_after_filtering")},
        "imputation": {"applied": bool("half-min" in method.lower()),
                       "method": method or "未记录"},
        "log2_transform": {"applied": log2_applied,
                           "form": "log2(x+1)",
                           "recorded_rule": method or "未记录",
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
            estimand = "个体内均值差（同一实验单位在两臂都有观测）"
        elif route == "between_unit_independent":
            estimand = "个体间均值差（每条臂只使用只在该臂观测的独立个体）"
        elif status != "completed":
            estimand = "无可识别的个体内或个体间估计目标"
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
            "status_zh": "已完成个体级敏感性分析" if status == "completed" else "不可估计，未做个体级检验",
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
              "imputation": "；".join(evidence.get("imputation") or []) or NOT_RECORDED}
    stage_text = next((x["value"] for x in evidence["items"] if x["item"] == "运行链路上的矩阵阶段"), "")
    params["matrix_stage"] = _table_cell(stage_text, 200)
    if thresholds:
        params["threshold"] = "校正 P≤%s 且 |效应量|≥%s" % (_fmt(thresholds.get("adj_p")),
                                                          _fmt(thresholds.get("abs_logFC")))
    if first.get("n_samples_group_a") is not None and first.get("n_samples_group_b") is not None:
        params["samples"] = "%s / %s（两臂样本数；按运行记录中第一个对比）" % (
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
                dimred = ("逐样本 %s 坐标已在 processed_proteins/qc_metrics.csv 给出（%d 行）；"
                          "UMAP 二维坐标为未产出坐标，不要写成缺少全部降维坐标。"
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
                params["imputation"] = "已执行（%s%s）" % (
                    _imp.get("method") or "方法未记录",
                    ("，填补 %s 个取值" % _imp.get("n_values_imputed"))
                    if _imp.get("n_values_imputed") not in (None, "") else "")
            else:
                params["imputation"] = "未执行"
        if _log.get("applied") is not None:
            params["log2_transform"] = ("已执行（%s）" % (_log.get("form") or "形式未记录")
                                        if _log.get("applied") else "未执行")
        if _filt:
            params["protein_filtering"] = "%s；保留蛋白 %s 个" % (
                _filt.get("rule") or NOT_RECORDED, _fmt(_filt.get("n_proteins_retained")))
        params["transform_record_source"] = _table_cell(str(transform.get("record_source") or ""), 140)
    else:
        params["imputation"] = NOT_RECORDED
        params["log2_transform"] = NOT_RECORDED
        params["transform_record_source"] = ("运行记录没有矩阵变换的完整记录：不能把「未记录」写成「未执行」，"
                                             "也不能写成已经执行")
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
        data["not_computed"].append("分层对比摘要（源表没有带分层的行）")

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
                for direction, label in (("upregulated", "上调方向"), ("downregulated", "下调方向")):
                    if len(arms) == 2:
                        label = "%s 较高" % (arms[0] if direction == "upregulated" else arms[1])
                    terms = [(str(term), str(genes).split("/"))
                             for genes, term in (node.get(direction) or {}).items()]
                    terms.sort(key=lambda item: (-len(item[1]), item[0]))
                    for term, members in terms[:1]:
                        picks.append({"database": database, "direction": label,
                                      "term": _table_cell(term, 80), "seed": len(members),
                                      "members": _table_cell(", ".join(members[:6]), 120)})
            total = sum(sum(counts.values()) for counts in per_db.values())
            overlap[contrast] = {"status": "已计算" if total else "有效空结果（该对比无条目）",
                                 "terms": total, "file_contrast": view.get("matched_key") or "",
                                 "reversed": bool(view.get("reversed")),
                                 "per_database": per_db, "top_terms": picks}
    data["overlap"] = overlap
    if overlap_state == "unparsable":
        data["not_computed"].append("预定义基因集的重叠富集（文件存在但解析失败——这是读取状态，"
                                    "不是未产出）")
    elif overlap_state == "empty":
        data["not_computed"].append("预定义基因集的重叠富集（文件存在但是空对象）")
    elif overlap_state == "missing":
        data["not_computed"].append("预定义基因集的重叠富集（本次运行没有该文件：分析未执行）")

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
            classes.append({"kind": "显著集",
                            "rule": _table_cell(significant_rows[0].get("query_definition"), 160),
                            "rows": len(significant_rows),
                            "query_sizes": _size_range([row.get("query_size")
                                                        for row in significant_rows]),
                            "contrasts": sorted({str(row.get("contrast") or "")
                                                 for row in significant_rows})[:8]})
        if fallback_rows:
            classes.append({"kind": "兜底集（该对比显著集过小时按 |logFC| 取前 10%）",
                            "rows": len(fallback_rows),
                            "significant_set_sizes": _size_range(_fallback_significant_sizes(fallback_rows)),
                            "query_sizes": _size_range([row.get("query_size") for row in fallback_rows]),
                            "contrasts": sorted({str(row.get("contrast") or "")
                                                 for row in fallback_rows})[:8]})
        data["ora"]["query_classes"] = classes
    else:
        data["not_computed"].append("正式富集检验（超几何 ORA）")

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
        data["not_computed"].append("剂量趋势检验（Spearman）")

    concordance_rows = _ext_csv_rows(run, "contrast_concordance.csv")
    data["concordance"] = {"pairs": len(concordance_rows),
                           "status": "已计算" if concordance_rows else "无法使用"}
    if not concordance_rows:
        data["not_computed"].append("对比一致性（源分析未产出可比较的对比对）")

    trace_rows = _ext_csv_rows(run, "candidate_exclusion_trace.csv")
    if trace_rows:
        tokens = _design_tokens(run)
        excluded = sorted({str(row.get("candidate")).strip() for row in trace_rows
                           if str(row.get("status") or "").strip().lower() == "absent"
                           and str(row.get("candidate") or "").strip()
                           and str(row.get("candidate")).strip().lower() not in tokens})
    else:
        excluded = []
        data["not_computed"].append("候选排除轨迹")
    data["exclusion"] = {"excluded": excluded, "status": "已计算" if trace_rows else "未计算"}

    titles = _figure_titles(run, index)
    data["figures"] = {"count": len(titles), "titles": titles[:24]}
    if not titles:
        data["not_computed"].append("图册与图注（visualize_results 未产出可引用图件）")
    return data


def format_request_evidence_index(data: Dict[str, Any]) -> str:
    """Render the shared evidence index as the request-side markdown block."""
    params = data.get("parameters") or {}

    def status(value: Any, ok: str = "已计算") -> str:
        return ok if str(value or "").strip() else "未计算"

    lines = [REQUEST_INDEX_TITLE, "", REQUEST_INDEX_INTRO, "", "### 数据与预处理", "",
             "| 项目 | 数值或状态 | 状态 |", "|---|---|---|",
             "| 运行链路上的矩阵阶段 | %s | %s |" % (_table_cell(params.get("matrix_stage"), 200),
                                                  status(params.get("matrix_stage"))),
             "| 统计检验所用矩阵规模 | %s 个蛋白 | %s |" % (params.get("tested_proteins"),
                                                        status(params.get("tested_proteins"))),
             "| 每组样本数 | %s | %s |" % (_table_cell(params.get("samples"), 120),
                                         status(params.get("samples"))),
             "| 原始交付矩阵的平均缺失率 | %s | %s |" % (params.get("missing_raw"),
                                                    status(params.get("missing_raw"))),
             "| 本轮分析矩阵的平均缺失率 | %s | %s |" % (params.get("missing_analysed"),
                                                    status(params.get("missing_analysed"))),
             "| 缺失值填补（按本轮实际执行记录） | %s | %s |" % (
                 params.get("imputation") or "未记录", status(params.get("imputation"), "已记录")),
             "| 对数变换（按本轮实际执行记录） | %s | %s |" % (
                 params.get("log2_transform") or "未记录", status(params.get("log2_transform"), "已记录")),
             "| 蛋白过滤（按本轮实际执行记录） | %s | %s |" % (
                 params.get("protein_filtering") or "未记录",
                 status(params.get("protein_filtering"), "已记录")),
             "| 变换记录的来源 | %s | 元数据 |" % _table_cell(
                 params.get("transform_record_source") or "未记录", 140),
             "| 统计模型 | %s | %s |" % (params.get("model"), status(params.get("model"), "已记录")),
             "| 多重校正 | %s | %s |" % (params.get("fdr"), status(params.get("fdr"), "已记录")),
             "| 实际生效阈值 | %s | %s |" % (params.get("threshold"),
                                          status(params.get("threshold"), "已记录")),
             "| 逐样本降维坐标 | %s | %s |" % (params.get("dimred") or "未记录",
                                            status(params.get("dimred"), "已计算")),
             ""]
    if data.get("contrasts"):
        lines += ["### 核心对比（task_id 与本节一致）", "",
                  "| task_id | 对比 | 两臂 | 通过筛选 | 第一臂较高 | 第二臂较高 | 方向写法 |",
                  "|---|---|---|---:|---:|---:|---|"]
        for rec in data["contrasts"]:
            direction = "按 %s" % rec.get("display_contrast")
            if rec.get("inverted"):
                direction = "按 %s（源差异表 %s 符号相反）" % (rec.get("display_contrast"),
                                                           rec.get("source_contrast"))
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                rec.get("task_id"), rec.get("display_contrast"), rec.get("arms"),
                rec.get("n_sig"), rec.get("n_up"), rec.get("n_down"), direction))
        lines.append("")
    unit_layer = data.get("unit_layer") or {}
    unit_items = list(unit_layer.get("contrasts") or [])
    if unit_items:
        lines += ["### 实验单位层（个体结构敏感性分析，已计算）", "",
                  "本节数值来自正式分析路径的单位层敏感性分析，覆盖本轮 %s 个对比（可做个体级检验 %s 个、"
                  "不可估计 %s 个）。它以个体均值计权，与以单个观测计权的观察级检验估计目标不同，"
                  "两套数值不可直接互换，也不可互相替代。"
                  % (unit_layer.get("n_contrasts"), unit_layer.get("n_completed"),
                     unit_layer.get("n_blocked")), "",
                  "| task_id | 本轮任务 | 状态 | 分析单位 | 单位语义来源 | 估计目标 | A/ B/ 共同/ 仅A/ 仅B | "
                  "检验蛋白/未检验 | FDR口径通过 | 联合口径通过 | 最小校正P |",
                  "|---|---|---|---|---|---|---|---|---:|---:|---:|"]
        for item in unit_items:
            lines.append("| %s | %s | %s | %s | %s | %s | %s/%s/%s/%s/%s | %s/%s | %s | %s | %s |" % (
                item.get("task_id"), "是" if item.get("in_task_list") else "否（背景对比）",
                item.get("status_zh"), item.get("analysis_unit"),
                item.get("unit_source"), item.get("estimand"),
                _fmt(item.get("n_units_a")), _fmt(item.get("n_units_b")),
                _fmt(item.get("n_units_shared")), _fmt(item.get("n_units_a_only")),
                _fmt(item.get("n_units_b_only")),
                _fmt(item.get("n_proteins_tested")), _fmt(item.get("n_proteins_not_tested")),
                _fmt(item.get("n_passing_fdr_only")), _fmt(item.get("n_passing_joint")),
                _fmt(item.get("min_adj_p_value"))))
        if any(not item.get("in_task_list") for item in unit_items):
            lines += ["", "标注为「否（背景对比）」的行是本轮分析已算出的其它目标对比；它们只作背景，"
                          "不得为它们另写任务小节，也不得把它们的通过数并入本节任务。"]
        lines += ["", "- 效应量列：%s（尺度：%s）；每个已检验蛋白的效应量、标准误与 95%% 置信区间"
                      "在该对比的单位层表中逐行给出，未通过最小样本条件的行标为未检验且不带 P 值。"
                      % (unit_layer.get("effect_column") or "未记录",
                         unit_layer.get("effect_scale") or "未记录")]
        for item in unit_items:
            summary = item.get("effect_summary") or {}
            if not summary:
                continue
            lines.append("  - %s：已检验 %s 个蛋白，效应量范围 %s 至 %s（中位 %s）；其中 %s 个蛋白的 "
                         "95%% 置信区间不跨 0（逐蛋白、未经多重校正）。"
                         % (item.get("task_id"), _fmt(summary.get("n")),
                            _fmt(summary.get("min")), _fmt(summary.get("max")),
                            _fmt(summary.get("median")), _fmt(summary.get("ci_excludes_zero"))))
        units_excluded = sorted({str(label) for item in unit_items
                                 for label in (item.get("excluded_units") or [])})
        if units_excluded:
            lines.append("- 已单列并排除在独立个体之外的实验单位：%s（按标签语义事先判定，不按检验结果挑选）。"
                         % "、".join(_cell(x) for x in units_excluded))
        for item in unit_items:
            if item.get("blocked_reason"):
                lines.append("- 未做个体级检验的对比：%s —— %s" % (item.get("task_id"),
                                                                    item.get("blocked_reason")))
        lines += ["- 引用规则：观察级与单位层各自的通过数只能写进各自小节；单位层不显著不等于「没有差异」，"
                  "不可估计的对比只能写「无法在本设计下识别该估计目标」。", ""]
    else:
        lines += ["### 实验单位层（个体结构敏感性分析）", "",
                  "本轮未记录单位层结果（状态：未记录）；不得据此写成「已确认个体间无差异」。", ""]
    strat = data.get("stratified") or {}
    if strat.get("rows"):
        lines += ["### 分层对比摘要（已计算：%d 行；分层变量：%s）"
                  % (strat["rows"], "、".join(strat.get("stratifiers") or ["未记录"])),
                  "", "| 对比 | 层 | 第一臂 n | 第二臂 n | 检验蛋白 | 通过筛选 | 分层判定 | 备注 |",
                  "|---|---|---:|---:|---:|---:|---|---|"]
        for row in strat.get("examples") or []:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row.get("contrast"), row.get("level"), row.get("n_a"), row.get("n_b"),
                row.get("tested"), row.get("n_sig"), row.get("verdict") or "未记录",
                row.get("note") or ""))
        shown = len(strat.get("examples") or [])
        if strat.get("truncated") or strat["rows"] > shown:
            # R21: the listing cap is display metadata. A model read the old wording as a finding
            # and wrote "分层变量包括 Cluster、Type2 及未列出的其余一行" into a reader-facing sentence.
            lines += ["", "覆盖范围（这是本清单的显示元数据，不是分析结论）：本清单为该源表的节选，"
                          "列出前 %d 行 / 共 %d 行，其余 %d 行未在本清单列出；节选只影响本清单的展示，"
                          "不代表分析范围，也不要把「未列出的其余行」写成科学结论。"
                          % (shown, strat["rows"], max(0, strat["rows"] - shown))]
        lines += ["", "分层结果按单个变量分别分组，不等于同时控制多个变量；主结论仍以主模型为准。", ""]
    else:
        lines += ["### 分层对比摘要", "", "本轮未产出分层对比（状态：未计算）。", ""]
    overlap = data.get("overlap") or {}
    if overlap:
        lines += ["### 预定义基因集的重叠富集（探索性，无 FDR/q）", "",
                  "来源：本次运行 enrichment_results 中的预定义基因集重叠结果；已读取并解析。", "",
                  "| 对比 | 状态 | 条目数 | 文件中的对比键（臂序） | 各库条目数 |",
                  "|---|---|---:|---|---|"]
        for contrast, info in overlap.items():
            detail = "；".join(
                "%s %s" % (db, "/".join("%s %d" % (key, value) for key, value in counts.items()))
                for db, counts in (info.get("per_database") or {}).items())
            lines.append("| %s | %s | %d | %s（%s） | %s |" % (
                contrast, info.get("status"), info.get("terms"),
                info.get("file_contrast") or "未记录",
                "与本对比臂序相反" if info.get("reversed") else "与本对比臂序一致", detail))
        picks = [pick for info in overlap.values() for pick in (info.get("top_terms") or [])]
        if picks:
            lines += ["", "报告将按同一规则各取一条（每个数据库每个方向种子基因数最多的一条）：", "",
                      "| 对比内的条目 | 库 | 方向 | 通路/语义条目 | 该条种子基因数 | 代表成员 |",
                      "|---|---|---|---|---:|---|"]
            for contrast, info in overlap.items():
                for pick in info.get("top_terms") or []:
                    lines.append("| %s | %s | %s | %s | %s | %s |" % (
                        contrast, pick.get("database"), pick.get("direction"), pick.get("term"),
                        pick.get("seed"), pick.get("members")))
            lines.append("")
        lines += ["", "该结果为集合重叠，未经多重检验校正、没有 FDR/q 值，只能作为探索性方向提示；"
                      "不得写成显著富集。", ""]
    else:
        state_note = {"missing": "本次运行没有该文件，该项分析未执行",
                      "unparsable": "文件存在但解析失败（读取状态，不是未产出）",
                      "empty": "文件存在但是空对象（已执行且无条目）"}.get(
            str(data.get("overlap_state") or ""), "文件不可用")
        lines += ["### 预定义基因集的重叠富集", "",
                  "状态：%s；不得据本项写通路结论，按边界说明处理。" % state_note, ""]
    ora = data.get("ora") or {}
    if ora:
        lines += ["### 正式富集检验（超几何）", "",
                  "- 被检验条目 %s 条，未检验 %s 条" % (ora.get("tested"), ora.get("skipped")),
                  "- 背景集大小 %s；统计检验所用矩阵蛋白数 %s（两个数字口径不同，引用时写明指哪一个）"
                  % (ora.get("background") or "未记录", ora.get("matrix_proteins") or "未记录"),
                  "- 通过 q(BH)≤0.05 的条目：%s 条（已计算；0 表示检验已执行但没有条目通过，不得写成未执行）"
                  % ora.get("significant"),
                  ""]
        classes = ora.get("query_classes") or []
        if classes:
            lines += ["- 查询集按对比分两类，引用 q 值时必须写明属于哪一类："]
            for item in classes:
                if item.get("kind", "").startswith("显著集"):
                    lines.append("  - %s：规则 %s；查询集 n=%s；涉及对比 %s"
                                 % (item.get("kind"), item.get("rule") or "未记录",
                                    item.get("query_sizes") or "未记录",
                                    "、".join(item.get("contrasts") or []) or "未记录"))
                else:
                    lines.append("  - %s：该对比显著集为 %s 个蛋白；兜底查询集 n=%s；涉及对比 %s"
                                 % (item.get("kind"), item.get("significant_set_sizes") or "未记录",
                                    item.get("query_sizes") or "未记录",
                                    "、".join(item.get("contrasts") or []) or "未记录"))
            lines += ["", "- 不得把某一对比的显著蛋白数写成整个数据集的数字；两类查询集的 q 值不可直接比较。",
                      ""]
        elif ora.get("query_rule"):
            lines += ["- 查询集定义：%s" % ora.get("query_rule"), ""]
        if ora.get("best"):
            lines += ["- 下表按 q(BH) 升序列出前 %d 条；被检验条目共 %s 条，其余条目未列出"
                      "（未列出不等于没有结果）。" % (len(ora["best"]), ora.get("tested")), ""]
            lines += ["| 库 | 通路 | 命中 | 查询集 | p | q(BH) |", "|---|---|---:|---:|---:|---:|"]
            for row in ora["best"]:
                lines.append("| %s | %s | %s | %s | %s | %s |" % (
                    row.get("namespace"), row.get("term"), row.get("hits"), row.get("query_size"),
                    row.get("p"), row.get("q")))
            lines.append("")
        for note in ora.get("skipped_notes") or []:
            lines.append("- 未执行的检验：%s" % note)
        lines.append("")
    else:
        lines += ["### 正式富集检验（超几何）", "", "本轮未产出（状态：未计算）。", ""]
    dose = data.get("dose") or {}
    if dose:
        lines += ["### 剂量趋势（Spearman，已计算）", "",
                  "- 模块级 %s 行、蛋白级 %s 行；蛋白级在 q≤0.05 下呈单调趋势 %s 个；最小 q=%s"
                  % (dose.get("module_rows"), dose.get("protein_rows"),
                     dose.get("protein_significant"), dose.get("best_q") or "未记录"),
                  "- 模块级通过 q≤0.05 的条目：%s 条（已计算；0 表示已计算且无模块级显著结果）"
                  % dose.get("module_significant"),
                  ""]
        for note in dose.get("skipped_notes") or []:
            lines.append("- 未执行的检验：%s" % note)
        lines.append("")
        if dose.get("modules"):
            lines += ["- 下表列出模块级前 %d 行；模块级共 %s 行（完整结果见本轮工件的 dose_trend.csv）。"
                      % (len(dose["modules"]), dose.get("module_rows")), ""]
            lines += ["| 药物 | 分层 | 模块 | 样本数 | ρ | p | q(BH) |", "|---|---|---|---:|---:|---:|---:|"]
            for row in dose["modules"]:
                lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                    row.get("drug"), row.get("stratum"), row.get("name"), row.get("n"),
                    row.get("rho"), row.get("p"), row.get("q")))
            lines.append("")
    conc = data.get("concordance") or {}
    excl = data.get("exclusion") or {}
    lines += ["### 对比一致性与候选排除", "",
              "- 对比一致性：%s（%s 对比较）" % (conc.get("status"), conc.get("pairs")),
              "- 候选排除轨迹：%s；被记录为未检出的候选：%s"
              % (excl.get("status"), "、".join(excl.get("excluded") or [])
                 or "无（记录项为分组与实验设计术语，不是蛋白候选）"),
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
        merged = "（一个图注条目可能覆盖同一编号段的多张图）" if png_count > figures["count"] else ""
        capped = ""
        if figures["count"] > len(listed):
            capped = "；此处列出前 %d 条，其余 %d 条未列出" % (len(listed),
                                                          figures["count"] - len(listed))
        lines += ["### 图册（报告按「图号 + 标题 + 文字结论」引用）", "",
                  "- 图注 %s 条%s；visualize_results 下 PNG %s 个%s：%s"
                  % (figures["count"], merged, png_count, capped, "；".join(listed)),
                  "- 图题中的数字由本运行的报告流水线从本运行工件计算或读取（其中 PCA 解释率来自对 "
                  "processed_proteins/ProteinQuant_ComBat.csv 的复算：列均值中心化、缺失以蛋白均值填补）。"
                  "与源表数值冲突时以源表为准，冲突处须标注需核验，不得直接写成已验证事实。",
                  ""]
    else:
        lines += ["### 图册", "", "本轮未产出可引用图件（状态：未计算）。", ""]
    if data.get("not_computed"):
        lines += ["### 本轮未计算的证据（不得写成已有结果）", ""]
        lines += ["- %s" % item for item in data["not_computed"]]
        lines.append("")
    if data.get("task_ids"):
        # R21/F4: the request hands over exactly one task id per scientific task, together with the
        # direction that id stands for; aliases stay valid input, but they are not extra tasks.
        lines += ["### 可用 task_id（每个对比一个；必须逐字使用）", ""]
        for rec in data.get("contrasts") or []:
            if not rec.get("task_id"):
                continue
            lines.append("- %s：对比 %s；正值表示 %s 较高；源差异表 %s%s"
                         % (rec.get("task_id"), rec.get("display_contrast"),
                            rec.get("positive_means") or "第一臂", rec.get("source_contrast") or "未记录",
                            "（符号与此相反）" if rec.get("inverted") else ""))
        aliases = {task: names for task, names in (data.get("task_aliases") or {}).items() if names}
        if aliases:
            lines += ["", "输入兼容：同一对比的其它写法（%s）会被识别为同一个任务，"
                          "它们不是额外任务，不得为它们另写小节。"
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
        body = "\n".join(block["lines"]).strip()
        if not body and not heading:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if not heading:
            sections["extra"].extend(paragraphs)
        elif ("速览" in heading) or ("结论" in heading and "边界" not in heading):
            sections["key_summary"].extend(paragraphs)
        elif "边界" in heading or "局限" in heading:
            sections["boundary"].extend(paragraphs)
        elif "总结" in heading or "推测" in heading or "讨论" in heading:
            sections["summary"].extend(paragraphs)
        elif "资产" in heading or "附录" in heading or "清单" in heading:
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
    first, second = (list(groups or []) + ["A 组", "B 组"])[:2]
    if token.startswith("up") or "group_a_higher" in token or token in {"higher_in_group_a"}:
        return "%s 较高" % first
    if token.startswith("down") or "group_b_higher" in token or token in {"lower_in_group_a"}:
        return "%s 较高" % second
    display = record.get("logFC_display")
    if isinstance(display, (int, float)):
        return "%s 较高" % (first if display >= 0 else second)
    return "方向未判定"
DISPLAY_KEYS = ("显示", "display")
SOURCE_KEYS = ("源表", "源差异表", "source")
ADJP_KEYS = ("adj.p", "adj.p.val", "fdr", "q值")
EFFECT_KEYS = ("logfc", "log2fc", "效应量", "δ", "delta")


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
