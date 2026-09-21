# -*- coding: utf-8 -*-
"""Generation-side evidence records, report-to-fact binding and tri-state fact check.

Scope fixed by the 2026-09-12 review
------------------------------------
* The check only reports contradiction / unverified / pass. It never declares a report
  scientifically correct, and a clean exit with unverified claims is impossible:
  exit 0 = pass (nothing unverified), 1 = contradiction, 2 = no contradiction but claims remain
  unverified. Callers must read the verdict field from the JSON; exit 0 is not a scientific
  acceptance.
* Low match coverage is a generation-side problem, not a rule-writing problem. The fix is to keep
  the records the report was written from - contrast pair (display vs source), direction, effect
  scale, thresholds, full protein group, source file/row - and to bind report sentences to them.
* Clauses that cannot be bound are written to the map as pending review. They are not errors and
  they are never counted as passes.

Artifacts written per run by write_artifacts()
----------------------------------------------
report_evidence_map.json
    fact records + clause bindings + convention notes (display/source sign flips).
report_fact_check.json
    tri-state findings, coverage counters and the verdict.

The checker module under docs/report_selfcontained_20260911 is a thin wrapper over this file, so the
protocol has exactly one implementation.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "2026-09-12"

# --------------------------------------------------------------------------- lexemes
CLAUSE_SPLIT = re.compile(r"[。！？!?；;，,]")
SIGNED_NUMBER = re.compile(r"(?<![A-Za-z0-9_])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
CLAIM_COUNT = re.compile(
    r"(?:显著|通过筛选|n_sig\s*=|通过\s*FDR)\s*[:：]?\s*(\d{1,5})"
    r"|(\d{1,5})\s*个\s*(?:蛋白|候选|条目|蛋白组)?\s*(?:通过筛选|显著)"
)
PROTEIN_GROUP = re.compile(r"\b(?:[A-Z][0-9][A-Z0-9]{4,9}|[OPQ][0-9][A-Z0-9]{3}[0-9](?:;[A-Z0-9]+)?)\b")
GENE = re.compile(r"\b[A-Z][A-Z0-9]{2,7}\b")
CONTRAST_NAME = re.compile(r"[A-Za-z0-9_]+_vs_[A-Za-z0-9_]+")
LABEL = re.compile(r"\b(?:C\d{1,2}|Cluster\s*\d{1,2}|亚群\s*\d{1,2}|Group\s*\d{1,2})\b")

CONV_RAW = ("原始p", "原始 p", "p 值", "p值", "p-value", "未校正", "raw p", "raw-p",
            "unadjusted p", "uncorrected p")
CONV_FDR = ("fdr", "adj.p", "adj p", "校正后", "bh", "q值", "q 值")
CONV_BOTH = ("双阈值", "|logfc|", "log2fc", "效应量")
UP_WORDS = ("上调", "更高", "升高", "上升", "增加", "富集", "positive")
DOWN_WORDS = ("下调", "更低", "降低", "下降", "减少", "negative")
NEGATION = re.compile(r"(没有|未|无|不含|不存在)[^。；，]{0,8}(显著|差异)|不显著|未见显著|无显著|不足以")

GENERIC_STOP = {"LOGFC", "LOG2FC", "FDR", "BH", "VS", "PADJ", "P", "N", "RNA", "DNA", "QC", "PCA", "UMAP",
                "GO", "KEGG", "MS", "ID", "NA", "SD", "CI", "IQR"}


# --------------------------------------------------------------------------- io helpers
def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_csv(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    delimiter = chr(9) if path.suffix.lower() == ".tsv" else ","
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out else None


def sha12_text(text: str) -> str:
    """Canonical short hash of report text: UTF-8 bytes with CRLF and CR folded to LF."""
    normalised = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8", errors="replace")).hexdigest()[:12]


def sha12_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    except Exception:
        return ""


# --------------------------------------------------------------------------- labels
def allowed_labels(run_folder: Path) -> set:
    """Group/cluster labels that exist in this run; aliases only from cluster columns."""
    labels = set()
    label_cols = re.compile(r"cluster|group|state|condition|type|label|batch|donor|亚群|分组|状态", re.I)
    alias_cols = re.compile(r"cluster|亚群|leiden|louvain", re.I)
    sampleinfo = Path(run_folder) / "processed_proteins" / "SampleInfo_Filtered.csv"
    for row in read_csv(sampleinfo):
        for key, value in row.items():
            if not key or not label_cols.search(str(key)) or value in (None, ""):
                continue
            text = str(value).strip()
            if len(text) > 40:
                continue
            labels.add(text)
            labels.add(text.replace("_", " "))
            if alias_cols.search(str(key)) and text.isdigit():
                labels.update({("C" + text), ("Cluster " + text), ("Cluster" + text)})
    for rel in ("evaluation_evidence/core_story_evidence.csv",
                "evaluation_evidence/curated_module_group_summary.csv",
                "evaluation_evidence/dataset_recipe_evidence.csv"):
        for row in read_csv(Path(run_folder) / rel):
            for key in ("group", "group_a", "group_b", "module", "state", "display_contrast", "contrast"):
                value = str(row.get(key) or "").strip()
                if value:
                    labels.add(value)
                    labels.add(value.replace("_", " "))
    expanded = set(labels)
    for value in labels:
        for token in re.split(r"[^A-Za-z0-9]+", value):
            if token and not token.isdigit():
                expanded.add(token)
    return expanded


# --------------------------------------------------------------------------- facts
def count_conventions(rows: List[Dict[str, str]], thresholds: Dict[str, Any]) -> Dict[str, int]:
    n_raw = n_fdr = n_both = n_raw_both = 0
    for row in rows:
        p = to_float(row.get("P.Value"))
        adj = to_float(row.get("adj.P.Val"))
        logfc = to_float(row.get("logFC"))
        if adj is not None and adj < thresholds["p"]:
            n_fdr += 1
            if logfc is not None and abs(logfc) > thresholds["logfc"]:
                n_both += 1
        if p is not None and p < thresholds["p"]:
            n_raw += 1
            if logfc is not None and abs(logfc) > thresholds["logfc"]:
                n_raw_both += 1
    return {"raw P": n_raw, "adj.P": n_fdr, "adj.P+effect": n_both, "raw P+effect": n_raw_both,
            "rows": len(rows)}


def _story_records(run_folder: Path) -> Dict[str, Any]:
    """Contrast / candidate / module records the report was written from."""
    payload = read_json(Path(run_folder) / "evaluation_evidence" / "core_story_evidence.json") or {}
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    contrasts: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    modules: List[Dict[str, Any]] = []
    conventions: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = row.get("row_type")
        display = str(row.get("display_contrast") or "")
        source = str(row.get("source_contrast") or "")
        if display and source and display != source:
            key = (display, source)
            conv = conventions.setdefault(key, {
                "display_contrast": display, "source_contrast": source, "inverted": True, "flip": -1,
                "sign_rule": "logFC_display = -logFC_source",
                "groups": {"display_a": str(row.get("group_a") or ""), "display_b": str(row.get("group_b") or "")},
                "source_file": str(row.get("source_file") or ""), "examples": [],
            })
            gene = str(row.get("candidate") or "")
            if gene and len(conv["examples"]) < 4:
                conv["examples"].append({
                    "subject": gene,
                    "protein_group": str(row.get("matched_protein") or ""),
                    "logFC_display": to_float(row.get("logFC_display")),
                    "logFC_source": None,
                    "source_file": str(row.get("source_file") or ""),
                })
        if kind == "contrast":
            contrasts.append({
                "claim_id": str(row.get("claim_id") or ""),
                "claim_title": str(row.get("claim_title") or ""),
                "display_contrast": display, "source_contrast": source,
                "group_a": str(row.get("group_a") or ""), "group_b": str(row.get("group_b") or ""),
                "inverted_from_source": bool(row.get("inverted_from_source")),
                "n_sig": to_float(row.get("n_sig")),
                "n_up_display_group_a": to_float(row.get("n_up_display_group_a")),
                "n_down_display_group_a": to_float(row.get("n_down_display_group_a")),
                "source_file": str(row.get("source_file") or ""),
                "evidence_source": str(row.get("evidence_source") or ""),
                "confidence": str(row.get("confidence") or ""),
            })
        elif kind == "candidate":
            candidates.append({
                "claim_id": str(row.get("claim_id") or ""),
                "candidate": str(row.get("candidate") or ""),
                "matched_gene": str(row.get("matched_gene") or ""),
                "protein_group": str(row.get("matched_protein") or ""),
                "display_contrast": display, "source_contrast": source,
                "logFC_display": to_float(row.get("logFC_display")),
                "logFC_source": None,
                "P.Value": to_float(row.get("P.Value")),
                "adj.P.Val": to_float(row.get("adj.P.Val")),
                "direction_display": str(row.get("direction_display") or ""),
                "source_file": str(row.get("source_file") or ""),
                "confidence": str(row.get("confidence") or ""),
            })
        elif kind == "module":
            modules.append({
                "claim_id": str(row.get("claim_id") or ""),
                "module": str(row.get("module") or ""),
                "display_contrast": display,
                "group_a": str(row.get("group_a") or ""), "group_b": str(row.get("group_b") or ""),
                "group_a_mean_score": to_float(row.get("group_a_mean_score")),
                "group_b_mean_score": to_float(row.get("group_b_mean_score")),
                "delta": to_float(row.get("module_delta_group_a_minus_group_b")),
                "matched_genes": str(row.get("matched_genes") or ""),
                "n_matched_genes": to_float(row.get("n_matched_genes")),
            })
    return {"contrasts": contrasts, "candidates": candidates, "modules": modules,
            "conventions": [conventions[key] for key in sorted(conventions)]}


def build_facts(run_folder: Path) -> Dict[str, Any]:
    """One structured record set per run; the checks and the map both read this."""
    run_folder = Path(run_folder)
    params = read_json(run_folder / "parameters.json") or {}
    design = params.get("analysis_design") or {}
    diff = design.get("differential") or {}
    thresholds = {"p": float(diff.get("p_thresh") or 0.05),
                  "logfc": float(diff.get("logfc_thresh") or 0.25),
                  "fdr_method": str(diff.get("fdr_method") or "BH")}
    matrix = design.get("matrix") or {}
    if matrix.get("log2_transform_applied") is True or matrix.get("matrix_state") in {"logged", "log2"}:
        scale = "log2"
    elif matrix.get("looks_logged") is True:
        scale = "log2 (inferred from value range; transform record absent)"
    else:
        scale = "unknown"

    contrasts: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(str(run_folder / "processed_proteins" / "differential_*.csv"))):
        name = os.path.basename(path).replace("differential_", "").replace(".csv", "")
        rows = read_csv(Path(path))
        by_group: Dict[str, Dict[str, str]] = {}
        by_gene: Dict[str, List[str]] = {}
        for row in rows:
            group = str(row.get("PG.ProteinGroups") or row.get("protein") or "").strip()
            gene = str(row.get("PG.Genes") or row.get("gene") or "").split(";")[0].strip()
            if group:
                by_group[group] = row
            if gene:
                by_gene.setdefault(gene, []).append(group)
        contrasts[name] = {"rows": rows, "by_group": by_group, "by_gene": by_gene,
                           "counts": count_conventions(rows, thresholds), "source": str(path)}

    design_contrasts = [{"name": str(c.get("name", "")), "a": str(c.get("group_a", "")),
                         "b": str(c.get("group_b", ""))} for c in (diff.get("contrasts") or [])]
    if not design_contrasts:
        for name in contrasts:
            if "_vs_" in name:
                a, b = name.split("_vs_", 1)
                design_contrasts.append({"name": name, "a": a, "b": b})

    story = _story_records(run_folder)
    for rec in story["candidates"]:
        table = contrasts.get(rec.get("source_contrast") or "")
        if not table:
            continue
        row = table["by_group"].get(rec.get("protein_group") or "")
        if row is None and rec.get("matched_gene"):
            groups = table["by_gene"].get(rec["matched_gene"]) or []
            if groups:
                row = table["by_group"].get(groups[0])
        if row is not None:
            rec["logFC_source"] = to_float(row.get("logFC"))
            rec["protein_group"] = rec.get("protein_group") or str(row.get("PG.ProteinGroups") or "")
    for conv in story["conventions"]:
        for example in conv["examples"]:
            table = contrasts.get(conv["source_contrast"])
            if table and example.get("logFC_source") is None:
                row = table["by_group"].get(example.get("protein_group") or "")
                if row is not None:
                    example["logFC_source"] = to_float(row.get("logFC"))

    conventions_by_display = {}
    for conv in story["conventions"]:
        conventions_by_display[conv["display_contrast"].lower()] = conv
    for rec in story["contrasts"]:
        display = str(rec.get("display_contrast") or "").lower()
        if display and display not in conventions_by_display and rec.get("source_contrast"):
            conventions_by_display[display] = {"display_contrast": rec.get("display_contrast"),
                                               "source_contrast": rec.get("source_contrast"),
                                               "inverted": False, "flip": 1,
                                               "sign_rule": "same orientation", "examples": [],
                                               "source_file": rec.get("source_file")}

    records: List[Dict[str, Any]] = []
    for rec in story["contrasts"]:
        records.append({"record_id": "contrast:%s" % (rec.get("display_contrast") or rec.get("source_contrast")),
                        "kind": "contrast", **rec})
    for rec in story["candidates"]:
        records.append({"record_id": "candidate:%s:%s" % (rec.get("display_contrast"), rec.get("candidate")),
                        "kind": "candidate", **rec})
    for rec in story["modules"]:
        records.append({"record_id": "module:%s:%s" % (rec.get("display_contrast"), rec.get("module")),
                        "kind": "module", **rec})

    contrast_record_names = set()
    for _rec in story["contrasts"]:
        for _name in (_rec.get("display_contrast"), _rec.get("source_contrast")):
            if _name:
                contrast_record_names.add(str(_name))
    for _name in sorted(contrasts):
        if _name in contrast_record_names:
            continue
        _counts = contrasts[_name].get("counts") or {}
        records.append({"record_id": "contrast:%s" % _name, "kind": "contrast",
                        "display_contrast": _name, "source_contrast": _name,
                        "group_a": "", "group_b": "",
                        "n_sig": _counts.get("adj.P+effect"),
                        "n_raw_p": _counts.get("raw P"),
                        "source_file": contrasts[_name].get("source", ""),
                        "evidence_source": "current_matrix", "confidence": "moderate",
                        "record_note": "auto record for a differential table referenced by the report"})
    alias_to_record: Dict[str, str] = {}
    for rec in records:
        for name in (rec.get("display_contrast"), rec.get("source_contrast")):
            key: str = _alias_key(name)
            if key:
                alias_to_record.setdefault(key, rec["record_id"])
    return {"run_folder": str(run_folder), "schema": SCHEMA_VERSION, "contrasts": contrasts,
            "design": design_contrasts, "thresholds": thresholds, "labels": allowed_labels(run_folder),
            "scale": scale, "story": {**story, "conventions_by_display": conventions_by_display},
            "records": records, "alias_to_record": alias_to_record}


# --------------------------------------------------------------------------- clause helpers
def clauses(text: str) -> List[Dict[str, Any]]:
    """Report text -> clauses with line number, section and the last contrast seen in context."""
    out: List[Dict[str, Any]] = []
    section = ""
    context = ""
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
        match = CONTRAST_NAME.search(line.replace(" ", "_"))
        if match:
            context = match.group(0).strip("_")
        for piece in CLAUSE_SPLIT.split(line):
            piece = piece.strip(" \t|*_\u3000")
            if piece:
                out.append({"line": idx, "section": section, "text": piece, "context_contrast": context})
    return out


def _find_token(text: str, token: str) -> int:
    if not token or len(token) < 3:
        return -1
    match = re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", text)
    return match.start() if match else -1


def _group_hits(text: str, token: str) -> int:
    """Position of a group token, accepting both the raw and the underscore-spaced spelling."""
    best = -1
    for form in {token, token.replace("_", " ")}:
        if not form or len(form) < 3:
            continue
        idx = _find_token(text, form)
        if idx >= 0 and (best < 0 or idx < best):
            best = idx
    return best


def resolve_contrast(clause: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a clause to (table, orientation, sign, flip).

    sign is +1 when the clause names the first group of the stated orientation first; flip converts
    source-table values into the display orientation (-1 when the display contrast is inverted).
    """
    text = clause.replace("_", " ")
    match = CONTRAST_NAME.search(clause.replace(" ", "_"))
    if match:
        key = match.group(0).strip("_")
        blocked_reason = ""
        for name in facts["contrasts"]:
            if name.lower() == key.lower():
                return {"table": name, "sign": 1, "orientation": "source", "flip": 1, "reason": ""}
        conv = facts["story"]["conventions_by_display"].get(key.lower())
        if conv and conv.get("source_contrast") in facts["contrasts"]:
            return {"table": conv["source_contrast"], "sign": 1, "orientation": "display",
                    "flip": conv.get("flip", -1), "reason": ""}
        blocked_reason = "对比名 %s 在本轮差异表中不存在" % key
    else:
        blocked_reason = ""
    story_both, story_single = [], []
    for rec in facts["story"]["contrasts"]:
        a, b = rec.get("group_a") or "", rec.get("group_b") or ""
        source = rec.get("source_contrast") or ""
        if not (a and b and source in facts["contrasts"]):
            continue
        flip = -1 if rec.get("inverted_from_source") else 1
        pa, pb = _group_hits(text, a), _group_hits(text, b)
        if pa >= 0 and pb >= 0 and a != b:
            story_both.append((source, (+1 if pa < pb else -1), "display", flip))
        elif pa >= 0 and pb < 0:
            story_single.append((source, +1, "display", flip))
        elif pb >= 0 and pa < 0:
            story_single.append((source, -1, "display", flip))
    if story_both:
        table, sign, orientation, flip = story_both[0]
        return {"table": table, "sign": sign, "orientation": orientation, "flip": flip, "reason": ""}
    design_both, design_single = [], []
    for entry in facts["design"]:
        a, b = entry["a"], entry["b"]
        if not (a and b) or entry["name"] not in facts["contrasts"]:
            continue
        pa, pb = _group_hits(text, a), _group_hits(text, b)
        if pa >= 0 and pb >= 0 and a != b:
            design_both.append((entry["name"], (+1 if pa < pb else -1), "source", 1))
        elif pa >= 0 and pb < 0:
            design_single.append((entry["name"], +1, "source", 1))
        elif pb >= 0 and pa < 0:
            design_single.append((entry["name"], -1, "source", 1))
    if design_both:
        table, sign, orientation, flip = design_both[0]
        return {"table": table, "sign": sign, "orientation": orientation, "flip": flip, "reason": ""}
    for matches, note in ((story_single, "display"), (design_single, "source")):
        unique = sorted(set(matches))
        if len(unique) == 1:
            table, sign, orientation, flip = unique[0]
            return {"table": table, "sign": sign, "orientation": orientation, "flip": flip, "reason": ""}
        if len(unique) > 1:
            return {"table": None, "sign": 0, "orientation": note, "flip": 1,
                    "reason": "句中只出现一个分组名，且可对应多个对比（需点明对比名）"}
    return {"table": None, "sign": 0, "orientation": "source", "flip": 1,
            "reason": blocked_reason or "未在句中识别出对比"}


def conventions_in(clause: str) -> set:
    """Single counting convention named by the clause, or an empty set when it names none.

    A compound statement of one convention ("adj.P<0.05 and |logFC|>0.25") is the double-threshold
    convention, not a mix-up; only raw-P and FDR tokens together (without an effect-size token)
    signal that the same number is being used for two different conventions.
    """
    low = clause.lower()
    raw = any(t in low for t in CONV_RAW)
    fdr = any(t in low for t in CONV_FDR)
    both = any(t in low for t in CONV_BOTH)
    if both and raw and not fdr:
        return {"raw P+effect"}
    if both and fdr and not raw:
        return {"adj.P+effect"}
    if both and not raw and not fdr:
        return {"adj.P+effect"}
    if fdr and not raw:
        return {"adj.P"}
    if raw and not fdr:
        return {"raw P"}
    return set()


def gene_stopwords(facts: Dict[str, Any]) -> set:
    stop = set(GENERIC_STOP)
    for entry in facts["design"]:
        for token in (entry["a"], entry["b"]):
            for part in re.split(r"[^A-Za-z0-9]+", token or ""):
                if part:
                    stop.add(part.upper())
    for name in facts["contrasts"]:
        for part in re.split(r"[^A-Za-z0-9]+", name):
            if part:
                stop.add(part.upper())
    return stop


def genes_in(clause: str, facts: Dict[str, Any]) -> List[str]:
    stop = gene_stopwords(facts)
    return [g for g in GENE.findall(clause) if len(g) >= 3 and g not in stop]


def subject_of(clause: str, facts: Dict[str, Any], table_name: str):
    """Resolve the protein a clause talks about: (kind, key, row, reason)."""
    table = facts["contrasts"].get(table_name)
    if not table:
        return None, None, None, "对比 %s 无差异表" % table_name
    for group in PROTEIN_GROUP.findall(clause):
        for part in (group, group.split(";")[0]):
            if part in table["by_group"]:
                return "protein_group", part, table["by_group"][part], ""
    for gene in genes_in(clause, facts):
        groups = table["by_gene"].get(gene, [])
        if len(groups) == 1:
            return "gene", gene, table["by_group"][groups[0]], ""
        if len(groups) > 1:
            signs = {1 if (to_float(table["by_group"][g].get("logFC")) or 0) > 0 else -1 for g in groups}
            if len(signs) > 1:
                return None, None, None, "基因 %s 对应多个蛋白组且符号不一致（需指定蛋白组）" % gene
            return "gene", gene, table["by_group"][groups[0]], ""
    return None, None, None, "句中未识别到可比对的蛋白"


# --------------------------------------------------------------------------- checks
def check_labels(clause_list: List[Dict[str, Any]], facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    for clause in clause_list:
        for match in LABEL.finditer(clause["text"]):
            token = match.group(0).strip()
            if token in facts["labels"] or token.replace(" ", "") in facts["labels"]:
                continue
            findings.append({"check": "label", "state": "contradiction", "label": token,
                             "reason": "该标签在本轮工件中不存在", "line": clause["line"],
                             "section": clause["section"], "clause": clause["text"][:140]})
    return findings


def check_counts(clause_list: List[Dict[str, Any]], facts: Dict[str, Any]):
    findings: List[Dict[str, Any]] = []
    unverified: List[Dict[str, Any]] = []
    passed = 0
    for clause in clause_list:
        text = clause["text"]
        if not any(w in text for w in ("显著", "通过筛选", "蛋白")):
            continue
        numbers = [int(x) for pair in CLAIM_COUNT.findall(text) for x in pair if x]
        if not numbers:
            continue
        resolved = resolve_contrast(text, facts)
        if not resolved["table"]:
            unverified.append({"check": "count", "state": "unverified", "reason": resolved["reason"],
                               "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        table = facts["contrasts"][resolved["table"]]
        convs = conventions_in(text)
        if len(convs) == 0:
            unverified.append({"check": "count", "state": "unverified",
                               "reason": "句子未标明计数口径（原始 P / FDR / 双阈值）",
                               "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        if len(convs) > 1:
            unverified.append({"check": "count", "state": "unverified",
                               "reason": "同一分句含多种口径，无法绑定：%s" % sorted(convs),
                               "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        conv = sorted(convs)[0]
        expected = table["counts"][conv]
        for value in numbers:
            if value == expected:
                passed += 1
            elif value > 9:
                findings.append({"check": "count", "state": "contradiction", "contrast": resolved["table"],
                                 "orientation": resolved["orientation"], "convention": conv,
                                 "stated": value, "expected": expected, "counts": table["counts"],
                                 "line": clause["line"], "section": clause["section"], "clause": text[:140]})
    return findings, unverified, passed


def check_direction(clause_list: List[Dict[str, Any]], facts: Dict[str, Any]):
    findings: List[Dict[str, Any]] = []
    unverified: List[Dict[str, Any]] = []
    passed = 0
    for clause in clause_list:
        text = clause["text"]
        # only a stated value ("logFC=1.2") is a direction claim; a threshold such as "|logFC|>0.25"
        # names the counting convention and carries no direction
        has_number_token = bool(re.search(r"(logfc|log2fc|倍数变化|效应量)\s*[:：=≈]\s*-?\d", text, re.I))
        has_word = any(w in text.lower() for w in UP_WORDS + DOWN_WORDS)
        if not (has_number_token or has_word):
            continue
        resolved = resolve_contrast(text, facts)
        if not resolved["table"]:
            unverified.append({"check": "direction", "state": "unverified", "reason": resolved["reason"],
                               "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        kind, key, row, why = subject_of(text, facts, resolved["table"])
        if row is None:
            unverified.append({"check": "direction", "state": "unverified", "reason": why,
                               "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        table_logfc = to_float(row.get("logFC"))
        if table_logfc is None:
            unverified.append({"check": "direction", "state": "unverified", "reason": "差异表缺少 logFC",
                               "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        oriented = table_logfc * resolved["flip"] * resolved["sign"]
        numbers = [to_float(x) for x in SIGNED_NUMBER.findall(text)]
        numbers = [n for n in numbers if n is not None and abs(n) > 1e-9]
        matched = None
        for value in numbers:
            if abs(abs(value) - abs(table_logfc)) <= max(0.02, abs(table_logfc) * 0.03):
                matched = value
                break
        words_up = any(w in text.lower() for w in UP_WORDS)
        words_down = any(w in text.lower() for w in DOWN_WORDS)
        if matched is None:
            if words_up or words_down:
                expected_positive = oriented > 0
                claimed_up = words_up and not words_down
                if claimed_up != expected_positive and (words_up ^ words_down):
                    findings.append({"check": "direction", "state": "contradiction",
                                     "contrast": resolved["table"], "orientation": resolved["orientation"],
                                     "subject": key, "table_logFC": table_logfc, "stated": "方向词",
                                     "line": clause["line"], "section": clause["section"], "clause": text[:140]})
                else:
                    passed += 1
            else:
                unverified.append({"check": "direction", "state": "unverified",
                                   "reason": "句中未给出可与差异表比对的数值或方向词",
                                   "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        expected_positive = oriented > 0
        claimed_positive = matched > 0
        if claimed_positive != expected_positive:
            findings.append({"check": "direction", "state": "contradiction", "contrast": resolved["table"],
                             "orientation": resolved["orientation"], "subject": key, "kind": kind,
                             "table_logFC": table_logfc, "stated": matched,
                             "line": clause["line"], "section": clause["section"], "clause": text[:140]})
        elif (words_up and not words_down and not expected_positive) or (words_down and not words_up and expected_positive):
            findings.append({"check": "direction", "state": "contradiction", "contrast": resolved["table"],
                             "orientation": resolved["orientation"], "subject": key, "table_logFC": table_logfc,
                             "stated": "方向词与数值/表格相反", "line": clause["line"],
                             "section": clause["section"], "clause": text[:140]})
        else:
            passed += 1
    return findings, unverified, passed


def check_claims(clause_list: List[Dict[str, Any]], facts: Dict[str, Any]):
    findings: List[Dict[str, Any]] = []
    unverified: List[Dict[str, Any]] = []
    passed = 0
    for clause in clause_list:
        text = clause["text"]
        if NEGATION.search(text):
            continue
        if not any(w in text for w in ("显著", "通过筛选")):
            continue
        resolved = resolve_contrast(text, facts)
        if not resolved["table"]:
            continue
        has_subject = bool(PROTEIN_GROUP.search(text)) or bool(genes_in(text, facts))
        if not has_subject:
            continue
        kind, key, row, why = subject_of(text, facts, resolved["table"])
        if row is None:
            unverified.append({"check": "claim", "state": "unverified", "reason": why,
                               "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        convs = conventions_in(text) or {"adj.P"}
        conv = sorted(convs)[0]
        p = to_float(row.get("adj.P.Val")) if conv == "adj.P" else to_float(row.get("P.Value"))
        if p is None:
            unverified.append({"check": "claim", "state": "unverified", "reason": "差异表缺少对应 p 值",
                               "line": clause["line"], "section": clause["section"], "clause": text[:140]})
            continue
        if p < facts["thresholds"]["p"]:
            passed += 1
        else:
            findings.append({"check": "claim", "state": "contradiction", "contrast": resolved["table"],
                             "subject": key, "convention": conv, "value": p,
                             "line": clause["line"], "section": clause["section"], "clause": text[:140]})
    return findings, unverified, passed


# --------------------------------------------------------------------------- binding / map
LEVEL_MEANING = {
    "unbound": "no contrast or fact could be attached to this clause",
    "contrast_context": "the comparison was recognised, but no specific protein/module/value was tied to it",
    "fact_located": "a specific fact record (protein group, gene or module) was located for this clause",
    "verified": "the number or direction stated in the clause matches that record's value in the stated orientation",
}


def _locate_record(facts: Dict[str, Any], table_name: str, key: Any) -> Optional[Dict[str, Any]]:
    """The candidate record a clause talks about, if the subject maps to one."""
    if not key or not table_name:
        return None
    for rec in facts["records"]:
        if rec.get("kind") != "candidate":
            continue
        if (rec.get("source_contrast") or "") != table_name:
            continue
        if str(key) in {str(rec.get("candidate") or ""), str(rec.get("matched_gene") or ""),
                        str(rec.get("protein_group") or "")}:
            return rec
    return None


def _expected_value(rec: Dict[str, Any], orientation: str, flip: int) -> Optional[float]:
    """The record's value expressed in the orientation the clause used."""
    if rec.get("kind") == "module":
        _delta = rec.get("delta")
        if not isinstance(_delta, (int, float)):
            return None
        return float(_delta) if orientation != "source" else float(_delta) * (flip or 1)
    display, source = rec.get("logFC_display"), rec.get("logFC_source")
    if orientation == "display":
        if display is not None:
            return display
        return None if source is None else source * flip
    if source is not None:
        return source
    return None if display is None else display * flip


def _clause_verified(text: str, resolved: Dict[str, Any], rec: Optional[Dict[str, Any]]) -> bool:
    """True when the stated number (or direction word) agrees with the located record's value."""
    if rec is None:
        return False
    expected = _expected_value(rec, resolved.get("orientation", "source"), resolved.get("flip", 1) or 1)
    if expected is None:
        return False
    numbers = [to_float(x) for x in SIGNED_NUMBER.findall(text)]
    for value in [n for n in numbers if n is not None and abs(n) > 1e-9]:
        if abs(value - expected) <= max(0.02, abs(expected) * 0.03):
            return True
    low = text.lower()
    words_up = any(w in low for w in UP_WORDS)
    words_down = any(w in low for w in DOWN_WORDS)
    if words_up != words_down:
        return (words_up and expected > 0) or (words_down and expected < 0)
    return False


def _alias_key(value: Any) -> str:
    """Normalised key for matching contrast / module names across spellings."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _contrast_record_id(facts: Dict[str, Any], table_name: str) -> str:
    """Canonical record id for a differential-table name (display or source spelling)."""
    rid = (facts.get("alias_to_record") or {}).get(_alias_key(table_name))
    return rid or ("contrast:%s" % table_name)


def _module_records_for(text: str, facts: Dict[str, Any], table_name: str = "") -> List[Dict[str, Any]]:
    """Module records whose module name appears in the clause (contrast-matched first)."""
    flat = _alias_key(text)
    canonical = _contrast_record_id(facts, table_name) if table_name else ""
    exact: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    for rec in facts["records"]:
        if rec.get("kind") != "module":
            continue
        name = _alias_key(rec.get("module"))
        if not name or len(name) < 4 or name not in flat:
            continue
        if canonical and ("contrast:%s" % (rec.get("display_contrast") or "")) == canonical:
            exact.append(rec)
        else:
            other.append(rec)
    return exact or other


def _value_check(text: str, resolved: Dict[str, Any], rec: Optional[Dict[str, Any]]) -> str:
    """Whether a clause cites a number that belongs to the located record.

    Deliberately conservative: any of the record's own values counts as a match, so a mismatch
    is reported only when no stated number corresponds to any of them.
    """
    if rec is None:
        return "no_record"
    if text.strip().startswith("|") or " | " in text:
        return "table_row"
    expected = _expected_value(rec, resolved.get("orientation", "source"), resolved.get("flip", 1) or 1)
    if expected is None:
        return "no_value"
    allowed = [expected, -expected]
    kind = str(rec.get("kind") or "")
    if kind == "module":
        keys = ("group_a_mean_score", "group_b_mean_score")
    elif kind == "candidate":
        keys = ("P.Value", "adj.P.Val", "logFC_display")
    else:
        keys = ("n_sig", "n_up_display_group_a", "n_down_display_group_a")
    for _key in keys:
        _value = rec.get(_key)
        if isinstance(_value, (int, float)):
            allowed.extend([float(_value), -float(_value)])
    numbers = [n for n in (to_float(x) for x in SIGNED_NUMBER.findall(text)) if n is not None and abs(n) > 1e-9]
    if not numbers:
        return "direction_only" if _clause_verified(text, resolved, rec) else "no_number"
    for value in numbers:
        for _allowed in allowed:
            if abs(value - _allowed) <= max(0.02, abs(_allowed) * 0.03):
                return "matched"
    return "value_mismatch"

def bind_report(report_text: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Bind report clauses to fact records at three levels (see LEVEL_MEANING).

    The levels are deliberately separated because "the comparison was recognised" and "the stated
    number was checked against the record" are different claims: model screening must not rank on
    the raw bound count, which only proves contrast context.
    """
    clause_list = clauses(report_text)
    record_ids_by_subject: Dict[Tuple[str, str], List[str]] = {}
    for rec in facts["records"]:
        if rec["kind"] != "candidate":
            continue
        source = rec.get("source_contrast") or ""
        for key in filter(None, (rec.get("candidate"), rec.get("matched_gene"))):
            record_ids_by_subject.setdefault((source, str(key)), []).append(rec["record_id"])
        if rec.get("protein_group"):
            record_ids_by_subject.setdefault((source, str(rec["protein_group"])), []).append(rec["record_id"])
    known_ids = {rec["record_id"] for rec in facts["records"]}
    records_by_id = {rec["record_id"]: rec for rec in facts["records"]}
    refs_total = 0
    refs_resolved = 0
    dangling: Dict[str, int] = {}

    subject_terms = set()
    for rec in facts["records"]:
        if rec["kind"] == "candidate":
            for key in filter(None, (rec.get("candidate"), rec.get("matched_gene"))):
                subject_terms.add(str(key).upper())

    bindings: List[Dict[str, Any]] = []
    n_claim = 0
    n_claim_bound = 0
    levels = {name: 0 for name in LEVEL_MEANING}
    for clause in clause_list:
        text = clause["text"]
        resolved = resolve_contrast(text, facts)
        route = ""
        table_name = resolved["table"]
        if table_name:
            route = "explicit_contrast" if CONTRAST_NAME.search(text.replace(" ", "_")) else "group_name"
        if not table_name and clause.get("context_contrast"):
            context = resolve_contrast(clause["context_contrast"], facts)
            if context["table"]:
                table_name = context["table"]
                route = "context"
        records: List[str] = []
        key = None
        row = None
        module_rec = None
        if table_name:
            records.append(_contrast_record_id(facts, table_name))
            _kind, key, row, _why = subject_of(text, facts, table_name)
            records.extend(sorted(set(record_ids_by_subject.get((table_name, str(key)), []))))
            for _mrec in _module_records_for(text, facts, table_name):
                records.append(_mrec["record_id"])
                if module_rec is None:
                    module_rec = _mrec
        record = _locate_record(facts, table_name, key) if table_name else None
        if record is None and module_rec is not None:
            record = module_rec
        level = "unbound"
        if records:
            level = "contrast_context"
            if (row is not None or module_rec is not None
                    or any(not str(r).startswith("contrast:") for r in records)):
                level = "fact_located"
            if _clause_verified(text, resolved, record):
                level = "verified"
        has_claim = (bool(SIGNED_NUMBER.search(text)) or "显著" in text or "n_sig" in text
                     or any(w in text.lower() for w in UP_WORDS + DOWN_WORDS)
                     or bool(subject_terms.intersection({g.upper() for g in GENE.findall(text)})))
        if has_claim:
            n_claim += 1
            levels[level] += 1
            if level != "unbound":
                n_claim_bound += 1
        value_check = _value_check(text, resolved, record)
        if value_check == "value_mismatch":
            _primary = (record or {}).get("record_id")
            for _rid in sorted(set(records)):
                if _rid == _primary:
                    continue
                _alt = records_by_id.get(_rid)
                if _alt is not None and _value_check(text, resolved, _alt) == "matched":
                    value_check = "matched_other_record"
                    break
        for _ref in sorted(set(records)):
            refs_total += 1
            if _ref in known_ids:
                refs_resolved += 1
            else:
                dangling[_ref] = dangling.get(_ref, 0) + 1
        bindings.append({
            "line": clause["line"], "section": clause["section"],
            "clause": text[:200], "has_claim": has_claim,
            "status": "bound" if records else "unbound",
            "level": level,
            "route": route if records else "",
            "contrast": resolved["table"], "orientation": resolved["orientation"],
            "subject": key, "records": sorted(set(records)),
            "value_check": value_check,
            "reason": "" if records else (resolved["reason"] or "未绑定到任何事实记录"),
        })
    reference_counts: Dict[str, int] = {}
    for _b in bindings:
        for _ref in _b["records"]:
            reference_counts[_ref] = reference_counts.get(_ref, 0) + 1
    never_referenced = sorted(known_ids - set(reference_counts))
    value_checks: Dict[str, int] = {}
    for _b in bindings:
        value_checks[_b["value_check"]] = value_checks.get(_b["value_check"], 0) + 1
    value_mismatch_lines = [{"line": _b["line"], "section": _b["section"],
                            "records": _b["records"], "clause": _b["clause"]}
                           for _b in bindings if _b["value_check"] == "value_mismatch"]
    coverage = {
        "clauses_total": len(clause_list),
        "record_refs_total": refs_total,
        "record_refs_resolved": refs_resolved,
        "dangling_refs": sum(dangling.values()),
        "dangling_examples": [{"ref": _k, "count": _v}
                              for _k, _v in sorted(dangling.items(), key=lambda kv: -kv[1])[:10]],
        "records_never_referenced": never_referenced,
        "value_checks": value_checks,
        "value_mismatch_lines": value_mismatch_lines[:20],
        "clauses_with_claim": n_claim,
        "claims_bound": n_claim_bound,
        "claims_unbound": n_claim - n_claim_bound,
        "levels": levels,
        "level_meaning": LEVEL_MEANING,
        "pending_review": [{"line": b["line"], "section": b["section"], "clause": b["clause"], "reason": b["reason"]}
                           for b in bindings if b["has_claim"] and b["status"] == "unbound"],
    }
    return {"bindings": bindings, "coverage": coverage}


def check_report(report_text: str, facts: Dict[str, Any], report_ref: str = "") -> Dict[str, Any]:
    clause_list = clauses(report_text)
    contradictions = check_labels(clause_list, facts)
    unverified: List[Dict[str, Any]] = []
    passed = 0
    for fn in (check_counts, check_direction, check_claims):
        found, unv, ok = fn(clause_list, facts)
        contradictions.extend(found)
        unverified.extend(unv)
        passed += ok
    verdict = "contradiction" if contradictions else ("unverified" if unverified else "pass")
    return {
        "report": report_ref,
        "report_sha12": sha12_text(report_text),
        "contrasts": sorted(facts["contrasts"]),
        "thresholds": facts["thresholds"],
        "effect_scale": facts["scale"],
        "n_labels_allowed": len(facts["labels"]),
        "contradictions": contradictions,
        "unverified": unverified,
        "n_contradictions": len(contradictions),
        "n_unverified": len(unverified),
        "n_passed": passed,
        "coverage": {"contradiction": len(contradictions), "unverified": len(unverified), "passed": passed},
        "verdict": verdict,
        "verdict_meaning": {
            "pass": "no contradiction and no unverified claim was found by this check",
            "unverified": "no contradiction found, but claims remain unbound - this is not a pass",
            "contradiction": "at least one bound claim contradicts the artifacts",
        }[verdict],
        "scope_note": "Check-item counts, not full-text fact coverage. The check reports contradictions "
                      "and unbound claims; it does not declare the report scientifically correct.",
    }


def run_checks(run_folder, report_path=None) -> Dict[str, Any]:
    """Backwards-compatible entry used by the paired tests and the docs wrapper."""
    run_folder = Path(run_folder)
    report_path = Path(report_path) if report_path else run_folder / "report.md"
    text = read_text(report_path)
    facts = build_facts(run_folder)
    result = check_report(text, facts, report_ref=str(report_path))
    result["conventions"] = facts["story"]["conventions"]
    return result


def write_artifacts(run_folder, report_text: Optional[str] = None, report_path=None,
                    out_dir=None) -> Dict[str, Any]:
    """Write report_evidence_map.json and report_fact_check.json; never raises on content."""
    run_folder = Path(run_folder)
    out_dir = Path(out_dir) if out_dir else run_folder
    report_path = Path(report_path) if report_path else run_folder / "report.md"
    if report_text is None:
        report_text = read_text(report_path)
    facts = build_facts(run_folder)
    result = check_report(report_text, facts, report_ref=str(report_path))
    binding = bind_report(report_text, facts)
    conventions = facts["story"]["conventions"]
    map_payload = {
        "schema": SCHEMA_VERSION,
        "run_folder": str(run_folder),
        "report": str(report_path),
        "report_sha12": result["report_sha12"],
        "effect_scale": facts["scale"],
        "thresholds": facts["thresholds"],
        "conventions": conventions,
        "records": facts["records"],
        "bindings": binding["bindings"],
        "coverage": binding["coverage"],
        "note": "Facts the report was written from. Binding levels: contrast_context (comparison "
                "recognised), fact_located (a specific record was located), verified (the stated "
                "value/direction matches that record). Unbound clauses are pending review, not "
                "errors; do not rank models on the bound count alone.",
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report_evidence_map.json").write_text(
            json.dumps(map_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "report_fact_check.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # diagnostics must never break a run
        result = dict(result)
        result["write_error"] = str(exc)
    result["map"] = {"records": len(facts["records"]), "bindings": len(binding["bindings"]),
                     "coverage": binding["coverage"], "conventions": len(conventions)}
    return result


# --------------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description="Generation-side evidence map + tri-state fact check")
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--map-out", default=None)
    args = ap.parse_args()
    run_folder = Path(args.run_folder)
    report_path = Path(args.report) if args.report else run_folder / "report.md"
    text = read_text(report_path)
    facts = build_facts(run_folder)
    result = check_report(text, facts, report_ref=str(report_path))
    binding = bind_report(text, facts)
    print("report %s | contrasts %d | passed %d | contradictions %d | unverified %d | verdict %s"
          % (result["report_sha12"], len(result["contrasts"]), result["n_passed"],
             result["n_contradictions"], result["n_unverified"], result["verdict"]))
    print("binding: claims %d | bound %d | pending review %d"
          % (binding["coverage"]["clauses_with_claim"], binding["coverage"]["claims_bound"],
             binding["coverage"]["claims_unbound"]))
    for item in result["contradictions"][:10]:
        print("  CONTRADICTION [%s] %s" % (item["check"],
                                           {k: v for k, v in item.items() if k not in {"check", "state", "clause"}}))
        print("      ...", item.get("clause", ""))
    for item in result["unverified"][:6]:
        print("  unverified [%s] %s" % (item["check"], item.get("reason", "")))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.map_out:
        Path(args.map_out).write_text(json.dumps({
            "schema": SCHEMA_VERSION, "run_folder": str(run_folder), "report": str(report_path),
            "report_sha12": result["report_sha12"], "conventions": facts["story"]["conventions"],
            "records": facts["records"], "bindings": binding["bindings"], "coverage": binding["coverage"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["n_contradictions"]:
        return 1
    if result["n_unverified"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
