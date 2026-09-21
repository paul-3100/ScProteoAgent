import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


GROUP_PRIORITY = [
    "MembraneStatus",
    "Cluster",
    "Group",
    "Condition",
    "Treatment",
    "Type1",
    "Type",
    "CellType",
    "cellType",
    "State",
    "SampleGroup",
    "Phenotype",
]

BATCH_PRIORITY = ["Batch", "batch", "Date", "Run", "MS_Run", "Plate", "TMT_Set"]
PAIR_PRIORITY = ["Donor", "donor", "Patient", "patient", "Subject", "subject", "Replicate", "BioReplicate"]
SAMPLE_COLUMNS = ["FileName", "R.FileName", "Sample", "SampleID", "sample", "sample_id"]
PROTEIN_COLUMNS = ["PG.ProteinGroups", "Protein.Group", "ProteinGroups", "ProteinID", "protein"]
GENE_COLUMNS = ["PG.Genes", "Genes", "Gene", "gene"]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _safe_yaml_dump(data: Dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)


def _safe_yaml_load(path: str | os.PathLike[str]) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}


def write_design_file(design: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(_safe_yaml_dump(design), encoding="utf-8")


def load_design_file(path: str | os.PathLike[str]) -> Dict[str, Any]:
    design = _safe_yaml_load(path)
    design.setdefault("schema_version", 1)
    design.setdefault("source", "user_provided")
    return design


def _first_existing(columns: List[str], priority: List[str]) -> Optional[str]:
    for col in priority:
        if col in columns:
            return col
    return None


def _sample_column(sampleinfo: pd.DataFrame) -> str:
    found = _first_existing(list(sampleinfo.columns), SAMPLE_COLUMNS)
    return found or str(sampleinfo.columns[0])


def _protein_columns(protein_quant: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(protein_quant.columns)
    protein_col = _first_existing(cols, PROTEIN_COLUMNS)
    gene_col = _first_existing(cols, GENE_COLUMNS)
    return {"protein_id_col": protein_col, "gene_col": gene_col}


def _usable_group_column(sampleinfo: pd.DataFrame) -> Optional[str]:
    excluded = set(SAMPLE_COLUMNS + BATCH_PRIORITY + PAIR_PRIORITY)
    for col in ["MembraneStatus", "Membrane_Status", "Membrane status", "Type1", "Type"]:
        if col not in sampleinfo.columns:
            continue
        values = sampleinfo[col].dropna().astype(str).str.strip()
        normalized = {v.lower() for v in values if v}
        if {"intact", "permeable"}.issubset(normalized):
            return col

    candidates = GROUP_PRIORITY + [c for c in sampleinfo.columns if c not in GROUP_PRIORITY and c not in excluded]
    for col in candidates:
        if col not in sampleinfo.columns:
            continue
        values = sampleinfo[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        n_unique = int(values.nunique())
        if 2 <= n_unique <= 30 and n_unique < len(values):
            return col
    return None


def _all_contrasts(sampleinfo: pd.DataFrame, group_col: Optional[str]) -> List[Dict[str, str]]:
    if not group_col or group_col not in sampleinfo.columns:
        return []
    levels = sorted(sampleinfo[group_col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist())
    return [
        {"name": f"{_safe_name(a)}_vs_{_safe_name(b)}", "group_a": str(a), "group_b": str(b)}
        for a, b in combinations(levels, 2)
    ]


def _safe_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return name or "group"


def _matrix_state(dataset_dir: Path, protein_quant: pd.DataFrame) -> Dict[str, Any]:
    docs = "\n".join(_read_text(p) for p in dataset_dir.glob("*.md")) + "\n" + _read_text(dataset_dir / "user_input.txt")
    docs_lower = docs.lower()
    values = protein_quant.select_dtypes(include=[np.number])
    numeric_values = values.to_numpy(dtype=float) if not values.empty else np.asarray([])
    finite = numeric_values[np.isfinite(numeric_values)] if numeric_values.size else np.asarray([])
    non_missing_fraction = float(np.isfinite(numeric_values).mean()) if numeric_values.size else 0.0
    max_value = float(np.nanmax(finite)) if finite.size else None
    min_value = float(np.nanmin(finite)) if finite.size else None

    already_processed = any(
        token in docs_lower
        for token in ["already processed", "已经过处理", "已归一化", "batch-corrected", "batch corrected", "无需 combat", "不需要 combat"]
    )
    batch_corrected = any(token in docs_lower for token in ["batch-corrected", "batch corrected", "limma batch", "combat", "批次校正"])
    needs_batch = any(token in docs_lower for token in ["未经过处理", "需 combat", "需要 combat", "raw", "unprocessed"])
    looks_logged = bool(finite.size and max_value is not None and max_value < 50 and min_value is not None and min_value > -50)

    state = "raw"
    if already_processed and batch_corrected:
        state = "batch_corrected"
    elif already_processed:
        state = "normalized"
    elif looks_logged:
        state = "normalized"

    return {
        "matrix_state": state,
        "already_processed": bool(already_processed),
        "batch_corrected_claimed": bool(batch_corrected),
        "needs_batch_correction_claimed": bool(needs_batch),
        "looks_logged": bool(looks_logged),
        "numeric_min": min_value,
        "numeric_max": max_value,
        "non_missing_fraction": round(non_missing_fraction, 6),
        "needs_log_transform": bool(not looks_logged),
        "needs_batch_correction": bool(needs_batch and state != "batch_corrected"),
    }


def infer_analysis_design(input_folder: str, sampleinfo_path: str, protein_quant_path: str) -> Dict[str, Any]:
    dataset_dir = Path(input_folder)
    sampleinfo = pd.read_csv(sampleinfo_path)
    protein_quant = pd.read_csv(protein_quant_path, nrows=250)
    sample_col = _sample_column(sampleinfo)
    group_col = _usable_group_column(sampleinfo)
    batch_col = _first_existing(list(sampleinfo.columns), BATCH_PRIORITY)
    pair_col = _first_existing(list(sampleinfo.columns), PAIR_PRIORITY)
    covariates = [
        col for col in sampleinfo.columns
        if col not in {sample_col, group_col, batch_col, pair_col}
        and 2 <= sampleinfo[col].dropna().nunique() < len(sampleinfo)
        and sampleinfo[col].dropna().nunique() <= 20
    ][:5]
    species = "mouse" if re.search(r"\b(mouse|murine|Mus musculus)\b", _read_text(dataset_dir / "user_input.txt"), re.I) else "human"
    matrix = _matrix_state(dataset_dir, protein_quant)
    # R28: the design step reads a head sample of the matrix for speed, and a detection-sorted file
    # makes that sample far denser than the delivered matrix (here 0.909 on the first 250 rows against
    # 0.313 over all rows). The sampled value is kept under its own key and the reported
    # non_missing_fraction is taken from the whole matrix, so a reader-facing missing rate no longer
    # describes the first rows of the file.
    matrix["non_missing_fraction_sampled"] = matrix.get("non_missing_fraction")
    matrix["non_missing_fraction_sampled_rows"] = 250
    try:
        _full = pd.read_csv(protein_quant_path)
        _full_values = _full.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if _full_values.size:
            matrix["non_missing_fraction"] = round(float(np.isfinite(_full_values).mean()), 6)
            matrix["non_missing_fraction_rows"] = int(_full.shape[0])
            matrix["non_missing_fraction_source"] = "full matrix (all rows, all numeric columns)"
        else:
            matrix["non_missing_fraction_source"] = "head sample only (matrix carried no numeric columns)"
    except Exception:  # noqa: BLE001
        matrix["non_missing_fraction_source"] = "head sample only (full read failed)"
    protein_cols = _protein_columns(protein_quant)

    design = {
        "schema_version": 1,
        "source": "auto_inferred",
        "input_folder": str(dataset_dir),
        "sample_id_col": sample_col,
        "protein_id_col": protein_cols["protein_id_col"],
        "gene_col": protein_cols["gene_col"],
        "group_col": group_col,
        "batch_col": batch_col,
        "pair_col": pair_col,
        "covariates": covariates,
        "species": species,
        "matrix": matrix,
        "differential": {
            "p_thresh": 0.05,
            "logfc_thresh": 0.25,
            "fdr_method": "BH",
            "contrasts": _all_contrasts(sampleinfo, group_col),
        },
        "confidence_policy": {
            "auto_inferred_design_max_confidence": "moderate",
            "python_fallback_max_confidence": "moderate",
            "external_knowledge_cannot_raise_confidence": True,
        },
        "warnings": [],
    }
    if design["source"] == "auto_inferred":
        design["warnings"].append("Analysis design was inferred from SampleInfo and dataset notes; review before publication.")
    if not group_col:
        design["warnings"].append("No usable primary group column was inferred.")
    if matrix["matrix_state"] == "batch_corrected":
        design["warnings"].append("Input appears already processed or batch-corrected; repeated batch correction should be skipped unless explicitly requested.")
    return design


def prepare_analysis_design(
    input_folder: str,
    sampleinfo_path: str,
    protein_quant_path: str,
    run_folder: str,
    explicit_design_path: Optional[str] = None,
) -> Dict[str, Any]:
    inferred = infer_analysis_design(input_folder, sampleinfo_path, protein_quant_path)
    inferred_path = Path(run_folder) / "analysis_design.inferred.yaml"
    write_design_file(inferred, inferred_path)

    dataset_design_path = Path(input_folder) / "analysis_design.yaml"
    if explicit_design_path:
        used = load_design_file(explicit_design_path)
        used["source"] = "cli_provided"
        used["source_path"] = str(explicit_design_path)
    elif dataset_design_path.exists():
        used = load_design_file(dataset_design_path)
        used["source"] = "dataset_provided"
        used["source_path"] = str(dataset_design_path)
    else:
        used = inferred

    used.setdefault("schema_version", 1)
    declared_input_folder = str(used.get("input_folder", "")).strip()
    runtime_input_folder = str(Path(input_folder).resolve())
    if declared_input_folder and declared_input_folder not in {".", runtime_input_folder}:
        used.setdefault("warnings", []).append(
            "analysis_design input_folder was normalized to the current runtime dataset directory for portability."
        )
    used["input_folder"] = runtime_input_folder
    used.setdefault("warnings", [])
    used_path = Path(run_folder) / "analysis_design.used.yaml"
    write_design_file(used, used_path)
    return {
        "analysis_design": used,
        "analysis_design_path": str(used_path),
        "analysis_design_inferred_path": str(inferred_path),
        "analysis_design_source": used.get("source", "unknown"),
    }
