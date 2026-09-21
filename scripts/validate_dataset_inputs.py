"""Offline input validator for ScProteoAgent dataset directories.

Checks one dataset folder (or every dataset folder directly below a parent folder):
  * required files: ProteinQuant.csv, SampleInfo.csv, user_input.txt
  * matrix orientation: proteins in rows, samples in columns
  * sample-ID key: SampleInfo ID column against the ProteinQuant sample columns, both directions
  * duplicate protein rows, duplicate sample IDs, missing-value markers, group/unit fields

Standard library only, no network access, no credentials. Nothing is written unless --json
is passed.

Exit codes:
  0  no validation errors (warnings may still be reported)
  2  validation errors found
  1  the validator could not complete (bad arguments or unreadable files)

Usage:
  py -3 scripts/validate_dataset_inputs.py <dataset_dir> [<dataset_dir> ...]
  py -3 scripts/validate_dataset_inputs.py --root <parent_dir_with_datasets>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_FILES = ("ProteinQuant.csv", "SampleInfo.csv", "user_input.txt")
MISSING_TOKENS = ("", "na", "nan", "n/a", "null", "none", "-inf", "+inf", "inf", "#n/a", "not found")
ID_COLUMN_NAMES = ("filename", "file_name", "sample", "sampleid", "sample_id", "samplename",
                   "sample_name", "run", "runs", "ms.run", "id", "cell_id", "cell")
UNIT_COLUMN_HINTS = ("donor", "mouse", "animal", "individual", "subject", "patient", "batch",
                     "replicate", "rep", "unit", "pool", "slide", "well", "plate", "channel")
GROUP_COLUMN_HINTS = ("cluster", "group", "condition", "type", "class", "label", "zone",
                      "treatment", "state", "subtype", "stain", "dataset", "region", "time")
MAX_SAMPLE_CANDIDATES = 4000


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be inspected at all."""


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a CSV file as (header, rows). Raises DatasetError for an empty file."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = [h.strip() for h in next(reader)]
            except StopIteration as exc:
                raise DatasetError("%s is empty" % path.name) from exc
            rows = [row for row in reader]
    except OSError as exc:
        raise DatasetError("cannot read %s: %s" % (path.name, exc)) from exc
    return header, rows


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS


def column_values(header: list[str], rows: list[list[str]], name: str) -> list[str]:
    index = header.index(name)
    return [(row[index] if index < len(row) else "") for row in rows]


def pick_id_column(header: list[str], values_by_column: dict[str, list[str]],
                   sample_ids: set[str]) -> tuple[str, int]:
    """Return the SampleInfo column whose values best identify the matrix sample columns."""
    best_name = header[0]
    best_overlap = -1
    for position, name in enumerate(header):
        unique = {v.strip() for v in values_by_column[name] if v.strip()}
        overlap = len(unique & sample_ids)
        preference = 0 if name.strip().lower() in ID_COLUMN_NAMES else 1
        if position == 0:
            preference = 0
        score = (overlap, -preference, -position)
        if overlap > best_overlap or (overlap == best_overlap and position == 0):
            best_name, best_overlap = name, overlap
        del score
    return best_name, max(best_overlap, 0)


def validate_dataset(path: Path) -> dict:
    report: dict = {"dataset": path.name, "path": str(path), "errors": [], "warnings": [],
                    "checks": {}, "files": {}}

    def error(message: str) -> None:
        report["errors"].append(message)

    def warn(message: str) -> None:
        report["warnings"].append(message)

    missing_files = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    report["checks"]["required_files_present"] = not missing_files
    if missing_files:
        error("missing required file(s): %s" % ", ".join(missing_files))
        return report

    matrix_header, matrix_rows = read_table(path / "ProteinQuant.csv")
    info_header, info_rows = read_table(path / "SampleInfo.csv")
    task_text = (path / "user_input.txt").read_text(encoding="utf-8", errors="replace")
    report["files"] = {
        "matrix_header_columns": len(matrix_header),
        "matrix_data_rows": len(matrix_rows),
        "metadata_columns": len(info_header),
        "metadata_rows": len(info_rows),
        "task_text_characters": len(task_text),
    }
    if len(matrix_header) < 2:
        error("ProteinQuant.csv has fewer than two columns; expected annotations plus samples")
        return report
    if not matrix_rows:
        error("ProteinQuant.csv has no data rows")
        return report
    if not info_rows:
        error("SampleInfo.csv has no data rows")
        return report
    if not task_text.strip():
        error("user_input.txt is empty")

    values_by_column = {name: column_values(info_header, info_rows, name) for name in info_header}
    matrix_columns = [c for c in matrix_header if c]
    matrix_column_set = set(matrix_columns)

    # --- orientation: sample IDs must be column names, not row labels ----------------
    first_column = [(row[0] if row else "").strip() for row in matrix_rows]
    info_id_guess = max(values_by_column.items(), key=lambda kv: len(set(kv[1]) & matrix_column_set))[0]
    info_ids = {v.strip() for v in values_by_column[info_id_guess] if v.strip()}
    row_overlap = len(set(first_column) & info_ids)
    col_overlap = len(info_ids & matrix_column_set)
    report["checks"]["orientation_proteins_in_rows"] = col_overlap >= row_overlap or row_overlap == 0
    if row_overlap > col_overlap and row_overlap > 0:
        error("matrix looks transposed: %d row labels match SampleInfo IDs but only %d column "
              "names do; ProteinQuant.csv must have proteins in rows and samples in columns"
              % (row_overlap, col_overlap))

    # --- sample-ID key ---------------------------------------------------------------
    id_column, id_overlap = pick_id_column(info_header, values_by_column, matrix_column_set)
    identified = {v.strip() for v in values_by_column[id_column] if v.strip()}
    samples_in_matrix = identified & matrix_column_set
    metadata_only = sorted(identified - matrix_column_set)
    matrix_only = sorted(c for c in matrix_columns if c not in identified)
    report["checks"].update({
        "id_column": id_column,
        "metadata_ids": len(identified),
        "matrix_columns": len(matrix_columns),
        "matched_ids": len(samples_in_matrix),
        "metadata_ids_absent_from_matrix": len(metadata_only),
        "matrix_columns_absent_from_metadata": len(matrix_only),
    })
    if id_overlap == 0:
        error("no SampleInfo value matches a ProteinQuant column name; the sample-ID key is "
              "broken (checked every SampleInfo column)")
    elif id_overlap < len(identified):
        error("%d of %d SampleInfo IDs have no matching ProteinQuant column (first missing: %s)"
              % (len(metadata_only), len(identified), ", ".join(metadata_only[:3])))
    if matrix_only:
        warn("matrix columns without a SampleInfo row are treated as annotation columns: %d "
             "(first: %s)" % (len(matrix_only), ", ".join(matrix_only[:3])))

    # --- duplicates ------------------------------------------------------------------
    duplicate_ids = sorted(name for name, count in Counter(values_by_column[id_column]).items()
                           if count > 1)
    duplicate_proteins = sorted(name for name, count in Counter(first_column).items() if count > 1)
    duplicate_columns = sorted(name for name, count in Counter(matrix_columns).items() if count > 1)
    report["checks"].update({
        "duplicate_metadata_ids": len(duplicate_ids),
        "duplicate_protein_ids": len(duplicate_proteins),
        "duplicate_matrix_column_names": len(duplicate_columns),
    })
    if duplicate_ids:
        error("duplicate sample IDs in SampleInfo.%s: %d (first: %s)"
              % (id_column, len(duplicate_ids), ", ".join(duplicate_ids[:3])))
    if duplicate_columns:
        error("duplicate column names in ProteinQuant.csv: %d (first: %s)"
              % (len(duplicate_columns), ", ".join(duplicate_columns[:3])))
    if duplicate_proteins:
        warn("duplicate protein identifiers in column 1 of ProteinQuant.csv: %d (first: %s)"
             % (len(duplicate_proteins), ", ".join(duplicate_proteins[:3])))
    if len(first_column) != len(set(first_column)) and not duplicate_proteins:
        warn("protein identifier column has trailing or embedded whitespace variants")

    # --- missing markers --------------------------------------------------------------
    sample_column_indexes = [i for i, name in enumerate(matrix_header) if name in samples_in_matrix]
    marker_counts: Counter = Counter()
    numeric_cells = 0
    non_numeric = 0
    for row in matrix_rows:
        for index in sample_column_indexes:
            value = row[index].strip() if index < len(row) else ""
            if is_missing(value):
                marker_counts[value if value else "<empty>"] += 1
            else:
                numeric_cells += 1
                try:
                    float(value)
                except ValueError:
                    non_numeric += 1
    total_cells = numeric_cells + sum(marker_counts.values())
    report["checks"].update({
        "sample_cells": total_cells,
        "missing_cells": sum(marker_counts.values()),
        "missing_fraction": round(sum(marker_counts.values()) / total_cells, 6) if total_cells else None,
        "missing_markers": dict(sorted(marker_counts.items())),
        "non_numeric_cells": non_numeric,
    })
    if non_numeric:
        warn("%d of %d quantified sample cells are neither a number nor a recognised missing "
             "marker" % (non_numeric, total_cells))
    if total_cells and sum(marker_counts.values()) == total_cells:
        error("every quantified sample cell is empty or a missing marker")

    # --- grouping and experimental-unit fields ---------------------------------------
    field_summary = []
    for name in info_header:
        values = [v.strip() for v in values_by_column[name]]
        unique = sorted({v for v in values if v})
        role = "id" if name == id_column else (
            "unit" if any(hint in name.lower() for hint in UNIT_COLUMN_HINTS) else (
                "group" if any(hint in name.lower() for hint in GROUP_COLUMN_HINTS) else "other"))
        field_summary.append({"column": name, "kind": role, "distinct_values": len(unique),
                              "examples": unique[:3], "missing": sum(1 for v in values if is_missing(v))})
    report["metadata_fields"] = field_summary
    grouping = [f for f in field_summary if f["kind"] == "group" and f["distinct_values"] > 0]
    units = [f for f in field_summary if f["kind"] == "unit" and f["distinct_values"] > 0]
    report["checks"]["grouping_fields"] = [f["column"] for f in grouping]
    report["checks"]["unit_fields"] = [f["column"] for f in units]
    if not grouping:
        warn("no grouping field recognised in SampleInfo.csv; a comparison must name the "
             "grouping column explicitly")
    for field in field_summary:
        if field["kind"] != "id" and field["distinct_values"] == 0:
            warn("SampleInfo column %r has no values" % field["column"])
        if field["kind"] != "id" and 0 < field["missing"] == len(info_rows):
            warn("SampleInfo column %r is missing for every row" % field["column"])
    if units:
        report["unit_summary"] = {
            name: len({v for v in values_by_column[name] if v})
            for name in report["checks"]["unit_fields"]}

    return report


def render(report: dict) -> None:
    print("=" * 72)
    print("dataset: %s" % report["dataset"])
    checks = report.get("checks", {})
    if report.get("files"):
        files = report["files"]
        print("  matrix: %d protein rows x %d columns (%d sample columns matched)"
              % (files["matrix_data_rows"], files["matrix_header_columns"],
                 checks.get("matched_ids", 0)))
        print("  metadata: %d rows x %d columns (ID column: %s)"
              % (files["metadata_rows"], files["metadata_columns"], checks.get("id_column", "?")))
        print("  missing: %s of %s sample cells (%.4f%%) markers=%s"
              % (checks.get("missing_cells"), checks.get("sample_cells"),
                 100.0 * (checks.get("missing_fraction") or 0.0), checks.get("missing_markers")))
        print("  grouping fields: %s" % ", ".join(checks.get("grouping_fields") or ["none"]))
        print("  experimental-unit candidates: %s"
              % ", ".join(report.get("unit_summary", {}) and
                          "%s(%d)" % (k, v) for k, v in report.get("unit_summary", {}).items()) if
              report.get("unit_summary") else "  experimental-unit candidates: none")
    for message in report["warnings"]:
        print("  WARN  %s" % message)
    for message in report["errors"]:
        print("  ERROR %s" % message)
    print("  result: %s" % ("FAIL" if report["errors"] else "PASS"))


def collect_targets(raw_paths: list[str]) -> tuple[list[Path], list[str]]:
    targets: list[Path] = []
    problems: list[str] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_dir():
            problems.append("not a directory: %s" % path)
            continue
        if all((path / name).is_file() for name in REQUIRED_FILES):
            targets.append(path)
            continue
        if any((path / name).is_file() for name in REQUIRED_FILES):
            # looks like a dataset that is missing part of its input: validate and report
            targets.append(path)
            continue
        nested = [child for child in sorted(path.iterdir())
                  if child.is_dir() and all((child / name).is_file() for name in REQUIRED_FILES)]
        if nested:
            targets.extend(nested)
        else:
            problems.append("no dataset with %s inside %s" % (", ".join(REQUIRED_FILES), path))
    return targets, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ScProteoAgent dataset inputs offline.")
    parser.add_argument("datasets", nargs="*", help="dataset directories, or parents of them")
    parser.add_argument("--root", type=Path, default=None,
                        help="parent directory whose immediate subdirectories are datasets")
    parser.add_argument("--json", type=Path, default=None, help="also write the summary as JSON")
    args = parser.parse_args(argv)

    raw_paths = list(args.datasets)
    if args.root is not None:
        raw_paths.append(str(args.root))
    if not raw_paths:
        parser.error("give at least one dataset directory or --root")

    targets, problems = collect_targets(raw_paths)
    for problem in problems:
        print("ERROR: %s" % problem, file=sys.stderr)
    if problems:
        return 1
    if len(targets) > MAX_SAMPLE_CANDIDATES:
        print("ERROR: refusing to validate %d datasets in one call" % len(targets), file=sys.stderr)
        return 1

    reports = []
    for path in targets:
        try:
            reports.append(validate_dataset(path))
        except DatasetError as exc:
            print("ERROR: %s" % exc, file=sys.stderr)
            return 1
    for report in reports:
        render(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"datasets": reports}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print("json written: %s" % args.json)
    errors = sum(len(r["errors"]) for r in reports)
    warnings = sum(len(r["warnings"]) for r in reports)
    print("SUMMARY: datasets=%d errors=%d warnings=%d" % (len(reports), errors, warnings))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
