"""Field-level comparison of a recomputed table with its frozen counterpart.

This module replaces the earlier per-case ``compare_table`` helpers. Those helpers could pass a
comparison that should have failed:

* a one-sided missing value was turned into a zero difference by ``nan_to_num(..., nan=0.0)``;
* the tolerance was inferred from the parsed floats, so an integer column produced a half-unit
  tolerance (``3`` versus ``3.1`` passed);
* a zero expected value fell back to an absolute tolerance of ``1.0``.

Every comparison here is driven by a pre-registered :class:`TableSpec`: which columns are compared,
how each one is compared, which ones are deliberately excluded (with a reason), and from where a
declared rounding allowance comes. Nothing is inferred from the difference being compared.

Comparison rules

1. The key set of the frozen table and of the recomputed frame must be equal. Missing keys, extra
   keys and duplicate keys are reported separately and fail the comparison; rows are then matched
   by key, so a different row order is legal and recorded.
2. Missing values are compared as a mask first: a value present on one side and missing on the
   other always fails. A value missing on both sides passes only where the field declares
   ``allows_missing``. ``nan_to_num`` is never used.
3. ``inf`` and ``-inf`` fail in every numeric field.
4. Counts, sample sizes, family sizes, states and flags compare exactly. ``3.0`` equals the count
   ``3``; ``3.1`` does not, and no half-unit tolerance exists.
5. Continuous values use a pre-registered ``rtol`` (default ``1e-7``) and an explicit ``atol``
   (default ``0``). P and q values use a purely relative comparison with ``atol = 0``, so
   ``1e-52`` is not equal to ``0``.
6. A rounding allowance exists only where the field declares ``format_decimals``. The declaration
   is checked against the raw text of the frozen file: if a stored token carries more decimals
   than declared, the declaration itself fails. Values are never re-rounded to force agreement.

Usage:

    from table_compare import Field, Ledger, TableSpec

    SPEC = TableSpec(key_fields=("protein",), fields=(Field("protein", "identity"),
                     Field("log2FC", "float", format_decimals=6)),
                     excluded=(("note", "prose column regenerated on every run"),))
    ledger = Ledger("example")
    ledger.add_table("example_effects", "per-protein effects", frozen_path, frame, SPEC)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REL_TOL = 1e-7
MISSING_TOKENS = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>", "nat"})
IDENTITY_KINDS = ("identity", "text")
NUMERIC_KINDS = ("int", "flag", "float", "pvalue")
KINDS = IDENTITY_KINDS + NUMERIC_KINDS
SOURCE_CLASSES = ("RECOMPUTED", "COPIED_REFERENCE", "IDENTITY_CHECK", "NOT_VERIFIED")
RESULTS = ("PASS", "FAIL", "BLOCKED", "OUT_OF_SCOPE")


@dataclass(frozen=True)
class Field:
    """How one column is compared. ``note`` documents the declared tolerance."""

    name: str
    kind: str = "float"
    rtol: float = REL_TOL
    atol: float = 0.0
    allows_missing: bool = False
    format_decimals: int | None = None
    rounds_to: int | None = None
    source_class: str = "RECOMPUTED"
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError("unknown field kind %r for %s" % (self.kind, self.name))
        if self.source_class not in SOURCE_CLASSES:
            raise ValueError("unknown source class %r for %s" % (self.source_class, self.name))
        if self.kind in IDENTITY_KINDS and (self.format_decimals is not None
                                            or self.rounds_to is not None):
            raise ValueError("%s is an identity field and cannot carry a numeric allowance"
                             % self.name)
        if self.kind in ("int", "flag") and (self.format_decimals is not None
                                              or self.rounds_to is not None):
            raise ValueError("%s is exact and cannot carry a rounding allowance" % self.name)
        if self.atol and self.kind == "pvalue":
            raise ValueError("%s is a p/q field: an absolute tolerance would hide small values"
                             % self.name)
        if self.kind == "pvalue" and self.format_decimals is not None:
            raise ValueError("%s is a p/q field: a declared rounding would add an absolute floor "
                             "and break the purely relative comparison" % self.name)


@dataclass(frozen=True)
class TableSpec:
    """Pre-registered comparison schema for one frozen table."""

    key_fields: tuple
    fields: tuple
    excluded: tuple = ()
    label: str = ""

    def field_names(self) -> list:
        return [f.name for f in self.fields]

    def excluded_names(self) -> list:
        return [name for name, _reason in self.excluded]

    def field(self, name: str) -> Field:
        for item in self.fields:
            if item.name == name:
                return item
        raise KeyError(name)


def text_of(value) -> str:
    """Text form of a value, with pandas/numpy missing markers becoming the empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def is_missing_token(token: str) -> bool:
    return token.strip().lower() in MISSING_TOKENS


def decimals_of(token: str):
    """Effective decimals of a stored numeric token, or None when it is not a plain number."""
    text = token.strip().lower().lstrip("+")
    exponent = 0
    if "e" in text:
        mantissa, _, exponent_text = text.partition("e")
        try:
            exponent = int(exponent_text)
        except ValueError:
            return None
    else:
        mantissa = text
    if mantissa.startswith("-"):
        mantissa = mantissa[1:]
    if "." in mantissa:
        whole, _, fraction = mantissa.partition(".")
        if not (whole.isdigit() and (fraction.isdigit() or not fraction)):
            return None
        return len(fraction) - exponent
    if not mantissa.isdigit():
        return None
    return -exponent


def read_frozen(path: Path) -> pd.DataFrame:
    """Read the frozen table as raw text: missing markers and stored precision stay visible."""
    separator = "," if Path(path).suffix.lower() == ".csv" else "\t"
    frame = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False, na_filter=False)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def frozen_decimals(path: Path, column: str) -> int | None:
    """Largest number of decimals stored in one column of a frozen file (raw text)."""
    best = None
    for token in read_frozen(path)[column].tolist():
        if is_missing_token(str(token)):
            continue
        value = decimals_of(str(token))
        if value is None:
            continue
        best = value if best is None else max(best, value)
    return best


def byte_identical(frozen_path: Path, frame: pd.DataFrame, spec: TableSpec, separator=None) -> bool:
    """True when the compared frame re-serialises to the frozen bytes (reported, never assumed)."""
    columns = [c for c in read_frozen(frozen_path).columns if c not in spec.excluded_names()]
    if list(frame.columns) != columns:
        return False
    separator = separator or ("," if Path(frozen_path).suffix.lower() == ".csv" else "\t")
    got = frame.to_csv(index=False, sep=separator).replace("\r\n", "\n").encode("utf-8")
    want = Path(frozen_path).read_bytes().replace(b"\xef\xbb\xbf", b"").replace(b"\r\n", b"\n")
    return got == want


def _numeric_column(frame: pd.DataFrame, name: str):
    """Values and the provided mask of one recomputed column, without coercing silently."""
    column = frame[name]
    provided = ~pd.isna(column).to_numpy(dtype=bool)
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    unparsed = provided & np.isnan(values)
    return values, provided, unparsed


def compare_table(frozen_path: Path, frame: pd.DataFrame, spec: TableSpec,
                  frozen_columns=None, row_filter=None) -> tuple:
    """Compare ``frame`` with the frozen table at ``frozen_path`` field by field.

    ``row_filter`` selects the subset of the frozen rows that this comparison covers when one
    frozen file holds two kinds of row (for example per-cell values and the state summaries of the
    same file). The filter is recorded in the detail, and byte identity is not claimed for a
    filtered comparison.
    """
    frozen_path = Path(frozen_path)
    frozen = read_frozen(frozen_path)
    detail_filter = {"row_filter": row_filter is not None, "rows_before_filter": int(len(frozen))}
    if row_filter is not None:
        frozen = frozen[row_filter(frozen)].reset_index(drop=True)
        detail_filter["rows_after_filter"] = int(len(frozen))
    declared = list(frozen.columns) if frozen_columns is None else list(frozen_columns)
    excluded = spec.excluded_names()
    compared = spec.field_names() + [k for k in spec.key_fields if k not in spec.field_names()]

    detail = {
        "target": frozen_path.name,
        "label": spec.label,
        "keys": list(spec.key_fields),
        "rows_frozen": int(len(frozen)),
        "rows_recomputed": int(len(frame)),
        "columns_declared_vs_file": {"declared": declared, "in_file": list(frozen.columns)},
        "excluded": [{"column": name, "reason": reason} for name, reason in spec.excluded],
        "columns_uncovered": [],
        "columns_missing_in_frame": [],
        "columns_extra_in_frame": [],
        "keys_missing": [],
        "keys_extra": [],
        "keys_duplicate_frozen": [],
        "keys_duplicate_recomputed": [],
        "row_order_matched": True,
        "identity_mismatches": 0,
        "numeric_mismatches": 0,
        "missing_mismatches": 0,
        "tolerance_rejections": 0,
        "declaration_errors": [],
        "parse_errors": [],
        "worst_abs_diff": 0.0,
        "worst_relative_to_tolerance": 0.0,
        "worst_relative": 0.0,
        "per_field": [],
        "column_classes": {},
        "failed_cells": [],
        "byte_identical": None,
        "columns_equal": False,
        "tolerance_basis": "pre-registered rtol/atol per field; declared rounding only",
    }
    detail.update(detail_filter)

    if set(declared) != set(frozen.columns) | set(excluded):
        detail["columns_uncovered"] = sorted((set(declared) - set(frozen.columns)) |
                                             (set(frozen.columns) - set(declared) - set(excluded)))

    detail["columns_missing_in_frame"] = [c for c in compared if c not in frame.columns]
    detail["columns_extra_in_frame"] = [c for c in frame.columns
                                        if c not in compared and c not in excluded]
    detail["columns_uncovered"] = list(detail["columns_uncovered"]) + [
        c for c in frozen.columns if c not in compared and c not in excluded]
    detail["columns_equal"] = not (detail["columns_missing_in_frame"]
                                   or detail["columns_extra_in_frame"]
                                   or detail["columns_uncovered"])
    if not detail["columns_equal"]:
        detail["byte_identical"] = False
        return False, detail

    def keys_of(table: pd.DataFrame, text: bool) -> list:
        if text:
            columns = [table[k].tolist() for k in spec.key_fields]
            return [tuple(str(v).strip().lstrip("\ufeff") for v in row) for row in zip(*columns)]
        columns = [table[k].to_numpy(dtype=object) for k in spec.key_fields]
        return [tuple(text_of(v) for v in row) for row in zip(*columns)]

    frozen_keys = keys_of(frozen, True)
    recomputed_keys = keys_of(frame, False)
    for name, keys in (("frozen", frozen_keys), ("recomputed", recomputed_keys)):
        seen, duplicates = set(), []
        for key in keys:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        detail["keys_duplicate_%s" % name] = sorted({str(k) for k in duplicates})[:10]
    frozen_set, recomputed_set = set(frozen_keys), set(recomputed_keys)
    detail["keys_missing"] = sorted(str(k) for k in frozen_set - recomputed_set)[:10]
    detail["keys_extra"] = sorted(str(k) for k in recomputed_set - frozen_set)[:10]
    detail["row_order_matched"] = frozen_keys == recomputed_keys
    if (detail["keys_missing"] or detail["keys_extra"]
            or detail["keys_duplicate_frozen"] or detail["keys_duplicate_recomputed"]):
        detail["byte_identical"] = False
        return False, detail

    frame_order = {key: index for index, key in enumerate(recomputed_keys)}
    frame = frame.iloc[[frame_order[key] for key in frozen_keys]].reset_index(drop=True)
    frozen = frozen.reset_index(drop=True)

    for field in spec.fields:
        detail["column_classes"][field.name] = field.source_class
        entry = {"field": field.name, "kind": field.kind, "source_class": field.source_class,
                 "allows_missing": field.allows_missing, "rtol": field.rtol, "atol": field.atol,
                 "format_decimals": field.format_decimals, "rounds_to": field.rounds_to,
                 "note": field.note, "compared": 0, "missing_both": 0,
                 "mismatches": 0, "max_abs_diff": 0.0, "declaration": "checked"}
        left_raw = frozen[field.name].astype(str).tolist()

        if field.kind in IDENTITY_KINDS:
            left = np.array([token.strip() for token in left_raw], dtype=object)
            right = np.array([text_of(v) for v in frame[field.name].to_numpy(dtype=object)],
                             dtype=object)
            both_empty = (left == "") & (right == "")
            mismatched = np.where(left != right)[0]
            illegal_empty = np.where(both_empty & ~bool(field.allows_missing))[0]
            entry["compared"] = int(len(left))
            entry["missing_both"] = int(both_empty.sum())
            entry["mismatches"] = int(mismatched.size + illegal_empty.size)
            detail["identity_mismatches"] += int(mismatched.size)
            detail["missing_mismatches"] += int(illegal_empty.size)
            for index in list(mismatched[:5]) + list(illegal_empty[:5]):
                detail["failed_cells"].append("%s row %d (key %s): %r vs %r"
                                               % (field.name, int(index),
                                                  "/".join(str(k) for k in frozen_keys[index]),
                                                  left[index], right[index]))
            detail["per_field"].append(entry)
            continue

        count = len(left_raw)
        left_value = np.full(count, np.nan)
        left_provided = np.zeros(count, dtype=bool)
        for index, token in enumerate(left_raw):
            stripped = token.strip()
            if is_missing_token(stripped):
                continue
            try:
                left_value[index] = float(stripped)
            except ValueError:
                detail["parse_errors"].append("%s row %d: frozen token %r is not numeric"
                                               % (field.name, index, stripped))
                continue
            left_provided[index] = True

        right_value, right_provided_raw, unparsed = _numeric_column(frame, field.name)
        right_provided = right_provided_raw & ~unparsed
        for index in np.where(unparsed)[0][:5]:
            detail["parse_errors"].append("%s row %d: recomputed value %r is not numeric"
                                           % (field.name, int(index), frame[field.name].iloc[index]))

        if field.format_decimals is not None:
            over = []
            for index, token in enumerate(left_raw):
                stripped = token.strip()
                if is_missing_token(stripped):
                    continue
                stored = decimals_of(stripped)
                if stored is None:
                    continue
                if stored > int(field.format_decimals):
                    over.append((index, stripped, stored))
            if over:
                detail["declaration_errors"].append(
                    "%s declares %d decimals but the frozen file stores %s"
                    % (field.name, int(field.format_decimals),
                       "; ".join("%r (%d)" % (token, stored) for _i, token, stored in over[:3])))
                entry["declaration"] = "rejected"

        left_inf = left_provided & ~np.isfinite(left_value)
        right_inf = right_provided & ~np.isfinite(right_value)
        both = left_provided & right_provided
        left_only = left_provided & ~right_provided
        right_only = ~left_provided & right_provided
        both_missing = ~left_provided & ~right_provided
        illegal_missing = (left_only | right_only | (both_missing & ~bool(field.allows_missing))
                           | left_inf | right_inf)

        for index in np.where(left_only)[0][:5]:
            detail["failed_cells"].append("%s row %d (key %s): frozen %r vs recomputed missing"
                                           % (field.name, int(index),
                                              "/".join(str(k) for k in frozen_keys[index]),
                                              left_raw[index]))
        for index in np.where(right_only)[0][:5]:
            detail["failed_cells"].append("%s row %d (key %s): frozen missing vs recomputed %r"
                                           % (field.name, int(index),
                                              "/".join(str(k) for k in frozen_keys[index]),
                                              frame[field.name].iloc[index]))
        for index in np.where(left_inf | right_inf)[0][:5]:
            detail["failed_cells"].append("%s row %d: non-finite value (%r vs %r)"
                                           % (field.name, int(index), left_value[index],
                                              right_value[index]))

        difference = np.where(both, np.abs(left_value - right_value), 0.0)
        allowed = np.full(count, np.inf)
        if field.kind in ("int", "flag"):
            not_integral = both & ((np.abs(left_value - np.round(left_value)) > 0)
                                   | (np.abs(right_value - np.round(right_value)) > 0))
            allowed = np.where(both, 0.0, np.inf)
            over_tolerance = not_integral | (both & (left_value != right_value))
            for index in np.where(not_integral)[0][:5]:
                detail["failed_cells"].append("%s row %d: %r is not an integer (%r vs %r)"
                                               % (field.name, int(index), left_raw[index],
                                                  left_value[index], right_value[index]))
        elif field.rounds_to is not None:
            rounds = int(field.rounds_to)
            allowed = np.where(both, 0.5 * 10.0 ** (-rounds), np.inf)
            over_tolerance = both & (left_value != np.round(right_value, rounds))
            entry["rounded_cells"] = int(np.sum(~over_tolerance & both))
        else:
            allowed = np.where(both, field.atol + field.rtol * np.abs(left_value), np.inf)
            if field.format_decimals is not None:
                allowed = np.where(both, np.maximum(allowed,
                                                    0.5 * 10.0 ** (-int(field.format_decimals))),
                                   np.inf)
            over_tolerance = both & (difference > allowed)
        entry["compared"] = int(both.sum())
        entry["missing_both"] = int(both_missing.sum())
        entry["mismatches"] = int(over_tolerance.sum() + illegal_missing.sum())
        detail["numeric_mismatches"] += int(over_tolerance.sum())
        detail["missing_mismatches"] += int(illegal_missing.sum())
        detail["tolerance_rejections"] += int(over_tolerance.sum())
        if both.any():
            entry["max_abs_diff"] = float(difference[both].max())
            detail["worst_abs_diff"] = max(detail["worst_abs_diff"], entry["max_abs_diff"])
            with np.errstate(divide="ignore", invalid="ignore"):
                relative = difference[both] / np.maximum(np.abs(left_value[both]), 1e-300)
            if np.isfinite(relative).any():
                detail["worst_relative"] = max(detail["worst_relative"],
                                                float(np.nanmax(relative)))
            if np.isfinite(allowed[both]).any():
                with np.errstate(divide="ignore", invalid="ignore"):
                    positive = allowed[both] > 0
                    ratio = np.where(positive,
                                     difference[both] / np.where(positive, allowed[both], 1.0),
                                     np.where(difference[both] > 0, np.inf, 0.0))
                if ratio.size:
                    detail["worst_relative_to_tolerance"] = max(
                        detail["worst_relative_to_tolerance"], float(np.nanmax(ratio)))
        for index in np.where(over_tolerance)[0][:5]:
            detail["failed_cells"].append("%s row %d (key %s): %r vs %r (|d| %.6g > %.6g)"
                                           % (field.name, int(index),
                                              "/".join(str(k) for k in frozen_keys[index]),
                                              left_value[index], right_value[index],
                                              difference[index], allowed[index]))
        if field.kind in ("int", "flag"):
            entry["tolerance"] = "exact (integral)"
        elif field.rounds_to is not None:
            entry["tolerance"] = "exact after rounding the recomputed value to %d decimals" \
                                 % int(field.rounds_to)
        else:
            entry["tolerance"] = "rtol %.3g + atol %.3g%s" % (
                field.rtol, field.atol,
                " + 0.5e-%d declared rounding" % int(field.format_decimals)
                if field.format_decimals is not None else "")
        detail["per_field"].append(entry)

    detail["byte_identical"] = (False if row_filter is not None
                                else byte_identical(frozen_path, frame, spec))
    ok = bool(detail["identity_mismatches"] == 0 and detail["numeric_mismatches"] == 0
              and detail["missing_mismatches"] == 0 and not detail["parse_errors"]
              and not detail["declaration_errors"])
    return ok, detail


def summarise(detail: dict) -> str:
    """One-line human summary of a comparison detail block."""
    classes = {}
    for name, klass in (detail.get("column_classes") or {}).items():
        classes[klass] = classes.get(klass, 0) + 1
    class_text = ", ".join("%d %s" % (count, name) for name, count in sorted(classes.items()))
    return ("rows %d/%d, columns equal %s, keys %d missing/%d extra/%d duplicate, "
            "identity mismatches %d, numeric mismatches %d, missing mismatches %d, "
            "worst |d| %.6g, worst ratio to tolerance %.3g, classes %s"
            % (detail["rows_recomputed"], detail["rows_frozen"], detail["columns_equal"],
               len(detail["keys_missing"]), len(detail["keys_extra"]),
               len(detail["keys_duplicate_frozen"]) + len(detail["keys_duplicate_recomputed"]),
               detail["identity_mismatches"], detail["numeric_mismatches"],
               detail["missing_mismatches"], detail["worst_abs_diff"],
               detail["worst_relative_to_tolerance"], class_text or "none"))


class Ledger:
    """Collects the check ledger that becomes ``case_regression.json``.

    Result vocabulary: ``PASS`` (checked and equal), ``FAIL`` (checked and different),
    ``BLOCKED`` (a number or claim of the manuscript cannot be checked with what the archive
    ships) and ``OUT_OF_SCOPE`` (the check asks for something the manuscript does not require;
    ``scope`` must say why).

    Exit codes of a case script: ``0`` every check passed, ``1`` a check failed, ``2`` a required
    input is missing (nothing written), ``3`` partial - no failure but at least one BLOCKED check.
    A check that is OUT_OF_SCOPE does not on its own make the run partial; it is reported with its
    scope in the summary and in the ledger.
    """

    def __init__(self, theme: str = "") -> None:
        self.theme = theme
        self.checks = []

    def add(self, check_id, check, target, expected, observed, result, method, evidence, notes="",
            scope=""):
        if result not in RESULTS:
            raise ValueError("unknown result %r" % result)
        if result == "OUT_OF_SCOPE" and not scope:
            raise ValueError("OUT_OF_SCOPE requires a scope: %s" % check_id)
        entry = {"check_id": check_id, "check": check, "target": target, "expected": str(expected),
                 "observed": str(observed), "result": result, "method": method,
                 "evidence": evidence, "notes": notes}
        if scope:
            entry["scope"] = scope
        self.checks.append(entry)
        print("CHECK %-34s %-13s %s" % (check_id, result, observed))
        return entry

    def add_table(self, check_id, check, frozen_path, frame, spec, frozen_columns=None,
                  evidence="", notes="", method_extra="", row_filter=None):
        ok, detail = compare_table(Path(frozen_path), frame, spec, frozen_columns, row_filter)
        observed = summarise(detail)
        entry = self.add(check_id, check, str(getattr(frozen_path, "name", frozen_path)),
                         "identity exact; numeric at the pre-registered per-field tolerance",
                         observed, "PASS" if ok else "FAIL",
                         "key set equal + missing mask + per-field tolerance%s" % method_extra,
                         evidence, notes)
        entry["comparison_detail"] = detail
        entry["column_classes"] = detail["column_classes"]
        return ok, detail

    def counts(self) -> dict:
        return {name: sum(1 for c in self.checks if c["result"] == name) for name in RESULTS}

    def exit_code(self) -> int:
        counts = self.counts()
        if counts["FAIL"]:
            return 1
        if counts["BLOCKED"]:
            return 3
        return 0

    def class_rollup(self) -> dict:
        rollup = {}
        for check in self.checks:
            for _name, klass in (check.get("column_classes") or {}).items():
                rollup[klass] = rollup.get(klass, 0) + 1
        return rollup

    def summary_line(self) -> str:
        counts = self.counts()
        return ("SUMMARY PASS=%d FAIL=%d BLOCKED=%d OUT_OF_SCOPE=%d; columns %s"
                % (counts["PASS"], counts["FAIL"], counts["BLOCKED"], counts["OUT_OF_SCOPE"],
                   self.class_rollup() or "none"))
