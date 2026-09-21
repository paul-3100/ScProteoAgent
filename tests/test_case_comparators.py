"""Synthetic positive and negative cases for ``reproduce/cases/table_compare.py``.

Every check writes a tiny synthetic frozen table into an isolated temporary directory and compares
it with an in-memory frame. No scientific data is read and no case script is imported or executed,
so this file can run anywhere.

Each check states why it must fail. A check that merely returns ``False`` for the wrong reason
(for example an unexpected column mismatch) is reported as a failure of this test.

Run: ``python tests/test_case_comparators.py`` (exit code 0 = all checks behaved as stated).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce" / "cases"))

from table_compare import Field, Ledger, TableSpec, compare_table  # noqa: E402


def write_frozen(path: Path, columns, rows, separator: str = "\t") -> None:
    lines = [separator.join(columns)]
    for row in rows:
        lines.append(separator.join(str(row[c]) for c in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def frame_of(columns, rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def run_case(name, expect_ok, columns, frozen_rows, frame, spec, predicate, frozen_columns=None):
    """Compare one synthetic pair and report whether the verdict and the reason are as stated."""
    with tempfile.TemporaryDirectory(prefix="comparators_") as tmp:
        path = Path(tmp) / "frozen_table.tsv"
        write_frozen(path, columns, frozen_rows)
        ok, detail = compare_table(path, frame, spec, frozen_columns)
    reason = bool(predicate(detail))
    verdict = "PASS" if ok else "FAIL"
    good = (ok == expect_ok) and reason
    print("%-4s %-52s expected %-4s got %-4s  %s"
          % ("ok" if good else "FAIL", name, "PASS" if expect_ok else "FAIL", verdict,
             "reason confirmed" if reason else "reason NOT confirmed"))
    return good


def main() -> int:
    failures = 0
    two_col = TableSpec(key_fields=("protein",),
                        fields=(Field("protein", "identity"),
                                Field("value", "float", format_decimals=2)))
    raw_float = TableSpec(key_fields=("protein",),
                          fields=(Field("protein", "identity"), Field("value", "float")))
    count = TableSpec(key_fields=("protein",),
                      fields=(Field("protein", "identity"), Field("n", "int")))
    four_dec = TableSpec(key_fields=("protein",),
                         fields=(Field("protein", "identity"),
                                 Field("value", "float", format_decimals=4)))
    identity_extra = TableSpec(key_fields=("protein",),
                               fields=(Field("protein", "identity"),
                                       Field("gene", "identity"),
                                       Field("value", "float", format_decimals=2)))

    # ---------------------------------------------------------------- negative cases
    failures += not run_case(
        "finite frozen value vs missing recomputed value", False, ["protein", "value"],
        [{"protein": "A", "value": "0.5"}], frame_of(["protein", "value"], [["A", None]]),
        two_col, lambda d: d["missing_mismatches"] == 1)

    failures += not run_case(
        "missing frozen value vs finite recomputed value", False, ["protein", "value"],
        [{"protein": "A", "value": ""}], frame_of(["protein", "value"], [["A", 0.5]]),
        two_col, lambda d: d["missing_mismatches"] == 1)

    failures += not run_case(
        "finite frozen value vs infinite recomputed value", False, ["protein", "value"],
        [{"protein": "A", "value": "0.5"}],
        frame_of(["protein", "value"], [["A", float("inf")]]),
        raw_float, lambda d: d["missing_mismatches"] == 1)

    failures += not run_case(
        "count 3 vs count 3.1", False, ["protein", "n"],
        [{"protein": "A", "n": "3"}], frame_of(["protein", "n"], [["A", 3.1]]),
        count, lambda d: d["numeric_mismatches"] == 1)

    failures += not run_case(
        "float 0.0 vs 0.1 without a declared format", False, ["protein", "value"],
        [{"protein": "A", "value": "0"}], frame_of(["protein", "value"], [["A", 0.1]]),
        raw_float, lambda d: d["numeric_mismatches"] == 1
                             and d["worst_relative_to_tolerance"] > 1.0)

    failures += not run_case(
        "p value 1e-52 vs 0", False, ["protein", "p"],
        [{"protein": "A", "p": "1e-52"}], frame_of(["protein", "p"], [["A", 0.0]]),
        TableSpec(key_fields=("protein",),
                  fields=(Field("protein", "identity"), Field("p", "pvalue"))),
        lambda d: d["numeric_mismatches"] == 1)

    failures += not run_case(
        "missing row in the recomputed table", False, ["protein", "value"],
        [{"protein": "A", "value": "1.00"}, {"protein": "B", "value": "2.00"}],
        frame_of(["protein", "value"], [["A", 1.0]]), two_col,
        lambda d: d["keys_missing"] == ["('B',)"])

    failures += not run_case(
        "extra row in the recomputed table", False, ["protein", "value"],
        [{"protein": "A", "value": "1.00"}],
        frame_of(["protein", "value"], [["A", 1.0], ["B", 2.0]]), two_col,
        lambda d: d["keys_extra"] == ["('B',)"])

    failures += not run_case(
        "duplicate key in the recomputed table", False, ["protein", "value"],
        [{"protein": "A", "value": "1.00"}],
        frame_of(["protein", "value"], [["A", 1.0], ["A", 1.0]]), two_col,
        lambda d: d["keys_duplicate_recomputed"] == ["('A',)"])

    failures += not run_case(
        "duplicate key in the frozen table", False, ["protein", "value"],
        [{"protein": "A", "value": "1.00"}, {"protein": "A", "value": "1.00"}],
        frame_of(["protein", "value"], [["A", 1.0], ["A", 1.0]]), two_col,
        lambda d: d["keys_duplicate_frozen"] == ["('A',)"])

    failures += not run_case(
        "string identity difference in a non-key column", False, ["protein", "gene", "value"],
        [{"protein": "P05164", "gene": "MPO", "value": "1.00"}],
        frame_of(["protein", "gene", "value"], [["P05164", "MPO1", 1.0]]), identity_extra,
        lambda d: d["identity_mismatches"] == 1)

    failures += not run_case(
        "required column missing from the recomputed frame", False, ["protein", "value"],
        [{"protein": "A", "value": "1.00"}], frame_of(["protein"], [["A"]]),
        two_col, lambda d: d["columns_missing_in_frame"] == ["value"])

    failures += not run_case(
        "undeclared precision (frozen stores 6 decimals, schema says 4)", False,
        ["protein", "value"], [{"protein": "A", "value": "0.123456"}],
        frame_of(["protein", "value"], [["A", 0.123456]]), four_dec,
        lambda d: bool(d["declaration_errors"])
                  and d["declaration_errors"][0].startswith("value declares 4 decimals"))

    failures += not run_case(
        "identity value empty on both sides while missing is not allowed", False,
        ["protein", "value"], [{"protein": "", "value": "1.00"}],
        frame_of(["protein", "value"], [["", 1.0]]), two_col,
        lambda d: d["missing_mismatches"] == 1)

    failures += not run_case(
        "uncovered frozen column (neither compared nor declared excluded)", False,
        ["protein", "value", "note"],
        [{"protein": "A", "value": "1.00", "note": "x"}],
        frame_of(["protein", "value"], [["A", 1.0]]), two_col,
        lambda d: d["columns_uncovered"] == ["note"])

    # ---------------------------------------------------------------- positive cases
    failures += not run_case(
        "identical integers (3.0 equals the count 3)", True, ["protein", "n"],
        [{"protein": "A", "n": "3"}], frame_of(["protein", "n"], [["A", 3.0]]), count,
        lambda d: d["numeric_mismatches"] == 0)

    failures += not run_case(
        "missing on both sides where the field allows it", True, ["protein", "value"],
        [{"protein": "A", "value": ""}], frame_of(["protein", "value"], [["A", None]]),
        TableSpec(key_fields=("protein",),
                  fields=(Field("protein", "identity"),
                          Field("value", "float", allows_missing=True))),
        lambda d: d["per_field"][1]["missing_both"] == 1 and d["missing_mismatches"] == 0)

    failures += not run_case(
        "continuous error inside the pre-registered relative tolerance", True,
        ["protein", "value"], [{"protein": "A", "value": "12.3456789"}],
        frame_of(["protein", "value"], [["A", 12.3456789 * (1 + 5e-8)]]), raw_float,
        lambda d: d["numeric_mismatches"] == 0 and d["worst_relative_to_tolerance"] <= 1.0)

    failures += not run_case(
        "declared rounding: 0.123532 against a 4-decimal field", True,
        ["protein", "value"], [{"protein": "A", "value": "0.1235"}],
        frame_of(["protein", "value"], [["A", 0.123532]]), four_dec,
        lambda d: d["numeric_mismatches"] == 0)

    failures += not run_case(
        "declared rounding still rejects a larger difference", False,
        ["protein", "value"], [{"protein": "A", "value": "0.1235"}],
        frame_of(["protein", "value"], [["A", 0.1237]]), four_dec,
        lambda d: d["numeric_mismatches"] == 1)

    failures += not run_case(
        "permuted rows with an identical key set", True, ["protein", "value"],
        [{"protein": "A", "value": "1.00"}, {"protein": "B", "value": "2.00"}],
        frame_of(["protein", "value"], [["B", 2.0], ["A", 1.0]]), two_col,
        lambda d: d["row_order_matched"] is False and d["identity_mismatches"] == 0
                  and d["numeric_mismatches"] == 0)

    failures += not run_case(
        "three permuted rows are matched by key, not by position", True, ["protein", "value"],
        [{"protein": "A", "value": "1.00"}, {"protein": "B", "value": "2.00"},
         {"protein": "C", "value": "3.00"}],
        frame_of(["protein", "value"], [["C", 3.0], ["A", 1.0], ["B", 2.0]]), two_col,
        lambda d: d["row_order_matched"] is False and d["identity_mismatches"] == 0
                  and d["numeric_mismatches"] == 0)

    failures += not run_case(
        "identity text equal after whitespace normalisation", True, ["protein", "value"],
        [{"protein": " P05164 ", "value": "1.00"}],
        frame_of(["protein", "value"], [["P05164", 1.0]]), two_col,
        lambda d: d["identity_mismatches"] == 0)

    failures += not run_case(
        "declared exclusion covers the frozen column", True, ["protein", "value", "note"],
        [{"protein": "A", "value": "1.00", "note": "regenerated"}],
        frame_of(["protein", "value"], [["A", 1.0]]),
        TableSpec(key_fields=("protein",),
                  fields=(Field("protein", "identity"),
                          Field("value", "float", format_decimals=2)),
                  excluded=(("note", "prose column regenerated on every run"),)),
        lambda d: d["columns_equal"] and d["columns_uncovered"] == [])

    # a declared row filter covers one kind of row of a file that holds two
    with tempfile.TemporaryDirectory(prefix="comparators_") as tmp:
        path = Path(tmp) / "frozen_table.tsv"
        write_frozen(path, ["row_kind", "protein", "value"],
                     [{"row_kind": "cell", "protein": "A", "value": "1.00"},
                      {"row_kind": "summary", "protein": "A", "value": "9.00"}])
        spec = TableSpec(key_fields=("protein",),
                         fields=(Field("row_kind", "identity"), Field("protein", "identity"),
                                 Field("value", "float", format_decimals=2)))
        ok, detail = compare_table(path, frame_of(["row_kind", "protein", "value"],
                                                  [["cell", "A", 1.0]]), spec,
                                   row_filter=lambda table: table["row_kind"] == "cell")
        good = ok and detail["row_filter"] is True and detail["rows_after_filter"] == 1
        print("%-4s %-52s expected %-4s got %-4s  %s"
              % ("ok" if good else "FAIL", "declared row filter covers one row kind", "PASS",
                 "PASS" if ok else "FAIL", "reason confirmed" if good else "reason NOT confirmed"))
        failures += not good

    # ---------------------------------------------------------------- ledger behaviour
    ledger = Ledger("synthetic")
    ledger.add("c1", "passing check", "-", "-", "-", "PASS", "-", "-")
    ledger.add("c2", "blocked check", "-", "-", "-", "BLOCKED", "-", "-")
    ledger.add("c3", "not needed for the manuscript", "-", "-", "-", "OUT_OF_SCOPE", "-", "-",
               scope="not cited in the manuscript")
    good = (ledger.counts() == {"PASS": 1, "FAIL": 0, "BLOCKED": 1, "OUT_OF_SCOPE": 1}
            and ledger.exit_code() == 3)
    print("%-4s %-52s expected %-4s got %-4s  %s"
          % ("ok" if good else "FAIL", "ledger counts and partial exit code", "3",
             ledger.exit_code(), "reason confirmed" if good else "reason NOT confirmed"))
    failures += not good

    try:
        Ledger("synthetic").add("c4", "-", "-", "-", "-", "OUT_OF_SCOPE", "-", "-")
        print("FAIL %-52s expected ValueError got none" % "OUT_OF_SCOPE without a scope")
        failures += 1
    except ValueError:
        print("ok   %-52s expected ValueError got ValueError" % "OUT_OF_SCOPE without a scope")

    for name, build in (("a p/q field with an absolute tolerance",
                         lambda: Field("p", "pvalue", atol=1e-6)),
                        ("a p/q field with a declared rounding",
                         lambda: Field("p", "pvalue", format_decimals=4)),
                        ("an exact field with a declared rounding",
                         lambda: Field("n", "int", format_decimals=2))):
        try:
            build()
            print("FAIL %-52s expected ValueError got none" % name)
            failures += 1
        except ValueError:
            print("ok   %-52s expected ValueError got ValueError" % name)

    print("RESULT: %s" % ("PASS" if not failures else "FAIL (%d check(s))" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
