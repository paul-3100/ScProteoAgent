"""Read a saved (frozen) result table and print a summary. No analysis is run.

This is the "inspect reported results" workflow: point it at a table that ships with the data
archive and read it as it is. The reader streams the file, treats empty/NA/NaN cells as missing,
and never rewrites the table.

Usage:
  python examples/saved_results/read_saved_table.py <table.tsv> [options]

Options:
  --filter COLUMN=VALUE     keep rows whose COLUMN equals VALUE (repeatable)
  --group-by COLUMN         print value counts for COLUMN
  --value COLUMN            summarise COLUMN numerically (repeatable, default: all numeric)
  --limit N                 stop after N rows (0 = all, default 0)
  --show-columns            print the column names and exit
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

MISSING = {"", "na", "nan", "n/a", "null", "none", "-inf", "inf"}


def read_rows(path: Path, limit: int):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit("FATAL: %s has no header" % path)
        for index, row in enumerate(reader):
            if limit and index >= limit:
                break
            yield row
        yield None  # sentinel so the header can be reported before the loop ends


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a saved result table without running analysis.")
    parser.add_argument("table", type=Path)
    parser.add_argument("--filter", action="append", default=[])
    parser.add_argument("--group-by", default=None)
    parser.add_argument("--value", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show-columns", action="store_true")
    args = parser.parse_args(argv)

    if not args.table.is_file():
        print("FATAL: no such table: %s" % args.table, file=sys.stderr)
        return 1
    filters = []
    for expression in args.filter:
        if "=" not in expression:
            print("FATAL: --filter expects COLUMN=VALUE, got %r" % expression, file=sys.stderr)
            return 1
        column, _, value = expression.partition("=")
        filters.append((column, value))

    with args.table.open(encoding="utf-8-sig", newline="") as handle:
        fieldnames = csv.DictReader(handle, delimiter="\t").fieldnames or []
    print("table: %s" % args.table)
    print("columns (%d): %s" % (len(fieldnames), ", ".join(fieldnames)))
    if args.show_columns:
        return 0

    counts: Counter = Counter()
    stats: dict[str, list[float]] = {name: [] for name in args.value}
    read = kept = 0
    numeric_candidates: dict[str, int] = {}
    for row in read_rows(args.table, args.limit):
        if row is None:
            break
        read += 1
        if any(str(row.get(column, "")) != value for column, value in filters):
            continue
        kept += 1
        if args.group_by:
            counts[str(row.get(args.group_by, ""))] += 1
        for name in (args.value or fieldnames):
            raw = str(row.get(name, "")).strip()
            if raw.lower() in MISSING or not name:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            numeric_candidates[name] = numeric_candidates.get(name, 0) + 1
            if name in stats:
                stats[name].append(value)
    print("rows read: %d   rows kept: %d" % (read, kept))
    if filters:
        print("filters: %s" % ", ".join("%s=%s" % f for f in filters))
    if args.value:
        for name in args.value:
            values = stats[name]
            if not values:
                print("  %s: no numeric values" % name)
                continue
            print("  %s: n=%d min=%.6g max=%.6g mean=%.6g"
                  % (name, len(values), min(values), max(values), sum(values) / len(values)))
    else:
        numeric = sorted(numeric_candidates.items(), key=lambda kv: -kv[1])[:8]
        if numeric:
            print("numeric columns (by non-missing count): %s"
                  % ", ".join("%s(%d)" % (name, count) for name, count in numeric))
    if args.group_by:
        print("group counts for %s (top 20):" % args.group_by)
        for value, count in counts.most_common(20):
            print("  %-32s %d" % (repr(value), count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
