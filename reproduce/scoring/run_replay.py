"""Offline replay driver for the frozen HuRunWen rule scorer (score_full_run.py).

The frozen scorer is executed unchanged, one pass per system, through guarded_launcher.py,
which blocks every socket operation. The LLM review layer is disabled: this driver
reproduces the rule layer only (rule_total and the six dimensions), which is the layer the
cross-system comparison uses.

What the driver does
====================
  * discovers one run root per system: a directory whose children are run directories that
    carry run_status.json (normally <scoring-root>/scoring/runs, assembled by
    reproduce/scoring/prepare_scoring_inputs.py);
  * calls the frozen scorer with --skip-llm for each system, in a subprocess whose credential
    environment variables are cleared;
  * merges the per-system score.json files into one summary keyed by (study, system);
  * compares that summary against a frozen rule-score table.

Completeness comes before numbers
=================================
A full-grid PASS requires all of the following:
  * the frozen table contributes exactly 88 unique (study, system) cells, none of them
    repeated, and every cell carries a finite rule_total and six finite dimensions; blank
    cells, empty strings, NaN, Infinity and non-numeric text are rejected;
  * the replay produced exactly the same 88 unique cells, each with
    scoring_mode == rule_only_fallback, i.e. the frozen rule layer. The historical
    language-model adjustment is never mixed into this comparison;
  * no cell differs by more than --tolerance, which defaults to 0.0 and is never widened
    automatically.
Zero rows, a short table, a repeated key, a missing dimension, an unknown method name or an
unmapped alias, and a partial grid all end in exit code 2 with an explicit failure reason.
Every numeric difference is reported as key / field / expected / observed.

An explicit subset (--datasets and/or --system) compares the frozen table projected onto the
requested cells and reports SUBSET_PASS with the coverage, so a partial replay can never be
presented as the 88-cell result.

Reported method names are converted to archived system directory names through the explicit
REPORTED_METHOD_ALIASES table only. A method name that is neither a replayed system nor a
listed alias is reported instead of being guessed at.

Usage:
  py -3 reproduce/scoring/run_replay.py --runs-root <scoring-root>/scoring/runs --examples-root <scoring-root>/examples --out-dir <out-dir> --expected-tsv <archive>/scores/HW_RULE_SCORES.tsv
  py -3 reproduce/scoring/run_replay.py ... --datasets Nat_Commun_PiSPA_2024 Science_BloodCell_2025
  py -3 reproduce/scoring/run_replay.py ... --check-report-sha256

Exit codes: 0 = full-grid PASS or subset SUBSET_PASS, 1 = usage/environment error,
            2 = FAIL or SUBSET_FAIL, including a replay that produced zero rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SCORING_DIR = Path(__file__).resolve().parent
FROZEN_SCORER = SCORING_DIR / "score_full_run.py"
GUARDED_LAUNCHER = SCORING_DIR / "guarded_launcher.py"
DIMENSIONS = ("scientific_coverage", "current_matrix_evidence", "boundary_compliance",
              "artifact_reproducibility", "chinese_report_quality", "penalty_credit")
SCORE_FIELDS = ("rule_total",) + DIMENSIONS
REQUIRED_SCORING_MODE = "rule_only_fallback"
STUDY_COUNT = 11
SYSTEM_COUNT = 8
FULL_GRID_CELLS = STUDY_COUNT * SYSTEM_COUNT
DIFFERENCE_REPORT_LIMIT = 50
KEY_REPORT_LIMIT = 20
CREDENTIAL_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "SCORE_MODEL", "GOOGLE_API_KEY",
                   "GOOGLE_BASE_URL", "GOOGLE_CX", "SERPAPI_KEY", "ENTREZ_EMAIL",
                   "R27_BUDGET_DIR", "R26_BUDGET_DIR")
# Reported method names that are not archived system directory names: the aliases the frozen
# cross-check used. Conversion happens through this table and nowhere else; an unknown name
# fails instead of being guessed at.
REPORTED_METHOD_ALIASES = {"scProteoAgent": "Proposed (gpt-5.5)",
                           "Claude Code": "Claudecode",
                           "BioAgent": "Bioagent"}


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "run_status.json").is_file()


def is_link(path: Path) -> bool:
    """True for POSIX symlinks and for Windows junctions (reparse points)."""
    checker = getattr(path, "is_junction", None)
    return bool(path.is_symlink() or (checker is not None and checker()))


def discover_system_roots(entry: Path, explicit_system: str | None) -> list[tuple[str, Path]]:
    """Return (system, root) pairs; a root is a directory whose children are run dirs."""
    if explicit_system:
        return [(explicit_system, entry)]
    # A symlink or junction is not followed here: it would replay the same run directories a
    # second time under a different name. The link's own target is replayed from its real path.
    children = [child for child in sorted(entry.iterdir()) if child.is_dir()]
    links = [child.name for child in children if is_link(child)]
    if links:
        print("skipping linked directories (not followed): %s" % ", ".join(links))
    children = [child for child in children if not is_link(child)]
    if any(is_run_dir(child) for child in children):
        return [(entry.name, entry)]
    systems = [(child.name, child) for child in children
               if any(is_run_dir(g) for g in child.iterdir() if g.is_dir() and not is_link(g))]
    if systems:
        return systems
    raise SystemExit("FATAL: %s holds neither run directories nor per-system directories" % entry)


def clean_environment() -> dict:
    env = dict(os.environ)
    for name in CREDENTIAL_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_public_report_hashes(path: Path) -> dict[tuple[str, str, str], tuple[str, str]]:
    """Load only explicitly registered original-to-public report hash pairs."""
    mapping = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parts = row["relative_path"].replace("\\", "/").split("/")
            if len(parts) != 6 or parts[:3] != ["data", "scoring", "runs"]:
                raise ValueError("Invalid public report hash path: %r" % row["relative_path"])
            key = (parts[3], parts[4], parts[5])
            pair = (row["sha256_original"].upper(), row["sha256_public"].upper())
            if key in mapping and mapping[key] != pair:
                raise ValueError("Conflicting public report hash registration: %r" % (key,))
            mapping[key] = pair
    return mapping


def run_system(system: str, runs_root: Path, examples_root: Path, out_dir: Path,
               datasets: list[str] | None) -> dict:
    # The child runs in its own directory, so every path handed to it must be absolute: relative
    # arguments would be resolved against the child's working directory, not the caller's.
    out_dir = Path(out_dir).resolve()
    runs_root = Path(runs_root).resolve()
    examples_root = Path(examples_root).resolve()
    system_out = out_dir / system.replace(" ", "_")
    system_out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(GUARDED_LAUNCHER), str(FROZEN_SCORER),
               "--examples-root", str(examples_root),
               "--runs-root", str(runs_root),
               "--out-json", str(system_out / "score.json"),
               "--out-tsv", str(system_out / "score.tsv"),
               "--out-md", str(system_out / "score.md"),
               "--details-dir", str(system_out / "details"),
               "--skip-llm"]
    if datasets:
        command.extend(["--datasets", *datasets])
    completed = subprocess.run(command, capture_output=True, text=True, env=clean_environment(),
                               cwd=str(system_out), encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit("FATAL: replay failed for system %s (exit %d)" % (system, completed.returncode))
    payload = json.loads((system_out / "score.json").read_text(encoding="utf-8"))
    print("system=%s datasets=%d scoring_system=%s"
          % (system, payload.get("dataset_count", -1), payload.get("scoring_system", "")))
    return payload


def finite_number(value, field: str, key: str, where: str, failures: list[str]):
    """Return a finite float, or None plus an explicit failure reason."""
    if isinstance(value, bool) or value is None:
        failures.append("%s: %s [%s] is missing (value %r)" % (where, key, field, value))
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            failures.append("%s: %s [%s] is a blank or empty string" % (where, key, field))
            return None
        try:
            number = float(text)
        except ValueError:
            failures.append("%s: %s [%s] is not numeric (value %r)" % (where, key, field, value))
            return None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        failures.append("%s: %s [%s] has the unsupported type %s (value %r)"
                        % (where, key, field, type(value).__name__, value))
        return None
    if math.isnan(number) or math.isinf(number):
        failures.append("%s: %s [%s] is not finite (value %r)" % (where, key, field, value))
        return None
    return number


def collect_rows(payloads: dict[str, dict], failures: list[str]) -> list[dict]:
    """Flatten the per-system payloads into (study, system) records, rejecting bad values."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for system, payload in payloads.items():
        payload_rows = payload.get("rows") or []
        if not payload_rows:
            failures.append("system %s produced zero rows" % system)
        for row in payload_rows:
            study = str(row.get("dataset", "")).strip()
            if not study:
                failures.append("system %s returned a row without a dataset name" % system)
                continue
            key = (study, system)
            label = "%s / %s" % key
            if key in seen:
                failures.append("replayed key %s appears twice; a duplicated key must not be "
                                "collapsed by a dictionary" % label)
                continue
            seen.add(key)
            rule_result = row.get("rule_result") or {}
            dimensions = rule_result.get("dimensions") or {}
            record = {"dataset": study, "system": system,
                      "scoring_mode": str(row.get("scoring_mode", "")),
                      "report_path": str(row.get("report_path", ""))}
            record["rule_total"] = finite_number(row.get("rule_total"), "rule_total", label,
                                                "replayed row", failures)
            for name in DIMENSIONS:
                record[name] = finite_number(dimensions.get(name), name, label,
                                             "replayed row", failures)
            rows.append(record)
    return sorted(rows, key=lambda item: (item["system"], item["dataset"]))


def check_scoring_modes(rows: list[dict], failures: list[str]) -> None:
    for row in rows:
        mode = row.get("scoring_mode", "")
        if mode != REQUIRED_SCORING_MODE:
            failures.append("replayed key %s / %s has scoring_mode %r; the frozen rule layer "
                            "requires %r (the historical LLM adjustment is not comparable)"
                            % (row["dataset"], row["system"], mode, REQUIRED_SCORING_MODE))


def load_expected_table(path: Path, systems: list[str], requested_datasets: list[str] | None,
                        expected_method: str | None, failures: list[str]):
    """Return (entries, meta); entries maps (study, system) to the seven required numbers."""
    entries: dict[tuple[str, str], dict] = {}
    meta = {"rows": 0, "mode_filtered_rows": 0, "columns_present": [], "aliases_used": {}}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        table_rows = list(reader)
    meta["columns_present"] = fieldnames
    meta["rows"] = len(table_rows)
    if not fieldnames:
        failures.append("the frozen table %s has no header row" % path)
        return entries, meta
    if not table_rows:
        failures.append("the frozen table %s holds zero data rows" % path)
        return entries, meta
    missing_columns = [name for name in SCORE_FIELDS if name not in fieldnames]
    if missing_columns:
        failures.append("the frozen table %s lacks the column(s) %s"
                        % (path, ", ".join(missing_columns)))
        return entries, meta
    for index, row in enumerate(table_rows, start=2):
        mode = (row.get("mode") or "").strip()
        if mode and mode != "with_run_dir":
            meta["mode_filtered_rows"] += 1
            continue
        study = (row.get("dataset") or row.get("dataset_id") or row.get("study") or "").strip()
        raw_method = (expected_method or row.get("method") or row.get("system") or "").strip()
        if not study or not raw_method:
            failures.append("the frozen table %s row %d lacks a dataset or method value"
                            % (path, index))
            continue
        system = REPORTED_METHOD_ALIASES.get(raw_method, raw_method)
        if system not in systems:
            failures.append("the frozen table %s row %d names the method %r, which is neither "
                            "one of the replayed systems (%s) nor an entry of the explicit alias "
                            "table %s" % (path, index, raw_method, ", ".join(sorted(systems)),
                                          REPORTED_METHOD_ALIASES))
            continue
        if raw_method != system:
            meta["aliases_used"][raw_method] = meta["aliases_used"].get(raw_method, 0) + 1
        key = (study, system)
        label = "%s / %s" % key
        if key in entries:
            failures.append("the frozen table %s repeats the key %s (row %d); the expected "
                            "table must hold %d unique cells" % (path, label, index, FULL_GRID_CELLS))
            continue
        entry: dict = {}
        broken = False
        for field in SCORE_FIELDS:
            number = finite_number(row.get(field), field, label, "expected row %d" % index, failures)
            if number is None:
                broken = True
            entry[field] = number
        if broken:
            continue
        entry["report_sha256"] = (row.get("report_sha256") or "").strip().upper()
        entries[key] = entry
    if requested_datasets is not None:
        entries = {key: value for key, value in entries.items() if key[0] in requested_datasets}
    return entries, meta


def compare_with_expected(rows: list[dict], expected_path: Path, systems: list[str],
                          requested_datasets: list[str] | None, subset_mode: bool,
                          tolerance: float, expected_method: str | None = None,
                          check_report_sha256: bool = False,
                          public_report_hashes: dict[tuple[str, str, str], tuple[str, str]] | None = None,
                          prior_failures: list[str] | None = None) -> dict:
    """Completeness first, then values. Returns the comparison record for the summary."""
    failures: list[str] = list(prior_failures or [])
    entries, meta = load_expected_table(expected_path, systems, requested_datasets,
                                       expected_method, failures)
    actual = {(row["dataset"], row["system"]): row for row in rows}
    expected_keys = set(entries)
    actual_keys = set(actual)

    if subset_mode:
        scope_datasets = requested_datasets
        if scope_datasets is None:
            scope_datasets = sorted({study for study, _ in expected_keys})
        requested_keys = {(study, system) for study in scope_datasets for system in systems}
        mode = "subset"
    else:
        scope_datasets = sorted({study for study, _ in expected_keys})
        requested_keys = expected_keys
        mode = "full"
        if len(expected_keys) != FULL_GRID_CELLS:
            failures.append("full-grid mode requires exactly %d unique expected cells; the "
                            "frozen table contributes %d (use --datasets and/or --system for an "
                            "explicit subset)" % (FULL_GRID_CELLS, len(expected_keys)))
        if len(actual_keys) != FULL_GRID_CELLS:
            failures.append("full-grid mode requires exactly %d replayed cells; the replay "
                            "produced %d" % (FULL_GRID_CELLS, len(actual_keys)))

    expected_only = sorted(requested_keys - expected_keys)
    replayed_only = sorted(requested_keys - actual_keys)
    outside_scope = sorted(actual_keys - requested_keys)
    for label, keys in (("requested cells missing from the frozen table", expected_only),
                        ("requested cells missing from the replay", replayed_only),
                        ("replayed cells outside the requested scope", outside_scope)):
        if keys:
            failures.append("%s (%d): %s%s"
                            % (label, len(keys),
                               ", ".join("%s / %s" % key for key in keys[:KEY_REPORT_LIMIT]),
                               " ..." if len(keys) > KEY_REPORT_LIMIT else ""))

    compared_keys = sorted(requested_keys & expected_keys & actual_keys)
    differences: list[dict] = []
    max_diff = {field: 0.0 for field in SCORE_FIELDS}
    for key in compared_keys:
        reference = entries[key]
        observed = actual[key]
        for field in SCORE_FIELDS:
            expected_value = reference.get(field)
            observed_value = observed.get(field)
            if expected_value is None or observed_value is None:
                continue
            delta = abs(observed_value - expected_value)
            max_diff[field] = max(max_diff[field], delta)
            if delta > tolerance:
                differences.append({"key": "%s / %s" % key, "study": key[0], "system": key[1],
                                    "field": field, "expected": expected_value,
                                    "observed": observed_value, "abs_diff": delta})

    report_sha_checked = 0
    if check_report_sha256:
        if "report_sha256" not in meta["columns_present"]:
            failures.append("--check-report-sha256 was requested but the frozen table %s has no "
                            "report_sha256 column" % expected_path)
        for key in compared_keys:
            expected_sha = entries[key].get("report_sha256", "")
            report_path = actual[key].get("report_path", "")
            if not expected_sha:
                failures.append("--check-report-sha256 was requested but the frozen table gives no "
                                "report_sha256 for %s / %s, so that cell cannot be bound to a report"
                                % (key[0], key[1]))
                continue
            if not report_path:
                failures.append("the replayed row for %s / %s carries no report path, so its report "
                                "cannot be hash-checked" % (key[0], key[1]))
                continue
            report_file = Path(report_path)
            if not report_file.is_file():
                failures.append("report file for %s / %s does not exist: %s"
                                % (key[0], key[1], report_path))
                continue
            observed_sha = sha256_file(report_file)
            report_sha_checked += 1
            if observed_sha != expected_sha:
                registered = (public_report_hashes or {}).get(
                    (key[1], report_file.parent.name, report_file.name))
                if registered != (expected_sha, observed_sha):
                    failures.append("report sha256 for %s / %s: expected %s, observed %s (%s)"
                                    % (key[0], key[1], expected_sha, observed_sha, report_path))

    differences.sort(key=lambda item: (-item["abs_diff"], item["key"], item["field"]))
    if differences:
        failures.append("%d cell field(s) differ by more than the tolerance %g; the largest "
                        "difference is %g (the differences list records key, field, expected and "
                        "observed for each one)"
                        % (len(differences), tolerance, differences[0]["abs_diff"]))
    pass_verdict = "SUBSET_PASS" if subset_mode else "PASS"
    fail_verdict = "SUBSET_FAIL" if subset_mode else "FAIL"
    verdict = fail_verdict if failures else pass_verdict
    return {
        "mode": mode,
        "expected_file": str(expected_path),
        "expected_table_rows": meta["rows"],
        "rows_filtered_by_mode": meta["mode_filtered_rows"],
        "expected_cells": len(expected_keys),
        "replayed_cells": len(actual_keys),
        "requested_cells": len(requested_keys),
        "compared_cells": len(compared_keys),
        "missing_cells": len(expected_only) + len(replayed_only),
        "rows_without_expected_cell": len(replayed_only),
        "expected_only_keys": ["%s / %s" % key for key in expected_only[:KEY_REPORT_LIMIT]],
        "replayed_only_keys": ["%s / %s" % key for key in replayed_only[:KEY_REPORT_LIMIT]],
        "outside_scope_keys": ["%s / %s" % key for key in outside_scope[:KEY_REPORT_LIMIT]],
        "required_scoring_mode": REQUIRED_SCORING_MODE,
        "scoring_modes_observed": sorted({row.get("scoring_mode", "") for row in rows}),
        "aliases_applied": meta["aliases_used"],
        "tolerance": tolerance,
        "max_abs_diff": max_diff,
        "max_abs_diff_overall": max(max_diff.values()) if max_diff else 0.0,
        "difference_count": len(differences),
        "differences": differences[:DIFFERENCE_REPORT_LIMIT],
        "report_sha256_checked": report_sha_checked,
        "coverage": {"cells": len(compared_keys), "grid_cells": FULL_GRID_CELLS,
                     "fraction": round(len(compared_keys) / FULL_GRID_CELLS, 6),
                     "systems": len(systems), "datasets": len(scope_datasets)},
        "failures": failures,
        "verdict": verdict,
        "exit_code": 0 if verdict == pass_verdict else 2,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the frozen rule scorer offline.")
    parser.add_argument("--runs-root", action="append", required=True,
                        help="run-root directory; repeat for several systems")
    parser.add_argument("--examples-root", required=True, type=Path,
                        help="directory containing the dataset directories (the scoring-only "
                             "assembly written by prepare_scoring_inputs.py)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="explicit subset of studies; the comparison then reports "
                             "SUBSET_PASS with its coverage instead of a full-grid PASS")
    parser.add_argument("--system", default=None,
                        help="system name when a run root holds one system's run directories; "
                             "also switches the comparison to the subset mode")
    parser.add_argument("--expected-tsv", type=Path, default=None,
                        help="frozen rule-score table to compare against")
    parser.add_argument("--expected-method", default=None,
                        help="method column value in --expected-tsv for these runs")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="absolute tolerance for the comparison (default 0.0; a failure is "
                             "never relaxed automatically)")
    parser.add_argument("--check-report-sha256", action="store_true",
                        help="also require the sha256 of every replayed report to equal the "
                             "report_sha256 column of the frozen table")
    parser.add_argument("--public-report-hashes", type=Path, default=None,
                        help="archive TSV registering exact original-to-public report hashes")
    args = parser.parse_args(argv)

    if args.tolerance < 0:
        print("FATAL: --tolerance must not be negative (got %r)" % args.tolerance, file=sys.stderr)
        return 1
    if args.public_report_hashes and not args.public_report_hashes.is_file():
        print("FATAL: --public-report-hashes file is missing: %s" % args.public_report_hashes,
              file=sys.stderr)
        return 1
    public_report_hashes = (load_public_report_hashes(args.public_report_hashes)
                            if args.public_report_hashes else {})
    for required in (FROZEN_SCORER, GUARDED_LAUNCHER):
        if not required.is_file():
            print("FATAL: missing %s" % required, file=sys.stderr)
            return 1
    if not args.examples_root.is_dir():
        print("FATAL: --examples-root is not a directory: %s" % args.examples_root, file=sys.stderr)
        return 1

    systems: list[tuple[str, Path]] = []
    for raw in args.runs_root:
        entry = Path(raw)
        if not entry.is_dir():
            print("FATAL: --runs-root is not a directory: %s" % entry, file=sys.stderr)
            return 1
        systems.extend(discover_system_roots(entry, args.system))
    names = [name for name, _ in systems]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        print("FATAL: the same system would be replayed more than once: %s. Each system must "
              "come from exactly one run root, otherwise duplicate cells could be collapsed "
              "silently." % ", ".join(repeated), file=sys.stderr)
        return 1
    subset_mode = bool(args.datasets) or bool(args.system)
    print("systems discovered: %s" % ", ".join(names))
    print("comparison mode: %s" % ("subset (SUBSET_PASS at most)" if subset_mode else "full grid"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {name: run_system(name, root, args.examples_root, args.out_dir, args.datasets)
                for name, root in systems}
    failures: list[str] = []
    rows = collect_rows(payloads, failures)
    check_scoring_modes(rows, failures)
    summary = {"frozen_scorer": str(FROZEN_SCORER),
               "frozen_scorer_sha256": sha256_file(FROZEN_SCORER),
               "llm_layer": "disabled (--skip-llm)",
               "network": "blocked by guarded_launcher.py",
               "systems": names, "rows": rows, "row_count": len(rows)}
    print("rows replayed: %d (key failures so far: %d)" % (len(rows), len(failures)))

    if args.expected_tsv:
        comparison = compare_with_expected(rows, args.expected_tsv, names, args.datasets,
                                          subset_mode, args.tolerance,
                                          expected_method=args.expected_method,
                                          check_report_sha256=args.check_report_sha256,
                                          public_report_hashes=public_report_hashes,
                                          prior_failures=failures)
        summary["comparison"] = comparison
        failures = comparison["failures"]
        exit_code = comparison["exit_code"]
        print("comparison: %s mode=%s compared=%d/%d missing=%d tolerance=%g max_abs_diff=%s"
              % (comparison["verdict"], comparison["mode"], comparison["compared_cells"],
                 comparison["requested_cells"], comparison["missing_cells"], args.tolerance,
                 {key: round(value, 6) for key, value in comparison["max_abs_diff"].items()}))
        print("coverage: %d/%d cells (%.1f%%)"
              % (comparison["coverage"]["cells"], comparison["coverage"]["grid_cells"],
                 100.0 * comparison["coverage"]["fraction"]))
        for difference in comparison["differences"][:10]:
            print("difference: %s [%s] expected=%r observed=%r abs_diff=%g"
                  % (difference["key"], difference["field"], difference["expected"],
                     difference["observed"], difference["abs_diff"]))
        if comparison["difference_count"] > len(comparison["differences"]):
            print("difference: ... %d more entries are recorded in replay_summary.json"
                  % (comparison["difference_count"] - len(comparison["differences"])))
    else:
        exit_code = 0 if rows else 2
        if not rows:
            failures.append("the replay produced zero rows and no frozen table was given")
        summary["comparison"] = {"verdict": "NO_TABLE", "note": "--expected-tsv was not given"}
        print("no --expected-tsv given: %d rows written, no comparison performed" % len(rows))

    for failure in failures:
        print("FAILURE: %s" % failure, file=sys.stderr)
    if failures:
        print("RESULT: %s (%d failure reason(s))"
              % (summary["comparison"].get("verdict", "FAIL"), len(failures)), file=sys.stderr)
    summary["row_failures"] = failures

    (args.out_dir / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    with (args.out_dir / "replay_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "system", "scoring_mode", "rule_total", *DIMENSIONS]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("summary: %s" % (args.out_dir / "replay_summary.json"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
