"""Negative and positive completeness tests for the scoring replay comparator.

The comparator is the only layer that turns a replay into a claim about the frozen 88-cell
table, so its failure modes are tested directly: zero rows, a short table, a repeated key, a
missing dimension, NaN, Infinity, blank and non-numeric cells, a wrong scoring mode, an
unknown method alias, a flipped value, swapped cells and a partial grid that could be mistaken
for the full grid. Every case asserts the verdict, the exit code and the shape of the failure
output; the positive cases assert that a complete grid of equal values is accepted and that an
explicit subset can only reach SUBSET_PASS with its coverage.

The tests need no third-party package, no network, no archive and no model: the fixtures are
synthetic (eighty-eight cells: eleven synthetic studies times the eight archived system names)
and are built in a temporary directory. The frozen scorer is executed as a subprocess, exactly
as the replay driver executes it.

Usage:
  py -3 tests/test_replay_completeness.py [--work-dir DIR] [--python EXE] [--report-json FILE] [--list]

Exit codes: 0 = every case behaved as asserted, 1 = at least one case did not,
            3 = environment cannot run the tests (frozen scorer or launcher missing).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCORING = REPO / "reproduce" / "scoring"
SCORER = SCORING / "score_full_run.py"
LAUNCHER = SCORING / "guarded_launcher.py"
RUN_REPLAY = SCORING / "run_replay.py"
FROZEN_SCORER_SHA256 = "2B37C1835D397FBF132BAD9DEB8900220BCD3ABCD299ECEE8112CC0B3BEEA74B"
FROZEN_LAUNCHER_SHA256 = "56A38AA5533AC50C48F6D6673D26167EE2A7AE9345B3E5C885078D43262CE9D7"
CREDENTIAL_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "SCORE_MODEL", "GOOGLE_API_KEY",
                   "GOOGLE_BASE_URL", "GOOGLE_CX", "SERPAPI_KEY", "ENTREZ_EMAIL")

# The eight archived system names, used verbatim so that the alias table can be exercised.
SYSTEM_NAMES = ("Bioagent", "Biomni", "CellVoyager", "Claudecode", "Codex", "Hermes",
                "Proposed (gpt-5.5)", "SpatialAgent")
# Synthetic study names: the fixtures never contain real study data.
STUDY_NAMES = ("S01_Study", "S02_Study", "S03_Study", "S04_Study", "S05_Study", "S06_Study",
               "S07_Study", "S08_Study", "S09_Study", "S10_Study", "S11_Study")
DIMENSION_NAMES = ("scientific_coverage", "current_matrix_evidence", "boundary_compliance",
                   "artifact_reproducibility", "chinese_report_quality", "penalty_credit")
REPORT_BODY = (
    "## 一、执行摘要\n\n本报告比较 A 组与 B 组，使用 current_matrix 证据。\n\n"
    "## 二、核心证据首页\n\nlogFC 与 adj.P.Val 已列出，差异为 1.2 倍，n = 4。\n\n"
    "## 三、主要发现\n\nGAPDH 上调；结果表见 tables/DE.csv。\n\n"
    "## 四、证据边界\n\n未使用 offline_enrichment，未做 external_annotation。\n\n"
    "## 五、可复核证据附录\n\nfigures/plot.png\n")
GRID_CELLS = len(SYSTEM_NAMES) * len(STUDY_NAMES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_run_replay():
    spec = importlib.util.spec_from_file_location("run_replay_under_test", RUN_REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_environment() -> dict:
    env = dict(os.environ)
    for name in CREDENTIAL_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def cell_values(offset: float = 0.0):
    """Deterministic value table for all 88 cells, keyed by (study, system)."""
    values = {}
    for study_index, study in enumerate(STUDY_NAMES):
        for system_index, system in enumerate(SYSTEM_NAMES):
            total = round(30.0 + study_index * 2.0 + system_index * 0.5 + offset, 2)
            dimensions = {name: round(total / 6.0, 2) for name in DIMENSION_NAMES}
            values[(study, system)] = {"rule_total": total, **dimensions}
    return values


def payload_rows(system: str, cells: dict, scoring_mode: str = "rule_only_fallback",
                 report_path: str = "") -> dict:
    """Mimic one per-system score.json payload for the given cells."""
    rows = []
    for (study, cell_system), values in cells.items():
        if cell_system != system:
            continue
        dimensions = {name: values.get(name) for name in DIMENSION_NAMES}
        rows.append({"dataset": study, "scoring_mode": scoring_mode, "report_path": report_path,
                     "rule_total": values["rule_total"],
                     "rule_result": {"dimensions": dimensions}})
    return {"dataset_count": len(rows), "scoring_system": "fixture", "rows": rows}


def by_system(cells: dict) -> dict:
    grouped = {system: {} for system in SYSTEM_NAMES}
    for (study, system), values in cells.items():
        grouped[system][(study, system)] = dict(values)
    return grouped


def collect(module, cells_by_system: dict):
    failures: list[str] = []
    payloads = {system: payload_rows(system, cells) for system, cells in cells_by_system.items()}
    rows = module.collect_rows(payloads, failures)
    module.check_scoring_modes(rows, failures)
    return rows, failures


def write_expected_tsv(path: Path, rows: list[dict]) -> Path:
    """Write a frozen-table-shaped TSV; cells keep their raw text so malformed cells survive."""
    fields = ["dataset", "method", "mode", "rule_total", *DIMENSION_NAMES, "report_sha256"]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(field, "")) for field in fields))
    write_text(path, "\n".join(lines) + "\n")
    return path


def expected_rows_from_values(values: dict, method_for=None) -> list[dict]:
    rows = []
    for (study, system), cell in values.items():
        method = (method_for(system) or system) if method_for else system
        rows.append({"dataset": study, "method": method, "mode": "with_run_dir",
                     "report_sha256": "", **cell})
    return rows


def compare(module, cells_by_system, expected_path, requested_datasets=None,
            subset_mode=False, tolerance=0.0, expected_method=None,
            check_report_sha256=False):
    rows, failures = collect(module, cells_by_system)
    return module.compare_with_expected(
        rows, Path(expected_path), list(SYSTEM_NAMES), requested_datasets, subset_mode, tolerance,
        expected_method=expected_method, check_report_sha256=check_report_sha256,
        prior_failures=failures)

def observed_facts(comparison: dict, extra: dict | None = None) -> dict:
    facts = {"verdict": comparison.get("verdict", ""),
             "exit_code": comparison.get("exit_code", -1),
             "compared_cells": comparison.get("compared_cells", -1),
             "missing_cells": comparison.get("missing_cells", -1),
             "difference_count": comparison.get("difference_count", -1),
             "failure_count": len(comparison.get("failures", [])),
             "failures_are_explicit": bool(comparison.get("failures")),
             "max_abs_diff_overall": comparison.get("max_abs_diff_overall", None)}
    if extra:
        facts.update(extra)
    return facts


def failure_text(comparison: dict) -> str:
    return " | ".join(comparison.get("failures", []))


def build_grid_fixture(root: Path):
    """Eleven synthetic studies times eight systems, small enough to score in seconds."""
    examples = root / "examples"
    runs = root / "runs"
    for study in STUDY_NAMES:
        dataset = examples / study
        write_text(dataset / "ProteinQuant.csv",
                   "PG.ProteinGroups,PG.Genes,c1,c2\nP1,GAPDH,10,11\nP2,ACTB,20,21\n")
        write_text(dataset / "SampleInfo.csv", "FileName,Cluster\nc1,A\nc2,B\n")
        write_text(dataset / "user_input.txt", "Compare group B with group A.\n")
        write_text(dataset / "ground_truth.txt", "GAPDH increases; groups differ by cluster.\n")
        write_text(dataset / "grading_standard.txt",
                   "The report must compare A and B and state limits.\n")
        for system in SYSTEM_NAMES:
            run_dir = runs / system / study
            write_text(run_dir / "run_status.json", json.dumps({"run_status": "completed"}))
            write_text(run_dir / "report.md", REPORT_BODY)
    return examples, runs


def run_replay_cli(python: str, args: list, out_dir: Path):
    command = [python, str(RUN_REPLAY), *args]
    completed = subprocess.run(command, capture_output=True, text=True,
                               env=clean_environment(), cwd=str(REPO))
    summary_path = out_dir / "replay_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return completed, summary


def cli_facts(completed, summary: dict, extra: dict | None = None) -> dict:
    comparison = summary.get("comparison") or {}
    facts = {"exit_code": completed.returncode,
             "verdict": comparison.get("verdict", "NO_SUMMARY"),
             "row_count": summary.get("row_count", -1),
             "compared_cells": comparison.get("compared_cells", -1),
             "coverage_fraction": round((comparison.get("coverage") or {}).get("fraction", -1), 6)}
    if extra:
        facts.update(extra)
    return facts


def tsv_with_report_hashes(summary: dict, path: Path) -> Path:
    """Build a frozen-table-shaped TSV from a completed replay, including report hashes."""
    rows = []
    for row in summary.get("rows", []):
        report = Path(row.get("report_path", ""))
        rows.append({"dataset": row["dataset"], "method": row["system"], "mode": "with_run_dir",
                     "rule_total": row["rule_total"],
                     "report_sha256": sha256_file(report) if report.is_file() else "",
                     **{name: row[name] for name in DIMENSION_NAMES}})
    return write_expected_tsv(path, rows)


def case_frozen_scorer_identity(env: dict) -> dict:
    return {"scorer_sha256": sha256_file(SCORER), "launcher_sha256": sha256_file(LAUNCHER)}


def case_positive_full_grid(env: dict) -> dict:
    values = cell_values()
    path = write_expected_tsv(env["root"] / "expected" / "full_grid.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], by_system(values), path)
    return observed_facts(comparison, {
        "expected_cells": comparison["expected_cells"],
        "report_sha256_checked": comparison["report_sha256_checked"]})


def case_positive_aliases(env: dict) -> dict:
    values = cell_values()
    aliases = {"Proposed (gpt-5.5)": "scProteoAgent", "Claudecode": "Claude Code",
               "Bioagent": "BioAgent"}
    path = write_expected_tsv(env["root"] / "expected" / "full_grid_aliases.tsv",
                              expected_rows_from_values(values, aliases.get))
    comparison = compare(env["module"], by_system(values), path)
    return observed_facts(comparison, {
        "aliases_applied": sum(comparison["aliases_applied"].values())})


def case_negative_empty_replay(env: dict) -> dict:
    values = cell_values()
    path = write_expected_tsv(env["root"] / "expected" / "empty_replay.tsv",
                              expected_rows_from_values(values))
    failures: list[str] = []
    payloads = {system: payload_rows(system, {}) for system in SYSTEM_NAMES}
    rows = env["module"].collect_rows(payloads, failures)
    env["module"].check_scoring_modes(rows, failures)
    comparison = env["module"].compare_with_expected(
        rows, Path(path), list(SYSTEM_NAMES), None, False, 0.0, prior_failures=failures)
    return observed_facts(comparison,
                          {"mentions_zero_rows": "zero rows" in failure_text(comparison)})


def case_negative_short_replay(env: dict) -> dict:
    values = cell_values()
    path = write_expected_tsv(env["root"] / "expected" / "short_replay.tsv",
                              expected_rows_from_values(values))
    short = dict(values)
    short.pop((STUDY_NAMES[-1], SYSTEM_NAMES[-1]))
    comparison = compare(env["module"], by_system(short), path)
    return observed_facts(comparison, {"replayed_cells": comparison["replayed_cells"]})


def case_negative_short_expected(env: dict) -> dict:
    values = cell_values()
    rows = expected_rows_from_values(values)
    rows.pop()
    path = write_expected_tsv(env["root"] / "expected" / "short_expected.tsv", rows)
    comparison = compare(env["module"], by_system(values), path)
    return observed_facts(comparison, {"expected_cells": comparison["expected_cells"]})


def case_negative_partial_grid(env: dict) -> dict:
    values = {(study, system): cell for (study, system), cell in cell_values().items()
              if system in SYSTEM_NAMES[:4]}
    path = write_expected_tsv(env["root"] / "expected" / "partial_grid.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], by_system(values), path)
    return observed_facts(comparison, {
        "expected_cells": comparison["expected_cells"],
        "grid_gate_reported": "full-grid mode requires" in failure_text(comparison)})


def case_negative_duplicate_expected(env: dict) -> dict:
    values = cell_values()
    rows = expected_rows_from_values(values)
    rows.append(dict(rows[0]))
    path = write_expected_tsv(env["root"] / "expected" / "duplicate_expected.tsv", rows)
    comparison = compare(env["module"], by_system(values), path)
    return observed_facts(comparison,
                          {"mentions_repeat": "repeats the key" in failure_text(comparison)})


def case_negative_duplicate_replayed(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    system = SYSTEM_NAMES[0]
    payloads = {name: payload_rows(name, cells) for name, cells in grouped.items()}
    rows = payloads[system]["rows"]
    rows.append(dict(rows[0]))
    failures: list[str] = []
    flat = env["module"].collect_rows(payloads, failures)
    path = write_expected_tsv(env["root"] / "expected" / "duplicate_replayed.tsv",
                              expected_rows_from_values(values))
    comparison = env["module"].compare_with_expected(
        flat, Path(path), list(SYSTEM_NAMES), None, False, 0.0, prior_failures=failures)
    facts = observed_facts(comparison)
    facts["mentions_duplicate_key"] = "appears twice" in failure_text(comparison)
    return facts


def case_negative_missing_dimension(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    system, study = SYSTEM_NAMES[1], STUDY_NAMES[0]
    cell = grouped[system][(study, system)]
    grouped[system][(study, system)] = {name: value for name, value in cell.items()
                                        if name != "penalty_credit"}
    path = write_expected_tsv(env["root"] / "expected" / "missing_dimension.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], grouped, path)
    facts = observed_facts(comparison)
    facts["mentions_missing_dimension"] = "penalty_credit" in failure_text(comparison)
    return facts


def case_negative_nan(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    system, study = SYSTEM_NAMES[2], STUDY_NAMES[1]
    grouped[system][(study, system)]["rule_total"] = float("nan")
    path = write_expected_tsv(env["root"] / "expected" / "nan.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], grouped, path)
    facts = observed_facts(comparison)
    facts["mentions_not_finite"] = "not finite" in failure_text(comparison)
    return facts


def case_negative_inf(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    system, study = SYSTEM_NAMES[3], STUDY_NAMES[2]
    grouped[system][(study, system)]["scientific_coverage"] = float("inf")
    path = write_expected_tsv(env["root"] / "expected" / "inf.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], grouped, path)
    facts = observed_facts(comparison)
    facts["mentions_not_finite"] = "not finite" in failure_text(comparison)
    return facts


def case_negative_blank_cell(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    system, study = SYSTEM_NAMES[4], STUDY_NAMES[3]
    grouped[system][(study, system)]["rule_total"] = "   "
    path = write_expected_tsv(env["root"] / "expected" / "blank.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], grouped, path)
    facts = observed_facts(comparison)
    facts["mentions_blank"] = "blank or empty string" in failure_text(comparison)
    return facts


def case_negative_text_cell(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    system, study = SYSTEM_NAMES[5], STUDY_NAMES[4]
    grouped[system][(study, system)]["rule_total"] = "sixty"
    path = write_expected_tsv(env["root"] / "expected" / "text.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], grouped, path)
    facts = observed_facts(comparison)
    facts["mentions_not_numeric"] = "not numeric" in failure_text(comparison)
    return facts

def case_negative_expected_nan(env: dict) -> dict:
    values = cell_values()
    rows = expected_rows_from_values(values)
    rows[0]["rule_total"] = "NaN"
    path = write_expected_tsv(env["root"] / "expected" / "expected_nan.tsv", rows)
    comparison = compare(env["module"], by_system(values), path)
    facts = observed_facts(comparison)
    facts["mentions_not_finite"] = "not finite" in failure_text(comparison)
    return facts


def case_negative_expected_inf(env: dict) -> dict:
    values = cell_values()
    rows = expected_rows_from_values(values)
    rows[0]["scientific_coverage"] = "Infinity"
    path = write_expected_tsv(env["root"] / "expected" / "expected_inf.tsv", rows)
    comparison = compare(env["module"], by_system(values), path)
    facts = observed_facts(comparison)
    facts["mentions_not_finite"] = "not finite" in failure_text(comparison)
    return facts


def case_negative_expected_blank(env: dict) -> dict:
    values = cell_values()
    rows = expected_rows_from_values(values)
    rows[0]["boundary_compliance"] = ""
    path = write_expected_tsv(env["root"] / "expected" / "expected_blank.tsv", rows)
    comparison = compare(env["module"], by_system(values), path)
    facts = observed_facts(comparison)
    facts["mentions_blank"] = "blank or empty string" in failure_text(comparison)
    return facts


def case_negative_wrong_mode(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    payloads = {}
    for system, cells in grouped.items():
        mode = "rule_plus_llm_review" if system == SYSTEM_NAMES[6] else "rule_only_fallback"
        payloads[system] = payload_rows(system, cells, scoring_mode=mode)
    failures: list[str] = []
    rows = env["module"].collect_rows(payloads, failures)
    env["module"].check_scoring_modes(rows, failures)
    path = write_expected_tsv(env["root"] / "expected" / "wrong_mode.tsv",
                              expected_rows_from_values(values))
    comparison = env["module"].compare_with_expected(
        rows, Path(path), list(SYSTEM_NAMES), None, False, 0.0, prior_failures=failures)
    facts = observed_facts(comparison)
    facts["mentions_scoring_mode"] = "scoring_mode" in failure_text(comparison)
    return facts


def case_negative_flipped_value(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    system, study = SYSTEM_NAMES[0], STUDY_NAMES[5]
    cell = grouped[system][(study, system)]
    cell["rule_total"] = -cell["rule_total"]
    path = write_expected_tsv(env["root"] / "expected" / "flipped.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], grouped, path)
    differences = comparison["differences"]
    return observed_facts(comparison, {
        "first_difference_has_fields": bool(differences)
        and {"key", "field", "expected", "observed"} <= set(differences[0]),
        "first_difference_field": differences[0]["field"] if differences else "",
        "difference_matches_cell": bool(differences)
        and differences[0]["key"] == "%s / %s" % (study, system)})


def case_negative_swapped_cells(env: dict) -> dict:
    values = cell_values()
    study_a, study_b = STUDY_NAMES[0], STUDY_NAMES[1]
    system = SYSTEM_NAMES[2]
    grouped = by_system(values)
    grouped[system][(study_a, system)], grouped[system][(study_b, system)] = (
        grouped[system][(study_b, system)], grouped[system][(study_a, system)])
    path = write_expected_tsv(env["root"] / "expected" / "swapped.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], grouped, path)
    keys = {entry["key"] for entry in comparison["differences"]}
    return observed_facts(comparison, {
        "swapped_cells_reported": {"%s / %s" % (study_a, system),
                                   "%s / %s" % (study_b, system)} <= keys})


def case_negative_unknown_alias(env: dict) -> dict:
    values = cell_values()
    rows = expected_rows_from_values(
        values, lambda system: "CodexCLI" if system == "Codex" else system)
    path = write_expected_tsv(env["root"] / "expected" / "unknown_alias.tsv", rows)
    comparison = compare(env["module"], by_system(values), path)
    facts = observed_facts(comparison)
    facts["mentions_unknown_method"] = "CodexCLI" in failure_text(comparison)
    return facts


def case_negative_tolerance_not_widened(env: dict) -> dict:
    values = cell_values()
    grouped = by_system(values)
    system, study = SYSTEM_NAMES[7], STUDY_NAMES[6]
    cell = grouped[system][(study, system)]
    cell["rule_total"] = round(cell["rule_total"] + 0.01, 4)
    path = write_expected_tsv(env["root"] / "expected" / "tolerance.tsv",
                              expected_rows_from_values(values))
    comparison = compare(env["module"], grouped, path)
    return observed_facts(comparison, {"tolerance_recorded": comparison["tolerance"]})


def case_positive_subset(env: dict) -> dict:
    values = cell_values()
    requested = [STUDY_NAMES[0], STUDY_NAMES[1]]
    path = write_expected_tsv(env["root"] / "expected" / "subset.tsv",
                              expected_rows_from_values(values))
    scoped = {key: value for key, value in values.items() if key[0] in requested}
    comparison = compare(env["module"], by_system(scoped), path, requested_datasets=requested,
                         subset_mode=True)
    return observed_facts(comparison, {
        "coverage_cells": comparison["coverage"]["cells"],
        "coverage_fraction": comparison["coverage"]["fraction"],
        "verdict_is_not_full_pass": comparison["verdict"] != "PASS"})


def case_negative_subset_missing_expected(env: dict) -> dict:
    values = {(study, system): cell for (study, system), cell in cell_values().items()
              if study != STUDY_NAMES[-1]}
    requested = [STUDY_NAMES[0], STUDY_NAMES[-1]]
    path = write_expected_tsv(env["root"] / "expected" / "subset_missing_expected.tsv",
                              expected_rows_from_values(values))
    grouped = by_system(values)
    for system in SYSTEM_NAMES:
        grouped[system][(STUDY_NAMES[-1], system)] = dict(list(grouped[system].values())[0])
    comparison = compare(env["module"], grouped, path, requested_datasets=requested,
                         subset_mode=True)
    facts = observed_facts(comparison)
    facts["mentions_missing_from_table"] = "missing from the frozen table" in failure_text(comparison)
    return facts


def case_negative_subset_missing_replay(env: dict) -> dict:
    values = cell_values()
    requested = [STUDY_NAMES[0], STUDY_NAMES[1]]
    path = write_expected_tsv(env["root"] / "expected" / "subset_missing_replay.tsv",
                              expected_rows_from_values(values))
    grouped = by_system({key: value for key, value in values.items() if key[0] != STUDY_NAMES[1]})
    comparison = compare(env["module"], grouped, path, requested_datasets=requested,
                         subset_mode=True)
    facts = observed_facts(comparison)
    facts["mentions_missing_from_replay"] = "missing from the replay" in failure_text(comparison)
    return facts


def case_negative_expected_sha_blank(env: dict) -> dict:
    """A present but empty report_sha256 column must not count as a verified binding."""
    values = cell_values()
    rows = expected_rows_from_values(values)
    path = write_expected_tsv(env["root"] / "expected" / "sha_blank.tsv", rows)
    comparison = compare(env["module"], by_system(values), path, check_report_sha256=True)
    facts = observed_facts(comparison)
    facts["report_sha256_checked"] = comparison["report_sha256_checked"]
    facts["mentions_no_report_sha256"] = "gives no report_sha256" in failure_text(comparison)
    return facts

def case_e2e_full_grid(env: dict) -> dict:
    grid = env["grid"]
    out_dir = env["root"] / "e2e" / "full"
    first, first_summary = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir)], out_dir)
    expected = tsv_with_report_hashes(first_summary, env["root"] / "e2e" / "full_expected.tsv")
    out_dir_second = env["root"] / "e2e" / "full_second"
    second, second_summary = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir_second), "--expected-tsv", str(expected),
        "--check-report-sha256"], out_dir_second)
    comparison = second_summary.get("comparison") or {}
    return cli_facts(second, second_summary, {
        "first_exit_code": first.returncode,
        "first_row_count": first_summary.get("row_count", -1),
        "report_sha256_checked": comparison.get("report_sha256_checked", -1),
        "difference_count": comparison.get("difference_count", -1)})


def case_e2e_cell_mismatch(env: dict) -> dict:
    grid = env["grid"]
    out_dir = env["root"] / "e2e" / "mismatch_base"
    _, summary = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir)], out_dir)
    expected = tsv_with_report_hashes(summary, env["root"] / "e2e" / "mismatch_expected.tsv")
    text = expected.read_text(encoding="utf-8").splitlines()
    header = text[0].split("\t")
    target = header.index("rule_total")
    inflated = text[1].split("\t")
    inflated[target] = str(float(inflated[target]) + 0.5)
    write_text(expected, "\n".join([text[0], "\t".join(inflated), *text[2:]]) + "\n")
    out_dir_second = env["root"] / "e2e" / "mismatch_second"
    completed, summary_second = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir_second), "--expected-tsv", str(expected)], out_dir_second)
    comparison = summary_second.get("comparison") or {}
    differences = comparison.get("differences") or []
    return cli_facts(completed, summary_second, {
        "difference_count": comparison.get("difference_count", -1),
        "difference_has_fields": bool(differences)
        and {"key", "field", "expected", "observed"} <= set(differences[0])})


def case_e2e_report_sha_mismatch(env: dict) -> dict:
    grid = env["grid"]
    out_dir = env["root"] / "e2e" / "sha_base"
    _, summary = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir)], out_dir)
    expected = tsv_with_report_hashes(summary, env["root"] / "e2e" / "sha_expected.tsv")
    text = expected.read_text(encoding="utf-8").splitlines()
    header = text[0].split("\t")
    target = header.index("report_sha256")
    tampered = text[1].split("\t")
    tampered[target] = "0" * 64
    write_text(expected, "\n".join([text[0], "\t".join(tampered), *text[2:]]) + "\n")
    out_dir_second = env["root"] / "e2e" / "sha_second"
    completed, summary_second = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir_second), "--expected-tsv", str(expected),
        "--check-report-sha256"], out_dir_second)
    comparison = summary_second.get("comparison") or {}
    return cli_facts(completed, summary_second, {
        "mentions_report_sha256": "report sha256" in " | ".join(comparison.get("failures", []))})


def case_e2e_subset(env: dict) -> dict:
    grid = env["grid"]
    out_dir = env["root"] / "e2e" / "subset_base"
    _, summary = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir)], out_dir)
    expected = tsv_with_report_hashes(summary, env["root"] / "e2e" / "subset_expected.tsv")
    out_dir_second = env["root"] / "e2e" / "subset_second"
    completed, summary_second = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir_second), "--datasets", STUDY_NAMES[0], STUDY_NAMES[1],
        "--expected-tsv", str(expected)], out_dir_second)
    comparison = summary_second.get("comparison") or {}
    return cli_facts(completed, summary_second, {
        "coverage_cells": (comparison.get("coverage") or {}).get("cells", -1),
        "verdict_is_not_full_pass": comparison.get("verdict") != "PASS"})


def case_e2e_partial_grid(env: dict) -> dict:
    grid = env["grid"]
    out_dir = env["root"] / "e2e" / "partial_base"
    _, summary = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(grid["examples"]),
        "--out-dir", str(out_dir)], out_dir)
    expected = tsv_with_report_hashes(summary, env["root"] / "e2e" / "partial_expected.tsv")
    partial_examples = env["root"] / "e2e" / "partial_examples"
    for study in STUDY_NAMES[:2]:
        shutil.copytree(grid["examples"] / study, partial_examples / study)
    out_dir_second = env["root"] / "e2e" / "partial_second"
    completed, summary_second = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(partial_examples),
        "--out-dir", str(out_dir_second), "--expected-tsv", str(expected)], out_dir_second)
    comparison = summary_second.get("comparison") or {}
    return cli_facts(completed, summary_second, {
        "mentions_full_grid_requirement": "full-grid mode requires"
        in " | ".join(comparison.get("failures", []))})


def case_e2e_duplicate_system_root(env: dict) -> dict:
    grid = env["grid"]
    out_dir = env["root"] / "e2e" / "duplicate_root"
    completed, summary = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--runs-root", str(grid["runs"]),
        "--examples-root", str(grid["examples"]), "--out-dir", str(out_dir)], out_dir)
    return cli_facts(completed, summary, {
        "mentions_repeat_system": "more than once" in (completed.stderr or "")})


def case_e2e_zero_rows(env: dict) -> dict:
    grid = env["grid"]
    empty = env["root"] / "e2e" / "empty_examples"
    empty.mkdir(parents=True, exist_ok=True)
    out_dir = env["root"] / "e2e" / "zero_rows"
    completed, summary = run_replay_cli(env["python"], [
        "--runs-root", str(grid["runs"]), "--examples-root", str(empty),
        "--out-dir", str(out_dir)], out_dir)
    return cli_facts(completed, summary)

CASES = (
    ("frozen_scorer_identity",
     "score_full_run.py and guarded_launcher.py still carry their frozen hashes",
     case_frozen_scorer_identity,
     {"scorer_sha256": FROZEN_SCORER_SHA256, "launcher_sha256": FROZEN_LAUNCHER_SHA256}),
    ("positive_full_grid", "88 equal cells, seven fields each, full-grid PASS",
     case_positive_full_grid,
     {"verdict": "PASS", "exit_code": 0, "expected_cells": GRID_CELLS,
      "compared_cells": GRID_CELLS, "missing_cells": 0, "failure_count": 0,
      "difference_count": 0, "max_abs_diff_overall": 0.0}),
    ("positive_aliases", "aliases convert through the explicit table only",
     case_positive_aliases,
     {"verdict": "PASS", "exit_code": 0, "aliases_applied": 33, "failure_count": 0}),
    ("negative_empty_replay", "zero replayed rows must not pass",
     case_negative_empty_replay,
     {"verdict": "FAIL", "exit_code": 2, "compared_cells": 0, "failures_are_explicit": True,
      "mentions_zero_rows": True}),
    ("negative_short_replay", "87 replayed rows must not pass",
     case_negative_short_replay,
     {"verdict": "FAIL", "exit_code": 2, "replayed_cells": GRID_CELLS - 1, "failure_count": 2}),
    ("negative_short_expected", "a frozen table with 87 cells must not pass",
     case_negative_short_expected,
     {"verdict": "FAIL", "exit_code": 2, "expected_cells": GRID_CELLS - 1,
      "failures_are_explicit": True}),
    ("negative_partial_grid", "a partial grid must not masquerade as the full grid",
     case_negative_partial_grid,
     {"verdict": "FAIL", "exit_code": 2, "expected_cells": 4 * len(STUDY_NAMES),
      "grid_gate_reported": True}),
    ("negative_duplicate_expected", "a repeated expected key must be reported",
     case_negative_duplicate_expected,
     {"verdict": "FAIL", "exit_code": 2, "mentions_repeat": True}),
    ("negative_duplicate_replayed", "a repeated replayed key must be reported",
     case_negative_duplicate_replayed,
     {"verdict": "FAIL", "exit_code": 2, "mentions_duplicate_key": True}),
    ("negative_missing_dimension", "a missing dimension must be reported",
     case_negative_missing_dimension,
     {"verdict": "FAIL", "exit_code": 2, "mentions_missing_dimension": True}),
    ("negative_nan", "NaN in a replayed cell must be rejected",
     case_negative_nan,
     {"verdict": "FAIL", "exit_code": 2, "mentions_not_finite": True}),
    ("negative_inf", "Infinity in a replayed cell must be rejected",
     case_negative_inf,
     {"verdict": "FAIL", "exit_code": 2, "mentions_not_finite": True}),
    ("negative_blank_cell", "a blank replayed cell must be rejected",
     case_negative_blank_cell,
     {"verdict": "FAIL", "exit_code": 2, "mentions_blank": True}),
    ("negative_text_cell", "non-numeric replayed text must be rejected",
     case_negative_text_cell,
     {"verdict": "FAIL", "exit_code": 2, "mentions_not_numeric": True}),
    ("negative_expected_nan", "NaN in the frozen table must be rejected",
     case_negative_expected_nan,
     {"verdict": "FAIL", "exit_code": 2, "mentions_not_finite": True}),
    ("negative_expected_inf", "Infinity in the frozen table must be rejected",
     case_negative_expected_inf,
     {"verdict": "FAIL", "exit_code": 2, "mentions_not_finite": True}),
    ("negative_expected_blank", "an empty frozen-table cell must be rejected",
     case_negative_expected_blank,
     {"verdict": "FAIL", "exit_code": 2, "mentions_blank": True}),
    ("negative_wrong_mode", "a scoring mode other than the frozen rule layer must be rejected",
     case_negative_wrong_mode,
     {"verdict": "FAIL", "exit_code": 2, "mentions_scoring_mode": True}),
    ("negative_flipped_value", "a flipped value is reported with key/field/expected/observed",
     case_negative_flipped_value,
     {"verdict": "FAIL", "exit_code": 2, "first_difference_has_fields": True,
      "first_difference_field": "rule_total", "difference_matches_cell": True,
      "difference_count": 1}),
    ("negative_swapped_cells", "values attached to the wrong cells must be reported",
     case_negative_swapped_cells,
     {"verdict": "FAIL", "exit_code": 2, "swapped_cells_reported": True,
      "difference_count": 2 * len(DIMENSION_NAMES) + 2}),
    ("negative_unknown_alias", "an unknown method name must not be silently mapped",
     case_negative_unknown_alias,
     {"verdict": "FAIL", "exit_code": 2, "mentions_unknown_method": True}),
    ("negative_tolerance_not_widened", "a difference above the default tolerance fails",
     case_negative_tolerance_not_widened,
     {"verdict": "FAIL", "exit_code": 2, "tolerance_recorded": 0.0, "difference_count": 1}),
    ("positive_subset", "an explicit subset reports SUBSET_PASS with its coverage",
     case_positive_subset,
     {"verdict": "SUBSET_PASS", "exit_code": 0, "coverage_cells": len(SYSTEM_NAMES) * 2,
      "coverage_fraction": round(len(SYSTEM_NAMES) * 2 / GRID_CELLS, 6),
      "verdict_is_not_full_pass": True, "failure_count": 0}),
    ("negative_subset_missing_expected", "a requested cell absent from the table fails",
     case_negative_subset_missing_expected,
     {"verdict": "SUBSET_FAIL", "exit_code": 2, "mentions_missing_from_table": True}),
    ("negative_subset_missing_replay", "a requested cell absent from the replay fails",
     case_negative_subset_missing_replay,
     {"verdict": "SUBSET_FAIL", "exit_code": 2, "mentions_missing_from_replay": True}),
    ("negative_expected_sha_blank",
     "an empty report_sha256 column cannot satisfy --check-report-sha256",
     case_negative_expected_sha_blank,
     {"verdict": "FAIL", "exit_code": 2, "mentions_no_report_sha256": True,
      "report_sha256_checked": 0}),
    ("e2e_full_grid", "end to end: 88 cells replayed and matched, reports hash-verified",
     case_e2e_full_grid,
     {"exit_code": 0, "verdict": "PASS", "row_count": GRID_CELLS, "first_exit_code": 0,
      "first_row_count": GRID_CELLS, "compared_cells": GRID_CELLS, "difference_count": 0,
      "report_sha256_checked": GRID_CELLS}),
    ("e2e_cell_mismatch", "end to end: one cell off by 0.5 fails with key/field/expected/observed",
     case_e2e_cell_mismatch,
     {"exit_code": 2, "verdict": "FAIL", "difference_count": 1, "difference_has_fields": True}),
    ("e2e_report_sha_mismatch", "end to end: a tampered report hash fails the check",
     case_e2e_report_sha_mismatch,
     {"exit_code": 2, "verdict": "FAIL", "mentions_report_sha256": True}),
    ("e2e_subset", "end to end: --datasets can only reach SUBSET_PASS",
     case_e2e_subset,
     {"exit_code": 0, "verdict": "SUBSET_PASS", "row_count": len(SYSTEM_NAMES) * 2,
      "coverage_cells": len(SYSTEM_NAMES) * 2, "verdict_is_not_full_pass": True}),
    ("e2e_partial_grid", "end to end: two studies scored against the 88-cell table fail",
     case_e2e_partial_grid,
     {"exit_code": 2, "verdict": "FAIL", "row_count": len(SYSTEM_NAMES) * 2,
      "mentions_full_grid_requirement": True}),
    ("e2e_duplicate_system_root", "end to end: the same run root twice is a usage error",
     case_e2e_duplicate_system_root,
     {"exit_code": 1, "mentions_repeat_system": True}),
    ("e2e_zero_rows", "end to end: a replay that produces nothing exits non-zero",
     case_e2e_zero_rows,
     {"exit_code": 2, "row_count": 0}),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay comparison completeness tests.")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    parser.add_argument("--only", default=None, help="run a single case by id")
    args = parser.parse_args(argv)

    if args.list:
        for case_id, description, _, _ in CASES:
            print("%-34s %s" % (case_id, description))
        return 0
    for required in (SCORER, LAUNCHER, RUN_REPLAY):
        if not required.is_file():
            print("BLOCKED: missing %s" % required)
            return 3

    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="replay_completeness_"))
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "expected").mkdir(parents=True, exist_ok=True)
    print("work dir: %s" % work_root)
    module = load_run_replay()
    examples, runs = build_grid_fixture(work_root / "grid")
    env = {"root": work_root, "module": module, "python": args.python,
           "grid": {"examples": examples, "runs": runs}}

    selected = [case for case in CASES if args.only is None or case[0] == args.only]
    if not selected:
        print("BLOCKED: --only %s matches no case" % args.only)
        return 3
    print("cases: %d" % len(selected))

    results = []
    failures = 0
    for case_id, description, func, expectations in selected:
        try:
            observed = func(env)
            mismatch = {key: {"expected": value, "observed": observed.get(key)}
                        for key, value in expectations.items() if observed.get(key) != value}
            state = "PASS" if not mismatch else "FAIL"
        except Exception as exc:  # a case that cannot run counts as a failure, never a pass
            observed = {}
            mismatch = {"exception": {"expected": "no exception",
                                      "observed": "%s: %s" % (type(exc).__name__, exc)}}
            state = "FAIL"
        if state == "FAIL":
            failures += 1
        results.append({"case_id": case_id, "description": description, "result": state,
                        "expected": expectations, "observed": observed, "mismatch": mismatch})
        print("%-4s %-34s %s" % (state, case_id, description))
        for key, delta in mismatch.items():
            print("      %s: expected %r, observed %r" % (key, delta["expected"], delta["observed"]))

    report = {"work_dir": str(work_root), "case_count": len(selected), "failure_count": failures,
              "frozen_scorer_sha256": sha256_file(SCORER),
              "frozen_launcher_sha256": sha256_file(LAUNCHER),
              "run_replay_sha256": sha256_file(RUN_REPLAY),
              "results": results}
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                    encoding="utf-8", newline="\n")
        print("report: %s" % args.report_json)
    print("RESULT: %s (%d of %d cases failed)"
          % ("FAIL" if failures else "PASS", failures, len(selected)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
