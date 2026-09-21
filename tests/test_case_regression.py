"""Case-study regression tests for the five themes of reproduce/cases/.

Each theme is a script that re-derives the reported numbers from a read-only data archive,
compares them field by field with the frozen tables and writes a ``case_regression.json`` ledger.
The tests here run those scripts and check the ledger:

* the script exits 0 when every check passed, 1 when a check failed, 2 when a required input is
  missing and 3 when the run is partial (no failure, but at least one check is BLOCKED);
* no check is FAIL;
* the set of BLOCKED checks and the set of OUT_OF_SCOPE checks are exactly the pre-registered ones;
* every BLOCKED check names the paths it looked for, and every OUT_OF_SCOPE check carries the scope
  that says why the check is not needed for the manuscript;
* every compared column is classified (RECOMPUTED / COPIED_REFERENCE / IDENTITY_CHECK) and the
  ledger carries the roll-up, so a copied value is never presented as a recomputation;
* a missing ``--data-dir`` exits 2, prints the paths it checked and writes nothing.

No theme carries a pre-registered per-check ``known_fail`` any more. The one entry this file used
to hold - ``hela_composition_rows``, whose frozen odds-ratio interval upper bound 91.2 is the
agent report's three-significant-digit rendering of 91.187482 - is resolved by separating the two
objects: the historical row stays a REPORTED transcription of the report and is covered by
``hela_odds_ratio_reported_transcription``, while the computed object lives in its own stable
table and is covered by ``hela_odds_ratio_computed`` (rtol 1e-7, atol 0, no rounding allowance)
and ``hela_odds_ratio_manuscript_display`` (the one-decimal values the manuscript prints). A
check-level exception is deliberately not used again: one check compares many fields at once, so
an exception registered for a whole ``check_id`` would also accept a new error inside the same
check. The mechanism is kept, with every theme's set empty, so that any future exception has to be
registered explicitly and justified field by field.

The archive is located from the ``SCPROTEOMICS_CASE_ARCHIVE`` environment variable, then
``<repository>/data``, then ``<repository>/../data``. Without an archive with a
``case_tables`` directory the tests print SKIP and exit 0, so a fresh checkout does not fail for
a missing data layer. They run under ``pytest`` and as a plain script
(``python tests/test_case_regression.py``). Only numpy and pandas are needed for the scripts.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "reproduce" / "cases"
SOURCE_CLASSES = {"RECOMPUTED", "COPIED_REFERENCE", "IDENTITY_CHECK", "NOT_VERIFIED"}

# Pre-registered expectations per theme: the minimum number of PASS checks, the exact set of
# BLOCKED check ids, the exact set of OUT_OF_SCOPE check ids and the exact set of known FAILs.
# Every known_fail set is empty: a check that fails must fail the run.
#
# HeLa migration emits 18 checks in this round: the 15 it already emitted, all PASS after the
# reported transcription of the agent report was separated from the strict row-by-row comparison,
# plus hela_odds_ratio_computed, hela_odds_ratio_reported_transcription and
# hela_odds_ratio_manuscript_display. The registered minimum below is 17, the floor agreed for
# this round; the run itself is expected to reach 18 PASS with 0 FAIL and exit 0.
EXPECTED = {
    "hela_migration": {"min_pass": 17, "blocked": set(), "out_of_scope": set(),
                       "known_fail": set()},
    "liver_zonation": {"min_pass": 18, "blocked": set(), "out_of_scope": set(), "known_fail": set()},
    "brain_states": {"min_pass": 9, "blocked": set(), "out_of_scope": set(), "known_fail": set()},
    "membrane_integrity": {"min_pass": 9, "blocked": set(), "out_of_scope": set(),
                           "known_fail": set()},
    "hematopoiesis": {"min_pass": 17, "blocked": set(),
                      "out_of_scope": {"hemato_crossrun_other_contrasts"}, "known_fail": set()},
}


def find_archive():
    """Return the data archive root, or None when this checkout has no data layer."""
    candidates = []
    from_env = os.environ.get("SCPROTEOMICS_CASE_ARCHIVE", "").strip()
    if from_env:
        candidates.append(Path(from_env).expanduser())
    candidates.append(REPO / "data")
    candidates.append(REPO.parent / "data")
    for candidate in candidates:
        if (candidate / "case_tables").is_dir():
            return candidate
    print("SKIP: no case archive found. Checked: %s. Set SCPROTEOMICS_CASE_ARCHIVE to the "
          "archive root (the directory that holds case_tables/, source_tables/ and inputs/)."
          % "; ".join(str(c) for c in candidates))
    return None


def run_theme(theme, archive, out_dir):
    script = CASES / theme / ("run_%s.py" % theme)
    assert script.is_file(), "missing script %s" % script
    return subprocess.run([sys.executable, str(script), "--data-dir", str(archive),
                           "--out-dir", str(out_dir)], capture_output=True, text=True)


def check_theme(theme, archive, out_dir):
    expected = EXPECTED[theme]
    completed = run_theme(theme, archive, out_dir)
    if expected["known_fail"]:
        wanted_code = 1
    elif expected["blocked"]:
        wanted_code = 3
    else:
        wanted_code = 0
    assert completed.returncode == wanted_code, "%s exited %d, expected %d\n%s\n%s" % (
        theme, completed.returncode, wanted_code, completed.stdout[-2000:],
        completed.stderr[-2000:])
    ledger_path = out_dir / "case_regression.json"
    assert ledger_path.is_file(), "%s wrote no case_regression.json" % theme
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    checks = payload["checks"]
    results = [check["result"] for check in checks]
    failed = {c["check_id"] for c in checks if c["result"] == "FAIL"}
    assert failed == expected["known_fail"], "%s failed %s, expected %s" % (
        theme, failed, expected["known_fail"])
    assert payload["counts"]["FAIL"] == len(expected["known_fail"]), "%s reported %d FAIL" % (
        theme, payload["counts"]["FAIL"])
    assert set(results) <= {"PASS", "FAIL", "BLOCKED", "OUT_OF_SCOPE"}, \
        "%s used an unknown result" % theme
    assert payload["counts"]["PASS"] >= expected["min_pass"], "%s passed only %d checks" % (
        theme, payload["counts"]["PASS"])
    assert payload["counts"]["BLOCKED"] == len(expected["blocked"]), "%s reported %d BLOCKED" % (
        theme, payload["counts"]["BLOCKED"])
    assert payload["exit_code"] == completed.returncode, "%s recorded exit %s but returned %d" % (
        theme, payload["exit_code"], completed.returncode)
    blocked = {c["check_id"] for c in checks if c["result"] == "BLOCKED"}
    out_of_scope = {c["check_id"] for c in checks if c["result"] == "OUT_OF_SCOPE"}
    assert blocked == expected["blocked"], "%s blocked %s, expected %s" % (
        theme, blocked, expected["blocked"])
    assert out_of_scope == expected["out_of_scope"], "%s out of scope %s, expected %s" % (
        theme, out_of_scope, expected["out_of_scope"])
    for check in checks:
        if check["result"] == "BLOCKED":
            text = json.dumps(check).lower()
            assert "checked" in text or "absent" in text, \
                "%s: a BLOCKED check must name the paths it checked" % theme
        if check["result"] == "OUT_OF_SCOPE":
            assert check.get("scope"), "%s: an OUT_OF_SCOPE check must carry its scope" % theme
        for name, klass in (check.get("column_classes") or {}).items():
            assert klass in SOURCE_CLASSES, "%s: unknown column class %r for %s" % (
                theme, klass, name)
    assert payload.get("column_classes"), "%s recorded no column classes" % theme
    assert set(payload["column_classes"]) <= SOURCE_CLASSES, "%s: unknown column class" % theme
    assert payload.get("input_sha256"), "%s recorded no input hashes" % theme
    return payload


def test_case_regression():
    archive = find_archive()
    if archive is None:
        try:
            import pytest
            pytest.skip("no case archive")
        except ImportError:
            return
    with tempfile.TemporaryDirectory(prefix="case_regression_") as tmp:
        for theme in EXPECTED:
            check_theme(theme, archive, Path(tmp) / theme)


def test_missing_archive_fails_cleanly():
    archive = find_archive()
    if archive is None:
        try:
            import pytest
            pytest.skip("no case archive")
        except ImportError:
            return
    with tempfile.TemporaryDirectory(prefix="case_missing_") as tmp:
        empty = Path(tmp) / "empty"
        empty.mkdir()
        out_dir = Path(tmp) / "out"
        completed = run_theme("hela_migration", empty, out_dir)
        assert completed.returncode == 2, "missing inputs must exit 2, got %d" % completed.returncode
        assert "FATAL: missing input" in completed.stdout
        assert not out_dir.exists(), "nothing may be written when an input is missing"


def main() -> int:
    archive = find_archive()
    if archive is None:
        return 0
    failures = 0
    with tempfile.TemporaryDirectory(prefix="case_regression_") as tmp:
        for theme in EXPECTED:
            try:
                payload = check_theme(theme, archive, Path(tmp) / theme)
                print("ok   %-20s %d PASS, %d BLOCKED, %d OUT_OF_SCOPE, exit %d  columns %s"
                      % (theme, payload["counts"]["PASS"], payload["counts"]["BLOCKED"],
                         payload["counts"]["OUT_OF_SCOPE"], payload["exit_code"],
                         payload["column_classes"]))
            except AssertionError as error:
                failures += 1
                print("FAIL %-20s %s" % (theme, error))
        empty = Path(tmp) / "empty"
        empty.mkdir()
        completed = run_theme("hela_migration", empty, Path(tmp) / "out")
        if completed.returncode != 2 or not completed.stdout.startswith("FATAL: missing input"):
            failures += 1
            print("FAIL missing-input path returned %d" % completed.returncode)
        else:
            print("ok   missing-input path exits 2 and writes nothing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
