"""Negative and positive cases for the HeLa odds-ratio precision closure.

The HeLa migration case carries two different objects for the same statistic and this suite keeps
them apart by perturbation:

* case_tables/pispa_hela/hela_cluster_odds_ratio.tsv - the computed object. Its cell counts and
  its Woolf log-odds interval are compared at rtol 1e-7 with atol 0 and with no rounding
  allowance, so a changed count or a re-rounded endpoint has to fail hela_odds_ratio_computed;
  its one-decimal display columns are checked by hela_odds_ratio_manuscript_display.
* source_tables/hela_composition.tsv - the frozen table of the agent report. Its odds-ratio
  statistics row is a REPORTED transcription, checked by hela_odds_ratio_reported_transcription
  and separated from the strict row-by-row comparison; all its other rows and fields stay
  compared, so a changed proportion difference still has to fail hela_composition_rows.

Every case runs in its own scratch archive: the files listed in SCRATCH_INPUTS are copied out of
the read-only case archive into a temporary directory, the perturbation is applied there and the
case script is run against that copy. Nothing in the real archive is written.

The archive is located from SCPROTEOMICS_CASE_ARCHIVE, then <repository>/data, then
<repository>/../data, exactly like tests/test_case_regression.py. Without an archive - or without
the files listed in SCRATCH_INPUTS - the suite prints SKIP and exits 0, so a fresh checkout does
not fail for a missing data layer. It runs under pytest and as a plain script
(python tests/test_hela_odds_ratio.py), printing one line per case. Only the standard library is
needed here; the case script itself needs numpy and pandas.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:  # the plain-script entry point works without pytest installed
    import pytest
except ImportError:  # pragma: no cover - only on a checkout without pytest
    pytest = None

REPO = Path(__file__).resolve().parents[1]
CASE_SCRIPT = REPO / "reproduce" / "cases" / "hela_migration" / "run_hela_migration.py"

ODDS_TABLE = "case_tables/pispa_hela/hela_cluster_odds_ratio.tsv"
COMPOSITION_TABLE = "source_tables/hela_composition.tsv"
ODDS_ROW = "odds ratio (Cluster 1 vs Cluster 2)"
PROPORTION_ROW = "proportion difference (Cluster 1 vs Cluster 2)"
CASE_KEY = ("comparison", "Cluster 1 vs Cluster 2")

# The complete input set of the HeLa theme: the ten archived files the case script reads, followed
# by the computed odds-ratio table of the same theme (a stable delivered table, not an agent
# output). Nothing else of the archive is needed, and nothing else is copied.
SCRATCH_INPUTS = (
    "case_tables/pispa_hela/run_processed/ProQuant_Normalized.csv",
    "case_tables/pispa_hela/run_processed/SampleInfo_Filtered.csv",
    "case_tables/pispa_hela/run_processed/matrix_transform_record.json",
    "inputs/Nat_Commun_PiSPA_2024/ProteinQuant.csv",
    "source_tables/hela_composition.tsv",
    "source_tables/hela_pca_coordinates.tsv",
    "source_tables/hela_candidates.tsv",
    "source_tables/hela_volcano.tsv",
    "source_tables/tbc1d10b_cell_values.tsv",
    "case_tables/pispa_hela/pispa_candidate_mask_counts.tsv",
    ODDS_TABLE,
)

# The floor agreed for the HeLa theme this round. The pristine run currently emits 18 PASS checks
# (15 pre-existing ones, all PASS after the reported transcription was separated, plus the three
# odds-ratio checks this suite perturbs); the minimum is a floor, the zero FAIL count is not.
MIN_PASS = 17
NO_ARCHIVE = ("SKIP: no case archive found (SCPROTEOMICS_CASE_ARCHIVE, <repository>/data or "
              "<repository>/../data); this suite needs the archive that ships the HeLa case.")


# --------------------------------------------------------------------------- archive handling
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
    return None


def missing_inputs(archive):
    """Relative paths of the files this suite needs and the archive does not carry."""
    return [rel for rel in SCRATCH_INPUTS if not (archive / rel).is_file()]


def build_scratch(archive: Path, scratch: Path) -> None:
    """Copy the HeLa input set into a scratch archive with its directory structure."""
    for rel in SCRATCH_INPUTS:
        source = archive / Path(rel)
        target = scratch / Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


# --------------------------------------------------------------------------- table surgery
def read_lines(path: Path):
    """Lines of a file with their original line endings (decoded as UTF-8)."""
    return path.read_bytes().decode("utf-8").split("\n")


def write_lines(path: Path, lines) -> None:
    path.write_bytes("\n".join(lines).encode("utf-8"))


def column_index(header: str, name: str) -> int:
    columns = [cell.rstrip("\r") for cell in header.split("\t")]
    if name not in columns:
        raise AssertionError("column %r not found in %r" % (name, header))
    return columns.index(name)


def read_cell(path: Path, key_column: str, key_value: str, column: str) -> str:
    """The stored token of one cell, addressed by a key column of its row."""
    lines = read_lines(path)
    key_at = column_index(lines[0], key_column)
    column_at = column_index(lines[0], column)
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) > max(key_at, column_at) and cells[key_at].rstrip("\r") == key_value:
            return cells[column_at].rstrip("\r")
    raise AssertionError("row %r not found in %s" % (key_value, path))


def write_cell(path: Path, key_column: str, key_value: str, column: str, value: str) -> None:
    """Replace one stored token, leaving every other byte of the file alone."""
    lines = read_lines(path)
    key_at = column_index(lines[0], key_column)
    column_at = column_index(lines[0], column)
    for index in range(1, len(lines)):
        cells = lines[index].split("\t")
        if len(cells) > max(key_at, column_at) and cells[key_at].rstrip("\r") == key_value:
            carriage_return = "\r" if cells[column_at].endswith("\r") else ""
            cells[column_at] = value + carriage_return
            lines[index] = "\t".join(cells)
            write_lines(path, lines)
            return
    raise AssertionError("row %r not found in %s" % (key_value, path))


# --------------------------------------------------------------------------- perturbations
def perturb_pristine(scratch: Path) -> str:
    """No change: the archive is used as delivered."""
    return "none"


def perturb_count(scratch: Path) -> str:
    """N1: one cell count of the computed table changes (7 -> 8)."""
    write_cell(scratch / ODDS_TABLE, CASE_KEY[0], CASE_KEY[1], "group_a_control_cells", "8")
    return "computed table group_a_control_cells 7 -> 8"


def perturb_ci_high_two_decimals(scratch: Path) -> str:
    """N2: the upper endpoint becomes its own two-decimal rendering (91.19)."""
    stored = read_cell(scratch / ODDS_TABLE, CASE_KEY[0], CASE_KEY[1], "ci_high")
    rendered = "%.2f" % float(stored)
    write_cell(scratch / ODDS_TABLE, CASE_KEY[0], CASE_KEY[1], "ci_high", rendered)
    return "computed table ci_high %s -> %s (two-decimal rendering)" % (stored, rendered)


def perturb_display(scratch: Path) -> str:
    """N3: the one-decimal value the manuscript prints changes (91.2 -> 91.3)."""
    stored = read_cell(scratch / ODDS_TABLE, CASE_KEY[0], CASE_KEY[1],
                       "manuscript_ci_high_display")
    write_cell(scratch / ODDS_TABLE, CASE_KEY[0], CASE_KEY[1], "manuscript_ci_high_display",
               "91.3")
    return "computed table manuscript_ci_high_display %s -> 91.3" % stored


def perturb_other_composition_field(scratch: Path) -> str:
    """N4: a different field of the frozen table changes (0.672 -> 0.700)."""
    path = scratch / COMPOSITION_TABLE
    stored = read_cell(path, "element", PROPORTION_ROW, "value")
    write_cell(path, "element", PROPORTION_ROW, "value", "0.700")
    return "frozen composition value %s -> 0.700" % stored


def perturb_restored(scratch: Path) -> str:
    """N5: a fresh pristine copy of the same archive."""
    return "none (archive rebuilt from the pristine copy)"


CASES = (
    {"case_id": "P", "perturb": perturb_pristine, "exit_code": 0, "failing": set()},
    {"case_id": "N1", "perturb": perturb_count, "exit_code": 1,
     "failing": {"hela_odds_ratio_computed"}},
    {"case_id": "N2", "perturb": perturb_ci_high_two_decimals, "exit_code": 1,
     "failing": {"hela_odds_ratio_computed"}},
    {"case_id": "N3", "perturb": perturb_display, "exit_code": 1,
     "failing": {"hela_odds_ratio_manuscript_display"}},
    {"case_id": "N4", "perturb": perturb_other_composition_field, "exit_code": 1,
     "failing": {"hela_composition_rows"}},
    {"case_id": "N5", "perturb": perturb_restored, "exit_code": 0, "failing": set()},
)


# --------------------------------------------------------------------------- case runner
class CaseResult:
    """What one case observed, and whether it matches the pre-registered expectation."""

    def __init__(self, case_id, perturbation, exit_code, expected_exit, failing, expected_failing,
                 counts, command):
        self.case_id = case_id
        self.perturbation = perturbation
        self.exit_code = exit_code
        self.expected_exit = expected_exit
        self.failing = failing
        self.expected_failing = expected_failing
        self.counts = counts
        self.command = command
        self.problems = []
        if exit_code != expected_exit:
            self.problems.append("exit code %s, expected %s" % (exit_code, expected_exit))
        if failing != expected_failing:
            self.problems.append("failing checks %s, expected %s"
                                 % (sorted(failing) or "none", sorted(expected_failing) or "none"))
        if counts:
            if counts.get("FAIL", 0) != len(expected_failing):
                self.problems.append("ledger FAIL count %s, expected %d"
                                     % (counts.get("FAIL"), len(expected_failing)))
            if counts.get("BLOCKED", 0):
                self.problems.append("ledger reported %s BLOCKED" % counts.get("BLOCKED"))
            if expected_exit == 0 and counts.get("PASS", 0) < MIN_PASS:
                self.problems.append("only %s PASS checks, the registered floor is %d"
                                     % (counts.get("PASS"), MIN_PASS))

    @property
    def ok(self):
        return not self.problems

    def describe(self):
        counts = self.counts or {}
        return ("%s | perturbation: %s | exit %s (expected %s) | FAIL %s (expected %s) | "
                "PASS %s | %s"
                % (self.case_id, self.perturbation, self.exit_code, self.expected_exit,
                   sorted(self.failing) or "none", sorted(self.expected_failing) or "none",
                   counts.get("PASS", "?"),
                   "ok" if self.ok else "; ".join(self.problems)))


def run_case(case, archive: Path) -> CaseResult:
    """Run one case in its own scratch archive and read the ledger it wrote."""
    with tempfile.TemporaryDirectory(prefix="hela_odds_ratio_%s_" % case["case_id"]) as tmp:
        scratch = Path(tmp) / "archive"
        out_dir = Path(tmp) / "out"
        build_scratch(archive, scratch)
        perturbation = case["perturb"](scratch)
        command = [sys.executable, str(CASE_SCRIPT), "--data-dir", str(scratch),
                   "--out-dir", str(out_dir)]
        completed = subprocess.run(command, capture_output=True, text=True)
        given = ['"%s"' % part if " " in part else part for part in command]
        ledger_path = out_dir / "case_regression.json"
        if not ledger_path.is_file():
            raise AssertionError("case %s wrote no case_regression.json (exit %d)\n%s\n%s"
                                 % (case["case_id"], completed.returncode, completed.stdout[-2000:],
                                    completed.stderr[-2000:]))
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        failing = {check["check_id"] for check in payload["checks"] if check["result"] == "FAIL"}
        return CaseResult(case["case_id"], perturbation, completed.returncode,
                          case["exit_code"], failing, case["failing"], payload.get("counts"),
                          " ".join(given))


def archive_or_skip():
    """The archive to use, or None - with a printed or raised SKIP - when there is none."""
    archive = find_archive()
    message = NO_ARCHIVE
    if archive is not None:
        missing = missing_inputs(archive)
        if missing:
            message = ("SKIP: the case archive %s does not carry the files this suite copies "
                       "(%s); it predates the HeLa odds-ratio table of this round"
                       % (archive, ", ".join(missing)))
            archive = None
    if archive is None:
        if pytest is not None:
            pytest.skip(message)
        print(message)
        return None
    return archive


def check_case(case):
    """Shared body of the pytest cases and of the plain-script loop."""
    archive = archive_or_skip()
    if archive is None:
        return None
    result = run_case(case, archive)
    print(result.describe())
    assert result.ok, result.describe()
    return result


def test_case_p_pristine():
    check_case(CASES[0])


def test_case_n1_cell_count_changed():
    check_case(CASES[1])


def test_case_n2_upper_endpoint_two_decimals():
    check_case(CASES[2])


def test_case_n3_manuscript_display_changed():
    check_case(CASES[3])


def test_case_n4_other_composition_field_changed():
    check_case(CASES[4])


def test_case_n5_restored():
    check_case(CASES[5])


def main() -> int:
    archive = find_archive()
    if archive is None:
        print(NO_ARCHIVE)
        return 0
    missing = missing_inputs(archive)
    if missing:
        print("SKIP: the case archive %s does not carry the files this suite copies (%s); it "
              "predates the HeLa odds-ratio table of this round" % (archive, ", ".join(missing)))
        return 0
    print("archive %s" % archive)
    failures = 0
    for case in CASES:
        try:
            result = run_case(case, archive)
        except Exception as error:  # a scratch build or ledger problem is a suite failure
            failures += 1
            print("FAIL %-3s %s" % (case["case_id"], error))
            continue
        print("%s %s" % ("ok  " if result.ok else "FAIL", result.describe()))
        if not result.ok:
            failures += 1
    print("%d of %d cases reproduced the pre-registered outcome"
          % (len(CASES) - failures, len(CASES)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
