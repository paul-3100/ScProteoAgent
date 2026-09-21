"""Determinism test for the paired percentile-bootstrap intervals.

The interval code is only useful if the same frozen inputs always produce the same endpoints.
This test builds a synthetic 11-study x 8-method rule table, runs
``reproduce/intervals/recompute_paired_ci.py`` twice into two different output directories, and
requires the four produced files to be byte-identical. It also checks the parity fields of
``environment.json``, the index-matrix hash, and the ordering property
``ci_low <= mean_difference <= ci_high``.

When a real interval data directory is available (``--data-dir``, ``data/intervals`` or
``../data/intervals``) the frozen-endpoint check is added by running ``ci_verify.py``, which
compares against the delivered values. Without it, the synthetic run still verifies determinism
and is reported as such.

Usage: python tests/test_pair_ci_determinism.py [--work-dir DIR] [--data-dir DIR] [--python EXE]
Exit codes: 0 pass (or determinism-only), 1 failure, 3 environment cannot run the test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INTERVALS = REPO / "reproduce" / "intervals"
RECOMPUTE = INTERVALS / "recompute_paired_ci.py"
VERIFY = INTERVALS / "ci_verify.py"
METHODS = ["scProteoAgent", "Hermes", "Codex", "Claude Code", "CellVoyager", "Biomni",
           "BioAgent", "SpatialAgent"]
STUDIES = ["Study_%02d" % index for index in range(1, 12)]
OUTPUT_FILES = ("paired_ci_recomputed.tsv", "paired_ci_old_new.tsv", "paired_ci_indices.npy",
                "environment.json")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_fixture(root: Path) -> Path:
    data_dir = root / "intervals_fixture"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = ["dataset_id\tstudy\tstudy_display\tmethod\tmode\trule_total"
            "\tscientific_coverage\tcurrent_matrix_evidence\tboundary_compliance"
            "\tartifact_reproducibility\tchinese_report_quality\tpenalty_credit"]
    values: dict[tuple[str, str], float] = {}
    for study_index, study in enumerate(STUDIES):
        for method_index, method in enumerate(METHODS):
            # deterministic synthetic scores; scProteoAgent (method 0) is the highest
            value = 50.0 + study_index + (7 - method_index) * 2.0
            values[(study, method)] = value
            rows.append("%s\t%s\t%s\t%s\twith_run_dir\t%.2f\t0\t0\t0\t0\t0\t0"
                        % (study, study.split("_")[0], study, method, value))
    write(data_dir / "benchmark_rule_scores.tsv", "\n".join(rows) + "\n")
    means = ["comparator\tmean_difference\tci_low\tci_high\tinterval_source\tcomputed_mean"]
    for index, method in enumerate(METHODS[1:]):
        mean = sum(values[(study, "scProteoAgent")] - values[(study, method)] for study in STUDIES) / len(STUDIES)
        means.append("%s\t%.6f\t%.6f\t%.6f\tfixture\t%.6f" % (method, mean, mean - 2, mean + 2, mean))
    write(data_dir / "benchmark_paired_difference_means.tsv", "\n".join(means) + "\n")
    return data_dir


def resolve_data_dir(explicit: Path | None) -> Path | None:
    """Find a real interval archive; the rule table is accepted under either of its two names."""
    rule_names = ("benchmark_rule_scores.tsv", "rule_scores.tsv")
    anchors = ["", "tables", "../source_tables", "../intervals"]
    candidates = []
    if explicit:
        for anchor in anchors:
            candidates.append(explicit if not anchor else (explicit / anchor))
    env_value = os.getenv("SCPROTEOMICS_INTERVALS_DIR", "").strip()
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(REPO / "data" / "intervals")
    candidates.append(REPO.parent / "data" / "intervals")
    candidates.append(REPO / "data" / "source_tables")
    candidates.append(REPO.parent / "data" / "source_tables")
    for candidate in candidates:
        if any((candidate / name).is_file() for name in rule_names):
            return candidate.resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired interval determinism test.")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    if not RECOMPUTE.is_file() or not VERIFY.is_file():
        print("BLOCKED: interval scripts missing under %s" % INTERVALS)
        return 3
    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="pair_ci_"))
    work_root.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture(work_root)
    print("work dir: %s" % work_root)
    failures: list[str] = []

    for tag in ("run_a", "run_b"):
        completed = subprocess.run(
            [args.python, str(RECOMPUTE), "--data-dir", str(fixture),
             "--out-dir", str(work_root / tag)],
            capture_output=True, text=True, cwd=str(INTERVALS),
            env=dict(os.environ, PYTHONPATH=""))
        print("%s exit=%d %s" % (tag, completed.returncode, completed.stdout.strip().splitlines()[0]
                                  if completed.stdout.strip() else ""))
        if completed.returncode != 0:
            print(completed.stderr)
            failures.append("%s exited %d" % (tag, completed.returncode))
    if failures:
        for failure in failures:
            print("FAIL %s" % failure)
        return 1

    for name in OUTPUT_FILES:
        left = (work_root / "run_a" / name).read_bytes()
        right = (work_root / "run_b" / name).read_bytes()
        if left != right:
            failures.append("%s differs between the two runs" % name)
        else:
            print("identical  %s  (%d bytes)" % (name, len(left)))

    environment = json.loads((work_root / "run_a" / "environment.json").read_text(encoding="utf-8"))
    parity = {"seed": environment["seed"], "n_resamples": environment["n_resamples"],
              "n_studies": environment["n_studies"], "n_comparisons": environment["n_comparisons"],
              "quantile_method": environment["quantile_method"],
              "generator": environment["generator"], "index_shape": environment["index_shape"]}
    print("parity: %s" % parity)
    if parity["seed"] != 20260919:
        failures.append("seed changed: %s" % parity["seed"])
    if parity["n_resamples"] != 20000 or parity["n_studies"] != 11 or parity["n_comparisons"] != 7:
        failures.append("resample/study/comparison counts changed: %s" % parity)
    if parity["quantile_method"] != "linear" or list(parity["index_shape"]) != [20000, 11]:
        failures.append("quantile method or index shape changed: %s" % parity)

    # environment.json records the hash of the raw array bytes, not of the .npy container, so the
    # hash is taken through numpy in a subprocess (this test itself stays numpy-free).
    index_path = work_root / "run_a" / "paired_ci_indices.npy"
    probe = subprocess.run(
        [args.python, "-c",
         "import hashlib, sys, numpy; "
         "print(hashlib.sha256(numpy.load(sys.argv[1]).tobytes()).hexdigest())", str(index_path)],
        capture_output=True, text=True)
    if probe.returncode != 0:
        print("BLOCKED: numpy is required to hash the index matrix")
        return 3
    index_sha = probe.stdout.strip()
    container_bytes = index_path.read_bytes()
    container_sha = hashlib.sha256(container_bytes).hexdigest()
    if index_sha != environment["index_sha256"]:
        failures.append("index hash does not match the recorded value")
    else:
        print("index array sha256 %s (container sha256 %s)"
              % (index_sha, container_sha[:16]))

    table = (work_root / "run_a" / "paired_ci_recomputed.tsv").read_text(encoding="utf-8").strip().splitlines()
    if len(table) != 8:
        failures.append("expected header + 7 comparisons, found %d lines" % len(table))
    for line in table[1:]:
        cells = line.split("\t")
        mean, low, high = float(cells[3]), float(cells[4]), float(cells[5])
        if not low <= mean <= high:
            failures.append("interval does not bracket its mean for %s" % cells[1])

    real_dirs = resolve_data_dir(args.data_dir)
    if real_dirs is None:
        print("NOTE: no interval data archive found; frozen-endpoint verification NOT_RUN")
    else:
        verify_out = work_root / "real"
        completed = subprocess.run(
            [args.python, str(RECOMPUTE), "--data-dir", str(real_dirs),
             "--out-dir", str(verify_out)], capture_output=True, text=True, cwd=str(INTERVALS),
            env=dict(os.environ, PYTHONPATH=""))
        completed = subprocess.run(
            [args.python, str(VERIFY), "--data-dir", str(real_dirs), "--out-dir", str(verify_out),
             "--verify-dir", str(verify_out), "--tag", "github_test"],
            capture_output=True, text=True, cwd=str(INTERVALS), env=dict(os.environ, PYTHONPATH=""))
        print("ci_verify exit=%d %s" % (completed.returncode, completed.stdout.strip()))
        if completed.returncode != 0:
            print(completed.stderr)
            failures.append("ci_verify failed against %s" % real_dirs)

    for failure in failures:
        print("FAIL %s" % failure)
    print("RESULT: %s" % ("FAIL" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
