"""Test the offline input validator and the packaged examples.

Checks:
  * ``examples/minimal_demo`` validates as PASS with exit code 0;
  * a dataset missing a required file fails with exit code 2;
  * a transposed matrix (samples in rows) fails with exit code 2;
  * duplicated sample IDs fail with exit code 2;
  * the validator itself imports only the standard library and contains no networking call.

Usage: python tests/test_example_input_validation.py [--work-dir DIR] [--python EXE]
Exit codes: 0 pass, 1 failure, 3 environment cannot run the test.
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VALIDATOR = REPO / "scripts" / "validate_dataset_inputs.py"
DEMO = REPO / "examples" / "minimal_demo"
NETWORK_MODULES = {"socket", "urllib", "http", "requests", "ftplib", "telnetlib", "asyncio"}
STDLIB_ALLOWED = {"argparse", "csv", "json", "sys", "collections", "pathlib", "__future__",
                  "typing", "re", "math", "io", "os", "textwrap", "dataclasses", "itertools"}


def run_validator(python: str, *targets: Path, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([python, str(VALIDATOR), *[str(t) for t in targets]],
                          capture_output=True, text=True, cwd=str(cwd),
                          env=dict(os.environ, PYTHONPATH=""))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validator and example-input test.")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    if not VALIDATOR.is_file() or not DEMO.is_dir():
        print("BLOCKED: %s or %s missing" % (VALIDATOR, DEMO))
        return 3
    failures: list[str] = []

    tree = ast.parse(VALIDATOR.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    extra = imported - STDLIB_ALLOWED
    network = imported & NETWORK_MODULES
    print("validator imports: %s" % ", ".join(sorted(imported)))
    if extra:
        failures.append("validator imports non-standard modules: %s" % ", ".join(sorted(extra)))
    if network:
        failures.append("validator imports networking modules: %s" % ", ".join(sorted(network)))

    demo = run_validator(args.python, DEMO, cwd=REPO)
    print("minimal_demo exit=%d" % demo.returncode)
    print(demo.stdout.strip())
    if demo.returncode != 0:
        failures.append("minimal_demo returned exit %d (expected 0)" % demo.returncode)
    if "result: PASS" not in demo.stdout:
        failures.append("minimal_demo did not report PASS")
    if "Donor(2)" not in demo.stdout:
        failures.append("minimal_demo did not report the Donor experimental-unit candidate")

    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="validator_"))
    work_root.mkdir(parents=True, exist_ok=True)
    print("work dir: %s" % work_root)

    missing = work_root / "missing_file"
    for name in ("ProteinQuant.csv", "SampleInfo.csv"):
        write(missing / name, (DEMO / name).read_text(encoding="utf-8"))
    result = run_validator(args.python, missing, cwd=REPO)
    print("missing user_input.txt exit=%d : %s" % (result.returncode, result.stdout.strip().splitlines()[-2:]))
    if result.returncode != 2:
        failures.append("missing required file returned exit %d (expected 2)" % result.returncode)

    transposed = work_root / "transposed"
    write(transposed / "ProteinQuant.csv",
          "PG.ProteinGroups,PG.Genes,P1,P2,P3\nc1,1,1,2,3\nc2,2,2,3,4\nc3,3,3,4,5\nc4,4,4,5,6\n")
    write(transposed / "SampleInfo.csv", "FileName,Cluster\nc1,A\nc2,A\nc3,B\nc4,B\n")
    write(transposed / "user_input.txt", "Compare A with B.\n")
    result = run_validator(args.python, transposed, cwd=REPO)
    print("transposed matrix exit=%d" % result.returncode)
    if result.returncode != 2 or "matrix looks transposed" not in result.stdout:
        failures.append("transposed matrix was not rejected (exit %d)" % result.returncode)

    duplicates = work_root / "duplicate_ids"
    write(duplicates / "ProteinQuant.csv", (DEMO / "ProteinQuant.csv").read_text(encoding="utf-8"))
    write(duplicates / "SampleInfo.csv", "FileName,Cluster\nCTRL_1,A\nCTRL_1,A\nCTRL_2,B\nCTRL_3,B\n")
    write(duplicates / "user_input.txt", "Compare A with B.\n")
    result = run_validator(args.python, duplicates, cwd=REPO)
    print("duplicate sample IDs exit=%d" % result.returncode)
    if result.returncode != 2 or "duplicate sample IDs" not in result.stdout:
        failures.append("duplicate sample IDs were not rejected (exit %d)" % result.returncode)

    for failure in failures:
        print("FAIL %s" % failure)
    print("RESULT: %s" % ("FAIL" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
