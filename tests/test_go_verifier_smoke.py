"""Smoke test for the GO background verifier.

Two modes:

**Full verification** - when a GO data archive is resolvable (``--data-dir``,
``SCPROTEOMICS_GO_DIR``, ``data/go_background`` or ``../data/go_background``) and the GMT files
are present, ``c4_verify.py`` is run for real and must report ``verdict: PASS`` with the frozen
family counts.

**Entry-point smoke** - without an archive, the test checks that the four scripts fail closed:
running each one must exit non-zero with the documented ``FATAL`` message about the missing
archive or GMT files, and none of the five ``go_background`` modules may import a networking
module. That mode is reported as SMOKE_ONLY.

Usage: python tests/test_go_verifier_smoke.py [--data-dir DIR] [--gmt-dir DIR] [--python EXE]
Exit codes: 0 pass, 1 failure, 3 environment cannot run the test.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GO_DIR = REPO / "reproduce" / "go_background"
SCRIPTS = ("c1_go_datasets.py", "c2_go_compute.py", "c3_display_compare.py", "c4_verify.py")
NETWORK_MODULES = {"socket", "urllib", "http", "requests", "ftplib", "telnetlib", "asyncio"}


def resolve_data_dir(explicit: Path | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_value = os.getenv("SCPROTEOMICS_GO_DIR", "").strip()
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(REPO / "data" / "go_background")
    candidates.append(REPO.parent / "data" / "go_background")
    for candidate in candidates:
        if (candidate / "tables" / "go_results_all.tsv").is_file():
            return candidate
    return None


def gmt_complete(data_dir: Path, explicit: Path | None) -> Path | None:
    candidates = [c for c in (explicit, data_dir / "gmt", data_dir) if c]
    for candidate in candidates:
        if all((candidate / name).is_file()
               for name in ("GO_BP.symbols.gmt", "GO_CC.symbols.gmt", "GO_MF.symbols.gmt")):
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GO verifier smoke test.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--gmt-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    failures: list[str] = []
    for name in (*SCRIPTS, "gb_env.py"):
        path = GO_DIR / name
        if not path.is_file():
            print("BLOCKED: %s missing" % path)
            return 3
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        network = imported & NETWORK_MODULES
        if network:
            failures.append("%s imports networking modules: %s" % (name, sorted(network)))
    print("checked %d go_background modules for networking imports" % (len(SCRIPTS) + 1))

    data_dir = resolve_data_dir(args.data_dir)
    if data_dir is None:
        empty = Path(tempfile.mkdtemp(prefix="go_smoke_"))
        print("NOTE: no GO data archive found; running the fail-closed smoke instead")
        for name in SCRIPTS:
            completed = subprocess.run(
                [args.python, str(GO_DIR / name), "--data-dir", str(empty)],
                capture_output=True, text=True, cwd=str(GO_DIR),
                env=dict(os.environ, PYTHONPATH=""))
            message = (completed.stderr or completed.stdout).strip().splitlines()
            tail = message[-1] if message else ""
            print("%s exit=%d %s" % (name, completed.returncode, tail[:120]))
            if completed.returncode == 0:
                failures.append("%s succeeded without a data archive" % name)
            elif "FATAL" not in tail:
                failures.append("%s failed without the documented FATAL message" % name)
        for failure in failures:
            print("FAIL %s" % failure)
        print("RESULT: %s (SMOKE_ONLY: full verification NOT_RUN)"
              % ("FAIL" if failures else "PASS"))
        return 1 if failures else 0

    gmt_dir = gmt_complete(data_dir, args.gmt_dir)
    if gmt_dir is None:
        print("BLOCKED: data archive %s has no complete GMT directory; full verification "
              "NOT_RUN (see reproduce/go_background/RESOURCE_ACQUISITION.md)" % data_dir)
        return 3

    command = [args.python, str(GO_DIR / "c4_verify.py"), "--data-dir", str(data_dir),
               "--gmt-dir", str(gmt_dir)]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(GO_DIR),
                               env=dict(os.environ, PYTHONPATH=""))
    print("c4_verify exit=%d" % completed.returncode)
    print(completed.stdout.strip())
    if completed.returncode != 0:
        print(completed.stderr)
        failures.append("c4_verify exited %d" % completed.returncode)
    else:
        try:
            summary = json.loads(completed.stdout.strip())
        except json.JSONDecodeError:
            failures.append("c4_verify output was not JSON")
            summary = {}
        if summary.get("verdict") != "PASS":
            failures.append("verdict is %r" % summary.get("verdict"))
        if summary.get("families_checked") != 12:
            failures.append("expected 12 families, got %r" % summary.get("families_checked"))
        if summary.get("rows_over_q_tol") not in (0, None):
            failures.append("q tolerance exceeded: %r" % summary.get("rows_over_q_tol"))

    for failure in failures:
        print("FAIL %s" % failure)
    print("RESULT: %s" % ("FAIL" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
