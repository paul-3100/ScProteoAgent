"""Contract test for the frozen HuRunWen rule scorer.

The scorer is an artefact, not a library: its identity is a hash, its behaviour is a rule
layer, and both must survive packaging. This test:

1. checks the recorded sha256 of ``reproduce/scoring/score_full_run.py`` and
   ``reproduce/scoring/guarded_launcher.py``;
2. runs the scorer end to end on a small fixture through the network kill-switch launcher with
   ``--skip-llm``, with every credential variable removed from the child environment;
3. checks the rule contract: six dimensions summing to 100, ``rule_total`` equal to their sum,
   ``scoring_mode == rule_only_fallback``, a dataset that is not executable when
   ``grading_standard.txt`` is absent, and identical output on a second run.

It needs no third-party package and no network. It does **not** compare the score against the
published table; that comparison is the replay driver's job
(``reproduce/scoring/run_replay.py --expected-tsv ...``).

Usage: python tests/test_frozen_scorer_contract.py [--work-dir DIR] [--python EXE]
Exit codes: 0 pass, 1 contract failure, 3 environment cannot run the test.
"""
from __future__ import annotations

import argparse
import hashlib
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
FROZEN_SCORER_SHA256 = "2B37C1835D397FBF132BAD9DEB8900220BCD3ABCD299ECEE8112CC0B3BEEA74B"
FROZEN_LAUNCHER_SHA256 = "56A38AA5533AC50C48F6D6673D26167EE2A7AE9345B3E5C885078D43262CE9D7"
EXPECTED_DIMENSIONS = {"scientific_coverage": 35, "current_matrix_evidence": 20,
                       "boundary_compliance": 15, "artifact_reproducibility": 15,
                       "chinese_report_quality": 10, "penalty_credit": 5}
CREDENTIAL_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "SCORE_MODEL", "GOOGLE_API_KEY",
                   "GOOGLE_BASE_URL", "GOOGLE_CX", "SERPAPI_KEY", "ENTREZ_EMAIL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "examples" / "DEMO_Study"
    write(dataset / "ProteinQuant.csv",
          "PG.ProteinGroups,PG.Genes,c1,c2,c3,c4\nP1,GAPDH,10,11,12,13\nP2,ACTB,20,21,22,23\n")
    write(dataset / "SampleInfo.csv", "FileName,Cluster\nc1,A\nc2,A\nc3,B\nc4,B\n")
    write(dataset / "user_input.txt", "Compare group B with group A.\n")
    write(dataset / "ground_truth.txt", "GAPDH increases; groups differ by cluster.\n")
    write(dataset / "grading_standard.txt", "The report must compare A and B and state limits.\n")

    run_dir = root / "runs" / "Agent-local-DEMO_Study"
    write(run_dir / "run_status.json", json.dumps({"run_status": "completed"}))
    write(run_dir / "report.md",
          "## 一、执行摘要\n\n本报告比较 A 组与 B 组，使用 current_matrix 证据。\n\n"
          "## 二、核心证据首页\n\nlogFC 与 adj.P.Val 已列出，差异为 1.2 倍，n = 4。\n\n"
          "## 三、主要发现\n\nGAPDH 上调；结果表见 tables/DE.csv。\n\n"
          "## 四、证据边界\n\n未使用 offline_enrichment，未做 external_annotation。\n\n"
          "## 五、可复核证据附录\n\nfigures/plot.png\n")
    return dataset, root / "runs"


def clean_environment() -> dict:
    env = dict(os.environ)
    for name in CREDENTIAL_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = ""
    return env


def run_scorer(python: str, dataset: Path, runs_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [python, str(LAUNCHER), str(SCORER),
               "--examples-root", str(dataset.parent), "--runs-root", str(runs_root),
               "--out-json", str(out_dir / "score.json"), "--out-tsv", str(out_dir / "score.tsv"),
               "--out-md", str(out_dir / "score.md"),
               "--details-dir", str(out_dir / "details"), "--skip-llm"]
    completed = subprocess.run(command, capture_output=True, text=True, env=clean_environment(),
                               cwd=str(out_dir))
    if completed.returncode != 0:
        raise SystemExit("scorer exited %d\n%s\n%s"
                         % (completed.returncode, completed.stdout, completed.stderr))
    return json.loads((out_dir / "score.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen scorer contract test.")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    failures: list[str] = []
    if not SCORER.is_file() or not LAUNCHER.is_file():
        print("BLOCKED: %s or %s missing" % (SCORER, LAUNCHER))
        return 3
    for path, expected in ((SCORER, FROZEN_SCORER_SHA256), (LAUNCHER, FROZEN_LAUNCHER_SHA256)):
        actual = sha256(path)
        state = "OK" if actual == expected else "MISMATCH"
        print("%-6s sha256 %s  %s" % (state, path.name, actual))
        if actual != expected:
            failures.append("%s hash %s != frozen %s" % (path.name, actual, expected))

    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="scorer_contract_"))
    work_root.mkdir(parents=True, exist_ok=True)
    dataset, runs_root = build_fixture(work_root)
    first = run_scorer(args.python, dataset, runs_root, work_root / "out1")
    second = run_scorer(args.python, dataset, runs_root, work_root / "out2")
    print("work dir: %s" % work_root)

    row = first["rows"][0]
    dimensions = row["rule_result"]["dimensions"]
    total = row["rule_total"]
    print("scoring_system=%s dataset_count=%d scoring_mode=%s"
          % (first["scoring_system"], first["dataset_count"], row["scoring_mode"]))
    print("rule_total=%s dimensions=%s" % (total, dimensions))
    if first["score_dimensions"] != EXPECTED_DIMENSIONS:
        failures.append("score dimension maxima changed: %s" % first["score_dimensions"])
    if row["scoring_mode"] != "rule_only_fallback":
        failures.append("--skip-llm did not produce rule_only_fallback (got %s)" % row["scoring_mode"])
    if round(sum(dimensions.values()), 2) != total:
        failures.append("rule_total %s != sum(dimensions) %s" % (total, round(sum(dimensions.values()), 2)))
    if not 0.0 <= total <= 100.0:
        failures.append("rule_total outside 0..100: %s" % total)
    if row["rule_result"]["checks"]["has_report"] is not True:
        failures.append("fixture report was not read")
    if second["rows"][0]["rule_total"] != total:
        failures.append("second run produced a different rule_total: %s vs %s"
                        % (second["rows"][0]["rule_total"], total))

    # a dataset without grading_standard.txt must not be treated as executable
    stripped = work_root / "examples" / "NO_STANDARD_Study"
    shutil.copytree(dataset, stripped)
    (stripped / "grading_standard.txt").unlink()
    stripped_run = run_scorer(args.python, stripped, runs_root, work_root / "out3")
    scored = sorted(row["dataset"] for row in stripped_run["rows"])
    if "NO_STANDARD_Study" in scored:
        failures.append("dataset without grading_standard.txt was scored anyway: %s" % scored)
    else:
        print("dataset without grading_standard.txt correctly skipped (scored: %s)" % scored)

    for failure in failures:
        print("FAIL %s" % failure)
    print("RESULT: %s" % ("FAIL" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
