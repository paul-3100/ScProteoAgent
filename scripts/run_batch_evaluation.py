import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from path_config import resolve_examples_root
DEFAULT_RUNS_ROOT = PROJECT_DIR / "runs_v3" / "manual_batch_runs"
REQUIRED_DATASET_FILES = [
    "ProteinQuant.csv",
    "SampleInfo.csv",
    "user_input.txt",
    "ground_truth.txt",
    "grading_standard.txt",
]
ANALYSIS_DATASET_FILES = [
    "ProteinQuant.csv",
    "SampleInfo.csv",
    "user_input.txt",
]

RUN_NAME_ALIASES = {
    "Nat_Methods_iPSC_2025": "iPSC",
    "Nat_Commun_PiSPA_2024": "PiSPA",
    "Nat_Biotech_Brain_2026": "Brain",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _run_input_folder(run_dir: Path) -> str:
    events = run_dir / "run_events.jsonl"
    if not events.exists():
        return ""
    try:
        with events.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                return str(event.get("input_folder", ""))
    except Exception:
        return ""
    return ""


def latest_run_for_name(runs_root: Path, run_name: str, dataset_name: str = ""):
    candidates = []
    for path in runs_root.iterdir() if runs_root.exists() else []:
        if not path.is_dir():
            continue
        if (
            run_name in path.name
            or (dataset_name and dataset_name in path.name)
            or (dataset_name and dataset_name in _run_input_folder(path))
        ):
            candidates.append(path)
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def is_executable_dataset(folder: Path) -> bool:
    return folder.is_dir() and all((folder / name).exists() for name in REQUIRED_DATASET_FILES)


def is_analysis_dataset(folder: Path) -> bool:
    return folder.is_dir() and all((folder / name).exists() for name in ANALYSIS_DATASET_FILES)


def run_name_for_dataset(prefix: str, dataset_name: str) -> str:
    suffix = RUN_NAME_ALIASES.get(dataset_name, dataset_name)
    return f"{prefix}_{suffix}"


def run_command(args, cwd: Path, timeout: int | None = None):
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return {
        "returncode": proc.returncode,
        "duration_sec": round(time.perf_counter() - started, 3),
        "stdout_tail": proc.stdout[-6000:],
    }


def run_one(
    dataset_dir: Path,
    runs_root: Path,
    run_name: str,
    timeout: int,
    retry_once: bool,
    publication_mode: bool,
    audit_hidden_file_access: bool,
    evaluation_dataset_dir: Path | None = None,
):
    attempts = []
    for attempt in range(1, 3 if retry_once else 2):
        name = run_name if attempt == 1 else f"{run_name}_retry{attempt}"
        cmd = [
            sys.executable,
            "main_agent.py",
            "-i",
            str(dataset_dir),
            "-rf",
            str(runs_root),
            "-n",
            name,
        ]
        if publication_mode:
            cmd.append("--publication-mode")
        result = run_command(cmd, PROJECT_DIR, timeout=timeout)
        run_dir = latest_run_for_name(runs_root, name, dataset_dir.name)
        status = ""
        if run_dir and (run_dir / "run_status.json").exists():
            try:
                status = json.loads(read_text(run_dir / "run_status.json")).get("run_status", "")
            except Exception:
                status = ""
        attempts.append({"attempt": attempt, "name": name, "run_dir": str(run_dir) if run_dir else "", "status": status, **result})
        if result["returncode"] == 0 and status == "completed":
            break

    final = attempts[-1]
    run_dir = Path(final["run_dir"]) if final.get("run_dir") else None
    validation = {}
    audit = {}
    if run_dir and run_dir.exists():
        validation_path = run_dir / "report_validation.json"
        validation_result = run_command(
            [sys.executable, "scripts/validate_scientific_report.py", str(run_dir), "--json-out", str(validation_path)],
            PROJECT_DIR,
            timeout=180,
        )
        validation = {
            "returncode": validation_result["returncode"],
            "json_path": str(validation_path),
        }
        try:
            validation.update(json.loads(read_text(validation_path)))
            (run_dir / "validation.json").write_text(
                json.dumps(validation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            validation["status"] = "invalid"
        if audit_hidden_file_access:
            audit_path = run_dir / "hidden_file_access_audit.json"
            audit_md = run_dir / "hidden_file_access_audit.md"
            audit_cmd = [
                sys.executable,
                "scripts/audit_hidden_file_access.py",
                "--run-dir",
                str(run_dir),
                "--analysis-root",
                str(dataset_dir),
                "--out-json",
                str(audit_path),
                "--out-md",
                str(audit_md),
            ]
            if evaluation_dataset_dir:
                audit_cmd.extend(["--evaluation-root", str(evaluation_dataset_dir)])
            audit_result = run_command(audit_cmd, PROJECT_DIR, timeout=180)
            audit = {
                "returncode": audit_result["returncode"],
                "json_path": str(audit_path),
                "md_path": str(audit_md),
            }
            try:
                audit.update(json.loads(read_text(audit_path)))
            except Exception:
                audit["status"] = "invalid"

    return {
        "dataset": dataset_dir.name,
        "run_name": run_name,
        "final_run_dir": str(run_dir) if run_dir else "",
        "final_status": final.get("status", ""),
        "attempts": attempts,
        "validation": validation,
        "hidden_file_audit": audit,
    }


def write_batch_markdown(summary_json: Path, score_json: Path, out_md: Path):
    summary = json.loads(read_text(summary_json)) if summary_json.exists() else {"rows": []}
    score = json.loads(read_text(score_json)) if score_json.exists() else {"rows": []}
    batch_rows = {row["dataset"]: row for row in summary.get("rows", [])}

    def run_status_from_dir(run_dir: str) -> str:
        if not run_dir:
            return ""
        status_path = Path(run_dir) / "run_status.json"
        if not status_path.exists():
            return ""
        try:
            return json.loads(read_text(status_path)).get("run_status", "")
        except Exception:
            return ""

    def md_cell(value, max_chars: int = 900) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\r", " ").replace("\n", "<br>").replace("|", "\\|").strip()
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    lines = [
        "# scProteomics Agent V3 11 数据集批量运行汇总",
        "",
        "本汇总由优化后的 Agent 自动运行和验证生成。正式分数以独立评分脚本输出的 `score_v3_11.*` 为准；本表只用于快速查看 run 状态、验证状态和报告入口。",
        "",
        "| Dataset | Run status | Validator | Score | Coverage | Report | Major issues |",
        "|---|---|---|---:|---:|---|---|",
    ]
    rows = score.get("rows", []) or summary.get("rows", [])
    for score_row in rows:
        dataset = score_row.get("dataset", "")
        batch_row = batch_rows.get(dataset, {})
        run_dir = score_row.get("run_dir") or batch_row.get("final_run_dir", "")
        report = ""
        if run_dir:
            run_path = Path(run_dir)
            candidates = [
                run_path / "report.md",
                run_path / f"analysis_report_{dataset}.md",
                run_path / "final_report.md",
                run_path / "output_report_1.md",
            ]
            report = str(next((path for path in candidates if path.exists()), candidates[-1]))
        run_status = batch_row.get("final_status", "") or run_status_from_dir(run_dir)
        validator_status = score_row.get("validator_status", "") or batch_row.get("validation", {}).get("status", "")
        lines.append(
            f"| {md_cell(dataset)} | {md_cell(run_status)} | "
            f"{md_cell(validator_status)} | "
            f"{md_cell(score_row.get('scorer_total_score', ''))} | "
            f"{md_cell(score_row.get('coverage_heuristic', ''))} | "
            f"`{md_cell(report, 300)}` | {md_cell(score_row.get('scorer_major_issues', ''))} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_print_json(payload):
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-root", type=Path, default=None)
    parser.add_argument("--analysis-root", type=Path, default=None, help="Publication-safe generation input root. Defaults to --examples-root.")
    parser.add_argument("--evaluation-root", type=Path, default=None, help="Evaluator-only scoring root. Defaults to --examples-root.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--name-prefix", default="batch_eval")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--retry-once", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_RUNS_ROOT / "batch_evaluation_runs_latest.json")
    parser.add_argument("--score-json", type=Path, default=DEFAULT_RUNS_ROOT / "scientific_report_score_summary_latest.json")
    parser.add_argument("--score-tsv", type=Path, default=DEFAULT_RUNS_ROOT / "scientific_report_score_summary_latest.tsv")
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_RUNS_ROOT / "batch_evaluation_summary.md")
    parser.add_argument("--publication-mode", action="store_true", default=False)
    parser.add_argument("--audit-hidden-file-access", action="store_true", default=False)
    args = parser.parse_args()
    args.examples_root = resolve_examples_root(args.examples_root)

    args.runs_root.mkdir(parents=True, exist_ok=True)
    analysis_root = args.analysis_root or args.examples_root
    evaluation_root = args.evaluation_root or args.examples_root
    dataset_predicate = is_analysis_dataset if args.publication_mode else is_executable_dataset
    datasets = [p for p in sorted(analysis_root.iterdir()) if dataset_predicate(p)]
    if args.datasets:
        wanted = set(args.datasets)
        datasets = [p for p in datasets if p.name in wanted]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        future_map = {
            pool.submit(
                run_one,
                dataset_dir,
                args.runs_root,
                run_name_for_dataset(args.name_prefix, dataset_dir.name),
                args.timeout_sec,
                args.retry_once,
                args.publication_mode,
                args.audit_hidden_file_access,
                evaluation_root / dataset_dir.name,
            ): dataset_dir.name
            for dataset_dir in datasets
        }
        for future in as_completed(future_map):
            rows.append(future.result())

    rows = sorted(rows, key=lambda row: row["dataset"])
    result = {"dataset_count": len(rows), "rows": rows}
    args.summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    score_cmd = [
        sys.executable,
        "scripts/score_reports_against_triplets.py",
        "--examples-root",
        str(evaluation_root),
        "--runs-root",
        str(args.runs_root),
        "--out",
        str(args.score_tsv),
        "--json-out",
        str(args.score_json),
    ]
    if args.datasets:
        score_cmd.extend(["--datasets", *args.datasets])
    score_result = run_command(score_cmd, PROJECT_DIR, timeout=300)
    result["score_summary_returncode"] = score_result["returncode"]
    result["analysis_root"] = str(analysis_root)
    result["evaluation_root"] = str(evaluation_root)
    result["publication_mode"] = args.publication_mode
    result["audit_hidden_file_access"] = args.audit_hidden_file_access
    args.summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_batch_markdown(args.summary_json, args.score_json, args.summary_md)

    safe_print_json(result)
    ok = all(row.get("final_status") == "completed" for row in rows) and score_result["returncode"] == 0
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
