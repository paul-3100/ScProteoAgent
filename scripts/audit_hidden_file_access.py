import argparse
import json
import re
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}
FORBIDDEN_FILE_NAMES = {
    "ground_truth.txt",
    "grading_standard.txt",
}
FORBIDDEN_CONTENT_RE = re.compile(
    r"(ground_truth_path|grading_standard_path|ground_truth\.txt|grading_standard\.txt)",
    re.IGNORECASE,
)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def iter_text_files(root: Path):
    if not root or not root.exists():
        return
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def scan_analysis_root(root: Path | None) -> list[dict[str, Any]]:
    findings = []
    if not root:
        return findings
    if not root.exists():
        findings.append({"severity": "error", "path": str(root), "issue": "analysis_root_missing"})
        return findings
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in FORBIDDEN_FILE_NAMES:
            findings.append({"severity": "error", "path": str(path), "issue": "forbidden_file_in_analysis_root"})
    return findings


def scan_run_dir(run_dir: Path | None) -> list[dict[str, Any]]:
    findings = []
    if not run_dir:
        return findings
    if not run_dir.exists():
        findings.append({"severity": "error", "path": str(run_dir), "issue": "run_dir_missing"})
        return findings
    for path in iter_text_files(run_dir):
        rel = path.relative_to(run_dir)
        if path.name in {"hidden_file_access_audit.json", "hidden_file_access_audit.md", "publication_eligibility.json"}:
            continue
        text = read_text(path)
        for match in FORBIDDEN_CONTENT_RE.finditer(text):
            findings.append({
                "severity": "error",
                "path": str(rel),
                "issue": "hidden_standard_reference_in_generation_output",
                "match": match.group(0),
            })
            break
    return findings


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(read_text(path))
    except Exception:
        return {}


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Hidden-file access audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Publication eligible: `{payload['publication_eligible']}`",
        f"- Run dir: `{payload.get('run_dir', '')}`",
        f"- Analysis root: `{payload.get('analysis_root', '')}`",
        f"- Evaluation root: `{payload.get('evaluation_root', '')}`",
        f"- Findings: {len(payload['findings'])}",
        "",
    ]
    if payload["findings"]:
        lines.extend(["| Severity | Issue | Path | Match |", "|---|---|---|---|"])
        for item in payload["findings"]:
            lines.append(
                f"| `{item.get('severity', '')}` | `{item.get('issue', '')}` | "
                f"`{item.get('path', '')}` | `{item.get('match', '')}` |"
            )
    else:
        lines.append("No evaluator-only file names or hidden standard path keys were detected in the scanned generation artifacts.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit generation artifacts for hidden evaluator-only file access.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--evaluation-root", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    findings = []
    findings.extend(scan_analysis_root(args.analysis_root))
    findings.extend(scan_run_dir(args.run_dir))

    run_metadata = load_json(args.run_dir / "run_metadata.json") if args.run_dir else {}
    if args.run_dir and not run_metadata:
        findings.append({"severity": "warning", "path": str(args.run_dir / "run_metadata.json"), "issue": "run_metadata_missing"})
    if run_metadata.get("hidden_standard_paths_in_generation_config"):
        findings.append({"severity": "error", "path": "run_metadata.json", "issue": "hidden_paths_present_in_generation_config"})
    if run_metadata.get("enable_internal_scorer"):
        findings.append({"severity": "error", "path": "run_metadata.json", "issue": "internal_scorer_enabled"})

    has_errors = any(item.get("severity") == "error" for item in findings)
    status = "fail" if has_errors else "pass"
    payload = {
        "status": status,
        "publication_eligible": not has_errors,
        "run_dir": str(args.run_dir) if args.run_dir else "",
        "analysis_root": str(args.analysis_root) if args.analysis_root else "",
        "evaluation_root": str(args.evaluation_root) if args.evaluation_root else "",
        "run_metadata": {
            "publication_mode": run_metadata.get("publication_mode"),
            "evaluator_mode": run_metadata.get("evaluator_mode"),
            "enable_internal_scorer": run_metadata.get("enable_internal_scorer"),
            "publication_eligible_initial": run_metadata.get("publication_eligible_initial"),
        },
        "findings": findings,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.out_md)
    if args.run_dir:
        (args.run_dir / "publication_eligibility.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "publication_eligible": not has_errors, "out_json": str(args.out_json)}, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
