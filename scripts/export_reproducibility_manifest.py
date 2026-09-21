import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
FORBIDDEN_RELEASE_NAMES = {
    ".env",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}
SECRET_PATTERNS = [
    "sk-",
    "api" + "_key=",
    "api" + "key=",
    "tok" + "en=",
    "bearer ",
]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def run_git_commit(project_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def scan_release_root(root: Path) -> dict[str, Any]:
    forbidden = []
    secret_like = []
    if not root.exists():
        return {"status": "missing", "forbidden_entries": [], "secret_like_hits": []}
    for path in root.rglob("*"):
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            if path.name in FORBIDDEN_RELEASE_NAMES:
                forbidden.append(str(path.relative_to(root)))
            continue
        if path.name in FORBIDDEN_RELEASE_NAMES:
            forbidden.append(str(path.relative_to(root)))
            continue
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".env", ".example"}:
            text = read_text(path).lower()
            if any(pattern in text for pattern in SECRET_PATTERNS):
                secret_like.append(str(path.relative_to(root)))
    status = "pass" if not forbidden and not secret_like else "review_required"
    return {
        "status": status,
        "forbidden_entries": sorted(set(forbidden)),
        "secret_like_hits": sorted(set(secret_like)),
    }


def collect_publication_eligibility(runs_root: Path | None) -> list[dict[str, Any]]:
    rows = []
    if not runs_root or not runs_root.exists():
        return rows
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir():
            continue
        eligibility_path = path / "publication_eligibility.json"
        if eligibility_path.exists():
            try:
                doc = json.loads(read_text(eligibility_path))
            except Exception:
                doc = {}
            rows.append({
                "run": path.name,
                "publication_eligible": doc.get("publication_eligible"),
                "audit_status": doc.get("status"),
                "findings_n": len(doc.get("findings", [])),
            })
    return rows


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Reproducibility manifest",
        "",
        f"- Project dir: `{payload['project_dir']}`",
        f"- Python: `{payload['python']['version']}`",
        f"- Platform: `{payload['platform']}`",
        f"- Git commit: `{payload.get('git_commit', '')}`",
        f"- Analysis root: `{payload.get('analysis_root', '')}`",
        f"- Evaluation root: `{payload.get('evaluation_root', '')}`",
        f"- Runs root: `{payload.get('runs_root', '')}`",
        f"- Release scan status: `{payload['release_scan']['status']}`",
        "",
        "## Publication eligibility",
        "",
        "| Run | Eligible | Audit status | Findings |",
        "|---|---|---|---:|",
    ]
    for row in payload["publication_eligibility"]:
        lines.append(f"| `{row['run']}` | `{row.get('publication_eligible')}` | `{row.get('audit_status')}` | {row.get('findings_n', 0)} |")
    if not payload["publication_eligibility"]:
        lines.append("| _none found_ |  |  |  |")
    lines.extend([
        "",
        "## Release scan",
        "",
        f"- Forbidden entries: {len(payload['release_scan']['forbidden_entries'])}",
        f"- Secret-like hits: {len(payload['release_scan']['secret_like_hits'])}",
        "",
        "This manifest does not include secret values. Any secret-like hit requires manual review before sharing code or artifacts.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a manuscript reproducibility manifest.")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--evaluation-root", type=Path, default=None)
    parser.add_argument("--release-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "project_dir": str(args.project_dir),
        "git_commit": run_git_commit(args.project_dir),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "platform": platform.platform(),
        "runs_root": str(args.runs_root) if args.runs_root else "",
        "analysis_root": str(args.analysis_root) if args.analysis_root else "",
        "evaluation_root": str(args.evaluation_root) if args.evaluation_root else "",
        "publication_eligibility": collect_publication_eligibility(args.runs_root),
        "release_scan": scan_release_root(args.release_root),
        "model_config": {
            "source": "runtime environment / project config",
            "note": "Do not export .env or secret values. Record model names and non-secret runtime settings only.",
        },
        "availability_drafts": {
            "data": "DATA_AVAILABILITY_DRAFT.md",
            "code": "CODE_AVAILABILITY_DRAFT.md",
            "ai_use": "AI_USE_STATEMENT_DRAFT.md",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "release_scan_status": payload["release_scan"]["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
