import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_FILES = {
    "ProteinQuant.csv",
    "SampleInfo.csv",
    "user_input.txt",
}
OPTIONAL_ALLOWED_FILES = {
    "analysis_design.yaml",
    "analysis_design.yml",
}
FORBIDDEN_NAMES = {
    ".env",
    "ground_truth.txt",
    "grading_standard.txt",
}
FORBIDDEN_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "runs_v3",
    "runs_smoke",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_is_usable(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in DEFAULT_ALLOWED_FILES)


def copy_dataset(source: Path, dest: Path, include_analysis_design: bool) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    excluded = []
    allowed = set(DEFAULT_ALLOWED_FILES)
    if include_analysis_design:
        allowed.update(OPTIONAL_ALLOWED_FILES)

    for item in sorted(source.iterdir()):
        rel_name = item.name
        if item.is_dir():
            excluded.append({"path": rel_name, "reason": "directory_not_in_blind_input"})
            continue
        if rel_name in allowed:
            target = dest / rel_name
            shutil.copy2(item, target)
            copied.append({
                "path": rel_name,
                "size": target.stat().st_size,
                "sha256": sha256_file(target),
            })
        else:
            reason = "forbidden_evaluator_only" if rel_name in FORBIDDEN_NAMES else "not_whitelisted"
            excluded.append({"path": rel_name, "reason": reason})

    return {
        "dataset": source.name,
        "source": str(source),
        "destination": str(dest),
        "copied": copied,
        "excluded": excluded,
        "status": "pass" if all((dest / name).exists() for name in DEFAULT_ALLOWED_FILES) else "missing_required",
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Blind analysis input manifest",
        "",
        f"- Source root: `{payload['source_examples_root']}`",
        f"- Analysis root: `{payload['out_analysis_root']}`",
        f"- Dataset count: {payload['dataset_count']}",
        f"- Status: `{payload['status']}`",
        "",
        "| Dataset | Status | Copied files | Excluded evaluator-only files |",
        "|---|---|---:|---:|",
    ]
    for row in payload["datasets"]:
        excluded_eval = sum(1 for item in row["excluded"] if item.get("reason") == "forbidden_evaluator_only")
        lines.append(f"| `{row['dataset']}` | `{row['status']}` | {len(row['copied'])} | {excluded_eval} |")
    lines.extend([
        "",
        "Allowed files: `ProteinQuant.csv`, `SampleInfo.csv`, `user_input.txt`, optional `analysis_design.yaml/yml`.",
        "Evaluator-only files such as `ground_truth.txt` and `grading_standard.txt` are deliberately excluded.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare publication-safe blind analysis inputs.")
    parser.add_argument("--source-examples-root", type=Path, required=True)
    parser.add_argument("--out-analysis-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--include-analysis-design", action="store_true")
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--manifest-md", type=Path, required=True)
    args = parser.parse_args()

    wanted = set(args.datasets or [])
    source_datasets = [p for p in sorted(args.source_examples_root.iterdir()) if dataset_is_usable(p)]
    if wanted:
        source_datasets = [p for p in source_datasets if p.name in wanted]
    args.out_analysis_root.mkdir(parents=True, exist_ok=True)

    rows = [
        copy_dataset(src, args.out_analysis_root / src.name, args.include_analysis_design)
        for src in source_datasets
    ]
    status = "pass" if rows and all(row["status"] == "pass" for row in rows) else "fail"
    payload = {
        "status": status,
        "source_examples_root": str(args.source_examples_root),
        "out_analysis_root": str(args.out_analysis_root),
        "dataset_count": len(rows),
        "datasets": rows,
        "forbidden_names": sorted(FORBIDDEN_NAMES),
        "forbidden_dirs": sorted(FORBIDDEN_DIRS),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.manifest_md)
    print(json.dumps({"status": status, "dataset_count": len(rows), "manifest_json": str(args.manifest_json)}, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
