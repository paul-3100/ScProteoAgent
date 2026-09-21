import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small project-owned annotation manifest without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = {"resources": []}
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("manifest_version:"):
            data["manifest_version"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("source_scope:"):
            data["source_scope"] = line.split(":", 1)[1].strip().strip('"')
        elif line.startswith("resources:"):
            continue
        elif line.startswith("  - "):
            if current:
                data["resources"].append(current)
            current = {}
            rest = line[4:]
            if ":" in rest:
                key, value = rest.split(":", 1)
                current[key.strip()] = parse_value(value.strip())
        elif line.startswith("    ") and current is not None and ":" in line:
            key, value = line.strip().split(":", 1)
            current[key.strip()] = parse_value(value.strip())
    if current:
        data["resources"].append(current)
    return data


def parse_value(value: str) -> Any:
    value = value.strip().strip('"')
    if value.startswith("[") and value.endswith("]"):
        inside = value[1:-1].strip()
        if not inside:
            return []
        return [item.strip().strip('"') for item in inside.split(",")]
    return value


def validate_resource(resource: dict[str, Any], project_dir: Path) -> list[str]:
    findings = []
    required = ["id", "path", "evidence_scope", "boundary_label", "citation_status", "license_status"]
    for key in required:
        if not resource.get(key):
            findings.append(f"missing required field `{key}`")
    path_text = str(resource.get("path", ""))
    if path_text:
        path = project_dir / path_text
        if not path.exists():
            findings.append(f"local path does not exist: {path_text}")
    if resource.get("citation_status") != "todo_verify":
        findings.append("citation_status should remain `todo_verify` until external references are manually checked")
    if resource.get("evidence_scope") not in {"user_visible", "external_annotation", "offline_enrichment", "evaluator_only"}:
        findings.append("evidence_scope is not one of the expected controlled values")
    if str(resource.get("id", "")).lower() in {"env", "api", "key"}:
        findings.append("resource id is too close to secret/config terminology")
    return findings


def write_summary(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Annotation manifest validation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Manifest: `{payload['manifest']}`",
        f"- Resource count: `{payload['resource_count']}`",
        "",
        "| Resource | Scope | Boundary | Citation status | Local path | Findings |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["resources"]:
        findings = "; ".join(row["findings"]) if row["findings"] else "pass"
        lines.append(
            f"| `{row['id']}` | `{row['evidence_scope']}` | `{row['boundary_label']}` | "
            f"`{row['citation_status']}` | `{row['path']}` | {findings} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local annotation resource manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    data = parse_simple_yaml(args.manifest)
    rows = []
    all_findings = []
    for resource in data.get("resources", []):
        findings = validate_resource(resource, args.project_dir)
        rows.append(
            {
                "id": resource.get("id", ""),
                "path": resource.get("path", ""),
                "evidence_scope": resource.get("evidence_scope", ""),
                "boundary_label": resource.get("boundary_label", ""),
                "citation_status": resource.get("citation_status", ""),
                "license_status": resource.get("license_status", ""),
                "findings": findings,
            }
        )
        all_findings.extend(findings)
    status = "pass" if not all_findings else "review_required"
    payload = {
        "status": status,
        "manifest": str(args.manifest),
        "source_scope": data.get("source_scope", ""),
        "resource_count": len(rows),
        "resources": rows,
        "policy": "External references are not verified by this script; citation_status remains todo_verify.",
    }
    out_json = args.out_json or args.manifest.with_suffix(".validation.json")
    out_md = args.out_md or args.manifest.with_suffix(".validation.md")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(payload, out_md)
    print(json.dumps({"status": status, "resources": len(rows), "out_json": str(out_json)}, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
