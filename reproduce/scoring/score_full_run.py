import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
REQUIRED_DATASET_FILES = [
    "ProteinQuant.csv",
    "SampleInfo.csv",
    "user_input.txt",
    "ground_truth.txt",
    "grading_standard.txt",
]
CHINESE_REPORT_HEADINGS = [
    "## 一、执行摘要",
    "## 二、核心证据首页",
    "## 三、主要发现",
    "## 四、证据边界",
    "## 五、可复核证据附录",
]
CLAUDE_STYLE_REPORT_HEADINGS = [
    "## 0. 关键结论速览",
    "## 1. 数据与预处理",
    "## 2. 任务一：聚类与迁移状态的对应",
    "## 3. 任务二：迁移相关 Cluster 的功能模块",
    "## 4. 任务三：骨架调控候选",
    "## 5. 任务四：对照群体的隐性差异",
    "## 6. 总结与启发式推测",
    "## 7. 可复核资产清单",
    "## 8. 结论边界与方法局限",
]
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|token)\s*=\s*[^`\s]+"),
]


def fs_path(path: Path) -> str:
    """Return a Windows long-path-safe string for direct filesystem calls."""
    text = str(path)
    if sys.platform.startswith("win"):
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = text
        if resolved.startswith("\\\\?\\"):
            return resolved
        if resolved.startswith("\\\\"):
            return "\\\\?\\UNC\\" + resolved.lstrip("\\")
        return "\\\\?\\" + resolved
    return text


def path_exists(path: Path) -> bool:
    return Path(fs_path(path)).exists()


def read_text(path: Path, limit: int | None = None) -> str:
    if not path_exists(path):
        return ""
    text = Path(fs_path(path)).read_text(encoding="utf-8", errors="ignore")
    return text[:limit] if limit else text


def read_json(path: Path) -> dict[str, Any]:
    if not path_exists(path):
        return {}
    try:
        return json.loads(read_text(path))
    except Exception:
        return {}


def is_executable_dataset(folder: Path) -> bool:
    return folder.is_dir() and all(path_exists(folder / name) for name in REQUIRED_DATASET_FILES)


def _run_input_folder(run_dir: Path) -> str:
    events = run_dir / "run_events.jsonl"
    if not path_exists(events):
        return ""
    try:
        with Path(fs_path(events)).open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                return str(event.get("input_folder", ""))
    except Exception:
        return ""
    return ""


def latest_run_for_dataset(runs_root: Path, dataset: str) -> Path | None:
    if not runs_root.exists():
        return None
    candidates: list[Path] = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        if dataset in path.name or dataset in _run_input_folder(path):
            has_report = (
                path_exists(path / "report.md")
                or path_exists(path / "final_report.md")
                or list(path.glob("analysis_report_*.md"))
                or list(path.glob("final_analysis_report_*.md"))
                or (path / "final_output_report.md").exists()
                or list(path.glob("output_report_*.md"))
            )
            if path_exists(path / "run_status.json") and has_report:
                candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def load_report(run_dir: Path | None) -> tuple[str, str]:
    if not run_dir:
        return "", ""
    preferred_reports = [
        run_dir / "report.md",
        run_dir / "final_report.md",
        run_dir / f"analysis_report_{run_dir.name}.md",
        run_dir / "analysis_report_Nat_Commun_PiSPA_2024.md",
        run_dir / f"final_analysis_report_{run_dir.name}.md",
        run_dir / f"final_analysis_report_Nat_Commun_PiSPA_2024.md",
        run_dir / "final_output_report.md",
        run_dir / "output_report_1.md",
        run_dir / "output_report_1_conclusions.md",
    ]
    preferred_reports.extend(sorted(run_dir.glob("analysis_report_*.md")))
    preferred_reports.extend(sorted(run_dir.glob("final_analysis_report_*.md")))
    for preferred in preferred_reports:
        if path_exists(preferred) and "evidence_appendix" not in preferred.name and "conclusions" not in preferred.name:
            return read_text(preferred), str(preferred)
    reports = sorted(
        path
        for path in run_dir.glob("output_report_*.md")
        if "evidence_appendix" not in path.name
    )
    if reports:
        return read_text(reports[0]), str(reports[0])
    return "", ""


def chinese_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z\u4e00-\u9fff]", text)
    if not letters:
        return 0.0
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return len(chinese) / len(letters)


def main_report_body(report: str) -> str:
    markers = [
        "## 五、证据附录",
        "## 7. 可复核资产清单",
        "### 可复核证据附录",
        "### 技术证据附录",
        "### 可复核证据附录",
        "## Evidence Appendix",
        "## Scoring Evidence Appendix",
    ]
    split_at = len(report)
    for marker in markers:
        idx = report.find(marker)
        if idx >= 0:
            split_at = min(split_at, idx)
    dynamic_assets = re.search(r"(?m)^##\s+\d+\.\s*可复核资产清单", report)
    if dynamic_assets:
        split_at = min(split_at, dynamic_assets.start())
    return report[:split_at]


def extract_keywords(text: str, limit: int = 80) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{3,}|[\u4e00-\u9fff]{2,8}", text)
    stop = {
        "current",
        "matrix",
        "ground",
        "truth",
        "grading",
        "standard",
        "report",
        "protein",
        "sample",
        "数据",
        "分析",
        "结果",
        "报告",
        "评分",
        "标准",
        "证据",
        "蛋白",
        "矩阵",
        "需要",
        "说明",
        "输出",
        "当前",
        "要求",
        "应该",
    }
    counts: dict[str, int] = {}
    for token in tokens:
        key = token.lower()
        if key in stop or len(key) < 2:
            continue
        counts[key] = counts.get(key, 0) + 1
    return [key for key, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def keyword_coverage(reference: str, report: str) -> tuple[float, list[str], list[str]]:
    keywords = extract_keywords(reference)
    report_lower = report.lower()
    matched = [kw for kw in keywords if kw in report_lower]
    missing = [kw for kw in keywords if kw not in report_lower]
    ratio = len(matched) / max(len(keywords), 1)
    return ratio, matched, missing


def has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def has_literal(text: str, *tokens: str) -> bool:
    return any(token in text for token in tokens)


def has_story_task_heading(text: str) -> bool:
    return bool(re.search(r"(?m)^##\s+\d+\.\s*任务", text))


def has_story_first_structure(text: str) -> bool:
    return (
        "## 0. 关键结论速览" in text
        and "## 1. 数据与预处理" in text
        and has_story_task_heading(text)
        and "总结与启发式推测" in text
        and "可复核资产清单" in text
        and "结论边界与方法局限" in text
    )


def required_story_contrasts(dataset: str) -> list[str]:
    return {
        "Nat_Commun_PiSPA_2024": [
            "Cluster_1_vs_Cluster_2",
            "Cluster_1_vs_Cluster_3",
            "Cluster_2_vs_Cluster_3",
            "Migrated_vs_Control",
        ],
        "Nat_Commun_Carr_2024": ["LPS_vs_DMSO"],
        "Nat_Commun_Nociceptor_2026": [
            "TrkA_Inflamed_vs_TrkA_Control",
            "IB4_Inflamed_vs_IB4_Control",
            "Mechano_Inflamed_vs_Mechano_Control",
        ],
        "Nat_Commun_SCPro_2024": [
            "CD4_CD25pos_Klrg1pos_vs_CD4_CD25pos_Klrg1neg",
            "CD4_Klrg1pos_vs_CD4_Klrg1neg",
            "CD8_Klrg1pos_vs_CD8_Klrg1neg",
        ],
        "Cell_TurnoverDynamics_2025": [
            "Bortezomib_High_vs_Control",
            "Bortezomib_Low_vs_Control",
            "Cycloheximide_High_vs_Control",
            "Cycloheximide_Low_vs_Control",
        ],
    }.get(dataset, [])


def story_quality_checks(dataset: str, run_dir: Path | None, report: str) -> dict[str, Any]:
    story_text = ""
    if run_dir:
        story_text = read_text(run_dir / "evaluation_evidence" / "core_story_evidence.csv")
    required = required_story_contrasts(dataset)
    missing = [token for token in required if token not in story_text and token not in report]
    has_core_table = has_literal(report, "核心对比表", "核心对比证据", "核心故事证据表")
    has_candidate_table = has_literal(report, "代表蛋白表", "候选蛋白表", "骨架调控候选")
    has_quant_story = bool(re.search(r"logFC|adj\.P\.Val|FDR", report, re.I))
    direction_standardized = True
    if dataset == "Nat_Commun_Carr_2024" and "DMSO_vs_LPS" in report and "LPS_vs_DMSO" not in report:
        direction_standardized = False
    if dataset == "Cell_TurnoverDynamics_2025" and "Control_vs_Cycloheximide" in report and "Cycloheximide_High_vs_Control" not in report:
        direction_standardized = False
    return {
        "has_core_story_evidence": bool(story_text.strip()),
        "required_contrasts": required,
        "missing_required_contrasts": missing,
        "has_core_contrast_table": has_core_table,
        "has_representative_protein_table": has_candidate_table,
        "has_quantitative_story_values": has_quant_story,
        "direction_standardized": direction_standardized,
        "story_text_length": len(story_text),
    }


def artifact_status(run_dir: Path | None, dataset: str) -> dict[str, Any]:
    if not run_dir:
        return {
            "run_status": "",
            "validator_status": "",
            "report_self_check_status": "",
            "has_candidate_evidence": False,
            "has_core_story_evidence": False,
            "has_group_qc": False,
            "has_module_sample_scores": False,
            "has_dataset_recipe": False,
            "has_coverage_table": False,
            "has_figure_index_or_manifest": False,
            "has_external_annotation": False,
            "external_annotation_applicable": dataset == "Nat_Biotech_Brain_2026",
        }
    evidence = run_dir / "evaluation_evidence"
    status_doc = read_json(run_dir / "run_status.json") or read_json(run_dir / "status.json")
    validation_doc = read_json(run_dir / "report_validation.json") or read_json(run_dir / "validation.json")
    self_check_doc = read_json(run_dir / "report_self_check.json") or read_json(run_dir / "self_check.json")
    status = status_doc.get("run_status") or status_doc.get("status", "")
    validator = validation_doc.get("status", "")
    self_check = self_check_doc.get("status", "")
    external_json = read_json(evidence / "external_annotation_asd_ndd_local.json")
    external_applicable = bool(external_json.get("applicable")) if external_json else dataset == "Nat_Biotech_Brain_2026"
    figure_index_exists = any(
        path_exists(candidate)
        for candidate in [
            run_dir / "figure_index.md",
            run_dir / "figure_manifest.json",
            run_dir / "figures.md",
            run_dir / "assets.txt",
            run_dir / "visualize_results" / "figure_index.md",
            run_dir / "visualize_results" / "figure_manifest.json",
        ]
    )
    return {
        "run_status": status,
        "validator_status": validator,
        "report_self_check_status": self_check,
        "has_candidate_evidence": path_exists(evidence / "candidate_protein_evidence.csv"),
        "has_core_story_evidence": path_exists(evidence / "core_story_evidence.csv"),
        "has_group_qc": path_exists(evidence / "group_composition_qc.csv"),
        "has_module_sample_scores": path_exists(evidence / "curated_module_sample_scores.csv"),
        "has_dataset_recipe": path_exists(evidence / "dataset_recipe_evidence.csv"),
        "has_coverage_table": path_exists(evidence / "scoring_standard_coverage.csv"),
        "has_figure_index_or_manifest": figure_index_exists,
        "has_external_annotation": path_exists(evidence / "external_annotation_asd_ndd_local.csv"),
        "external_annotation_applicable": external_applicable,
    }


def score_rule_based(dataset_dir: Path, run_dir: Path | None, report: str) -> dict[str, Any]:
    dataset = dataset_dir.name
    ground_truth = read_text(dataset_dir / "ground_truth.txt")
    grading = read_text(dataset_dir / "grading_standard.txt")
    reference = ground_truth + "\n" + grading
    coverage, matched, missing = keyword_coverage(reference, report)
    artifacts = artifact_status(run_dir, dataset)
    story = story_quality_checks(dataset, run_dir, report)
    report_main = main_report_body(report)
    report_len = len(report_main)

    has_report = bool(report.strip())
    has_old_headings = all(heading in report for heading in CHINESE_REPORT_HEADINGS)
    has_story_first = has_story_first_structure(report)
    has_claude_headings = has_story_first or all(heading in report for heading in CLAUDE_STYLE_REPORT_HEADINGS)
    has_all_headings = has_old_headings or has_story_first
    has_current_matrix = "current_matrix" in report
    has_offline_enrichment = "offline_enrichment" in report
    has_external_annotation_label = "external_annotation" in report
    has_quant = has_any(
        report,
        [
            r"\blogFC\b",
            r"\bFDR\b",
            r"adj\.P\.Val",
            r"p[- ]?value",
            r"检出率",
            r"均值",
            r"差异",
            r"\bn\s*=",
            r"\d+\.\d+",
        ],
    )
    has_boundary_heading = "## 四、证据边界" in report or "结论边界与方法局限" in report
    has_scoring_first_screen = "## 二、核心证据首页" in report or has_story_task_heading(report)
    has_main_findings = "## 三、主要发现" in report or has_story_task_heading(report)
    has_csv_refs = ".csv" in report
    has_figure_refs = (
        "figure_index" in report
        or "figure_manifest" in report
        or "figures/" in report
        or ".png" in report
        or "图" in report
    )
    has_raw_json_or_traceback = has_any(report, [r"Traceback \(most recent call last\)", r"```json", r"\{\s*\"[A-Za-z0-9_]+\"\s*:"])
    has_secret = any(pattern.search(report) for pattern in SECRET_PATTERNS)
    nonbrain_asd = dataset != "Nat_Biotech_Brain_2026" and has_any(report, [r"\bASD\b", r"\bNDD\b", r"SFARI", r"自闭症"])
    brain_external_ok = dataset != "Nat_Biotech_Brain_2026" or (
        artifacts["has_external_annotation"] and has_external_annotation_label and has_boundary_heading
    )

    scientific = 0.0
    scientific += min(18.0, coverage * 24.0)
    scientific += 5.0 if has_scoring_first_screen else 0.0
    scientific += 5.0 if has_main_findings else 0.0
    scientific += 4.0 if artifacts["has_dataset_recipe"] else 0.0
    scientific += 3.0 if artifacts["has_coverage_table"] else 0.0
    scientific += 4.0 if story["has_core_story_evidence"] and not story["missing_required_contrasts"] else 0.0
    scientific += 2.0 if story["has_core_contrast_table"] else 0.0
    scientific = min(scientific, 35.0)

    matrix = 0.0
    matrix += 4.0 if has_current_matrix else 0.0
    matrix += 4.0 if has_quant else 0.0
    matrix += 4.0 if artifacts["has_candidate_evidence"] else 0.0
    matrix += 3.0 if artifacts["has_core_story_evidence"] and story["has_quantitative_story_values"] else 0.0
    matrix += 3.0 if artifacts["has_group_qc"] else 0.0
    matrix += 3.0 if has_csv_refs else 0.0
    matrix += 2.0 if has_figure_refs else 0.0
    matrix = min(matrix, 20.0)

    boundary = 0.0
    boundary += 4.0 if has_boundary_heading else 0.0
    boundary += 3.0 if has_current_matrix else 0.0
    boundary += 3.0 if has_offline_enrichment else 0.0
    boundary += 3.0 if brain_external_ok else 0.0
    boundary += 2.0 if not nonbrain_asd else 0.0
    boundary = min(boundary, 15.0)

    artifacts_score = 0.0
    artifacts_score += 3.0 if artifacts["run_status"] == "completed" else 0.0
    artifacts_score += 3.0 if artifacts["validator_status"] == "pass" else 0.0
    artifacts_score += 3.0 if artifacts["report_self_check_status"] == "pass" else 0.0
    artifacts_score += 2.0 if artifacts["has_candidate_evidence"] else 0.0
    artifacts_score += 2.0 if artifacts["has_module_sample_scores"] else 0.0
    artifacts_score += 1.0 if artifacts["has_dataset_recipe"] else 0.0
    artifacts_score += 1.0 if artifacts["has_core_story_evidence"] else 0.0
    artifacts_score += 1.0 if artifacts["has_figure_index_or_manifest"] else 0.0
    artifacts_score = min(artifacts_score, 15.0)

    language = 0.0
    language += 4.0 if has_all_headings else 0.0
    main_chinese_ratio = chinese_ratio(report_main)
    language += 3.0 if main_chinese_ratio >= 0.35 else 0.0
    language += 2.0 if 1200 <= report_len <= 18000 else 0.0
    language += 1.0 if not has_raw_json_or_traceback else 0.0
    language += 1.0 if story["has_core_contrast_table"] and story["has_representative_protein_table"] else 0.0
    language = min(language, 10.0)

    penalty_credit = 5.0
    penalties: list[str] = []
    if not has_report:
        penalty_credit -= 5.0
        penalties.append("缺少报告正文")
    if artifacts["run_status"] != "completed":
        penalty_credit -= 2.0
        penalties.append(f"run_status={artifacts['run_status'] or 'missing'}")
    if has_raw_json_or_traceback:
        penalty_credit -= 2.0
        penalties.append("报告包含 raw JSON 或 traceback 风险")
    if has_secret:
        penalty_credit -= 5.0
        penalties.append("报告疑似包含 key/token")
    if nonbrain_asd:
        penalty_credit -= 2.0
        penalties.append("非 Brain 数据集出现 ASD/NDD 外部注释")
    if not artifacts["has_core_story_evidence"]:
        penalty_credit -= 2.0
        penalties.append("缺少 core_story_evidence 核心故事证据表")
    if story["missing_required_contrasts"]:
        penalty_credit -= 2.0
        penalties.append("缺少必需核心对比: " + ",".join(story["missing_required_contrasts"]))
    if not story["direction_standardized"]:
        penalty_credit -= 2.0
        penalties.append("核心对比方向未按评分标准标准化")
    if required_story_contrasts(dataset) and not story["has_core_contrast_table"]:
        penalty_credit -= 1.0
        penalties.append("报告前两页缺少核心对比表")
    if dataset == "Nat_Biotech_Brain_2026" and not artifacts["has_external_annotation"]:
        penalty_credit -= 1.0
        penalties.append("Brain 缺少本地 ASD/NDD external_annotation 产物")
    penalty_credit = max(0.0, penalty_credit)

    dimensions = {
        "scientific_coverage": round(scientific, 2),
        "current_matrix_evidence": round(matrix, 2),
        "boundary_compliance": round(boundary, 2),
        "artifact_reproducibility": round(artifacts_score, 2),
        "chinese_report_quality": round(language, 2),
        "penalty_credit": round(penalty_credit, 2),
    }
    rule_total = round(sum(dimensions.values()), 2)
    return {
        "dataset": dataset,
        "rule_total": rule_total,
        "dimensions": dimensions,
        "keyword_coverage": round(coverage, 4),
        "matched_keywords": matched[:40],
        "missing_keywords": missing[:30],
        "checks": {
            "has_report": has_report,
            "has_all_chinese_headings": has_all_headings,
            "has_legacy_five_part_headings": has_old_headings,
            "has_claude_style_headings": has_claude_headings,
            "has_story_first_structure": has_story_first,
            "has_current_matrix": has_current_matrix,
            "has_offline_enrichment": has_offline_enrichment,
            "has_external_annotation_label": has_external_annotation_label,
            "has_quantitative_evidence": has_quant,
            "story_quality": story,
            "nonbrain_asd_or_ndd_mention": nonbrain_asd,
            "has_raw_json_or_traceback": has_raw_json_or_traceback,
            "has_secret_like_text": has_secret,
            "chinese_ratio": round(main_chinese_ratio, 4),
            "main_report_length": report_len,
            "full_report_length": len(report),
        },
        "artifacts": artifacts,
        "rule_penalties": penalties,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def run_llm_review(
    rule_result: dict[str, Any],
    ground_truth: str,
    grading: str,
    report: str,
    score_model: str | None = None,
    llm_timeout_sec: float | None = None,
    max_llm_adjust: float = 5.0,
) -> dict[str, Any]:
    try:
        if score_model:
            from langchain_openai import ChatOpenAI
            from llm import OPENAI_API_KEY, OPENAI_BASE_URL

            if not OPENAI_API_KEY or not OPENAI_BASE_URL:
                return {"available": False, "error": "missing_openai_compatible_config_for_score_model_override"}
            SCORELLM = ChatOpenAI(
                model=score_model,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                timeout=llm_timeout_sec or 180,
                max_retries=1,
            )
        else:
            from llm import SCORELLM
    except Exception as exc:
        return {"available": False, "error": f"import_llm_failed: {exc}"}

    prompt = f"""
你是单细胞蛋白质组报告的独立评分复核者。请只基于给定 ground_truth、grading_standard、报告正文和规则评分摘要复核分数。

要求：
1. 不使用报告作者自评分。
2. 只允许给出 -{max_llm_adjust:g} 到 +{max_llm_adjust:g} 的 llm_adjustment。
3. 如果报告缺少当前矩阵证据、边界不清、外部注释越界、没有可复核产物，应扣分。
4. 输出严格 JSON，不要 Markdown。

JSON schema:
{{
  "llm_adjustment": 0,
  "major_findings": ["..."],
  "major_penalties": ["..."],
  "confidence": "high|moderate|low",
  "rationale": "..."
}}

规则评分摘要:
{json.dumps(rule_result, ensure_ascii=False)[:6000]}

ground_truth:
{ground_truth[:6000]}

grading_standard:
{grading[:6000]}

报告正文:
{report[:14000]}
"""
    try:
        response = SCORELLM.invoke(prompt)
        content = getattr(response, "content", str(response))
        parsed = extract_json_object(content)
        if not parsed:
            return {"available": False, "error": "llm_returned_non_json", "raw_excerpt": content[:500]}
        adjustment = parsed.get("llm_adjustment", 0)
        try:
            adjustment = float(adjustment)
        except Exception:
            adjustment = 0.0
        parsed["llm_adjustment"] = max(-float(max_llm_adjust), min(float(max_llm_adjust), adjustment))
        parsed["available"] = True
        return parsed
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def score_one(
    dataset_dir: Path,
    runs_root: Path,
    details_dir: Path,
    skip_llm: bool = False,
    score_model: str | None = None,
    llm_timeout_sec: float | None = None,
    max_llm_adjust: float = 5.0,
) -> dict[str, Any]:
    run_dir = latest_run_for_dataset(runs_root, dataset_dir.name)
    report, report_path = load_report(run_dir)
    ground_truth = read_text(dataset_dir / "ground_truth.txt")
    grading = read_text(dataset_dir / "grading_standard.txt")
    rule = score_rule_based(dataset_dir, run_dir, report)
    llm = (
        {"available": False, "error": "skip_llm_requested"}
        if skip_llm
        else run_llm_review(rule, ground_truth, grading, report, score_model=score_model, llm_timeout_sec=llm_timeout_sec, max_llm_adjust=max_llm_adjust)
    )
    if llm.get("available"):
        adjustment = float(llm.get("llm_adjustment", 0.0))
        scoring_mode = "rule_plus_llm_review"
    else:
        adjustment = 0.0
        scoring_mode = "rule_only_fallback"
    final_score = round(max(0.0, min(100.0, rule["rule_total"] + adjustment)), 2)
    result = {
        "dataset": dataset_dir.name,
        "run_dir": str(run_dir) if run_dir else "",
        "report_path": report_path,
        "scoring_mode": scoring_mode,
        "rule_total": rule["rule_total"],
        "llm_adjustment": round(adjustment, 2),
        "final_score": final_score,
        "confidence": llm.get("confidence", "low" if not llm.get("available") else ""),
        "major_findings": llm.get("major_findings", []),
        "major_penalties": (llm.get("major_penalties", []) or []) + rule.get("rule_penalties", []),
        "llm_review": llm,
        "rule_result": rule,
    }
    details_dir.mkdir(parents=True, exist_ok=True)
    detail_path = details_dir / f"independent_score_detail_{dataset_dir.name}.json"
    detail_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["detail_json"] = str(detail_path)
    return result


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset",
        "final_score",
        "rule_total",
        "llm_adjustment",
        "scoring_mode",
        "confidence",
        "run_dir",
        "report_path",
        "major_penalties",
        "detail_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "; ".join(row.get(key, [])) if isinstance(row.get(key), list) else row.get(key, "")
                for key in fields
            })


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    def cell(value: Any, limit: int = 280) -> str:
        if isinstance(value, list):
            text = "; ".join(str(v) for v in value)
        else:
            text = "" if value is None else str(value)
        text = text.replace("\r", " ").replace("\n", "<br>").replace("|", "\\|").strip()
        return text[: limit - 3] + "..." if len(text) > limit else text

    fallback_rows = [row for row in rows if row.get("scoring_mode") != "rule_plus_llm_review"]
    first_error = ""
    for row in fallback_rows:
        error = (row.get("llm_review") or {}).get("error", "")
        if error:
            first_error = error
            break
    lines = [
        "# full-run 独立评分汇总",
        "",
        "评分来源为 `independent_scoring/score_full_run.py`：规则基础分 + LLM 语义复核；不采用 Agent 内置 `[Scorer]` 自评分作为最终分数。",
        "",
    ]
    if fallback_rows:
        lines.extend([
            "> 说明：本次存在 LLM 复核不可用的数据集，因此表中的 fallback 分数是规则层临时参考，不应等同于正式 DeepSeek 语义复核分。",
            f"> LLM 不可用原因示例：{cell(first_error, 420) if first_error else '未返回可用 LLM 复核结果'}",
            "",
        ])
    lines.extend([
        "| Dataset | Final | Rule | LLM adj | Mode | Confidence | Report | Major penalties |",
        "|---|---:|---:|---:|---|---|---|---|",
    ])
    for row in sorted(rows, key=lambda r: r["dataset"]):
        lines.append(
            f"| {cell(row['dataset'])} | {cell(row['final_score'])} | {cell(row['rule_total'])} | "
            f"{cell(row['llm_adjustment'])} | {cell(row['scoring_mode'])} | {cell(row.get('confidence', ''))} | "
            f"`{cell(row.get('report_path', ''), 360)}` | {cell(row.get('major_penalties', []))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently score V3 scProteomics reports.")
    parser.add_argument("--examples-root", type=Path, default=PROJECT_DIR / "examples" / "V3")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    parser.add_argument("--skip-llm", action="store_true", help="Use rule-only fallback scoring and do not call the LLM review layer.")
    parser.add_argument("--score-model", default=None, help="Override SCORE_MODEL for this scoring run without editing .env.")
    parser.add_argument("--llm-timeout-sec", type=float, default=180.0, help="Per LLM scoring call timeout when supported by the OpenAI-compatible client.")
    parser.add_argument("--max-llm-adjust", type=float, default=5.0, help="Maximum absolute LLM calibration adjustment.")
    args = parser.parse_args()

    folders = [p for p in sorted(args.examples_root.iterdir()) if is_executable_dataset(p)]
    if args.datasets:
        wanted = set(args.datasets)
        folders = [p for p in folders if p.name in wanted]
    rows = [
        score_one(
            folder,
            args.runs_root,
            args.details_dir,
            skip_llm=args.skip_llm,
            score_model=args.score_model,
            llm_timeout_sec=args.llm_timeout_sec,
            max_llm_adjust=args.max_llm_adjust,
        )
        for folder in folders
    ]
    rows = sorted(rows, key=lambda row: row["dataset"])
    payload = {
        "dataset_count": len(rows),
        "scoring_system": "independent_rule_plus_llm_review" if not args.skip_llm else "independent_rule_only_fallback",
        "score_model_override": args.score_model or "",
        "llm_timeout_sec": args.llm_timeout_sec,
        "max_llm_adjust": args.max_llm_adjust,
        "score_dimensions": {
            "scientific_coverage": 35,
            "current_matrix_evidence": 20,
            "boundary_compliance": 15,
            "artifact_reproducibility": 15,
            "chinese_report_quality": 10,
            "penalty_credit": 5,
        },
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tsv(rows, args.out_tsv)
    write_markdown(rows, args.out_md)
    print(json.dumps({"dataset_count": len(rows), "out_json": str(args.out_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
