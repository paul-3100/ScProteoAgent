import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import report_language  # noqa: E402  (repo root; the shared zh/en registry)

# --------------------------------------------------------------------------- language
# The wording probes below are the words that identify each structure in the report of the
# active language. The zh value of every entry is the released literal copied byte for byte,
# so a zh run probes exactly what it probed before; the en value is the equivalent English
# wording. Heading and token probes shared with the rest of the engine live in the "core"
# namespace of the same registry (report_language.canonical_headings / story_headings /
# core.tok_*); only the validator-specific probes are registered here.
VALIDATE_TEXT = {
    "banned_scoring_evidence": ("评分证据", "scoring evidence"),
    "banned_scoring_coverage": ("评分覆盖", "scoring coverage"),
    "banned_scoring_standard_coverage": ("评分标准覆盖", "scoring standard coverage"),
    "banned_easy_to_lose_points": ("容易失分", "easy to lose points"),
    "banned_score_farming": ("刷分", "score farming"),
    "banned_supplementary_boundary_note": ("补充边界说明", "supplementary boundary note"),
    "banned_mechanism_name_clue": (
        "作为机制命名的补充线索",
        "supplementary clue for naming the mechanism",
    ),
    "module_evidence_table": ("模块证据表", "module evidence table"),
    "module_word": ("模块", "module"),
    "confidence_word": ("置信度", "confidence"),
    "offline_enrichment_section": (
        "### 离线富集补充表",
        "### Offline Enrichment Supplement Table",
    ),
    "enrichment_family_header": (
        "| 语义家族 | 代表条目 | 对比与方向 | p.adjust/FDR | Count | 相关效应量 | 解读 |",
        "| Semantic family | Representative entry | Contrast and direction | p.adjust/FDR | "
        "Count | Effect size | Interpretation |",
    ),
    "enrichment_family_cell": ("语义家族", "semantic family"),
    "legacy_enrichment_header": (
        "| 证据来源 | 对比 | 方向 | 富集条目 |",
        "| Evidence source | Contrast | Direction | Enriched entries |",
    ),
}
report_language.register("validate", VALIDATE_TEXT)


def probe(lang, key):
    """Wording probe for the requested language (the registry never falls back to Chinese)."""
    return report_language.t_in(lang, key)


def run_record_language(run_dir):
    """The language recorded by the run, or None. A pre-feature run has no field."""
    for name in ("run_metadata.json", "parameters.json"):
        payload = load_json(Path(run_dir) / name)
        if isinstance(payload, dict) and "report_language" in payload:
            try:
                return report_language.normalize(payload.get("report_language"))
            except report_language.UnknownReportLanguage:
                continue
    return None


def resolve_report_language(cli_value, run_dir):
    """(language, source). An explicit flag wins; otherwise the run record; otherwise zh."""
    if cli_value:
        return report_language.normalize(cli_value), "cli"
    recorded = run_record_language(run_dir)
    if recorded:
        return recorded, "run_record"
    return report_language.DEFAULT_LANGUAGE, "default"


# Historical check identifiers that are evaluated against the active language. The identifiers
# are kept for continuity (see report_language core note_historical_check_names); the wording
# each one is evaluated against is listed here so an operator can see the active-language
# expectation with --explain.
FAILURE_HINTS = (
    ("report_missing_chinese_executive_summary", ("core.h_executive", "core.s0")),
    ("report_missing_chinese_scoring_evidence_first_screen", ("core.h_scoring", "core.task_word")),
    ("report_missing_chinese_main_findings", ("core.h_findings", "core.task_word")),
    ("report_missing_chinese_evidence_boundary", ("core.h_boundary", "core.story_conclusion")),
    ("report_missing_chinese_evidence_appendix", ("core.h_appendix", "core.story_assets")),
    ("v3_report_missing_story_first_headings", ("core.s0", "core.s1", "core.s6", "core.s7", "core.s8")),
    ("report_missing_core_contrast_table", ("core.tok_core_contrast_table",)),
    ("missing_confidence_labels", ("validate.confidence_word",)),
    ("pispa_module_table_missing_statistical_basis", ("validate.module_evidence_table",)),
    ("pispa_offline_enrichment_not_semantic_family_table", ("validate.enrichment_family_header",)),
    ("pispa_legacy_offline_enrichment_table_header", ("validate.legacy_enrichment_header",)),
    ("pispa_task4_missing", ("core.tok_representative_header", "validate.module_word", "core.tok_followup")),
    ("pispa_missing_story_element", ("core.tok_heuristic", "core.tok_followup")),
)


SECRET_PATTERNS = {
    "openai_or_deepseek_key": re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{10,}\b"),
    "google_key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    "bearer_token": re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
}


def fs_path(path: Path) -> str:
    text = str(path)
    if path.exists() or len(str(path.resolve())) < 240:
        return text
    resolved = str(path.resolve())
    if not resolved.startswith("\\\\?\\"):
        if resolved.startswith("\\\\"):
            resolved = "\\\\?\\UNC\\" + resolved.lstrip("\\")
        else:
            resolved = "\\\\?\\" + resolved
    return resolved


def path_exists(path: Path) -> bool:
    return path.exists() or Path(fs_path(path)).exists()


def read_text(path: Path) -> str:
    return Path(fs_path(path)).read_text(encoding="utf-8", errors="ignore") if path_exists(path) else ""


def load_jsonl(path: Path):
    rows = []
    if not path_exists(path):
        return rows
    for line in Path(fs_path(path)).read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def load_json(path: Path):
    if not path_exists(path):
        return {}
    try:
        return json.loads(Path(fs_path(path)).read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def has_h2(text: str, heading: str) -> bool:
    return bool(re.search(rf"(?m)^## {re.escape(heading)}\s*$", text))


def has_story_task_heading(text: str, lang: str = "zh") -> bool:
    return bool(re.search(r"(?m)^## \d+\.\s*" + re.escape(probe(lang, "core.task_word")), text))


def story_heading_re(lang: str, index: int):
    """The "## <n>. <story heading>" line of the active language, as a compiled pattern."""
    heading = report_language.t_in(lang, "core.s%d" % index)
    title = heading.split(".", 1)[1].strip() if "." in heading else heading
    return re.compile(r"(?m)^## \d+\.\s*" + re.escape(title) + r"\s*$")


def has_story_first_structure(text: str, lang: str = "zh") -> bool:
    return (
        has_h2(text, report_language.t_in(lang, "core.s0"))
        and has_h2(text, report_language.t_in(lang, "core.s1"))
        and has_story_task_heading(text, lang)
        and bool(story_heading_re(lang, 6).search(text))
        and bool(story_heading_re(lang, 7).search(text))
        and bool(story_heading_re(lang, 8).search(text))
    )


CHINESE_HEADINGS = {
    "executive": "一、执行摘要",
    "scoring": "二、核心证据首页",
    "findings": "三、主要发现",
    "boundary": "四、证据边界",
    "appendix": "五、可复核证据附录",
}

CLAUDE_STYLE_HEADINGS = [
    "0. 关键结论速览",
    "1. 数据与预处理",
    "2. 任务一：聚类与迁移状态的对应",
    "3. 任务二：迁移相关 Cluster 的功能模块",
    "4. 任务三：骨架调控候选",
    "5. 任务四：对照群体的隐性差异",
    "6. 总结与启发式推测",
    "7. 可复核资产清单",
    "8. 结论边界与方法局限",
]

V3_MAIN_DATASETS = {
    "Nat_Commun_PiSPA_2024",
    "Nat_Methods_iPSC_2025",
    "Nat_Biotech_Brain_2026",
    "Nat_Commun_Carr_2024",
    "Nat_Methods_pSCoPE_2023",
    "Nat_Methods_DVP_2023",
    "Nat_Commun_SCPro_2024",
    "Nat_Commun_Nociceptor_2026",
    "Science_BloodCell_2025",
    "Cell_TurnoverDynamics_2025",
    "Nat_Commun_ProteinLeakage_2025",
}

LEGACY_PRIMARY_HEADINGS = [
    "Executive Result Summary",
    "Scoring Evidence First Screen",
    "Main Findings",
    "Evidence Boundary",
    "Evidence Appendix",
]


def main_report_text_for_language_check(text: str) -> str:
    markers = [
        f"\n## {CHINESE_HEADINGS['appendix']}",
        f"\n## {CLAUDE_STYLE_HEADINGS[7]}",
        "\n## Evidence Appendix",
        "\n## Scoring Evidence Appendix",
        "\n## Protein Leakage Evidence Appendix",
    ]
    split_at = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            split_at = min(split_at, idx)
    return text[:split_at]


def chinese_char_ratio(text: str) -> float:
    body = main_report_text_for_language_check(text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    latin = len(re.findall(r"[A-Za-z]", body))
    denom = cjk + latin
    return round(cjk / denom, 4) if denom else 0.0


def latin_char_ratio(text: str) -> float:
    """The same body-language share for an en report: the Latin share of the main body."""
    body = main_report_text_for_language_check(text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    latin = len(re.findall(r"[A-Za-z]", body))
    denom = cjk + latin
    return round(latin / denom, 4) if denom else 0.0


def enrichment_header_present(text: str, lang: str) -> bool:
    """The semantic-family table header of the active language.

    A zh run requires the released header literally. For en the family cell wording and the
    p.adjust column are checked case-insensitively instead of literally, because that table is
    emitted by another module and only the family cell and the p.adjust column are fixed.
    """
    if probe(lang, "validate.enrichment_family_header") in text:
        return True
    if lang == "zh":
        return False
    cell = probe(lang, "validate.enrichment_family_cell").lower()
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("|") and cell in low and "p.adjust" in low:
            return True
    return False


def has_long_english_paragraph(text: str) -> bool:
    body = main_report_text_for_language_check(text)
    for line in body.splitlines():
        stripped = line.strip()
        if len(stripped) < 160:
            continue
        lower = stripped.lower()
        if (
            stripped.startswith(("|", "-", "*", "`"))
            or "\\" in stripped
            or "/" in stripped
            or ".csv" in lower
            or ".json" in lower
            or ".png" in lower
            or "current_matrix" in lower
            or "offline_enrichment" in lower
            or "external_annotation" in lower
        ):
            continue
        cjk = len(re.findall(r"[\u4e00-\u9fff]", stripped))
        latin = len(re.findall(r"[A-Za-z]", stripped))
        if latin > 120 and cjk < 3:
            return True
    return False


def validate_run(run_dir: Path, lang=None) -> dict:
    # The historical check identifiers below (report_missing_chinese_*) are kept unchanged for a
    # zh report and are reused for an en report; each one is evaluated against the headings and
    # wording of the active language, which is what report_language reserves that name for.
    lang, language_source = resolve_report_language(lang, run_dir)
    headings = report_language.canonical_headings(lang)
    reports = (
        ([run_dir / "report.md"] if path_exists(run_dir / "report.md") else [])
        + ([run_dir / "final_report.md"] if path_exists(run_dir / "final_report.md") else [])
        + sorted(run_dir.glob("analysis_report_*.md"))
        + sorted(run_dir.glob("final_analysis_report_*.md"))
        + sorted(run_dir.glob("output_report_*.md"))
        + sorted(run_dir.glob("final_output_report.md"))
    )
    reports = [p for p in reports if "evidence_appendix" not in p.name and "conclusions" not in p.name]
    report_text = "\n".join(read_text(path) for path in reports)
    ledger = load_jsonl(run_dir / "evidence_ledger.jsonl")
    external = load_jsonl(run_dir / "external_knowledge.jsonl")
    failures = []
    warnings = []

    if not reports:
        failures.append("missing_report")
    if not path_exists(run_dir / "analysis_design.used.yaml"):
        failures.append("missing_analysis_design_used")
    if not path_exists(run_dir / "analysis_design.inferred.yaml"):
        failures.append("missing_analysis_design_inferred")
    if not ledger:
        failures.append("missing_or_empty_evidence_ledger")
    evidence_dir = run_dir / "evaluation_evidence"
    requirements_candidates = [
        run_dir / "evaluation_requirements.used.json",
        run_dir / "user_visible_requirements.used.json",
        run_dir / "evaluator_requirements.used.json",
    ]
    requirements_used = next(
        (path for path in requirements_candidates if path_exists(path)),
        requirements_candidates[0],
    )
    required_evidence = [
        requirements_used,
        evidence_dir / "group_composition_qc.csv",
        evidence_dir / "core_story_evidence.csv",
        evidence_dir / "candidate_protein_evidence.csv",
        evidence_dir / "curated_module_group_summary.csv",
        evidence_dir / "curated_module_sample_scores.csv",
        evidence_dir / "dataset_recipe_evidence.csv",
        evidence_dir / "scoring_standard_coverage.csv",
        run_dir / "report_self_check.json",
    ]
    external_annotation = load_json(evidence_dir / "external_annotation_asd_ndd_local.json")
    if external_annotation.get("applicable"):
        required_evidence.append(evidence_dir / "external_annotation_asd_ndd_local.csv")
    for path in required_evidence:
        if not path_exists(path):
            failures.append(f"missing_scoring_evidence:{path.name}")
    req = load_json(requirements_used)
    dataset = str(req.get("dataset", "")) if isinstance(req, dict) else ""
    if dataset in V3_MAIN_DATASETS and list(run_dir.glob("output_report_*.md")):
        failures.append("legacy_output_report_alias_present")
    has_story_first = has_story_first_structure(report_text, lang)
    has_exact_pispa_style = all(has_h2(report_text, heading)
                                for heading in report_language.story_headings(lang))
    if not has_h2(report_text, headings["executive"]) and not has_h2(report_text, report_language.t_in(lang, "core.s0")):
        failures.append("report_missing_chinese_executive_summary")
    if not has_h2(report_text, headings["scoring"]) and not has_story_task_heading(report_text, lang):
        failures.append("report_missing_chinese_scoring_evidence_first_screen")
    if not has_h2(report_text, headings["findings"]) and not has_story_task_heading(report_text, lang):
        failures.append("report_missing_chinese_main_findings")
    if not has_h2(report_text, headings["boundary"]) and not story_heading_re(lang, 8).search(report_text):
        failures.append("report_missing_chinese_evidence_boundary")
    if not has_h2(report_text, headings["appendix"]) and not story_heading_re(lang, 7).search(report_text):
        failures.append("report_missing_chinese_evidence_appendix")
    if dataset in V3_MAIN_DATASETS and not has_story_first:
        failures.append("v3_report_missing_story_first_headings")
    if dataset == "Nat_Commun_PiSPA_2024" and not has_exact_pispa_style:
        failures.append("pispa_report_missing_claude_style_headings")
    if any(has_h2(report_text, heading) for heading in LEGACY_PRIMARY_HEADINGS):
        failures.append("legacy_english_primary_headings_present")
    # The two body-language checks keep their historical identifiers in both languages; for en
    # the ratio is the Latin share and the long-paragraph check looks for a purely Chinese
    # paragraph, so the identifier means "the body-language check of the active language".
    if lang == "zh":
        language_ratio = chinese_char_ratio(report_text)
        if language_ratio < 0.10:
            failures.append(f"report_chinese_body_ratio_low:{language_ratio}")
        if has_long_english_paragraph(report_text):
            failures.append("long_english_paragraph_in_main_report")
    else:
        body = main_report_text_for_language_check(report_text)
        latin_ratio = latin_char_ratio(body)
        if not report_language.body_language_ok(body, "en"):
            failures.append(f"report_chinese_body_ratio_low:{latin_ratio}")
        if report_language.wrong_language_paragraph_present(report_text, "en"):
            failures.append("long_english_paragraph_in_main_report")
    if re.search(r"```json\s*(?:(?!```).){1000,}(?:\"tool\"|\"result\"|\"tool_call_id\")", report_text, re.I | re.S):
        failures.append("raw_tool_json_in_report")
    if "## Task" in report_text or "[Analyzer] Local Fallback Summary" in report_text:
        failures.append("long_tool_log_or_task_text_in_report")
    banned_terms = [
        probe(lang, "validate.banned_scoring_evidence"),
        probe(lang, "validate.banned_scoring_coverage"),
        probe(lang, "validate.banned_scoring_standard_coverage"),
        probe(lang, "validate.banned_easy_to_lose_points"),
        probe(lang, "validate.banned_score_farming"),
        "validator",
        "report_self_check",
        probe(lang, "validate.banned_supplementary_boundary_note"),
        "output_report_1",
        probe(lang, "validate.banned_mechanism_name_clue"),
    ]
    for term in banned_terms:
        if term in report_text:
            failures.append(f"user_report_contains_meta_term:{term}")
    if dataset == "Nat_Commun_PiSPA_2024":
        if not enrichment_header_present(report_text, lang) and "offline_enrichment" in report_text:
            failures.append("pispa_offline_enrichment_not_semantic_family_table")
        if probe(lang, "validate.legacy_enrichment_header") in report_text:
            failures.append("pispa_legacy_offline_enrichment_table_header")
        enrichment_idx = report_text.find(probe(lang, "validate.offline_enrichment_section"))
        if enrichment_idx >= 0:
            next_heading = report_text.find("\n## ", enrichment_idx + 1)
            next_subheading = report_text.find("\n### ", enrichment_idx + 1)
            candidates = [idx for idx in [next_heading, next_subheading] if idx > enrichment_idx]
            end_idx = min(candidates) if candidates else len(report_text)
            enrichment_text = report_text[enrichment_idx:end_idx]
            family_counts = {}
            for line in enrichment_text.splitlines():
                if (not line.startswith("|") or line.startswith("|---")
                        or probe(lang, "validate.enrichment_family_cell") in line):
                    continue
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 7:
                    family_counts[cells[0]] = family_counts.get(cells[0], 0) + 1
            for family, count in family_counts.items():
                if count > 3:
                    failures.append(f"pispa_enrichment_family_over_represented:{family}:{count}")
        if probe(lang, "validate.module_evidence_table") in report_text and "FDR: NA" not in report_text:
            failures.append("pispa_module_table_missing_statistical_basis")
        task4_idx = report_text.find("## " + report_language.t_in(lang, "core.s5"))
        next_idx = report_text.find("## " + report_language.t_in(lang, "core.s6"),
                                    task4_idx if task4_idx >= 0 else 0)
        task4_text = report_text[task4_idx:next_idx if next_idx > task4_idx else len(report_text)] if task4_idx >= 0 else ""
        for token in ["Cluster 2", "Cluster 3", "C2/C3",
                      report_language.t_in(lang, "core.tok_representative_header"),
                      probe(lang, "validate.module_word"), "extension",
                      report_language.t_in(lang, "core.tok_followup")]:
            if token not in task4_text:
                failures.append(f"pispa_task4_missing:{token}")

    for label in ["current_matrix", "offline_enrichment"]:
        if label not in report_text:
            warnings.append(f"report_missing_label:{label}")
    if external_annotation.get("applicable") and "external_annotation" not in report_text:
        failures.append("external_annotation_used_without_report_label")
    if "confidence:" not in report_text and not re.search(r"confidence\s*[:：]\s*(high|moderate|low)", report_text, re.I):
        if probe(lang, "validate.confidence_word") not in report_text:
            failures.append("missing_confidence_labels")
    if "normalized/uncorrected fallback" in report_text and re.search(r"successfully\s+batch[- ]corrected|成功.*批次校正", report_text, re.I):
        failures.append("fallback_described_as_successful_batch_correction")
    if re.search(r"exploratory_not_fdr_significant.{0,120}(significant|显著)", report_text, re.I | re.S):
        failures.append("exploratory_enrichment_described_as_significant")

    core_story_text = read_text(evidence_dir / "core_story_evidence.csv")
    if not core_story_text.strip():
        failures.append("missing_core_story_evidence")
    story_requirements = {
        "Nat_Commun_PiSPA_2024": [
            "Cluster_1_vs_Cluster_2",
            "Cluster_1_vs_Cluster_3",
            "Cluster_2_vs_Cluster_3",
            "Migrated_vs_Control",
            "EZR",
            "MSN",
            "MYL9",
            "CDC42",
            "RAC1",
            "RHOA",
            "TLN1",
            "VCL",
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
    }
    for token in story_requirements.get(dataset, []):
        if token not in core_story_text and token not in report_text:
            failures.append(f"missing_required_core_contrast:{token}")
    if dataset in story_requirements:
        if report_language.t_in(lang, "core.tok_core_contrast_table") not in report_text:
            failures.append("report_missing_core_contrast_table")
        if not re.search(r"logFC|adj\.P\.Val|FDR", report_text, re.I):
            failures.append("report_missing_logfc_fdr_story_table")
    if dataset in V3_MAIN_DATASETS:
        if not path_exists(run_dir / "assets.txt"):
            failures.append("missing_assets_txt")
        if not path_exists(run_dir / "figures.md"):
            failures.append("missing_figures_md")
        if not path_exists(run_dir / "figures_preview_local.md"):
            failures.append("missing_figures_preview_local_md")
    if dataset == "Nat_Commun_PiSPA_2024":
        for token in ["Rho GTPase", "ERM", "myosin", "talin", "vinculin",
                      report_language.t_in(lang, "core.tok_heuristic"),
                      report_language.t_in(lang, "core.tok_followup")]:
            if token not in report_text:
                failures.append(f"pispa_missing_story_element:{token}")
        if "up_in_display_group_a" in report_text or "up_in_display_group_b" in report_text:
            failures.append("pispa_internal_direction_label_in_report")
    if dataset == "Nat_Commun_Carr_2024" and "DMSO_vs_LPS" in report_text and "LPS_vs_DMSO" not in report_text:
        failures.append("carr_direction_not_standardized_to_lps_vs_dmso")
    if dataset == "Cell_TurnoverDynamics_2025" and "Control_vs_Cycloheximide" in report_text and "Cycloheximide_High_vs_Control" not in report_text:
        failures.append("turnover_direction_not_standardized_to_cycloheximide_vs_control")

    if external:
        if "external_literature" not in report_text and any(r.get("evidence_source") == "external_literature" for r in external):
            failures.append("external_literature_used_without_report_label")
        if "drug_database" not in report_text and any(r.get("evidence_source") == "drug_database" for r in external):
            failures.append("drug_database_used_without_report_label")
    if re.search(r"\b(ASD|NDD|autism|neurodevelopmental)\b", report_text, re.I) and "external_annotation" not in report_text:
        failures.append("disease_annotation_without_external_annotation_boundary")

    figure_candidates = [
        run_dir / "figures" / "figure_manifest.json",
        run_dir / "figure_manifest.json",
        run_dir / "visualize_results" / "figure_manifest.json",
        run_dir / "visualize_results" / "figure_index.md",
        run_dir / "figure_index.md",
    ]
    if not any(path_exists(path) for path in figure_candidates):
        warnings.append("missing_figure_manifest_or_index")

    report_self_check = load_json(run_dir / "report_self_check.json")
    if report_self_check and report_self_check.get("status") != "pass":
        failures.append("report_self_check_failed")

    scan_paths = reports + [
        run_dir / "record_file.md",
        run_dir / "run_events.jsonl",
        run_dir / "run_status.json",
        run_dir / "evidence_ledger.jsonl",
        run_dir / "external_knowledge.jsonl",
    ]
    for path in scan_paths:
        text = read_text(path)
        for name, rx in SECRET_PATTERNS.items():
            if rx.search(text):
                failures.append(f"secret_like_hit:{name}:{path.name}")

    return {
        "run_dir": str(run_dir),
        "status": "pass" if not failures else "fail",
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
        "reports": [str(p) for p in reports],
        "evidence_rows": len(ledger),
        "external_knowledge_rows": len(external),
    }


def explain_failures(failures, lang, source=""):
    """For each failing check id, the active-language wording it was evaluated against.

    The identifiers are language-neutral and are kept for continuity; this listing is what makes
    them readable in either language, and it never invents wording: every line is the registry
    value of the probe the check actually used.
    """
    lines = ["report language: %s%s" % (lang, (" (source: %s)" % source) if source else "")]
    seen = set()
    for failure in failures:
        code = str(failure).split(":", 1)[0]
        if code in seen:
            continue
        keys = [keys for name, keys in FAILURE_HINTS if name == code]
        if not keys:
            continue
        seen.add(code)
        lines.append("%s -> %s" % (code, "; ".join(report_language.t_in(lang, key)
                                                   for key in keys[0])))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--language", choices=list(report_language.SUPPORTED), default=None,
                        help="report language; default: the language recorded by the run, else zh")
    parser.add_argument("--explain", action="store_true",
                        help="also print the active-language wording behind each failing check")
    args = parser.parse_args()
    lang, language_source = resolve_report_language(args.language, args.run_dir)
    result = validate_run(args.run_dir, lang)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.explain:
        for line in explain_failures(result.get("failures") or [], lang, language_source):
            print(line)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
