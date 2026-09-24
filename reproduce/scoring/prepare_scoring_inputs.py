#!/usr/bin/env python3
"""Assemble the scoring-only input tree for the frozen rule scorer.

What the frozen scorer reads
============================
reproduce/scoring/score_full_run.py (frozen, sha256
2B37C1835D397FBF132BAD9DEB8900220BCD3ABCD299ECEE8112CC0B3BEEA74B) is executed unchanged.
For every study it reads five files from <examples-root>/<study>/:

  * the three generation inputs the archive ships in inputs/<study>/ --
    ProteinQuant.csv, SampleInfo.csv, user_input.txt
    (is_executable_dataset requires all of them, otherwise the dataset is skipped and no row
    is produced at all);
  * the two independent evaluation references from evaluation_references/<study>/ --
    ground_truth.txt and grading_standard.txt, read by score_rule_based at lines 355-359
    (ground_truth = read_text(dataset_dir / "ground_truth.txt"), grading =
    read_text(dataset_dir / "grading_standard.txt")).

For every (study, system) cell it reads one run directory below --runs-root: the directory
whose name contains the study name and that carries run_status.json plus a report
(latest_run_for_dataset -> load_report). Inside that directory the rule layer additionally
reads run_status.json, report_validation.json, report_self_check.json and the
evaluation_evidence artefacts, and it scores the presence and absence of further files.

The archive keeps those three trees apart:

    <archive>/inputs/<study>/{ProteinQuant.csv,SampleInfo.csv,user_input.txt}
    <archive>/evaluation_references/<study>/{ground_truth.txt,grading_standard.txt}
    <archive>/scoring/runs/<system>/<run-directory>/

so a replay command cannot point --examples-root at the archive directly. This script
assembles the three trees, byte for byte, into one throw-away scoring root:

    <out-dir>/examples/<study>/                       eleven directories, five files each
    <out-dir>/scoring/runs/<system>/<run-directory>/  88 mirrored run directories
    <out-dir>/scoring_input_manifest.json (.tsv)      every copied file with its sha256
    <out-dir>/SCORING_ONLY_README.txt                 the audit note below

SCORING-ONLY TREE, NEVER AN AGENT INPUT
=======================================
<out-dir>/examples carries ground_truth.txt and grading_standard.txt. It exists so that the
frozen scorer can read the same bytes that were scored. Do not point main_agent.py or any
live generation run at it, and keep it in a temporary directory.

Explicit registries, no directory-name guessing
===============================================
The eleven studies with their five required files and the 88 (system, study) run directories
are literal tables in this file. Nothing is discovered by listing the archive, so a directory
that merely looks newest is never selected. A required file that is missing stops the script
with a non-zero exit status; an empty string or an empty file is never substituted for it.
Absent files inside a run directory stay absent, because the rule layer reads part of its
score from file presence.

Usage:
  py -3 reproduce/scoring/prepare_scoring_inputs.py --archive <archive>/data --out-dir <tmp>/scoring_inputs
  py -3 reproduce/scoring/prepare_scoring_inputs.py --archive <archive>/data --out-dir <tmp>/scoring_inputs --check-only

Exit codes: 0 = assembled (or --check-only found no problem), 1 = usage or layout error,
2 = a registered input file is missing or unreadable, 3 = the archive contradicts the
registry tables in this file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

STUDIES = (
    "Cell_TurnoverDynamics_2025",
    "Nat_Biotech_Brain_2026",
    "Nat_Commun_Carr_2024",
    "Nat_Commun_Nociceptor_2026",
    "Nat_Commun_PiSPA_2024",
    "Nat_Commun_ProteinLeakage_2025",
    "Nat_Commun_SCPro_2024",
    "Nat_Methods_DVP_2023",
    "Nat_Methods_iPSC_2025",
    "Nat_Methods_pSCoPE_2023",
    "Science_BloodCell_2025",
)

SYSTEMS = (
    "Bioagent",
    "Biomni",
    "CellVoyager",
    "Claudecode",
    "Codex",
    "Hermes",
    "Proposed (gpt-5.5)",
    "SpatialAgent",
)

# (system, study) -> run directory name inside <archive>/scoring/runs/<system>/.
# The archive is not uniform: the Proposal run directories carry a model/date
# suffix, so every name is listed explicitly instead of being derived.
RUN_DIRECTORIES = {
    "Bioagent": {
        "Cell_TurnoverDynamics_2025": "Cell_TurnoverDynamics_2025",
        "Nat_Biotech_Brain_2026": "Nat_Biotech_Brain_2026",
        "Nat_Commun_Carr_2024": "Nat_Commun_Carr_2024",
        "Nat_Commun_Nociceptor_2026": "Nat_Commun_Nociceptor_2026",
        "Nat_Commun_PiSPA_2024": "Nat_Commun_PiSPA_2024",
        "Nat_Commun_ProteinLeakage_2025": "Nat_Commun_ProteinLeakage_2025",
        "Nat_Commun_SCPro_2024": "Nat_Commun_SCPro_2024",
        "Nat_Methods_DVP_2023": "Nat_Methods_DVP_2023",
        "Nat_Methods_iPSC_2025": "Nat_Methods_iPSC_2025",
        "Nat_Methods_pSCoPE_2023": "Nat_Methods_pSCoPE_2023",
        "Science_BloodCell_2025": "Science_BloodCell_2025",
    },
    "Biomni": {
        "Cell_TurnoverDynamics_2025": "Cell_TurnoverDynamics_2025",
        "Nat_Biotech_Brain_2026": "Nat_Biotech_Brain_2026",
        "Nat_Commun_Carr_2024": "Nat_Commun_Carr_2024",
        "Nat_Commun_Nociceptor_2026": "Nat_Commun_Nociceptor_2026",
        "Nat_Commun_PiSPA_2024": "Nat_Commun_PiSPA_2024",
        "Nat_Commun_ProteinLeakage_2025": "Nat_Commun_ProteinLeakage_2025",
        "Nat_Commun_SCPro_2024": "Nat_Commun_SCPro_2024",
        "Nat_Methods_DVP_2023": "Nat_Methods_DVP_2023",
        "Nat_Methods_iPSC_2025": "Nat_Methods_iPSC_2025",
        "Nat_Methods_pSCoPE_2023": "Nat_Methods_pSCoPE_2023",
        "Science_BloodCell_2025": "Science_BloodCell_2025",
    },
    "CellVoyager": {
        "Cell_TurnoverDynamics_2025": "Cell_TurnoverDynamics_2025",
        "Nat_Biotech_Brain_2026": "Nat_Biotech_Brain_2026",
        "Nat_Commun_Carr_2024": "Nat_Commun_Carr_2024",
        "Nat_Commun_Nociceptor_2026": "Nat_Commun_Nociceptor_2026",
        "Nat_Commun_PiSPA_2024": "Nat_Commun_PiSPA_2024",
        "Nat_Commun_ProteinLeakage_2025": "Nat_Commun_ProteinLeakage_2025",
        "Nat_Commun_SCPro_2024": "Nat_Commun_SCPro_2024",
        "Nat_Methods_DVP_2023": "Nat_Methods_DVP_2023",
        "Nat_Methods_iPSC_2025": "Nat_Methods_iPSC_2025",
        "Nat_Methods_pSCoPE_2023": "Nat_Methods_pSCoPE_2023",
        "Science_BloodCell_2025": "Science_BloodCell_2025",
    },
    "Claudecode": {
        "Cell_TurnoverDynamics_2025": "Cell_TurnoverDynamics_2025",
        "Nat_Biotech_Brain_2026": "Nat_Biotech_Brain_2026",
        "Nat_Commun_Carr_2024": "Nat_Commun_Carr_2024",
        "Nat_Commun_Nociceptor_2026": "Nat_Commun_Nociceptor_2026",
        "Nat_Commun_PiSPA_2024": "Nat_Commun_PiSPA_2024",
        "Nat_Commun_ProteinLeakage_2025": "Nat_Commun_ProteinLeakage_2025",
        "Nat_Commun_SCPro_2024": "Nat_Commun_SCPro_2024",
        "Nat_Methods_DVP_2023": "Nat_Methods_DVP_2023",
        "Nat_Methods_iPSC_2025": "Nat_Methods_iPSC_2025",
        "Nat_Methods_pSCoPE_2023": "Nat_Methods_pSCoPE_2023",
        "Science_BloodCell_2025": "Science_BloodCell_2025",
    },
    "Codex": {
        "Cell_TurnoverDynamics_2025": "Cell_TurnoverDynamics_2025",
        "Nat_Biotech_Brain_2026": "Nat_Biotech_Brain_2026",
        "Nat_Commun_Carr_2024": "Nat_Commun_Carr_2024",
        "Nat_Commun_Nociceptor_2026": "Nat_Commun_Nociceptor_2026",
        "Nat_Commun_PiSPA_2024": "Nat_Commun_PiSPA_2024",
        "Nat_Commun_ProteinLeakage_2025": "Nat_Commun_ProteinLeakage_2025",
        "Nat_Commun_SCPro_2024": "Nat_Commun_SCPro_2024",
        "Nat_Methods_DVP_2023": "Nat_Methods_DVP_2023",
        "Nat_Methods_iPSC_2025": "Nat_Methods_iPSC_2025",
        "Nat_Methods_pSCoPE_2023": "Nat_Methods_pSCoPE_2023",
        "Science_BloodCell_2025": "Science_BloodCell_2025",
    },
    "Hermes": {
        "Cell_TurnoverDynamics_2025": "Cell_TurnoverDynamics_2025",
        "Nat_Biotech_Brain_2026": "Nat_Biotech_Brain_2026",
        "Nat_Commun_Carr_2024": "Nat_Commun_Carr_2024",
        "Nat_Commun_Nociceptor_2026": "Nat_Commun_Nociceptor_2026",
        "Nat_Commun_PiSPA_2024": "Nat_Commun_PiSPA_2024",
        "Nat_Commun_ProteinLeakage_2025": "Nat_Commun_ProteinLeakage_2025",
        "Nat_Commun_SCPro_2024": "Nat_Commun_SCPro_2024",
        "Nat_Methods_DVP_2023": "Nat_Methods_DVP_2023",
        "Nat_Methods_iPSC_2025": "Nat_Methods_iPSC_2025",
        "Nat_Methods_pSCoPE_2023": "Nat_Methods_pSCoPE_2023",
        "Science_BloodCell_2025": "Science_BloodCell_2025",
    },
    "Proposed (gpt-5.5)": {
        "Cell_TurnoverDynamics_2025": "Cell_TurnoverDynamics_2025_gpt-5.5_20260818",
        "Nat_Biotech_Brain_2026": "Nat_Biotech_Brain_2026_gpt-5.5_20260818",
        "Nat_Commun_Carr_2024": "Nat_Commun_Carr_2024_gpt-5.5_20260818",
        "Nat_Commun_Nociceptor_2026": "Nat_Commun_Nociceptor_2026_gpt-5.5_20260818",
        "Nat_Commun_PiSPA_2024": "Nat_Commun_PiSPA_2024_gpt-5.5_20260818",
        "Nat_Commun_ProteinLeakage_2025": "Nat_Commun_ProteinLeakage_2025_gpt-5.5_20260818",
        "Nat_Commun_SCPro_2024": "Nat_Commun_SCPro_2024_gpt-5.5_20260818",
        "Nat_Methods_DVP_2023": "Nat_Methods_DVP_2023_gpt-5.5_20260818",
        "Nat_Methods_iPSC_2025": "Nat_Methods_iPSC_2025_gpt-5.5_20260818",
        "Nat_Methods_pSCoPE_2023": "Nat_Methods_pSCoPE_2023_gpt-5.5_20260818",
        "Science_BloodCell_2025": "Science_BloodCell_2025_gpt-5.5_20260818",
    },
    "SpatialAgent": {
        "Cell_TurnoverDynamics_2025": "Cell_TurnoverDynamics_2025",
        "Nat_Biotech_Brain_2026": "Nat_Biotech_Brain_2026",
        "Nat_Commun_Carr_2024": "Nat_Commun_Carr_2024",
        "Nat_Commun_Nociceptor_2026": "Nat_Commun_Nociceptor_2026",
        "Nat_Commun_PiSPA_2024": "Nat_Commun_PiSPA_2024",
        "Nat_Commun_ProteinLeakage_2025": "Nat_Commun_ProteinLeakage_2025",
        "Nat_Commun_SCPro_2024": "Nat_Commun_SCPro_2024",
        "Nat_Methods_DVP_2023": "Nat_Methods_DVP_2023",
        "Nat_Methods_iPSC_2025": "Nat_Methods_iPSC_2025",
        "Nat_Methods_pSCoPE_2023": "Nat_Methods_pSCoPE_2023",
        "Science_BloodCell_2025": "Science_BloodCell_2025",
    },
}

GENERATION_INPUTS = ("ProteinQuant.csv", "SampleInfo.csv", "user_input.txt")
EVALUATION_REFERENCES = ("ground_truth.txt", "grading_standard.txt")
REQUIRED_DATASET_FILES = GENERATION_INPUTS + EVALUATION_REFERENCES
ARCHIVE_INPUTS = "inputs"
ARCHIVE_REFERENCES = "evaluation_references"
ARCHIVE_RUNS = ("scoring", "runs")
ARCHIVE_REGISTRY_TSV = ("scoring", "FROZEN_REGISTRY.tsv")
ARCHIVE_INPUT_MAP_TSV = ("study_mapping", "STUDY_INPUT_MAP.tsv")

SCORING_ONLY_README = """SCORING-ONLY STAGING TREE -- DO NOT USE AS AGENT INPUT

Assembled by reproduce/scoring/prepare_scoring_inputs.py from
  {archive}

examples/<study>/ holds the three generation inputs (ProteinQuant.csv, SampleInfo.csv,
user_input.txt) together with the independent evaluation references (ground_truth.txt,
grading_standard.txt). Pointing main_agent.py or any live generation run at this directory
would put the evaluation references in front of the agent. Keep the tree in a temporary
directory and delete it when the replay is finished.

scoring/runs/<system>/<run-directory>/ mirrors the archived run directories, including which
files exist and which are missing: the rule layer reads part of its score from file presence,
so missing files are intentionally not recreated.

scoring_input_manifest.json and scoring_input_manifest.tsv list every copied file with its
sha256 and byte count. The archive itself was not modified.
"""


class AssemblyError(RuntimeError):
    """Raised when a path or a registry table contradicts the frozen archive layout."""


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


def as_path(path: Path) -> Path:
    return Path(fs_path(path))


def path_exists(path: Path) -> bool:
    return as_path(path).exists()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with as_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def select_report(run_dir: Path) -> Path | None:
    """Mirror of the load_report() preference order in the frozen scorer (read-only audit)."""
    preferred = [
        run_dir / "report.md",
        run_dir / "final_report.md",
        run_dir / ("analysis_report_%s.md" % run_dir.name),
        run_dir / "analysis_report_Nat_Commun_PiSPA_2024.md",
        run_dir / ("final_analysis_report_%s.md" % run_dir.name),
        run_dir / "final_analysis_report_Nat_Commun_PiSPA_2024.md",
        run_dir / "final_output_report.md",
        run_dir / "output_report_1.md",
        run_dir / "output_report_1_conclusions.md",
    ]
    preferred.extend(sorted(as_path(run_dir).glob("analysis_report_*.md")))
    preferred.extend(sorted(as_path(run_dir).glob("final_analysis_report_*.md")))
    for candidate in preferred:
        if path_exists(candidate) and "evidence_appendix" not in candidate.name and "conclusions" not in candidate.name:
            return candidate
    reports = sorted(path for path in as_path(run_dir).glob("output_report_*.md")
                     if "evidence_appendix" not in path.name)
    return reports[0] if reports else None


def planned_entries() -> list[dict]:
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for system in SYSTEMS:
        table = RUN_DIRECTORIES.get(system, {})
        for study in STUDIES:
            run_dir = table.get(study, "")
            if not run_dir:
                raise AssemblyError("registry has no run directory for %s / %s" % (system, study))
            key = (system, study)
            if key in seen:
                raise AssemblyError("registry repeats the cell %s / %s" % key)
            seen.add(key)
            entries.append({"system": system, "study": study, "run_dir": run_dir})
    if len(entries) != len(STUDIES) * len(SYSTEMS):
        raise AssemblyError("registry holds %d cells, expected %d"
                            % (len(entries), len(STUDIES) * len(SYSTEMS)))
    return entries


def resolve_archive(raw: Path) -> Path:
    archive = raw
    inputs_ok = (archive / ARCHIVE_INPUTS).is_dir()
    runs_ok = (archive / ARCHIVE_RUNS[0] / ARCHIVE_RUNS[1]).is_dir()
    if inputs_ok and runs_ok:
        return archive
    hint = ""
    nested = archive / "data"
    if (nested / ARCHIVE_INPUTS).is_dir() and (nested / ARCHIVE_RUNS[0] / ARCHIVE_RUNS[1]).is_dir():
        hint = ("; pass the archive root that holds inputs/, evaluation_references/ and "
                "scoring/runs/, i.e. %s" % nested)
    raise AssemblyError("--archive %s does not contain %s/ and %s/%s/%s%s"
                        % (archive, ARCHIVE_INPUTS, ARCHIVE_RUNS[0], ARCHIVE_RUNS[1], "", hint))


def dataset_source(archive: Path, study: str, name: str) -> Path:
    if name in GENERATION_INPUTS:
        return archive / ARCHIVE_INPUTS / study / name
    return archive / ARCHIVE_REFERENCES / study / name


def studies_in_run_name(run_dir_name: str) -> list[str]:
    return [study for study in STUDIES if study in run_dir_name]


def preflight(archive: Path, entries: list[dict], problems: list[str], warnings: list[str]) -> None:
    for entry in entries:
        run_source = archive / ARCHIVE_RUNS[0] / ARCHIVE_RUNS[1] / entry["system"] / entry["run_dir"]
        entry["run_source"] = run_source
        entry["report_file"] = ""
        entry["report_sha256"] = ""
        if not run_source.is_dir():
            problems.append("missing run directory: %s" % run_source)
            continue
        matches = studies_in_run_name(entry["run_dir"])
        if matches != [entry["study"]]:
            problems.append("run directory name %s maps to %s, expected only %s"
                            % (entry["run_dir"], matches, entry["study"]))
        if not path_exists(run_source / "run_status.json"):
            problems.append("run directory without run_status.json (the scorer would skip it): %s"
                            % run_source)
        report = select_report(run_source)
        if report is None:
            problems.append("run directory without a report the scorer can read: %s" % run_source)
        else:
            entry["report_file"] = report.name
            entry["report_sha256"] = sha256_file(report)
    for study in STUDIES:
        for name in REQUIRED_DATASET_FILES:
            source = dataset_source(archive, study, name)
            if not path_exists(source):
                problems.append("missing dataset file: %s" % source)
            elif as_path(source).stat().st_size == 0:
                warnings.append("zero-byte required file in the archive (copied unchanged): %s" % source)


def parse_braced_names(value: str) -> list[str]:
    start = value.find("{")
    end = value.find("}", start + 1)
    if start < 0 or end < 0:
        return []
    return [part.strip() for part in value[start + 1:end].split(";") if part.strip()]


def parse_hash_pairs(value: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for chunk in value.split(";"):
        if "=" not in chunk:
            continue
        name, _, digest = chunk.partition("=")
        pairs[name.strip()] = digest.strip()
    return pairs


def registered_public_report_hashes(archive: Path) -> dict[tuple[str, str, str], tuple[str, str]]:
    """Read the archive's exact original-to-public report hash registrations."""
    path = archive / "scoring" / "PUBLIC_REPORT_HASHES.tsv"
    if not path.is_file():
        return {}
    mapping = {}
    with as_path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parts = row["relative_path"].replace("\\", "/").split("/")
            if len(parts) != 6 or parts[:3] != ["data", "scoring", "runs"]:
                raise ValueError("Invalid public report hash path: %r" % row["relative_path"])
            key = (parts[3], parts[4], parts[5])
            pair = (row["sha256_original"].upper(), row["sha256_public"].upper())
            if key in mapping and mapping[key] != pair:
                raise ValueError("Conflicting public report hash registration: %r" % (key,))
            mapping[key] = pair
    return mapping


def cross_check_archive(archive: Path, entries: list[dict], contradictions: list[str],
                        warnings: list[str]) -> int:
    """Compare the registry tables in this file with registry files shipped by the archive."""
    checks = 0
    registry_tsv = archive / ARCHIVE_REGISTRY_TSV[0] / ARCHIVE_REGISTRY_TSV[1]
    by_cell = {(entry["system"], entry["study"]): entry for entry in entries}
    public_hashes = registered_public_report_hashes(archive)
    if registry_tsv.is_file():
        with as_path(registry_tsv).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        archive_cells = {(row.get("method", ""), row.get("dataset", "")): row for row in rows}
        for key in sorted(set(archive_cells) | set(by_cell)):
            checks += 1
            if key not in archive_cells:
                contradictions.append("archive registry lacks cell %s / %s" % key)
                continue
            if key not in by_cell:
                contradictions.append("registry in this script lacks cell %s / %s" % key)
                continue
            row = archive_cells[key]
            entry = by_cell[key]
            case_dir = str(row.get("case_dir", "")).replace("/", "\\")
            archive_name = case_dir.rstrip("\\").split("\\")[-1] if case_dir else ""
            if archive_name != entry["run_dir"]:
                contradictions.append("run directory name for %s / %s: archive %r, registry %r"
                                      % (key[0], key[1], archive_name, entry["run_dir"]))
            report_filename = row.get("report_filename", "")
            if report_filename and entry["report_file"] and report_filename != entry["report_file"]:
                contradictions.append("report file for %s / %s: archive registry %r, scorer order %r"
                                      % (key[0], key[1], report_filename, entry["report_file"]))
            sha = str(row.get("report_sha256", "")).upper()
            if sha and entry["report_sha256"] and sha != entry["report_sha256"]:
                registered = public_hashes.get(
                    (entry["system"], entry["run_dir"], entry["report_file"]))
                if registered == (sha, entry["report_sha256"]):
                    warnings.append("registered report normalisation accepted for %s / %s"
                                    % key)
                else:
                    contradictions.append("report sha256 for %s / %s: archive registry %s, run directory %s"
                                          % (key[0], key[1], sha, entry["report_sha256"]))
    else:
        warnings.append("archive ships no %s; the registry tables in this script were used alone"
                        % "/".join(ARCHIVE_REGISTRY_TSV))

    input_map = archive / ARCHIVE_INPUT_MAP_TSV[0] / ARCHIVE_INPUT_MAP_TSV[1]
    if not input_map.is_file():
        warnings.append("archive ships no %s; the five dataset files were checked against the "
                        "registry only" % "/".join(ARCHIVE_INPUT_MAP_TSV))
        return checks
    with as_path(input_map).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        study = row.get("study_id", "")
        if study not in STUDIES:
            continue
        checks += 1
        names = parse_braced_names(row.get("input_files", ""))
        references = parse_braced_names(row.get("evaluation_reference_files", ""))
        if set(names) != set(GENERATION_INPUTS) or set(references) != set(EVALUATION_REFERENCES):
            contradictions.append("study input map lists unexpected files for %s: %s / %s"
                                  % (study, sorted(names), sorted(references)))
            continue
        hashes = dict(parse_hash_pairs(row.get("input_sha256", "")))
        hashes.update(parse_hash_pairs(row.get("evaluation_sha256", "")))
        for name in REQUIRED_DATASET_FILES:
            source = dataset_source(archive, study, name)
            if not path_exists(source):
                continue
            expected = hashes.get(name, "")
            if expected and expected.upper() != sha256_file(source):
                contradictions.append("study input map sha256 for %s / %s does not match the archive file"
                                      % (study, name))
    return checks


def copy_file(source: Path, dest: Path, records: list[dict], kind: str,
              system: str, study: str, relative: str) -> int:
    as_path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fs_path(source), fs_path(dest))
    source_size = as_path(source).stat().st_size
    size = as_path(dest).stat().st_size
    source_sha = sha256_file(source)
    dest_sha = sha256_file(dest)
    if size != source_size or source_sha != dest_sha:
        raise AssemblyError("copy verification failed for %s" % dest)
    records.append({"kind": kind, "system": system, "study": study, "relative_path": relative,
                    "source": str(source), "dest": str(dest), "bytes": size, "sha256": dest_sha})
    return size


def copy_tree(source: Path, dest: Path, records: list[dict], system: str, study: str,
              prefix: str = "") -> tuple[int, int]:
    as_path(dest).mkdir(parents=True, exist_ok=True)
    files = 0
    total = 0
    for child in sorted(as_path(source).iterdir(), key=lambda item: item.name):
        relative = child.name if not prefix else prefix + "/" + child.name
        if child.is_dir():
            sub_files, sub_bytes = copy_tree(source / child.name, dest / child.name, records,
                                             system, study, relative)
            files += sub_files
            total += sub_bytes
        else:
            total += copy_file(source / child.name, dest / child.name, records, "run_file",
                               system, study, relative)
            files += 1
    return files, total


def write_manifest(out_dir: Path, payload: dict, records: list[dict]) -> None:
    as_path(out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "scoring_input_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    fields = ["kind", "system", "study", "relative_path", "bytes", "sha256", "source", "dest"]
    with (out_dir / "scoring_input_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})
    (out_dir / "SCORING_ONLY_README.txt").write_text(
        SCORING_ONLY_README.format(archive=payload["archive"]), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the scoring-only input tree for the frozen rule scorer.")
    parser.add_argument("--archive", type=Path, required=True,
                        help="archive root holding inputs/, evaluation_references/ and scoring/runs/")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="empty temporary scoring root to create (never an agent input directory)")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the archive against the registries and print the plan, writing nothing")
    args = parser.parse_args(argv)

    try:
        archive = resolve_archive(args.archive)
        entries = planned_entries()
    except AssemblyError as exc:
        print("FATAL: %s" % exc, file=sys.stderr)
        return 1

    problems: list[str] = []
    warnings: list[str] = []
    contradictions: list[str] = []
    preflight(archive, entries, problems, warnings)
    checks = cross_check_archive(archive, entries, contradictions, warnings)
    for problem in problems:
        print("PROBLEM: %s" % problem, file=sys.stderr)
    for contradiction in contradictions:
        print("CONTRADICTION: %s" % contradiction, file=sys.stderr)
    if problems:
        print("FATAL: %d registered input file(s) or run directories are missing or ambiguous"
              % len(problems), file=sys.stderr)
        return 2
    if contradictions:
        print("FATAL: the archive contradicts the registry tables in this script (%d item(s))"
              % len(contradictions), file=sys.stderr)
        return 3
    for warning in warnings:
        print("NOTE: %s" % warning)

    print("archive: %s" % archive)
    print("studies: %d   systems: %d   cells: %d" % (len(STUDIES), len(SYSTEMS), len(entries)))
    print("dataset files to copy: %d" % (len(STUDIES) * len(REQUIRED_DATASET_FILES)))
    if args.check_only:
        print("CHECK ONLY: registries and archive agree (%d archive cross-checks); nothing written"
              % checks)
        return 0

    out_dir = args.out_dir
    archive_resolved = archive.resolve()
    try:
        out_resolved = out_dir.resolve()
    except Exception:
        out_resolved = out_dir
    if out_resolved == archive_resolved or archive_resolved in out_resolved.parents \
            or out_resolved in archive_resolved.parents:
        print("FATAL: --out-dir must be outside the archive and not equal to it: %s" % out_dir,
              file=sys.stderr)
        return 1
    if path_exists(out_dir) and any(as_path(out_dir).iterdir()):
        print("FATAL: --out-dir already exists and is not empty: %s (use a fresh temporary "
              "directory so that a stale tree cannot be mixed into the replay)" % out_dir,
              file=sys.stderr)
        return 1

    records: list[dict] = []
    run_files = 0
    run_bytes = 0
    for entry in entries:
        dest = out_dir / "scoring" / "runs" / entry["system"] / entry["run_dir"]
        files, size = copy_tree(entry["run_source"], dest, records, entry["system"], entry["study"])
        entry["file_count"] = files
        entry["byte_count"] = size
        run_files += files
        run_bytes += size

    example_files = 0
    example_bytes = 0
    for study in STUDIES:
        for name in REQUIRED_DATASET_FILES:
            source = dataset_source(archive, study, name)
            dest = out_dir / "examples" / study / name
            example_bytes += copy_file(source, dest, records, "dataset_file", "", study, name)
            example_files += 1

    payload = {
        "tool": "reproduce/scoring/prepare_scoring_inputs.py",
        "archive": str(archive),
        "out_dir": str(out_dir),
        "scoring_only": True,
        "agent_input_warning": ("This tree carries the independent evaluation references "
                                "(ground_truth.txt, grading_standard.txt) for the frozen scorer; "
                                "never use it as agent or generation input."),
        "registry": {
            "studies": list(STUDIES),
            "systems": list(SYSTEMS),
            "cells": len(entries),
            "generation_inputs": list(GENERATION_INPUTS),
            "evaluation_references": list(EVALUATION_REFERENCES),
        },
        "totals": {
            "run_directories": len(entries),
            "run_files": run_files,
            "run_bytes": run_bytes,
            "example_directories": len(STUDIES),
            "example_files": example_files,
            "example_bytes": example_bytes,
        },
        "run_dirs": [
            {
                "system": entry["system"],
                "study": entry["study"],
                "run_dir": entry["run_dir"],
                "source": str(entry["run_source"]),
                "dest": str(out_dir / "scoring" / "runs" / entry["system"] / entry["run_dir"]),
                "files": entry.get("file_count", 0),
                "bytes": entry.get("byte_count", 0),
                "report_file": entry["report_file"],
                "report_sha256": entry["report_sha256"],
            }
            for entry in entries
        ],
        "examples": [
            {
                "study": study,
                "files": [
                    {"name": name, "source": str(dataset_source(archive, study, name)),
                     "dest": str(out_dir / "examples" / study / name),
                     "sha256": sha256_file(dataset_source(archive, study, name))}
                    for name in REQUIRED_DATASET_FILES
                ],
            }
            for study in STUDIES
        ],
        "archive_cross_checks": checks,
        "problems": problems,
        "warnings": warnings,
    }
    write_manifest(out_dir, payload, records)
    print("assembled: %d run files (%.1f MB), %d dataset files (%.1f MB) -> %s"
          % (run_files, run_bytes / 1048576.0, example_files, example_bytes / 1048576.0, out_dir))
    print("manifest: %s" % (out_dir / "scoring_input_manifest.json"))
    print("SCORING-ONLY TREE: do not use %s as agent input" % (out_dir / "examples"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
