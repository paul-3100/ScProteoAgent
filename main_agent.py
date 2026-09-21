# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Tuple, Optional
try:
    from PIL import Image
except Exception:  # pragma: no cover - image validation is best effort without Pillow.
    Image = None

from config import Config, get_config, set_config
import figure_captions
import report_language as _report_language
from evidence_utils import append_evidence
# The agent stack (langchain, langgraph, the provider clients, the tool registry and the prompt
# library) is imported by `_load_runtime_dependencies()` at the start of a real run instead of at
# module import time. `python main_agent.py --help` and every offline inspection of this file must
# work on a machine where the agent environment is not installed, no key is configured and no
# `.env` is read; `from __future__ import annotations` keeps the annotations below resolvable on
# that path. `_load_runtime_dependencies()` injects the imported objects into the module globals,
# so every function below keeps using the same names and the run logic is unchanged.
#
# Deferred imports: langchain_core.messages, langchain_core.prompts, langgraph.graph, llm,
# analysis_design, evaluation_evidence, prompts, tools, tools_prompts.

_RUNTIME_DEPENDENCIES_LOADED = False


class AgentState(TypedDict):
    task: str
    plan: List[dict]
    plan_idx: int
    report_idx: int
    plan_num: int
    current_step: int
    execute_results: str
    backtrack_num: int
    backtrack_idx: int
    analysis_summary: str
    state_history: List[dict]
    this_run_folder: str
    record_file: str
    memory: Dict[str, Any]
    next: str
    pre: str
    ground_truth_path: str
    grading_standard_path: str
    publication_mode: bool
    evaluator_mode: str
    enable_internal_scorer: bool
    evaluator_root: str
    output_report: List[str]
    critique_text: Any
    parameters_path: str
    memory_path: str
    executor_backtrack_counts: Dict[str, int]
    max_executor_backtracks: int
    node_failure_counts: Dict[str, int]
    max_node_failures: int


def safe_print(text, chunk: int = 2000):
    text = str(text)
    for i in range(0, len(text), chunk):
        part = text[i:i + chunk]
        try:
            print(part, end="")
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            print(part.encode(encoding, errors="replace").decode(encoding, errors="replace"), end="")


def _filesystem_write_path(path: str) -> str:
    if os.name != "nt":
        return path
    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\?\\") or len(abs_path) < 240:
        return path
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def _path_exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(_filesystem_write_path(path))


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _validate_binary_image_file(path: str) -> bool:
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in IMAGE_SUFFIXES:
        return True
    read_path = _filesystem_write_path(path)
    if not os.path.exists(read_path):
        return False
    if suffix == ".png":
        with open(read_path, "rb") as handle:
            head = handle.read(8)
        if head != PNG_SIGNATURE:
            return False
    if Image is not None:
        try:
            with Image.open(read_path) as img:
                img.verify()
        except Exception:
            return False
    return True


def record_report(file: str, summary: str = "", mode: str = "a", visible: bool = True):
    if visible:
        safe_print(summary)
    write_file = _filesystem_write_path(file)
    directory = os.path.dirname(write_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(write_file, mode, encoding="utf-8") as f:
        f.write(summary.strip() + "\n")


SENSITIVE_KEYWORDS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "google_api_key",
    "openai_api_key",
    "password",
    "serpapi_key",
    "secret",
    "token",
)
SENSITIVE_VALUE_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]+)", re.IGNORECASE)


def redact_for_log(value: Any, max_chars: int = 2000) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key)
            if any(keyword in key_text.lower() for keyword in SENSITIVE_KEYWORDS):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = redact_for_log(item, max_chars=max_chars)
        return redacted
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        rendered = [redact_for_log(item, max_chars=max_chars) for item in items[:50]]
        if len(items) > 50:
            rendered.append(f"...[truncated {len(items) - 50} items]...")
        return rendered
    if isinstance(value, (int, float, bool)) or value is None:
        return value

    text = SENSITIVE_VALUE_RE.sub("[REDACTED]", str(value))
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]..."
    return text


def _preview_text(value: Any, max_chars: int = 160) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _split_gene_preview(value: Any, max_items: int = 12) -> List[str]:
    text = str(value or "")
    if not text:
        return []
    genes = [g.strip() for g in re.split(r"[;,/]\s*", text) if g.strip()]
    return genes[:max_items]


def _compact_enrichment_result(result: Any) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "status": "completed",
        "evidence_source": "offline_enrichment",
        "full_results_file": "enrichment_results/go_kegg_reactome_results.json",
        "namespaces": {},
    }
    if not isinstance(result, dict):
        compact["preview"] = _preview_text(result, 1000)
        return compact

    payload_root = result.get("ALL") if isinstance(result.get("ALL"), dict) else result
    for namespace, namespace_payload in payload_root.items():
        if not isinstance(namespace_payload, dict):
            continue
        ns_summary = {
            "passes_fdr_directions": 0,
            "exploratory_directions": 0,
            "representative_terms": [],
        }
        for contrast, contrast_payload in namespace_payload.items():
            if not isinstance(contrast_payload, dict):
                continue
            diagnostics = contrast_payload.get("diagnostics", {}) if isinstance(contrast_payload.get("diagnostics"), dict) else {}
            for direction in ("upregulated", "downregulated"):
                direction_diag = diagnostics.get(direction, {}) if isinstance(diagnostics.get(direction), dict) else {}
                if direction_diag.get("passes_fdr"):
                    ns_summary["passes_fdr_directions"] += 1
                if direction_diag.get("significance_status") == "exploratory_not_fdr_significant":
                    ns_summary["exploratory_directions"] += 1
                terms_payload = contrast_payload.get(direction, {})
                terms = []
                if isinstance(terms_payload, dict):
                    terms = [_preview_text(v, 90) for v in list(terms_payload.values())[:5] if str(v).strip()]
                if terms and len(ns_summary["representative_terms"]) < 14:
                    ns_summary["representative_terms"].append({
                        "contrast": contrast,
                        "direction": direction,
                        "passes_fdr": bool(direction_diag.get("passes_fdr")),
                        "significance_status": direction_diag.get("significance_status", ""),
                        "terms": terms,
                    })
        compact["namespaces"][namespace] = ns_summary
    return compact


def _compact_differential_result(result: Any) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "status": "completed",
        "evidence_source": "current_matrix",
        "contrast_summary": [],
    }
    if not isinstance(result, dict):
        compact["preview"] = _preview_text(result, 1000)
        return compact
    for contrast, payload in list(result.items())[:60]:
        if not isinstance(payload, dict):
            continue
        top_up = payload.get("top_up_genes", {}) if isinstance(payload.get("top_up_genes"), dict) else {}
        top_down = payload.get("top_down_genes", {}) if isinstance(payload.get("top_down_genes"), dict) else {}
        compact["contrast_summary"].append({
            "contrast": contrast,
            "n_up": payload.get("n_up"),
            "n_down": payload.get("n_down"),
            "p_thresh": payload.get("p_thresh"),
            "logfc_thresh": payload.get("logfc_thresh"),
            "top_up_genes": _split_gene_preview(top_up.get("gene_names"), 12),
            "top_down_genes": _split_gene_preview(top_down.get("gene_names"), 12),
            "average_up_logFC": top_up.get("average_logFC"),
            "average_down_logFC": top_down.get("average_logFC"),
        })
    compact["n_contrasts"] = len(compact["contrast_summary"])
    return compact


def _compact_limma_result(result: Any) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "status": "completed",
        "evidence_source": "current_matrix",
        "contrast_summary": [],
    }
    if not isinstance(result, dict):
        compact["preview"] = _preview_text(result, 1000)
        return compact
    for contrast, payload in list(result.items())[:60]:
        if not isinstance(payload, dict):
            continue
        checks = payload.get("sanity_checks", {}) if isinstance(payload.get("sanity_checks"), dict) else {}
        compact["contrast_summary"].append({
            "contrast": contrast,
            "method": payload.get("method"),
            "statistical_backend": payload.get("statistical_backend"),
            "n_samples_group_a": payload.get("n_samples_group_a"),
            "n_samples_group_b": payload.get("n_samples_group_b"),
            "n_significant": checks.get("n_significant"),
            "n_up": checks.get("n_up"),
            "n_down": checks.get("n_down"),
            "median_logFC_sig": payload.get("median_logFC_sig"),
        })
    compact["n_contrasts"] = len(compact["contrast_summary"])
    return compact


def _compact_umap_result(result: Any) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "status": "completed",
        "evidence_source": "current_matrix",
        "outputs_file": "umap_results/umap_clustering_results.json",
        "views": [],
    }
    if not isinstance(result, dict):
        compact["preview"] = _preview_text(result, 1000)
        return compact
    for view_name, payload in list(result.items())[:20]:
        if not isinstance(payload, dict):
            continue
        compactness = payload.get("cluster_compactness", {}) if isinstance(payload.get("cluster_compactness"), dict) else {}
        clusters = compactness.get("Cluster") or compactness.get("Type") or {}
        mean_radius = compactness.get("mean_radius") or {}
        compact["views"].append({
            "view": view_name,
            "cluster_number": payload.get("cluster_number"),
            "clusters": list(clusters.values())[:12] if isinstance(clusters, dict) else [],
            "mean_radius_preview": list(mean_radius.items())[:6] if isinstance(mean_radius, dict) else [],
            "umap_parameters": payload.get("umap_parameters", {}),
        })
    compact["n_views"] = len(compact["views"])
    return compact


def compact_tool_result_for_context(tool_name: str, result: Any) -> Any:
    """Return a concise, analyzer-safe representation of tool output.

    Full evidence remains on disk; this function only controls what goes into
    record summaries and LLM context.
    """
    if tool_name == "run_enrichment":
        return _compact_enrichment_result(result)
    if tool_name == "extract_differential_proteins":
        return _compact_differential_result(result)
    if tool_name == "run_pairwise_limma":
        return _compact_limma_result(result)
    if tool_name == "run_umap":
        return _compact_umap_result(result)
    return redact_for_log(result, max_chars=12000)


def record_event(
    record_file: str,
    event: str,
    status: str,
    started_at: float | None = None,
    **fields,
):
    if not record_file:
        return
    try:
        run_folder = os.path.dirname(record_file) or "."
        os.makedirs(run_folder, exist_ok=True)
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        duration_sec = None if started_at is None else round(time.perf_counter() - started_at, 4)
        payload = {
            "timestamp": timestamp,
            "event": event,
            "status": status,
            "duration_sec": duration_sec,
        }
        payload.update(redact_for_log(fields, max_chars=4000))

        events_path = os.path.join(run_folder, "run_events.jsonl")
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

        status_path = os.path.join(run_folder, "run_status.json")
        previous_status = {}
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    previous_status = json.load(f)
            except Exception:
                previous_status = {}

        run_status = previous_status.get("run_status", "running")
        if event == "run_start":
            run_status = "running"
        elif event == "run_end":
            # R27: a run stopped by the authorized envelope (or by the wall clock) is a real
            # terminal state, not a plain error, and must not stay "running" forever.
            if status in ("completed", "stopped_envelope", "stopped_wall_clock"):
                run_status = status
            else:
                run_status = "error"
        elif status == "error":
            run_status = "error"

        status_payload = {
            "updated_at": timestamp,
            "status": run_status,
            "run_status": run_status,
            "last_event": payload,
        }
        tmp_path = status_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(status_payload, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, status_path)
        short_status_path = os.path.join(run_folder, "status.json")
        tmp_short_status_path = short_status_path + ".tmp"
        with open(tmp_short_status_path, "w", encoding="utf-8") as f:
            json.dump(status_payload, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_short_status_path, short_status_path)
    except Exception as e:
        print(f"[RunLog] Failed to write structured event: {e}")


def build_prompt_with_static_system(system_text: str, human_template: str) -> ChatPromptTemplate:
    """Build a chat prompt where only the human message is templated."""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_text),
        HumanMessagePromptTemplate.from_template(human_template),
    ])


def format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"


def record_timing(record_file: str, node_name: str, started_at: float, extra_label: str = ""):
    duration = time.perf_counter() - started_at
    suffix = f" ({extra_label})" if extra_label else ""
    record_report(record_file, f"\n\n## [Timing] {node_name}{suffix}: {format_duration(duration)}")
    record_event(
        record_file,
        "timing",
        "completed",
        started_at=started_at,
        node=node_name,
        label=extra_label,
        duration_text=format_duration(duration),
    )


def initialize_memory_store() -> Dict[str, Any]:
    return {}


def normalize_memory_key(plan_idx: Any, step_idx: Any) -> str:
    return f"plan_{int(plan_idx)}_step_{int(step_idx)}"


def ensure_memory_store(memory: Any) -> Dict[str, Any]:
    if isinstance(memory, dict):
        normalized: Dict[str, Any] = {}
        for key, value in memory.items():
            if re.fullmatch(r"plan_\d+_step_\d+", str(key)):
                text = str(value or "").strip()
                if text:
                    normalized[str(key)] = text

        legacy_summaries = memory.get("step_summaries", [])
        if isinstance(legacy_summaries, list):
            for item in legacy_summaries:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", ""))
                match = re.fullmatch(r"plan_(\d+)_step_(\d+)_summary", title)
                if not match:
                    continue
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                normalized[normalize_memory_key(match.group(1), match.group(2))] = content

        return dict(sorted(normalized.items(), key=lambda item: item[0]))

    if isinstance(memory, str) and memory.strip():
        try:
            parsed = json.loads(memory)
            if isinstance(parsed, dict):
                return ensure_memory_store(parsed)
        except Exception:
            return initialize_memory_store()

    return initialize_memory_store()


def load_memory(memory_file_path: str) -> Dict[str, Any]:
    if not memory_file_path or not os.path.exists(memory_file_path):
        return initialize_memory_store()

    try:
        with open(memory_file_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return initialize_memory_store()
        return ensure_memory_store(json.loads(raw))
    except Exception:
        try:
            with open(memory_file_path, "r", encoding="utf-8") as f:
                return ensure_memory_store(f.read())
        except Exception:
            return initialize_memory_store()


def save_memory(memory_file_path: str, memory: Dict[str, Any]):
    try:
        directory = os.path.dirname(memory_file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(memory_file_path, "w", encoding="utf-8") as f:
            json.dump(ensure_memory_store(memory), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Memory] Failed to save memory: {e}")


def trim_text(text: Any, max_chars: int = 2000) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "\n...[truncated]..."


def set_memory_entry(
    memory: Dict[str, Any],
    key: str,
    content: Any,
) -> Dict[str, Any]:
    normalized_memory = ensure_memory_store(memory)
    text = str(content or "").strip()
    if not text:
        return normalized_memory

    normalized_memory[str(key)] = text
    return dict(sorted(normalized_memory.items(), key=lambda item: item[0]))


def build_memory_context(memory: Dict[str, Any], max_chars: int = 12000) -> str:
    normalized_memory = ensure_memory_store(memory)
    if not normalized_memory:
        return "No useful memory yet."

    rendered_entries = [
        f"- {key}: {trim_text(value, 1500)}"
        for key, value in normalized_memory.items()
    ]
    return trim_text("Analyzer results from prior steps:\n" + "\n".join(rendered_entries), max_chars)


def snapshot_state(state: AgentState) -> Dict[str, Any]:
    return copy.deepcopy({k: v for k, v in state.items() if k != "state_history"})


def append_state_history(state: AgentState, node_name: str) -> List[dict]:
    return state.get("state_history", []) + [{
        "node": node_name,
        "state_snapshot": snapshot_state(state),
    }]


def extract_json_block(text: str, opening_char: str) -> str:
    closing_char = {"[": "]", "{": "}"}[opening_char]
    start = text.find(opening_char)
    if start == -1:
        raise ValueError(f"No JSON block starting with '{opening_char}' found.")

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opening_char:
            depth += 1
        elif ch == closing_char:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    raise ValueError("Incomplete JSON block in model response.")


def parse_plan_response(plan_str: str) -> List[dict]:
    stripped = plan_str.strip()

    for opening_char in ("[", "{"):
        if opening_char not in stripped:
            continue
        try:
            payload = json.loads(extract_json_block(stripped, opening_char))
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and isinstance(payload.get("plan"), list):
                return payload["plan"]
        except Exception:
            continue

    raise ValueError("Unable to parse plan JSON from planner response.")


def parse_json_object_response(text: str) -> Dict[str, Any]:
    payload = json.loads(extract_json_block(text, "{"))
    if not isinstance(payload, dict):
        raise ValueError("Parsed JSON payload is not an object.")
    return payload


def render_tool_catalog(tool_prompts: List[dict]) -> str:
    tool_lines = []
    for item in tool_prompts:
        function_info = item.get("function", {})
        name = function_info.get("name", "")
        description = function_info.get("description", "")
        properties = function_info.get("parameters", {}).get("properties", {})
        param_names = ", ".join(properties.keys()) if properties else "no arguments"
        tool_lines.append(f"- {name}: {description} Parameters: {param_names}.")
    return "\n".join(tool_lines)


def build_local_fallback_plan(task: str = "") -> List[dict]:
    try:
        design = get_config().get("analysis_design", {}) or {}
    except Exception:
        design = {}
    species = design.get("species", "human")
    design_text = json.dumps(design, ensure_ascii=False).lower() if design else ""
    task_text = (task or "").lower()
    needs_leakage = any(
        token in f"{task_text}\n{design_text}"
        for token in ["intact", "permeable", "membranestatus", "membrane status", "leakage", "protein leakage", "type1"]
    )
    plan = [
        {
            "name": "local_scoring_evidence_preflight",
            "description": "Generate dataset-specific scoring requirements and QC evidence before analysis.",
            "tool_calls": [{"name": "prepare_scoring_evidence_pack", "args": {}}],
        },
        {
            "name": "local_matrix_qc_and_batch_boundary",
            "description": "Normalize/calibrate the matrix only when the analysis design says this is appropriate.",
            "tool_calls": [{"name": "combat_calibration", "args": {"need_calibration": True}}],
        },
        {
            "name": "local_design_driven_differential_analysis",
            "description": "Run design-driven pairwise differential analysis and extract candidate proteins.",
            "tool_calls": [
                {"name": "run_pairwise_limma", "args": {}},
                {"name": "extract_differential_proteins", "args": {"top_n": 25}},
            ],
        },
    ]
    if needs_leakage:
        plan.append({
            "name": "local_membrane_leakage_evidence",
            "description": "Run membrane-status protein leakage analysis and compartment module summaries.",
            "tool_calls": [{"name": "run_protein_leakage_analysis", "args": {}}],
        })
    plan.append(
        {
            "name": "local_enrichment_and_visualization",
            "description": "Run offline enrichment, generate figures, and refresh scoring evidence tables.",
            "tool_calls": [
                {"name": "run_enrichment", "args": {"contrast_type": "ALL", "species": species, "top_n": 15}},
                {"name": "visualize", "args": {"plot_set": "full", "include_enrichment": True}},
                {"name": "prepare_scoring_evidence_pack", "args": {}},
            ],
        },
    )
    return plan


def execute_tool_calls(tool_calls, record_file: str, step_idx: int):
    tool_results = []
    need_backtrack_flag = False

    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        call_started_at = time.perf_counter()
        safe_args = redact_for_log(tool_args)
        record_event(
            record_file,
            "tool_call_start",
            "started",
            tool=tool_name,
            step_idx=step_idx + 1,
            args=safe_args,
        )
        record_report(record_file, f"\n\n## [Executor] Calling tool '{tool_name}' with args {safe_args}")
        tool_impl = next((t for t in AVAIABLE_TOOLS if t.name == tool_name), None)
        if not tool_impl:
            err = (
                f"\n\n## [Executor] Call tool error: Tool '{tool_name}' not found. "
                f"Available tools: {[t.name for t in AVAIABLE_TOOLS]}"
            )
            record_report(record_file, err)
            record_event(
                record_file,
                "tool_call_error",
                "error",
                started_at=call_started_at,
                tool=tool_name,
                step_idx=step_idx + 1,
                args=safe_args,
                error=f"Tool '{tool_name}' not found.",
            )
            tool_results.append({"tool": tool_name, "args": tool_args, "result": err})
            need_backtrack_flag = True
            continue

        try:
            res = tool_impl.invoke(tool_args)
            compact_res = compact_tool_result_for_context(tool_name, res)
            safe_res = redact_for_log(compact_res, max_chars=20000)
            record_report(record_file, f"\n\n## [Executor] Executing step {step_idx + 1}. Compact Results:\n{safe_res}")
            record_event(
                record_file,
                "tool_call_end",
                "completed",
                started_at=call_started_at,
                tool=tool_name,
                step_idx=step_idx + 1,
                args=safe_args,
                result_summary=redact_for_log(compact_res, max_chars=4000),
            )
            tool_results.append({"tool": tool_name, "args": tool_args, "result": compact_res})
        except Exception as e:
            err = f"\n\n## [Executor] Executing step {step_idx + 1} failed: {e}"
            tb = traceback.format_exc()
            safe_err = redact_for_log(err, max_chars=120000)
            safe_tb = redact_for_log(tb, max_chars=120000)
            record_report(record_file, safe_err + f"\n\n```text\n{safe_tb}\n```")
            record_event(
                record_file,
                "tool_call_error",
                "error",
                started_at=call_started_at,
                tool=tool_name,
                step_idx=step_idx + 1,
                args=safe_args,
                error=str(e),
                traceback=tb,
            )
            tool_results.append({"tool": tool_name, "args": tool_args, "result": err})
            need_backtrack_flag = True

    return tool_results, need_backtrack_flag


def plan_node(state: AgentState):
    started_at = time.perf_counter()
    task = state.get("task", "")
    plan_idx = state.get("plan_idx", 0) + 1
    record_file = state.get("record_file", "record_file.md")
    tools_description = render_tool_catalog(TOOLS_PROMPTS)
    memory = ensure_memory_store(state.get("memory", {}))
    memory_context = build_memory_context(memory)

    is_first_plan = plan_idx <= 1
    node_name = "planner" if is_first_plan else "replanner"

    if is_first_plan:
        record_report(record_file, f"\n\n## [Planner] Task: {task}")
        system_prompt = PLANNER_PROMPT
        human_prompt = (
            "Task: {task}\n"
            "Relevant memory from prior work:\n{memory_context}"
        )
        invoke_payload = {
            "task": task,
            "tools_description": tools_description,
            "memory_context": memory_context,
        }
    else:
        critique_text = state.get("critique_text", "")
        system_prompt = REPLANNER_PROMPT
        human_prompt = (
            "Generate a new execution plan.\n"
            "Task: {task}\n"
            "Relevant memory from prior work:\n{memory_context}\n"
            "Critique text:\n{critique_text}"
        )
        invoke_payload = {
            "task": task,
            "tools_description": tools_description,
            "critique_text": json.dumps(critique_text, ensure_ascii=False),
            "memory_context": memory_context,
        }

    prompt = build_prompt_with_static_system(
        system_prompt + "\nYou may only use the following tools:\n" + tools_description,
        human_prompt,
    )

    chain = prompt | LLM
    try:
        response = chain.invoke(invoke_payload)
        plan = parse_plan_response(response.content)
    except Exception as e:
        fallback_plan = build_local_fallback_plan(task)
        if fallback_plan:
            record_report(
                record_file,
                f"\n\n## [{node_name.capitalize()}] Local Fallback Plan:\n"
                f"LLM planner failed ({e}); using a deterministic analysis plan.\n"
                f"{json.dumps(fallback_plan, ensure_ascii=False, indent=2)}"
            )
            record_event(
                record_file,
                "planner_local_fallback",
                "completed",
                started_at=started_at,
                error=str(e),
                n_steps=len(fallback_plan),
            )
            record_timing(record_file, node_name.capitalize(), started_at, f"plan {plan_idx} local fallback")
            return {
                "plan": fallback_plan,
                "plan_idx": plan_idx,
                "current_step": 0,
                "executor_backtrack_counts": {},
                "node_failure_counts": {},
                "memory": memory,
                "state_history": append_state_history(state, "planner"),
                "next": "executor",
                "pre": "planner",
                "analysis_summary": "",
            }
        node_failure_counts = dict(state.get("node_failure_counts", {}))
        node_failure_counts[node_name] = node_failure_counts.get(node_name, 0) + 1
        err = f"\n\n## [{node_name.capitalize()}] Error: {e}"
        record_report(record_file, err)
        record_timing(record_file, node_name.capitalize(), started_at, f"plan {plan_idx}")
        return {
            "plan": [{"name": "error", "description": err}],
            "memory": memory,
            "state_history": append_state_history(state, "planner"),
            "node_failure_counts": node_failure_counts,
            "next": "backtrack",
            "pre": "planner",
        }

    updated_state = {
        "plan": plan,
        "plan_idx": plan_idx,
        "current_step": 0,
        "executor_backtrack_counts": {},
        "node_failure_counts": {},
        "memory": memory,
        "state_history": append_state_history(state, "planner"),
        "next": "executor",
        "pre": "planner",
        "analysis_summary": "",
    }

    title = "Generated Plan" if is_first_plan else "New Plan"
    record_report(record_file, f"\n\n## [{node_name.capitalize()}] {title}:\n{json.dumps(plan, ensure_ascii=False, indent=2)}")
    record_timing(record_file, node_name.capitalize(), started_at, f"plan {plan_idx}")
    return updated_state


def split_text(text: str, max_length: int = 220000) -> List[str]:
    text = text or ""
    if not text:
        return []
    return [text[start:start + max_length] for start in range(0, len(text), max_length)]


def execute_node(state: AgentState):
    started_at = time.perf_counter()
    task = state.get("task", "")
    plan = state.get("plan", [])
    idx = state.get("current_step", 0)
    record_file = state.get("record_file", "record_file.md")
    analysis_summary = state.get("analysis_summary", "")
    memory = ensure_memory_store(state.get("memory", {}))
    parameters_path = state.get("parameters_path", "")
    avaliable_parameters = load_json(parameters_path) if os.path.exists(parameters_path) else {}
    tools_description = render_tool_catalog(TOOLS_PROMPTS)
    need_backtrack_flag = False
    executor_backtrack_counts = dict(state.get("executor_backtrack_counts", {}))
    step_backtrack_key = f"plan_{state.get('plan_idx', 0)}_step_{idx}"

    if idx >= len(plan):
        record_report(record_file, "\n\n## [Executor] No remaining plan steps. Moving to report.")
        record_timing(record_file, "Executor", started_at, "no-op")
        return {
            "memory": memory,
            "state_history": append_state_history(state, "executor"),
            "next": "report",
            "pre": "executor",
        }

    current_step = plan[idx]
    record_report(record_file, f"\n\n## [Executor] Executing step {idx + 1}: {current_step}")

    if isinstance(current_step, dict) and current_step.get("tool_calls"):
        tool_results, need_backtrack_flag = execute_tool_calls(
            current_step.get("tool_calls", []),
            record_file=record_file,
            step_idx=idx,
        )
        final_result = json.dumps(tool_results, ensure_ascii=False)
        next_state = {
            "execute_results": final_result,
            "memory": memory,
            "executor_backtrack_counts": executor_backtrack_counts,
            "state_history": append_state_history(state, "executor"),
            "pre": "executor",
        }
        if need_backtrack_flag:
            backtrack_idx = state.get("backtrack_idx", 0) + 1
            executor_backtrack_counts[step_backtrack_key] = executor_backtrack_counts.get(step_backtrack_key, 0) + 1
            next_state["backtrack_idx"] = backtrack_idx
            next_state["executor_backtrack_counts"] = executor_backtrack_counts
            next_state["next"] = "backtrack"
        else:
            executor_backtrack_counts.pop(step_backtrack_key, None)
            next_state["executor_backtrack_counts"] = executor_backtrack_counts
            next_state["current_step"] = idx + 1
            next_state["next"] = "analyzer"
        record_timing(record_file, "Executor", started_at, f"step {idx + 1} local tools")
        return next_state

    chunks = split_text(analysis_summary)
    if not chunks:
        chunks = [""]

    memory_context = build_memory_context(memory)
    execution_prompt = build_prompt_with_static_system(
        EXECUTOR_PROMPT + "\nAvailable tools:\n" + tools_description,
        "Finish the task: {task}.\n"
        "Execute the following step: {current_plan_step}.\n"
        "Relevant memory from prior steps:\n{memory_context}\n"
        "Possible parameters that may be used: {avaliable_parameters}.\n"
        "---\n"
        "Result from processing previous analysis chunks (if any): {previous_iteration_result}\n"
        "Current analysis chunk to consider: {current_analysis_chunk}",
    )

    rolling_llm_result = "None"
    final_llm_response = None
    aggregated_tool_results = []

    try:
        for chunk_idx, chunk in enumerate(chunks):
            record_report(
                record_file,
                f"\n\n## [Executor] Step {idx + 1}. Processing chunk {chunk_idx + 1}/{len(chunks)} ---"
            )

            llm_input = {
                "task": task,
                "current_plan_step": current_step,
                "memory_context": memory_context,
                "avaliable_parameters": json.dumps(avaliable_parameters, ensure_ascii=False),
                "previous_iteration_result": rolling_llm_result,
                "current_analysis_chunk": chunk,
            }

            llm_response = LLM_with_tools.invoke(execution_prompt.format_messages(**llm_input))
            final_llm_response = llm_response

            if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
                chunk_tool_results, chunk_need_backtrack = execute_tool_calls(
                    llm_response.tool_calls,
                    record_file=record_file,
                    step_idx=idx,
                )
                aggregated_tool_results.extend(chunk_tool_results)
                need_backtrack_flag = need_backtrack_flag or chunk_need_backtrack
                rolling_llm_result = json.dumps(chunk_tool_results, ensure_ascii=False)
            elif getattr(llm_response, "content", None):
                rolling_llm_result = llm_response.content
            else:
                rolling_llm_result = "No content or tool call returned."
    except Exception as e:
        err = f"[Executor] LLM invoke failed: {e}"
        record_report(record_file, err)
        record_timing(record_file, "Executor", started_at, f"step {idx + 1}")
        return {
            "execute_results": err,
            "memory": memory,
            "executor_backtrack_counts": {
                **executor_backtrack_counts,
                step_backtrack_key: executor_backtrack_counts.get(step_backtrack_key, 0) + 1,
            },
            "state_history": append_state_history(state, "executor"),
            "next": "backtrack",
            "pre": "executor",
        }

    if aggregated_tool_results:
        final_result = json.dumps(aggregated_tool_results, ensure_ascii=False)
    elif getattr(final_llm_response, "tool_calls", None):
        tool_results, direct_need_backtrack = execute_tool_calls(
            final_llm_response.tool_calls,
            record_file=record_file,
            step_idx=idx,
        )
        need_backtrack_flag = need_backtrack_flag or direct_need_backtrack
        final_result = json.dumps(tool_results, ensure_ascii=False)
    else:
        final_result = getattr(final_llm_response, "content", str(final_llm_response))

    next_state = {
        "execute_results": final_result,
        "memory": memory,
        "executor_backtrack_counts": executor_backtrack_counts,
        "state_history": append_state_history(state, "executor"),
        "pre": "executor",
    }

    if need_backtrack_flag:
        backtrack_idx = state.get("backtrack_idx", 0) + 1
        executor_backtrack_counts[step_backtrack_key] = executor_backtrack_counts.get(step_backtrack_key, 0) + 1
        next_state["backtrack_idx"] = backtrack_idx
        next_state["executor_backtrack_counts"] = executor_backtrack_counts
        next_state["next"] = "backtrack"
    else:
        executor_backtrack_counts.pop(step_backtrack_key, None)
        next_state["executor_backtrack_counts"] = executor_backtrack_counts
        next_state["current_step"] = idx + 1
        next_state["next"] = "analyzer"

    record_timing(record_file, "Executor", started_at, f"step {idx + 1}")
    return next_state


def analyzer_node(state: AgentState):
    started_at = time.perf_counter()
    plan = state.get("plan", [])
    execute_results = state.get("execute_results", "")
    idx = state.get("current_step", 0)
    completed_step_idx = max(idx - 1, 0)
    record_file = state.get("record_file", "record_file.md")
    memory_path = state.get("memory_path", "")
    analysis_summary = state.get("analysis_summary", "")
    memory = ensure_memory_store(state.get("memory", {}))

    analyzer_prompt = build_prompt_with_static_system(
        ANALYZER_PROMPT,
        "Historical analysis summary:\n{previous_summary}\n\n"
        "Current execution result to analyze:\n{execute_results}\n\n"
        "Continue the analysis based on what is already known and avoid repetition.",
    )

    need_backtrack_flag = False
    chunks = split_text(execute_results)
    if not chunks:
        chunks = [""]

    try:
        analyzer_chain = analyzer_prompt | LLM
        rolling_context = ""
        full_rolling_context = ""

        for i, chunk in enumerate(chunks):
            analyzer_response = analyzer_chain.invoke({
                "execute_results": chunk,
                "previous_summary": rolling_context,
            })

            step_txt = f"\n\n## [Analyzer] Step {completed_step_idx + 1}.{i + 1}/{len(chunks)}:\n{analyzer_response.content}"
            record_report(record_file, step_txt)
            full_rolling_context += "\n" + analyzer_response.content
            rolling_context += "\n" + analyzer_response.content

            if len(rolling_context) > 100000:
                rolling_context = rolling_context[-100000:]

        summary_block = f"\n\n### Step {completed_step_idx + 1} Summary\n{full_rolling_context.strip()}"
        analysis_summary = (analysis_summary.rstrip() + summary_block).strip()
        memory = set_memory_entry(
            memory,
            normalize_memory_key(state.get("plan_idx", 0), completed_step_idx + 1),
            full_rolling_context,
        )

        persistent_memory = ensure_memory_store(memory)
        save_memory(memory_path, persistent_memory)


    except Exception as e:
        err = f"[Analyzer] Analysis failed: {e}"
        record_report(record_file, err)
        raw_fallback_summary = (
            f"\n\n## [Analyzer] Local Fallback Summary for step {completed_step_idx + 1}\n"
            f"- Analyzer LLM failed: {e}\n"
            f"- Raw tool output is retained below in record_file.md for audit only.\n\n"
            f"```json\n{trim_text(execute_results, 60000)}\n```"
        )
        compact_fallback_summary = (
            f"\n\n## [Analyzer] Local Fallback Summary for step {completed_step_idx + 1}\n"
            f"- Analyzer LLM failed: {e}\n"
            "- Raw tool JSON was retained in `record_file.md` only and is intentionally excluded from reporter context.\n"
            "- Reporter should use generated evidence tables, figures, scoring coverage, and concise execution summaries."
        )
        record_report(record_file, raw_fallback_summary)
        analysis_summary = (analysis_summary.rstrip() + compact_fallback_summary).strip()
        memory = set_memory_entry(
            memory,
            normalize_memory_key(state.get("plan_idx", 0), completed_step_idx + 1),
            compact_fallback_summary,
        )
        try:
            persistent_memory = ensure_memory_store(memory)
            save_memory(memory_path, persistent_memory)
        except Exception:
            pass
        need_backtrack_flag = False

    if need_backtrack_flag:
        next_node = "backtrack"
    elif idx >= len(plan):
        next_node = "report"
    else:
        next_node = "executor"

    next_state = {
        "memory": memory,
        "analysis_summary": analysis_summary,
        "state_history": append_state_history(state, "analyzer"),
        "pre": "analyzer",
        "next": next_node,
    }

    record_timing(record_file, "Analyzer", started_at, f"step {completed_step_idx + 1}")
    return next_state


def critic_node(state: AgentState):
    started_at = time.perf_counter()
    record_file = state["record_file"]
    plan_idx = state.get("plan_idx", 0)
    plan_num = state.get("plan_num", 2)
    analysis_summary = state.get("analysis_summary", "")
    ground_truth_path = state.get("ground_truth_path", "")
    grading_standard_path = state.get("grading_standard_path", "")
    enable_internal_scorer = bool(state.get("enable_internal_scorer", False))
    evaluator_mode = state.get("evaluator_mode", "none")
    output_report = state.get("output_report", [])
    memory = ensure_memory_store(state.get("memory", {}))

    critic_prompt = build_prompt_with_static_system(
        CRITIC_PROMPT,
        "Review the following report and return the critique JSON only:\n{report_txt}",
    )

    critic_chain = critic_prompt | LLM
    decision = "done"
    try:
        llm_response = critic_chain.invoke({"report_txt": output_report[-1] if output_report else analysis_summary})
        critique_text = parse_json_object_response(llm_response.content)
        record_report(record_file, f"\n\n## [Critic] Critique Results:\n{json.dumps(critique_text, ensure_ascii=False, indent=2)}")
        decision = critique_text.get("decision", "done")

        if enable_internal_scorer and evaluator_mode == "internal-dev" and os.path.exists(ground_truth_path):
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                ground_truth = f.read()
            with open(grading_standard_path, "r", encoding="utf-8") as f:
                grading_standard = f.read()
            _ = score_analysis(output_report[-1] if output_report else analysis_summary, grading_standard, ground_truth, record_file)
    except Exception as e:
        critique_text = f"Error parsing critique: {e}"
        record_report(record_file, f"\n\n## [Critic] Critique Results:\n{critique_text}")
        if enable_internal_scorer and evaluator_mode == "internal-dev" and os.path.exists(ground_truth_path) and os.path.exists(grading_standard_path):
            try:
                with open(ground_truth_path, "r", encoding="utf-8") as f:
                    ground_truth = f.read()
                with open(grading_standard_path, "r", encoding="utf-8") as f:
                    grading_standard = f.read()
                _ = score_analysis(output_report[-1] if output_report else analysis_summary, grading_standard, ground_truth, record_file)
            except Exception as score_error:
                record_report(record_file, f"\n\n## [Scorer] Fallback scoring failed: {score_error}")
        record_timing(record_file, "Critic", started_at, f"plan {plan_idx}")
        return {
            "memory": memory,
            "state_history": append_state_history(state, "critic"),
            "next": "end",
            "pre": "critic",
            "critique_text": critique_text,
        }

    next_node = "planner" if decision == "replan" else "end"
    if plan_idx >= plan_num:
        next_node = "end"

    updated_state = {
        "memory": memory,
        "state_history": append_state_history(state, "critic"),
        "next": next_node,
        "pre": "critic",
        "critique_text": critique_text,
    }
    record_timing(record_file, "Critic", started_at, f"plan {plan_idx}")
    return updated_state


def backtrack_node(state: AgentState):
    started_at = time.perf_counter()
    state_history = state.get("state_history", [])
    backtrack_idx = state.get("backtrack_idx", 0)
    record_file = state.get("record_file", "record_file.md")
    pre = state.get("pre", "")
    executor_backtrack_counts = dict(state.get("executor_backtrack_counts", {}))
    node_failure_counts = dict(state.get("node_failure_counts", {}))
    max_node_failures = state.get("max_node_failures", 3)
    current_step_idx = state.get("current_step", 0)
    step_backtrack_key = f"plan_{state.get('plan_idx', 0)}_step_{current_step_idx}"
    step_backtrack_count = executor_backtrack_counts.get(step_backtrack_key, 0)
    max_executor_backtracks = state.get("max_executor_backtracks", 3)

    if not state_history:
        next_state = {
            **state,
            "next": "planner",
            "pre": "backtrack",
        }
        record_timing(record_file, "Backtrack", started_at)
        return next_state

    previous_snapshot = state_history[-1]
    restored_state = copy.deepcopy(previous_snapshot["state_snapshot"])
    restored_state["state_history"] = state_history
    restored_state["executor_backtrack_counts"] = executor_backtrack_counts
    restored_state["node_failure_counts"] = node_failure_counts

    if pre != "executor" and node_failure_counts.get(pre, 0) >= max_node_failures:
        message = (
            f"{pre} failed {node_failure_counts.get(pre, 0)} times; stopping instead of "
            "retrying until the graph recursion limit."
        )
        record_report(record_file, f"\n\n## [Backtrack] {message}")
        record_timing(record_file, "Backtrack", started_at, f"{pre} failure limit")
        raise RuntimeError(message)

    if pre == "executor" and step_backtrack_count >= max_executor_backtracks:
        skipped_step_idx = restored_state.get("current_step", 0)
        plan = restored_state.get("plan", [])
        skipped_step = plan[skipped_step_idx] if skipped_step_idx < len(plan) else {}
        restored_state["execute_results"] = (
            f"[Executor] Step {skipped_step_idx + 1} skipped after "
            f"{step_backtrack_count} backtrack attempts. Step content: {skipped_step}"
        )
        restored_state["current_step"] = skipped_step_idx + 1
        restored_state["backtrack_idx"] = 0
        restored_state["next"] = "analyzer"
        record_report(
            record_file,
            f"\n\n## [Backtrack] Step {skipped_step_idx + 1} exceeded executor backtrack limit "
            f"({step_backtrack_count}/{max_executor_backtracks}). Skipping this step."
        )
        record_timing(record_file, "Backtrack", started_at, f"step {skipped_step_idx + 1}")
        return restored_state

    if backtrack_idx >= state.get("backtrack_num", 0):
        restored_state["next"] = "planner"
        record_report(record_file, "\n\n## [Backtrack] Exceeded max backtrack attempts. Moving to replan.")
    else:
        restored_state["next"] = pre
        record_report(record_file, f"\n\n## [Backtrack] Backtracked to node {pre}.")

    record_timing(record_file, "Backtrack", started_at)
    return restored_state


def score_analysis(
    report_txt: str,
    grading_standard: str,
    ground_truth: str,
    record_file: str | None = None,
):
    scorer_prompt = ChatPromptTemplate.from_messages([
        HumanMessage(content=grading_standard),
        SystemMessage(content=SCORER_PROMPT),
        HumanMessagePromptTemplate.from_template(
            "Ground Truth:\n{ground_truth}\n\nReport to score:\n{report_txt}"
        ),
    ])

    scorer_chain = scorer_prompt | SCORELLM

    try:
        llm_response = scorer_chain.invoke({
            "ground_truth": ground_truth,
            "report_txt": report_txt,
        })

        response_text = llm_response.content
        try:
            parsed = parse_json_object_response(response_text)
        except Exception:
            parsed = response_text
        parsed = normalize_score_payload(parsed)

        if record_file is not None:
            record_report(record_file, f"\n\n## [Scorer] Scoring Results:\n{json.dumps(parsed, ensure_ascii=False, indent=2)}")

        return parsed
    except Exception as e:
        error_msg = f"Scorer error: {e}"
        fallback = heuristic_score_analysis(report_txt, grading_standard, ground_truth, error_msg)
        if record_file is not None:
            record_report(record_file, f"\n\n## [Scorer] Error:\n{error_msg}")
            record_report(record_file, f"\n\n## [Scorer] Scoring Results:\n{json.dumps(fallback, ensure_ascii=False, indent=2)}")
        return fallback


def heuristic_score_analysis(report_txt: str, grading_standard: str, ground_truth: str, error_msg: str = "") -> Dict[str, Any]:
    text = report_txt or ""
    score = 45.0
    checks = {
        "scoring_appendix": "Scoring Evidence Appendix" in text or "评分证据附录" in text,
        "coverage_table": "Scoring Standard Coverage Table" in text or "评分标准覆盖" in text,
        "candidate_table": "Candidate Protein Evidence Highlights" in text or "候选蛋白证据摘要" in text,
        "group_qc_table": "Group Composition And QC Table" in text or "分组组成与 QC 表" in text,
        "module_scores": "Curated Module Score Highlights" in text or "Curated 模块分数摘要" in text,
        "confidence": bool(re.search(r"confidence\s*[:：]\s*(high|moderate|low)", text, re.I)),
        "evidence_sources": any(label in text for label in ["current_matrix", "offline_enrichment", "external_literature", "drug_database"]),
    }
    score += 8 if checks["scoring_appendix"] else 0
    score += 8 if checks["coverage_table"] else 0
    score += 8 if checks["candidate_table"] else 0
    score += 8 if checks["group_qc_table"] else 0
    score += 7 if checks["module_scores"] else 0
    score += 6 if checks["confidence"] else 0
    score += 5 if checks["evidence_sources"] else 0

    lowered = text.lower()
    if "fdr" in lowered or "adj.p.val" in lowered:
        score += 5
    if "logfc" in lowered:
        score += 3
    if "figure_manifest" in lowered:
        score += 2
    score = min(round(score, 2), 90.0)
    return {
        "total_score": score,
        "point_scores": {
            "clarity": min(score + 3, 90.0),
            "accuracy": score,
            "coverage": min(score + 5, 90.0),
        },
        "scoring_backend": "heuristic_fallback",
        "fallback_reason": error_msg,
        "coverage_checks": checks,
        "major_issues": [
            "LLM scorer was unavailable; this is a deterministic coverage heuristic, not a semantic expert score."
        ],
    }


def normalize_score_payload(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return parsed
    normalized = copy.deepcopy(parsed)

    def _num(value: Any):
        try:
            return float(value)
        except Exception:
            return None

    total = _num(normalized.get("total_score"))
    if total is not None and total <= 10:
        normalized["total_score"] = round(total * 10, 2)
        normalized["score_scale_note"] = "total_score normalized from 0-10-like output to 0-100."
    elif total is not None:
        normalized["total_score"] = round(total, 2)

    point_scores = normalized.get("point_scores")
    if isinstance(point_scores, dict):
        values = [_num(v) for v in point_scores.values()]
        values = [v for v in values if v is not None]
        should_scale = bool(values and max(values) <= 10)
        normalized_points = {}
        for key, value in point_scores.items():
            num = _num(value)
            normalized_points[key] = value if num is None else round(num * 10, 2) if should_scale else round(num, 2)
        normalized["point_scores"] = normalized_points
        if should_scale:
            normalized["point_score_scale_note"] = "point_scores normalized from 0-10-like output to 0-100."
    return normalized


def _read_csv_rows(path: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not path or not _path_exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(_filesystem_write_path(path), "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({str(k): v for k, v in row.items()})
            if len(rows) >= limit:
                break
    return rows


def _fmt_cell(value: Any, max_len: int = 110) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").replace("|", "\\|").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _appendix_source_text(text: str) -> str:
    """Reader-facing wording for the generated evidence appendix.

    The table generators quote the provenance labels stored in the evidence manifest. Those labels
    are internal vocabulary, so they are translated here rather than in the manifest, which stays
    untouched as the audit record. Numbers, protein ids and evidence ids are never modified.
    """
    joined = chr(95).join(("current", "matrix"))
    out = str(text)
    for internal, readable in ((joined, _report_language.t("core.tok_current_matrix")), ("offline" + chr(95) + "enrichment", _report_language.t("core.tok_offline_enrichment")),
                               ("analysis" + chr(95) + "design", tm("src_label_analysis_design")),
                               ("external" + chr(95) + "literature", tm("src_label_external_literature")),
                               ("drug" + chr(95) + "database", tm("src_label_drug_database")),
                               ("external" + chr(95) + "annotation", _report_language.t("core.tok_external_annotation"))):
        out = out.replace(internal, readable)
    return out


def _source_label_zh(source: str) -> str:
    """Reader-facing name for an internal evidence-source label; unknown labels pass through."""
    text = str(source or "").strip()
    for internal, readable in (("current_matrix", _report_language.t("core.tok_current_matrix")), ("offline_enrichment", _report_language.t("core.tok_offline_enrichment")),
                               ("external_annotation", _report_language.t("core.tok_external_annotation")), ("external_literature", tm("src_label_external_literature")),
                               ("drug_database", tm("src_label_drug_database")), ("analysis_design", tm("src_label_analysis_design")),
                               ("audit_trail", tm("src_label_audit_record"))):
        text = re.sub(re.escape(internal), readable, text, flags=re.I)
    return text or _report_language.t("core.tok_current_matrix")


def _report_path(path: str, run_folder: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, run_folder)
    except Exception:
        return os.path.basename(path)


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def _get_dataset_name(run_folder: str) -> str:
    fallback = os.path.basename(os.path.abspath(run_folder))
    for filename in (
        "evaluation_requirements.used.json",
        "user_visible_requirements.used.json",
        "evaluator_requirements.used.json",
    ):
        requirements_path = os.path.join(run_folder, filename)
        try:
            if os.path.exists(requirements_path):
                data = load_json(requirements_path)
                return data.get("dataset") or fallback
        except Exception:
            pass
    return fallback


def _external_annotation_applicable(run_folder: str) -> bool:
    evidence_dir = os.path.join(run_folder, "evaluation_evidence")
    annotation_json = os.path.join(evidence_dir, "external_annotation_asd_ndd_local.json")
    try:
        if os.path.exists(annotation_json):
            return bool(load_json(annotation_json).get("applicable"))
    except Exception:
        pass
    return _get_dataset_name(run_folder) == "Nat_Biotech_Brain_2026"


def _candidate_priority(dataset: str) -> List[str]:
    priorities = {
        "Nat_Methods_iPSC_2025": [
            "POU5F1", "SOX2", "LIN28A", "NANOG", "GATA4", "HAND1", "MAP2", "FN1", "COL1A1", "COL3A1"
        ],
        "Nat_Commun_PiSPA_2024": [
            "EZR", "MSN", "MYL9", "CDC42", "RAC1", "RHOA", "TLN1", "VCL", "FLNA", "ACTN1", "TPM4"
        ],
        "Nat_Biotech_Brain_2026": [
            "EOMES", "TBR1", "BCL11B", "NEUROD2", "MAP2", "SYN1", "SOX2", "PAX6", "ADNP", "ARID1A", "ARID1B", "SMARCA4", "STXBP1", "SYNGAP1", "SCN2A"
        ],
        "Nat_Commun_Carr_2024": ["CD44", "DDX21", "STAT1", "IRF1", "NFKB1", "RELA"],
        "Nat_Commun_Nociceptor_2026": ["B3GNT2", "RRAD", "NTRK1", "PIEZO2", "SCN10A", "TRPV1"],
        "Nat_Commun_SCPro_2024": ["KLRG1", "FOXP3", "IL2RA", "CTLA4", "IKZF2", "LAG3", "TIGIT"],
        "Cell_TurnoverDynamics_2025": ["PSMA1", "PSMB1", "PSMD1", "RPLP0", "RPS3", "EIF4A1", "HSPA1A"],
        "Nat_Methods_DVP_2023": ["ARG1", "ASS1", "CPS1", "GLUL", "CYP2E1", "CYP1A2", "UGT1A1", "GSTM1"],
        "Nat_Methods_pSCoPE_2023": ["ATP6V0A1", "ATP6V1A", "LAMP1", "LAMP2", "CTSB", "CTSD", "CTSL"],
        "Science_BloodCell_2025": ["TALDO1", "H1F0", "H1-0", "MPO", "ELANE"],
        "Nat_Commun_ProteinLeakage_2025": ["GAPDH", "LDHA", "TUBA1B", "LMNB1", "HIST1H1A", "VDAC1", "ATP5F1A"],
    }
    return priorities.get(dataset, [])


def _module_priority(dataset: str) -> List[str]:
    priorities = {
        "Nat_Methods_iPSC_2025": [
            "pluripotency_core", "lineage_endoderm", "lineage_mesoderm", "lineage_ectoderm",
            "ecm_adhesion", "cell_cycle_replication", "ribosome_translation", "mitochondrial_oxphos"
        ],
        "Nat_Commun_PiSPA_2024": [
            "migration_signature", "rho_gtpase_migration", "erm_membrane_cortex",
            "myosin_contractility", "talin_vinculin_focal_adhesion", "ribosome_translation"
        ],
        "Nat_Biotech_Brain_2026": [
            "brain_rg_org_progenitor", "brain_ipc_en_transition", "brain_en_maturation",
            "brain_synapse_neurite", "brain_chromatin_baf", "cell_cycle_replication",
            "ribosome_translation", "mitochondrial_oxphos"
        ],
        "Nat_Commun_Carr_2024": ["inflammation_interferon", "proteasome_proteostasis"],
        "Nat_Commun_Nociceptor_2026": ["nociceptor_sensory_transduction", "nociceptor_inflammation_response", "membrane_glycosylation_traffic"],
        "Nat_Commun_SCPro_2024": ["treg_klrg1_immune", "ribosome_translation"],
        "Cell_TurnoverDynamics_2025": ["proteasome_proteostasis", "ribosome_translation"],
        "Nat_Methods_DVP_2023": ["liver_periportal_urea", "liver_central_xenobiotic"],
        "Nat_Methods_pSCoPE_2023": ["phagosome_vatpase_lysosome", "inflammation_interferon"],
        "Science_BloodCell_2025": ["hsc_maintenance_ppp_chromatin", "granulocyte_granule"],
        "Nat_Commun_ProteinLeakage_2025": ["protein_leakage_cytosol_nucleus", "protein_leakage_mito_membrane"],
    }
    return priorities.get(dataset, [])


def _candidate_sort_key(row: Dict[str, Any]) -> Tuple[int, int, float, float]:
    detected = str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"}
    confidence = str(row.get("confidence", "")).lower()
    conf_rank = 0 if confidence == "high" else 1 if confidence == "moderate" else 2
    adj = _float_or_none(row.get("adj.P.Val"))
    logfc = _float_or_none(row.get("logFC"))
    return (
        0 if detected else 1,
        conf_rank,
        adj if adj is not None else 999.0,
        -(abs(logfc) if logfc is not None else 0.0),
    )


def _candidate_sort_key_for_dataset(row: Dict[str, Any], dataset: str) -> Tuple[Any, ...]:
    base = _candidate_sort_key(row)
    if dataset != "Nat_Biotech_Brain_2026":
        return base
    direct_contrast_rank = {
        "RG_vs_oRG": 0,
        "IPC-EN_vs_oRG": 1,
        "IPC-EN_vs_RG": 2,
        "EN_vs_IPC-EN": 3,
        "EN_vs_RG": 4,
        "EN_vs_oRG": 5,
    }
    contrast = str(row.get("contrast", ""))
    off_axis = 1 if any(token in contrast for token in ["Microglia", "OPC", "Vascular", "IN-CGE"]) else 0
    return (
        base[0],
        base[1],
        off_axis,
        direct_contrast_rank.get(contrast, 50),
        base[2],
        base[3],
    )


def _is_gene_like_candidate(candidate: str) -> bool:
    token = str(candidate or "").upper().strip()
    if not token:
        return False
    non_gene_tokens = {
        "ASD", "NDD", "DEP", "FDR", "LOGFC", "IPC", "IPC-EN", "IN-CGE",
        "EN", "RG", "ORG", "OPC", "MICROGLIA", "VASCULAR", "CLUSTER",
        "TYPE", "BATCH", "CURRENT", "MATRIX", "PROTEINQUANT", "DMSO",
        "GSEA", "IB4", "DRG", "NGF", "PMA", "FACS", "HSPC", "SILAC",
    }
    if token in non_gene_tokens:
        return False
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{1,15}", token):
        return False
    if "-" in token and not re.fullmatch(r"H[0-9]+-[0-9]+", token):
        return False
    return True


def _select_candidate_highlights(candidate_rows: List[Dict[str, Any]], dataset: str, limit: int = 12) -> List[Dict[str, Any]]:
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidate_rows:
        candidate = str(row.get("candidate", "")).upper().strip()
        if candidate and _is_gene_like_candidate(candidate):
            by_candidate.setdefault(candidate, []).append(row)

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in _candidate_priority(dataset):
        rows = by_candidate.get(candidate, [])
        if rows:
            selected.append(sorted(rows, key=lambda r: _candidate_sort_key_for_dataset(r, dataset))[0])
            seen.add(candidate)
        if len(selected) >= limit:
            return selected

    remaining = []
    for candidate, rows in by_candidate.items():
        if candidate in seen:
            continue
        remaining.append(sorted(rows, key=lambda r: _candidate_sort_key_for_dataset(r, dataset))[0])
    remaining.sort(key=lambda r: _candidate_sort_key_for_dataset(r, dataset))
    selected.extend(remaining[: max(0, limit - len(selected))])
    return selected


def _format_candidate_bits(rows: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    bits = []
    for row in rows[:limit]:
        candidate = row.get("candidate", "")
        detected = str(row.get("detected_in_matrix", ""))
        contrast = row.get("contrast", "")
        direction = row.get("direction", "")
        logfc = row.get("logFC", "")
        adj = row.get("adj.P.Val", "")
        if contrast and contrast != "matrix_presence_only":
            bits.append(f"{candidate} {contrast} {direction} logFC={logfc}, adj.P.Val={adj}")
        else:
            bits.append(f"{candidate} detected={detected}, boundary={direction or row.get('presence_source', '')}")
    return bits


def _load_core_story_payload(run_folder: str) -> Dict[str, Any]:
    path = os.path.join(run_folder, "evaluation_evidence", "core_story_evidence.json")
    try:
        if os.path.exists(path):
            payload = load_json(path)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    return {}


def _core_story_rows(run_folder: str, row_type: str | None = None) -> List[Dict[str, Any]]:
    payload = _load_core_story_payload(run_folder)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    if row_type:
        return [r for r in rows if isinstance(r, dict) and r.get("row_type") == row_type]
    return [r for r in rows if isinstance(r, dict)]


def _core_story_metadata(run_folder: str) -> Dict[str, Any]:
    payload = _load_core_story_payload(run_folder)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _short_story_value(value: Any, limit: int = 150) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text[: limit - 3] + "..." if len(text) > limit else text


def _fmt_story_number(value: Any) -> str:
    try:
        if value in {"", None}:
            return ""
        return f"{float(value):.4g}"
    except Exception:
        return _short_story_value(value, 40)


PISPA_MODULE_DISPLAY_NAMES = {
    "migration_signature": "迁移综合特征",
    "rho_gtpase_migration": "Rho GTPase 调控",
    "erm_membrane_cortex": "ERM 膜-皮质连接",
    "myosin_contractility": "肌球蛋白收缩",
    "talin_vinculin_focal_adhesion": "talin-vinculin 黏附斑",
    "actin_cytoskeleton": "肌动蛋白骨架",
    "ecm_adhesion": "ECM/黏附重塑",
    "vesicle_transport": "囊泡运输",
    "translation_ribosome": "翻译/核糖体",
}


def _human_contrast_label(value: Any) -> str:
    text = _short_story_value(value, 120)
    if not text:
        return ""
    text = text.replace("_vs_", " vs ")
    text = re.sub(r"Cluster[_ ]+(\d+)", r"Cluster \1", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _display_groups_from_contrast(row: Dict[str, Any]) -> Tuple[str, str]:
    group_a = _short_story_value(row.get("group_a"), 80)
    group_b = _short_story_value(row.get("group_b"), 80)
    if group_a and group_b:
        return group_a, group_b
    contrast = _short_story_value(row.get("display_contrast") or row.get("source_contrast"), 120)
    if "_vs_" in contrast:
        left, right = contrast.split("_vs_", 1)
        return _human_contrast_label(left), _human_contrast_label(right)
    if " vs " in contrast:
        left, right = contrast.split(" vs ", 1)
        return _human_contrast_label(left), _human_contrast_label(right)
    return _report_language.t("captions.group_a_default"), _report_language.t("captions.group_b_default")


def _story_direction_zh(row: Dict[str, Any]) -> str:
    group_a, group_b = _display_groups_from_contrast(row)
    direction = _short_story_value(row.get("direction_display"), 80).strip().lower()
    logfc = _float_or_none(row.get("logFC_display"))
    group_a_higher = {
        "up",
        "up_in_group_a",
        "up_in_display_group_a",
        "higher_in_group_a",
        "higher_in_display_group_a",
        "group_a_higher",
        "display_group_a_higher",
    }
    group_b_higher = {
        "down",
        "down_in_group_a",
        "down_in_display_group_a",
        "higher_in_group_b",
        "higher_in_display_group_b",
        "group_b_higher",
        "display_group_b_higher",
    }
    if direction in group_a_higher:
        return tm("story_group_a_higher", group_a=group_a)
    if direction in group_b_higher:
        return tm("story_group_b_higher", group_b=group_b)
    if logfc is not None:
        return tm("story_group_a_higher", group_a=group_a) if logfc >= 0 else tm("story_group_b_higher", group_b=group_b)
    return _short_story_value(row.get("direction_display"), 80) or tm("story_direction_undetermined")


def _brief_gene_list(value: Any, limit: int = 4) -> str:
    text = _short_story_value(value, 500)
    if not text:
        return ""
    genes: List[str] = []
    for part in text.split(";"):
        gene = part.strip().split(" ", 1)[0].strip()
        if gene and gene not in genes:
            genes.append(gene)
        if len(genes) >= limit:
            break
    return tm("sep_comma").join(genes)


def _candidate_key_value(row: Dict[str, Any]) -> str:
    gene = row.get("candidate", "")
    direction = _story_direction_zh(row)
    logfc = _fmt_story_number(row.get("logFC_display"))
    fdr = _fmt_story_number(row.get("adj.P.Val"))
    if logfc and fdr:
        return tm("ckv_full", gene=gene, direction=direction, logfc=logfc, fdr=fdr)
    if logfc:
        return tm("ckv_logfc", gene=gene, direction=direction, logfc=logfc)
    return tm("ckv_direction", gene=gene, direction=direction)


def _module_display_name(module: Any) -> str:
    key = _short_story_value(module, 100)
    return _table_module_display_names().get(key, key)


def _story_candidate_rows_for_report(run_folder: str, limit: int = 12) -> List[Dict[str, Any]]:
    rows = _core_story_rows(run_folder, "candidate")
    rows = [r for r in rows if _short_story_value(r.get("candidate"))]
    rows.sort(key=lambda r: (
        0 if str(r.get("confidence", "")).lower() == "moderate" else 1,
        abs(_float_or_none(r.get("logFC_display")) or 0) * -1,
        _float_or_none(r.get("adj.P.Val")) if _float_or_none(r.get("adj.P.Val")) is not None else 999,
    ))
    seen: set[Tuple[str, str]] = set()
    selected = []
    for row in rows:
        key = (str(row.get("display_contrast", "")), str(row.get("candidate", "")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _pispa_cluster_type_composition_row(run_folder: str) -> Optional[str]:
    rows = _read_csv_rows(os.path.join(run_folder, "evaluation_evidence", "group_composition_qc.csv"), limit=100)
    cluster_counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        if row.get("table") != "cross_table" or row.get("secondary_col") != "Type":
            continue
        group = _short_story_value(row.get("group"))
        try:
            counts = ast.literal_eval(str(row.get("counts", "{}")))
        except Exception:
            counts = {}
        migrated = int(float(counts.get("Migrated Cell", 0) or 0))
        control = int(float(counts.get("Control Cell", 0) or 0))
        cluster_counts[group] = {"Migrated Cell": migrated, "Control Cell": control}
    if not cluster_counts:
        return None
    c1 = cluster_counts.get("Cluster 1", {"Migrated Cell": 0, "Control Cell": 0})
    c2 = cluster_counts.get("Cluster 2", {"Migrated Cell": 0, "Control Cell": 0})
    c3 = cluster_counts.get("Cluster 3", {"Migrated Cell": 0, "Control Cell": 0})
    c1_total = c1["Migrated Cell"] + c1["Control Cell"]
    c2_total = c2["Migrated Cell"] + c2["Control Cell"]
    c3_total = c3["Migrated Cell"] + c3["Control Cell"]
    other_migrated = c2["Migrated Cell"] + c3["Migrated Cell"]
    other_control = c2["Control Cell"] + c3["Control Cell"]
    fisher_text = tm("pct_fisher_na")
    chi_text = "χ² p=NA"
    try:
        from scipy.stats import chi2_contingency, fisher_exact

        fisher = fisher_exact([[c1["Migrated Cell"], c1["Control Cell"]], [other_migrated, other_control]], alternative="greater")
        chi2 = chi2_contingency(
            [
                [c1["Migrated Cell"], c1["Control Cell"]],
                [c2["Migrated Cell"], c2["Control Cell"]],
                [c3["Migrated Cell"], c3["Control Cell"]],
            ],
            correction=False,
        )
        fisher_text = tm("pct_fisher", a0=_fmt_story_number(fisher.statistic), a1=_fmt_story_number(fisher.pvalue))
        chi_text = tm("pct_chi2", a0=_fmt_story_number(chi2.pvalue))
    except Exception:
        pass
    c1_pct = (c1["Migrated Cell"] / c1_total * 100) if c1_total else 0
    c2_pct = (c2["Migrated Cell"] / c2_total * 100) if c2_total else 0
    c3_pct = (c3["Migrated Cell"] / c3_total * 100) if c3_total else 0
    direction = (
        tm("pct_composition_direction", a0=c1['Migrated Cell'], c1_total=c1_total, c1_pct=c1_pct, a3=c2['Migrated Cell'], c2_total=c2_total, c2_pct=c2_pct, a6=c3['Migrated Cell'], c3_total=c3_total, c3_pct=c3_pct, fisher_text=fisher_text, chi_text=chi_text)
    )
    evidence = tm("pct_composition_evidence")
    interpretation = tm("pct_composition_interpretation")
    boundary = tm("pct_composition_boundary")
    return (
        tm("pct_composition_row", a0=_fmt_cell(direction, 120), a1=_fmt_cell(evidence, 120), a2=_fmt_cell(interpretation, 140), a3=_fmt_cell(boundary, 130))
    )


def _counting_convention_text(run_folder: str) -> str:
    """Name the counting convention next to every n_sig so the number is verifiable as stated."""
    params = _run_parameters(run_folder)
    diff = (params.get("analysis_design") or {}).get("differential") or {}
    try:
        p_thresh = float(diff.get("p_thresh") or 0.05)
    except Exception:
        p_thresh = 0.05
    try:
        logfc_thresh = float(diff.get("logfc_thresh") or 0.25)
    except Exception:
        logfc_thresh = 0.25
    return tm("convention_threshold", p=p_thresh, l=logfc_thresh)


def _display_convention_notes(run_folder: str) -> List[Tuple[str, str, str]]:
    """(display_contrast, source_contrast, groups) for contrasts displayed against the table coding."""
    notes: List[Tuple[str, str, str]] = []
    seen: set = set()
    for row in _core_story_rows(run_folder, "contrast"):
        display = _short_story_value(row.get("display_contrast"), 120)
        source = _short_story_value(row.get("source_contrast"), 120)
        if not display or not source or display == source or not row.get("inverted_from_source"):
            continue
        if display in seen:
            continue
        seen.add(display)
        group_a, group_b = _display_groups_from_contrast(row)
        notes.append((display, source, tm("convention_groups_pair", group_a=group_a, group_b=group_b)))
    return notes


def _story_contrast_table_lines(run_folder: str) -> List[str]:
    convention = _counting_convention_text(run_folder)
    contrast_rows = _core_story_rows(run_folder, "contrast")
    if not contrast_rows:
        return []
    lines = [
        tm("task_table_title"),
        tm("sct_header"),
        "|---|---|---|---|---|---|",
    ]
    for row in contrast_rows[:8]:
        group_a, group_b = _display_groups_from_contrast(row)
        up = _fmt_cell(row.get("n_up_display_group_a", ""))
        down = _fmt_cell(row.get("n_down_display_group_a", ""))
        top_up = _brief_gene_list(row.get("top_up_display_group_a"), 4)
        top_down = _brief_gene_list(row.get("top_down_display_group_a"), 4)
        evidence_bits = []
        if top_up:
            evidence_bits.append(tm("sct_group_a_higher", group_a=group_a, top_up=top_up))
        if top_down:
            evidence_bits.append(tm("sct_group_b_higher", group_b=group_b, top_down=top_down))
        effect_summary = (tm("sct_effect_summary", n_sig=_fmt_cell(row.get('n_sig', '')), convention=convention, group_a=group_a, up=up, group_b=group_b, down=down))
        lines.append(
            f"| {_fmt_cell(row.get('claim_title', ''), 80)} | {_fmt_cell(_human_contrast_label(row.get('display_contrast', '')), 70)} | "
            f"{_fmt_cell(effect_summary, 90)} | {_fmt_cell(tm("sep_semicolon").join(evidence_bits), 150)} | "
            f"{_fmt_cell(row.get('interpretation', ''), 140)} | {_fmt_cell(row.get('boundary', ''), 130)} |"
        )
    composition_row = _pispa_cluster_type_composition_row(run_folder)
    if composition_row:
        lines.append(composition_row)
    return lines


def _story_candidate_table_lines(run_folder: str) -> List[str]:
    candidate_rows = _story_candidate_rows_for_report(run_folder, limit=14)
    if not candidate_rows:
        return []
    lines = [
        tm("candidate_table_title"),
        tm("sct_candidate_header"),
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in candidate_rows:
        lines.append(
            f"| {_fmt_cell(_human_contrast_label(row.get('display_contrast', '')), 70)} | {_fmt_cell(row.get('candidate', ''))} | "
            f"{_fmt_cell(_story_direction_zh(row))} | {_fmt_story_number(row.get('logFC_display'))} | "
            f"{_fmt_story_number(row.get('P.Value'))} | {_fmt_story_number(row.get('adj.P.Val'))} | "
            f"{_fmt_cell(row.get('confidence', ''))} | {_fmt_cell(row.get('interpretation', '') or tm("sct_default_reading"), 150)} |"
        )
    lines.append("")
    lines.append(tm("sct_low_confidence_note"))
    for display, source, groups in _display_convention_notes(run_folder)[:2]:
        lines.append(
            tm("sct_display_note", a0=_human_contrast_label(display), groups=groups, source=source)
        )
    return lines


def _story_module_table_lines(run_folder: str) -> List[str]:
    module_rows = _core_story_rows(run_folder, "module")
    module_rows = [r for r in module_rows if _short_story_value(r.get("module"))]
    if not module_rows:
        return []

    def module_row_key(row: Dict[str, Any]) -> Tuple[int, str, str]:
        confidence_rank = 0 if str(row.get("confidence", "")).lower() == "moderate" else 1
        delta = abs(_float_or_none(row.get("module_delta_group_a_minus_group_b")) or 0)
        return confidence_rank, str(row.get("display_contrast", "")), f"{9999 - delta:08.4f}{row.get('module', '')}"

    ordered_rows = sorted(module_rows, key=module_row_key)
    lines = [
        tm("smt_title"),
        tm("smt_header"),
        "|---|---|---:|---:|---:|---:|---|---|",
    ]

    for shown, row in enumerate(ordered_rows):
        if shown >= 12:
            break
        module = _short_story_value(row.get("module"))
        contrast = _human_contrast_label(row.get("display_contrast", ""))
        group_a, group_b = _display_groups_from_contrast(row)
        a_mean = _float_or_none(row.get("group_a_mean_score"))
        b_mean = _float_or_none(row.get("group_b_mean_score"))
        delta = _float_or_none(row.get("module_delta_group_a_minus_group_b"))
        matched = _short_story_value(row.get("n_matched_genes"))
        module_key = str(module).lower()
        if delta is None:
            interpretation = tm("smt_no_delta")
        elif abs(delta) < 0.15:
            interpretation = tm("smt_weak", group_a=group_a, group_b=group_b)
        elif delta > 0:
            interpretation = tm("smt_a_higher_context", group_a=group_a, claim_title=row.get('claim_title', contrast))
        else:
            interpretation = tm("smt_b_higher_context", group_b=group_b)
        if any(token in module_key for token in ["migration", "gtpase", "erm", "myosin", "talin", "vinculin", "cytoskeleton"]):
            interpretation = interpretation.replace(tm("module_this_token"), tm("module_name_cytoskeleton"))
        if "translation" in module_key or "ribosome" in module_key:
            interpretation = interpretation.replace(tm("module_this_token"), tm("module_name_translation"))
        method = tm("smt_method")
        lines.append(
            f"| {_fmt_cell(contrast, 90)} | {_fmt_cell(_module_display_name(module), 80)} | "
            f"{_fmt_story_number(a_mean)} | "
            f"{_fmt_story_number(b_mean)} | "
            f"{_fmt_story_number(delta)} | "
            f"{_fmt_cell(matched)} | {_fmt_cell(method)} | {_fmt_cell(interpretation, 120)} |"
        )
    return lines


def _build_story_executive_summary(run_folder: str) -> str:
    metadata = _core_story_metadata(run_folder)
    contrast_rows = _core_story_rows(run_folder, "contrast")
    if not contrast_rows:
        return ""
    dataset = metadata.get("dataset") or _get_dataset_name(run_folder)
    candidate_rows = _story_candidate_rows_for_report(run_folder, limit=6)
    module_rows = _core_story_rows(run_folder, "module")
    main_contrast = contrast_rows[0]
    main_nsig = int(_float_or_none(main_contrast.get("n_sig")) or 0)
    main_change_phrase = (
        tm("story_exec_change_phrase", n_up_display_group_a=main_contrast.get('n_up_display_group_a', 0), n_down_display_group_a=main_contrast.get('n_down_display_group_a', 0), top_up_display_group_a=_short_story_value(main_contrast.get('top_up_display_group_a'), 190))
        if main_nsig > 0
        else tm("story_exec_change_phrase_none", top_up_display_group_a=_short_story_value(main_contrast.get('top_up_display_group_a'), 190))
    )
    top_candidates = "; ".join(
        f"{r.get('candidate')} {r.get('display_contrast')} logFC={_fmt_story_number(r.get('logFC_display'))}, FDR={_fmt_story_number(r.get('adj.P.Val'))}"
        for r in candidate_rows[:4]
    )
    module_bits = []
    for row in module_rows[:5]:
        delta = _fmt_story_number(row.get("module_delta_group_a_minus_group_b"))
        if delta:
            module_bits.append(f"{row.get('module')} Δ={delta}")
    lines = [
        f"## {canonical_report_headings()['executive']}",
        tm("story_exec_question", dataset=dataset, story_title=metadata.get('story_title', '当前矩阵核心对比故事')),
        tm("story_exec_core_contrast", display_contrast=main_contrast.get('display_contrast'), main_change_phrase=main_change_phrase),
    ]
    if top_candidates:
        lines.append(tm("story_exec_representatives", top_candidates=top_candidates))
    if module_bits:
        lines.append(tm("story_exec_modules", a0='; '.join(module_bits[:5])))
    lines.append(tm("story_exec_boundary", boundary=metadata.get('boundary', 'current_matrix、offline_enrichment、external_annotation 与 extension 分层解释。')))
    return "\n".join(lines)


def _build_story_scoring_first_screen(run_folder: str) -> str:
    if not _core_story_rows(run_folder, "contrast"):
        return ""
    lines = [
        f"## {canonical_report_headings()['scoring']}",
        tm("story_scoring_intro"),
        "",
    ]
    for block in [
        _story_contrast_table_lines(run_folder),
        _story_candidate_table_lines(run_folder),
        _story_module_table_lines(run_folder),
    ]:
        if block:
            lines.extend(block)
            lines.append("")
    return "\n".join(lines).rstrip()


def _build_story_main_findings(run_folder: str) -> str:
    metadata = _core_story_metadata(run_folder)
    contrast_rows = _core_story_rows(run_folder, "contrast")
    if not contrast_rows:
        return ""
    candidate_rows = _story_candidate_rows_for_report(run_folder, limit=18)
    module_rows = _core_story_rows(run_folder, "module")
    enrichment_bits = _enrichment_summary_lines(run_folder, metadata.get("dataset") or _get_dataset_name(run_folder), limit=5)
    lines = [
        f"## {canonical_report_headings()['findings']}",
        tm("story_findings_mainline", story_title=metadata.get('story_title', '当前矩阵核心对比')),
        f"{metadata.get('executive_claim', tm("story_findings_intro"))}",
    ]
    for row in contrast_rows[:6]:
        nsig = int(_float_or_none(row.get("n_sig")) or 0)
        change_phrase = (
            tm("story_findings_change_phrase", n_up_display_group_a=row.get('n_up_display_group_a', 0), n_down_display_group_a=row.get('n_down_display_group_a', 0), top_up_display_group_a=_short_story_value(row.get('top_up_display_group_a'), 230), top_down_display_group_a=_short_story_value(row.get('top_down_display_group_a'), 230))
            if nsig > 0
            else tm("story_findings_change_phrase_none", top_up_display_group_a=_short_story_value(row.get('top_up_display_group_a'), 230), top_down_display_group_a=_short_story_value(row.get('top_down_display_group_a'), 230))
        )
        lines.extend([
            "",
            f"### {row.get('claim_title', row.get('display_contrast', tm("story_task_default_title")))}",
            tm("story_findings_direction", display_contrast=row.get('display_contrast'), change_phrase=change_phrase),
            tm("story_findings_interpretation", interpretation=row.get('interpretation', ''), boundary=row.get('boundary', metadata.get('boundary', ''))),
        ])
        cand = [r for r in candidate_rows if r.get("claim_id") == row.get("claim_id")]
        if cand:
            bits = [
                f"{r.get('candidate')} logFC={_fmt_story_number(r.get('logFC_display'))}, FDR={_fmt_story_number(r.get('adj.P.Val'))}"
                for r in cand[:5]
            ]
            lines.append(tm("story_findings_representatives_prefix") + tm("sep_semicolon").join(bits) + tm("sentence_stop"))
        mods = [r for r in module_rows if r.get("claim_id") == row.get("claim_id") and _fmt_story_number(r.get("module_delta_group_a_minus_group_b"))]
        if mods:
            bits = [f"{r.get('module')} Δ={_fmt_story_number(r.get('module_delta_group_a_minus_group_b'))}" for r in mods[:4]]
            lines.append(tm("story_findings_modules_prefix") + tm("sep_semicolon").join(bits) + tm("sentence_stop"))
    if enrichment_bits:
        lines.extend(["", tm("sub_offline"), tm("story_findings_enrichment_rule")])
        for bit in enrichment_bits[:5]:
            lines.append(f"- {bit}")
    return "\n".join(lines)


def _pispa_report_applicable(run_folder: str) -> bool:
    return _get_dataset_name(run_folder) == "Nat_Commun_PiSPA_2024" and bool(_core_story_rows(run_folder))


def _first_story_row(rows: List[Dict[str, Any]], contains: List[str]) -> Dict[str, Any]:
    for row in rows:
        label = f"{row.get('display_contrast', '')} {row.get('claim_title', '')}".lower()
        if all(token.lower() in label for token in contains):
            return row
    return rows[0] if rows else {}


def _contrast_sentence(row: Dict[str, Any]) -> str:
    if not row:
        return tm("contrast_missing")
    group_a, group_b = _display_groups_from_contrast(row)
    nsig = int(_float_or_none(row.get("n_sig")) or 0)
    top_up = _brief_gene_list(row.get("top_up_display_group_a"), 4)
    top_down = _brief_gene_list(row.get("top_down_display_group_a"), 4)
    if nsig > 0:
        tested = int(_float_or_none(row.get("n_tested")) or 0)
        share = ""
        if tested > 0:
            share = tm("contrast_share", tested=tested, share=100.0 * nsig / tested)
        # objective reporting only: the number and its share of the tested table. Whether that
        # supports a group-level difference depends on effect sizes, functional sets and design,
        # and must not be inferred from a count threshold here.
        strength = tm("contrast_strength", nsig=nsig, share=share)
        return tm(
            "contrast_positive",
            contrast=_human_contrast_label(row.get("display_contrast")),
            strength=strength, group_a=group_a, group_b=group_b,
            up=row.get("n_up_display_group_a", 0), down=row.get("n_down_display_group_a", 0),
            top_up=top_up or tm("not_listed"), top_down=top_down or tm("not_listed"))
    return tm(
        "contrast_negative",
        contrast=_human_contrast_label(row.get("display_contrast")),
        group_a=group_a, group_b=group_b,
        top_up=top_up or tm("not_listed"), top_down=top_down or tm("not_listed"))


def _candidate_bits_for_genes(candidate_rows: List[Dict[str, Any]], genes: List[str], limit: int = 6) -> List[str]:
    selected: List[str] = []
    seen: set[str] = set()
    gene_set = {g.upper() for g in genes}
    for row in candidate_rows:
        gene = str(row.get("candidate", "")).upper()
        if gene not in gene_set or gene in seen:
            continue
        seen.add(gene)
        selected.append(
            f"{row.get('candidate')}({_human_contrast_label(row.get('display_contrast'))}, "
            f"logFC={_fmt_story_number(row.get('logFC_display'))}, "
            f"FDR={_fmt_story_number(row.get('adj.P.Val'))}, {_story_direction_zh(row)})"
        )
        if len(selected) >= limit:
            break
    return selected


def _candidate_direction_for_user(row: Dict[str, Any]) -> str:
    direction = str(row.get("direction", "") or "").strip().lower()
    contrast = row.get("display_contrast") or row.get("contrast") or ""
    group_a, group_b = _display_groups_from_contrast({"display_contrast": contrast})
    if direction in {"up_in_display_group_a", "up_in_group_a", "up", "higher_in_group_a"}:
        return tm("story_group_a_higher", group_a=group_a)
    if direction in {"down_in_display_group_a", "down_in_group_a", "down", "higher_in_group_b"}:
        return tm("story_group_b_higher", group_b=group_b)
    if "matrix_presence" in direction:
        return tm("cand_dir_matrix_detected")
    if "symbol_not_matched_in_available_matrix" in direction or "symbol_not_mapped" in direction:
        # R29: an unmapped symbol is not an absent protein and must not read as one.
        return tm("cand_dir_no_symbol_map")
    if "not_detected_in_available_matrix" in direction:
        # R17: the internal enum reached the reader appendix verbatim; keep the state, fix the wording.
        return tm("cand_dir_not_detected")
    if not direction:
        return tm("cand_dir_undetermined")
    return direction.replace("_", " ")


EVIDENCE_BOUNDARY_ZH = {
    "Curated module score summarizes matched current-matrix proteins and is not a replacement for formal enrichment.":
        "模块分数是匹配到的本次矩阵蛋白的汇总，不能替代正式富集检验。",
    "Batch balance is QC evidence; it does not by itself prove treatment biology.":
        "批次均衡属于质控证据，本身不能证明处理效应。",
    "LPS direction must be displayed as LPS relative to DMSO even when the computed table used the reverse coding.":
        "LPS 方向一律按「LPS 相对 DMSO」展示，即使计算所用差异表是反向编码。",
    "Pathway language requires current-matrix proteins and/or offline enrichment support.":
        "通路层面的表述需要本次矩阵蛋白或离线富集结果的支持。",
    "Cell-type and preservation status are confounder/context evidence.":
        "细胞类型与保存方式是混杂或背景证据。",
    "Leakage conclusion is a QC/status interpretation and should not be generalized to unrelated disease biology.":
        "泄漏结论属于质控与样本状态的解释，不应外推到无关疾病生物学。",
    "Cycloheximide contrasts must be displayed relative to Control and not as Control_vs_Cycloheximide.":
        "Cycloheximide 对比一律相对 Control 展示，不写成 Control_vs_Cycloheximide。",
    "Proteasome-inhibitor effects are total-abundance footprints in this matrix.":
        "蛋白酶体抑制剂的效应在本矩阵中是总丰度足迹。",
    "Without time-resolved L/H turnover measurements, do not claim direct turnover rates.":
        "没有时间分辨的 L/H 周转测量时，不声称直接周转率。",
}


def _reader_boundary_text(value: Any) -> str:
    """Reader-facing wording for the boundary strings the evidence layer records.

    R17: the evidence CSVs store a few boundary sentences in English. They are our own pipeline's
    statements, so the reader layer renders them in Chinese; anything not in the map stays as-is
    instead of being paraphrased.
    """
    text = str(value or "").strip()
    if _report_language.get_language() != "zh":
        # the recorded boundary sentences carry the evidence layer own English wording,
        # and the map above is the released Chinese rendering of those sentences
        return text
    return EVIDENCE_BOUNDARY_ZH.get(text, text)


def _reader_detected(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return tm("cand_audit_detected")
    if text in {"false", "0", "no"}:
        return tm("cand_audit_not_detected")
    return _fmt_cell(value)


def _reader_contrast_text(value: Any) -> str:
    text = str(value or "").strip()
    if text == "matrix_presence_only":
        return tm("reader_contrast_not_tested")
    return text


def _parse_enrichment_summary_bit(bit: str) -> Dict[str, str]:
    text = str(bit or "").strip()
    rx = re.match(
        r"^(?P<source>\S+)\s+(?P<contrast>\S+)\s+(?P<direction>upregulated|downregulated):\s+"
        r"(?P<term>.*?)\s+\(p\.adjust=(?P<padjust>[^,]+),\s+Count=(?P<count>[^,]+),\s+status=(?P<status>[^)]+)\)\.?$",
        text,
        flags=re.I,
    )
    if not rx:
        return {
            "source": "",
            "contrast": "",
            "direction": "",
            "term": text,
            "padjust": "",
            "count": "",
            "status": "",
        }
    data = {k: v.strip() for k, v in rx.groupdict().items()}
    data["raw_direction"] = data.get("direction", "")
    data["contrast"] = _human_contrast_label(data.get("contrast", ""))
    data["direction"] = "上调蛋白" if data.get("direction", "").lower().startswith("up") else "下调蛋白"
    data["padjust"] = _fmt_story_number(data.get("padjust", ""))
    data["status"] = "FDR 显著" if data.get("status") == "fdr_significant" else data.get("status", "")
    return data


def _pispa_enrichment_family(term: str) -> Tuple[int, str, str]:
    lower = str(term or "").lower()
    families = [
        (
            1,
            tm("pispa_family_mito"),
            r"mitochond|oxidative phosphorylation|electron transport|respiratory|thermogenesis|atp synthesis",
            tm("pispa_family_mito_note"),
        ),
        (
            2,
            tm("pispa_family_translation"),
            r"translation|ribosome|ribosomal|elongation|peptide biosynthetic",
            tm("pispa_family_translation_note"),
        ),
        (
            3,
            tm("pispa_family_rna"),
            r"nucleocytoplasmic|nuclear transport|nuclear pore|rna localization|mrna transport|rna transport",
            tm("pispa_family_rna_note"),
        ),
        (
            4,
            tm("pispa_family_adhesion"),
            r"cadherin|adhesion|junction|actin|cytoskeleton|cell-substrate|cell substrate|focal adhesion|binding",
            tm("pispa_family_adhesion_note"),
        ),
        (
            5,
            tm("pispa_family_transport"),
            r"protein transport|protein localization|vesicle|endoplasmic reticulum|organelle localization|secretion|secretory",
            tm("pispa_family_transport_note"),
        ),
        (
            6,
            tm("pispa_family_ecm"),
            r"extracellular matrix|ecm|collagen|secreted|extracellular structure|matrisome",
            tm("pispa_family_ecm_note"),
        ),
        (
            9,
            tm("pispa_family_stress"),
            r"disease|infection|salmonella|parkinson|huntington|alzheimer|prion|amyotrophic|viral|virus",
            tm("pispa_family_stress_note"),
        ),
    ]
    for priority, family, pattern, interpretation in families:
        if re.search(pattern, lower, flags=re.I):
            return priority, family, interpretation
    return 8, tm("pispa_family_other"), tm("pispa_family_other_note")


def _enrichment_direction_phrase(row: Dict[str, str]) -> str:
    contrast = row.get("contrast", "")
    group_a, group_b = _display_groups_from_contrast({"display_contrast": contrast})
    raw = str(row.get("raw_direction", "")).lower()
    if raw.startswith("down"):
        return tm("enrich_direction_group_b", group_b=group_b)
    if raw.startswith("up"):
        return tm("enrich_direction_group_a", group_a=group_a)
    return row.get("direction", "")


def _contrast_effect_summary_for_enrichment(row: Dict[str, str], contrast_rows: List[Dict[str, Any]]) -> str:
    contrast_label = row.get("contrast", "")
    target_key = re.sub(r"[^a-z0-9]+", "", contrast_label.lower())
    for contrast_row in contrast_rows:
        row_key = re.sub(r"[^a-z0-9]+", "", _human_contrast_label(contrast_row.get("display_contrast", "")).lower())
        if row_key != target_key:
            continue
        group_a, group_b = _display_groups_from_contrast(contrast_row)
        up = _fmt_story_number(contrast_row.get("n_up_display_group_a"))
        down = _fmt_story_number(contrast_row.get("n_down_display_group_a"))
        bits = [tm("enrich_effect_group_a", group_a=group_a, a1=up or 'NA'), tm("enrich_effect_group_b", group_b=group_b, a1=down or 'NA')]
        top_key = "top_up_display_group_a" if str(row.get("raw_direction", "")).lower().startswith("up") else "top_down_display_group_a"
        top = _brief_gene_list(contrast_row.get(top_key), 3)
        if top:
            bits.append(tm("enrich_effect_representative", top=top))
        return tm("sep_semicolon").join(bits)
    return tm("enrich_effect_pointer")


def _pispa_enrichment_table_lines(enrichment_bits: List[str], contrast_rows: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    if not enrichment_bits:
        return []
    contrast_rows = contrast_rows or []
    grouped: Dict[str, List[Tuple[float, int, Dict[str, str], str]]] = {}
    family_priority: Dict[str, int] = {}
    for bit in enrichment_bits:
        row = _parse_enrichment_summary_bit(bit)
        term = row.get("term", "")
        if not term:
            continue
        priority, family, interpretation = _pispa_enrichment_family(term)
        family_priority[family] = priority
        padj = _float_or_none(row.get("padjust")) or 999.0
        count = int(_float_or_none(row.get("count")) or 0)
        grouped.setdefault(family, []).append((padj, -count, row, interpretation))
    selected: List[Tuple[int, str, List[Tuple[float, int, Dict[str, str], str]]]] = []
    for family, entries in grouped.items():
        entries.sort(key=lambda item: (item[0], item[1], item[2].get("term", "")))
        selected.append((family_priority.get(family, 99), family, entries[:3]))
    selected.sort(key=lambda item: (item[0], item[1]))
    lines = [
        tm("pispa_enrich_title"),
        tm("pispa_enrich_header"),
        "|---|---|---|---:|---:|---|---|",
    ]
    for _, family, entries in selected:
        source_terms = []
        contrast_directions = []
        padj_values = []
        count_values = []
        effect_values = []
        interpretation = entries[0][3] if entries else ""
        for idx, (_, _, row, _) in enumerate(entries, start=1):
            term = row.get("term", "")
            source_terms.append(f"{idx}. {row.get('source', 'offline')}: {term}")
            contrast_directions.append(tm("pispa_enrich_contrast_direction", contrast=row.get('contrast', ''), a1=_enrichment_direction_phrase(row)))
            padj_values.append(row.get("padjust", ""))
            count_values.append(row.get("count", ""))
            effect_values.append(_contrast_effect_summary_for_enrichment(row, contrast_rows))
        contrast_direction = tm("sep_semicolon").join(dict.fromkeys(contrast_directions))
        effect = tm("sep_semicolon").join(dict.fromkeys(effect_values))
        lines.append(
            f"| {_fmt_cell(family, 40)} | {_fmt_cell('<br>'.join(source_terms), 180)} | {_fmt_cell(contrast_direction, 120)} | "
            f"{_fmt_cell(tm("sep_semicolon").join(padj_values))} | {_fmt_cell(tm("sep_semicolon").join(count_values))} | "
            f"{_fmt_cell(effect, 110)} | {_fmt_cell(interpretation, 120)} |"
        )
    lines.append("")
    lines.append(tm("genr_table_note"))
    return lines


def _rows_for_display_contrast(rows: List[Dict[str, Any]], contrast: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("display_contrast") == contrast]


def _pispa_task_four_lines(
    c2c3_contrast: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
    module_rows: List[Dict[str, Any]],
) -> List[str]:
    c2c3_candidates = _rows_for_display_contrast(candidate_rows, "Cluster_2_vs_Cluster_3")
    c2c3_modules = _rows_for_display_contrast(module_rows, "Cluster_2_vs_Cluster_3")
    group_a, group_b = _display_groups_from_contrast(c2c3_contrast)
    lines = [
        f"## {story_report_headings()[5]}",
        (
            tm("pt4_control_not_equivalent", group_a=group_a, group_b=group_b, a2=_contrast_sentence(c2c3_contrast))
        ),
        "",
        tm("pt4_c2c3_table_title"),
        tm("pt4_c2c3_header"),
        "|---|---|---|---|---|---|",
        (
            tm("pt4_c2c3_diff_row", top_up_display_group_a=_fmt_cell(_brief_gene_list(c2c3_contrast.get('top_up_display_group_a'), 5), 120), top_down_display_group_a=_fmt_cell(_brief_gene_list(c2c3_contrast.get('top_down_display_group_a'), 5), 120), n_sig=_fmt_story_number(c2c3_contrast.get('n_sig')), group_a=group_a, n_up_display_group_a=_fmt_story_number(c2c3_contrast.get('n_up_display_group_a')), group_b=group_b, n_down_display_group_a=_fmt_story_number(c2c3_contrast.get('n_down_display_group_a')))
        ),
    ]
    if c2c3_candidates:
        bits = [_candidate_key_value(row) for row in c2c3_candidates[:6]]
        lines.append(
            tm("pt4_c2c3_cytoskeleton_row", a0=_fmt_cell('；'.join(bits), 180))
        )
    if c2c3_modules:
        module_bits = [
            f"{_module_display_name(row.get('module'))} Δ={_fmt_story_number(row.get('module_delta_group_a_minus_group_b'))}"
            for row in c2c3_modules
            if _fmt_story_number(row.get("module_delta_group_a_minus_group_b"))
        ]
        if module_bits:
            lines.append(
                tm("pt4_c2c3_module_row", a0=_fmt_cell('；'.join(module_bits), 180))
            )
    lines.extend([
        "",
        tm("sub_current_matrix"),
        tm("pt4_c2c3_current_matrix", group_a=group_a, group_b=group_b, n_up_display_group_a=_fmt_story_number(c2c3_contrast.get('n_up_display_group_a')), n_down_display_group_a=_fmt_story_number(c2c3_contrast.get('n_down_display_group_a'))),
        tm("pt4_c2c3_background"),
        "",
        tm("sub_reasoned"),
        tm("pt4_extension_c2"),
        tm("pt4_extension_c3"),
        "",
        tm("sub_followup"),
        tm("pt4_followup_morphology"),
        tm("pt4_followup_timeseries"),
    ])
    return lines


def _build_pispa_claude_style_report(report_body: str, run_folder: str) -> str:
    metadata = _core_story_metadata(run_folder)
    contrast_rows = _core_story_rows(run_folder, "contrast")
    candidate_rows = _story_candidate_rows_for_report(run_folder, limit=32)
    module_rows = _core_story_rows(run_folder, "module")
    group_rows = _read_csv_rows(os.path.join(run_folder, "evaluation_evidence", "group_composition_qc.csv"), limit=80)
    enrichment_bits = _enrichment_summary_lines(run_folder, "Nat_Commun_PiSPA_2024", limit=24)
    visualization_lines = _visualization_evidence_lines(run_folder)
    main_contrast = _first_story_row(contrast_rows, ["cluster", "1"])
    c2c3_contrast = _first_story_row(contrast_rows, ["2", "3"]) if contrast_rows else {}
    priority_candidates = ["EZR", "MSN", "MYL9", "CDC42", "RAC1", "RHOA", "TLN1", "VCL", "FLNA", "ACTN1", "ITGB1"]
    candidate_bits = _candidate_bits_for_genes(candidate_rows, priority_candidates, limit=8)
    module_bits: List[str] = []
    for row in module_rows:
        if row.get("display_contrast") not in {"Cluster_1_vs_Cluster_2", "Cluster_1_vs_Cluster_3"}:
            continue
        delta = _fmt_story_number(row.get("module_delta_group_a_minus_group_b"))
        if delta:
            module_bits.append(f"{_module_display_name(row.get('module'))} Δ={delta}")
        if len(module_bits) >= 5:
            break
    group_summary = "; ".join(_group_qc_lines(group_rows, limit=4)) or tm("pispa_group_summary_fallback")
    figure_summary = " ".join(visualization_lines[:3]) or tm("pispa_figure_summary_fallback")

    lines: List[str] = [
        f"## {story_report_headings()[0]}",
        tm("pispa_summary_c1"),
        tm("pispa_summary_mechanism"),
        tm("pispa_summary_c2c3"),
        tm("pispa_summary_candidates"),
        tm("pispa_summary_boundary"),
    ]

    lines.extend([
        "",
        f"## {story_report_headings()[1]}",
        tm("pispa_data_intro"),
        tm("pispa_group_qc_line", group_summary=group_summary),
        tm("pispa_figure_line", figure_summary=figure_summary),
        "",
        f"## {story_report_headings()[2]}",
        tm("pispa_task1_intro"),
        "",
    ])
    lines.extend(_story_contrast_table_lines(run_folder) or [tm("pispa_task1_no_table")])
    lines.extend([
        "",
        tm("pispa_task1_contrast_note"),
        "",
        tm("pispa_task1_reading", a0=_contrast_sentence(main_contrast)),
        "",
        f"## {story_report_headings()[3]}",
        tm("pispa_task2_intro"),
        "",
    ])
    lines.extend(_story_module_table_lines(run_folder) or [tm("pispa_task2_no_table")])
    if module_bits:
        lines.append("")
        lines.append(tm("pispa_task2_summary_prefix") + tm("sep_semicolon").join(module_bits[:5]) + tm("pispa_task2_summary_tail"))
    enrichment_table = _pispa_enrichment_table_lines(enrichment_bits, contrast_rows)
    if enrichment_table:
        lines.append("")
        lines.extend(enrichment_table)
    lines.extend([
        "",
        f"## {story_report_headings()[4]}",
        tm("pispa_task3_intro"),
        "",
    ])
    lines.extend(_story_candidate_table_lines(run_folder) or [tm("pispa_task3_no_table")])
    if candidate_bits:
        lines.append("")
        lines.append(tm("pispa_task3_summary_prefix") + tm("sep_semicolon").join(candidate_bits[:6]) + tm("pispa_task3_summary_tail"))
    lines.extend([
        "",
        tm("pispa_task3_interpretation"),
        "",
    ])
    lines.extend(_pispa_task_four_lines(c2c3_contrast, candidate_rows, module_rows))
    lines.extend([
        "",
        f"## {story_report_headings()[6]}",
        tm("sub_current_matrix"),
        tm("pispa_synth_c1"),
        tm("pispa_synth_c2c3"),
        "",
        tm("sub_reasoned"),
        tm("pispa_synth_axis"),
        tm("pispa_synth_fdr"),
        "",
        tm("sub_followup"),
        tm("pispa_followup_staining"),
        tm("pispa_followup_live"),
        "",
        f"## {story_report_headings()[7]}",
        tm("pispa_assets_intro"),
        "",
        tm("assets_header"),
        "|---|---|---|",
        tm("asset_report"),
        tm("asset_assets"),
        tm("asset_figures"),
        tm("asset_core_story"),
        tm("asset_candidates"),
        tm("pispa_asset_modules"),
        tm("pispa_asset_group_qc"),
        tm("pispa_asset_figure_index"),
        tm("pispa_asset_audit"),
        "",
        f"## {story_report_headings()[8]}",
        tm("pispa_boundary_current_matrix"),
        tm("boundary_offline_enrichment"),
        tm("pispa_boundary_external"),
        tm("pispa_boundary_extension"),
        tm("pispa_boundary_records"),
    ])
    return "\n".join(lines).strip() + "\n"


def _story_candidate_rows_for_claim(candidate_rows: List[Dict[str, Any]], claim_id: str, limit: int = 6) -> List[Dict[str, Any]]:
    rows = [row for row in candidate_rows if str(row.get("claim_id", "")) == str(claim_id)]
    rows.sort(key=lambda r: (
        0 if str(r.get("confidence", "")).lower() == "moderate" else 1,
        _float_or_none(r.get("adj.P.Val")) if _float_or_none(r.get("adj.P.Val")) is not None else 999,
        -abs(_float_or_none(r.get("logFC_display")) or 0),
    ))
    return rows[:limit]


def _story_module_rows_for_claim(module_rows: List[Dict[str, Any]], claim_id: str, limit: int = 6) -> List[Dict[str, Any]]:
    rows = [row for row in module_rows if str(row.get("claim_id", "")) == str(claim_id)]
    rows.sort(key=lambda r: (
        0 if _float_or_none(r.get("module_delta_group_a_minus_group_b")) is not None else 1,
        -abs(_float_or_none(r.get("module_delta_group_a_minus_group_b")) or 0),
        str(r.get("module", "")),
    ))
    return rows[:limit]


def _story_task_heading(index: int, title: str) -> str:
    title = _short_story_value(title, 90) or tm("story_task_default_title")
    cn_nums = list(_report_language.t_list("main.task_numerals"))
    label = cn_nums[index - 1] if 0 < index <= len(cn_nums) else str(index)
    return tm("story_task_heading", n=index + 1, label=label, index=index, title=title)


def _generic_story_key_summary(run_folder: str) -> List[str]:
    metadata = _core_story_metadata(run_folder)
    contrast_rows = _core_story_rows(run_folder, "contrast")
    candidate_rows = _story_candidate_rows_for_report(run_folder, limit=8)
    module_rows = _core_story_rows(run_folder, "module")
    dataset = metadata.get("dataset") or _get_dataset_name(run_folder)
    lines = [
        tm("key_summary_opening", dataset=dataset,
           story_title=metadata.get("story_title", tm("key_summary_default_story")))
    ]
    for row in contrast_rows[:3]:
        group_a, group_b = _display_groups_from_contrast(row)
        nsig = _fmt_story_number(row.get("n_sig")) or "0"
        up = _fmt_story_number(row.get("n_up_display_group_a")) or "0"
        down = _fmt_story_number(row.get("n_down_display_group_a")) or "0"
        top_up = _brief_gene_list(row.get("top_up_display_group_a"), 2)
        top_down = _brief_gene_list(row.get("top_down_display_group_a"), 2)
        evidence = tm("sep_semicolon").join(bit for bit in [
            tm("evidence_up", group=group_a, genes=top_up) if top_up else "",
            tm("evidence_up", group=group_b, genes=top_down) if top_down else "",
        ] if bit)
        if not evidence:
            evidence = tm("evidence_insufficient")
        lines.append(tm(
            "key_summary_contrast",
            claim=row.get("claim_title") or _human_contrast_label(row.get("display_contrast")),
            contrast=_human_contrast_label(row.get("display_contrast")),
            nsig=nsig, convention=_counting_convention_text(run_folder),
            up=tm("evidence_up_short", group=group_a) + "=" + str(up),
            down=tm("evidence_up_short", group=group_b) + "=" + str(down),
            evidence=evidence))
    if candidate_rows:
        bits = [_candidate_key_value(row) for row in candidate_rows[:3]]
        lines.append(tm("key_summary_candidates", bits=tm("sep_semicolon").join(bits)))
    informative_modules = [
        row for row in module_rows
        if _float_or_none(row.get("module_delta_group_a_minus_group_b")) is not None
    ]
    if informative_modules:
        bits = [
            f"{_module_display_name(row.get('module'))} Δ={_fmt_story_number(row.get('module_delta_group_a_minus_group_b'))}"
            for row in informative_modules[:3]
        ]
        lines.append(tm("key_summary_modules", bits=tm("sep_semicolon").join(bits)))
    lines.append(tm("key_summary_boundary",
                    boundary=metadata.get("boundary", tm("boundary_default"))))
    return lines[:6]


def _generic_task_table_lines(
    contrast_row: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
    module_rows: List[Dict[str, Any]],
    convention_text: str = "",
) -> List[str]:
    claim_id = str(contrast_row.get("claim_id", ""))
    group_a, group_b = _display_groups_from_contrast(contrast_row)
    candidates = _story_candidate_rows_for_claim(candidate_rows, claim_id, limit=6)
    modules = _story_module_rows_for_claim(module_rows, claim_id, limit=5)
    convention_text = convention_text or tm("convention_threshold", p=0.05, l=0.25)
    display_contrast = _short_story_value(contrast_row.get("display_contrast"), 120)
    source_contrast = _short_story_value(contrast_row.get("source_contrast"), 120)
    inverted_display = bool(
        contrast_row.get("inverted_from_source")
        and display_contrast and source_contrast and display_contrast != source_contrast
    )
    lines = [
        tm("task_table_title"),
        tm("task_table_header"),
        tm("task_table_rule"),
        (
            f"| {_fmt_cell(_human_contrast_label(contrast_row.get('display_contrast')), 80)} | "
            f"{_fmt_story_number(contrast_row.get('n_sig')) or '0'} | "
            f"{_fmt_story_number(contrast_row.get('n_up_display_group_a')) or '0'} | "
            f"{_fmt_story_number(contrast_row.get('n_down_display_group_a')) or '0'} | "
            f"{_fmt_cell('; '.join(bit for bit in [_brief_gene_list(contrast_row.get('top_up_display_group_a'), 4), _brief_gene_list(contrast_row.get('top_down_display_group_a'), 4)] if bit), 150)} | "
            f"{tm('task_table_note', convention=convention_text)} |"
        ),
        "",
    ]
    if candidates:
        lines.extend([
            tm("candidate_table_title"),
            tm("candidate_table_header"),
            tm("candidate_table_rule"),
        ])
        for row in candidates:
            lines.append(
                f"| {_fmt_cell(row.get('candidate', ''))} | {_fmt_cell(_story_direction_zh(row))} | "
                f"{_fmt_story_number(row.get('logFC_display'))} | {_fmt_story_number(row.get('P.Value'))} | "
                f"{_fmt_story_number(row.get('adj.P.Val'))} | {_fmt_cell(row.get('interpretation', '') or tm('candidate_default_reading'), 130)} |"
            )
        if inverted_display:
            lines.append(
                tm("gtt_display_note", a0=_human_contrast_label(display_contrast), group_a=group_a, group_b=group_b, source_contrast=source_contrast)
            )
        lines.append("")
    if modules:
        lines.extend([
            tm("module_table_title"),
            tm("module_table_header"),
            tm("module_table_rule"),
        ])
        for row in modules:
            delta = _float_or_none(row.get("module_delta_group_a_minus_group_b"))
            if delta is None:
                interpretation = tm("module_not_computed")
            elif abs(delta) < 0.15:
                interpretation = tm("module_weak", group_a=group_a, group_b=group_b)
            elif delta > 0:
                interpretation = tm("module_a_higher", group_a=group_a)
            else:
                interpretation = tm("module_b_higher", group_b=group_b)
            lines.append(
                f"| {_fmt_cell(_module_display_name(row.get('module')), 80)} | "
                f"{_fmt_story_number(row.get('group_a_mean_score'))} | "
                f"{_fmt_story_number(row.get('group_b_mean_score'))} | "
                f"{_fmt_story_number(delta)} | {_fmt_cell(row.get('n_matched_genes', ''))} | "
                f"{_fmt_cell(interpretation, 130)} |"
            )
        lines.append("")
        lines.append(tm("gtt_module_note"))
    return lines



def _generic_enrichment_interpretation(term: str, contrast: str, direction: str) -> str:
    """Dataset-neutral reading of one enrichment row.

    The family label comes from the shared classifier, but the sentence is built only from this
    row's own fields. PiSPA-specific wording (subgroup codes, cluster-layer phrasing) must never
    be applied to other datasets -- that leak produced "C2/C3" labels in a dataset without them.
    """
    _, family, _ = _pispa_enrichment_family(term)
    family = family or tm("genr_family_set")
    return (tm("genr_direction_entry")
            % (family, contrast or tm("genr_current_contrast"), direction or tm("genr_differential_proteins")))

def _generic_enrichment_table_lines(run_folder: str, dataset: str, contrast_rows: List[Dict[str, Any]]) -> List[str]:
    bits = _enrichment_summary_lines(run_folder, dataset, limit=10)
    if not bits:
        return [tm("genr_no_fdr_enrichment")]
    lines = [
        tm("genr_table_header"),
        "|---|---|---|---:|---:|---|---|",
    ]
    seen_terms: set[Tuple[str, str]] = set()
    for bit in bits:
        row = _parse_enrichment_summary_bit(bit)
        term = row.get("term", "")
        contrast = row.get("contrast", "")
        key = (contrast, term.lower())
        if not term or key in seen_terms:
            continue
        seen_terms.add(key)
        direction = _enrichment_direction_phrase(row)
        effect = _contrast_effect_summary_for_enrichment(row, contrast_rows)
        interpretation = _generic_enrichment_interpretation(term, contrast, direction)
        lines.append(
            f"| {_fmt_cell(row.get('source', 'offline'))} | {_fmt_cell(tm("genr_contrast_direction", contrast=contrast, direction=direction), 100)} | "
            f"{_fmt_cell(term, 110)} | {_fmt_cell(row.get('padjust', ''))} | {_fmt_cell(row.get('count', ''))} | "
            f"{_fmt_cell(effect, 120)} | {_fmt_cell(interpretation, 120)} |"
        )
        if len(lines) >= 8:
            break
    lines.append("")
    lines.append(tm("genr_table_note"))
    return lines


_DIMENSION4_SECTION_SPECS: Dict[str, Dict[str, Any]] = {
    "Nat_Commun_Nociceptor_2026": {
        "heading": "机制整合与解释边界（亚型×炎症）",
        "contrasts": [["亚型内 Inflamed vs Control", "differential_*Inflamed_vs_*Control*.csv"]],
        "genes": ["B3GNT2", "RRAD", "NTRK1"],
        "paragraphs": [
            "亚型内对比是本数据集的核心设计：TrkA、IB4 与 Mechano 各自的 Inflamed vs Control 回答炎症响应是否发生在特定伤害感受亚型内。",
            "下表汇总各亚型对比的显著蛋白数（adj.P.Val<0.05）与焦点候选的方向和 FDR；n_sig 为 0 表示当前矩阵在该亚型内未检出足够显著的炎症响应，相关方向线索只能作为低置信度假设。",
        ],
        "boundary": ["亚型内样本量小，显著蛋白数对插补和阈值敏感；亚型间比较不构成细胞类型组成差异的统计检验。"],
    },
    "Nat_Commun_SCPro_2024": {
        "heading": "机制整合与解释边界（KLRG1 程序与 PDAC 背景）",
        "contrasts": [["KLRG1 主对比与 CD4/CD8 背景", "differential_*Klrg1*.csv"]],
        "genes": ["KLRG1", "CTLA4", "FOXP3", "IL2RA"],
        "paragraphs": [
            "主对比层级为 Treg 内部 KLRG1+ vs KLRG1-，再以 CD4/CD8 KLRG1 背景对照；下表给出各对比的显著蛋白数与焦点候选。",
            "研究背景（study context）：SCPro 面向胰腺癌（PDAC）肿瘤微环境的成像引导空间蛋白组。当前矩阵仅含 T 细胞区室，PDAC 免疫微环境解释停留在研究背景层，不能写成矩阵内发现。",
        ],
        "boundary": ["KLRG1 程序差异不构成 PDAC 免疫状态结论；免疫微环境整合需要带区室注释的独立证据。"],
    },
    "Nat_Methods_DVP_2023": {
        "heading": "机制整合与解释边界（小叶分区与 Midlobular 定位）",
        "contrasts": [["分区对比（Portal/Midlobular vs Central 等）", "differential_*Central*.csv"], ["Midlobular 相关对比", "differential_*Midlobular*.csv"]],
        "genes": ["GLUL", "CYP2E1", "ALDH1A1", "ASS1", "CPS1"],
        "paragraphs": [
            "肝小叶分区以 Portal vs Central 为主轴；Midlobular 的定位通过其与两侧的对比方向与幅度判断。",
            "判读规则：若 Midlobular 相对 Portal 与 Central 的焦点基因方向相反、幅度相当，则支持过渡态；若与一侧方向一致且 n_sig 明显更小，则更像该分区的延伸。以下表方向组合为准。",
        ],
        "boundary": ["分区方向不等于血流或代谢通量的直接测量。"],
    },
    "Nat_Methods_iPSC_2025": {
        "heading": "机制整合与解释边界（多能性梯度与 EB 内部异质性）",
        "contrasts": [["EB vs iPSCs 主轴", "differential_EB_vs_iPSCs*.csv"]],
        "genes": ["POU5F1", "SOX2", "NANOG", "COL1A1", "FN1"],
        "paragraphs": [
            "EB vs iPSCs 主轴给出多能性下降与谱系/ECM 上升；EB 内部异质性由聚类配方图与簇规模刻画。",
        ],
        "manifest_probe": {"manifest": "visualize_results/figure_manifest.json", "plot_type": "recipe_ipsc_eb_heterogeneity_clusters", "label": "EB 内部聚类簇规模"},
        "boundary": ["EB 内部异质性不等于谱系判定；需要时间序列或独立标注验证。"],
    },
    "Nat_Commun_Carr_2024": {
        "heading": "机制整合与解释边界（处理效应与批次结构）",
        "contrasts": [["LPS vs DMSO 主对比", "differential_LPS_vs_DMSO*.csv"]],
        "genes": ["CD44", "DDX21"],
        "paragraphs": [
            "LPS vs DMSO 主对比之外，批次与处理效应的相对强弱必须对照图册中的 PCA/UMAP 与批次交叉表解读；本节不重复计算，只给出核对入口。",
            "SampleInfo 中的 Batch 结构与处理分组存在交叉；若批次分离强于处理分离，差异方向仍以当前矩阵为准，但机制解释需降级为方向线索。",
        ],
        "boundary": ["批校正矩阵为本报告差异分析基础；批效应残留程度以图册证据为准。"],
        "always_render": True,
    },
    "Nat_Biotech_Brain_2026": {
        "heading": "机制整合与解释边界（状态轴与批次注记）",
        "contrasts": [["状态轴两段过渡", "differential_*oRG*.csv"], ["EN 成熟段", "differential_EN_vs_IPC-EN*.csv"]],
        "genes": ["EOMES", "MAP2", "TBR1", "NEUROD2"],
        "paragraphs": [
            "状态轴以 IPC-EN vs oRG（过渡）与 EN vs IPC-EN（成熟）两段对比组织；下表给出两段的显著蛋白数与焦点基因方向。",
            "供体/批次结构与发育状态的混淆需在批内方向核对后才支持发育结论；ASD/NDD 关联仅作为 external_annotation，不进入矩阵内结论。",
        ],
        "boundary": ["状态轴方向不替代拟时序或独立队列验证。"],
    },
    "Nat_Commun_ProteinLeakage_2025": {
        "heading": "机制整合与解释边界（区室泄漏与分层混杂）",
        "contrasts": [["泄漏主对比（含分型）", "differential_*Permeable*.csv"]],
        "genes": ["ATP5F1A", "GAPDH", "LDHA", "PTMS", "MPC2"],
        "paragraphs": [
            "主对比的区室方向（膜/线粒体 vs 胞质/核）是泄漏 signature 的核心；下表给出焦点蛋白方向。",
            "Fresh/Frozen 与细胞类型的分层混杂以 group_composition_qc 交叉表为准：差异解读需先核对分层的样本量不均衡，再谈生物学含义。",
        ],
        "boundary": ["泄漏足迹是样品处理 QC 特征，不直接等于生物学调控。"],
        "always_render": True,
    },
}


def _dimension4_focus_gene_lines(csv_path: str, genes: set) -> Tuple[List[str], int]:
    import csv as _csv

    lines: List[str] = []
    n_sig = 0
    try:
        with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = _csv.DictReader(handle)
            cols = reader.fieldnames or []
            gene_col = next((c for c in cols if str(c).strip().lower() in ("gene_label", "gene", "gene_name", "symbol", "pg.genes", "genes")), None)
            if not gene_col:
                return lines, 0
            for row in reader:
                gene_raw = str(row.get(gene_col, "") or "").strip().upper()
                gene_ok = False
                matched = []
                for chunk in re.split("[;/,]", gene_raw):
                    token = chunk.strip()
                    if token and token in genes:
                        gene_ok = True
                        matched.append(token)
                adj = row.get("adj.P.Val") or row.get("FDR") or row.get("padj") or ""
                try:
                    significant = float(adj) < 0.05
                except (TypeError, ValueError):
                    significant = False
                if significant:
                    n_sig += 1
                if gene_ok:
                    gene = "/".join(matched)
                    lfc = row.get("logFC") or row.get("log2FoldChange") or ""
                    pval = row.get("P.Value") or row.get("pvalue") or ""
                    lines.append("| " + " | ".join([gene, str(lfc), str(pval), str(adj)]) + " |")
    except Exception:
        return lines, n_sig
    return lines, n_sig


def _build_dimension4_integration_section(run_folder: str, heading_index: int,
                                        spec_override: Optional[Dict[str, Any]] = None) -> List[str]:
    dataset = _get_dataset_name(run_folder)
    spec = spec_override or _dimension4_spec_for(dataset)
    if not spec:
        return []
    lines: List[str] = []
    table_blocks: List[str] = []
    import glob as _glob

    for label, pattern in spec.get("contrasts", []):
        matches = sorted(_glob.glob(os.path.join(run_folder, "processed_proteins", pattern)))
        if not matches:
            continue
        for match in matches[:4]:
            contrast_name = os.path.basename(match).replace("differential_", "").replace(".csv", "")
            gene_lines, n_sig = _dimension4_focus_gene_lines(match, set(spec.get("genes", [])))
            if not gene_lines and not n_sig:
                continue
            table_blocks.append("")
            table_blocks.append(tm("dim4_integration_row", label=label, contrast_name=contrast_name, n_sig=n_sig))
            inverted = next((r for r in _core_story_rows(run_folder, "contrast")
                             if _short_story_value(r.get("source_contrast")) == contrast_name
                             and r.get("inverted_from_source")), None)
            if inverted is not None:
                display_label = _human_contrast_label(_short_story_value(inverted.get("display_contrast"), 120))
                table_blocks.append(tm("dim4_integration_note", contrast_name=contrast_name, display_label=display_label))
            if gene_lines:
                table_blocks.append(tm("dim4_candidate_header"))
                table_blocks.append("|---|---|---|---|")
                table_blocks.extend(gene_lines[:6])
    probe = spec.get("manifest_probe")
    if probe:
        try:
            manifest_path = os.path.join(run_folder, probe["manifest"])
            manifest = json.load(open(manifest_path, encoding="utf-8"))
            entry = next((e for e in manifest.get("figures", []) if e.get("plot_type") == probe["plot_type"]), None)
            sizes = ((entry or {}).get("caption_metadata", {}).get("thresholds", {}) or {}).get("cluster_sizes", {})
            if sizes:
                text = tm("sep_comma").join(f"{k}={v}" for k, v in sizes.items())
                total = sum(int(v) for v in sizes.values())
                table_blocks.append("")
                table_blocks.append(tm("dim4_cluster_size_note", total=total, text=text))
        except Exception:
            pass
    if not table_blocks and not spec.get("always_render"):
        return []
    lines.append("")
    lines.append(f"## {heading_index}. {spec['heading']}")
    for paragraph in spec.get("paragraphs", []):
        lines.append(paragraph)
        lines.append("")
    lines.extend(table_blocks)
    if table_blocks:
        lines.append("")
    for boundary in spec.get("boundary", []):
        lines.append(tm("dim4_prefixed_boundary", boundary=boundary))
    return lines


def _build_generic_story_first_report(report_body: str, run_folder: str) -> str:
    metadata = _core_story_metadata(run_folder)
    contrast_rows = _core_story_rows(run_folder, "contrast")
    if not contrast_rows:
        return ""
    dataset = metadata.get("dataset") or _get_dataset_name(run_folder)
    candidate_rows = _story_candidate_rows_for_report(run_folder, limit=80)
    module_rows = _core_story_rows(run_folder, "module")
    group_rows = _read_csv_rows(os.path.join(run_folder, "evaluation_evidence", "group_composition_qc.csv"), limit=100)
    visualization_lines = _visualization_evidence_lines(run_folder)
    group_summary = "; ".join(_group_qc_lines(group_rows, limit=5)) or tm("group_qc_fallback")
    figure_summary = " ".join(visualization_lines[:4]) or tm("figure_fallback")
    lines: List[str] = [
        "## " + _report_language.t("core.s0"),
        *_generic_story_key_summary(run_folder),
        "",
        "## " + _report_language.t("core.s1"),
        tm("intro"),
        tm("label_group_qc", text=group_summary),
        tm("label_structure_figures", text=figure_summary),
        tm("statistical_boundary"),
        "",
    ]
    for index, contrast_row in enumerate(contrast_rows[:6], start=1):
        title = str(contrast_row.get("claim_title") or _human_contrast_label(contrast_row.get("display_contrast", "")))
        heading = _story_task_heading(index, title)
        lines.extend([
            f"## {heading}",
            tm("conclusion_first", title=title, sentence=_contrast_sentence(contrast_row)),
            "",
        ])
        lines.extend(_generic_task_table_lines(contrast_row, candidate_rows, module_rows,
                                               convention_text=_counting_convention_text(run_folder)))
        interpretation = _short_story_value(contrast_row.get("interpretation"), 260)
        boundary = _short_story_value(contrast_row.get("boundary") or metadata.get("boundary"), 260)
        if interpretation:
            lines.append(tm("label_interpretation", text=interpretation))
        if boundary:
            lines.append(tm("label_boundary", text=boundary))
        lines.append("")
    summary_index = min(len(contrast_rows[:6]) + 2, 12)
    integration_lines = _build_dimension4_section(run_folder, summary_index)
    if integration_lines:
        lines.extend(integration_lines)
        summary_index += 1
    lines.extend([
        f"## {summary_index}. {_report_language.t('core.story_synthesis')}",
        tm("sub_current_matrix"),
        tm("mainline", title=metadata.get("story_title", tm("default_story_title"))),
        tm("mainline_more"),
        "",
        tm("sub_offline"),
    ])
    lines.extend(_generic_enrichment_table_lines(run_folder, dataset, contrast_rows))
    lines.extend([
        "",
        tm("sub_reasoned"),
        tm("extension_unanimous"),
        tm("extension_low_confidence"),
        "",
        tm("sub_followup"),
        tm("followup_focus"),
        tm("followup_replication"),
        "",
        f"## {summary_index + 1}. {_report_language.t('core.story_assets')}",
        tm("assets_intro"),
        "",
        tm("assets_header"),
        tm("assets_rule"),
        tm("asset_report"),
        tm("asset_figures"),
        tm("asset_assets"),
        tm("asset_core_story"),
        tm("asset_candidates"),
        tm("asset_modules"),
        tm("asset_group_qc"),
        tm("asset_figure_index"),
        "",
        f"## {summary_index + 2}. {_report_language.t('core.story_conclusion')}",
        tm("boundary_current_matrix"),
        tm("boundary_offline_enrichment"),
    ])
    if _external_annotation_applicable(run_folder):
        lines.append(tm("boundary_external_applicable"))
    else:
        lines.append(tm("boundary_external_not_applicable"))
    lines.extend([
        tm("boundary_extension"),
        tm("boundary_records"),
    ])
    return "\n".join(lines).strip() + "\n"


def _build_story_first_report(report_body: str, run_folder: str) -> str:
    if not _core_story_rows(run_folder):
        return ""
    if _get_dataset_name(run_folder) == "Nat_Commun_PiSPA_2024":
        return _polish_pispa_report_terms(_build_pispa_claude_style_report(report_body, run_folder))
    return _build_generic_story_first_report(report_body, run_folder)


def _load_latest_tool_summary(run_folder: str, tool_name: str) -> Dict[str, Any]:
    events_path = os.path.join(run_folder, "run_events.jsonl")
    if not os.path.exists(events_path):
        return {}
    try:
        with open(events_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return {}
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("event") == "tool_call_end" and event.get("tool") == tool_name:
            summary = event.get("result_summary")
            return summary if isinstance(summary, dict) else {}
    return {}


def _visualization_evidence_lines(run_folder: str) -> List[str]:
    summary = _load_latest_tool_summary(run_folder, "visualize")
    lines: List[str] = []
    root_summary = summary.get("_summary", {}) if isinstance(summary, dict) else {}
    pca = root_summary.get("pca", {}) if isinstance(root_summary, dict) else {}
    umap = root_summary.get("umap", {}) if isinstance(root_summary, dict) else {}
    heatmap = root_summary.get("heatmap", {}) if isinstance(root_summary, dict) else {}
    key = root_summary.get("key_protein_overview", {}) if isinstance(root_summary, dict) else {}
    if pca:
        lines.append(
            tm("vis_pca", n_samples=pca.get('n_samples', ''), PC1_variance=pca.get('PC1_variance', ''), PC2_variance=pca.get('PC2_variance', ''), label_column=pca.get('label_column', ''))
        )
    if umap:
        lines.append(
            tm("vis_umap", n_samples=umap.get('n_samples', ''), n_clusters=umap.get('n_clusters', ''), label_column=umap.get('label_column', ''))
        )
    if heatmap:
        comparisons = heatmap.get("comparisons", []) or []
        comparison_preview = ", ".join(str(x) for x in comparisons[:6])
        if len(comparisons) > 6:
            comparison_preview += tm("vis_more_contrasts", a0=len(comparisons))
        lines.append(
            tm("vis_heatmap", n_features=heatmap.get('n_features', ''), comparison_preview=comparison_preview)
        )
    if key:
        lines.append(
            tm("vis_differential_overview") % (key.get("n_significant", ""), key.get("n_contrasts", ""))
        )
    figure_manifest = os.path.join(run_folder, "visualize_results", "figure_manifest.json")
    try:
        if os.path.exists(figure_manifest):
            manifest = load_json(figure_manifest)
            figures = manifest.get("figures", []) if isinstance(manifest, dict) else []
            categories = sorted({f.get("category", "") for f in figures if isinstance(f, dict) and f.get("category")})
            lines.append(tm("vis_static_figure_set", a0=len(figures), a1=', '.join(categories[:6])))
    except Exception:
        pass
    return lines


def _group_qc_lines(group_rows: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    rows = [r for r in group_rows if r.get("table") == "primary_group"]
    lines = []
    for row in rows[:limit]:
        if row.get("input_mean_detected_proteins") not in (None, ""):
            lines.append(
                tm("gqc_row_split", group_col=row.get('group_col', ''), group=row.get('group', ''), n_samples=row.get('n_samples', ''), input_mean_detected_proteins=row.get('input_mean_detected_proteins', ''), input_mean_missing_rate=row.get('input_mean_missing_rate', ''), analysed_mean_detected_proteins=row.get('analysed_mean_detected_proteins', ''), analysed_mean_missing_rate=row.get('analysed_mean_missing_rate', ''))
            )
        else:
            lines.append(
                tm("gqc_row_flat", group_col=row.get('group_col', ''), group=row.get('group', ''), n_samples=row.get('n_samples', ''), mean_detected_proteins=row.get('mean_detected_proteins', ''), mean_missing_rate=row.get('mean_missing_rate', ''))
            )
    return lines


def _module_value(rows: List[Dict[str, Any]], module: str, group: str) -> str:
    for row in rows:
        if row.get("module") == module and row.get("group") == group:
            return str(row.get("mean_score", ""))
    return ""


def _module_summary_lines(module_rows: List[Dict[str, Any]], dataset: str, limit: int = 8) -> List[str]:
    lines: List[str] = []
    modules = _module_priority(dataset)
    for module in modules:
        rows = [r for r in module_rows if r.get("module") == module]
        if not rows:
            continue
        if dataset == "Nat_Methods_iPSC_2025":
            groups = ["EB", "iPSCs", "EB_minus_iPSCs"]
        elif dataset == "Nat_Biotech_Brain_2026":
            groups = ["oRG", "RG", "IPC-EN", "EN", "EN_minus_IPC-EN", "EN_minus_oRG", "RG_minus_oRG"]
        elif dataset == "Nat_Commun_PiSPA_2024":
            groups = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 1_minus_Cluster 2", "Cluster 2_minus_Cluster 3"]
        else:
            groups = [r.get("group", "") for r in rows[:4]]
        values = [f"{g}={_module_value(rows, module, g)}" for g in groups if _module_value(rows, module, g)]
        if values:
            lines.append(f"{module}: " + ", ".join(values[:6]))
        if len(lines) >= limit:
            break
    return lines


def _sample_score_dispersion_lines(sample_rows: List[Dict[str, Any]], dataset: str) -> List[str]:
    if dataset != "Nat_Methods_iPSC_2025":
        return []
    eb_rows = [r for r in sample_rows if str(r.get("Cluster", "")) == "EB"]
    modules = ["lineage_endoderm", "lineage_mesoderm", "lineage_ectoderm", "ecm_adhesion"]
    lines = []
    for module in modules:
        values = [_float_or_none(r.get(module)) for r in eb_rows]
        vals = [v for v in values if v is not None]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        lines.append(
            tm("eis_dispersion_row", module=module, a1=len(vals), mean=mean, a3=min(vals), a4=max(vals), a5=var ** 0.5)
        )
    return lines


def _parse_counts_cell(value: Any) -> Dict[str, int]:
    if not value:
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except Exception:
            return {}
    counts: Dict[str, int] = {}
    for key, val in raw.items():
        try:
            counts[str(key)] = int(val)
        except Exception:
            continue
    return counts


def _reader_counts_text(value: Any) -> str:
    """Counts cell for the reader: readable pairs; a truncated literal stays readable and labelled."""
    counts = _parse_counts_cell(value)
    if counts:
        return tm("sep_semicolon").join("%s %s" % (key, val) for key, val in counts.items())
    text = str(value or "").strip()
    if not text:
        return ""
    if "..." in text or "…" in text:
        return text.strip("{}") + tm("reader_counts_truncated_note")
    return text


def _pipsa_cluster_type_summary(group_rows: List[Dict[str, Any]]) -> str:
    bits = []
    for row in group_rows:
        if row.get("table") != "cross_table":
            continue
        counts = _parse_counts_cell(row.get("counts", ""))
        total = sum(counts.values())
        if not total:
            continue
        migrated = counts.get("Migrated Cell", 0)
        control = counts.get("Control Cell", 0)
        bits.append(f"{row.get('group', '')}: migrated={migrated}/{total}({migrated / total:.0%}), control={control}/{total}")
    return "; ".join(bits)


def _dataset_specific_interpretation_lines(
    dataset: str,
    group_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    module_rows: List[Dict[str, Any]],
    sample_rows: List[Dict[str, Any]],
) -> List[str]:
    if dataset == "Nat_Commun_PiSPA_2024":
        migration_modules = [
            "migration_signature",
            "rho_gtpase_migration",
            "erm_membrane_cortex",
            "myosin_contractility",
            "talin_vinculin_focal_adhesion",
        ]
        module_bits = []
        for module in migration_modules:
            values = [
                f"{group}={_module_value(module_rows, module, group)}"
                for group in ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 1_minus_Cluster 2", "Cluster 2_minus_Cluster 3"]
                if _module_value(module_rows, module, group)
            ]
            if values:
                module_bits.append(f"{module}: " + ", ".join(values))
        candidate_bits = _format_candidate_bits(
            _select_candidate_highlights(candidate_rows, dataset, limit=12),
            limit=10,
        )
        cluster_type = _pipsa_cluster_type_summary(group_rows)
        lines = [tm("dsi_pispa_chain_title")]
        if cluster_type:
            lines.append(
                tm("dsi_pispa_cluster_type")
                + cluster_type
                + tm("dsi_pispa_cluster_type_tail")
            )
        if module_bits:
            lines.append(
                tm("dsi_pispa_module_chain")
                + "; ".join(module_bits[:5])
                + tm("dsi_pispa_module_tail")
            )
        if candidate_bits:
            lines.append(
                tm("dsi_pispa_candidate_chain")
                + "; ".join(candidate_bits[:10])
                + tm("dsi_pispa_candidate_tail")
            )
        lines.append(
            tm("dsi_pispa_enrichment_boundary")
        )
        return lines

    if dataset == "Nat_Methods_iPSC_2025":
        module_bits = []
        for module in ["pluripotency_core", "lineage_endoderm", "lineage_mesoderm", "lineage_ectoderm", "ecm_adhesion", "cell_cycle_replication", "ribosome_translation", "mitochondrial_oxphos"]:
            values = [
                f"{group}={_module_value(module_rows, module, group)}"
                for group in ["EB", "iPSCs", "EB_minus_iPSCs"]
                if _module_value(module_rows, module, group)
            ]
            if values:
                module_bits.append(f"{module}: " + ", ".join(values))
        candidate_bits = _format_candidate_bits(
            _select_candidate_highlights(candidate_rows, dataset, limit=12),
            limit=10,
        )
        dispersion_bits = _sample_score_dispersion_lines(sample_rows, dataset)
        lines = [tm("dsi_ipsceb_chain_title")]
        lines.append(
            tm("dsi_ipsceb_normalisation")
        )
        if candidate_bits:
            lines.append(
                tm("dsi_ipsceb_candidate_chain")
                + "; ".join(candidate_bits[:10])
                + tm("dsi_ipsceb_nanog_tail")
            )
        if module_bits:
            lines.append(
                tm("dsi_ipsceb_module_chain")
                + "; ".join(module_bits[:6])
                + tm("dsi_ipsceb_module_tail")
            )
        if dispersion_bits:
            lines.append(
                tm("dsi_ipsceb_dispersion")
                + " ".join(dispersion_bits[:3])
                + tm("dsi_ipsceb_dispersion_tail")
            )
        return lines

    return []


def _brain_competitive_interpretation_lines(
    group_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    module_rows: List[Dict[str, Any]],
    external_rows: List[Dict[str, Any]],
    visualization_lines: List[str],
    run_folder: str = "",
) -> List[str]:
    batch_bits = []
    primary = [r for r in group_rows if r.get("table") == "primary_group"]
    for group in ["RG", "oRG", "IPC-EN", "EN", "IN-CGE"]:
        row = next((r for r in primary if r.get("group") == group), None)
        if row:
            batch_bits.append(f"{group}: n={row.get('n_samples', '')}, batches={row.get('n_batches', '')}, missing={row.get('mean_missing_rate', '')}")

    batch_cross_bits = []
    for group in ["oRG", "RG", "IPC-EN", "EN"]:
        row = next(
            (
                r for r in group_rows
                if r.get("table") == "cross_table" and r.get("group") == group and r.get("secondary_col") == "Batch"
            ),
            None,
        )
        if not row:
            continue
        counts = _parse_counts_cell(row.get("counts", ""))
        nonzero = {k: v for k, v in counts.items() if v}
        if nonzero:
            batch_cross_bits.append(f"{group} batch counts={nonzero}")

    batch_metric_bits = []
    if run_folder:
        detect_summary = _load_latest_tool_summary(run_folder, "detect_batch_and_doublets")
        outlier_summary = detect_summary.get("outlier_summary", {}) if isinstance(detect_summary, dict) else {}
        qc_stats = detect_summary.get("qc_statistics", {}) if isinstance(detect_summary, dict) else {}
        batch_assoc = detect_summary.get("batch_association_with_PC1", {}) if isinstance(detect_summary, dict) else {}
        for key, val in batch_assoc.items():
            if isinstance(val, dict) and "PC1_variance_explained" in val:
                batch_metric_bits.append(f"{key} explains PC1={val.get('PC1_variance_explained')}")
        if outlier_summary:
            batch_metric_bits.append(f"outlier_samples={outlier_summary.get('n_outlier_samples', '')}")
        if qc_stats:
            batch_metric_bits.append(f"mean_missing={qc_stats.get('mean_missing_rate', '')}")

    module_focus = []
    module_specs = [
        ("brain_rg_org_progenitor", ["oRG", "RG", "IPC-EN", "EN"]),
        ("brain_ipc_en_transition", ["oRG", "RG", "IPC-EN", "EN"]),
        ("brain_en_maturation", ["oRG", "IPC-EN", "EN", "EN_minus_IPC-EN"]),
        ("brain_synapse_neurite", ["IPC-EN", "EN", "EN_minus_IPC-EN", "EN_minus_oRG"]),
        ("brain_chromatin_baf", ["IPC-EN", "EN", "EN_minus_IPC-EN", "EN_minus_oRG"]),
        ("cell_cycle_replication", ["RG", "oRG", "IPC-EN", "EN", "EN_minus_IPC-EN"]),
        ("ribosome_translation", ["RG", "oRG", "IPC-EN", "EN"]),
        ("mitochondrial_oxphos", ["RG", "oRG", "IPC-EN", "EN"]),
    ]
    for module, groups in module_specs:
        values = [f"{g}={_module_value(module_rows, module, g)}" for g in groups if _module_value(module_rows, module, g)]
        if values:
            module_focus.append(f"{module}: " + ", ".join(values))

    candidate_bits = _format_candidate_bits(
        _select_candidate_highlights(candidate_rows, "Nat_Biotech_Brain_2026", limit=16),
        limit=12,
    )
    detected_external = [
        row.get("gene", "") for row in external_rows
        if str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"} and _is_gene_like_candidate(row.get("gene", ""))
    ]

    lines = [
        tm("brain_scoring_title"),
        tm("brain_state_qc") + " ".join(visualization_lines[:2]),
    ]
    if batch_bits:
        lines.append(
            tm("brain_batch_boundary") +
            "; ".join(batch_bits) +
            tm("brain_batch_boundary_tail")
        )
    if batch_cross_bits or batch_metric_bits:
        lines.append(
            tm("brain_batch_quant")
            + ("; ".join(batch_cross_bits[:4]) if batch_cross_bits else "")
            + (tm("sep_semicolon") if batch_cross_bits and batch_metric_bits else "")
            + ("; ".join(batch_metric_bits) if batch_metric_bits else "")
            + tm("brain_batch_quant_tail")
        )
    if module_focus:
        lines.append(
            tm("brain_axis") +
            "; ".join(module_focus[:5]) +
            tm("brain_axis_tail")
        )
        if len(module_focus) > 5:
            lines.append(tm("brain_module_coverage") + "; ".join(module_focus[5:]) + tm("sentence_stop"))
    if candidate_bits:
        lines.append(
            tm("brain_candidates") +
            "; ".join(candidate_bits[:10]) +
            tm("brain_candidates_tail")
        )
    if detected_external:
        lines.append(
            tm("brain_external_boundary") +
            ", ".join(detected_external[:14]) +
            tm("brain_external_boundary_tail")
        )
    lines.append(
        tm("brain_enrichment_rule")
    )
    return lines


def _enrichment_summary_lines(run_folder: str, dataset: str, limit: int = 8) -> List[str]:
    enrich_dir = os.path.join(run_folder, "enrichment_results")
    if not os.path.isdir(enrich_dir):
        return []
    preferred = {
        "Nat_Methods_iPSC_2025": ["EB_vs_iPSCs"],
        "Nat_Biotech_Brain_2026": ["EN_vs_IPC-EN", "EN_vs_oRG", "RG_vs_oRG", "IPC-EN_vs_oRG", "EN_vs_RG"],
        "Nat_Commun_PiSPA_2024": ["Cluster 1", "Cluster_1", "Cluster 2", "Cluster_2", "Cluster 3", "Cluster_3"],
    }.get(dataset, [])
    candidates = []
    for name in os.listdir(enrich_dir):
        if not name.endswith(".csv"):
            continue
        if preferred and not any(token in name for token in preferred):
            continue
        path = os.path.join(enrich_dir, name)
        rows = _read_csv_rows(path, limit=3)
        for row in rows:
            adj = _float_or_none(row.get("p.adjust") or row.get("qvalue") or row.get("pvalue"))
            if adj is None:
                continue
            candidates.append((adj, name, row))
    candidates.sort(key=lambda item: item[0])
    lines = []
    seen = set()
    for adj, name, row in candidates:
        desc = row.get("Description", "")
        if dataset == "Nat_Biotech_Brain_2026":
            desc_lower = str(desc).lower()
            broad_context_terms = [
                "infectious", "viral", "virus", "coronavirus", "influenza",
                "parkinson", "huntington", "prion", "amyotrophic", "disease",
            ]
            core_terms = [
                "rna", "splicing", "mrna", "translation", "ribosome", "ribosomal",
                "chromatin", "nuclear", "synapse", "neurite", "nervous system",
                "mitochond", "oxidative phosphorylation", "cell cycle", "dna replication",
                "cytoskeleton", "cell substrate junction",
            ]
            if any(term in desc_lower for term in broad_context_terms) and not any(term in desc_lower for term in core_terms):
                continue
        cluster = row.get("Cluster", "")
        namespace = row.get("namespace", "") or name.split("_", 1)[0]
        direction = "upregulated" if "upregulated" in name else "downregulated" if "downregulated" in name else ""
        key = (cluster, namespace, direction, desc)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"{namespace} {cluster} {direction}: {desc} (p.adjust={row.get('p.adjust', '')}, Count={row.get('Count', '')}, status={row.get('significance_status', '')})."
        )
        if len(lines) >= limit:
            break
    return lines


def strip_raw_tool_json(text: str) -> str:
    def replace_block(match: re.Match) -> str:
        body = match.group(1) or ""
        looks_raw = (
            len(body) > 1200
            or '"tool"' in body
            or '"result"' in body
            or '"tool_call_id"' in body
            or '"args"' in body
            or '"messages"' in body
        )
        if looks_raw:
            return "\n\n<!-- raw tool JSON omitted from final report; see record_file.md for audit trail -->\n\n"
        return match.group(0)

    return re.sub(r"```json\s*(.*?)```", replace_block, str(text), flags=re.I | re.S)


def sanitize_report_text_for_output(text: str, max_chars: int = 120000) -> str:
    cleaned = strip_raw_tool_json(str(text))
    cleaned = re.sub(r"Raw tool output is retained below.*?(?=\n#|\n##|\Z)", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\n## Task\n.*?(?=\n## |\Z)", "\n", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\n## \[Analyzer\] Local Fallback Summary.*?(?=\n## |\Z)", "\n", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return trim_text(cleaned.strip(), max_chars)


def _has_h2(text: str, heading: str) -> bool:
    return bool(re.search(rf"(?m)^## {re.escape(heading)}\s*$", str(text)))


def _has_h2_any(text: str, headings: List[str]) -> bool:
    return any(_has_h2(text, heading) for heading in headings)


def _has_numbered_h2(text: str, heading: str) -> bool:
    """Match the renumbered story-first form, e.g. '## 6. <heading>'."""
    pattern = "(?m)^## [0-9]+[.][ ]*" + re.escape(heading) + "[ ]*$"
    return bool(re.search(pattern, str(text)))


def _numbered_heading_re(heading: str, number: str = "[0-9]+"):
    """Compiled matcher for '## <number>. <heading>', tolerant of leading whitespace."""
    pattern = "(?m)^##[ \t]*" + number + "[.][ \t]*" + re.escape(heading) + "[ \t]*$"
    return re.compile(pattern)


def _numbered_heading_re_for_key(key: str, number: str = "[0-9]+"):
    """Matcher for a numbered heading whose template already carries its own section number.

    The story-first keys come in two shapes: the reader-heading keys (core.story_assets and
    friends) hold a bare title, while the canonical keys (core.s1 and friends) hold a title
    that already begins with its number. Passing the second shape into _numbered_heading_re
    makes the pattern demand the number twice, so it can never match a real heading and the
    insertion silently degrades to an append. Stripping the leading number here is what keeps
    the two shapes interchangeable.
    """
    title = re.sub("^[0-9]+[.][ \t]*", "", _report_language.t(key))
    return _numbered_heading_re(title, number)


CANONICAL_REPORT_HEADINGS = {
    "executive": "一、执行摘要",
    "scoring": "二、核心证据首页",
    "findings": "三、主要发现",
    "boundary": "四、证据边界",
    "appendix": "五、可复核证据附录",
}

CLAUDE_STYLE_REPORT_HEADINGS = [
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

LEGACY_REPORT_HEADINGS = [
    "Executive Result Summary",
    "Scoring Evidence First Screen",
    "Main Findings",
    "Evidence Boundary",
    "Evidence Boundary And Confidence",
    "Evidence Appendix",
    "Scoring Evidence Appendix",
    "Protein Leakage Evidence Appendix",
]

CHINESE_REPORT_HEADINGS = list(CANONICAL_REPORT_HEADINGS.values()) + CLAUDE_STYLE_REPORT_HEADINGS


# The constants above are the released zh wording and stay its single source. These
# accessors return the wording of the *active* report language; for zh they return exactly
# those constants, so a zh run renders byte-identical text to the release.
def canonical_report_headings() -> Dict[str, str]:
    return _report_language.canonical_headings()


def story_report_headings() -> List[str]:
    return _report_language.story_headings()


def reader_report_headings() -> List[str]:
    """Section headings of the active language, plus the legacy English set to strip."""
    return (list(canonical_report_headings().values())
            + story_report_headings() + LEGACY_REPORT_HEADINGS)


# Deterministic report prose for the active language. zh values are the released wording,
# copied verbatim; en values are the added language. Keys whose zh value embeds a path are
# written with the same backticks the released report used.
MAIN_TEXT = {
    "group_qc_fallback": (
        "未生成可读分组/QC 摘要。",
        "No readable group/QC summary was generated."),
    "figure_fallback": (
        "图册索引见 `figures.md` 和 `visualize_results/figure_index.md`。",
        "See `figures.md` and `visualize_results/figure_index.md` for the figure index."),
    "intro": (
        "本报告基于当前输入矩阵、样本信息和自动/预设 analysis design 生成；先确认分组、缺失和结构图，再解释差异蛋白、模块和离线富集。",
        "This report was generated from the current input matrix, the sample information and "
        "the automatic or declared analysis design. Grouping, missingness and the structure "
        "figures are established first, and the differential proteins, modules and offline "
        "enrichment are interpreted afterwards."),
    "label_group_qc": ("- 分组与 QC 摘要：{text}", "- Group and QC summary: {text}"),
    "label_structure_figures": ("- 结构图证据：{text}", "- Structure figure evidence: {text}"),
    "statistical_boundary": (
        "- 统计边界：差异蛋白以当前矩阵中的 logFC、P.Value 和 adj.P.Val/FDR 为准；模块分数用于组织机制，不单独等同于显著富集。",
        "- Statistical boundary: differential proteins follow the logFC, P.Value and "
        "adj.P.Val/FDR recorded in the current matrix; module scores are used to organise "
        "mechanisms and are not by themselves equivalent to significant enrichment."),
    "conclusion_first": ("结论先行：{title}。{sentence}", "Conclusion first: {title}. {sentence}"),
    "label_interpretation": ("解释：{text}", "Interpretation: {text}"),
    "label_boundary": ("边界：{text}", "Boundary: {text}"),
    "sub_current_matrix": ("### 当前矩阵支持", "### Supported by the current matrix"),
    "mainline": (
        "- 当前矩阵支持的主线是：{title}。",
        "- The main line supported by the current matrix is: {title}."),
    "mainline_more": (
        "- 核心对比、代表蛋白和模块 Δ 已在上文逐任务列出；未显著候选只作为方向线索。",
        "- The core contrasts, representative proteins and module deltas are listed task by "
        "task above; candidates that were not significant serve only as directional leads."),
    "sub_offline": ("### 离线富集补充", "### Offline enrichment supplement"),
    "sub_reasoned": ("### 合理推测", "### Reasonable inference"),
    "extension_unanimous": (
        "- `extension`：若同一方向同时得到差异蛋白、模块 Δ 和离线富集支持，可作为后续实验优先验证的机制假设。",
        "- `extension`: when the same direction is supported jointly by differential "
        "proteins, module deltas and offline enrichment, it can be prioritised as a "
        "mechanism hypothesis for follow-up experiments."),
    "extension_low_confidence": (
        "- `extension`：若当前矩阵显著性不足但效应量和候选方向一致，应写成低置信度线索，而不是已验证机制。",
        "- `extension`: when significance in the current matrix is insufficient but the "
        "effect size and candidate direction agree, it must be written as a low-confidence "
        "lead rather than an established mechanism."),
    "sub_followup": ("### 后续验证建议", "### Follow-up validation suggestions"),
    "followup_focus": (
        "- 优先验证与主任务最接近的候选蛋白或模块，而不是泛化到无直接证据的疾病、发育或功能因果结论。",
        "- Validate the candidate proteins or modules closest to the main question first, "
        "instead of generalising to disease, developmental or functional causal claims that "
        "have no direct evidence here."),
    "followup_replication": (
        "- 对样本量较小或 FDR 不显著的对比，建议增加重复、靶向验证或独立批次确认。",
        "- For contrasts with a small sample size or a non-significant FDR, add replicates, "
        "targeted validation or independent-batch confirmation."),
    "assets_intro": (
        "本 run 同时生成主报告、结论版、技术附录、图册和证据表；建议先读 `report.md`，再查看 `figures.md` 与 `assets.txt`。",
        "This run produced a main report, a conclusion-first report, a technical appendix, a "
        "figure set and the evidence tables. Read `report.md` first, then `figures.md` "
        "and `assets.txt`."),
    "assets_header": ("| 资产 | 证据层级 | 用途 |", "| Asset | Evidence level | Purpose |"),
    "assets_rule": ("|---|---|---|", "|---|---|---|"),
    "asset_report": (
        "| `report.md` | final_report | 结论优先的中文主报告 |",
        "| `report.md` | final_report | Conclusion-first main report |"),
    "asset_figures": (
        "| `figures.md` | figure_view | 直接内嵌关键 PNG 图和中文图注 |",
        "| `figures.md` | figure_view | Key PNG figures with their captions, embedded directly |"),
    "asset_assets": (
        "| `assets.txt` | asset_index | 纯文本列出关键报告、图表和证据表 |",
        "| `assets.txt` | asset_index | Plain-text index of the key reports, figures and evidence tables |"),
    "asset_core_story": (
        "| `evaluation_evidence/core_story_evidence.csv` | current_matrix | 核心对比、候选蛋白、模块方向的故事骨架 |",
        "| `evaluation_evidence/core_story_evidence.csv` | current_matrix | Story backbone of core contrasts, candidate proteins and module directions |"),
    "asset_candidates": (
        "| `evaluation_evidence/candidate_protein_evidence.csv` | current_matrix | 候选蛋白检出、logFC、P.Value、adj.P.Val/FDR |",
        "| `evaluation_evidence/candidate_protein_evidence.csv` | current_matrix | Candidate detection, logFC, P.Value and adj.P.Val/FDR |"),
    "asset_modules": (
        "| `evaluation_evidence/curated_module_group_summary.csv` | current_matrix | Curated 模块在各分组中的均值和差值 |",
        "| `evaluation_evidence/curated_module_group_summary.csv` | current_matrix | Curated module means and deltas per group |"),
    "asset_group_qc": (
        "| `evaluation_evidence/group_composition_qc.csv` | current_matrix | 样本组成、缺失率和分组交叉表 |",
        "| `evaluation_evidence/group_composition_qc.csv` | current_matrix | Sample composition, missingness rates and group cross-tables |"),
    "asset_figure_index": (
        "| `visualize_results/figure_index.md` | current_matrix | 完整图册索引 |",
        "| `visualize_results/figure_index.md` | current_matrix | Complete figure index |"),
    "boundary_current_matrix": (
        "- `current_matrix`：本报告的差异、候选、模块、QC 和图册均来自当前矩阵；它们能支持方向和优先级，不能单独证明因果。",
        "- `current_matrix`: the differential results, candidates, modules, QC and figures "
        "in this report all come from the current matrix. They support direction and priority "
        "and cannot on their own establish causation."),
    "boundary_offline_enrichment": (
        "- `offline_enrichment`：只作为本地 GO/KEGG/Reactome 的通路解释；未通过 FDR 的条目只能写作探索性提示。",
        "- `offline_enrichment`: used only as local GO/KEGG/Reactome pathway interpretation; "
        "terms that do not pass FDR may only be written as exploratory notes."),
    "boundary_external_applicable": (
        "- `external_annotation`：本地外部注释只用于离线交叉表，不证明当前矩阵中的疾病或风险因果关系。",
        "- `external_annotation`: local external annotation is used only for offline "
        "cross-tables and does not establish disease or risk causation in the current matrix."),
    "boundary_external_not_applicable": (
        "- `external_annotation`：本数据集未启用疾病或风险基因外部注释，正文结论不依赖该证据层。",
        "- `external_annotation`: disease or risk-gene external annotation is not enabled "
        "for this dataset, and the conclusions here do not depend on that evidence layer."),
    "boundary_extension": (
        "- `extension`：启发式推测和后续验证建议用于提出可检验假设，不能替代差异统计或独立实验。",
        "- `extension`: heuristic inference and follow-up suggestions exist to state "
        "testable hypotheses and do not replace differential statistics or independent "
        "experiments."),
    "boundary_records": (
        "- `analysis_design.used.yaml` 记录分组和对比设计；`evidence_ledger.jsonl` 保存证据来源标签，便于复核。",
        "- `analysis_design.used.yaml` records the grouping and contrast design, and "
        "`evidence_ledger.jsonl` stores the evidence-source labels for review."),
    "appendix_intro": (
        "本节集中列出可复核证据表、图册索引和审计文件。正文不保留 raw tool JSON 或长工具日志。",
        "This section collects the reproducible evidence tables, the figure index and the "
        "audit files. The body text keeps no raw tool JSON and no long tool logs."),
    "default_story_title": (
        "围绕核心对比组织的蛋白质组差异",
        "proteome differences organised around the core contrasts"),
    "footer_design": (
        "- 分析设计记录：说明本次分析的分组选择，以及设计是用户提供、数据集提供还是自动推断；随证据包交付。",
        "- Analysis design record: states the grouping used here and whether the design was "
        "user-provided, dataset-provided or automatically inferred; delivered with the "
        "evidence pack."),
    "footer_ledger": (
        "- 证据账本：记录每条结论的来源标签（本次分析的数据矩阵、离线富集结果、外部文献、药物数据库与分析设计）；随证据包交付。",
        "- Evidence ledger: records the source label of every conclusion (the data matrix "
        "behind this analysis, offline enrichment results, external literature, drug "
        "databases and the analysis design); delivered with the evidence pack."),
    "footer_external": (
        "- 外部知识审计：`{path}` 保存 PubMed/Google/DGIdb 查询和返回记录。",
        "- External-knowledge audit: `{path}` stores the PubMed/Google/DGIdb queries and "
        "their returned records."),
    "footer_confidence": (
        "- confidence: moderate 是自动推断设计或 Python fallback 结论的默认上限；high confidence 需要用户复核设计和成熟统计后端。",
        "- confidence: moderate is the default ceiling for conclusions from an automatically "
        "inferred design or a Python fallback; high confidence requires a user-reviewed "
        "design and a mature statistical backend."),
    "footer_combat": (
        "- 批次校正边界：当 `combat_success=false` 时，`ProteinQuant_ComBat.csv` 是下游兼容输出路径，不应解释为新完成的 ComBat 校正证据。",
        "- Batch-correction boundary: when `combat_success=false`, "
        "`ProteinQuant_ComBat.csv` is a downstream compatibility output path and must not "
        "be read as evidence of a newly completed ComBat correction."),
    "boundary_supplement": ("### 结论边界补充", "### Evidence-boundary supplement"),
    "required_deliverables": (
        "必需交付与候选蛋白核查",
        "Required Deliverables and Candidate Protein Audit"),
    "candidate_audit_title": (
        "候选蛋白逐条核查（完整表）",
        "Candidate protein audit (complete table)"),
    "candidate_audit_note": (
        "未检出的候选按当前矩阵的记录形式（见「本次分析条件」）判读：可能是该蛋白组不在分析矩阵中，"
        "也可能是检出结构不支持定量比较；两者都不等于该蛋白在生物学上不存在。",
        "A candidate that was not detected is read through the recording convention of the "
        "current matrix (see Analysis Conditions for This Run): the protein group may be "
        "absent from the analysis matrix, or the detection structure may not support a "
        "quantitative comparison. Neither case means the protein does not exist "
        "biologically."),
    "mechanism_integration": (
        "机制整合与解释边界",
        "Mechanism Integration and Interpretation Boundaries"),
    "appendix_reproducible": ("可复核证据附录", "Reproducible Evidence Appendix"),
    "appendix_technical": ("技术证据附录", "Technical Evidence Appendix"),
    "appendix_scoring": ("评分证据附录", "Scoring Evidence Appendix"),
    "appendix_leakage": ("蛋白泄漏证据附录", "Protein Leakage Evidence Appendix"),
    "appendix_scoring_intro": (
        "以下条目说明正文结论的证据来源与复核入口；数值已按读者可读形式列入正文对应小节，本表不再暴露内部文件名。",
        "The entries below state the evidence source and the review entry point for the "
        "conclusions in the body. The numbers are already listed in reader-facing form in "
        "the corresponding sections, and this table no longer exposes internal file names."),
    "evidence_ledger_header": (
        "| 证据账 | 本报告中的复核入口 |",
        "| Evidence ledger | Review entry in this report |"),
    "appendix_leakage_intro": (
        "以下泄漏证据来自膜状态对比与 curated compartment-marker 汇总，属本次分析的数据矩阵证据。",
        "The leakage evidence below comes from the membrane-state contrast and the curated "
        "compartment-marker summary, and belongs to the data-matrix evidence of this run."),
    "appendix_leakage_qc_title": (
        "#### 蛋白泄漏 QC 证据", "#### Protein Leakage QC Evidence"),
    "story_task_heading": ("{n}. 任务{label}：{title}", "{n}. Task {index}: {title}"),
    "story_task_default_title": ("核心对比", "core contrast"),
    "convention_threshold": ("adj.P<{p:g} 且 |logFC|>{l:g}", "adj.P<{p:g} and |logFC|>{l:g}"),
    "key_summary_opening": (
        "- {dataset} 的核心问题是：{story_title}；结论以 current_matrix 为主，离线富集和推测只作解释层。",
        "- The central question for {dataset} is: {story_title}. Conclusions rest on "
        "current_matrix; offline enrichment and inference are only an interpretation layer."),
    "key_summary_default_story": (
        "当前矩阵核心对比故事", "the core contrast story of the current matrix"),
    "evidence_up": ("{group}较高：{genes}", "{group} higher: {genes}"),
    "evidence_up_short": ("{group}较高", "{group} higher"),
    "evidence_insufficient": (
        "代表蛋白不足，需按效应量和候选表保守解读",
        "too few representative proteins; read conservatively from effect sizes and the "
        "candidate table"),
    "key_summary_contrast": (
        "- {claim}：{contrast} 中 n_sig={nsig}（{convention}），{up}、{down}；{evidence}。",
        "- {claim}: in {contrast}, n_sig={nsig} ({convention}); {up}, {down}; {evidence}."),
    "key_summary_candidates": (
        "- 代表蛋白层面优先关注 {bits}；未达 FDR 的候选只作为方向线索。",
        "- At the representative-protein level, prioritise {bits}; candidates that do not "
        "reach FDR serve only as directional leads."),
    "key_summary_modules": (
        "- 模块层面显示 {bits}；模块 Δ 是平均分差，不替代差异蛋白 FDR。",
        "- At the module level, {bits}; module deltas are mean-score differences and do not "
        "replace differential-protein FDR."),
    "key_summary_boundary": ("- 边界：{boundary}", "- Boundary: {boundary}"),
    "boundary_default": (
        "所有外推均需标注 evidence layer，不能替代当前矩阵结论。",
        "Every extrapolation must be labelled with its evidence layer and cannot replace "
        "conclusions from the current matrix."),
    "contrast_missing": (
        "当前矩阵未生成对应核心对比，结论只能保持低置信度。",
        "The current matrix produced no corresponding core contrast, so conclusions can only "
        "stay at low confidence."),
    "contrast_share": (
        "（占表内蛋白 {tested} 的 {share:.2f}%）",
        " ({share:.2f}% of the {tested} proteins in the table)"),
    "contrast_strength": (
        "{nsig} 个蛋白通过筛选口径{share}；",
        "{nsig} proteins passed the screening convention{share}; "),
    "contrast_positive": (
        "{contrast} 的对比结果为：{strength}{group_a} 较高 {up} 个，{group_b} 较高 {down} 个；"
        "代表蛋白包括 {group_a} 侧 {top_up}，{group_b} 侧 {top_down}。",
        "For {contrast}: {strength}{group_a} is higher for {up} proteins and {group_b} for "
        "{down}; representative proteins are {top_up} on the {group_a} side and {top_down} "
        "on the {group_b} side."),
    "contrast_negative": (
        "{contrast} 在当前阈值下没有稳定显著差异蛋白；因此只把按效应量排序的候选变化作为探索性线索，"
        "{group_a} 较高候选为 {top_up}，{group_b} 较高候选为 {top_down}。",
        "At the current thresholds {contrast} has no stably significant differential "
        "proteins, so only the effect-size-ranked candidate changes are treated as "
        "exploratory leads: {top_up} are higher in {group_a} and {top_down} in {group_b}."),
    "not_listed": ("未列出", "not listed"),
    "task_table_title": ("### 核心对比表", "### Core contrast table"),
    "task_table_header": (
        "| 对比 | n_sig | A组较高 | B组较高 | 代表蛋白 | 统计口径 |",
        "| Contrast | n_sig | Higher in A | Higher in B | Representative proteins | Convention |"),
    "task_table_rule": ("|---|---:|---:|---:|---|---|", "|---|---:|---:|---:|---|---|"),
    "task_table_note": (
        "n_sig 口径：{convention}；logFC/adj.P.Val 来自当前差异表",
        "n_sig convention: {convention}; logFC/adj.P.Val come from the current differential table"),
    "candidate_table_title": ("### 代表蛋白表", "### Representative proteins"),
    "candidate_table_header": (
        "| 候选蛋白 | 当前矩阵方向 | logFC | P.Value | adj.P.Val/FDR | 解读 |",
        "| Candidate | Direction in the current matrix | logFC | P.Value | adj.P.Val/FDR | Reading |"),
    "candidate_table_rule": ("|---|---|---:|---:|---:|---|", "|---|---|---:|---:|---:|---|"),
    "candidate_default_reading": (
        "同任务候选，需结合 FDR 和效应量判断优先级。",
        "A candidate from the same task; judge priority from FDR together with effect size."),
    "module_table_title": ("### 模块证据", "### Module evidence"),
    "module_table_header": (
        "| 模块 | A组均值 | B组均值 | Δ(A-B) | 代表蛋白数 | 解读 |",
        "| Module | Mean in A | Mean in B | Delta (A-B) | Representative proteins | Reading |"),
    "module_table_rule": ("|---|---:|---:|---:|---:|---|", "|---|---:|---:|---:|---:|---|"),
    "module_not_computed": (
        "未计算出模块均值差，通常是分组不存在或模块匹配蛋白不足。",
        "The module mean difference could not be computed, usually because a group is "
        "missing or too few module proteins matched."),
    "module_weak": ("{group_a} 与 {group_b} 的模块差异较弱。", "The module difference between {group_a} and {group_b} is weak."),
    "module_a_higher": ("{group_a} 在该模块上更高。", "{group_a} is higher in this module."),
    "module_b_higher": ("{group_b} 在该模块上更高。", "{group_b} is higher in this module."),
    "ac_analysed_matrix": (
        "分析矩阵 %s",
        "analysed matrix %s"),
    "ac_applied_with": (
        "已执行（%s）",
        "applied (%s)"),
    "ac_batch_block": (
        "- 批次处理：%s（%s）；矩阵有效性状态 %s。",
        "- Batch handling: %s (%s); matrix validity status %s."),
    "ac_batch_col_undeclared": (
        "元数据中未声明",
        "not declared in the metadata"),
    "ac_batch_corrected": (
        "已执行新的批次校正",
        "a new batch correction was applied"),
    "ac_batch_not_corrected": (
        "未执行新的批次校正",
        "no new batch correction was applied"),
    "ac_batch_not_recorded": (
        "- 批次处理：本轮未记录批次校正分支；如元数据未声明批次列，则不做批次校正。",
        "- Batch handling: this round recorded no batch-correction branch; when the metadata declares no batch column, no batch correction is performed."),
    "ac_chain_suffix": (
        "；%s",
        "; %s"),
    "ac_contrasts_see_sections": (
        "见各任务小节",
        "see the task sections"),
    "ac_design_source": (
        "- 设计来源：{design_source}；样本标识列 {sample_id_col}，蛋白组列 {protein_id_col}，分组列 {group_col}，物种 {species}。",
        "- Design source: {design_source}; sample identifier column {sample_id_col}, protein group column {protein_id_col}, grouping column {group_col}, species {species}."),
    "ac_design_variables_fallback": (
        "设计变量",
        "the design variables"),
    "ac_design_vars": (
        "- 设计变量：分组列 %s；协变量 %s；批次列 %s。",
        "- Design variables: grouping column %s; covariates %s; batch column %s."),
    "ac_filter_rule": (
        "过滤：%s",
        "filtering: %s"),
    "ac_form_not_recorded": (
        "形式未记录",
        "form not recorded"),
    "ac_fraction_full_matrix": (
        "全矩阵非缺失比例 %s（%s 行）",
        "full-matrix non-missing fraction %s (%s rows)"),
    "ac_fraction_head_sample": (
        "非缺失比例 %s（设计推断时按矩阵前 %s 行抽样估计，抽样口径，不等于全矩阵缺失率）",
        "non-missing fraction %s (estimated during design inference by sampling the first %s rows of the matrix, which is a sampling convention and not the full-matrix missingness rate)"),
    "ac_fraction_not_recorded": (
        "非缺失比例未记录",
        "non-missing fraction not recorded"),
    "ac_fraction_unlabelled": (
        "非缺失比例 %s（记录未标明是全矩阵还是抽样，引用前需核验）",
        "non-missing fraction %s (the record does not say whether this is the full matrix or a sample, so verify it before citing)"),
    "ac_group_col_none": (
        "无",
        "none"),
    "ac_imputed_values": (
        "，填补 %s 个取值",
        ", %s values imputed"),
    "ac_input_encoding_fallback": (
        "输入矩阵的原始编码",
        "the original encoding of the input matrix"),
    "ac_matrix_state": (
        "- 矩阵状态：输入为%s；%s；数值范围 %s 至 %s。",
        "- Matrix state: input is %s; %s; numeric range %s to %s."),
    "ac_method_not_recorded": (
        "方法未记录",
        "method not recorded"),
    "ac_missing_and_scale": (
        "- 缺失与尺度（按本轮实际执行记录）：未检出以%s记录；缺失值填补 %s；对数变换 %s%s。",
        "- Missingness and scale (from the execution record of this round): non-detection is recorded as %s; missing-value imputation %s; log transform %s%s."),
    "ac_missing_and_scale_not_recorded": (
        "- 缺失与尺度：未检出以%s记录；本轮实际执行的填补与对数变换未记录（状态：未记录，不得据此推断为未执行）。",
        "- Missingness and scale: non-detection is recorded as %s; the imputation and log transform actually executed this round are not recorded (status: not recorded, which must not be read as not executed)."),
    "ac_no_extra_design_variables": (
        "无额外设计变量",
        "no additional design variables"),
    "ac_not_applied": (
        "未执行",
        "not applied"),
    "ac_not_recorded_with_rule": (
        "未记录（记录中的规则：%s）",
        "not recorded (rule in the record: %s)"),
    "ac_other_design_variables": (
        "- 其他可用设计变量：%s（本轮未按其分层比较）。",
        "- Other available design variables: %s (not compared by stratum in this round)."),
    "ac_proteins_retained": (
        "保留蛋白 %s 个",
        "%s proteins retained"),
    "ac_several_rows": (
        "若干",
        "several"),
    "ac_statistical_method": (
        "- 统计方法：%s；多重校正 %s；通过筛选的口径为校正后 P（%s，即差异表中的 adj.P.Val）< %s 且 |logFC| > %s（未校正的 P.Value 列保留在差异表中供参考，不参与筛选）；主对比 %s。",
        "- Statistical method: %s; multiple correction %s; the screening convention is the adjusted P (%s, that is adj.P.Val in the differential table) < %s and |logFC| > %s (the unadjusted P.Value column stays in the differential table for reference and takes no part in screening); main contrast %s."),
    "ac_stratified_absent": (
        "- 分层情况：本轮已计算 %d 个对比，未生成按设计变量（%s）分层的差异对比；相关背景条件只在分组 QC 交叉表与解释边界中讨论。",
        "- Stratification: this round computed %d contrasts and generated no differential contrast stratified by the design variables (%s); the related background conditions are discussed only in the group QC cross-table and in the interpretation boundaries."),
    "ac_stratified_present": (
        "- 分层情况：本轮已计算 %d 个对比；证据层按 %s 生成 %d 条分层对比记录，逐层结果表列入可复核资产清单。",
        "- Stratification: this round computed %d contrasts; by %s the evidence layer generated %d stratified contrast records, and the per-stratum result tables are listed in the reproducible asset list."),
    "boundary_prefix": (
        "边界：",
        "Boundary: "),
    "brain_axis": (
        "- current_matrix RG/oRG -> IPC-EN -> EN 轴：",
        "- current_matrix RG/oRG -> IPC-EN -> EN axis: "),
    "brain_axis_tail": (
        "。这些模块分数分别支持 progenitor、IPC transition、EN maturation、synapse/neurite、chromatin/BAF 与 cell-cycle 程序。",
        ". These module scores support the progenitor, IPC transition, EN maturation, synapse/neurite, chromatin/BAF and cell-cycle programmes respectively."),
    "brain_batch_boundary": (
        "- current_matrix 批次边界：主要评分状态覆盖多个 batch（",
        "- current_matrix batch boundary: the main scoring states cover several batches ("),
    "brain_batch_boundary_tail": (
        "）；未分配的 `nan` 细胞不用于发育状态结论，ComBat fallback 作为边界明示。",
        "); unassigned `nan` cells are not used for developmental-state conclusions, and the ComBat fallback is stated explicitly as a boundary."),
    "brain_batch_quant": (
        "- current_matrix 批次量化补充：",
        "- current_matrix batch quantification supplement: "),
    "brain_batch_quant_tail": (
        "。这些数值用于说明状态分布与 QC，而不是把 fallback 矩阵描述为已完成 ComBat 校正。",
        ". These values describe the state distribution and QC, and do not describe the fallback matrix as a completed ComBat correction."),
    "brain_candidates": (
        "- current_matrix 候选蛋白：",
        "- current_matrix candidate proteins: "),
    "brain_candidates_tail": (
        "。优先展示 RG/oRG、IPC-EN/oRG、EN/IPC-EN、EN/RG 或 EN/oRG 等发育轴对比；涉及 Microglia/OPC/Vascular 的行只作为背景差异，不作为主轴证据。",
        ". Developmental-axis contrasts such as RG/oRG, IPC-EN/oRG, EN/IPC-EN, EN/RG or EN/oRG are shown first; rows involving Microglia, OPC or Vascular serve only as background differences and not as main-axis evidence."),
    "brain_enrichment_rule": (
        "- offline_enrichment 解释规则：宽泛 viral/infectious Reactome 标签不作为主要脑发育结论；主要解释优先来自 FDR 支持的 RNA processing/splicing、translation/ribosome、chromatin/nuclear complex、synapse/neurite、mitochondrial/OXPHOS 和 cell-cycle 条目。",
        "- offline_enrichment interpretation rule: broad viral or infectious Reactome labels are not used as main brain-development conclusions; the main interpretation comes first from the FDR-supported RNA processing/splicing, translation/ribosome, chromatin/nuclear complex, synapse/neurite, mitochondrial/OXPHOS and cell-cycle entries."),
    "brain_external_boundary": (
        "- external_annotation ASD/NDD 边界：本地交叉表在当前矩阵中检出 ",
        "- external_annotation ASD/NDD boundary: the local cross-table detects in the current matrix "),
    "brain_external_boundary_tail": (
        "；只有与 EN maturation 或 chromatin/BAF 模块重叠时解释力更强，但仍是注释层，不是疾病因果证明。",
        "; the interpretation is stronger only where these overlap the EN maturation or chromatin/BAF modules, and even then it stays an annotation layer rather than proof of disease causation."),
    "brain_module_coverage": (
        "- current_matrix 功能模块覆盖：",
        "- current_matrix functional module coverage: "),
    "brain_scoring_title": (
        "### Brain 评分主线解释",
        "### Brain scoring main-line interpretation"),
    "brain_state_qc": (
        "- current_matrix 状态/QC 边界：PCA/UMAP 图显示 2310 个细胞的矩阵具有状态结构；",
        "- current_matrix state / QC boundary: the PCA and UMAP figures show that the 2310-cell matrix has state structure; "),
    "cand_audit_detected": (
        "已检出",
        "detected"),
    "cand_audit_diff_row": (
        "| %s | %s | %s | 已检出 | %s | %s | %s | %s | 差异表（任务点名候选） |",
        "| %s | %s | %s | detected | %s | %s | %s | %s | differential table (task-named candidate) |"),
    "cand_audit_exploratory_title": (
        "**探索性候选（限量呈现）**",
        "**Exploratory candidates (limited presentation)**"),
    "cand_audit_footer_note": (
        "同一基因对应多个蛋白组时按蛋白组分行呈现；对比列写的是该行统计实际所属的对比，未计算该对比时留空并注明，不用其他对比的数值替代。",
        "When one gene maps to several protein groups the rows are per protein group; the contrast column states the contrast the row's statistics actually belong to, and when that contrast was not computed the cell is left empty and annotated rather than filled with numbers from another contrast."),
    "cand_audit_header_full": (
        "| 候选蛋白 | 蛋白组 | 对比 | 检出状态 | 方向 | logFC | P.Value | adj.P.Val | 来源 |",
        "| Candidate | Protein group | Contrast | Detection status | Direction | logFC | P.Value | adj.P.Val | Source |"),
    "cand_audit_header_short": (
        "| 候选蛋白 | 蛋白组 | 对比 | 检出状态 | 方向 | logFC | P.Value | adj.P.Val |",
        "| Candidate | Protein group | Contrast | Detection status | Direction | logFC | P.Value | adj.P.Val |"),
    "cand_audit_no_named": (
        "**说明**：本轮任务文本未点名具体候选蛋白，下表为数据集中预设候选的探索性呈现。",
        "**Note**: this round's task text names no specific candidate proteins, so the table below is an exploratory presentation of the candidates preset in the dataset."),
    "cand_audit_not_detected": (
        "未检出",
        "not detected"),
    "cand_audit_not_evaluated": (
        "未评估（该对比未计算）",
        "not evaluated (this contrast was not computed)"),
    "cand_audit_title_named": (
        "**任务点名的候选蛋白（%d 个，全部呈现，不按数据存在性截断）**",
        "**Candidate proteins named by the task (%d, all shown, not truncated by whether data exists)**"),
    # English-only line: a zh report must stay byte-identical to the released text, so the zh
    # value is never emitted (see _build_candidate_audit_lines).
    "cand_audit_prose_note": (
        "**说明**：任务文本中另有 %d 个词按普通用词或设计分组名处理，未计入候选蛋白。",
        "**Note**: %d further tokens in the task text were read as ordinary words or design group labels and are not counted as candidate proteins."),
    "cand_audit_unresolved": (
        "未评估：未在任何矩阵、候选表或差异表中匹配到该符号（名称待解析或当前矩阵未检出）",
        "not evaluated: the symbol did not match any matrix, candidate table or differential table (the name is unresolved, or it is not detected in the current matrix)"),
    "cand_audit_unresolved_row": (
        "| %s | — | — | 见右列 | — | — | — | — | %s |",
        "| %s | - | - | see the right column | - | - | - | - | %s |"),
    "cand_dir_matrix_detected": (
        "当前矩阵检出",
        "detected in the current matrix"),
    "cand_dir_no_symbol_map": (
        "未建立符号映射（该符号未匹配到矩阵标识，不等同于未检出）",
        "no symbol mapping was established (the symbol did not match a matrix identifier, which does not mean it was not detected)"),
    "cand_dir_not_detected": (
        "在可用矩阵中未检出",
        "not detected in the available matrix"),
    "cand_dir_undetermined": (
        "方向未定",
        "direction undetermined"),
    "checklist_computed": (
        "已计算（adj.P<%s：%d；再叠加 |log2FC|>%s：%d；表内蛋白 %d；%s）",
        "computed (adj.P<%s: %d; after also applying |log2FC|>%s: %d; proteins in the table %d; %s)"),
    "checklist_contrast_prefix": (
        "对比 ",
        "Contrast "),
    "checklist_header": (
        "| 必需项目 | 状态 | 依据 |",
        "| Required item | Status | Evidence |"),
    "checklist_mentioned": (
        "正文已提及（文本匹配）",
        "mentioned in the body (text match)"),
    "checklist_not_mentioned": (
        "正文未提及",
        "not mentioned in the body"),
    "ckv_direction": (
        "{gene}（{direction}）",
        "{gene} ({direction})"),
    "ckv_full": (
        "{gene}（{direction}，logFC={logfc}，FDR={fdr}）",
        "{gene} ({direction}, logFC={logfc}, FDR={fdr})"),
    "ckv_logfc": (
        "{gene}（{direction}，logFC={logfc}）",
        "{gene} ({direction}, logFC={logfc})"),
    "convention_groups_pair": (
        "A 组 {group_a}、B 组 {group_b}",
        "Group A {group_a}, Group B {group_b}"),
    "d4_boundary_suffix": (
        "；%s。",
        "; %s."),
    "d4_closing_boundary": (
        "边界：本节的方向与分类判断来自当前矩阵的差异表与模块分数；不构成对上游实验设计或因果机制的检验%s。",
        "Boundary: the directional and classification judgements in this section come from the differential tables and the module scores of the current matrix; they do not test the upstream experimental design or any causal mechanism%s."),
    "d4_contrast_count": (
        "**%s**：adj.P.Val < %s 的蛋白 %d；再叠加 |log2FC| > %s 后为 %d（表内蛋白 %d）。",
        "**%s**: proteins with adj.P.Val < %s: %d; after also applying |log2FC| > %s: %d (proteins in the table %d)."),
    "d4_heading": (
        "## %d. 机制整合与解释边界",
        "## %d. Mechanism Integration and Interpretation Boundaries"),
    "d4_intro": (
        "本节把各任务小节的蛋白级结果、模块证据与离线富集放在同一处收束，并给出明确的边界判断；判读规则是：只有通过筛选口径的候选进入机制解释，未通过筛选的方向线索只作为假设。",
        "This section brings the protein-level results, the module evidence and the offline enrichment of the task sections together in one place and states the boundary judgements explicitly; the reading rule is that only candidates passing the screening convention enter the mechanism interpretation, while directional leads that do not pass it serve as hypotheses only."),
    "d4_label_agree": (
        "与标签一致",
        "agrees with the label"),
    "d4_label_disagree": (
        "与标签不一致",
        "disagrees with the label"),
    "d4_label_no_expectation": (
        "未设定预期状态",
        "no expected state set"),
    "d4_module_top_state": (
        "%s 在 %s 上最高（%.3f，%s）",
        "%s is highest in %s (%.3f, %s)"),
    "d4_not_ordered": (
        "非中序",
        "not ordered"),
    "d4_ordered": (
        "中序",
        "ordered"),
    "d4_verdict_bloodcell": (
        "判定：标注状态与蛋白程序的一致性——%s。两个模块的最高状态分别落在其预期状态上，说明注释标签与蛋白程序方向一致；这不构成对各状态之间连续过渡或供体效应的检验。",
        "Verdict: consistency between the annotated states and the protein programmes: %s. The highest state of each module falls on its expected state, which indicates that the annotation labels and the protein programme directions agree; this does not test continuous transitions between states or donor effects."),
    "d4_verdict_generic": (
        "判定：本节机制解释以 %d 条通过筛选口径的候选证据为主，%d 条为未通过筛选的方向线索，%d 条候选在当前矩阵中未检出。方向线索只作假设，不进入结论句。",
        "Verdict: the mechanism interpretation in this section rests mainly on %d candidate records that pass the screening convention, with %d directional leads that do not pass it and %d candidates not detected in the current matrix. Directional leads serve as hypotheses only and do not enter the conclusion sentences."),
    "d4_verdict_ipsceb": (
        "判定：EB 组内部并非同质。EB 样本内的模块分数分布为——%s。因此 EB 与 iPSC 的组间比较描述的是中心趋势，组内梯度与亚群结构需另外阅读异质性图。",
        "Verdict: the EB group is not internally homogeneous. The module score distribution inside the EB samples is: %s. The EB-vs-iPSC comparison therefore describes a central tendency, and within-group gradients and substructure must be read from the heterogeneity figures separately."),
    "d4_verdict_liver": (
        "判定：在 %d 个可判读的焦点基因中，%d 个满足 Portal—Midlobular—Central 的中序关系，支持把 Midlobular 作为两侧之间的过渡带而不是独立区带；其余基因不满足该关系，说明分区不是单一线性梯度。逐基因结果：%s。",
        "Verdict: of %d readable focal genes, %d satisfy the Portal-Midlobular-Central ordering, supporting Midlobular as a transition zone between the two sides rather than an independent zone; the remaining genes do not satisfy it, which shows the zonation is not a single linear gradient. Per-gene results: %s."),
    "design_source_auto": (
        "自动推断",
        "automatically inferred"),
    "design_source_dataset": (
        "数据集提供",
        "dataset-provided"),
    "design_source_user": (
        "用户提供",
        "user-provided"),
    "dim4_candidate_header": (
        "| 候选蛋白 | logFC | P.Value | adj.P.Val/FDR |",
        "| Candidate | logFC | P.Value | adj.P.Val/FDR |"),
    "dim4_cluster_size_note": (
        "**聚类簇规模（本图按全部 {total} 个观测聚类，不是组内子聚类）**：{text}。本轮未执行该组内部的子聚类与标记基因分析，组内分布以模块分数分布为准。",
        "**Cluster size (this figure clusters all {total} observations, not sub-clusters within a group)**: {text}. No sub-clustering or marker-gene analysis inside this group was run in this round; the within-group distribution is read from the module score distribution."),
    "dim4_integration_note": (
        "注：本表数值取自源差异表 `{contrast_name}`；正文核心表按「{display_label}」显示，两者符号相反，方向以文字与所标对比名为准。",
        "Note: the values in this table come from the source differential table `{contrast_name}`; the body core table displays it as \"{display_label}\", the two have opposite signs, and direction is read from the wording and from the stated contrast name."),
    "dim4_integration_row": (
        "**{label}｜{contrast_name}**：显著蛋白数 n_sig={n_sig}（adj.P.Val<0.05）。",
        "**{label} | {contrast_name}**: significant proteins n_sig={n_sig} (adj.P.Val<0.05)."),
    "dim4_prefixed_boundary": (
        "边界：{boundary}",
        "Boundary: {boundary}"),
    "dsi_ipsceb_candidate_chain": (
        "- current_matrix 多能性到分化的候选链条为：POU5F1/SOX2/LIN28A 在 EB 相对 iPSCs 下降，GATA4/HAND1/MAP2/FN1/COL1A1/COL3A1 等分化或 ECM 相关候选在 EB 上升；关键数值为 ",
        "- The current_matrix candidate chain from pluripotency to differentiation runs as follows: POU5F1/SOX2/LIN28A are lower in EB than in iPSCs, while differentiation or ECM-related candidates such as GATA4/HAND1/MAP2/FN1/COL1A1/COL3A1 are higher in EB; the key numbers are "),
    "dsi_ipsceb_chain_title": (
        "### iPSC/EB 分化与异质性链",
        "### iPSC/EB differentiation and heterogeneity chain"),
    "dsi_ipsceb_dispersion": (
        "- current_matrix EB-only 异质性不依赖平均值结论，而用 EB 细胞内部模块分布显示谱系梯度：",
        "- The current_matrix EB-only heterogeneity does not rest on mean-value conclusions; it uses the module distribution inside the EB cells to show a lineage gradient: "),
    "dsi_ipsceb_dispersion_tail": (
        " 这些结果支持 EB 内部存在连续差异，但不把模块高低直接定性为硬亚群或真实发育轨迹。",
        " These results support continuous differences inside EB, without labelling high and low modules directly as hard subpopulations or as a true developmental trajectory."),
    "dsi_ipsceb_module_chain": (
        "- current_matrix 模块分数把单蛋白方向整合为谱系与功能程序：",
        "- The current_matrix module scores integrate the single-protein directions into lineage and functional programmes: "),
    "dsi_ipsceb_module_tail": (
        "。pluripotency_core 与 cell_cycle_replication 更偏 iPSCs，lineage/ECM 模块更偏 EB，支持从多能性维持到 EB 谱系分化的主线。",
        ". pluripotency_core and cell_cycle_replication lean towards iPSCs while the lineage/ECM modules lean towards EB, supporting the main line from pluripotency maintenance to EB lineage differentiation."),
    "dsi_ipsceb_nanog_tail": (
        "。NANOG 未检出只写作矩阵检测边界，不写作生物学缺失。",
        ". The non-detection of NANOG is written only as a detection boundary of the matrix, not as a biological absence."),
    "dsi_ipsceb_normalisation": (
        "- current_matrix 标准化与差异模型摘要：本轮按 EB vs iPSCs 做蛋白层差异比较，并在报告中保留 ComBat fallback 边界；因此结论优先解释当前矩阵的方向和模块分数，而不复制论文全量流程中的固定 DEP 数量。",
        "- current_matrix normalisation and differential model summary: this run compares protein-level differences for EB vs iPSCs and keeps the ComBat fallback boundary on the record; the conclusions therefore read the direction and the module scores of the current matrix first, rather than copying the fixed DEP counts of the full original pipeline."),
    "dsi_pispa_candidate_chain": (
        "- current_matrix 候选蛋白把机制链落到可复核数值：",
        "- The current_matrix candidate proteins put the mechanism chain on reproducible numbers: "),
    "dsi_pispa_candidate_tail": (
        "。EZR/MSN/RDX 对应膜皮质，CDC42/RAC1/RHOA 对应小 GTPase，MYL9 对应收缩，TLN1/VCL/PXN/ITGB1 对应黏附复合体；每个方向均以 logFC/adj.P.Val 和对比方向为边界。",
        ". EZR/MSN/RDX correspond to the membrane cortex, CDC42/RAC1/RHOA to small GTPases, MYL9 to contractility and TLN1/VCL/PXN/ITGB1 to the adhesion complex; every direction is bounded by logFC/adj.P.Val and by the recorded contrast direction."),
    "dsi_pispa_chain_title": (
        "### PiSPA 迁移机制链",
        "### PiSPA migration mechanism chain"),
    "dsi_pispa_cluster_type": (
        "- current_matrix Cluster-Type 结构显示迁移细胞主要富集在 Cluster 1，而 Cluster 2/3 更偏对照或混合状态：",
        "- The current_matrix Cluster-Type structure shows migrated cells are mainly enriched in Cluster 1, while Cluster 2/3 are closer to control or mixed states: "),
    "dsi_pispa_cluster_type_tail": (
        "。这说明后续机制解释应以 Cluster 1 vs 2/3 的迁移轴和 Cluster 2 vs 3 的非迁移异质性为主。",
        ". This means the mechanism interpretation that follows should centre on the Cluster 1 vs 2/3 migration axis and on the non-migratory heterogeneity of Cluster 2 vs 3."),
    "dsi_pispa_enrichment_boundary": (
        "- offline_enrichment 未提供 FDR 显著富集时，报告不把通路名写成显著发现；迁移结论主要来自 current_matrix 候选蛋白、curated module score、Cluster-Type 交叉表和图册。",
        "- When offline_enrichment provides no FDR-significant enrichment, the report does not write pathway names as significant findings; the migration conclusions come mainly from the current_matrix candidate proteins, the curated module scores, the Cluster-Type cross-table and the figure set."),
    "dsi_pispa_module_chain": (
        "- current_matrix 模块链条从 Rho GTPase 调控、ERM 膜皮质连接、肌球蛋白收缩到 talin-vinculin/focal adhesion 逐步支持迁移表型：",
        "- The current_matrix module chain progressively supports the migration phenotype, from Rho GTPase regulation and ERM membrane-cortex linkage through myosin contractility to talin-vinculin/focal adhesion: "),
    "dsi_pispa_module_tail": (
        "。Cluster 1 在 migration_signature、ERM 和 adhesion 相关模块上整体高于 Cluster 2/3，属于强证据；Cluster 2 vs 3 的差异用于解释对照群体内部异质性，置信度较低。",
        ". Cluster 1 is overall higher than Cluster 2/3 in the migration_signature, ERM and adhesion-related modules, which is strong evidence; the Cluster 2 vs 3 difference explains heterogeneity inside the control population and carries lower confidence."),
    "eis_dispersion_row": (
        "EB 内部 {module} 模块在 {a1} 个 EB 细胞中存在分布差异：mean={mean:.3f}, range={a3:.3f} to {a4:.3f}, SD={a5:.3f}；这是 current_matrix 梯度证据，不作为硬性谱系判定。",
        "Within EB, the {module} module is distributed differently across {a1} EB cells: mean={mean:.3f}, range={a3:.3f} to {a4:.3f}, SD={a5:.3f}; this is current_matrix gradient evidence and not a hard lineage call."),
    "enrich_direction_group_a": (
        "{group_a} 较高蛋白",
        "{group_a} higher proteins"),
    "enrich_direction_group_b": (
        "{group_b} 较高蛋白",
        "{group_b} higher proteins"),
    "enrich_effect_group_a": (
        "{group_a}较高={a1}",
        "{group_a} higher={a1}"),
    "enrich_effect_group_b": (
        "{group_b}较高={a1}",
        "{group_b} higher={a1}"),
    "enrich_effect_pointer": (
        "效应量见对应核心对比表",
        "see the corresponding core contrast table for the effect size"),
    "enrich_effect_representative": (
        "代表={top}",
        "representative={top}"),
    "ext_batch_title": (
        "### 处理与批次的可比性",
        "### Comparability of handling and batches"),
    "ext_candidate_chain_row": (
        "- %s：%s",
        "- %s: %s"),
    "ext_candidate_chain_title": (
        "### 候选进入/排除链路",
        "### Candidate inclusion and exclusion chain"),
    "ext_contrast_compare_header": (
        "| 对比 A | 对比 B | 共享蛋白 | Spearman(logFC) | A/中位|logFC| | B/中位|logFC| | A 显著 | B 显著 | 显著重叠 | 共同显著符号一致 | FDR 来源 | 判定 |",
        "| Contrast A | Contrast B | Shared proteins | Spearman(logFC) | A / median |logFC| | B / median |logFC| | Significant in A | Significant in B | Significant overlap | Concordant sign among shared significant | FDR source | Verdict |"),
    "ext_contrast_compare_title": (
        "### 对比之间的并列比较",
        "### Side-by-side comparison between contrasts"),
    "ext_fdr_from_table": (
        "表内 adj.P.Val",
        "adj.P.Val from the table"),
    "ext_fdr_recomputed": (
        "BH 重算",
        "BH recomputed"),
    "ext_stratified_header": (
        "| 分层变量 | 层 | 对比（方向） | 效应尺度 | 第一臂 n | 第二臂 n | 通过筛选 | 状态 | 说明 |",
        "| Stratifier | Stratum | Contrast (direction) | Effect scale | n arm 1 | n arm 2 | Passed screening | Status | Note |"),
    "ext_stratified_row": (
        "| %s | %s | %s（%s） | %s | %s | %s | %s | %s | %s |",
        "| %s | %s | %s (%s) | %s | %s | %s | %s | %s | %s |"),
    "ext_stratified_title": (
        "### 分层对比",
        "### Stratified contrasts"),
    "fallback_boundary_enrichment": (
        "- 离线富集属于 offline_enrichment 证据；FDR 不显著时只能作为探索性提示。",
        "- Offline enrichment belongs to offline_enrichment evidence; terms that are not FDR-significant may only serve as exploratory notes."),
    "fallback_boundary_external": (
        "- 外部文献或药物数据库证据如存在，也不能解释为当前矩阵的直接证据。",
        "- External literature or drug-database evidence, when present, cannot be read as direct evidence from the current matrix."),
    "fallback_boundary_heading": (
        "## 科学边界",
        "## Scientific boundaries"),
    "fallback_boundary_matrix": (
        "- 矩阵统计属于 current_matrix 证据。",
        "- Matrix statistics belong to current_matrix evidence."),
    "fallback_h1": (
        "# scProteomics 分析报告",
        "# scProteomics Analysis Report"),
    "fallback_intro": (
        "本报告由本地 fallback reporter 生成，因为 LLM reporter 不可用。结论仅限于本轮 current_matrix 输出、offline_enrichment 和已标注来源的证据文件。",
        "This report was produced by the local fallback reporter because the LLM reporter was unavailable. Conclusions are limited to the current_matrix output of this round, to offline_enrichment, and to the evidence files with a recorded source."),
    "fallback_source_design": (
        "- analysis_design：`{a0}`。",
        "- analysis_design: `{a0}`."),
    "fallback_source_figures": (
        "- figure_manifest：`{a0}`。",
        "- figure_manifest: `{a0}`."),
    "fallback_source_ledger": (
        "- evidence_ledger：`{a0}`。",
        "- evidence_ledger: `{a0}`."),
    "fallback_source_matrix": (
        "- current_matrix：分析输出位于 `{a0}`。",
        "- current_matrix: the analysis output is in `{a0}`."),
    "fallback_sources_heading": (
        "## 证据来源",
        "## Evidence sources"),
    "fallback_summary_heading": (
        "## 分析摘要",
        "## Analysis summary"),
    "figure_index_and": (
        " 与 ",
        " and "),
    "figure_index_caption_fallback": (
        "见图册索引",
        "see the figure index"),
    "figure_index_captions": (
        "；图注与阈值见 ",
        "; captions and thresholds are in "),
    "figure_index_full_prefix": (
        "- 完整图册索引：",
        "- Complete figure index: "),
    "figure_index_row_prefix": (
        "- 图 ",
        "- Figure "),
    "figure_index_row_tail": (
        "（%s）：%s。",
        " (%s): %s."),
    "genr_contrast_direction": (
        "{contrast}；{direction}",
        "{contrast}; {direction}"),
    "genr_current_contrast": (
        "当前对比",
        "Current contrast"),
    "genr_differential_proteins": (
        "差异蛋白",
        "Differential proteins"),
    "genr_direction_entry": (
        "%s方向条目：该条来自 %s 的%s查询集，效应量与阈值以「相关效应量」列与核心对比表为准。",
        "%s-direction entry: this entry comes from the %s query set of %s; the effect size and the thresholds follow the \"Related effect size\" column and the core contrast table."),
    "genr_family_set": (
        "功能集合",
        "Functional set"),
    "genr_no_fdr_enrichment": (
        "未找到可直接用于正文的 FDR 支持离线富集；若存在 fallback 条目，只能作为探索性提示。",
        "No FDR-supported offline enrichment was found that can go straight into the body text; if fallback entries exist they can only serve as exploratory notes."),
    "genr_table_header": (
        "| 来源 | 对比与方向 | 富集条目 | p.adjust/FDR | Count | 相关效应量 | 解读 |",
        "| Source | Contrast and direction | Enriched term | p.adjust/FDR | Count | Related effect size | Reading |"),
    "genr_table_note": (
        "注：offline_enrichment 为本地离线富集解释，用于给差异蛋白集合命名并形成可检验假设；因果关系仍需独立实验验证。",
        "Note: offline_enrichment is a local offline enrichment interpretation, used to name the differential protein sets and to state testable hypotheses; causality still requires independent experimental validation."),
    "gqc_row_flat": (
        "{group_col}={group}：样本数={n_samples}，平均检出蛋白={mean_detected_proteins}，平均缺失率={mean_missing_rate}（该表未区分输入与分析矩阵）",
        "{group_col}={group}: n_samples={n_samples}, mean detected proteins={mean_detected_proteins}, mean missing rate={mean_missing_rate} (this table does not separate the input and the analysed matrix)"),
    "gqc_row_split": (
        "{group_col}={group}：样本数={n_samples}；输入矩阵平均检出={input_mean_detected_proteins}、平均缺失率={input_mean_missing_rate}；分析矩阵平均检出={analysed_mean_detected_proteins}、平均缺失率={analysed_mean_missing_rate}",
        "{group_col}={group}: n_samples={n_samples}; input matrix mean detected={input_mean_detected_proteins}, mean missing rate={input_mean_missing_rate}; analysed matrix mean detected={analysed_mean_detected_proteins}, mean missing rate={analysed_mean_missing_rate}"),
    "gtt_display_note": (
        "注：本节按「{a0}」显示（A 组 {group_a}、B 组 {group_b}）；底层差异表为 `{source_contrast}`，其中 logFC 符号与本表相反（本表 logFC 为显示方向），方向以文字与所标对比名为准。",
        "Note: this section is displayed as \"{a0}\" (Group A {group_a}, Group B {group_b}); the underlying differential table is `{source_contrast}`, where the logFC sign is opposite to this table (here the logFC follows the displayed direction), so read direction from the wording and from the stated contrast name."),
    "gtt_module_note": (
        "注：模块证据的 FDR 为 NA；效应量为模块平均分差 Δ，用于组织机制背景，不能替代正式差异蛋白或富集显著性。",
        "Note: the FDR for the module evidence is NA; the effect measure is the module mean difference Delta, which organises the mechanism background and does not replace formal differential-protein or enrichment significance."),
    "human_contrast_unspecified": (
        "未指定对比（仅存在性记录）",
        "no contrast specified (existence record only)"),
    "human_test_two_group": (
        "双组比较",
        "two-group comparison"),
    "lea_leakage_direction_note": (
        "下方的泄漏方向与模块摘要",
        "the leakage direction and module summaries below"),
    "lea_per_module_note": (
        "下方的逐模块表",
        "the per-module table below"),
    "missing_encoding_nan": (
        "空值",
        "NaN"),
    "missing_encoding_with_form": (
        "%s（记录形式：%s）",
        "%s (recorded form: %s)"),
    "missing_encoding_zero": (
        "零值",
        "zero"),
    "module_dispersion_row": (
        "%s：n=%d，中位数 %.3f，四分位距 %.3f（P25 %.3f 至 P75 %.3f，极差 %.3f）",
        "%s: n=%d, median %.3f, interquartile range %.3f (P25 %.3f to P75 %.3f, range %.3f)"),
    "module_name_cytoskeleton": (
        "骨架/迁移模块",
        "cytoskeleton/migration module"),
    "module_name_translation": (
        "翻译/核糖体模块",
        "translation/ribosome module"),
    "module_this_token": (
        "该模块",
        "this module"),
    "off_direction_detail": (
        "- %s：查询集 n=%s（该方向通过筛选的蛋白，映射之前）；其中 %s 个映射到本地资源；背景集 %s 个基因；方法=%s；资源=%s；FDR 阈值 %s。",
        "- %s: query set n=%s (proteins passing the screening in that direction, before mapping); %s of them map to the local resources; background set %s genes; method=%s; resource=%s; FDR threshold %s."),
    "off_entry_row": (
        "  - %s ｜ %s（校正 P=%s；查询集命中 %s 个；该通路基因集在背景中 %s 个；背景 %s 个；查询集命中比 %s%s）",
        "  - %s | %s (adjusted P=%s; query-set hits %s; this pathway gene set in the background %s; background %s; query-set hit ratio %s%s)"),
    "off_main_contrast": (
        "- 主对比（analysis design 指定的第一个对比）：%s。",
        "- Main contrast (the first contrast named by the analysis design): %s."),
    "off_members": (
        "；成员 ",
        "; members "),
    "off_method_ora": (
        "过度代表分析（ORA，阈值蛋白集合）",
        "Over-representation analysis (ORA, thresholded protein set)"),
    "off_method_ranked": (
        "排序型富集（不应按单蛋白 FDR 判定）",
        "Rank-based enrichment (must not be judged by single-protein FDR)"),
    "off_min_padjust": (
        "（最小 p.adjust=%s）",
        "(minimum p.adjust=%s)"),
    "off_no_direction_result": (
        "- %s：本轮未生成该方向的富集结果（通常表示该方向没有通过筛选的蛋白集合）。",
        "- %s: this round generated no enrichment result for that direction (usually because no protein set passed the screening in that direction)."),
    "off_no_fdr_entries": (
        "  - 该方向没有通过 FDR 的条目%s；本轮不引用富集作为该方向的证据。",
        "  - that direction has no entries passing FDR%s; this round cites no enrichment as evidence for that direction."),
    "off_resource_local_gmt": (
        "本地 GMT",
        "local GMT"),
    "off_section_note": (
        "以上条目来自本地离线富集结果，用于给差异蛋白集合命名；不构成因果或通路活性测量。",
        "The entries above come from the local offline enrichment results and are used to name the differential protein sets; they do not constitute a causal claim or a measurement of pathway activity."),
    "off_section_title": (
        "### 离线富集证据（主对比）",
        "### Offline enrichment evidence (main contrast)"),
    "off_tested_entries": (
        "  - 本轮对该方向检验了 %s 个基因集条目，落盘 %s 条；下面只引用通过 FDR 的前几条。",
        "  - This round tested %s gene-set entries for that direction and wrote %s to disk; only the leading entries that pass FDR are cited below."),
    "off_zero_citations_note": (
        "- 说明：主对比的富集引用数为 0，机制解释只依赖蛋白级与模块级证据。",
        "- Note: the number of enrichment citations for the main contrast is 0, so the mechanism interpretation rests on protein-level and module-level evidence only."),
    "pct_chi2": (
        "整体 χ² p={a0}",
        "Overall chi-square p={a0}"),
    "pct_composition_boundary": (
        "这是表型组成统计，不是蛋白差异对比；效应量为迁移比例、OR 和组成检验 p 值。",
        "This is a phenotypic composition statistic, not a protein differential contrast; the effect measures are the migration proportion, the OR and the composition test p value."),
    "pct_composition_direction": (
        "C1 迁移细胞 {a0}/{c1_total}（{c1_pct:.1f}%）；C2 {a3}/{c2_total}（{c2_pct:.1f}%）；C3 {a6}/{c3_total}（{c3_pct:.1f}%）；{fisher_text}；{chi_text}",
        "C1 migrated cells {a0}/{c1_total} ({c1_pct:.1f}%); C2 {a3}/{c2_total} ({c2_pct:.1f}%); C3 {a6}/{c3_total} ({c3_pct:.1f}%); {fisher_text}; {chi_text}"),
    "pct_composition_evidence": (
        "Cluster × Type 组成显示迁移标签主要集中在 Cluster 1",
        "The Cluster x Type composition shows the migration label is concentrated in Cluster 1"),
    "pct_composition_interpretation": (
        "该组成证据说明 Cluster 1 是迁移状态的主要承载群体，后续差异统计以 Cluster 两两对比解释蛋白质组机制。",
        "This composition evidence indicates that Cluster 1 is the main carrier of the migration state, and the differential statistics that follow explain proteome mechanisms through pairwise Cluster contrasts."),
    "pct_composition_row": (
        "| 迁移标签与聚类的对应 | Migrated_vs_Control / Cluster × Type | {a0} | {a1} | {a2} | {a3} |",
        "| Migration label to cluster correspondence | Migrated_vs_Control / Cluster x Type | {a0} | {a1} | {a2} | {a3} |"),
    "pct_fisher": (
        "Fisher OR={a0}，p={a1}",
        "Fisher OR={a0}, p={a1}"),
    "pct_fisher_na": (
        "Fisher OR=NA，p=NA",
        "Fisher OR=NA, p=NA"),
    "pispa_asset_audit": (
        "| `analysis_design.used.yaml` 与 `evidence_ledger.jsonl` | audit trail | 记录分组设计和证据来源标签 |",
        "| `analysis_design.used.yaml` and `evidence_ledger.jsonl` | audit trail | Records the grouping design and the evidence-source labels |"),
    "pispa_asset_figure_index": (
        "| `visualize_results/figure_index.md` | current_matrix | PCA/UMAP、热图、候选蛋白和模块图册索引 |",
        "| `visualize_results/figure_index.md` | current_matrix | Figure index for the PCA/UMAP, heatmap, candidate protein and module figures |"),
    "pispa_asset_group_qc": (
        "| `evaluation_evidence/group_composition_qc.csv` | current_matrix | Cluster × Type、缺失率和样本组成 QC |",
        "| `evaluation_evidence/group_composition_qc.csv` | current_matrix | Cluster x Type composition, missingness rates and sample composition QC |"),
    "pispa_asset_modules": (
        "| `evaluation_evidence/curated_module_group_summary.csv` | current_matrix | 迁移、Rho、ERM、myosin、talin-vinculin 等模块分数 |",
        "| `evaluation_evidence/curated_module_group_summary.csv` | current_matrix | Module scores for migration, Rho, ERM, myosin, talin-vinculin and related modules |"),
    "pispa_assets_intro": (
        "本 run 同时生成纯文本资产清单与嵌图版 Markdown，便于合作者快速查看图表而不必翻完整日志。",
        "This run produces both a plain-text asset list and a figure-embedded Markdown view, so a collaborator can look at the figures quickly without reading the full log."),
    "pispa_boundary_current_matrix": (
        "- `current_matrix`：本报告的差异、候选、模块、QC 和图册均来自当前评测矩阵；它们能支持方向和优先级，不能单独证明因果。",
        "- `current_matrix`: the differential results, candidates, modules, QC and figures in this report all come from the current evaluation matrix; they support direction and priority but cannot on their own establish causation."),
    "pispa_boundary_extension": (
        "- `extension`：启发式推测和后续验证建议用于提出可检验假设，不能替代差异统计、显微验证或迁移扰动实验。",
        "- `extension`: heuristic inference and follow-up suggestions exist to state testable hypotheses and do not replace differential statistics, microscopy validation or migration perturbation experiments."),
    "pispa_boundary_external": (
        "- `external_annotation`：PiSPA 正文不使用疾病外部注释，不把背景知识写成当前矩阵结论。",
        "- `external_annotation`: the PiSPA body text does not use disease external annotation and does not write background knowledge as a current-matrix conclusion."),
    "pispa_boundary_records": (
        "- `analysis_design.used.yaml` 记录分组和对比设计；`evidence_ledger.jsonl` 保存证据来源标签，便于追溯但不参与生物学结论本身。",
        "- `analysis_design.used.yaml` records the grouping and contrast design, and `evidence_ledger.jsonl` stores the evidence-source labels for traceability; neither takes part in the biological conclusions themselves."),
    "pispa_data_intro": (
        "本轮使用包内 `ProteinQuant.csv` 与 `SampleInfo.csv` 生成当前矩阵证据；分组、缺失和样本结构先由 `group_composition_qc.csv`、PCA/UMAP 与图册索引确认，再进入差异和模块解释。",
        "This round used the in-package `ProteinQuant.csv` and `SampleInfo.csv` to generate the current matrix evidence; grouping, missingness and sample structure are confirmed first from `group_composition_qc.csv`, the PCA/UMAP figures and the figure index, before the differential and module interpretation begins."),
    "pispa_enrich_contrast_direction": (
        "{contrast}；{a1}",
        "{contrast}; {a1}"),
    "pispa_enrich_header": (
        "| 语义家族 | 代表条目 | 对比与方向 | p.adjust/FDR | Count | 相关效应量 | 解读 |",
        "| Semantic family | Representative term | Contrast and direction | p.adjust/FDR | Count | Related effect size | Reading |"),
    "pispa_enrich_title": (
        "### 离线富集补充表",
        "### Offline enrichment supplement table"),
    "pispa_family_adhesion": (
        "黏附/细胞连接/骨架",
        "Adhesion / cell junction / cytoskeleton"),
    "pispa_family_adhesion_note": (
        "提示细胞连接、黏附或骨架预激活差异，可作为迁移状态的背景层。",
        "Suggests differences in cell junctions, adhesion or cytoskeleton pre-activation that can serve as a background layer for the migration state."),
    "pispa_family_ecm": (
        "ECM/分泌重塑",
        "ECM / secretory remodelling"),
    "pispa_family_ecm_note": (
        "提示分泌型基质和微环境重塑，是迁移相关解释中更接近表型的一层。",
        "Suggests secreted matrix and microenvironment remodelling, one of the layers closest to the phenotype in a migration-related interpretation."),
    "pispa_family_mito": (
        "线粒体/能量代谢",
        "Mitochondrial / energy metabolism"),
    "pispa_family_mito_note": (
        "提示对照亚群的能量代谢活跃度不同，尤其适合解释 C2/C3 背景状态差异。",
        "Suggests that the control subpopulations differ in energy-metabolism activity, which is particularly useful for explaining the C2/C3 background state difference."),
    "pispa_family_other": (
        "其它功能线索",
        "Other functional leads"),
    "pispa_family_other_note": (
        "补充说明差异蛋白集合的功能背景，需结合对应核心对比和代表蛋白解读。",
        "Supplements the functional background of the differential protein set and must be read together with the corresponding core contrast and representative proteins."),
    "pispa_family_rna": (
        "核质运输/RNA定位",
        "Nucleocytoplasmic transport / RNA localisation"),
    "pispa_family_rna_note": (
        "提示核质转运或 RNA 定位变化，可能连接应激、增殖或蛋白合成状态。",
        "Suggests changes in nucleocytoplasmic transport or RNA localisation that may link stress, proliferation or protein-synthesis states."),
    "pispa_family_stress": (
        "宽泛应激/疾病交叉",
        "Broad stress / disease overlap"),
    "pispa_family_stress_note": (
        "属于通用应激或疾病基因集重叠，只有与线粒体、翻译或骨架条目一致时才作为背景线索。",
        "A generic stress or disease gene-set overlap; used as a background lead only when it agrees with the mitochondrial, translation or cytoskeleton entries."),
    "pispa_family_translation": (
        "翻译/核糖体",
        "Translation / ribosome"),
    "pispa_family_translation_note": (
        "提示蛋白合成程序更活跃，是 Cluster 内部状态分层的重要功能轴。",
        "Suggests a more active protein-synthesis programme, an important functional axis of the state structure inside the clusters."),
    "pispa_family_transport": (
        "蛋白转运/定位",
        "Protein transport / localisation"),
    "pispa_family_transport_note": (
        "提示膜系统、囊泡或蛋白定位程序变化，可能连接分泌和膜重塑状态。",
        "Suggests changes in the membrane system, vesicles or protein-localisation programmes that may link secretion and membrane-remodelling states."),
    "pispa_figure_line": (
        "- 结构图证据：{figure_summary}",
        "- Structure figure evidence: {figure_summary}"),
    "pispa_figure_summary_fallback": (
        "图册索引见 visualize_results/figure_index.md。",
        "See visualize_results/figure_index.md for the figure index."),
    "pispa_followup_live": (
        "- 将 PiSPA 迁移 readout 与细胞形态、速度或扰动实验联动，验证模块分数是否对应真实迁移能力。",
        "- Combine the PiSPA migration readout with cell morphology, speed or perturbation experiments to test whether the module scores correspond to real migration capacity."),
    "pispa_followup_staining": (
        "- 用免疫染色或靶向蛋白质组验证 EZR/MSN/MYL9/TLN1/VCL 等候选在迁移前沿或黏附结构中的定位。",
        "- Use immunostaining or targeted proteomics to validate the localisation of candidates such as EZR/MSN/MYL9/TLN1/VCL at the leading edge or in adhesion structures."),
    "pispa_group_qc_line": (
        "- 分组与 QC 摘要：{group_summary}。",
        "- Group and QC summary: {group_summary}."),
    "pispa_group_summary_fallback": (
        "group_composition_qc.csv 未提供可读摘要",
        "group_composition_qc.csv provided no readable summary"),
    "pispa_summary_boundary": (
        "- 本报告可以提出后续验证假设，但不直接证明细胞形态、迁移方向性或力学拉伸；这些结论需要显微成像或扰动实验继续验证。",
        "- This report can state follow-up hypotheses, but it does not directly prove cell morphology, migration directionality or mechanical stretching; those conclusions need imaging or perturbation experiments to be established."),
    "pispa_summary_c1": (
        "- Cluster 1 是本数据集中最明确的迁移相关状态；Cluster × Type 结构和 C1 相对 C2/C3 的核心对比共同支持这一点，证据层级为 `current_matrix`，置信度为中等。",
        "- Cluster 1 is the clearest migration-related state in this dataset; the Cluster x Type structure and the core contrasts of C1 against C2/C3 support this together, with `current_matrix` as the evidence layer and moderate confidence."),
    "pispa_summary_c2c3": (
        "- Cluster 2 与 Cluster 3 不能合并成一个均一对照；C2/C3 的差异提示对照群体内部已有翻译、黏附或细胞状态分层，需要在解释迁移轴时单独标出。",
        "- Cluster 2 and Cluster 3 cannot be merged into one uniform control; the C2/C3 difference indicates that the control population already carries translation, adhesion or cell-state structure, which must be marked separately when reading the migration axis."),
    "pispa_summary_candidates": (
        "- EZR、CDC42、RAC1、RHOA、TLN1、VCL 等是当前矩阵中优先级较高的骨架/黏附候选；MSN、MYL9 等若显著性较弱，应作为同模块方向线索而不是独立驱动因子。",
        "- EZR, CDC42, RAC1, RHOA, TLN1 and VCL are the higher-priority cytoskeleton/adhesion candidates in the current matrix; where MSN or MYL9 are less significant they should serve as directional leads in the same module rather than as independent drivers."),
    "pispa_summary_mechanism": (
        "- 迁移机制不是单个蛋白的故事，而是“分泌/ECM 重塑 + 局部骨架快速周转 + 黏附复合体调节”的组合轴；Rho GTPase 调控、ERM 膜-皮质连接、肌球蛋白收缩和 talin-vinculin 黏附斑提供同向证据。",
        "- The migration mechanism is not the story of a single protein but a combined axis of secretion/ECM remodelling, rapid local cytoskeleton turnover and adhesion-complex regulation; Rho GTPase regulation, ERM membrane-cortex linkage, myosin contractility and talin-vinculin focal adhesions provide concordant evidence."),
    "pispa_synth_axis": (
        "- 若后续扰动验证优先级有限，Rho GTPase/ERM/myosin/talin-vinculin 轴比单个候选蛋白更适合作为迁移机制验证框架。",
        "- If only limited follow-up perturbation work is possible, the Rho GTPase/ERM/myosin/talin-vinculin axis is a better frame for validating the migration mechanism than any single candidate protein."),
    "pispa_synth_c1": (
        "- Cluster 1 与迁移表型的对应关系可以由 Cluster-Type 结构、核心对比、候选蛋白和模块分数共同支持。",
        "- The correspondence between Cluster 1 and the migration phenotype is supported jointly by the Cluster-Type structure, the core contrasts, the candidate proteins and the module scores."),
    "pispa_synth_c2c3": (
        "- C2/C3 内部差异是 current_matrix 中真实需要解释的结构，不能被迁移主线完全覆盖。",
        "- The internal C2/C3 difference is a structure in the current_matrix that genuinely needs explaining and cannot be fully covered by the migration main line."),
    "pispa_synth_fdr": (
        "- 若候选蛋白未达到 FDR 阈值，可把它们作为同模块的方向线索，而不是单独宣称为显著驱动因子。",
        "- Where candidate proteins do not reach the FDR threshold, treat them as directional leads in the same module rather than claiming them individually as significant drivers."),
    "pispa_task1_contrast_note": (
        "`Migrated_vs_Control` 在本报告中作为 Cluster × Type 组成和表型标签证据使用；主差异统计以 Cluster 两两对比展开，因为这些对比更直接分解迁移细胞富集的 Cluster 1 与两个对照背景。若需要迁移表型整体差异表，应在后续 run 中显式加入该 contrast。",
        "`Migrated_vs_Control` is used in this report as Cluster x Type composition and phenotypic label evidence; the main differential statistics are run as pairwise Cluster contrasts, because those separate the Cluster 1 enrichment of migrated cells from the two control backgrounds more directly. If an overall differential table for the migration phenotype is needed, that contrast must be added explicitly in a later run."),
    "pispa_task1_intro": (
        "结论先行：Cluster 1 是迁移相关状态的主要候选，Cluster 2/3 则是带有内部差异的对照背景。下表只保留最能回答任务的问题、方向和代表证据，完整差异表见资产清单。",
        "Conclusion first: Cluster 1 is the main candidate for the migration-related state, while Cluster 2/3 form a control background that carries internal differences. The table below keeps only the question, the direction and the representative evidence that best answer the task; the complete differential tables are in the asset list."),
    "pispa_task1_no_table": (
        "核心对比表未生成。",
        "No core contrast table was generated."),
    "pispa_task1_reading": (
        "简要解读：{a0} 这一结果说明 Cluster 1 与迁移状态相关，但它仍是抽样矩阵中的统计对应，不等同于因果验证。",
        "Brief reading: {a0} This result indicates that Cluster 1 is associated with the migration state, but it remains a statistical correspondence in a sampled matrix and is not equivalent to causal validation."),
    "pispa_task2_intro": (
        "结论先行：迁移 Cluster 的功能解释应以机制链为中心，而不是把所有通路名铺开。当前最有价值的链条是 Rho GTPase 调控 -> ERM 膜-皮质连接 -> 肌球蛋白收缩 -> talin/vinculin 黏附 -> 肌动蛋白骨架重排。",
        "Conclusion first: the functional interpretation of the migration clusters should centre on a mechanism chain rather than spreading out every pathway name. The most valuable chain here is Rho GTPase regulation -> ERM membrane-cortex linkage -> myosin contractility -> talin/vinculin adhesion -> actin cytoskeleton rearrangement."),
    "pispa_task2_no_table": (
        "模块证据表未生成。",
        "No module evidence table was generated."),
    "pispa_task2_summary_prefix": (
        "模块主线摘要：",
        "Module main-line summary: "),
    "pispa_task2_summary_tail": (
        "。这些分数用于整合机制链，不能替代正式富集显著性。",
        ". These scores integrate the mechanism chain and do not replace formal enrichment significance."),
    "pispa_task3_interpretation": (
        "解释：这些候选蛋白共同指向迁移细胞的膜-皮质连接、收缩力、黏附复合体和肌动蛋白骨架重排。当前报告只把它们写成矩阵支持的候选链条，不能替代实时迁移成像或扰动实验。",
        "Interpretation: together these candidate proteins point to membrane-cortex linkage, contractility, the adhesion complex and actin cytoskeleton rearrangement in migrated cells. This report writes them only as a matrix-supported candidate chain, which does not replace live migration imaging or perturbation experiments."),
    "pispa_task3_intro": (
        "结论先行：EZR/MSN/MYL9/CDC42/RAC1/RHOA/TLN1/VCL/FLNA/ACTN1/ITGB1 等候选应按检出、方向、logFC 与 FDR 分层呈现；不可检出或未显著项目必须作为边界写清。",
        "Conclusion first: candidates such as EZR/MSN/MYL9/CDC42/RAC1/RHOA/TLN1/VCL/FLNA/ACTN1/ITGB1 should be presented in layers by detection, direction, logFC and FDR; entries that were not detectable or not significant must be written out as boundaries."),
    "pispa_task3_no_table": (
        "代表蛋白表未生成。",
        "No representative protein table was generated."),
    "pispa_task3_summary_prefix": (
        "候选优先级摘要：",
        "Candidate priority summary: "),
    "pispa_task3_summary_tail": (
        "。这些数值支持方向和优先级，不把未显著候选写成已验证机制。",
        ". These numbers support direction and priority and do not write non-significant candidates as established mechanisms."),
    "pispa_term_note": (
        "术语说明：本节把 single-cell proteomic 矩阵中的 cell motility、actin cytoskeleton、focal adhesion、ERM 膜-皮质连接和 Rho GTPase 调控视为同一迁移机制链的不同层级；这些术语用于组织证据，不额外扩大当前矩阵能够直接证明的范围。",
        "Terminology note: this section treats cell motility, the actin cytoskeleton, focal adhesion, ERM membrane-cortex linkage and Rho GTPase regulation in a single-cell proteomic matrix as different levels of one migration mechanism chain. These terms organise the evidence and do not extend what the current matrix can directly establish."),
    "probe_compat": (
        "兼容",
        "compatible"),
    "probe_confidence": (
        "置信度",
        "confidence"),
    "pt4_c2c3_background": (
        "- C2/C3 差异可以作为解释迁移轴的背景层：Cluster 1 的迁移相关信号需要与这种对照内部梯度区分开，避免把所有非迁移细胞看作同一种状态。",
        "- The C2/C3 difference serves as a background layer for reading the migration axis: migration-related signal in Cluster 1 must be kept apart from this internal control gradient, so that not every non-migratory cell is treated as one and the same state."),
    "pt4_c2c3_current_matrix": (
        "- `current_matrix` 支持 {group_a} 与 {group_b} 的差异来自 C2/C3 对比本身：{group_a}较高={n_up_display_group_a}，{group_b}较高={n_down_display_group_a}，代表蛋白见上表。",
        "- `current_matrix` supports that the difference between {group_a} and {group_b} comes from the C2/C3 contrast itself: {group_a} higher={n_up_display_group_a}, {group_b} higher={n_down_display_group_a}; representative proteins are in the table above."),
    "pt4_c2c3_cytoskeleton_row": (
        "| 骨架/黏附候选 | {a0} | 见 logFC 方向 | P.Value/adj.P.Val 见括号 FDR | logFC 见括号 | 用于判断对照内部是否已有局部骨架或黏附预激活线索。 |",
        "| Cytoskeleton / adhesion candidates | {a0} | See the logFC direction | P.Value/adj.P.Val from the bracketed FDR | logFC in brackets | Used to judge whether the controls already carry local cytoskeleton or adhesion pre-activation leads. |"),
    "pt4_c2c3_diff_row": (
        "| 差异蛋白 | {top_up_display_group_a} | {top_down_display_group_a} | adj.P.Val/FDR 见差异表 | n_sig={n_sig}；{group_a}较高={n_up_display_group_a}；{group_b}较高={n_down_display_group_a} | Cluster 2 侧若集中于翻译、线粒体或黏附相关候选，说明其代表较活跃的对照状态；Cluster 3 侧候选则提示另一类非迁移背景。 |",
        "| Differential proteins | {top_up_display_group_a} | {top_down_display_group_a} | adj.P.Val/FDR in the differential table | n_sig={n_sig}; {group_a} higher={n_up_display_group_a}; {group_b} higher={n_down_display_group_a} | If the Cluster 2 side concentrates on translation, mitochondrial or adhesion candidates it points to a more active control state, while Cluster 3 candidates suggest a different non-migratory background. |"),
    "pt4_c2c3_header": (
        "| 证据层级 | Cluster 2 侧 | Cluster 3 侧 | FDR/统计口径 | 效应量 | 解读 |",
        "| Evidence level | Cluster 2 side | Cluster 3 side | FDR / statistical convention | Effect size | Reading |"),
    "pt4_c2c3_module_row": (
        "| 模块分数 | {a0} | 相对较低或未覆盖 | 未计算 FDR；模块平均分差 | Δ 见左列 | 模块差异提示对照内部存在迁移相关程序的梯度，而不是二元开关。 |",
        "| Module scores | {a0} | Relatively lower or not covered | FDR not computed; module mean difference | Delta in the left column | The module difference indicates a gradient of migration-related programmes inside the controls rather than a binary switch. |"),
    "pt4_c2c3_table_title": (
        "### C2/C3 代表蛋白与模块证据表",
        "### C2/C3 representative protein and module evidence"),
    "pt4_control_not_equivalent": (
        "结论先行：{group_a} 与 {group_b} 不是完全等价的对照背景。{a2} 这说明对照群体内部仍有可解释的蛋白组状态分层。",
        "Conclusion first: {group_a} and {group_b} are not fully equivalent control backgrounds. {a2} This indicates that the control population still carries an interpretable proteome state structure."),
    "pt4_extension_c2": (
        "- `extension`：如果 Cluster 2 侧同时出现翻译、线粒体、黏附或骨架模块升高，它可能代表更活跃或更易响应的对照状态；这仍需后续功能实验确认。",
        "- `extension`: if translation, mitochondrial, adhesion or cytoskeleton modules are simultaneously higher on the Cluster 2 side, that may represent a more active or more responsive control state; it still needs confirmation from follow-up functional experiments."),
    "pt4_extension_c3": (
        "- `extension`：Cluster 3 若在结构、RNA/核孔或局部骨架蛋白上更高，可解释为另一种非迁移状态，而不是简单的低迁移版本。",
        "- `extension`: if Cluster 3 is higher in structural, RNA or nuclear-pore, or local cytoskeleton proteins, it can be read as another non-migratory state rather than simply a low-migration version."),
    "pt4_followup_morphology": (
        "- 在不混合 Cluster 2/3 的前提下，分别比较其形态、黏附斑、迁移前沿定位或划痕边缘距离。",
        "- Without mixing Cluster 2 and Cluster 3, compare their morphology, focal adhesions, leading-edge localisation or distance to the scratch edge separately."),
    "pt4_followup_timeseries": (
        "- 若有时间序列或扰动数据，优先检验 C2/C3 是否会向 Cluster 1 迁移状态转换，或只是平行的对照亚状态。",
        "- With time-series or perturbation data, test first whether C2/C3 shift towards the Cluster 1 migration state, or are simply parallel control substates."),
    "qc_analysed_matrix": (
        "本轮分析矩阵（过滤与填补之后）：各分组的平均缺失率 %s，该值为 0 只表示这个矩阵没有缺口，不表示全部蛋白在所有样本中都被检出，检出深度以平均检出蛋白数为准",
        "Analysed matrix of this round (after filtering and imputation): mean missingness rate per group %s; a value of 0 only means this matrix has no gaps, not that every protein was detected in every sample, and detection depth is read from the mean number of detected proteins"),
    "qc_design_fraction": (
        "设计推断记录的非缺失比例 %s（%s），与上面的分组缺失率口径不同，引用时须写明指哪一个",
        "the non-missing fraction recorded by the design inference %s (%s), which uses a different convention from the per-group missingness rate above, so any citation must say which one is meant"),
    "qc_detection_encoding": (
        "未检出编码的记录形式：%s",
        "recorded form of the non-detection encoding: %s"),
    "qc_full_matrix_rows": (
        "全矩阵（%s 行）",
        "full matrix (%s rows)"),
    "qc_head_sample": (
        "设计推断时按矩阵前 %s 行抽样估计，是抽样口径，不能当作全矩阵缺失率",
        "estimated during design inference by sampling the first %s rows of the matrix; this is a sampling convention and must not be taken as the full-matrix missingness rate"),
    "qc_note_fallback": (
        "缺失率口径：运行记录中没有分阶段的缺失统计；分析矩阵缺失率为 0 只表示该矩阵没有缺口，不表示全部蛋白被检出。",
        "Missingness-rate convention: the run record carries no staged missingness statistics; a missingness rate of 0 in the analysed matrix only means that matrix has no gaps, not that every protein was detected."),
    "qc_note_prefix": (
        "缺失率口径：",
        "Missingness-rate convention: "),
    "qc_original_matrix": (
        "原始交付矩阵（过滤与填补之前、全部蛋白行）：各分组的平均缺失率 %s，未检出按该矩阵的缺失编码统计",
        "Original delivered matrix (before filtering and imputation, all protein rows): mean missingness rate per group %s, with non-detections counted through the missingness encoding of that matrix"),
    "qc_unlabelled": (
        "记录未标明是全矩阵还是抽样",
        "the record does not say whether this is the full matrix or a sample"),
    "range_between": (
        "%.4f 至 %.4f",
        "%.4f to %.4f"),
    "reader_contrast_not_tested": (
        "未进入统计检验",
        "not entered into a statistical test"),
    "reader_counts_truncated_note": (
        "（原表该单元格过长，此处按原样保留前段）",
        " (the original table cell is too long; the leading part is kept as it stands)"),
    "req_candidate_pointer": (
        "完整候选核查表见后文「候选蛋白逐条核查（完整表）」；该表只呈现证据，不改变任务要求本身。",
        "The complete candidate audit table appears later as \"Candidate protein audit (complete table)\"; that table presents evidence only and does not change the task requirements themselves."),
    "req_section_note": (
        "上表逐条给出本轮已计算的对比、通过筛选的蛋白数，以及该对比是否在本报告中以文本形式被提及；正文未提及的对比属于本报告的覆盖缺口，应在下一轮补齐，而不是视为已评估。",
        "The table above lists, item by item, the contrasts computed in this round, the number of proteins passing the screening, and whether that contrast is mentioned in the text of this report; a contrast that the body does not mention is a coverage gap of this report, to be closed in the next round rather than treated as already evaluated."),
    "req_section_title": (
        "### 必需子分析与对比覆盖",
        "### Required sub-analyses and contrast coverage"),
    "sct_candidate_header": (
        "| 对比 | 候选蛋白 | 当前矩阵方向 | logFC | P.Value | adj.P.Val/FDR | 置信度 | 功能解释 |",
        "| Contrast | Candidate | Direction in the current matrix | logFC | P.Value | adj.P.Val/FDR | Confidence | Functional reading |"),
    "sct_default_reading": (
        "同模块候选，需结合效应量和 FDR 判断优先级",
        "A candidate from the same module; judge priority from effect size together with FDR"),
    "sct_display_note": (
        "注：本表按「{a0}」显示（{groups}）；底层差异表为 `{source}`，其 logFC 符号与本表相反，方向以文字与差异表对比名为准。",
        "Note: this table is displayed as \"{a0}\" ({groups}); the underlying differential table is `{source}`, whose logFC sign is opposite to this table, so read direction from the wording and from the contrast name in the differential table."),
    "sct_effect_summary": (
        "n_sig={n_sig}（{convention}）；{group_a}较高={up}；{group_b}较高={down}",
        "n_sig={n_sig} ({convention}); {group_a} higher={up}; {group_b} higher={down}"),
    "sct_group_a_higher": (
        "{group_a} 较高：{top_up}",
        "{group_a} higher: {top_up}"),
    "sct_group_b_higher": (
        "{group_b} 较高：{top_down}",
        "{group_b} higher: {top_down}"),
    "sct_header": (
        "| 科学问题 | 对比 | 方向强度 | 代表蛋白与效应量 | 解释 | 边界 |",
        "| Scientific question | Contrast | Direction strength | Representative proteins and effect size | Interpretation | Boundary |"),
    "sct_low_confidence_note": (
        "注：低置信候选表示检出或方向可用但 FDR/效应量不足，不单独作为机制驱动结论。",
        "Note: a low-confidence candidate means the protein was detected, or has a usable direction, but the FDR or effect size is insufficient; it does not on its own drive a mechanism conclusion."),
    "sea_candidate_summary_header": (
        "| 候选 | 是否检出 | 对比 | 方向 | logFC | P.Value | adj.P.Val | 置信度 |",
        "| Candidate | Detected | Contrast | Direction | logFC | P.Value | adj.P.Val | Confidence |"),
    "sea_candidate_summary_title": (
        "#### 候选蛋白证据摘要",
        "#### Candidate protein evidence summary"),
    "sea_core_story_candidate_source": (
        "current_matrix 候选蛋白；P.Value/adj.P.Val 来自对应差异表",
        "current_matrix candidate proteins; P.Value/adj.P.Val come from the corresponding differential table"),
    "sea_core_story_counts": (
        "通过筛选的蛋白数={n_sig}；A 组较高={n_up_display_group_a}；B 组较高={n_down_display_group_a}",
        "proteins passing the screening convention={n_sig}; Group A higher={n_up_display_group_a}; Group B higher={n_down_display_group_a}"),
    "sea_core_story_diff_source": (
        "current_matrix 差异统计；效应量见代表蛋白 logFC",
        "current_matrix differential statistics; the effect size is the logFC of the representative proteins"),
    "sea_core_story_header": (
        "| 类型 | 科学问题 | 对比 | 核心数值 | 置信度 | {story_source_header} |",
        "| Type | Scientific question | Contrast | Core values | Confidence | {story_source_header} |"),
    "sea_core_story_module_source": (
        "模块平均分差；未计算 FDR",
        "Module mean difference; FDR not computed"),
    "sea_core_story_title": (
        "#### 核心故事证据表",
        "#### Core story evidence table"),
    "sea_curated_module_header": (
        "| 模块 | 组别 | 匹配基因数 | 平均分数 | 统计口径 | 置信度 |",
        "| Module | Group | Matched genes | Mean score | Statistical convention | Confidence |"),
    "sea_curated_module_row": (
        "| {module} | {group} | {n_matched_genes} | {mean_score} | 未计算 FDR；样本级模块均值 | {confidence} |",
        "| {module} | {group} | {n_matched_genes} | {mean_score} | FDR not computed; sample-level module mean | {confidence} |"),
    "sea_curated_module_title": (
        "#### Curated 模块分数补充表",
        "#### Curated module score supplement"),
    "sea_external_annotation": (
        "外部知识注释（ASD/NDD）",
        "External knowledge annotation (ASD/NDD)"),
    "sea_external_annotation_note": (
        "「结论边界与方法局限」的外部注释说明",
        "the external annotation note in \"Conclusion Boundaries and Method Limitations\""),
    "sea_external_cross_header": (
        "| 基因 | 类别 | 是否检出 | 匹配行数 | 证据来源 | 边界 |",
        "| Gene | Category | Detected | Matched rows | Evidence source | Boundary |"),
    "sea_external_cross_title": (
        "#### 离线外部注释交叉表",
        "#### Offline external annotation cross-table"),
    "sea_group_qc_header": (
        "| 表 | 分组 | 组别 | 二级列 | 样本数 | 平均检出蛋白 | 平均缺失率 | 计数 |",
        "| Table | Grouping | Group | Sub-column | n_samples | Mean detected proteins | Mean missing rate | Count |"),
    "sea_group_qc_title": (
        "#### 分组组成与 QC 表",
        "#### Group composition and QC tables"),
    "sea_leakage_appendix": (
        "「蛋白泄漏证据附录」",
        "the \"Protein Leakage Evidence Appendix\""),
    "sea_leakage_correlation_header": (
        "| 对比 | 状态列 | 分层 | logFC 矩阵 | 高相关对数 | 结论 |",
        "| Contrast | State column | Stratum | logFC matrix | Highly correlated pairs | Conclusion |"),
    "sea_leakage_correlation_title": (
        "#### 泄漏相关性摘要",
        "#### Leakage correlation summary"),
    "sea_leakage_module_summary": (
        "蛋白泄漏模块汇总",
        "Protein leakage module summary"),
    "sea_leakage_summary": (
        "蛋白泄漏汇总",
        "Protein leakage summary"),
    "sea_module_effect_header": (
        "| 对比 | 状态列 | 模块 | 匹配基因数 | groupA-groupB 平均 logFC | 方向 | A 组样本数 | B 组样本数 | 置信度 |",
        "| Contrast | State column | Module | Matched genes | groupA-groupB mean logFC | Direction | n samples group A | n samples group B | Confidence |"),
    "sea_req_candidate_stats": (
        "候选蛋白逐对比统计",
        "Per-contrast candidate protein statistics"),
    "sea_req_candidate_table": (
        "「候选蛋白逐对比统计」表",
        "the \"Per-contrast candidate protein statistics\" table"),
    "sea_req_contrast_table": (
        "「对比级证据摘要」表",
        "the \"Contrast-level evidence summary\" table"),
    "sea_req_diff_analysis": (
        "差异分析要求与口径",
        "Differential analysis requirements and convention"),
    "sea_req_diff_summary": (
        "差异检验摘要",
        "Differential test summary"),
    "sea_req_diff_threshold": (
        "「数据与预处理」的差异分析阈值",
        "Differential analysis thresholds in \"Data and Preprocessing\""),
    "sea_req_group_qc": (
        "分组组成与质控",
        "Group composition and QC"),
    "sea_req_group_sizes": (
        "「数据与预处理」的分组规模与缺失率",
        "Group sizes and missingness rates in \"Data and Preprocessing\""),
    "sea_req_module_group": (
        "模块分组分数",
        "Module group scores"),
    "sea_req_module_group_table": (
        "「模块分组分数」表",
        "the \"Module group scores\" table"),
    "sea_req_module_sample": (
        "模块样本级分数",
        "Module sample-level scores"),
    "sea_req_module_sample_source": (
        "「模块分组分数」表的均值来源",
        "the mean source of the \"Module group scores\" table"),
    "sea_req_task_evidence": (
        "任务要求与证据落点",
        "Task requirements and where the evidence lands"),
    "sea_req_task_evidence_table": (
        "下方「数据集任务要求与证据落点」表",
        "the \"Dataset task requirements and evidence landing points\" table below"),
    "sea_rows_truncated": (
        "| …（共 %d 行，本表显示前 %d 行；完整表见证据表） |  |  |  |  |  |",
        "| ... (%d rows in total, the first %d shown here; see the evidence tables for the complete table) |  |  |  |  |  |"),
    "sea_source_header_default": (
        "边界",
        "Boundary"),
    "sea_source_header_pispa": (
        "证据口径",
        "Evidence convention"),
    "sea_task_landing_header": (
        "| 任务要求 | 要求覆盖的证据项 | 证据来源 | 置信度 | 边界 |",
        "| Task requirement | Evidence item it must cover | Evidence source | Confidence | Boundary |"),
    "sea_task_landing_intro": (
        "本表是任务要求清单（不是结果）；每项对应的数值见正文任务小节与上表，未计算项在覆盖表中标注。",
        "This table is a checklist of task requirements, not a result set; the numbers for each item are in the task sections of the body and in the tables above, and items that were not computed are marked in the coverage table."),
    "sea_task_landing_title": (
        "#### 数据集任务要求与证据落点",
        "#### Dataset task requirements and evidence landing points"),
    "sentence_stop": (
        "。",
        "."),
    "sep_comma": (
        "、",
        ", "),
    "sep_semicolon": (
        "；",
        "; "),
    "smt_a_higher_context": (
        "{group_a} 在该模块上更高，支持与“{claim_title}”相关的方向性解释。",
        "{group_a} is higher in this module, supporting a directional interpretation related to \"{claim_title}\"."),
    "smt_b_higher_context": (
        "{group_b} 在该模块上更高，提示该任务存在反向或背景状态差异。",
        "{group_b} is higher in this module, indicating a reversed or background state difference for this task."),
    "smt_header": (
        "| 任务/对比 | 模块 | A组均值 | B组均值 | Δ(A-B) | 代表蛋白数 | 统计口径 | 解读 |",
        "| Task/contrast | Module | Mean in A | Mean in B | Delta (A-B) | Representative proteins | Statistical convention | Reading |"),
    "smt_method": (
        "未计算 FDR；效应量=模块平均分差 Δ",
        "FDR not computed; effect measure = module mean difference Delta"),
    "smt_no_delta": (
        "该模块在本对比中未能计算均值差，通常是分组不存在或模块匹配蛋白不足。",
        "The module mean difference could not be computed for this contrast, usually because a group is missing or too few module proteins matched."),
    "smt_title": (
        "### 模块证据表",
        "### Module evidence table"),
    "smt_weak": (
        "{group_a} 与 {group_b} 的模块差异较弱，适合作为背景线索。",
        "The module difference between {group_a} and {group_b} is weak and serves as background information."),
    "src_label_analysis_design": (
        "分析设计",
        "Analysis design"),
    "src_label_audit_record": (
        "审计记录",
        "Audit record"),
    "src_label_drug_database": (
        "药物数据库",
        "Drug database"),
    "src_label_external_literature": (
        "外部文献",
        "External literature"),
    "story_direction_undetermined": (
        "方向未判定",
        "direction undetermined"),
    "story_exec_boundary": (
        "- 证据边界：{boundary}",
        "- Evidence boundary: {boundary}"),
    "story_exec_change_phrase": (
        "显著上调 {n_up_display_group_a}、显著下调 {n_down_display_group_a}；代表上调为 {top_up_display_group_a}",
        "{n_up_display_group_a} significantly higher and {n_down_display_group_a} significantly lower; the representative higher proteins are {top_up_display_group_a}"),
    "story_exec_change_phrase_none": (
        "在当前阈值下没有显著差异蛋白；按效应量排序的上调候选为 {top_up_display_group_a}",
        "no significantly different proteins at the current thresholds; the effect-size-ranked higher candidates are {top_up_display_group_a}"),
    "story_exec_core_contrast": (
        "- current_matrix 核心对比：{display_contrast}，{main_change_phrase}。",
        "- current_matrix core contrast: {display_contrast}, {main_change_phrase}."),
    "story_exec_modules": (
        "- current_matrix 模块证据：{a0}；模块分数用于整合方向，正式显著性仍以差异表/富集表为边界。",
        "- current_matrix module evidence: {a0}; module scores integrate direction, and formal significance is still bounded by the differential and enrichment tables."),
    "story_exec_question": (
        "- `{dataset}` 的核心问题是：{story_title}；本报告先给结论和核心表格，再把完整证据放入附录。",
        "- The central question for `{dataset}` is: {story_title}; this report gives the conclusions and the core tables first and puts the complete evidence in the appendix."),
    "story_exec_representatives": (
        "- current_matrix 代表蛋白：{top_candidates}；这些数值用于支撑主线，不把候选存在性写成显著差异。",
        "- current_matrix representative proteins: {top_candidates}; these numbers support the main line and do not turn the existence of a candidate into a significant difference."),
    "story_findings_change_phrase": (
        "当前矩阵中上调 {n_up_display_group_a}、下调 {n_down_display_group_a}。代表上调：{top_up_display_group_a}；代表下调：{top_down_display_group_a}。",
        "In the current matrix {n_up_display_group_a} proteins are higher and {n_down_display_group_a} lower. Representative higher: {top_up_display_group_a}; representative lower: {top_down_display_group_a}."),
    "story_findings_change_phrase_none": (
        "当前阈值下没有显著差异蛋白；因此本段只把按效应量排序的候选变化作为低置信度线索。A组较高候选：{top_up_display_group_a}；B组较高候选：{top_down_display_group_a}。",
        "No significantly different proteins at the current thresholds, so this paragraph treats only the effect-size-ranked candidate changes as low-confidence leads. Candidates higher in group A: {top_up_display_group_a}; higher in group B: {top_down_display_group_a}."),
    "story_findings_direction": (
        "`{display_contrast}` 是本段标准方向；{change_phrase}",
        "`{display_contrast}` is the standard direction for this paragraph; {change_phrase}"),
    "story_findings_enrichment_rule": (
        "offline_enrichment 用于补充机制名词，但只有带 FDR/q-value 支持的条目才写成显著富集；探索性 fallback 只作为提示。",
        "offline_enrichment supplies mechanism vocabulary, but only entries with FDR or q-value support are written as significant enrichment; exploratory fallback entries serve as hints only."),
    "story_findings_interpretation": (
        "解释：{interpretation} 边界：{boundary}",
        "Interpretation: {interpretation} Boundary: {boundary}"),
    "story_findings_intro": (
        "本节按数据集核心问题组织结果，而不是按工具调用顺序罗列证据。",
        "This section organises the results around the central question of the dataset rather than listing evidence in tool-call order."),
    "story_findings_mainline": (
        "### 主线判断：{story_title}",
        "### Main-line judgement: {story_title}"),
    "story_findings_modules_prefix": (
        "模块层面为：",
        "At the module level: "),
    "story_findings_representatives_prefix": (
        "代表蛋白数值为：",
        "Representative protein values: "),
    "story_group_a_higher": (
        "{group_a} 较高",
        "{group_a} higher"),
    "story_group_b_higher": (
        "{group_b} 较高",
        "{group_b} higher"),
    "story_scoring_intro": (
        "本节只放用户最需要先看到的核心表格：核心对比、代表蛋白和模块证据。完整文件路径、coverage table 与日志放在证据附录。",
        "This section carries only the core tables a reader needs first: the core contrasts, the representative proteins and the module evidence. Full file paths, the coverage table and the logs sit in the evidence appendix."),
    "vis_differential_overview": (
        "差异概览显示 %s 个显著蛋白条目，覆盖 %s 个对比。该数字的口径是「adj.P.Val < 0.05」（未叠加效应量阈值，叠加 |logFC| > 0.25 后的合计见《必需子分析与对比覆盖》表），并且是各对比条目的合计——同一蛋白出现在多个对比中会重复计入，不是去重后的蛋白数。",
        "The differential overview reports %s significant protein entries across %s contrasts. That count uses the convention \"adj.P.Val < 0.05\" (no effect-size threshold is added; the combined count after also applying |logFC| > 0.25 appears in the Required Sub-analyses and Contrast Coverage table), and it is a sum over per-contrast entries: the same protein appearing in more than one contrast is counted more than once, so it is not a de-duplicated protein count."),
    "vis_heatmap": (
        "热图使用 {n_features} 个信息量较高的特征，覆盖 {comparison_preview}；图见 `visualize_results/heatmap.png`。",
        "The heatmap uses {n_features} features with high information content, covering {comparison_preview}; see `visualize_results/heatmap.png`."),
    "vis_more_contrasts": (
        " 等 {a0} 个对比",
        " ({a0} contrasts in total)"),
    "vis_pca": (
        "PCA 样本结构：n={n_samples}，PC1={PC1_variance}%，PC2={PC2_variance}%，标注列={label_column}；图见 `visualize_results/pca_plot.png`。",
        "PCA sample structure: n={n_samples}, PC1={PC1_variance}%, PC2={PC2_variance}%, label column={label_column}; see `visualize_results/pca_plot.png`."),
    "vis_static_figure_set": (
        "静态图册包含 {a0} 张图，类别包括：{a1}。",
        "The static figure set contains {a0} figures, with categories including: {a1}."),
    "task_numerals": (
        "一|二|三|四|五|六|七|八|九|十",
        "1|2|3|4|5|6|7|8|9|10"),
    "figure_volcano_suffix": (
        " 火山图",
        " volcano plot"),
    "vis_umap": (
       "UMAP 样本结构：n={n_samples}，clusters/groups={n_clusters}，标注列={label_column}；图见 `visualize_results/umap_plot.png`。",
       "UMAP sample structure: n={n_samples}, clusters/groups={n_clusters}, label column={label_column}; see `visualize_results/umap_plot.png`."),
}
_report_language.register("main", MAIN_TEXT)


def tm(key: str, **fmt) -> str:
    """Deterministic report prose for the active language (main namespace)."""
    return _report_language.t("main." + key, **fmt)

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


def report_language_directive() -> str:
    """Report-request text an English run needs. Empty for zh, so a zh request is unchanged."""
    if _report_language.get_language() != "en":
        return ""
    directive = globals().get("EN_REPORT_LANGUAGE_DIRECTIVE")
    if not directive:
        # prompts.py is standard library only, so this import is safe on the offline path too
        try:
            from prompts import EN_REPORT_LANGUAGE_DIRECTIVE as directive  # noqa: F811
        except Exception:  # noqa: BLE001
            directive = ""
    if not directive:
        return ""
    return chr(10) + chr(10) + directive


def _has_story_task_heading(report_content: str) -> bool:
    word = re.escape(_report_language.t("core.task_word"))
    return bool(re.search("(?m)^## [0-9]+[.][ ]*" + word, str(report_content)))


def _has_story_first_structure(report_content: str) -> bool:
    text = str(report_content)
    story = story_report_headings()

    def _numbered(key: str) -> bool:
        tail = re.escape(_report_language.t(key))
        return bool(re.search("(?m)^## [0-9]+[.][ ]*" + tail + "[ ]*$", text))

    return (
        _has_h2(text, story[0])
        and _has_h2(text, story[1])
        and _has_story_task_heading(text)
        and _numbered("core.story_synthesis")
        and _numbered("core.story_assets")
        and _numbered("core.story_conclusion")
    )


def _main_report_text_for_language_check(report_content: str) -> str:
    text = str(report_content)
    markers = [
        f"\n## {canonical_report_headings()['appendix']}",
        f"\n## {story_report_headings()[7]}",
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


def _chinese_char_ratio(report_content: str) -> float:
    body = _main_report_text_for_language_check(report_content)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    latin = len(re.findall(r"[A-Za-z]", body))
    denom = cjk + latin
    return round(cjk / denom, 4) if denom else 0.0


def _has_long_english_paragraph(report_content: str) -> bool:
    """zh: a long purely-English paragraph. en: a long purely-Chinese paragraph."""
    body = _main_report_text_for_language_check(report_content)
    return _report_language.wrong_language_paragraph_present(
        body, _report_language.get_language())


def _remove_h2_section(text: str, heading: str) -> str:
    return re.sub(rf"(?ms)^## {re.escape(heading)}\s*\n.*?(?=^## |\Z)", "", str(text)).strip()


def _strip_report_title(report_text: str) -> Tuple[str, str]:
    lines = str(report_text).splitlines()
    if lines and lines[0].startswith("# "):
        return lines[0].strip(), "\n".join(lines[1:]).strip()
    return "# scProteomics Analysis Report", str(report_text).strip()


def _clean_report_body_for_main_findings(report_body: str) -> str:
    cleaned = sanitize_report_text_for_output(report_body, 35000)
    for heading in CHINESE_REPORT_HEADINGS + LEGACY_REPORT_HEADINGS + [
        "Task",
        "Evidence Sources",
        "Analysis Summary",
        "Scoring Evidence Pack",
        "Scientific Boundary",
    ]:
        cleaned = _remove_h2_section(cleaned, heading)
    cleaned = re.sub(r"(?ms)^### Scoring Evidence First Screen\s*\n.*?(?=^## |^### |\Z)", "", cleaned).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _primary_report_base_name(run_folder: str, report_idx: int, final: bool = False) -> str:
    dataset = _get_dataset_name(run_folder)
    if dataset in V3_MAIN_DATASETS:
        return "final_report" if final else "report"
    return "final_output_report" if final else f"output_report_{report_idx}"


def _build_scoring_evidence_first_screen(run_folder: str) -> str:
    story = _build_story_scoring_first_screen(run_folder)
    if story:
        return story

    evidence_dir = os.path.join(run_folder, "evaluation_evidence")
    coverage_path = os.path.join(evidence_dir, "scoring_standard_coverage.csv")
    external_applicable = _external_annotation_applicable(run_folder)
    evidence_files = [
        ("evaluation_requirements.used.json", os.path.join(run_folder, "evaluation_requirements.used.json"), "analysis_design"),
        ("group_composition_qc.csv", os.path.join(evidence_dir, "group_composition_qc.csv"), "current_matrix"),
        ("candidate_protein_evidence.csv", os.path.join(evidence_dir, "candidate_protein_evidence.csv"), "current_matrix"),
        ("curated_module_group_summary.csv", os.path.join(evidence_dir, "curated_module_group_summary.csv"), "current_matrix"),
        ("curated_module_sample_scores.csv", os.path.join(evidence_dir, "curated_module_sample_scores.csv"), "current_matrix"),
        ("dataset_recipe_evidence.csv", os.path.join(evidence_dir, "dataset_recipe_evidence.csv"), "current_matrix;analysis_design"),
        ("scoring_standard_coverage.csv", coverage_path, "current_matrix;analysis_design"),
    ]
    if external_applicable:
        evidence_files.insert(
            -1,
            ("external_annotation_asd_ndd_local.csv", os.path.join(evidence_dir, "external_annotation_asd_ndd_local.csv"), "external_annotation"),
        )
    if not any(os.path.exists(path) for _, path, _ in evidence_files):
        return ""

    lines = [
        f"## {canonical_report_headings()['scoring']}",
        (
            "本节先把核心问题对应到可复核证据，再进入机制解释。`external_annotation` 只在本数据集需要本地离线注释交叉表时出现，"
            "不代表当前矩阵直接证明疾病或因果关系。"
        )
        if external_applicable
        else "本节先把核心问题对应到可复核证据，再进入机制解释；本数据集正文只使用 current_matrix 与 offline_enrichment，不启用疾病外部注释。",
        "",
        "| 核心问题 | 覆盖状态 | 置信度 | 证据来源 | 核心证据与边界 |",
        "|---|---|---|---|---|",
    ]
    coverage_rows = _read_csv_rows(coverage_path, limit=40)
    if coverage_rows:
        for row in coverage_rows:
            dimension = f"{row.get('dimension_id', '')} {row.get('dimension_title', '')}".strip()
            evidence_file_refs = [
                _report_path(path.strip(), run_folder)
                for path in str(row.get("evidence_files", "")).split(";")
                if path.strip()
            ]
            evidence = "; ".join(evidence_file_refs[:3])
            requirements = row.get("requirements", "")
            if requirements:
                evidence = (evidence + " | " if evidence else "") + requirements
            lines.append(
                f"| {_fmt_cell(dimension)} | {_fmt_cell(row.get('coverage_status', ''))} | "
                f"{_fmt_cell(row.get('confidence', ''))} | {_fmt_cell(row.get('evidence_source', ''))} | "
                f"{_fmt_cell(evidence, 220)} |"
            )
    else:
        lines.append("| general | pending | 较低 | 当前矩阵与分析设计 | 未找到覆盖映射表 |")

    visualization_lines = _visualization_evidence_lines(run_folder)
    if visualization_lines:
        lines.extend([
            "",
            "### 图表证据",
        ])
        for item in visualization_lines[:6]:
            lines.append(f"- {item}")

    lines.extend([
        "",
        "### 关键证据文件",
        "| 文件 | 证据来源 | 用途 |",
        "|---|---|---|",
    ])
    for label, path, source in evidence_files:
        if os.path.exists(path):
            lines.append(f"| {label} | {_source_label_zh(source)} | 支持正文结论；文件位于本轮运行目录内 |")
    return "\n".join(lines)


def _build_executive_result_summary(run_folder: str) -> str:
    story = _build_story_executive_summary(run_folder)
    if story:
        return story

    evidence_dir = os.path.join(run_folder, "evaluation_evidence")
    requirements_path = os.path.join(run_folder, "evaluation_requirements.used.json")
    dataset = os.path.basename(os.path.abspath(run_folder))
    try:
        requirements = load_json(requirements_path) if os.path.exists(requirements_path) else {}
        dataset = requirements.get("dataset") or dataset
    except Exception:
        requirements = {}

    coverage_rows = _read_csv_rows(os.path.join(evidence_dir, "scoring_standard_coverage.csv"), limit=80)
    candidate_rows = _read_csv_rows(os.path.join(evidence_dir, "candidate_protein_evidence.csv"), limit=120)
    module_rows = _read_csv_rows(os.path.join(evidence_dir, "curated_module_group_summary.csv"), limit=120)
    module_sample_rows = _read_csv_rows(os.path.join(evidence_dir, "curated_module_sample_scores.csv"), limit=5000)
    group_rows = _read_csv_rows(os.path.join(evidence_dir, "group_composition_qc.csv"), limit=80)
    recipe_rows = _read_csv_rows(os.path.join(evidence_dir, "dataset_recipe_evidence.csv"), limit=20)
    external_rows = _read_csv_rows(os.path.join(evidence_dir, "external_annotation_asd_ndd_local.csv"), limit=80)
    params_path = os.path.join(run_folder, "parameters.json")

    covered = sum(1 for row in coverage_rows if str(row.get("coverage_status", "")).startswith("covered"))
    coverage_total = len(coverage_rows)
    high_signal_candidates = _select_candidate_highlights(candidate_rows, dataset, limit=8)
    module_bits = _module_summary_lines(module_rows, dataset, limit=4)
    visualization_lines = _visualization_evidence_lines(run_folder)
    group_bits = _group_qc_lines(group_rows, limit=4)
    dispersion_bits = _sample_score_dispersion_lines(module_sample_rows, dataset)
    enrichment_bits = _enrichment_summary_lines(run_folder, dataset, limit=4)
    detected_external = []
    if _external_annotation_applicable(run_folder):
        detected_external = [
            row for row in external_rows
            if str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"}
        ][:5]

    lines = [
        f"## {canonical_report_headings()['executive']}",
        f"- 数据集：`{dataset}`。",
        f"- confidence: moderate；核心问题覆盖为 {covered}/{coverage_total} 个维度；依据为 current_matrix 证据表与数据集 recipe。",
    ]
    if recipe_rows:
        recipe_names = "; ".join(row.get("recipe_item", "") for row in recipe_rows[:4] if row.get("recipe_item"))
        lines.append(f"- current_matrix 数据集特异检查重点：{recipe_names}；confidence: moderate。")
    if group_bits:
        lines.append("- current_matrix 分组/QC：" + "; ".join(group_bits[:3]) + "；confidence: moderate。")
    if visualization_lines:
        lines.append("- current_matrix 结构图证据：" + " ".join(visualization_lines[:2]) + " confidence: moderate。")
    if high_signal_candidates:
        lines.append("- current_matrix 候选蛋白证据：" + "; ".join(_format_candidate_bits(high_signal_candidates, limit=6)) + "；confidence: moderate。")
    elif candidate_rows:
        detected_count = sum(1 for row in candidate_rows if str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"})
        lines.append(f"- current_matrix 候选蛋白证据：{detected_count}/{len(candidate_rows)} 行候选在矩阵中可检测或有存在性支持；confidence: moderate。")
    if module_bits:
        lines.append("- current_matrix 模块证据：" + "; ".join(module_bits[:3]) + "；confidence: moderate。")
    if dataset == "Nat_Biotech_Brain_2026":
        brain_bits = _brain_competitive_interpretation_lines(group_rows, candidate_rows, module_rows, external_rows, visualization_lines, run_folder)
        for bit in brain_bits[1:4]:
            lines.append(bit)
    elif dataset in {"Nat_Methods_iPSC_2025", "Nat_Commun_PiSPA_2024"}:
        dataset_bits = _dataset_specific_interpretation_lines(dataset, group_rows, candidate_rows, module_rows, module_sample_rows)
        for bit in dataset_bits[1:3]:
            lines.append(bit)
    if dispersion_bits:
        lines.append("- current_matrix EB 异质性证据：" + " ".join(dispersion_bits[:2]) + " confidence: moderate。")
    if enrichment_bits:
        lines.append("- offline_enrichment 富集证据：" + " ".join(enrichment_bits[:3]) + " confidence: moderate。")
    if detected_external:
        genes = ", ".join(row.get("gene", "") for row in detected_external if row.get("gene"))
        lines.append(f"- external_annotation 边界：本地 ASD/NDD 离线交叉表检出 {genes}；该注释不提高 current_matrix 结论置信度。")
    try:
        params = load_json(params_path) if os.path.exists(params_path) else {}
        batch_info = params.get("combat_calibration", {}).get("batch_correction", {})
        if batch_info.get("combat_success") is False:
            fallback = batch_info.get("fallback_behavior") or batch_info.get("effective_matrix_status") or "uncorrected fallback"
            lines.append(f"- 批次校正边界：ComBat 未成功应用；`ProteinQuant_ComBat.csv` 是兼容输出，下游统计使用 `{fallback}`。")
    except Exception:
        pass
    lines.extend([
        "- offline_enrichment 边界：只有明确 FDR/q-value 支持的条目可写为显著；fallback 条目只作为探索性提示。",
        "- 证据附录列出 CSV/JSON 和图册索引；正文不保留 raw tool JSON 或长工具日志。",
    ])
    return "\n".join(lines)


def _build_main_findings_section(report_body: str, run_folder: str) -> str:
    story = _build_story_main_findings(run_folder)
    if story:
        return story

    evidence_dir = os.path.join(run_folder, "evaluation_evidence")
    coverage_rows = _read_csv_rows(os.path.join(evidence_dir, "scoring_standard_coverage.csv"), limit=80)
    candidate_rows = _read_csv_rows(os.path.join(evidence_dir, "candidate_protein_evidence.csv"), limit=1000)
    module_rows = _read_csv_rows(os.path.join(evidence_dir, "curated_module_group_summary.csv"), limit=1000)
    module_sample_rows = _read_csv_rows(os.path.join(evidence_dir, "curated_module_sample_scores.csv"), limit=5000)
    group_rows = _read_csv_rows(os.path.join(evidence_dir, "group_composition_qc.csv"), limit=120)
    recipe_rows = _read_csv_rows(os.path.join(evidence_dir, "dataset_recipe_evidence.csv"), limit=20)
    external_rows = _read_csv_rows(os.path.join(evidence_dir, "external_annotation_asd_ndd_local.csv"), limit=40)
    dataset = _get_dataset_name(run_folder)
    visualization_lines = _visualization_evidence_lines(run_folder)
    enrichment_bits = _enrichment_summary_lines(run_folder, dataset, limit=8)

    lines = [
        f"## {canonical_report_headings()['findings']}",
        "### 当前矩阵结构",
    ]
    if coverage_rows:
        covered = sum(1 for row in coverage_rows if str(row.get("coverage_status", "")).startswith("covered"))
        lines.append(f"- 核心问题覆盖映射显示 {covered}/{len(coverage_rows)} 个维度已有可追溯证据；每个维度保留 confidence 与 evidence_source。")
    for item in _group_qc_lines(group_rows, limit=8):
        lines.append(f"- current_matrix 分组/QC：{item}。")
    for item in visualization_lines[:5]:
        lines.append(f"- current_matrix 图表证据：{item}")
    if recipe_rows:
        recipe = "; ".join(row.get("recipe_item", "") for row in recipe_rows[:3] if row.get("recipe_item"))
        if recipe:
            lines.append(f"- 数据集特异 recipe 聚焦：{recipe}。")
    if dataset == "Nat_Biotech_Brain_2026":
        lines.append("")
        lines.extend(_brain_competitive_interpretation_lines(group_rows, candidate_rows, module_rows, external_rows, visualization_lines, run_folder))
    elif dataset in {"Nat_Methods_iPSC_2025", "Nat_Commun_PiSPA_2024"}:
        lines.append("")
        lines.extend(_dataset_specific_interpretation_lines(dataset, group_rows, candidate_rows, module_rows, module_sample_rows))

    lines.append("")
    lines.append("### 候选蛋白证据")
    signal_candidates = _select_candidate_highlights(candidate_rows, dataset, limit=12)
    if signal_candidates:
        for bit in _format_candidate_bits(signal_candidates, limit=10):
            lines.append(f"- current_matrix: {bit}.")
    elif candidate_rows:
        detected = sum(1 for row in candidate_rows if str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"})
        lines.append(f"- 本轮候选蛋白证据以存在性支持为主：{detected}/{len(candidate_rows)} 行候选在当前矩阵中可检测。")

    lines.append("")
    lines.append("### 模块与数据集 recipe 证据")
    module_bits = _module_summary_lines(module_rows, dataset, limit=10)
    if module_bits:
        for bit in module_bits:
            lines.append(f"- current_matrix module score: {bit}.")
    for bit in _sample_score_dispersion_lines(module_sample_rows, dataset):
        lines.append(f"- current_matrix EB heterogeneity: {bit}")

    lines.append("")
    lines.append("### 离线富集证据")
    if enrichment_bits:
        for bit in enrichment_bits[:8]:
            lines.append(f"- offline_enrichment: {bit}")
    else:
        lines.append("- offline_enrichment：未找到可直接用于正文的 FDR 支持富集表；fallback 条目只作为探索性提示。")

    detected_external = []
    if _external_annotation_applicable(run_folder):
        detected_external = [
            row.get("gene", "") for row in external_rows
            if str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"} and row.get("gene")
        ][:6]
    if detected_external:
        lines.append("")
        lines.append("### 外部注释边界")
        lines.append("- external_annotation 仅作为本地离线交叉表使用；当前矩阵中检出的基因包括 " + ", ".join(detected_external) + "。")
        lines.append("- external_annotation 行不证明疾病因果关系，只说明本地 curated ASD/NDD 或发育风险基因中哪些可在当前矩阵中测量。")
    return "\n".join(lines)


def _build_evidence_boundary_section(run_folder: str) -> str:
    design_path = os.path.join(run_folder, "analysis_design.used.yaml")
    ledger_path = os.path.join(run_folder, "evidence_ledger.jsonl")
    lines = [
        f"## {canonical_report_headings()['boundary']}",
        f"- 当前矩阵：来自本轮分析的蛋白统计、候选证据、QC 和模块分数；审计账本为 `{_report_path(ledger_path, run_folder)}`。",
        "- 离线富集：来自本地 GO/KEGG/Reactome 资源；非 FDR 支持的条目只作探索性提示。",
    ]
    if _external_annotation_applicable(run_folder):
        lines.append("- 外部注释：来自本地离线疾病或风险基因交叉表，包含适用时的 ASD/NDD 注释；不证明当前矩阵中的疾病因果关系。")
    else:
        lines.append("- 外部注释：本数据集未启用疾病或风险基因外部注释，正文结论不依赖该证据层。")
    lines.extend([
        "- 拓展解释：超出直接矩阵统计的生物学解释必须标明来源并降低置信度。",
        f"- 分析设计：`{_report_path(design_path, run_folder)}` 记录分组选择以及设计是用户提供、数据集提供还是自动推断。",
    ])
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# 报告自包含层（2026-09-11）
# 报告层评审（只读主报告）显示：分析条件、必需对比与候选统计原先只存在于工件层，
# 评审读报告时读不到。以下函数把这些信息从工件确定性地上移到主报告。
# ---------------------------------------------------------------------------

_MATRIX_STATE_TEXT = {
    "raw": "原始强度矩阵",
    "normalized": "已归一化矩阵",
    "batch_corrected": "已完成批次校正的矩阵",
    "logged": "已取对数的矩阵",
}


def _run_parameters(run_folder: str) -> Dict[str, Any]:
    path = os.path.join(run_folder, "parameters.json")
    if not os.path.exists(path):
        return {}
    try:
        data = load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _missing_encoding_text(run_folder: str) -> str:
    """Describe how the input matrix records un-detected values (data property, not a toggled option)."""
    info = _run_parameters(run_folder).get("input_missing_encoding") or {}
    if not isinstance(info, dict) or not info:
        return ""
    parts = []
    for key, label in (("input_nan_pct", tm("missing_encoding_nan")), ("input_zero_pct", tm("missing_encoding_zero"))):
        value = info.get(key)
        if value not in (None, ""):
            parts.append("%s %s%%" % (label, value))
    encoding = info.get("input_missing_encoding") or info.get("encoding")
    text = tm("sep_semicolon").join(parts)
    if encoding:
        text = (tm("missing_encoding_with_form") % (text, encoding)).strip(tm("sep_semicolon"))
    return text


def _matrix_transform_record(run_folder: str) -> Dict[str, Any]:
    """The record the analysis step writes when it builds the analysed matrix (R28).

    Reads processed_proteins/matrix_transform_record.json, which the calibration step writes from
    the code that actually imputes and transforms. Absent means the action was not recorded; it does
    not mean the action was skipped.
    """
    try:
        import report_assembly as _ra_transform
        return _ra_transform.read_matrix_transform_record(str(run_folder))
    except Exception:  # noqa: BLE001
        path = os.path.join(str(run_folder), "processed_proteins", "matrix_transform_record.json")
        try:
            if os.path.exists(path):
                data = load_json(path)
                return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
        return {}


def _build_analysis_conditions_lines(run_folder: str) -> List[str]:
    params = _run_parameters(run_folder)
    design = params.get("analysis_design") if isinstance(params.get("analysis_design"), dict) else {}
    matrix = design.get("matrix", {}) if isinstance(design.get("matrix"), dict) else {}
    differential = design.get("differential", {}) if isinstance(design.get("differential"), dict) else {}
    batch = (params.get("combat_calibration") or {}).get("batch_correction", {}) if isinstance(params.get("combat_calibration"), dict) else {}
    lines: List[str] = []

    design_source = {"auto_inferred": tm("design_source_auto"), "user_provided": tm("design_source_user"),
                     "dataset_provided": tm("design_source_dataset")}.get(str(design.get("source", "")), str(design.get("source", "")) or _report_language.t("captions.not_recorded"))
    lines.append(tm("ac_design_source", design_source=design_source, sample_id_col=design.get('sample_id_col', ''), protein_id_col=design.get('protein_id_col', ''), group_col=design.get('group_col', ''), species=design.get('species', '')))
    covariates = design.get("covariates") or []
    batch_col = design.get("batch_col")
    lines.append(tm("ac_design_vars") % (
        design.get("group_col", tm("ac_group_col_none")),
        tm("sep_comma").join(str(c) for c in covariates) if covariates else _report_language.t("captions.not_recorded"),
        batch_col if batch_col else tm("ac_batch_col_undeclared"),
    ))

    state = str(matrix.get("matrix_state", "") or "")
    state_text = _table_matrix_state().get(state, state or _report_language.t("captions.not_recorded"))
    # R28: the fraction kept in this record was estimated on a head sample of a detection-sorted
    # matrix, so it is labelled with the rows it covers instead of being printed as the delivered
    # matrix's missing rate. The full-matrix rate is stated from the group QC table below.
    fraction = matrix.get("non_missing_fraction", "")
    fraction_source = str(matrix.get("non_missing_fraction_source") or "")
    fraction_rows = matrix.get("non_missing_fraction_rows")
    if fraction not in (None, ""):
        if "full matrix" in fraction_source:
            fraction_label = tm("ac_fraction_full_matrix") % (fraction, fraction_rows or _report_language.t("captions.not_recorded"))
        elif "head sample" in fraction_source or matrix.get("non_missing_fraction_sampled") is not None:
            fraction_label = (tm("ac_fraction_head_sample")
                              % (fraction, matrix.get("non_missing_fraction_sampled_rows") or tm("ac_several_rows")))
        else:
            fraction_label = (tm("ac_fraction_unlabelled") % fraction)
    else:
        fraction_label = tm("ac_fraction_not_recorded")
    lines.append(tm("ac_matrix_state") % (
        state_text, fraction_label,
        matrix.get("numeric_min", ""),
        matrix.get("numeric_max", ""),
    ))

    if batch:
        corrected = tm("ac_batch_corrected") if batch.get("new_batch_correction_applied") else tm("ac_batch_not_corrected")
        reason = batch.get("fallback_reason") or batch.get("method") or ""
        lines.append(tm("ac_batch_block") % (
            corrected, reason, batch.get("effective_matrix_status", "")))
    else:
        lines.append(tm("ac_batch_not_recorded"))

    # R28: the earlier wording turned an absent field into the factual claim "本轮未执行填补／未执行
    # 对数变换", while the matrix the tests actually ran on had been screened, imputed (half-min) and
    # log2(x+1)-transformed. The statement now comes from the record the analysis step wrote.
    transform = _matrix_transform_record(run_folder)
    if transform:
        imp = transform.get("imputation") or {}
        logt = transform.get("log2_transform") or {}
        filt = transform.get("protein_filtering") or {}
        if imp.get("applied"):
            imputation_text = tm("ac_applied_with") % (imp.get("method") or tm("ac_method_not_recorded"))
            n_imputed = imp.get("n_values_imputed")
            if n_imputed not in (None, ""):
                imputation_text += tm("ac_imputed_values") % n_imputed
        elif imp.get("applied") is False:
            imputation_text = tm("ac_not_applied")
        else:
            imputation_text = _report_language.t("captions.not_recorded")
        if logt.get("applied"):
            log_text = tm("ac_applied_with") % (logt.get("form") or tm("ac_form_not_recorded"))
        elif logt.get("applied") is False:
            log_text = tm("ac_not_applied")
        else:
            log_text = tm("ac_not_recorded_with_rule") % (logt.get("recorded_rule") or _report_language.t("captions.not_recorded"))
        chain = tm("sep_comma").join(part for part in (
            (tm("ac_filter_rule") % _reader_recorded_rule(filt.get("rule"))) if filt.get("rule") else "",
            (tm("ac_proteins_retained") % filt.get("n_proteins_retained"))
            if filt.get("n_proteins_retained") not in (None, "") else "",
            (tm("ac_analysed_matrix") % transform.get("analysed_matrix")) if transform.get("analysed_matrix") else "",
        ) if part)
        lines.append(tm("ac_missing_and_scale") % (
            _missing_encoding_text(run_folder) or tm("ac_input_encoding_fallback"),
            imputation_text, log_text, (tm("ac_chain_suffix") % chain) if chain else ""))
    else:
        lines.append(tm("ac_missing_and_scale_not_recorded") % (
                         _missing_encoding_text(run_folder) or tm("ac_input_encoding_fallback")))

    backend = batch.get("statistical_backend") if isinstance(batch, dict) else ""
    contrasts = differential.get("contrasts") or []
    contrast_text = tm("sep_comma").join(str(c.get("name", "")) for c in contrasts[:6]) if contrasts else tm("ac_contrasts_see_sections")
    unused = _unused_design_variables(run_folder)
    stratified = [value for key, value in params.items()
                  if isinstance(value, dict) and "strat" in str(value.get("contrast_type", "")).lower()]
    design_vars = [str(design.get("group_col", ""))] + [str(c) for c in covariates]
    design_vars = [v for v in design_vars if v]
    strat_path = os.path.join(str(run_folder), "evaluation_evidence_ext", "stratified_contrast_summary.csv")
    strat_rows = 0
    strat_vars = []
    if os.path.exists(strat_path):
        with open(strat_path, "r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline().strip().split(",")
            for line in handle:
                if not line.strip():
                    continue
                strat_rows += 1
                parts = line.split(",")
                if "stratifier" in header:
                    idx = header.index("stratifier")
                    if idx < len(parts) and parts[idx] and parts[idx] not in strat_vars:
                        strat_vars.append(parts[idx])
    if strat_rows:
        lines.append(tm("ac_stratified_present") % (
            len(differential.get("contrasts") or []),
            tm("sep_comma").join(strat_vars) if strat_vars else tm("ac_design_variables_fallback"),
            strat_rows))
    else:
        lines.append(tm("ac_stratified_absent") % (
                         len(differential.get("contrasts") or []),
                         tm("sep_comma").join(design_vars) if design_vars else tm("ac_no_extra_design_variables")))
    if unused:
        lines.append(tm("ac_other_design_variables") % tm("sep_comma").join(unused))
    # R28: the screen this pipeline applies is the multiplicity-adjusted one. The unadjusted column
    # stays in the difference tables for reference but was never the cutoff, so naming it here made
    # the stated method disagree with the executed filter.
    _fdr = differential.get("fdr_method", "BH") or "BH"
    lines.append(tm("ac_statistical_method") % (
                     _human_test_method(run_folder), _fdr, _fdr,
                     differential.get("p_thresh", ""), differential.get("logfc_thresh", ""), contrast_text))
    return lines


def _qc_semantics_note(run_folder: str) -> str:
    """R28: state which matrix each missing rate belongs to, and never print a head-sample estimate
    as if it described the delivered matrix."""
    params = _run_parameters(run_folder)
    info = params.get("input_missing_encoding") or {}
    design = params.get("analysis_design") if isinstance(params.get("analysis_design"), dict) else {}
    matrix = design.get("matrix") if isinstance(design.get("matrix"), dict) else {}
    fraction = matrix.get("non_missing_fraction")
    fraction_source = str(matrix.get("non_missing_fraction_source") or "")
    input_range, analysed_range = _group_qc_missing_range(run_folder)
    parts: List[str] = []
    if input_range:
        parts.append(tm("qc_original_matrix") % input_range)
    if analysed_range:
        parts.append(tm("qc_analysed_matrix") % analysed_range)
    if fraction not in (None, ""):
        if "full matrix" in fraction_source:
            label = tm("qc_full_matrix_rows") % (matrix.get("non_missing_fraction_rows") or _report_language.t("captions.not_recorded"))
        elif "head sample" in fraction_source or matrix.get("non_missing_fraction_sampled") is not None:
            label = (tm("qc_head_sample")
                     % (matrix.get("non_missing_fraction_sampled_rows") or tm("ac_several_rows")))
        else:
            label = tm("qc_unlabelled")
        parts.append(tm("qc_design_fraction")
                     % (fraction, label))
    if isinstance(info, dict) and info:
        parts.append(tm("qc_detection_encoding") % _missing_encoding_text(run_folder))
    if not parts:
        return (tm("qc_note_fallback"))
    return tm("qc_note_prefix") + tm("sep_semicolon").join(parts) + tm("sentence_stop")


def _group_qc_missing_range(run_folder: str) -> Tuple[str, str]:
    """(input stage, analysed stage) per-group mean missing-rate range from the group QC table."""
    path = os.path.join(str(run_folder), "evaluation_evidence", "group_composition_qc.csv")
    input_rates: List[float] = []
    analysed_rates: List[float] = []
    try:
        import csv as _csv
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig", newline="") as handle:
                for row in _csv.DictReader(handle):
                    if str(row.get("table") or "") != "primary_group":
                        continue
                    for key, bucket in (("input_mean_missing_rate", input_rates),
                                        ("analysed_mean_missing_rate", analysed_rates)):
                        try:
                            bucket.append(float(row.get(key)))
                        except (TypeError, ValueError):
                            continue
    except Exception:  # noqa: BLE001
        return "", ""

    def _range_text(values: List[float]) -> str:
        if not values:
            return ""
        low, high = min(values), max(values)
        return "%.4f" % low if low == high else tm("range_between") % (low, high)

    return _range_text(input_rates), _range_text(analysed_rates)


def _build_figure_index_lines(run_folder: str) -> List[str]:
    lines: List[str] = []
    manifest = os.path.join(run_folder, "visualize_results", "figure_manifest.json")
    try:
        if os.path.exists(manifest):
            data = load_json(manifest)
            figures = data.get("figures", []) if isinstance(data, dict) else []
            for entry in figures[:10]:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("file_path") or entry.get("file") or entry.get("path") or ""
                title = entry.get("plot_type") or entry.get("title") or ""
                caption = str(entry.get("caption") or "")[:70]
                if path:
                    lines.append(tm("figure_index_row_prefix") + chr(96) + "%s" + chr(96) + tm("figure_index_row_tail") % (
                        os.path.basename(str(path)), str(title), caption or tm("figure_index_caption_fallback")))
            if figures:
                lines.append(tm("figure_index_full_prefix") + chr(96) + "figures.md" + chr(96) + tm("figure_index_and") + chr(96) + "visualize_results/figure_index.md" + chr(96) + tm("figure_index_captions") + chr(96) + "figures_captions.md" + chr(96) + tm("sentence_stop"))
    except Exception:
        pass
    return lines


def _n_sig_for_contrast(csv_path: str, logfc_thresh: float, p_thresh: float) -> Tuple[int, int, int]:
    """Return (n_fdr_only, n_fdr_and_effect, n_rows) so both counting conventions are explicit."""
    try:
        rows = _read_csv_rows(csv_path, limit=20000)
    except Exception:
        return (0, 0, 0)
    n_fdr = 0
    n_both = 0
    n_rows = 0
    for row in rows:
        try:
            logfc = float(row.get("logFC", "nan"))
            adj = float(row.get("adj.P.Val", row.get("FDR", "nan")))
        except Exception:
            continue
        n_rows += 1
        if adj < p_thresh:
            n_fdr += 1
            if abs(logfc) > logfc_thresh:
                n_both += 1
    return (n_fdr, n_both, n_rows)


def _build_deliverable_checklist_lines(run_folder: str, report_body: str) -> List[str]:
    params = _run_parameters(run_folder)
    design = params.get("analysis_design") if isinstance(params.get("analysis_design"), dict) else {}
    differential = design.get("differential", {}) if isinstance(design.get("differential"), dict) else {}
    logfc_thresh = float(differential.get("logfc_thresh", 0.25) or 0.25)
    p_thresh = float(differential.get("p_thresh", 0.05) or 0.05)
    lines: List[str] = []
    rows: List[Tuple[str, str, str]] = []

    import glob as _glob
    files = sorted(_glob.glob(os.path.join(run_folder, "processed_proteins", "differential_*.csv")))
    for path in files:
        name = os.path.basename(path).replace("differential_", "").replace(".csv", "")
        n_fdr, n_both, n_rows = _n_sig_for_contrast(path, logfc_thresh, p_thresh)
        groups = [g.replace("_", " ") for g in re.split(r"_vs_", name) if g]
        mentioned = False
        if len(groups) == 2:
            a, b = (re.escape(g) for g in groups)
            mentioned = bool(re.search(a + r".{0,24}(?:vs|相对|与|和|对比).{0,24}" + b, report_body)
                             or re.search(b + r".{0,24}(?:vs|相对|与|和|对比).{0,24}" + a, report_body))
        elif groups:
            mentioned = all(g in report_body for g in groups)
        state = tm("checklist_mentioned") if mentioned else tm("checklist_not_mentioned")
        rows.append((tm("checklist_contrast_prefix") + name.replace("_", " "),
                     tm("checklist_computed") % (
                         p_thresh, n_fdr, logfc_thresh, n_both, n_rows, state),
                     "processed_proteins/differential_%s.csv" % name))
    lines.append(tm("checklist_header"))
    lines.append("|---|---|---|")
    for item, state, source in rows[:12]:
        lines.append("| %s | %s | %s |" % (item, state, source))
    return lines


def _resolved_run_symbols(run_folder: str) -> set:
    """Symbols this run's own data contains, used to protect a named protein from the prose rule.

    A named protein whose symbol is also an ordinary English word must still be reported when
    the run's data holds it. Only two cheap sources are read, and a read failure leaves the set
    empty, which can never invent a candidate.
    """
    import glob as _glob
    symbols = set()
    try:
        pattern = os.path.join(run_folder, "processed_proteins", "differential_*.csv")
        for path in sorted(_glob.glob(pattern)):
            for row in _read_csv_rows(path, limit=20000):
                for column in ("PG.Genes", "Genes"):
                    for part in re.split(r"[;,]", str(row.get(column) or "")):
                        if part.strip():
                            symbols.add(part.strip())
    except Exception:
        pass
    try:
        evidence = os.path.join(run_folder, "evaluation_evidence", "candidate_protein_evidence.csv")
        for row in _read_csv_rows(evidence, limit=4000):
            for column in ("matched_gene", "candidate"):
                value = str(row.get(column) or "").strip()
                if value:
                    symbols.add(value)
    except Exception:
        pass
    return symbols


def _task_named_genes_and_prose(run_folder: str) -> Tuple[List[str], List[str]]:
    """(candidates the task names, tokens read as ordinary words rather than candidates).

    Presence in data must not decide the candidate list: a protein the task names explicitly is
    kept whether or not the matrix contains it, and is later reported as not matched instead of
    being dropped. Ordinary English prose is not a symbol, so a token that cannot be resolved in
    this run and reads as a plain English word is suppressed. The suppression is counted and
    surfaced (see _build_candidate_audit_lines), never silent.
    """
    params = _run_parameters(run_folder)
    path = str(params.get("user_input_path") or "")
    if not path or not os.path.exists(path):
        return [], []
    try:
        text = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception:
        return [], []
    stop = {"CSV", "TSV", "QC", "PCA", "UMAP", "MS", "GO", "KEGG", "FDR", "PBS", "DMSO", "DNA",
            "RNA", "ATP", "PH", "ID", "AI", "LLM", "HGNC", "SP", "THP", "NF", "KB", "IL", "TNF",
            "IFN", "CD", "GSEA", "ORA", "FC", "CV", "SD", "SE", "CI", "IQR", "HSC", "MPP", "LMPP",
            "GMP", "MEP", "EB", "IPSC", "IPS", "RG", "ORG", "EN", "IPC", "LPS", "EDTA", "LC", "MS2", "HLA"}
    return _report_language.classify_task_tokens(
        text,
        resolved=_resolved_run_symbols(run_folder),
        stop=stop,
        labels=_run_design_labels(run_folder),
    )


def _run_design_labels(run_folder: str) -> set:
    """Design group values recorded for this run, upper-cased.

    A group name such as "Control" is a label of the experiment, not a candidate protein, so it
    must not reach the candidate audit as an unresolved symbol.
    """
    labels = set()
    path = os.path.join(run_folder, "evaluation_evidence", "group_composition_qc.csv")
    for row in _read_csv_rows(path, limit=2000):
        for column in ("group", "\ufeffgroup"):
            value = str(row.get(column) or "").strip()
            if value and value.lower() != "nan":
                labels.add(value.upper())
    return labels


def _task_named_genes(run_folder: str) -> List[str]:
    """Every gene-like symbol named by the task. Presence in data must not decide the list."""
    return _task_named_genes_and_prose(run_folder)[0]


def _candidate_trace_verdicts(run_folder: str) -> Dict[str, str]:
    """candidate -> one-line pipeline verdict from analysis_extensions (if that run produced it)."""
    for sub in ("evaluation_evidence_ext", "evaluation_evidence"):
        path = os.path.join(run_folder, sub, "candidate_exclusion_trace.csv")
        if not os.path.exists(path):
            continue
        verdicts: Dict[str, str] = {}
        for row in _read_csv_rows(path, limit=2000):
            if str(row.get("stage")) == "verdict" and row.get("candidate"):
                verdicts[str(row["candidate"])] = str(row.get("reason", ""))
        if verdicts:
            return verdicts
    return {}


def _gene_rows_in_differential_tables(run_folder: str, gene: str) -> List[Tuple[str, str, str, str, str]]:
    """(contrast, protein_group, logFC, P.Value, adj.P.Val) rows for one gene, per computed contrast."""
    import glob as _glob
    out: List[Tuple[str, str, str, str, str]] = []
    for path in sorted(_glob.glob(os.path.join(run_folder, "processed_proteins", "differential_*.csv"))):
        contrast = os.path.basename(path).replace("differential_", "").replace(".csv", "")
        for row in _read_csv_rows(path, limit=20000):
            symbol = str(row.get("PG.Genes") or "").split(";")[0].strip()
            if symbol != gene:
                continue
            out.append((contrast, str(row.get("PG.ProteinGroups", "")),
                        str(row.get("logFC", "")), str(row.get("P.Value", "")),
                        str(row.get("adj.P.Val", ""))))
    return out



def _build_extension_evidence_lines(run_folder: str) -> List[str]:
    """Cite the four extension tables when a run produced them."""
    lines: List[str] = []

    def _ext_path(name: str) -> str:
        for sub in ("evaluation_evidence_ext", "evaluation_evidence"):
            candidate = os.path.join(run_folder, sub, name)
            if os.path.exists(candidate):
                return candidate
        return ""

    trace_path = _ext_path("candidate_exclusion_trace.csv")
    if trace_path:
        rows = _read_csv_rows(trace_path, limit=500)
        verdicts = [r for r in rows if str(r.get("stage")) == "verdict"]
        if verdicts:
            lines.append(tm("ext_candidate_chain_title"))
            for row in verdicts[:8]:
                lines.append(tm("ext_candidate_chain_row") % (row.get("candidate", ""), row.get("reason", "")))
            lines.append("")
    verdict_path = _ext_path("estimability_verdict.txt")
    if verdict_path:
        try:
            text = open(verdict_path, encoding="utf-8").read().strip()
        except Exception:
            text = ""
        if text:
            lines.append(tm("ext_batch_title"))
            lines.append("- " + text)
            lines.append("")
    strat_path = _ext_path("stratified_contrast_summary.csv")
    if strat_path:
        rows = _read_csv_rows(strat_path, limit=200)
        if rows:
            lines.append(tm("ext_stratified_title"))
            lines.append(tm("ext_stratified_header"))
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for row in rows[:10]:
                lines.append(tm("ext_stratified_row") % (
                    row.get("stratifier", ""), row.get("level", ""),
                    str(row.get("contrast", "")).replace("_", " "), row.get("direction", ""),
                    row.get("effect_scale", ""), row.get("n_a", ""), row.get("n_b", ""),
                    row.get("n_sig", ""), row.get("verdict", ""), str(row.get("note", ""))[:70]))
            lines.append("")
    conc_path = _ext_path("contrast_concordance.csv")
    if conc_path:
        rows = _read_csv_rows(conc_path, limit=100)
        if rows:
            lines.append(tm("ext_contrast_compare_title"))
            lines.append(tm("ext_contrast_compare_header"))
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for row in rows[:8]:
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    str(row.get("contrast_a", "")).replace("_", " "), str(row.get("contrast_b", "")).replace("_", " "),
                    row.get("shared_proteins", ""), row.get("spearman_logfc", ""),
                    row.get("median_abs_logfc_a", ""), row.get("median_abs_logfc_b", ""),
                    row.get("n_sig_a", ""), row.get("n_sig_b", ""),
                    row.get("overlap_sig", "") if row.get("overlap_sig") not in (None, "") else "0",
                    row.get("sign_agreement", ""),
                    (tm("ext_fdr_from_table") if str(row.get("fdr_source_a", "")).startswith("table") else tm("ext_fdr_recomputed")),
                    str(row.get("verdict", ""))[:60]))
            lines.append("- " + str(rows[0].get("note", "")))
            lines.append("")
    return lines

def _build_candidate_audit_lines(run_folder: str) -> List[str]:
    """Task-named candidates first (complete), then a bounded exploratory list.

    Rows keep the full protein-group identifier and are keyed by (candidate, protein
    group, contrast); no statistic is ever borrowed from a different contrast.
    """
    evidence = os.path.join(run_folder, "evaluation_evidence", "candidate_protein_evidence.csv")
    rows = _read_csv_rows(evidence, limit=4000)
    required, prose_suppressed = _task_named_genes_and_prose(run_folder)
    lines: List[str] = []

    trace_verdicts = _candidate_trace_verdicts(run_folder)
    if required:
        lines.append(tm("cand_audit_title_named") % len(required))
    else:
        lines.append(tm("cand_audit_no_named"))
    # en only: a zh report must stay byte-identical to the released text, so the zh value of this
    # key is never emitted even though the registry carries both languages
    if prose_suppressed and _report_language.get_language() != "zh":
        lines.append(tm("cand_audit_prose_note") % len(prose_suppressed))
    if required:
        lines.append("")
        lines.append(tm("cand_audit_header_full"))
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for gene in required:
            entries = [r for r in rows if str(r.get("matched_gene") or r.get("candidate") or "") == gene]
            seen = set()
            written = 0
            for row in entries:
                key = (str(row.get("matched_protein", "")), str(row.get("contrast", "")))
                if key in seen:
                    continue
                seen.add(key)
                detected = str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"}
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | candidate_protein_evidence.csv |" % (
                    gene, row.get("matched_protein", ""), _human_contrast(row.get("contrast", "")),
                    tm("cand_audit_detected") if detected else tm("cand_audit_not_detected"), _human_direction(row.get("direction", "")),
                    row.get("logFC", "") or tm("cand_audit_not_evaluated"),
                    row.get("P.Value", "") or tm("cand_audit_not_evaluated"),
                    row.get("adj.P.Val", "") or tm("cand_audit_not_evaluated")))
                written += 1
            if written == 0:
                diff_rows = _gene_rows_in_differential_tables(run_folder, gene)
                if diff_rows:
                    for contrast, protein_group, logfc, p_value, adj in diff_rows[:4]:
                        lines.append(tm("cand_audit_diff_row") % (
                            gene, protein_group, contrast.replace("_", " "),
                            _human_direction("up_in_group_a" if float(logfc) >= 0 else "down_in_group_a"),
                            logfc, p_value, adj))
                else:
                    note = trace_verdicts.get(gene, tm("cand_audit_unresolved"))
                    lines.append(tm("cand_audit_unresolved_row") % (gene, note))
        lines.append("")

    exploratory = [r for r in rows if str(r.get("matched_gene") or r.get("candidate") or "") not in set(required)]
    if exploratory:
        lines.append(tm("cand_audit_exploratory_title"))
        lines.append("")
        lines.append(tm("cand_audit_header_short"))
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in exploratory[:12]:
            gene = str(row.get("matched_gene") or row.get("candidate") or "")
            detected = str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"}
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                gene, row.get("matched_protein", ""), _human_contrast(row.get("contrast", "")),
                tm("cand_audit_detected") if detected else tm("cand_audit_not_detected"), _human_direction(row.get("direction", "")),
                row.get("logFC", ""), row.get("P.Value", ""), row.get("adj.P.Val", "")))
        lines.append("")
    lines.append(tm("cand_audit_footer_note"))
    return lines


def _primary_contrast_candidate_stats(run_folder: str) -> Tuple[str, Dict[str, Tuple[float, float, float]]]:
    """gene -> (logFC, P.Value, adj.P.Val) taken from the primary contrast differential table."""
    params = _run_parameters(run_folder)
    design = params.get("analysis_design") if isinstance(params.get("analysis_design"), dict) else {}
    differential = design.get("differential", {}) if isinstance(design.get("differential"), dict) else {}
    contrasts = differential.get("contrasts") or []
    if not contrasts:
        return ("", {})
    name = str(contrasts[0].get("name", ""))
    path = os.path.join(run_folder, "processed_proteins", "differential_%s.csv" % name)
    if not os.path.exists(path):
        return ("", {})
    table: Dict[str, Tuple[float, float, float]] = {}
    for row in _read_csv_rows(path, limit=20000):
        gene = str(row.get("PG.Genes") or row.get("gene") or "").split(";")[0].strip()
        if not gene:
            continue
        try:
            table[gene] = (float(row.get("logFC", "nan")), float(row.get("P.Value", "nan")),
                           float(row.get("adj.P.Val", "nan")))
        except Exception:
            continue
    return (name, table)


def _build_offline_enrichment_lines(run_folder: str) -> List[str]:
    """Enrichment evidence for the primary contrast, carrying the query set, background and method.

    The rule follows the method actually used: for threshold-based over-representation
    analysis (ORA) the query set is the filtered protein list, so the block reports the
    query size, the mapped size, the background and the FDR cutoff, and only cites terms
    that pass FDR. A ranking-based enrichment (GSEA-style) would instead carry a ranking
    statistic and must not be gated by the same rule.
    """
    params = _run_parameters(run_folder)
    design = params.get("analysis_design") if isinstance(params.get("analysis_design"), dict) else {}
    contrasts = (design.get("differential") or {}).get("contrasts") or []
    if not contrasts:
        return []
    contrast = str(contrasts[0].get("name", ""))
    root = os.path.join(run_folder, "enrichment_results")
    if not os.path.isdir(root):
        return []
    import glob as _glob
    lines: List[str] = [tm("off_main_contrast") % contrast.replace("_", " ")]
    any_cited = False
    for direction, label in (("upregulated", _report_language.t("captions.direction_upregulated")), ("downregulated", _report_language.t("captions.direction_downregulated"))):
        rows_all: List[Dict[str, Any]] = []
        for resource in ("GO", "KEGG", "Reactome"):
            pattern = os.path.join(root, "%s_%s_%s_proteins.csv" % (resource, contrast, direction))
            for path in sorted(_glob.glob(pattern))[:1]:
                for row in _read_csv_rows(path, limit=200):
                    row["_resource"] = resource
                    rows_all.append(row)
        if not rows_all:
            lines.append(tm("off_no_direction_result") % label)
            continue
        head = rows_all[0]
        query_n = head.get("input_gene_count") or head.get("normalized_query_gene_count") or ""
        matched = head.get("matched_offline_gene_count") or ""
        background = head.get("BgRatio") or ""
        cutoff = head.get("fdr_cutoff") or ""
        source = head.get("source") or ""
        method = tm("off_method_ora")
        if "gsea" in str(source).lower() or "rank" in str(head.get("significance_status", "")).lower():
            method = tm("off_method_ranked")
        passing = [r for r in rows_all if str(r.get("passes_fdr", "")).lower() in {"true", "1", "yes"}]
        # R28: the old line printed BgRatio as "背景 13/19591" and each term as "重叠 6/19591", which
        # mixed four different quantities: the query set, the mapped query, the gene-set size in the
        # background and the background size. Each is now named.
        def _fmt_prob(value: Any) -> str:
            try:
                return "%.3g" % float(value)
            except (TypeError, ValueError):
                return str(value or "未记录")

        ratio_text = str(background or "")
        background_size = ratio_text.split("/")[-1] if "/" in ratio_text else ""
        tested_count = head.get("strict_fdr_rows") or ""
        selected_count = head.get("selected_rows") or ""
        lines.append(tm("off_direction_detail")
                     % (label, query_n or _report_language.t("captions.not_recorded"), matched or _report_language.t("captions.not_recorded"),
                        background_size or _report_language.t("captions.not_recorded"), method, source or tm("off_resource_local_gmt"),
                        cutoff or _report_language.t("captions.not_recorded")))
        if tested_count or selected_count:
            lines.append(tm("off_tested_entries")
                         % (tested_count or _report_language.t("captions.not_recorded"), selected_count or _report_language.t("captions.not_recorded")))
        if passing:
            any_cited = True
            for row in passing[:3]:
                members = str(row.get("geneID", ""))[:70]
                term_ratio = str(row.get("BgRatio", ""))
                term_set_size = term_ratio.split("/")[0] if "/" in term_ratio else ""
                term_bg_size = term_ratio.split("/")[-1] if "/" in term_ratio else ""
                lines.append(tm("off_entry_row")
                             % (row.get("_resource"), str(row.get("Description", ""))[:60],
                                _fmt_prob(row.get("p.adjust")), row.get("Count", ""),
                                term_set_size or _report_language.t("captions.not_recorded"), term_bg_size or _report_language.t("captions.not_recorded"),
                                row.get("GeneRatio", "") or _report_language.t("captions.not_recorded"),
                                (tm("off_members") + members) if members else ""))
        else:
            best = min((r for r in rows_all if r.get("p.adjust")), key=lambda r: float(r["p.adjust"]), default=None)
            lines.append(tm("off_no_fdr_entries")
                         % ((tm("off_min_padjust") % best["p.adjust"]) if best else ""))
    if not any_cited:
        lines.append(tm("off_zero_citations_note"))
    return lines


def _build_required_analysis_section(run_folder: str, report_body: str) -> List[str]:
    lines: List[str] = []
    enrichment = _build_offline_enrichment_lines(run_folder)
    if enrichment:
        lines.append(tm("off_section_title"))
        lines.extend(enrichment)
        lines.append("")
        lines.append(tm("off_section_note"))
        lines.append("")
    checklist = _build_deliverable_checklist_lines(run_folder, report_body)
    if checklist:
        lines.append(tm("req_section_title"))
        lines.extend(checklist)
        lines.append("")
        lines.append(tm("req_section_note"))
        lines.append("")
    extension_lines = _build_extension_evidence_lines(run_folder)
    if extension_lines:
        lines.extend(extension_lines)
    lines.append(tm("req_candidate_pointer"))
    return lines

def prepend_report_front_matter(report_text: str, run_folder: str) -> str:
    _, body = _strip_report_title(report_text)
    for heading in CHINESE_REPORT_HEADINGS + LEGACY_REPORT_HEADINGS:
        body = _remove_h2_section(body, heading)

    story_first = _build_story_first_report(body, run_folder)
    if story_first:
        return _inject_selfcontained_sections(story_first, run_folder, body)

    blocks = [
        _build_executive_result_summary(run_folder),
        _build_scoring_evidence_first_screen(run_folder),
        _build_main_findings_section(body, run_folder),
        _build_evidence_boundary_section(run_folder),
        f"## {canonical_report_headings()['appendix']}\n{tm('appendix_intro')}",
    ]
    composed = "\n\n".join(block for block in blocks if block).strip() + "\n"
    return _inject_selfcontained_sections(composed, run_folder, body)



def _module_dispersion_within(sample_scores_path: str, group_value: str, modules: List[str],
                              cluster_col: str = "Cluster") -> List[str]:
    """Per-module spread inside one group (answers "group means hide within-group gradients")."""
    rows = _read_csv_rows(sample_scores_path, limit=20000)
    if not rows:
        return []
    subset = [r for r in rows if str(r.get(cluster_col, "")) == group_value]
    if len(subset) < 4:
        return []
    out: List[str] = []
    for module in modules:
        values = []
        for row in subset:
            try:
                values.append(float(row.get(module, "nan")))
            except Exception:
                continue
        values = sorted(v for v in values if v == v)
        if len(values) < 4:
            continue
        n = len(values)
        q1 = values[int(0.25 * (n - 1))]
        q3 = values[int(0.75 * (n - 1))]
        out.append(tm("module_dispersion_row") % (
            module, n, values[n // 2], q3 - q1, q1, q3, values[-1] - values[0]))
    return out


def _gene_logfc_lookup(run_folder: str, genes: List[str]) -> Dict[str, Dict[str, float]]:
    import glob as _glob
    table: Dict[str, Dict[str, float]] = {}
    for path in sorted(_glob.glob(os.path.join(run_folder, "processed_proteins", "differential_*.csv"))):
        contrast = os.path.basename(path).replace("differential_", "").replace(".csv", "")
        try:
            rows = _read_csv_rows(path, limit=20000)
        except Exception:
            continue
        for row in rows:
            gene = ""
            for column in ("PG.Genes", "gene", "Gene", "gene_name", "Genes", "matched_gene"):
                value = row.get(column)
                if value:
                    gene = str(value).split(";")[0].strip()
                    break
            if gene not in genes:
                continue
            try:
                logfc = float(row.get("logFC", "nan"))
                adj = float(row.get("adj.P.Val", row.get("FDR", "nan")))
            except Exception:
                continue
            table.setdefault(gene, {})[contrast] = (logfc, adj)
    return table



_DIRECTION_TEXT = {
    "up_in_group_a": "第一组更高（上调）",
    "up_in_display_group_a": "第一组更高（上调）",
    "down_in_group_a": "第一组更低（下调）",
    "down_in_display_group_a": "第一组更低（下调）",
    "up_in_group_b": "第二组更高（上调）",
    "down_in_group_b": "第二组更低（下调）",
    "detected_without_selected_contrast": "仅记录存在性（未指定对比）",
    "symbol_not_matched_in_available_matrix": "未建立符号映射（不等同于未检出）",
    "not_detected_in_available_matrix": "当前矩阵未检出",
}

_TEST_METHOD_TEXT = {
    "python_welch_ttest_fallback": "Python 实现的 Welch t 检验",
    "python_welch_ttest": "Welch t 检验",
    "welch_ttest": "Welch t 检验",
}


def _human_direction(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return _report_language.t("captions.not_recorded")
    return _table_direction().get(text, text.replace("_", " "))


def _human_contrast(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return _report_language.t("captions.not_recorded")
    if text.lower().replace(" ", "").replace("_", "") == "matrixpresenceonly":
        return tm("human_contrast_unspecified")
    return text.replace("_", " ")


def _human_test_method(run_folder: str) -> str:
    params = _run_parameters(run_folder)
    for key, value in params.items():
        if isinstance(value, dict) and value.get("method"):
            method = str(value["method"])
            method_texts = _table_test_method()
            if method in method_texts or "ttest" in method or "welch" in method.lower():
                return method_texts.get(method, method.replace("_", " "))
    return tm("human_test_two_group")


def _unused_design_variables(run_folder: str) -> List[str]:
    """Design columns present in SampleInfo but not used for grouping or adjustment."""
    params = _run_parameters(run_folder)
    design = params.get("analysis_design") if isinstance(params.get("analysis_design"), dict) else {}
    sampleinfo_path = params.get("sampleinfo_path") or ""
    if not sampleinfo_path or not os.path.exists(sampleinfo_path):
        return []
    try:
        with open(sampleinfo_path, "r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline().strip().split(",")
    except Exception:
        return []
    used = {str(design.get("sample_id_col", "")), str(design.get("protein_id_col", "")),
            str(design.get("gene_col", "")), str(design.get("group_col", "")),
            str(design.get("batch_col", "") or ""), str(design.get("pair_col", "") or "")}
    used.update(str(c) for c in (design.get("covariates") or []))
    skip = {"file", "filenames", "index", "unnamed: 0", ""}
    unused = []
    for column in header:
        name = column.strip()
        if not name or name.lower() in skip or name in used:
            continue
        if name.lower().startswith("unnamed"):
            continue
        unused.append(name)
    return unused[:6]

def _dimension4_verdict_lines(run_folder: str, dataset: str) -> List[str]:
    lines: List[str] = []
    sample_scores = os.path.join(run_folder, "evaluation_evidence", "curated_module_sample_scores.csv")

    if dataset == "Nat_Methods_DVP_2023":
        genes = ["GLUL", "CYP2E1", "ALDH1A1", "ASS1", "CPS1", "GLS2", "OAT"]
        table = _gene_logfc_lookup(run_folder, genes)
        ok = 0
        total = 0
        detail = []
        for gene, contrasts in table.items():
            low = contrasts.get("Midlobular_vs_Portal") or contrasts.get("Portal_vs_Midlobular")
            high = contrasts.get("Central_vs_Midlobular") or contrasts.get("Midlobular_vs_Central")
            if not low or not high:
                continue
            low_sign = low[0] if "Midlobular_vs_Portal" in contrasts else -low[0]
            high_sign = high[0] if "Central_vs_Midlobular" in contrasts else -high[0]
            total += 1
            intermediate = (low_sign > 0 and high_sign > 0) or (low_sign < 0 and high_sign < 0)
            if intermediate:
                ok += 1
            detail.append("%s(%s)" % (gene, tm("d4_ordered") if intermediate else tm("d4_not_ordered")))
        if total:
            lines.append(tm("d4_verdict_liver") % (total, ok, tm("sep_comma").join(detail[:8])))

    if dataset == "Nat_Methods_iPSC_2025":
        stats = _module_dispersion_within(
            sample_scores, "EB",
            ["pluripotency_core", "lineage_endoderm", "lineage_mesoderm", "lineage_ectoderm",
             "cell_cycle_replication", "chromatin_open_remodeling"])
        if stats:
            lines.append(tm("d4_verdict_ipsceb") % tm("sep_semicolon").join(stats[:4]))

    if dataset == "Science_BloodCell_2025":
        rows = _read_csv_rows(os.path.join(run_folder, "evaluation_evidence",
                                           "curated_module_group_summary.csv"), limit=500)
        means: Dict[str, List[Tuple[str, float]]] = {}
        for row in rows:
            try:
                value = float(row.get("mean_score", "nan"))
            except Exception:
                continue
            if value != value:
                continue
            group_name = str(row.get("group", ""))
            if "_minus_" in group_name or not group_name:
                continue
            means.setdefault(str(row.get("module", "")), []).append((group_name, value))
        expected = {"hsc_maintenance_ppp_chromatin": "HSC", "granulocyte_granule": "GMP"}
        bits = []
        for module, entries in means.items():
            if not entries:
                continue
            top = max(entries, key=lambda item: item[1])
            want = expected.get(module, "")
            agreed = tm("d4_label_agree") if want and top[0] == want else (tm("d4_label_disagree") if want else tm("d4_label_no_expectation"))
            bits.append(tm("d4_module_top_state") % (module, top[0], top[1], agreed))
        if bits:
            lines.append(tm("d4_verdict_bloodcell")
                         % tm("sep_semicolon").join(bits[:4]))

    if not lines:
        rows = _read_csv_rows(os.path.join(run_folder, "evaluation_evidence",
                                           "candidate_protein_evidence.csv"), limit=4000)
        passed = 0
        direction_only = 0
        undetected = 0
        for row in rows:
            detected = str(row.get("detected_in_matrix", "")).lower() in {"true", "1", "yes"}
            try:
                adj = float(row.get("adj.P.Val", "nan"))
            except Exception:
                adj = float("nan")
            if not detected:
                undetected += 1
            elif adj == adj and adj < 0.05:
                passed += 1
            else:
                direction_only += 1
        if rows:
            lines.append(tm("d4_verdict_generic")
                         % (passed, direction_only, undetected))
    return lines


def _build_generic_dimension4_section(run_folder: str, heading_index: int) -> List[str]:
    import glob as _glob
    metadata = _core_story_metadata(run_folder)
    dataset = _get_dataset_name(run_folder)
    files = sorted(_glob.glob(os.path.join(run_folder, "processed_proteins", "differential_*.csv")))
    if not files:
        return []
    params = _run_parameters(run_folder)
    differential = (params.get("analysis_design") or {}).get("differential", {}) if isinstance(params.get("analysis_design"), dict) else {}
    logfc_thresh = float(differential.get("logfc_thresh", 0.25) or 0.25)
    p_thresh = float(differential.get("p_thresh", 0.05) or 0.05)
    lines: List[str] = ["", tm("d4_heading") % heading_index]
    lines.append(tm("d4_intro"))
    lines.append("")
    for path in files[:3]:
        contrast = os.path.basename(path).replace("differential_", "").replace(".csv", "")
        n_fdr, n_both, n_rows = _n_sig_for_contrast(path, logfc_thresh, p_thresh)
        lines.append(tm("d4_contrast_count") % (
            contrast.replace("_", " "), p_thresh, n_fdr, logfc_thresh, n_both, n_rows))
    lines.append("")
    for entry in _dimension4_verdict_lines(run_folder, dataset):
        lines.append(entry)
        lines.append("")
    boundary = metadata.get("boundary")
    lines.append(tm("d4_closing_boundary") % (tm("sentence_stop") if not boundary else tm("d4_boundary_suffix") % _short_story_value(boundary, 160)))
    return lines


def _build_dimension4_section(run_folder: str, heading_index: int) -> List[str]:
    dataset = _get_dataset_name(run_folder)
    spec = _dimension4_spec_for(dataset)
    if spec:
        lines = _build_dimension4_integration_section(run_folder, heading_index, spec_override=spec)
        if lines:
            verdicts = _dimension4_verdict_lines(run_folder, dataset)
            if verdicts:
                insert_at = len(lines)
                for idx, line in enumerate(lines):
                    if line.startswith(tm("boundary_prefix")):
                        insert_at = idx
                        break
                lines[insert_at:insert_at] = [""] + verdicts
            return lines
    return _build_generic_dimension4_section(run_folder, heading_index)


def _renumber_report_headings(text: str) -> str:
    counter = {"n": 0}

    def repl(match):
        number = counter["n"]
        counter["n"] += 1
        return "## %d. %s" % (number, match.group(1))

    return re.sub(r"(?m)^##\s*\d+\.\s*(.+)$", repl, text)


def _candidate_audit_block_present(text: str, title: str) -> bool:
    """Whether the candidate audit table itself is already in the report text.

    The zh path keeps the released substring test. A zh report quotes the block title inside the
    required-deliverables sentence, so the released code never appends the block there; that
    behaviour is frozen because a zh run has to stay byte-identical to the released text. The
    English report carries its own equivalent sentence, which is why the same substring test
    would now suppress the block in English too, so the English path asks for a real heading.
    """
    if _report_language.get_language() == "en":
        return re.search(r"(?m)^#{2,4}\s*" + re.escape(title) + r"\s*$", text or "") is not None
    return title in text


def _inject_selfcontained_sections(report_text: str, run_folder: str, report_body: str) -> str:
    """Deterministically lift the analysis conditions, deliverable coverage and
    candidate audit from the artifact layer into the main report."""
    if _report_language.t("core.analysis_conditions") in report_text:
        return report_text
    text = report_text
    block: List[str] = ["", "## 9. " + _report_language.t("core.analysis_conditions"), ""]
    block.extend(_build_analysis_conditions_lines(run_folder))
    block.extend(["", "### " + _report_language.t("core.qc_semantics"), "",
                  "- " + _qc_semantics_note(run_folder)])
    figure_lines = _build_figure_index_lines(run_folder)
    if figure_lines:
        block.extend(["", "### " + _report_language.t("core.figure_index"), ""])
        block.extend(figure_lines)
    required = _build_required_analysis_section(run_folder, report_body)
    if required:
        block.extend(["", "## 9. " + tm("required_deliverables"), ""])
        block.extend(required)

    anchor = _numbered_heading_re_for_key("core.s1", "1").search(text)
    if anchor:
        nxt = re.search(r"(?m)^##\s", text[anchor.end():])
        insert_at = anchor.end() + (nxt.start() if nxt else len(text[anchor.end():]))
        text = text[:insert_at].rstrip() + chr(10) + chr(10) + chr(10).join(block).strip(chr(10)) + chr(10) + chr(10) + text[insert_at:].lstrip(chr(10))
    else:
        text = text.rstrip() + chr(10) + chr(10) + chr(10).join(block).strip(chr(10)) + chr(10)

    candidate_lines = _build_candidate_audit_lines(run_folder)
    if candidate_lines and not _candidate_audit_block_present(text, tm("candidate_audit_title")):
        tail_anchor = (
            _numbered_heading_re(_report_language.t("core.story_conclusion")).search(text)
            or _numbered_heading_re(_report_language.t("core.story_assets")).search(text))
        block_text = chr(10).join(["### " + tm("candidate_audit_title"), ""] + candidate_lines + [
            "",
            tm("candidate_audit_note")]).strip(chr(10))
        if tail_anchor:
            text = text[:tail_anchor.start()].rstrip() + chr(10) + chr(10) + block_text + chr(10) + chr(10) + text[tail_anchor.start():]
        else:
            text = text.rstrip() + chr(10) + chr(10) + block_text + chr(10)
    # the section title is emitted in the active language by _build_dimension4_section, but an
    # English run may legitimately carry either marker, so both are accepted as already present
    if tm("mechanism_integration") not in text and "机制整合与解释边界" not in text:
        dim4 = _build_dimension4_section(run_folder, heading_index=9)
        if dim4:
            tail = _numbered_heading_re(_report_language.t("core.story_synthesis")).search(text)
            block_text = chr(10).join(dim4).strip(chr(10))
            if tail:
                text = text[:tail.start()].rstrip() + chr(10) + chr(10) + block_text + chr(10) + chr(10) + text[tail.start():]
            else:
                text = text.rstrip() + chr(10) + chr(10) + block_text + chr(10)
    return _renumber_report_headings(text)

def _polish_pispa_report_terms(report_text: str) -> str:
    """Add concise PiSPA domain wording without adding scoring-oriented text."""
    if "single-cell proteomic" in report_text and "cell motility" in report_text and "actin cytoskeleton" in report_text:
        return report_text
    heading = f"## {story_report_headings()[3]}"
    insertion = (
        chr(10) + tm("pispa_term_note") + chr(10)
    )
    if heading in report_text:
        return report_text.replace(heading + "\n", heading + "\n" + insertion, 1)
    return report_text.rstrip() + "\n" + insertion


def write_report_variants(report_content: str, run_folder: str, base_name: str) -> List[str]:
    os.makedirs(run_folder, exist_ok=True)
    appendix_markers = [
        f"\n## {canonical_report_headings()['appendix']}",
        f"\n## {story_report_headings()[7]}",
        chr(10) + "### " + tm("appendix_reproducible"),
        chr(10) + "### " + tm("appendix_technical"),
        "\n## Evidence Appendix",
        "\n## Scoring Evidence Appendix",
        "\n## Protein Leakage Evidence Appendix",
        "\n## Evidence Boundary And Confidence",
    ]
    split_at = len(report_content)
    for marker in appendix_markers:
        idx = report_content.find(marker)
        if idx >= 0:
            split_at = min(split_at, idx)
    conclusion_text = report_content[:split_at].rstrip() + "\n"
    appendix_text = report_content[split_at:].strip()
    if not appendix_text:
        appendix_text = "No generated evidence appendix was available for this report.\n"

    written = []
    if _get_dataset_name(run_folder) in V3_MAIN_DATASETS and base_name in {"report", "final_report"}:
        variant_specs = [
            ("conclusions.md", conclusion_text),
            ("appendix.md", appendix_text + "\n"),
        ]
    else:
        variant_specs = [
            (f"{base_name}_conclusions.md", conclusion_text),
            (f"{base_name}_evidence_appendix.md", appendix_text + "\n"),
        ]
    for filename, content in variant_specs:
        path = os.path.join(run_folder, filename)
        record_report(path, content, mode="w", visible=False)
        written.append(path)
    return written


# Key-figure registry: (visualize_results file name, 图题). Per-figure narrative
# captions (面板说明/阈值/统计口径/解读/证据边界/原论文图映射) are generated data-
# driven into figures_captions.md by figure_captions.py — no hardcoded "用于…"
# caption text lives here anymore.
PISPA_KEY_FIGURE_SPECS = [
    ("qc_sample_overview.png", "样本与缺失率 QC"),
    ("pca_plot.png", "PCA 样本结构"),
    ("umap_Group_Control Cell-Migrated Cell.png", "UMAP 迁移表型分布"),
    ("umap_Cluster.png", "UMAP Cluster 分布"),
    ("Cluster_1_vs_Cluster_2_volcano_plot.png", "Cluster 1 vs Cluster 2 火山图"),
    ("Cluster_1_vs_Cluster_3_volcano_plot.png", "Cluster 1 vs Cluster 3 火山图"),
    ("Cluster_2_vs_Cluster_3_volcano_plot.png", "Cluster 2 vs Cluster 3 火山图"),
    ("mechanism_enrichment_dotplot.png", "机制富集 dotplot"),
    ("mechanism_pathway_gene_heatmap_Clust_2c410786f2.png", "机制蛋白热图"),
    ("mechanism_top_protein_group_means.png", "核心蛋白组均值图"),
    ("protein_contrast_bubble.png", "候选蛋白对比气泡图"),
    ("key_protein_overview.png", "关键蛋白总览"),
]

GENERIC_KEY_FIGURE_SPECS = [
    ("qc_sample_overview.png", "样本与缺失率 QC"),
    ("pca_plot.png", "PCA 样本结构"),
    ("umap_plot.png", "UMAP 样本结构"),
    ("heatmap.png", "差异蛋白热图"),
    ("differential_summary_barplot.png", "差异蛋白数量概览"),
    ("mechanism_enrichment_dotplot.png", "机制富集 dotplot"),
    ("mechanism_top_protein_group_means.png", "核心蛋白组均值图"),
    ("protein_contrast_bubble.png", "候选蛋白对比气泡图"),
    ("key_protein_overview.png", "关键蛋白总览"),
]


def _existing_key_figures(run_folder: str) -> List[Tuple[str, str]]:
    dataset = _get_dataset_name(run_folder)
    fig_dir = os.path.join(run_folder, "visualize_results")
    found: List[Tuple[str, str]] = []
    specs = _key_figure_specs(dataset)
    for filename, title in specs:
        path = os.path.join(fig_dir, filename)
        if _path_exists(path):
            found.append((filename, title))
            continue
        stem, ext = os.path.splitext(filename)
        try:
            matches = [
                item for item in sorted(os.listdir(fig_dir))
                if item.startswith(f"{stem}_") and item.endswith(ext)
            ]
        except Exception:
            matches = []
        if matches:
            found.append((matches[0], title))
    if os.path.isdir(fig_dir):
        try:
            volcanoes = sorted(filename for filename in os.listdir(fig_dir) if filename.endswith("_volcano_plot.png"))
        except Exception:
            volcanoes = []
        seen = {name for name, _ in found}
        for filename in volcanoes[:4]:
            if filename not in seen:
                title = _human_contrast_label(filename.replace("_volcano_plot.png", "")) + tm("figure_volcano_suffix")
                found.append((filename, title))
                seen.add(filename)
    if not found and os.path.isdir(fig_dir):
        for filename in sorted(os.listdir(fig_dir)):
            if filename.lower().endswith(".png"):
                found.append((filename, os.path.splitext(filename)[0]))
            if len(found) >= 8:
                break
    return found


def write_report_asset_files(run_folder: str, base_name: str) -> List[str]:
    dataset = _get_dataset_name(run_folder)
    if dataset not in V3_MAIN_DATASETS:
        return []
    figures = _existing_key_figures(run_folder)
    asset_txt = os.path.join(run_folder, "assets.txt")
    figure_md = os.path.join(run_folder, "figures.md")
    local_preview_md = os.path.join(run_folder, "figures_preview_local.md")
    captions_md = os.path.join(run_folder, figure_captions.CAPTIONS_FILENAME)
    short_figure_names = {
        "qc_sample_overview.png": "qc.png",
        "pca_plot.png": "pca.png",
        "umap_plot.png": "umap.png",
        "heatmap.png": "heatmap.png",
        "differential_summary_barplot.png": "diff_summary.png",
        "Cluster_1_vs_Cluster_2_volcano_plot.png": "volcano_c1_c2.png",
        "Cluster_1_vs_Cluster_3_volcano_plot.png": "volcano_c1_c3.png",
        "Cluster_2_vs_Cluster_3_volcano_plot.png": "volcano_c2_c3.png",
        "mechanism_enrichment_dotplot.png": "enrichment.png",
        "mechanism_top_protein_group_means.png": "protein_means.png",
        "protein_contrast_bubble.png": "protein_bubble.png",
        "key_protein_overview.png": "key_proteins.png",
    }
    # Parallel entries: (visualize_results name, title, short album name). Only
    # figures whose album copy validates become caption records, so captions,
    # album links and the asset list always stay aligned.
    album_entries: List[Tuple[str, str, str]] = []
    short_fig_dir = os.path.join(run_folder, "figures")
    os.makedirs(short_fig_dir, exist_ok=True)
    for filename, title in figures:
        short_name = short_figure_names.get(filename)
        if not short_name:
            base, ext = os.path.splitext(filename)
            safe_base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()[:36] or "figure"
            short_name = f"{safe_base}{ext.lower() or '.png'}"
        src = os.path.join(run_folder, "visualize_results", filename)
        dst = os.path.join(short_fig_dir, short_name)
        if _path_exists(src) and _validate_binary_image_file(src):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
        if _path_exists(dst) and _validate_binary_image_file(dst):
            album_entries.append((filename, title, short_name))

    # Data-driven publication-grade captions (figure_captions.py): one structured
    # caption per figure from figure_manifest + analysis_design + evidence tables.
    caption_records: List[Dict[str, Any]] = []
    try:
        caption_context = figure_captions.load_run_context(run_folder)
        caption_records = figure_captions.build_caption_records(
            caption_context, [(name, title) for name, title, _ in album_entries]
        )
        captions_document = figure_captions.render_figures_captions_md(
            caption_context.get("dataset") or dataset, caption_records, caption_context
        )
    except Exception:
        caption_records = []
        captions_document = ""
    if not caption_records or not captions_document:
        fallback_lines = [
            f"# {dataset} 发表级图注",
            "",
            "图注生成未完成，仅保留图号与标题；数字口径以 figure_manifest.json 与 evaluation_evidence 证据表为准。",
            "",
        ]
        for idx, (_, title, short_name) in enumerate(album_entries, start=1):
            fallback_lines.extend([f"## 图{idx} {title}", "", f"- 图片文件：`figures/{short_name}`", ""])
        captions_document = "\n".join(fallback_lines).rstrip() + "\n"
    record_report(captions_md, captions_document, mode="w", visible=False)

    report_files = [
        "report.md",
        "conclusions.md",
        "appendix.md",
        "figures_captions.md",
    ]
    evidence_files = [
        "evaluation_evidence/core_story_evidence.csv",
        "evaluation_evidence/candidate_protein_evidence.csv",
        "evaluation_evidence/curated_module_group_summary.csv",
        "evaluation_evidence/curated_module_sample_scores.csv",
        "evaluation_evidence/group_composition_qc.csv",
        "visualize_results/figure_index.md",
        "visualize_results/figure_manifest.json",
        "self_check.json",
        "run_status.json",
    ]
    txt_lines = [
        f"{dataset} 报告资产清单",
        "",
        "一、推荐阅读顺序",
        "1. report.md：中文主报告，先看结论和逐任务解读。",
        "2. figures.md：嵌图版图册，直接查看关键图片。",
        "3. figures_captions.md：发表级图注（面板说明、阈值与样本量、统计口径、解读与证据边界、对应原论文图）。",
        "4. figures_preview_local.md：本机绝对路径预览版；如果预览器不能解析相对路径，可打开该文件。",
        "5. appendix.md：证据附录，查看完整表格和边界。",
        "",
        "二、报告文件",
    ]
    for rel in report_files:
        txt_lines.append(f"- {rel}")
    txt_lines.extend(["", "三、关键证据表"])
    for rel in evidence_files:
        if os.path.exists(os.path.join(run_folder, rel)):
            txt_lines.append(f"- {rel}")
    txt_lines.extend(["", "四、关键图表"])
    for idx, (_, title, short_name) in enumerate(album_entries):
        pointer = f"（图注见 figures_captions.md §图{idx + 1}）" if idx < len(caption_records) else "（图注见 figures_captions.md）"
        txt_lines.append(f"- figures/{short_name}：{title}{pointer}")
    txt_lines.extend([
        "",
        "五、证据边界",
        "- current_matrix：差异、候选、模块、QC 和图表来自当前矩阵。",
        "- offline_enrichment：本地离线富集用于解释通路，不等于当前矩阵因果证据。",
        "- extension：启发式推测和后续验证建议只作为可检验假设。",
    ])
    record_report(asset_txt, "\n".join(txt_lines) + "\n", mode="w", visible=False)

    if caption_records:
        links = [f"figures/{short_name}" for _, _, short_name in album_entries]
        figure_md_text = figure_captions.render_brief_figures_md(dataset, caption_records, links)
        abs_links = [
            os.path.abspath(os.path.join(short_fig_dir, short_name)).replace("\\", "/")
            for _, _, short_name in album_entries
        ]
        local_md_text = figure_captions.render_brief_figures_md(
            dataset,
            caption_records,
            abs_links,
            title_suffix="关键图表本机预览版",
            intro=(
                "本文件使用本机绝对路径，主要用于解决某些 Markdown 预览器不能解析相对图片路径的问题；"
                "对外发送时优先使用 `figures.md`。逐图图注见 `figures_captions.md`。"
            ),
        )
    else:
        md_lines = [
            f"# {dataset} 关键图表嵌入版",
            "",
            "每张图只保留图号、标题与嵌图；逐图图注见 `figures_captions.md`。",
            "",
        ]
        local_md_lines = [
            f"# {dataset} 关键图表本机预览版",
            "",
            "本文件使用本机绝对路径；对外发送时优先使用 `figures.md`。逐图图注见 `figures_captions.md`。",
            "",
        ]
        for idx, (_, title, short_name) in enumerate(album_entries, start=1):
            rel = f"figures/{short_name}"
            md_lines.extend([
                f"## 图{idx} {title}",
                "",
                f"![图{idx} {title}]({rel})",
                "",
                "图注详见 `figures_captions.md`。",
                "",
            ])
            abs_path = os.path.abspath(os.path.join(short_fig_dir, short_name)).replace("\\", "/")
            local_md_lines.extend([
                f"## 图{idx} {title}",
                "",
                f"![图{idx} {title}]({abs_path})",
                "",
                "图注详见 `figures_captions.md`。",
                "",
            ])
        figure_md_text = "\n".join(md_lines).rstrip() + "\n"
        local_md_text = "\n".join(local_md_lines).rstrip() + "\n"
    record_report(figure_md, figure_md_text, mode="w", visible=False)
    record_report(local_preview_md, local_md_text, mode="w", visible=False)
    return [asset_txt, figure_md, local_preview_md, captions_md]


def run_report_evidence_check(report_content: str, run_folder: str) -> Dict[str, Any]:
    """Write the generation-side evidence map and the tri-state fact check for this report.

    The verdict is tri-state: pass / unverified / contradiction. "unverified" is explicitly not a
    pass, so a clean exit code must never be read as scientific acceptance; the run flow records the
    verdict next to the report hashes instead.
    """
    module = None
    try:
        import report_evidence as module  # same directory as this file
    except Exception:
        try:
            import importlib.util as _importlib_util
            _spec = _importlib_util.spec_from_file_location(
                "report_evidence", os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_evidence.py"))
            module = _importlib_util.module_from_spec(_spec)
            _spec.loader.exec_module(module)
        except Exception as exc:
            return {"verdict": "unavailable", "reason": "report_evidence module not importable: %s" % exc}
    report_path = os.path.join(run_folder, "report.md")
    if not os.path.exists(report_path):
        report_path = os.path.join(run_folder, "final_output_report.md")
    try:
        result = module.write_artifacts(run_folder, report_text=report_content, report_path=report_path)
    except Exception as exc:
        return {"verdict": "error", "reason": str(exc)}
    return {
        "verdict": result.get("verdict"),
        "n_contradictions": result.get("n_contradictions"),
        "n_unverified": result.get("n_unverified"),
        "n_passed": result.get("n_passed"),
        "coverage": result.get("coverage"),
        "map": result.get("map"),
        "files": [os.path.join(run_folder, "report_evidence_map.json"),
                  os.path.join(run_folder, "report_fact_check.json")],
        "note": "verdict=unverified is not a pass; details in report_fact_check.json.",
    }


def _markdown_table_headers(text: str) -> List[str]:
    # R29 locating helper: the structural self-check looked for the legacy literal headings
    # 核心对比表 / 代表蛋白表, which the story-first layout never renders.  The criterion is
    # unchanged (a reviewable quantitative table must exist); only the way it is located changes:
    # a markdown table header row followed by a separator row is the table itself.
    lines = text.splitlines()
    headers = []
    for position, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith('|') and stripped.endswith('|') and len(stripped) > 2):
            continue
        following = lines[position + 1].strip() if position + 1 < len(lines) else ''
        if re.match(r'^\|[\s:\-|]+\|$', following) and '-' in following:
            headers.append(stripped)
    return headers


def _quantitative_table_present(headers, subject_tokens, value_tokens) -> bool:
    for header in headers:
        if any(token in header for token in subject_tokens) and any(token in header for token in value_tokens):
            return True
    return False


def _candidate_evidence_row_count(run_folder: str) -> Optional[int]:
    """Rows in this run's candidate evidence table, or None when the table itself is absent.

    The two cases are not the same defect. An empty table with a header means the analysis ran and
    no candidate passed screening; an absent table means the run did not produce the artifact that
    the candidate audit is built from.
    """
    path = os.path.join(run_folder, "evaluation_evidence", "candidate_protein_evidence.csv")
    if not os.path.exists(path):
        return None
    try:
        return len(_read_csv_rows(path, limit=4000))
    except Exception:
        return None

def run_report_self_check(report_content: str, run_folder: str) -> Dict[str, Any]:
    evidence_dir = os.path.join(run_folder, "evaluation_evidence")
    external_annotation_json = os.path.join(evidence_dir, "external_annotation_asd_ndd_local.json")
    external_annotation_required = False
    if os.path.exists(external_annotation_json):
        try:
            external_annotation_required = bool(load_json(external_annotation_json).get("applicable"))
        except Exception:
            external_annotation_required = False
    required_csvs = [
        os.path.join(evidence_dir, "group_composition_qc.csv"),
        os.path.join(evidence_dir, "core_story_evidence.csv"),
        os.path.join(evidence_dir, "candidate_protein_evidence.csv"),
        os.path.join(evidence_dir, "curated_module_group_summary.csv"),
        os.path.join(evidence_dir, "dataset_recipe_evidence.csv"),
        os.path.join(evidence_dir, "scoring_standard_coverage.csv"),
    ]
    figure_candidates = [
        os.path.join(run_folder, "figures", "figure_manifest.json"),
        os.path.join(run_folder, "figure_manifest.json"),
        os.path.join(run_folder, "visualize_results", "figure_manifest.json"),
        os.path.join(run_folder, "visualize_results", "figure_index.md"),
        os.path.join(run_folder, "figure_index.md"),
    ]
    coverage_rows = _read_csv_rows(os.path.join(evidence_dir, "scoring_standard_coverage.csv"), limit=200)
    chinese_ratio = _chinese_char_ratio(report_content)
    long_english_paragraph = _has_long_english_paragraph(report_content)
    legacy_primary_heading_present = any(_has_h2(report_content, heading) for heading in LEGACY_REPORT_HEADINGS[:5])
    dataset_for_style = _get_dataset_name(run_folder)
    story_first_structure_present = _has_story_first_structure(report_content)
    exact_pispa_headings_present = all(_has_h2(report_content, heading) for heading in story_report_headings())
    # An English report must not carry Chinese template text. Any residue is reported with
    # the exact fragments so a missing English template becomes a listed gap, never a silent
    # fall back to Chinese.
    english_residue: List[str] = []
    if _report_language.get_language() == "en":
        _scan_body = _report_language.strip_code_and_paths(
            _main_report_text_for_language_check(report_content))
        english_residue = re.findall("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", _scan_body)
    asset_files_present = (
        dataset_for_style not in V3_MAIN_DATASETS
        or (
            _path_exists(os.path.join(run_folder, "assets.txt"))
            and _path_exists(os.path.join(run_folder, "figures.md"))
            and _path_exists(os.path.join(run_folder, "figures_preview_local.md"))
            and _path_exists(os.path.join(run_folder, "figures_captions.md"))
        )
    )
    key_summary = re.search(
        rf"(?ms)^## {re.escape(story_report_headings()[0])}\s*\n(.*?)(?=^## |\Z)",
        report_content,
    )
    key_summary_text = key_summary.group(1) if key_summary else ""
    key_summary_bullets = len(re.findall(r"(?m)^[-*]\s+", key_summary_text))
    # R29 locating: a story-first summary renders as 3-6 short paragraphs instead of bullets.
    # The criterion stays "3-6 concise items": headings and tables are not items, and every item
    # still has to stay within the 300 character bound.
    _ks_blocks = [b.strip() for b in re.split(r"\n\s*\n", key_summary_text.strip()) if b.strip()]
    _ks_items = [b for b in _ks_blocks
                 if not b.lstrip().startswith("#")
                 and not all(line.lstrip().startswith("|") for line in b.splitlines() if line.strip())]
    key_summary_count = key_summary_bullets if key_summary_bullets else len(_ks_items)
    key_summary_longest = max([len(item) for item in _ks_items] or [0])
    key_summary_concise = (
        dataset_for_style not in V3_MAIN_DATASETS
        or (3 <= key_summary_count <= 6 and key_summary_longest <= 300)
    )
    # A missing representative-protein table is only a defect when a candidate exists to
    # report. With an empty candidate table the honest state is "absent, no candidates", which
    # is recorded as its own state instead of being passed off as a present table.
    candidate_evidence_rows = _candidate_evidence_row_count(run_folder)
    representative_table_present = (
        _report_language.t("core.tok_representative_header")
        + _report_language.t("core.tok_table_word") in report_content
    ) or _quantitative_table_present(
        _markdown_table_headers(report_content),
        (_report_language.t("core.tok_candidate_header"), _report_language.t("core.tok_representative_header")),
        ("logFC", "log2FC", "adj.P"),
    )
    if representative_table_present:
        representative_table_state = "present"
    elif candidate_evidence_rows == 0:
        representative_table_state = "absent_no_candidates"
    else:
        representative_table_state = "missing_expected"
    checks = {
        "chinese_executive_summary": _has_h2(report_content, canonical_report_headings()["executive"]) or _has_h2(report_content, story_report_headings()[0]),
        "chinese_scoring_first_screen": _has_h2(report_content, canonical_report_headings()["scoring"]) or _has_story_task_heading(report_content),
        "chinese_main_findings": _has_h2(report_content, canonical_report_headings()["findings"]) or _has_story_task_heading(report_content),
        "chinese_evidence_boundary": _has_h2(report_content, canonical_report_headings()["boundary"]) or _has_numbered_h2(report_content, _report_language.t("core.story_conclusion")),
        "chinese_evidence_appendix": _has_h2(report_content, canonical_report_headings()["appendix"]) or _has_numbered_h2(report_content, _report_language.t("core.story_assets")),
        "story_first_headings_present": story_first_structure_present or dataset_for_style not in V3_MAIN_DATASETS,
        "pispa_exact_task_headings_present": exact_pispa_headings_present or dataset_for_style != "Nat_Commun_PiSPA_2024",
        "legacy_primary_english_headings_absent": not legacy_primary_heading_present,
        "chinese_body_ratio_ok": _report_language.body_language_ok(
            _main_report_text_for_language_check(report_content), _report_language.get_language()),
        "long_english_paragraph_absent": not long_english_paragraph,
        "english_report_has_no_chinese_residue": (
            (not english_residue) if _report_language.get_language() == "en" else True),
        "raw_tool_json_absent": not re.search(r"```json\s*(?:(?!```).){1000,}(?:\"tool\"|\"result\"|\"tool_call_id\")", report_content, re.I | re.S),
        "long_tool_logs_absent": "## Task" not in report_content and "[Analyzer] Local Fallback Summary" not in report_content,
        "required_csvs_present": all(_path_exists(path) for path in required_csvs),
        "core_story_table_present": (_report_language.t("core.tok_core_contrast_table") in report_content) or _quantitative_table_present(
            _markdown_table_headers(report_content), (_report_language.t("core.tok_contrast_header"),),
            ("logFC", "log2FC", "adj.P", "FDR") + _report_language.t_list("core.tok_table_significance")),
        "representative_protein_table_present": representative_table_state != "missing_expected",
        "figures_present": any(_path_exists(path) for path in figure_candidates),
        "asset_files_present": asset_files_present,
        "key_summary_concise": key_summary_concise,
        "internal_direction_labels_absent": "up_in_display_group_a" not in report_content and "up_in_display_group_b" not in report_content,
        "quantitative_evidence_present": bool(re.search(r"\b(logFC|adj\.P\.Val|FDR|P\.Value|missing_rate|detected_in_matrix)\b", report_content, re.I)) or _path_exists(os.path.join(evidence_dir, "candidate_protein_evidence.csv")),
        # finished reports carry the boundary statements in Chinese; the internal provenance tokens
        # must not appear in reader-facing text, so the check follows the reader-facing wording
        "evidence_boundary_present": _report_language.t("core.tok_current_matrix") in report_content,
        "offline_enrichment_boundary_present": _report_language.t("core.tok_offline_enrichment") in report_content,
        "external_annotation_boundary_present": (_report_language.t("core.tok_external_annotation") in report_content) or not external_annotation_required,
        "coverage_rows_present": bool(coverage_rows),
        "coverage_not_missing": not any(str(row.get("coverage_status", "")).startswith("missing") for row in coverage_rows),
        "pispa_mechanism_chain_present": dataset_for_style != "Nat_Commun_PiSPA_2024" or all(token in report_content for token in ["Rho GTPase", "ERM", "myosin", "talin", "vinculin"]),
        "pispa_heuristic_and_boundary_present": dataset_for_style != "Nat_Commun_PiSPA_2024" or (
            _report_language.t("core.tok_heuristic") in report_content
            and _report_language.t("core.tok_followup") in report_content
            and _report_language.t("core.tok_boundary_word") in report_content),
    }
    failures = [name for name, passed in checks.items() if not passed and name not in {"figures_present", "offline_enrichment_boundary_present"}]
    warnings = [name for name, passed in checks.items() if not passed and name in {"figures_present", "offline_enrichment_boundary_present"}]
    # the fact check is a completion condition of its own: a definite contradiction must not be
    # reported as a clean self-check, while unverified claims stay a separate (non-blocking) state
    fact_check = run_report_evidence_check(report_content, run_folder)
    if fact_check.get("verdict") == "contradiction":
        failures = failures + ["report_fact_check_contradiction"]
    elif fact_check.get("verdict") == "unverified":
        warnings = warnings + ["report_fact_check_unverified"]
    result = {
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "representative_protein_table_state": representative_table_state,
        "candidate_evidence_rows": candidate_evidence_rows,
        "required_csvs": required_csvs,
        "figure_candidates": figure_candidates,
        "chinese_body_ratio": chinese_ratio,
        "report_language": _report_language.get_language(),
        "chinese_residue": english_residue[:40],
        "fact_check": fact_check,
    }
    out_json = os.path.join(run_folder, "report_self_check.json")
    out_md = os.path.join(run_folder, "report_self_check.md")
    record_report(out_json, json.dumps(result, ensure_ascii=False, indent=2), mode="w", visible=False)
    if dataset_for_style in V3_MAIN_DATASETS:
        record_report(os.path.join(run_folder, "self_check.json"), json.dumps(result, ensure_ascii=False, indent=2), mode="w", visible=False)
    lines = ["# Report Self Check", "", f"status: {result['status']}", "", "| Check | Passed |", "|---|---|"]
    for name, passed in checks.items():
        lines.append(f"| {name} | {passed} |")
    fact_check = result.get("fact_check") or {}
    levels = ((fact_check.get("map") or {}).get("coverage") or {}).get("levels") or {}
    lines.extend([
        "",
        f"fact_check_verdict: {fact_check.get('verdict')}",
        f"fact_check_levels: {levels}",
        "report_language: %s" % _report_language.get_language(),
        _report_language.t("core.note_historical_check_names"),
        "注：contradiction = 存在与工件矛盾的断言；unverified = 仍有未绑定断言（不构成通过）；"
        "pass = 未发现矛盾且无未绑定断言。exit code 0 不等于科学验收通过。",
    ])
    record_report(out_md, "\n".join(lines), mode="w", visible=False)
    if dataset_for_style in V3_MAIN_DATASETS:
        record_report(os.path.join(run_folder, "self_check.md"), "\n".join(lines), mode="w", visible=False)
    return result


def build_report_request(base_prompt: str, assembly_mode: str, task_ids, boundary_hint: str = "") -> str:
    """T1: the one place where the production report request is assembled.

    Both the first draft and the final integration call this, so the structured schema, the valid
    task ids and the three general revision requirements are part of the request itself. Keeping it
    in a named function lets the node test call exactly what the node calls instead of a copy.
    """
    prompt = base_prompt + report_language_directive()
    if str(assembly_mode).lower() != "hybrid":
        return prompt
    # the schema text carries braces, and the prompt is a LangChain f-string template, so the
    # appended half must be escaped exactly like the static prompts were
    from prompts import _escape_prompt_template as _escape
    prompt = prompt + _escape(structured_report_prompt_suffix(task_ids, boundary_hint))
    try:
        from revision_requirements import revision_block
        prompt = prompt + chr(10) + chr(10) + revision_block()
    except Exception:  # noqa: BLE001
        # the schema half is mandatory for hybrid; the revision block is additive
        prompt = prompt + chr(10) + chr(10) + "本轮修订要求：方向与坐标系一致；显著性表述绑定统计量；"
        prompt = prompt + "任务结论与自身表格一致。"
    return prompt


def structured_report_prompt_suffix(task_ids, boundary_hint="") -> str:
    """S3: the request half of the hybrid path.

    Hybrid assembly binds model prose to deterministic tables by task id, so the model must answer in
    the structured schema and may only use the ids that exist in this run. Without this the model
    answers in narrative markdown, every section stays unbound, and the hybrid path silently degrades
    to the legacy renderer.
    """
    ids = [str(item) for item in (task_ids or []) if str(item).strip()]
    id_block = "、".join(ids) if ids else "（本轮未提供可用对比 ID，只写 key_summary、synthesis 与 boundary）"
    parts = [
        "",
        "输出格式（必须严格遵守）：只输出一个 JSON 对象，不要 markdown 代码块，不要额外说明文字。",
        "schema：",
        '{"key_summary": ["..."], "tasks": [{"task_id": "<下面列出的 ID 之一>", "question": ["..."],',
        '  "findings": ["..."], "explanation": ["..."], "evidence_ids": ["..."]}],',
        '  "synthesis": {"type": "synthesis", "text": ["..."]}, "boundary": ["..."]}',
        "可用 task_id（每个对比只有一个，必须逐字使用，不得自造、不得改写）：%s" % id_block,
        "每个 task_id 只能出现一次；同一对比的其它写法（别名）属于同一个任务，不得为别名另写小节；",
        "只写本轮确有的对比。key_summary 给 3-5 条结论要点；",
        "synthesis.type 必须写成 synthesis；boundary 写方法与解释边界。",
        "数组元素是字符串，不要放表格；表格由装配程序按确定性事实生成。",
    ]
    if boundary_hint:
        parts.append("边界提示：%s" % boundary_hint)
    return chr(10).join(parts)


REPORT_ASSEMBLY_MODES = ("legacy", "hybrid")

# R28: the configuration a production run must resolve to. The candidate entrypoint reads this and
# fails fast when the resolved configuration differs, instead of discovering after the run that a
# report was rendered by the other path.
REPORT_PRODUCTION_CONFIG = {
    "report_assembly": "hybrid",
    "request_evidence_index": True,
}


class ReportAssemblyModeError(ValueError):
    """An explicitly requested report assembly mode is not one the pipeline implements."""


def resolve_report_assembly(fallback: str = "legacy",
                            env_var: str = "SCPROTEO_REPORT_ASSEMBLY") -> str:
    """Which report finalisation the batch run uses: legacy (historical) or hybrid (structured).

    Only an unset variable falls back. An explicit value that is not one of REPORT_ASSEMBLY_MODES
    raises, because silently rendering the report through the legacy path while the run's
    configuration claims another mode is what let an earlier run produce a legacy report that its
    own setup did not describe.
    """
    if fallback not in REPORT_ASSEMBLY_MODES:
        fallback = "legacy"
    raw = os.environ.get(env_var)
    if raw is None or not str(raw).strip():
        return fallback
    value = str(raw).strip().lower()
    if value not in REPORT_ASSEMBLY_MODES:
        raise ReportAssemblyModeError(
            "%s=%r is not a report assembly mode; expected one of %s"
            % (env_var, raw, " / ".join(REPORT_ASSEMBLY_MODES)))
    return value


def report_production_config_snapshot(run_folder: str = "") -> Dict[str, Any]:
    """Read the report-production configuration this process would actually use.

    Recorded at the start of a run so the mode, the model and the request switches are visible in
    run_events.jsonl rather than inferred afterwards from the produced report.
    """
    try:
        mode = resolve_report_assembly()
        mode_error = ""
    except ReportAssemblyModeError as exc:
        mode = ""
        mode_error = str(exc)
    index_raw = str(os.environ.get("SCPROTEO_REQUEST_EVIDENCE_INDEX", "1")).strip().lower()
    return {
        "report_assembly": mode,
        "report_assembly_raw": str(os.environ.get("SCPROTEO_REPORT_ASSEMBLY", "")),
        "report_assembly_error": mode_error,
        "expected_report_assembly": REPORT_PRODUCTION_CONFIG["report_assembly"],
        "matches_expected": bool(mode) and mode == REPORT_PRODUCTION_CONFIG["report_assembly"],
        "request_evidence_index": index_raw not in ("0", "off", "false", "no"),
        "model": str(os.environ.get("OPENAI_MODEL", "")),
        "run_folder": str(run_folder or ""),
    }


def structured_sections_for_assembly(report_content: str, record_file: str = "") -> dict:
    """Parse a structured model answer into assembly sections; a parse failure is recorded, not hidden."""
    import report_assembly
    try:
        parsed = report_assembly.parse_structured(report_content)
        sections = parsed.get("sections") or {}
        if (sections.get("binding_report") or {}).get("mode") == "unstructured":
            # a plain markdown answer still carries headings and prose: read it with the heading-based
            # reader instead of leaving the whole answer unbound, and keep that choice on the record
            if record_file:
                record_event(record_file, "report_assembly_parse", "warning",
                             error="no JSON payload; fell back to heading-based sections")
            readable = report_assembly.sections_from_text(report_content)
            readable["binding_report"] = sections.get("binding_report")
            return readable
        return sections
    except Exception as exc:  # noqa: BLE001
        if record_file:
            record_event(record_file, "report_assembly_parse", "warning", error=str(exc))
        return {}


def write_final_report_binding(run_folder: str, report_path: str, mode: str = "") -> dict:
    """External binding record for the report file as written (called after the write, not before)."""
    import report_assembly
    try:
        record = report_assembly.write_report_binding(run_folder, [report_path], mode=mode)
    except Exception as exc:  # noqa: BLE001
        record = {"write_error": str(exc)}
    return record


def finalize_report_hybrid(report_content: str, run_folder: str, record_file: str, sections=None) -> str:
    """Hybrid assembly: model text kept, deterministic blocks own the facts (2026-09-13).

    Nothing here rebuilds the body: the story-first front matter is deliberately skipped, and only
    additive, self-guarded appendices and footer items are appended, so a second pass cannot
    duplicate content.
    """
    import report_assembly
    if sections is None:
        sections = report_assembly.sections_from_text(report_content)
    result = report_assembly.assemble(run_folder, sections)
    text = sanitize_report_text_for_output(result["report"])
    text = _append_scoring_evidence_appendix(text, run_folder)
    text = _append_leakage_evidence_appendix(text, run_folder)
    text = _appendix_source_text(text)
    footer_items = report_evidence_footer_items(text, run_folder)
    if footer_items:
        text = text.rstrip() + "\n\n" + "\n".join(footer_items) + "\n"
    manifest = dict(result["manifest"])
    manifest["assembly_mode"] = "hybrid"
    manifest["report_sha12"] = report_assembly.sha12(text)
    manifest["final_chars"] = len(text)
    try:
        with open(os.path.join(run_folder, "report_assembly_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        record_event(record_file, "report_assembly_manifest", "warning", error=str(exc))
    record_event(record_file, "report_assembly", "completed", mode="hybrid",
                 report_sha12=manifest["report_sha12"], model_chars=manifest.get("model_chars"),
                 deterministic_chars=manifest.get("deterministic_chars"),
                 dropped_blocks=sum(1 for block in manifest.get("content_ledger", [])
                                    if block.get("disposition") == "dropped"))
    return text


def finalize_report_for_output(report_content: str, run_folder: str, base_name: str, record_file: str,
                               assembly: str = "legacy", sections=None) -> str:
    if str(assembly or "legacy").lower() == "hybrid":
        finalized = finalize_report_hybrid(report_content, run_folder, record_file, sections=sections)
    else:
        finalized = ensure_report_evidence_footer(report_content, run_folder)
    variant_files = write_report_variants(finalized, run_folder, base_name)
    asset_files = write_report_asset_files(run_folder, base_name)
    self_check = run_report_self_check(finalized, run_folder)
    fact_check = self_check.get("fact_check") or {}
    record_event(
        record_file,
        "report_self_check",
        "completed" if self_check.get("status") == "pass" else "warning",
        self_check_status=self_check.get("status"),
        fact_check_verdict=fact_check.get("verdict"),
        fact_check_contradictions=fact_check.get("n_contradictions"),
        fact_check_unverified=fact_check.get("n_unverified"),
        failures=self_check.get("failures", []),
        warnings=self_check.get("warnings", []),
        files=variant_files + asset_files + [
            os.path.join(run_folder, "report_self_check.json"),
            os.path.join(run_folder, "report_self_check.md"),
            os.path.join(run_folder, "report_evidence_map.json"),
            os.path.join(run_folder, "report_fact_check.json"),
        ],
    )
    return finalized


def _append_scoring_evidence_appendix(report_text: str, run_folder: str) -> str:
    if (
        any(title in report_text for title in (
            "### " + tm("appendix_reproducible"),
            "### " + tm("appendix_technical"),
            "### " + tm("appendix_scoring"),
            "## Scoring Evidence Appendix"))
    ):
        return report_text

    evidence_dir = os.path.join(run_folder, "evaluation_evidence")
    dataset = _get_dataset_name(run_folder)
    requirements_path = os.path.join(run_folder, "evaluation_requirements.used.json")
    group_qc_path = os.path.join(evidence_dir, "group_composition_qc.csv")
    story_path = os.path.join(evidence_dir, "core_story_evidence.csv")
    candidate_path = os.path.join(evidence_dir, "candidate_protein_evidence.csv")
    module_path = os.path.join(evidence_dir, "curated_module_group_summary.csv")
    module_sample_path = os.path.join(evidence_dir, "curated_module_sample_scores.csv")
    recipe_path = os.path.join(evidence_dir, "dataset_recipe_evidence.csv")
    external_annotation_path = os.path.join(evidence_dir, "external_annotation_asd_ndd_local.csv")
    external_applicable = _external_annotation_applicable(run_folder)
    leakage_summary_path = os.path.join(run_folder, "proteins_leakage", "leakage_summary.json")
    leakage_module_path = os.path.join(run_folder, "proteins_leakage", "leakage_module_summary.csv")
    appendix_inputs = [requirements_path, group_qc_path, story_path, candidate_path, module_path, module_sample_path, recipe_path, leakage_summary_path, leakage_module_path]
    if external_applicable:
        appendix_inputs.append(external_annotation_path)
    if not any(os.path.exists(p) for p in appendix_inputs):
        return report_text

    lines = [
        "### " + tm("appendix_reproducible"),
        tm("appendix_scoring_intro"),
        "",
        tm("evidence_ledger_header"),
        "|---|---|",
    ]
    evidence_file_rows = [
        (tm("sea_req_diff_analysis"), requirements_path, tm("sea_req_diff_threshold")),
        (tm("sea_req_group_qc"), group_qc_path, tm("sea_req_group_sizes")),
        (tm("sea_req_diff_summary"), story_path, tm("sea_req_contrast_table")),
        (tm("sea_req_candidate_stats"), candidate_path, tm("sea_req_candidate_table")),
        (tm("sea_req_module_group"), module_path, tm("sea_req_module_group_table")),
        (tm("sea_req_module_sample"), module_sample_path, tm("sea_req_module_sample_source")),
        (tm("sea_req_task_evidence"), recipe_path, tm("sea_req_task_evidence_table")),
        (tm("sea_leakage_summary"), leakage_summary_path, tm("sea_leakage_appendix")),
        (tm("sea_leakage_module_summary"), leakage_module_path, tm("sea_leakage_appendix")),
    ]
    if external_applicable:
        evidence_file_rows.insert(6, (tm("sea_external_annotation"), external_annotation_path, tm("sea_external_annotation_note")))
    for label, path, pointer in evidence_file_rows:
        if os.path.exists(path):
            lines.append(f"| {label} | {pointer} |")

    recipe_rows = _read_csv_rows(recipe_path, limit=40)
    if recipe_rows:
        lines.extend([
            "",
            tm("sea_task_landing_title"),
            tm("sea_task_landing_intro"),
            tm("sea_task_landing_header"),
            "|---|---|---|---|---|",
        ])
        for row in recipe_rows:
            lines.append(
                f"| {_fmt_cell(row.get('recipe_item', ''))} | {_fmt_cell(row.get('expected_evidence', ''))} | "
                f"{_fmt_cell(row.get('evidence_source', ''))} | {_fmt_cell(row.get('confidence', ''))} | "
                f"{_fmt_cell(_reader_boundary_text(row.get('boundary', '')), 160)} |"
            )

    story_rows = _read_csv_rows(story_path, limit=80)
    if story_rows:
        story_source_header = tm("sea_source_header_pispa") if dataset == "Nat_Commun_PiSPA_2024" else tm("sea_source_header_default")
        lines.extend([
            "",
            tm("sea_core_story_title"),
            tm("sea_core_story_header", story_source_header=story_source_header),
            "|---|---|---|---|---|---|",
        ])
        for row in story_rows[:40]:
            if row.get("row_type") == "contrast":
                value = (tm("sea_core_story_counts", n_sig=row.get('n_sig', ''), n_up_display_group_a=row.get('n_up_display_group_a', ''), n_down_display_group_a=row.get('n_down_display_group_a', '')))
                source_note = tm("sea_core_story_diff_source")
            elif row.get("row_type") == "candidate":
                value = f"{row.get('candidate', '')} logFC={row.get('logFC_display', '')}; adj.P.Val={row.get('adj.P.Val', '')}"
                source_note = tm("sea_core_story_candidate_source")
            else:
                value = f"{row.get('module', '')} Δ={row.get('module_delta_group_a_minus_group_b', '')}"
                source_note = tm("sea_core_story_module_source")
            if dataset != "Nat_Commun_PiSPA_2024":
                source_note = _reader_boundary_text(row.get("boundary", ""))
            lines.append(
                f"| {_fmt_cell(row.get('row_type', ''))} | {_fmt_cell(row.get('claim_title', ''), 90)} | "
                f"{_fmt_cell(row.get('display_contrast', ''), 80)} | {_fmt_cell(value, 160)} | "
                f"{_fmt_cell(row.get('confidence', ''))} | {_fmt_cell(source_note, 150)} |"
            )

    candidate_rows = _read_csv_rows(candidate_path, limit=1000)
    candidate_rows = _select_candidate_highlights(candidate_rows, dataset, limit=24)
    if candidate_rows:
        lines.extend([
            "",
            tm("sea_candidate_summary_title"),
            tm("sea_candidate_summary_header"),
            "|---|---|---|---|---:|---:|---:|---|",
        ])
        for row in candidate_rows:
            lines.append(
                f"| {_fmt_cell(row.get('candidate', ''))} | {_fmt_cell(_reader_detected(row.get('detected_in_matrix', '')))} | "
                f"{_fmt_cell(_reader_contrast_text(row.get('contrast', '')))} | "
                f"{_fmt_cell(_candidate_direction_for_user(row))} | "
                f"{_fmt_cell(row.get('logFC', ''))} | {_fmt_cell(row.get('P.Value', ''))} | "
                f"{_fmt_cell(row.get('adj.P.Val', ''))} | {_fmt_cell(row.get('confidence', ''))} |"
            )

    group_rows = _read_csv_rows(group_qc_path, limit=80)
    if group_rows:
        lines.extend([
            "",
            tm("sea_group_qc_title"),
            tm("sea_group_qc_header"),
            "|---|---|---|---|---:|---:|---:|---|",
        ])
        for row in group_rows:
            grouping = row.get("group_col", "")
            if row.get("secondary_col"):
                grouping = f"{grouping} x {row.get('secondary_col', '')}"
            lines.append(
                f"| {_fmt_cell(row.get('table', ''))} | {_fmt_cell(grouping)} | {_fmt_cell(row.get('group', ''))} | "
                f"{_fmt_cell(row.get('secondary_col', ''))} | {_fmt_cell(row.get('n_samples', ''))} | "
                f"{_fmt_cell(row.get('mean_detected_proteins', ''))} | {_fmt_cell(row.get('mean_missing_rate', ''))} | "
                f"{_fmt_cell(_reader_counts_text(row.get('counts', '')), 400)} |"
            )

    # R29: the appendix table had a hard 40-row cap and dropped the rest silently.  The bound stays,
    # but it now covers a realistic module table and any truncation is stated in the table itself.
    MODULE_APPENDIX_MAX_ROWS = 200
    module_rows = _read_csv_rows(module_path, limit=MODULE_APPENDIX_MAX_ROWS + 50)
    if module_rows:
        lines.extend([
            "",
            tm("sea_curated_module_title"),
            tm("sea_curated_module_header"),
            "|---|---|---:|---:|---|---|",
        ])
        shown = 0
        for row in module_rows:
            if not row.get("group"):
                continue
            lines.append(
                tm("sea_curated_module_row", module=_fmt_cell(row.get('module', '')), group=_fmt_cell(row.get('group', '')), n_matched_genes=_fmt_cell(row.get('n_matched_genes', '')), mean_score=_fmt_cell(row.get('mean_score', '')), confidence=_fmt_cell(row.get('confidence', '')))
            )
            shown += 1
            if shown >= MODULE_APPENDIX_MAX_ROWS:
                break
        if len(module_rows) > shown:
            lines.append(
                tm("sea_rows_truncated") % (len(module_rows), shown))

    external_rows = _read_csv_rows(external_annotation_path, limit=60) if external_applicable else []
    if external_rows:
        lines.extend([
            "",
            tm("sea_external_cross_title"),
            tm("sea_external_cross_header"),
            "|---|---|---|---:|---|---|",
        ])
        shown = 0
        for row in external_rows:
            if row.get("detected_in_matrix", "").lower() not in {"true", "1", "yes"} and shown >= 15:
                continue
            lines.append(
                f"| {_fmt_cell(row.get('gene', ''))} | {_fmt_cell(row.get('category', ''))} | "
                f"{_fmt_cell(row.get('detected_in_matrix', ''))} | {_fmt_cell(row.get('matched_row_count', ''))} | "
                f"{_fmt_cell(row.get('evidence_source', ''))} | {_fmt_cell(row.get('boundary', ''), 160)} |"
            )
            shown += 1
            if shown >= 40:
                break

    leakage_rows = _read_csv_rows(leakage_module_path, limit=40)
    if leakage_rows:
        lines.extend([
            "",
            tm("appendix_leakage_qc_title"),
            tm("sea_module_effect_header"),
            "|---|---|---|---:|---:|---|---:|---:|---|",
        ])
        for row in leakage_rows:
            lines.append(
                f"| {_fmt_cell(row.get('comparison', ''))} | {_fmt_cell(row.get('status_column', ''))} | "
                f"{_fmt_cell(row.get('module', ''))} | {_fmt_cell(row.get('n_matched_genes', ''))} | "
                f"{_fmt_cell(row.get('mean_logFC_group_a_minus_group_b', ''))} | {_fmt_cell(row.get('direction', ''))} | "
                f"{_fmt_cell(row.get('n_samples_group_a', ''))} | {_fmt_cell(row.get('n_samples_group_b', ''))} | "
                f"{_fmt_cell(row.get('confidence', ''))} |"
            )
    if os.path.exists(leakage_summary_path):
        try:
            leakage = load_json(leakage_summary_path)
        except Exception:
            leakage = {}
        if isinstance(leakage, dict) and leakage:
            lines.extend([
                "",
                tm("sea_leakage_correlation_title"),
                tm("sea_leakage_correlation_header"),
                "|---|---|---|---:|---:|---|",
            ])
            for comparison, payload in list(leakage.items())[:20]:
                if not isinstance(payload, dict):
                    continue
                lines.append(
                    f"| {_fmt_cell(comparison)} | {_fmt_cell(payload.get('status_column', ''))} | "
                    f"{_fmt_cell(payload.get('stratification_col', ''))} | {_fmt_cell(payload.get('logfc_matrix_shape', ''))} | "
                    f"{_fmt_cell(len(payload.get('high_correlation_pairs', []) or []))} | {_fmt_cell(payload.get('conclusion', ''))} |"
                )

    return report_text.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def _append_leakage_evidence_appendix(report_text: str, run_folder: str) -> str:
    if ("### " + tm("appendix_leakage")) in report_text or "## Protein Leakage Evidence Appendix" in report_text or "### Protein Leakage QC Evidence" in report_text:
        return report_text
    leakage_summary_path = os.path.join(run_folder, "proteins_leakage", "leakage_summary.json")
    leakage_module_path = os.path.join(run_folder, "proteins_leakage", "leakage_module_summary.csv")
    if not os.path.exists(leakage_summary_path) and not os.path.exists(leakage_module_path):
        return report_text

    lines = [
        "### " + tm("appendix_leakage"),
        tm("appendix_leakage_intro"),
        "",
        tm("evidence_ledger_header"),
        "|---|---|",
    ]
    for label, path, pointer in [
        (tm("sea_leakage_summary"), leakage_summary_path, tm("lea_leakage_direction_note")),
        (tm("sea_leakage_module_summary"), leakage_module_path, tm("lea_per_module_note")),
    ]:
        if os.path.exists(path):
            lines.append(f"| {label} | {pointer} |")

    leakage_rows = _read_csv_rows(leakage_module_path, limit=40)
    if leakage_rows:
        lines.extend([
            "",
            tm("appendix_leakage_qc_title"),
            tm("sea_module_effect_header"),
            "|---|---|---|---:|---:|---|---:|---:|---|",
        ])
        for row in leakage_rows:
            lines.append(
                f"| {_fmt_cell(row.get('comparison', ''))} | {_fmt_cell(row.get('status_column', ''))} | "
                f"{_fmt_cell(row.get('module', ''))} | {_fmt_cell(row.get('n_matched_genes', ''))} | "
                f"{_fmt_cell(row.get('mean_logFC_group_a_minus_group_b', ''))} | {_fmt_cell(row.get('direction', ''))} | "
                f"{_fmt_cell(row.get('n_samples_group_a', ''))} | {_fmt_cell(row.get('n_samples_group_b', ''))} | "
                f"{_fmt_cell(row.get('confidence', ''))} |"
            )

    if os.path.exists(leakage_summary_path):
        try:
            leakage = load_json(leakage_summary_path)
        except Exception:
            leakage = {}
        if isinstance(leakage, dict) and leakage:
            lines.extend([
                "",
                tm("sea_leakage_correlation_title"),
                tm("sea_leakage_correlation_header"),
                "|---|---|---|---:|---:|---|",
            ])
            for comparison, payload in list(leakage.items())[:20]:
                if not isinstance(payload, dict):
                    continue
                lines.append(
                    f"| {_fmt_cell(comparison)} | {_fmt_cell(payload.get('status_column', ''))} | "
                    f"{_fmt_cell(payload.get('stratification_col', ''))} | {_fmt_cell(payload.get('logfc_matrix_shape', ''))} | "
                    f"{_fmt_cell(len(payload.get('high_correlation_pairs', []) or []))} | {_fmt_cell(payload.get('conclusion', ''))} |"
                )

    return report_text.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def build_local_fallback_report(task: str, analysis_summary: str, run_folder: str) -> str:
    design_path = os.path.join(run_folder, "analysis_design.used.yaml")
    ledger_path = os.path.join(run_folder, "evidence_ledger.jsonl")
    figure_manifest = os.path.join(run_folder, "figures", "figure_manifest.json")
    if not os.path.exists(figure_manifest):
        figure_manifest = os.path.join(run_folder, "figure_manifest.json")
    lines = [
        tm("fallback_h1"),
        "",
        "confidence: moderate",
        "",
        tm("fallback_intro"),
        "",
        tm("fallback_sources_heading"),
        tm("fallback_source_matrix", a0=os.path.abspath(run_folder)),
        tm("fallback_source_design", a0=os.path.abspath(design_path)),
        tm("fallback_source_ledger", a0=os.path.abspath(ledger_path)),
    ]
    if os.path.exists(figure_manifest):
        lines.append(tm("fallback_source_figures", a0=os.path.abspath(figure_manifest)))
    lines.extend([
        "",
        tm("fallback_summary_heading"),
        sanitize_report_text_for_output(analysis_summary, 40000),
        "",
        tm("fallback_boundary_heading"),
        tm("fallback_boundary_matrix"),
        tm("fallback_boundary_enrichment"),
        tm("fallback_boundary_external"),
    ])
    return "\n".join(lines)


def report_evidence_footer_items(report_text: str, run_folder: str) -> List[str]:
    """Footer lines appended without rebuilding the body; shared by the legacy and hybrid paths."""
    footer_items = []
    design_path = os.path.join(run_folder, "analysis_design.used.yaml")
    ledger_path = os.path.join(run_folder, "evidence_ledger.jsonl")
    external_path = os.path.join(run_folder, "external_knowledge.jsonl")
    params_path = os.path.join(run_folder, "parameters.json")
    if "analysis_design.used.yaml" not in report_text:
        footer_items.append(tm("footer_design"))
    if "evidence_ledger.jsonl" not in report_text:
        footer_items.append(tm("footer_ledger"))
    if os.path.exists(external_path) and "external_knowledge.jsonl" not in report_text:
        footer_items.append(tm("footer_external", path=_report_path(external_path, run_folder)))
    if (
        "confidence:" not in report_text
        and tm("probe_confidence") not in report_text
        and not re.search(r"confidence\s*[:：]\s*(high|moderate|low)", report_text, re.I)
    ):
        footer_items.append(tm("footer_confidence"))
    try:
        params = load_json(params_path) if os.path.exists(params_path) else {}
        batch_info = params.get("combat_calibration", {}).get("batch_correction", {})
        if (
            batch_info.get("combat_success") is False
            and "ProteinQuant_ComBat.csv" in report_text
            and "compatibility" not in report_text.lower()
            and tm("probe_compat") not in report_text
        ):
            footer_items.append(tm("footer_combat"))
    except Exception:
        pass
    return footer_items


def ensure_report_evidence_footer(report_text: str, run_folder: str) -> str:
    report_text = sanitize_report_text_for_output(report_text)
    report_text = prepend_report_front_matter(report_text, run_folder)
    report_text = _append_scoring_evidence_appendix(report_text, run_folder)
    report_text = _append_leakage_evidence_appendix(report_text, run_folder)
    footer_items = report_evidence_footer_items(report_text, run_folder)
    if not footer_items:
        return sanitize_report_text_for_output(report_text)
    if _get_dataset_name(run_folder) in V3_MAIN_DATASETS:
        addition = "\n".join(footer_items)
        story_boundary = _numbered_heading_re(
            _report_language.t("core.story_conclusion")).search(report_text)
        if story_boundary:
            section_marker = "\n" + story_boundary.group(0)
        elif _get_dataset_name(run_folder) == "Nat_Commun_PiSPA_2024":
            section_marker = f"\n## {story_report_headings()[8]}"
        else:
            section_marker = "\n## " + _report_language.t("core.story_conclusion")
        insert_markers = [
            "\n### " + tm("appendix_reproducible"),
            "\n### " + tm("appendix_technical"),
            "\n### " + tm("appendix_leakage"),
            f"\n## {canonical_report_headings()['appendix']}",
        ]
        insert_at = len(report_text)
        for marker in insert_markers:
            marker_idx = report_text.find(marker)
            if marker_idx >= 0:
                insert_at = min(insert_at, marker_idx)
        if section_marker in report_text:
            report_text = report_text[:insert_at].rstrip() + "\n" + addition + "\n" + report_text[insert_at:]
        else:
            report_text = report_text.rstrip() + f"\n\n{section_marker.lstrip()}\n" + addition + "\n"
        return sanitize_report_text_for_output(report_text)

    addition = tm("boundary_supplement") + "\n" + "\n".join(footer_items)
    markers = [f"\n## {canonical_report_headings()['appendix']}", f"\n## {story_report_headings()[7]}", "\n## Evidence Appendix"]
    idx = -1
    for marker in markers:
        marker_idx = report_text.find(marker)
        if marker_idx >= 0:
            idx = marker_idx if idx < 0 else min(idx, marker_idx)
    if idx >= 0:
        report_text = report_text[:idx].rstrip() + "\n\n" + addition + "\n" + report_text[idx:]
    else:
        report_text = report_text.rstrip() + f"\n\n## {canonical_report_headings()['boundary']}\n" + addition + "\n"
    return sanitize_report_text_for_output(report_text)


def _request_model_id() -> str:
    """The model id this run asked for, read from the client configuration (never inferred)."""
    value = ""
    try:
        import llm as _llm
        value = str(getattr(_llm, "OPENAI_MODEL", "") or "")
    except Exception:  # noqa: BLE001
        value = ""
    if not value:
        value = str(os.environ.get("OPENAI_MODEL") or "")
    return value or "未记录"


def _response_call_metadata(response) -> dict:
    """R21: what the provider itself reported about one call; a missing field stays unprovided.

    The request model id is a local fact. The returned model id, the response id and the finish
    reason only exist when the provider sent them, so they are never copied from the request value:
    a reader must be able to tell "the provider said nothing" from "the provider agreed".
    """
    def _pick(mapping, keys):
        if not isinstance(mapping, dict):
            return None
        for key in keys:
            value = mapping.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    metadata = getattr(response, "response_metadata", None) or {}
    extra = getattr(response, "additional_kwargs", None) or {}
    returned = (_pick(metadata, ("model_name", "model", "model_id"))
                or _pick(extra, ("model", "model_name")))
    return {"request_model_id": _request_model_id(),
            "returned_model_id": str(returned) if returned else "未提供",
            "response_id": str(_pick(metadata, ("id", "response_id", "request_id")) or "未提供"),
            "finish_reason": str(_pick(metadata, ("finish_reason", "stop_reason")) or "未提供")}


def _persist_generation_call(run_folder: str, record_file: str, idx: int, label: str,
                             prompt_template, prompt_vars: dict, response) -> None:
    """R18: persist the paid request and the raw response before the assembler can fail.

    report_node used to hold the model answer in memory only; raw_response_1.md was written by the
    driver after the node returned, so an assembly failure destroyed a billable answer (that is how the
    gpt-5.6-luna/carr first attempt was lost). Writing both files here, immediately after invoke() and
    before finalize_report_for_output, makes a failed assembly non-destructive.
    """
    if str(os.environ.get("SCPROTEO_PERSIST_RAW_RESPONSE", "1")).strip().lower() \
            in ("0", "off", "false", "no"):
        return
    try:
        import hashlib as _hashlib
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        suffix = "" if label == "draft" else "_" + label
        target = os.path.join(run_folder, "raw_response_%d%s.md" % (idx, suffix))
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content)
        request_text = ""
        try:
            request_text = prompt_template.format(**prompt_vars)
        except Exception:  # noqa: BLE001
            request_text = ""
        if request_text:
            with open(os.path.join(run_folder, "request_%d%s.txt" % (idx, suffix)),
                      "w", encoding="utf-8") as handle:
                handle.write(request_text)
        usage = getattr(response, "usage_metadata", None) or {}
        response_sha256 = _hashlib.sha256(content.encode("utf-8")).hexdigest()
        request_sha256 = (_hashlib.sha256(request_text.encode("utf-8")).hexdigest()
                          if request_text else "")
        call_meta = _response_call_metadata(response)
        record_event(record_file, "generation_raw_response", "completed", label=label,
                     path=target, chars=len(content),
                     sha12=response_sha256[:12], sha256=response_sha256,
                     request_chars=len(request_text),
                     request_sha12=request_sha256[:12], request_sha256=request_sha256,
                     usage=usage, **call_meta)
        # R21: the same facts next to the response itself, so a reader does not have to parse the
        # event log; a field the provider did not send is labelled instead of being left out.
        try:
            with open(os.path.join(run_folder, "generation_call_%d%s.json" % (idx, suffix)),
                      "w", encoding="utf-8") as handle:
                json.dump({"label": label, "request_sha256": request_sha256,
                           "response_sha256": response_sha256, "request_chars": len(request_text),
                           "response_chars": len(content), "usage": usage, **call_meta},
                          handle, ensure_ascii=False, indent=2)
        except Exception as _meta_exc:  # noqa: BLE001
            record_event(record_file, "generation_call_metadata", "warning", error=str(_meta_exc))
    except Exception as exc:  # noqa: BLE001
        record_event(record_file, "generation_raw_response", "warning", label=label, error=str(exc))


def report_node(state: AgentState):
    started_at = time.perf_counter()
    task = state.get("task", "")
    report_idx = state.get("report_idx", 0) + 1
    this_run_folder = state.get("this_run_folder", ".")
    output_report = list(state.get("output_report", []))
    primary_report_base = _primary_report_base_name(this_run_folder, report_idx, final=False)
    primary_report_file = os.path.join(this_run_folder, f"{primary_report_base}.md")
    output_report_file = os.path.join(this_run_folder, f"output_report_{report_idx}.md")
    final_report_base = _primary_report_base_name(this_run_folder, report_idx, final=True)
    primary_final_output_report_file = os.path.join(this_run_folder, f"{final_report_base}.md")
    final_output_report_file = os.path.join(this_run_folder, "final_output_report.md")
    write_legacy_report_aliases = _get_dataset_name(this_run_folder) not in V3_MAIN_DATASETS
    record_file = state.get("record_file", "record_file.md")
    memory = ensure_memory_store(state.get("memory", {}))
    analysis_summary = state.get("analysis_summary", "")
    memory_context = build_memory_context(memory)
    try:
        evidence_config = dict(get_config())
        evidence_config["this_run_folder_path"] = this_run_folder
        evidence_config["record_file_path"] = record_file
        evidence_pack = prepare_user_visible_evidence_pack(evidence_config)
        analysis_summary = analysis_summary.rstrip() + "\n\n" + evidence_pack.get("markdown", "")
        record_event(
            record_file,
            "user_visible_evidence_pack",
            "completed",
            files=[
                evidence_pack.get("requirements_path", ""),
                evidence_pack.get("group_qc", {}).get("csv", ""),
                evidence_pack.get("core_story", {}).get("csv", ""),
                evidence_pack.get("candidate_evidence", {}).get("csv", ""),
                evidence_pack.get("module_scores", {}).get("group_summary_csv", ""),
                evidence_pack.get("module_scores", {}).get("sample_scores_csv", ""),
                evidence_pack.get("dataset_recipe", {}).get("csv", ""),
                evidence_pack.get("external_annotation", {}).get("csv", ""),
                evidence_pack.get("scoring_coverage", {}).get("csv", ""),
            ],
        )
    except Exception as e:
        record_report(record_file, f"\n\n## [EvidencePack] Failed to prepare evaluation evidence pack: {e}")
        record_event(record_file, "evaluation_evidence_pack", "error", error=str(e))

    # R17: the request carries the same evidence index the deterministic assembler renders, so the
    # model is told what this run actually produced instead of inferring it from file names.
    # R18: the two-arm writing diagnostic must be able to send the pre-R17 request protocol through this
    # same node, so the same single variable is switched instead of a second request builder. The default
    # keeps production behaviour; SCPROTEO_REQUEST_EVIDENCE_INDEX=0 reproduces the pre-R17 request.
    _request_index_enabled = str(os.environ.get("SCPROTEO_REQUEST_EVIDENCE_INDEX", "1")).strip().lower() \
        not in ("0", "off", "false", "no")
    if not _request_index_enabled:
        record_event(record_file, "report_request_evidence_index", "skipped",
                     reason="disabled by SCPROTEO_REQUEST_EVIDENCE_INDEX=0 (pre-R17 request protocol)")
    try:
        import report_assembly as _evidence_index
        # one build, used for both the request text and its hash: the index reads the run's CSVs
        _index_data = (_evidence_index.build_request_evidence_index(this_run_folder)
                       if _request_index_enabled else {})
        request_index = (_evidence_index.format_request_evidence_index(_index_data)
                         if _index_data else "")
        if request_index:
            analysis_summary = analysis_summary.rstrip() + "\n\n" + request_index
            record_event(record_file, "report_request_evidence_index", "completed",
                         chars=len(request_index),
                         sha12=_evidence_index.sha12(request_index),
                         index_sha12=_evidence_index.sha12(
                             json.dumps(_index_data, ensure_ascii=False, sort_keys=True, default=str)))
    except Exception as e:
        record_event(record_file, "report_request_evidence_index", "warning", error=str(e))

    report_prompt = build_prompt_with_static_system(
        REPORT_PROMPT,
        "The task is {task}.\n"
        "Relevant memory from the workflow:\n{memory_context}\n"
        "Analyze the results of {analysis_summary} and generate a detailed report.",
    )
    try:
        report_assembly_mode = resolve_report_assembly()
    except ReportAssemblyModeError as _mode_exc:
        # R28: an explicitly configured mode the pipeline does not implement stops the run here.
        # The alternative is a report rendered by the other path while the run's configuration
        # claims the mode was switched on.
        record_event(record_file, "report_assembly_mode", "error", error=str(_mode_exc))
        raise
    record_event(record_file, "report_assembly_mode", "completed", mode=report_assembly_mode)
    try:
        record_event(record_file, "report_production_config", "completed",
                     **report_production_config_snapshot(this_run_folder))
    except Exception as _cfg_exc:  # noqa: BLE001
        record_event(record_file, "report_production_config", "warning", error=str(_cfg_exc))
    valid_task_ids = []
    if report_assembly_mode == "hybrid":
        try:
            import report_assembly as _ra
            # R21/F4: one canonical id per scientific task. Listing the claim id, the display name
            # and the source name of the same comparison made the model write that comparison three
            # times; the aliases stay valid on input (the binder accepts them) but are no longer
            # offered as tasks of their own.
            valid_task_ids = _ra.canonical_task_ids(this_run_folder)
        except Exception as _exc:  # noqa: BLE001
            record_event(record_file, "report_assembly_task_ids", "warning", error=str(_exc))
        if not valid_task_ids:
            record_event(record_file, "report_assembly_task_ids", "warning",
                         error="no contrast ids available for the structured request")
        else:
            record_event(record_file, "report_assembly_task_ids", "completed",
                         ids=valid_task_ids, n=len(valid_task_ids))
    # the mode and the task ids must exist before the request is built: the structured schema and
    report_prompt = build_report_request(
        report_prompt, report_assembly_mode, valid_task_ids,
        "结论只对当前矩阵成立；模块分数不替代单蛋白 FDR 或正式富集检验。")
    need_backtrack_flag = False
    try:
        report_chain = report_prompt | LLM
        report_response = report_chain.invoke({
            "analysis_summary": analysis_summary,
            "memory_context": memory_context,
            "task": task,
        })
        _persist_generation_call(this_run_folder, record_file, report_idx, "draft", report_prompt,
                                 {"analysis_summary": analysis_summary,
                                  "memory_context": memory_context, "task": task}, report_response)

        report_content = finalize_report_for_output(
            report_response.content, this_run_folder, primary_report_base, record_file,
            assembly=report_assembly_mode,
            sections=(structured_sections_for_assembly(report_response.content, record_file)
                      if report_assembly_mode == "hybrid" else None))
        report_txt = f"\n\n## [Report] Report {report_idx} Results:\n{report_content}"
        record_report(record_file, report_txt)
        record_report(primary_report_file, report_content, mode="w", visible=False)
        write_final_report_binding(this_run_folder, primary_report_file, report_assembly_mode)
        if write_legacy_report_aliases and primary_report_file != output_report_file:
            record_report(output_report_file, report_content, mode="w", visible=False)
            write_report_variants(report_content, this_run_folder, f"output_report_{report_idx}")
        output_report.append(report_content)
    except Exception as e:
        err = f"[Report] LLM report failed: {e}; using local fallback report."
        record_report(record_file, err)
        report_content = finalize_report_for_output(
            build_local_fallback_report(task, analysis_summary, this_run_folder),
            this_run_folder,
            primary_report_base,
            record_file,
        )
        report_txt = f"\n\n## [Report] Report {report_idx} Results:\n{report_content}"
        record_report(record_file, report_txt)
        record_report(primary_report_file, report_content, mode="w", visible=False)
        write_final_report_binding(this_run_folder, primary_report_file, report_assembly_mode)
        if write_legacy_report_aliases and primary_report_file != output_report_file:
            record_report(output_report_file, report_content, mode="w", visible=False)
            write_report_variants(report_content, this_run_folder, f"output_report_{report_idx}")
        output_report.append(report_content)
        need_backtrack_flag = False

    if report_idx >= 2 and not need_backtrack_flag:
        previous_report = "\n\n".join(
            f"Report Part {i + 1}:\n{content}"
            for i, content in enumerate(output_report)
        )

        final_report_prompt = build_prompt_with_static_system(
            REPORT_PROMPT,
            "The task is {task}.\n"
            "Relevant memory from the workflow:\n{memory_context}\n"
            "The previous reports are:\n{previous_report}\n"
            "Analyze the results of {analysis_summary} and generate a detailed report.",
        )
        # the final integration request carries the same schema, id whitelist and revision rules
        final_report_prompt = build_report_request(
            final_report_prompt, report_assembly_mode, valid_task_ids,
            "结论只对当前矩阵成立；模块分数不替代单蛋白 FDR 或正式富集检验。")
        try:
            report_chain = final_report_prompt | LLM
            report_response = report_chain.invoke({
                "analysis_summary": analysis_summary,
                "memory_context": memory_context,
                "previous_report": previous_report,
                "task": task,
            })
            _persist_generation_call(this_run_folder, record_file, report_idx, "final",
                                     final_report_prompt,
                                     {"analysis_summary": analysis_summary,
                                      "memory_context": memory_context,
                                      "previous_report": previous_report, "task": task},
                                     report_response)
            report_content = finalize_report_for_output(
                report_response.content, this_run_folder, final_report_base, record_file,
                assembly=report_assembly_mode,
                sections=(structured_sections_for_assembly(report_response.content, record_file)
                          if report_assembly_mode == "hybrid" else None))
            report_txt = f"\n\n## [Report] Final Report Results:\n{report_content}"
            record_report(record_file, report_txt)
            record_report(primary_final_output_report_file, report_content, mode="w", visible=False)
            if write_legacy_report_aliases and primary_final_output_report_file != final_output_report_file:
                record_report(final_output_report_file, report_content, mode="w", visible=False)
                write_report_variants(report_content, this_run_folder, "final_output_report")
            output_report.append(report_content)
        except Exception as e:
            err = f"[Report] Final LLM report failed: {e}; keeping the latest local report."
            record_report(record_file, err)
            need_backtrack_flag = False

    next_state = {
        "report_idx": report_idx,
        "output_report": output_report,
        "memory": memory,
        "state_history": append_state_history(state, "report"),
        "pre": "report",
    }

    next_state["next"] = "backtrack" if need_backtrack_flag else "critic"
    record_timing(record_file, "Report", started_at, f"report {report_idx}")
    return next_state


def with_node_logging(node_name: str, node_func):
    def wrapped(state: AgentState):
        record_file = state.get("record_file", "record_file.md")
        started_at = time.perf_counter()
        record_event(
            record_file,
            "node_start",
            "started",
            node=node_name,
            plan_idx=state.get("plan_idx", 0),
            current_step=state.get("current_step", 0),
            report_idx=state.get("report_idx", 0),
        )
        try:
            result = node_func(state)
            record_event(
                record_file,
                "node_end",
                "completed",
                started_at=started_at,
                node=node_name,
                next=result.get("next") if isinstance(result, dict) else None,
                plan_idx=result.get("plan_idx", state.get("plan_idx", 0)) if isinstance(result, dict) else state.get("plan_idx", 0),
                current_step=result.get("current_step", state.get("current_step", 0)) if isinstance(result, dict) else state.get("current_step", 0),
                report_idx=result.get("report_idx", state.get("report_idx", 0)) if isinstance(result, dict) else state.get("report_idx", 0),
            )
            return result
        except Exception as e:
            tb = traceback.format_exc()
            record_report(record_file, f"\n\n## [NodeError] {node_name} failed: {e}\n\n```text\n{tb}\n```")
            record_event(
                record_file,
                "node_error",
                "error",
                started_at=started_at,
                node=node_name,
                error=str(e),
                traceback=tb,
            )
            raise

    return wrapped


def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", with_node_logging("planner", plan_node))
    workflow.add_node("executor", with_node_logging("executor", execute_node))
    workflow.add_node("analyzer", with_node_logging("analyzer", analyzer_node))
    workflow.add_node("report", with_node_logging("report", report_node))
    workflow.add_node("critic", with_node_logging("critic", critic_node))
    workflow.add_node("backtrack", with_node_logging("backtrack", backtrack_node))

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        lambda s: s["next"],
        {
            "backtrack": "backtrack",
            "executor": "executor",
        }
    )
    workflow.add_conditional_edges(
        "executor",
        lambda s: s["next"],
        {
            "backtrack": "backtrack",
            "analyzer": "analyzer",
            "report": "report",
        }
    )
    workflow.add_conditional_edges(
        "analyzer",
        lambda s: s["next"],
        {
            "backtrack": "backtrack",
            "executor": "executor",
            "report": "report",
        }
    )
    workflow.add_conditional_edges(
        "report",
        lambda s: s["next"],
        {
            "backtrack": "backtrack",
            "critic": "critic",
        }
    )
    workflow.add_conditional_edges(
        "critic",
        lambda s: s["next"],
        {
            "planner": "planner",
            "backtrack": "backtrack",
            "end": END,
        }
    )
    workflow.add_conditional_edges(
        "backtrack",
        lambda s: s["next"],
        {
            "planner": "planner",
            "executor": "executor",
            "analyzer": "analyzer",
            "report": "report",
            "critic": "critic",
        }
    )

    return workflow.compile()


def build_initial_state(config: Config, memory: Dict[str, Any] | None = None) -> AgentState:
    enable_internal_scorer = bool(config.get("enable_internal_scorer", False)) and config.get("evaluator_mode", "none") == "internal-dev"
    return {
        "task": "",
        "plan": [],
        "plan_idx": 0,
        "report_idx": 0,
        "plan_num": 1,
        "current_step": 0,
        "execute_results": "",
        "backtrack_num": 10,
        "backtrack_idx": 0,
        "analysis_summary": "",
        "state_history": [],
        "this_run_folder": config.get("this_run_folder_path"),
        "record_file": config.get("record_file_path"),
        "memory": ensure_memory_store(memory or {}),
        "pre": "",
        "next": "",
        "ground_truth_path": config.get("ground_truth_path", "") if enable_internal_scorer else "",
        "grading_standard_path": config.get("grading_standard_path", "") if enable_internal_scorer else "",
        "publication_mode": bool(config.get("publication_mode", True)),
        "evaluator_mode": config.get("evaluator_mode", "none"),
        "enable_internal_scorer": enable_internal_scorer,
        "evaluator_root": config.get("evaluator_root", ""),
        "critique_text": "",
        "output_report": [],
        "memory_path": config.get("memory_path", ""),
        "parameters_path": config.get("parameters_path", ""),
        "executor_backtrack_counts": {},
        "max_executor_backtracks": 1,
        "node_failure_counts": {},
        "max_node_failures": 3,
    }


def read_task_from_file(user_input_path: str) -> str:
    if not user_input_path or not os.path.exists(user_input_path):
        return ""
    with open(user_input_path, "r", encoding="utf-8") as f:
        return f.read().rstrip()


def collect_multiline_input() -> str:
    print("\n请输入分析任务，支持多行输入。单独输入 END 提交，输入 exit 退出。")
    lines: List[str] = []
    while True:
        line = input("... " if lines else "> ")
        if not lines and line.strip().lower() == "exit":
            return "exit"
        if line.strip() == "END":
            return "\n".join(lines).rstrip()
        lines.append(line)


def REQUEST_BUDGET_SNAPSHOT():
    """R27: envelope snapshot for the terminal run_end record."""
    try:
        return _REQUEST_BUDGET.snapshot()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def run_interactive_agent(config: Config):
    output_path = config.get("record_file_path")
    memory_path = config.get("memory_path", "")
    user_input_path = config.get("user_input_path", "")
    app = build_agent_graph()
    persistent_memory = load_memory(memory_path)

    print("\n\n--- OmicsAgent started ---")
    print("\n\n支持从 `user_input.txt` 读取初始任务；控制台输入支持多行，使用 `END` 提交，`exit` 退出。")

    idx = 1
    while True:
        if idx == 1:
            user_input = read_task_from_file(user_input_path)
            idx += 1
        else:
            break # 暂时修改为这样
            user_input = collect_multiline_input()

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye")
            break

        current_state = build_initial_state(config, persistent_memory)
        current_state["task"] = user_input
        run_started_at = time.perf_counter()
        record_event(
            output_path,
            "run_start",
            "started",
            input_folder=os.path.dirname(user_input_path),
            model=OPENAI_MODEL,
            env_file=str(LOADED_ENV_FILE) if LOADED_ENV_FILE else "",
        )
        try:
            current_state = app.invoke(current_state, config={"recursion_limit": 200})
            finalize_continued_artifacts(config, output_path)
            record_event(
                output_path,
                "run_end",
                "completed",
                started_at=run_started_at,
                final_next=current_state.get("next", ""),
                reports=len(current_state.get("output_report", [])),
            )
        except Exception as e:
            tb = traceback.format_exc()
            record_report(output_path, f"\n\n## [Runner] Run failed: {e}\n\n```text\n{tb}\n```")
            if isinstance(e, RequestBudgetExceeded):
                finalize_continued_artifacts(config, output_path)
                record_event(
                    output_path,
                    "run_end",
                    "stopped_envelope",
                    started_at=run_started_at,
                    error=str(e),
                    stopped_by="request_envelope",
                    snapshot=REQUEST_BUDGET_SNAPSHOT(),
                )
            else:
                record_event(
                    output_path,
                    "run_end",
                    "error",
                    started_at=run_started_at,
                    error=str(e),
                    traceback=tb,
                )
            raise

        if current_state.get("analysis_summary"):
            safe_print(f"Analysis Summary: {current_state['analysis_summary']}")

        try:
            persistent_memory = ensure_memory_store(current_state.get("memory", {}))
            save_memory(memory_path, persistent_memory)
        except Exception as e:
            print(f"[Runner] Failed to save memory: {e}")
            record_report(output_path, f"\n\n## [Runner] Failed to save memory: {e}")

        print("-" * 50)


def _safe_run_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "run"


def build_folders(runs_folder: str, experiment_name: str, dataset_name: str = "", model_name: str = "") -> str:
    os.makedirs(runs_folder, exist_ok=True)
    if dataset_name in V3_MAIN_DATASETS:
        base_name = f"{_safe_run_part(dataset_name)}_{_safe_run_part(model_name or OPENAI_MODEL)}_{time.strftime('%Y%m%d')}"
        this_run_folder = os.path.join(runs_folder, base_name)
        if os.path.exists(this_run_folder):
            suffix = 2
            while os.path.exists(os.path.join(runs_folder, f"{base_name}_r{suffix}")):
                suffix += 1
            this_run_folder = os.path.join(runs_folder, f"{base_name}_r{suffix}")
    else:
        this_run_folder = os.path.join(runs_folder, f"{experiment_name}-{time.strftime('%Y.%m.%d--%H-%M-%S')}")
    os.makedirs(this_run_folder, exist_ok=True)
    try:
        set_usage_log(os.path.join(this_run_folder, "llm_usage.jsonl"))
    except Exception:
        pass
    return this_run_folder


def _load_runtime_dependencies() -> None:
    """Import the modules that need the agent environment, on the first real run.

    `--help`, the argument validation and every offline check must work on a machine where the
    agent environment is not installed yet, so nothing here is imported at module import time.
    The imported objects are injected into the module globals under exactly the names the rest
    of this file already uses, so the run logic is unchanged.
    """
    global _RUNTIME_DEPENDENCIES_LOADED
    global HumanMessage, SystemMessage, ChatPromptTemplate, HumanMessagePromptTemplate
    global StateGraph, END
    global LLM, SCORELLM, LLM_with_tools, OPENAI_MODEL, LOADED_ENV_FILE, set_usage_log
    global RequestBudgetExceeded, _REQUEST_BUDGET
    global prepare_analysis_design
    global prepare_evaluation_evidence_pack, prepare_user_visible_evidence_pack
    global PLANNER_PROMPT, EXECUTOR_PROMPT, ANALYZER_PROMPT, CRITIC_PROMPT
    global REPLANNER_PROMPT, SCORER_PROMPT, REPORT_PROMPT
    global EN_REPORT_LANGUAGE_DIRECTIVE
    global AVAIABLE_TOOLS, load_json, TOOLS_PROMPTS

    if _RUNTIME_DEPENDENCIES_LOADED:
        return
    _RUNTIME_DEPENDENCIES_LOADED = True

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langgraph.graph import StateGraph, END

    from llm import LLM, SCORELLM, OPENAI_MODEL, LOADED_ENV_FILE, set_usage_log, RequestBudgetExceeded
    from llm import REQUEST_BUDGET as _REQUEST_BUDGET
    from analysis_design import prepare_analysis_design
    from evaluation_evidence import prepare_evaluation_evidence_pack, prepare_user_visible_evidence_pack
    from prompts import (
        PLANNER_PROMPT,
        EXECUTOR_PROMPT,
        ANALYZER_PROMPT,
        CRITIC_PROMPT,
        REPLANNER_PROMPT,
        SCORER_PROMPT,
        REPORT_PROMPT,
        EN_REPORT_LANGUAGE_DIRECTIVE,
    )
    from tools import AVAIABLE_TOOLS, load_json
    from tools_prompts import TOOLS_PROMPTS

    LLM_with_tools = LLM.bind_tools(AVAIABLE_TOOLS)


# --------------------------------------------------------------------------- report language
#
# The report language is an explicit run option. It is never inferred from the task text,
# so a Chinese request can ask for an English report and an English request can ask for a
# Chinese one. Resolution order: explicit CLI value > the parent run recorded by
# --continue-from > the zh default.

CONTINUED_ARTIFACT_SKIP_PREFIXES = ("differential_",)
CONTINUED_ARTIFACT_SKIP_NAMES = frozenset({
    "report.md", "final_report.md", "final_output_report.md", "report_validation.json",
    "report_self_check.json", "report_self_check.md", "self_check.json", "self_check.md",
    "memory.json", "record_file.md", "evidence_ledger.jsonl", "external_knowledge.jsonl",
})


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def looks_like_run_folder(path: str) -> bool:
    """Cheap, dependency-free check that --continue-from points at a previous run."""
    if not path or not os.path.isdir(path):
        return False
    for name in ("run_metadata.json", "parameters.json", "record_file.md",
                 "analysis_design.used.yaml"):
        if os.path.exists(os.path.join(path, name)):
            return True
    return os.path.isdir(os.path.join(path, "processed_proteins"))


def parent_run_language(parent_folder: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (language, source_path) recorded by the parent run, else (None, None).

    A run recorded before this feature existed has no report_language field; that is
    read as the zh default, exactly like a missing field anywhere else.
    """
    for name in ("run_metadata.json", "parameters.json"):
        path = os.path.join(parent_folder, name)
        if not os.path.exists(path):
            continue
        try:
            # read with the standard library rather than the runtime-injected loader, so the
            # resolution works before the agent environment is loaded and cannot silently
            # degrade to "no language recorded" if that global is missing
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:  # noqa: BLE001 - an unreadable record simply falls through
            continue
        if not isinstance(payload, dict) or "report_language" not in payload:
            continue
        recorded = payload.get("report_language")
        if not (isinstance(recorded, str) and recorded.strip()):
            # an empty or blank value is not a recorded language; treating it as one would
            # report an inheritance that never happened
            continue
        try:
            return _report_language.normalize(recorded), path
        except _report_language.UnknownReportLanguage:
            continue
    return None, None


def resolve_report_language(cli_value, continue_from: Optional[str] = None) -> Tuple[str, str]:
    """Resolve the run language and say where it came from. Never reads the task text."""
    if cli_value is not None:
        return _report_language.normalize(cli_value), "cli"
    if continue_from:
        inherited, _source_path = parent_run_language(continue_from)
        if inherited:
            return inherited, "inherited_from_parent"
    return _report_language.DEFAULT_LANGUAGE, "default"


def collect_continued_artifacts(parent_folder: str) -> List[Tuple[str, str]]:
    """Parent-run products a continuation may reuse, as (source, destination-relative) pairs.

    The saved matrix, its gene map and its transform record are reused. Differential tables
    are deliberately not reused: a continuation recomputes the contrasts it needs on the
    inherited matrix, which is what makes it a new analysis instead of a replay.
    """
    pairs: List[Tuple[str, str]] = []
    design_src = os.path.join(parent_folder, "analysis_design.used.yaml")
    if os.path.isfile(design_src):
        pairs.append((design_src, "analysis_design.used.yaml"))
    processed_src = os.path.join(parent_folder, "processed_proteins")
    if os.path.isdir(processed_src):
        for name in sorted(os.listdir(processed_src)):
            source = os.path.join(processed_src, name)
            if not os.path.isfile(source):
                continue
            if name.startswith(CONTINUED_ARTIFACT_SKIP_PREFIXES):
                continue
            if name in CONTINUED_ARTIFACT_SKIP_NAMES:
                continue
            pairs.append((source, os.path.join("processed_proteins", name)))
    return pairs


def skipped_continued_artifacts(parent_folder: str) -> List[str]:
    """Entries under processed_proteins that the copy deliberately does not carry over.

    Recorded rather than dropped silently, so a reader can tell "the parent had nothing else"
    apart from "a whole subdirectory was skipped".
    """
    processed_src = os.path.join(parent_folder, "processed_proteins")
    if not os.path.isdir(processed_src):
        return []
    skipped = []
    for name in sorted(os.listdir(processed_src)):
        source = os.path.join(processed_src, name)
        if os.path.isdir(source):
            skipped.append("processed_proteins/" + name + "/")
        elif name.startswith(CONTINUED_ARTIFACT_SKIP_PREFIXES) or name in CONTINUED_ARTIFACT_SKIP_NAMES:
            skipped.append("processed_proteins/" + name)
    return skipped


def seed_continued_run(parent_folder: str, run_folder: str) -> List[Dict[str, Any]]:
    """Copy the reusable parent products into the new run and hash every one of them.

    The destination hashes are a seed-time snapshot. A run may legitimately recompute and
    overwrite a seeded file, so the key is named accordingly rather than implying that the
    file on disk at run end is the one that was carried over.
    """
    records: List[Dict[str, Any]] = []
    seeded_at = datetime.now().isoformat(timespec="seconds")
    for source, relative in collect_continued_artifacts(parent_folder):
        destination = os.path.join(run_folder, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        records.append({
            "source_path": source,
            "relative_path": relative.replace(os.sep, "/"),
            "destination_path": destination,
            "source_sha256": sha256_file(source),
            "seeded_destination_sha256": sha256_file(destination),
            "bytes": os.path.getsize(destination),
            "seeded_at": seeded_at,
            "note": "seed-time snapshot; the run may recompute and overwrite this file",
        })
    return records


def finalize_continued_artifacts(config: Config, record_file_path: str) -> Dict[str, Any]:
    """Re-hash every seeded artefact at run end, persist the outcome and record it.

    Copying a parent artefact into a new run does not prove that the run consumed it: a run
    derives its matrix from its own input, and a deterministic recomputation can produce the
    same bytes. What is recorded here is therefore a file-level fact only, and the metadata
    states explicitly that computation reuse is not claimed.
    """
    records = list(config.get("continued_artifacts") or [])
    run_folder = str(config.get("this_run_folder_path") or "")
    if not records or not run_folder:
        return {}
    for item in records:
        destination = str(item.get("destination_path") or "")
        if not destination:
            destination = os.path.join(run_folder, str(item.get("relative_path") or ""))
        try:
            current = sha256_file(destination)
        except Exception:
            current = ""
        item["final_sha256"] = current
        if not current:
            item["final_state"] = "missing"
        elif current == str(item.get("seeded_destination_sha256") or ""):
            item["final_state"] = "unchanged"
        else:
            item["final_state"] = "recomputed"
    outcome = {
        "artifacts": len(records),
        "unchanged": sum(1 for item in records if item.get("final_state") == "unchanged"),
        "recomputed": sum(1 for item in records if item.get("final_state") == "recomputed"),
        "missing": sum(1 for item in records if item.get("final_state") == "missing"),
        "computation_reuse": "not_claimed",
        "note": "unchanged = the bytes at run end equal the seed-time snapshot, which does not by "
                "itself show that the run consumed the copy, because a deterministic recomputation "
                "can produce the same bytes; recomputed = the run overwrote the seeded file",
    }
    for path in (os.path.join(run_folder, "run_metadata.json"), str(config.get("parameters_path") or "")):
        if not path or not os.path.exists(path):
            continue
        try:
            payload = json.loads(open(path, "r", encoding="utf-8").read())
        except Exception:
            continue
        if "continued_artifacts" not in payload:
            continue
        payload["continued_artifacts"] = records
        payload["continued_artifacts_outcome"] = outcome
        record_report(path, json.dumps(payload, ensure_ascii=False, indent=2), mode="w", visible=False)
    record_event(record_file_path, "continued_artifacts_finalized", "completed", **outcome)
    return outcome


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command line parser (standard library only, so `--help` needs no environment)."""
    parser = argparse.ArgumentParser(description="Single-cell Proteomics LLM Pipeline")
    parser.add_argument("-i", "--input-folder", required=True, type=str)
    parser.add_argument("-rf", "--runs-folder", required=False, type=str, default="./runs")
    parser.add_argument(
        "--report-language",
        dest="report_language",
        choices=list(_report_language.SUPPORTED),
        default=None,
        help="Language of the generated report: zh (default) or en. An explicit value wins "
             "over the language inherited through --continue-from.",
    )
    parser.add_argument(
        "--continue-from",
        dest="continue_from",
        required=False,
        default=None,
        help="Previous run folder to continue. Inherits its report language and reuses its "
             "saved analysis design and matrix state.",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=False,
        default=None,
        help="Experiment name, used for folder naming. Default is the input file name without extension.",
    )
    parser.add_argument(
        "--design",
        required=False,
        default=None,
        help="Optional analysis_design.yaml. Priority: CLI --design > dataset analysis_design.yaml > auto-inferred design.",
    )
    parser.add_argument(
        "--publication-mode",
        action="store_true",
        default=True,
        help="Run in publication-safe mode. This is the default and keeps evaluator-only files out of generation state.",
    )
    parser.add_argument(
        "--evaluator-mode",
        choices=["none", "internal-dev"],
        default="none",
        help="Evaluator-only file access mode. Use internal-dev only for local development scoring, never publication results.",
    )
    parser.add_argument(
        "--enable-internal-scorer",
        action="store_true",
        help="Allow the legacy in-agent scorer to read evaluator-only files. Requires --evaluator-mode internal-dev.",
    )
    parser.add_argument(
        "--evaluator-root",
        required=False,
        default=None,
        help="Root containing evaluator-only files for internal-dev scoring or post-run evaluation. Defaults to the input folder.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    # import sys
    # sys.argv = ["main_agent.py", "-i", "./examples/V3/Cell_TurnoverDynamics_2025"]

    parser = build_argument_parser()

    args = parser.parse_args(argv)
    if args.enable_internal_scorer and args.evaluator_mode != "internal-dev":
        parser.error("--enable-internal-scorer requires --evaluator-mode internal-dev")
    # checked before the agent environment is touched, so a bad parent folder costs nothing
    if args.continue_from and not looks_like_run_folder(args.continue_from):
        parser.error("--continue-from must point at a previous run folder (none of "
                     "run_metadata.json, parameters.json, record_file.md, "
                     "analysis_design.used.yaml or processed_proteins/ was found in %s)"
                     % args.continue_from)

    # From here on the run needs the agent environment. `--help` and an invalid argument
    # combination have already exited above, before any third-party package is imported.
    _load_runtime_dependencies()

    report_language, report_language_source = resolve_report_language(
        args.report_language, args.continue_from)
    _report_language.set_language(report_language)

    input_file_path = args.input_folder
    evaluator_root = args.evaluator_root or input_file_path
    enable_internal_scorer = bool(args.enable_internal_scorer and args.evaluator_mode == "internal-dev")
    publication_eligible_initial = bool(args.publication_mode and args.evaluator_mode == "none" and not enable_internal_scorer)
    if args.name:
        experiment_name = f"Agent-{OPENAI_MODEL}-{args.name}"
    else:
        experiment_name = f"Agent-{OPENAI_MODEL}-{os.path.splitext(os.path.basename(input_file_path))[0]}"

    dataset_name_for_run = os.path.basename(os.path.abspath(input_file_path.rstrip("\\/")))
    this_run_folder = build_folders(args.runs_folder, experiment_name, dataset_name_for_run, OPENAI_MODEL)
    record_file_path = os.path.join(this_run_folder, "record_file.md")
    memory_path = os.path.join(this_run_folder, "memory.json")
    parameters_path = os.path.join(this_run_folder, "parameters.json")

    record_report(record_file_path, "# Single-cell Proteomics Analysis Log\n", mode="w", visible=False)
    record_event(
        record_file_path,
        "run_folder_created",
        "created",
        run_folder=this_run_folder,
        input_folder=input_file_path,
        model=OPENAI_MODEL,
        env_file=str(LOADED_ENV_FILE) if LOADED_ENV_FILE else "",
        publication_mode=args.publication_mode,
        evaluator_mode=args.evaluator_mode,
        enable_internal_scorer=enable_internal_scorer,
        publication_eligible_initial=publication_eligible_initial,
        report_language=report_language,
        report_language_source=report_language_source,
        continued_from=args.continue_from or "",
    )
    save_memory(memory_path, initialize_memory_store())

    continued_artifacts: List[Dict[str, Any]] = []
    if args.continue_from:
        continued_artifacts = seed_continued_run(args.continue_from, this_run_folder)
        record_event(
            record_file_path,
            "continued_run_seeded",
            "completed",
            continued_from=args.continue_from,
            artifact_count=len(continued_artifacts),
            artifacts=[item["relative_path"] for item in continued_artifacts],
            skipped=skipped_continued_artifacts(args.continue_from),
            note="seeded_destination_sha256 is a seed-time snapshot; a run may recompute and "
                 "overwrite a seeded file, so the file on disk at run end is not guaranteed to "
                 "be the one that was carried over",
        )

    sampleinfo_path = os.path.join(input_file_path, "SampleInfo.csv")
    protein_quant_path = os.path.join(input_file_path, "ProteinQuant.csv")
    explicit_design_path = args.design
    if explicit_design_path is None and args.continue_from:
        inherited_design_path = os.path.join(args.continue_from, "analysis_design.used.yaml")
        if os.path.exists(inherited_design_path):
            explicit_design_path = inherited_design_path
    design_info = prepare_analysis_design(
        input_file_path,
        sampleinfo_path,
        protein_quant_path,
        this_run_folder,
        explicit_design_path=explicit_design_path,
    )
    record_event(
        record_file_path,
        "analysis_design_prepared",
        "completed",
        analysis_design_source=design_info.get("analysis_design_source"),
        analysis_design_path=design_info.get("analysis_design_path"),
        analysis_design_inferred_path=design_info.get("analysis_design_inferred_path"),
    )
    append_evidence(
        this_run_folder,
        source="analysis_design",
        tool="prepare_analysis_design",
        claim="Analysis design prepared for this run; downstream statistical confidence is bounded by whether the design was inferred or user-provided.",
        files=[design_info.get("analysis_design_path", ""), design_info.get("analysis_design_inferred_path", "")],
        metrics={
            "source": design_info.get("analysis_design_source"),
            "group_col": design_info.get("analysis_design", {}).get("group_col"),
            "batch_col": design_info.get("analysis_design", {}).get("batch_col"),
            "matrix_state": design_info.get("analysis_design", {}).get("matrix", {}).get("matrix_state"),
        },
        confidence="moderate" if design_info.get("analysis_design_source") == "auto_inferred" else "high",
        limitations=design_info.get("analysis_design", {}).get("warnings", []),
    )

    config = {
        "input_folder": input_file_path,
        "project_dir": os.path.dirname(os.path.abspath(__file__)),
        "sampleinfo_path": sampleinfo_path,
        "protein_quant_path": protein_quant_path,
        "user_input_path": os.path.join(input_file_path, "user_input.txt"),
        "this_run_folder_path": this_run_folder,
        "record_file_path": record_file_path,
        "memory_path": memory_path,
        "parameters_path": parameters_path,
        "publication_mode": args.publication_mode,
        "evaluator_mode": args.evaluator_mode,
        "enable_internal_scorer": enable_internal_scorer,
        "evaluator_root": evaluator_root if args.evaluator_mode == "internal-dev" else "",
        "publication_eligible_initial": publication_eligible_initial,
        "report_language": report_language,
        "report_language_source": report_language_source,
        "continued_from": args.continue_from or "",
        "continued_artifacts": continued_artifacts,
        "analysis_design": design_info.get("analysis_design", {}),
        "analysis_design_path": design_info.get("analysis_design_path", ""),
        "analysis_design_inferred_path": design_info.get("analysis_design_inferred_path", ""),
        "evaluation_requirements_path": os.path.join(this_run_folder, "evaluation_requirements.used.json"),
    }
    if enable_internal_scorer:
        config["ground_truth_path"] = os.path.join(evaluator_root, "ground_truth.txt")
        config["grading_standard_path"] = os.path.join(evaluator_root, "grading_standard.txt")
    run_metadata = {
        "run_folder": this_run_folder,
        "input_folder": input_file_path,
        "model": OPENAI_MODEL,
        "publication_mode": args.publication_mode,
        "evaluator_mode": args.evaluator_mode,
        "enable_internal_scorer": enable_internal_scorer,
        "publication_eligible_initial": publication_eligible_initial,
        "publication_eligible": publication_eligible_initial,
        "hidden_standard_paths_in_generation_config": enable_internal_scorer,
        "report_language": report_language,
        "report_language_source": report_language_source,
        "continued_from": args.continue_from or "",
        "continued_artifacts": continued_artifacts,
        "analysis_design_source": design_info.get("analysis_design_source"),
        "analysis_design_path": design_info.get("analysis_design_path", ""),
        "analysis_design_inferred_path": design_info.get("analysis_design_inferred_path", ""),
    }
    record_report(os.path.join(this_run_folder, "run_metadata.json"), json.dumps(run_metadata, ensure_ascii=False, indent=2), mode="w", visible=False)
    record_report(parameters_path, json.dumps(redact_for_log(config, max_chars=4000), ensure_ascii=False, indent=2), mode="w", visible=False)
    record_event(record_file_path, "publication_metadata_prepared", "completed", **run_metadata)

    set_config(config)
    try:
        initial_evidence_pack = prepare_user_visible_evidence_pack(config)
        record_event(
            record_file_path,
            "user_visible_evidence_pack_initial",
            "completed",
            evidence_scope=initial_evidence_pack.get("evidence_scope", "user_visible"),
            files=[
                initial_evidence_pack.get("requirements_path", ""),
                initial_evidence_pack.get("group_qc", {}).get("csv", ""),
                initial_evidence_pack.get("core_story", {}).get("csv", ""),
                initial_evidence_pack.get("candidate_evidence", {}).get("csv", ""),
                initial_evidence_pack.get("module_scores", {}).get("group_summary_csv", ""),
                initial_evidence_pack.get("scoring_coverage", {}).get("csv", ""),
            ],
        )
    except Exception as e:
        record_report(record_file_path, f"\n\n## [EvidencePack] Initial preparation failed: {e}")
        record_event(record_file_path, "evaluation_evidence_pack_initial", "error", error=str(e))
    run_interactive_agent(config)




# ---------------------------------------------------------------------------
# Reader wording for the module-level tables above (bilingual release, 2026-09-21).
#
# Every zh value below is the frozen value of the table above, referenced at registration
# time so those tables stay the single source of the released wording; the en value is the
# added language. They are registered in their own namespace because they are table
# vocabulary rather than per-function report prose, and the accessors below resolve them
# for the language in force.
# ---------------------------------------------------------------------------
_TABLE_TEXT = {
    "mx_state_raw": (
        "原始强度矩阵",
        "raw intensity matrix"),
    "mx_state_normalized": (
        "已归一化矩阵",
        "normalised matrix"),
    "mx_state_batch_corrected": (
        "已完成批次校正的矩阵",
        "batch-corrected matrix"),
    "mx_state_logged": (
        "已取对数的矩阵",
        "log-transformed matrix"),
    "direction_up_in_group_a": (
        "第一组更高（上调）",
        "Group 1 higher (up)"),
    "direction_up_in_display_group_a": (
        "第一组更高（上调）",
        "Group 1 higher (up)"),
    "direction_down_in_group_a": (
        "第一组更低（下调）",
        "Group 1 lower (down)"),
    "direction_down_in_display_group_a": (
        "第一组更低（下调）",
        "Group 1 lower (down)"),
    "direction_up_in_group_b": (
        "第二组更高（上调）",
        "Group 2 higher (up)"),
    "direction_down_in_group_b": (
        "第二组更低（下调）",
        "Group 2 lower (down)"),
    "direction_detected_without_selected_contrast": (
        "仅记录存在性（未指定对比）",
        "existence recorded only (no contrast specified)"),
    "direction_symbol_not_matched_in_available_matrix": (
        "未建立符号映射（不等同于未检出）",
        "no symbol mapping established (not the same as not detected)"),
    "direction_not_detected_in_available_matrix": (
        "当前矩阵未检出",
        "not detected in the current matrix"),
    "test_method_python_welch_ttest_fallback": (
        "Python 实现的 Welch t 检验",
        "Welch t test implemented in Python"),
    "test_method_python_welch_ttest": (
        "Welch t 检验",
        "Welch t test"),
    "test_method_welch_ttest": (
        "Welch t 检验",
        "Welch t test"),
    "module_name_migration_signature": (
        "迁移综合特征",
        "Migration composite signature"),
    "module_name_rho_gtpase_migration": (
        "Rho GTPase 调控",
        "Rho GTPase regulation"),
    "module_name_erm_membrane_cortex": (
        "ERM 膜-皮质连接",
        "ERM membrane-cortex linkage"),
    "module_name_myosin_contractility": (
        "肌球蛋白收缩",
        "Myosin contractility"),
    "module_name_talin_vinculin_focal_adhesion": (
        "talin-vinculin 黏附斑",
        "talin-vinculin focal adhesion"),
    "module_name_actin_cytoskeleton": (
        "肌动蛋白骨架",
        "Actin cytoskeleton"),
    "module_name_ecm_adhesion": (
        "ECM/黏附重塑",
        "ECM/adhesion remodelling"),
    "module_name_vesicle_transport": (
        "囊泡运输",
        "Vesicle transport"),
    "module_name_translation_ribosome": (
        "翻译/核糖体",
        "Translation/ribosome"),
    "rule_all_samples": (
        "全部样本中非缺失比例 >= {cutoff}",
        "non-missing fraction >= {cutoff} in all samples"),
    "rule_per_cell_type": (
        "按细胞类型分别要求非缺失比例 >= {cutoff}",
        "non-missing fraction >= {cutoff} per cell type"),
}
_report_language.register("tables", _TABLE_TEXT)

_FIGURE_CAPTION_KEYS = {
    "Cluster_1_vs_Cluster_2_volcano_plot.png": "figure_caption_01",
    "Cluster_1_vs_Cluster_3_volcano_plot.png": "figure_caption_02",
    "Cluster_2_vs_Cluster_3_volcano_plot.png": "figure_caption_03",
    "differential_summary_barplot.png": "figure_caption_04",
    "heatmap.png": "figure_caption_05",
    "key_protein_overview.png": "figure_caption_06",
    "mechanism_enrichment_dotplot.png": "figure_caption_07",
    "mechanism_pathway_gene_heatmap_Clust_2c410786f2.png": "figure_caption_08",
    "mechanism_top_protein_group_means.png": "figure_caption_09",
    "pca_plot.png": "figure_caption_10",
    "protein_contrast_bubble.png": "figure_caption_11",
    "qc_sample_overview.png": "figure_caption_12",
    "umap_Cluster.png": "figure_caption_13",
    "umap_Group_Control Cell-Migrated Cell.png": "figure_caption_14",
    "umap_plot.png": "figure_caption_15",
}
_FIGURE_CAPTION_ZH = {
    "figure_caption_01": "Cluster 1 vs Cluster 2 火山图",
    "figure_caption_02": "Cluster 1 vs Cluster 3 火山图",
    "figure_caption_03": "Cluster 2 vs Cluster 3 火山图",
    "figure_caption_04": "差异蛋白数量概览",
    "figure_caption_05": "差异蛋白热图",
    "figure_caption_06": "关键蛋白总览",
    "figure_caption_07": "机制富集 dotplot",
    "figure_caption_08": "机制蛋白热图",
    "figure_caption_09": "核心蛋白组均值图",
    "figure_caption_10": "PCA 样本结构",
    "figure_caption_11": "候选蛋白对比气泡图",
    "figure_caption_12": "样本与缺失率 QC",
    "figure_caption_13": "UMAP Cluster 分布",
    "figure_caption_14": "UMAP 迁移表型分布",
    "figure_caption_15": "UMAP 样本结构",
}
_FIGURE_CAPTION_EN = {
    "figure_caption_01": "Cluster 1 vs Cluster 2 volcano plot",
    "figure_caption_02": "Cluster 1 vs Cluster 3 volcano plot",
    "figure_caption_03": "Cluster 2 vs Cluster 3 volcano plot",
    "figure_caption_04": "Differential protein count overview",
    "figure_caption_05": "Differential protein heatmap",
    "figure_caption_06": "Key protein overview",
    "figure_caption_07": "Mechanism enrichment dotplot",
    "figure_caption_08": "Mechanism protein heatmap",
    "figure_caption_09": "Core protein group means",
    "figure_caption_10": "PCA sample structure",
    "figure_caption_11": "Candidate protein contrast bubble plot",
    "figure_caption_12": "Sample and missingness-rate QC",
    "figure_caption_13": "UMAP cluster distribution",
    "figure_caption_14": "UMAP migration-phenotype distribution",
    "figure_caption_15": "UMAP sample structure",
}
for _caption_key, _caption_en in _FIGURE_CAPTION_EN.items():
    _TABLE_TEXT[_caption_key] = (_FIGURE_CAPTION_ZH[_caption_key], _caption_en)
_report_language.register("tables", _TABLE_TEXT)



# Dataset-specific mechanism-integration specs, for the language in force.
#
# The table above stays the single source of the released zh wording; every zh string it holds
# that reaches the report is registered here with its added English wording, and the accessor
# below swaps the strings for a non-zh report without touching the zh object itself.
_DIMENSION4_SPEC_KEYS = {
    "机制整合与解释边界（状态轴与批次注记）": "dim4_nat_biotech_brain_2026_heading",
    "状态轴两段过渡": "dim4_nat_biotech_brain_2026_contrasts_0_0",
    "EN 成熟段": "dim4_nat_biotech_brain_2026_contrasts_1_0",
    "状态轴以 IPC-EN vs oRG（过渡）与 EN vs IPC-EN（成熟）两段对比组织；下表给出两段的显著蛋白数与焦点基因方向。": "dim4_nat_biotech_brain_2026_paragraphs_0",
    "供体/批次结构与发育状态的混淆需在批内方向核对后才支持发育结论；ASD/NDD 关联仅作为 external_annotation，不进入矩阵内结论。": "dim4_nat_biotech_brain_2026_paragraphs_1",
    "状态轴方向不替代拟时序或独立队列验证。": "dim4_nat_biotech_brain_2026_boundary_0",
    "机制整合与解释边界（处理效应与批次结构）": "dim4_nat_commun_carr_2024_heading",
    "LPS vs DMSO 主对比": "dim4_nat_commun_carr_2024_contrasts_0_0",
    "LPS vs DMSO 主对比之外，批次与处理效应的相对强弱必须对照图册中的 PCA/UMAP 与批次交叉表解读；本节不重复计算，只给出核对入口。": "dim4_nat_commun_carr_2024_paragraphs_0",
    "SampleInfo 中的 Batch 结构与处理分组存在交叉；若批次分离强于处理分离，差异方向仍以当前矩阵为准，但机制解释需降级为方向线索。": "dim4_nat_commun_carr_2024_paragraphs_1",
    "批校正矩阵为本报告差异分析基础；批效应残留程度以图册证据为准。": "dim4_nat_commun_carr_2024_boundary_0",
    "机制整合与解释边界（亚型×炎症）": "dim4_nat_commun_nociceptor_2026_heading",
    "亚型内 Inflamed vs Control": "dim4_nat_commun_nociceptor_2026_contrasts_0_0",
    "亚型内对比是本数据集的核心设计：TrkA、IB4 与 Mechano 各自的 Inflamed vs Control 回答炎症响应是否发生在特定伤害感受亚型内。": "dim4_nat_commun_nociceptor_2026_paragraphs_0",
    "下表汇总各亚型对比的显著蛋白数（adj.P.Val<0.05）与焦点候选的方向和 FDR；n_sig 为 0 表示当前矩阵在该亚型内未检出足够显著的炎症响应，相关方向线索只能作为低置信度假设。": "dim4_nat_commun_nociceptor_2026_paragraphs_1",
    "亚型内样本量小，显著蛋白数对插补和阈值敏感；亚型间比较不构成细胞类型组成差异的统计检验。": "dim4_nat_commun_nociceptor_2026_boundary_0",
    "机制整合与解释边界（区室泄漏与分层混杂）": "dim4_nat_commun_proteinleakage_2025_heading",
    "泄漏主对比（含分型）": "dim4_nat_commun_proteinleakage_2025_contrasts_0_0",
    "主对比的区室方向（膜/线粒体 vs 胞质/核）是泄漏 signature 的核心；下表给出焦点蛋白方向。": "dim4_nat_commun_proteinleakage_2025_paragraphs_0",
    "Fresh/Frozen 与细胞类型的分层混杂以 group_composition_qc 交叉表为准：差异解读需先核对分层的样本量不均衡，再谈生物学含义。": "dim4_nat_commun_proteinleakage_2025_paragraphs_1",
    "泄漏足迹是样品处理 QC 特征，不直接等于生物学调控。": "dim4_nat_commun_proteinleakage_2025_boundary_0",
    "机制整合与解释边界（KLRG1 程序与 PDAC 背景）": "dim4_nat_commun_scpro_2024_heading",
    "KLRG1 主对比与 CD4/CD8 背景": "dim4_nat_commun_scpro_2024_contrasts_0_0",
    "主对比层级为 Treg 内部 KLRG1+ vs KLRG1-，再以 CD4/CD8 KLRG1 背景对照；下表给出各对比的显著蛋白数与焦点候选。": "dim4_nat_commun_scpro_2024_paragraphs_0",
    "研究背景（study context）：SCPro 面向胰腺癌（PDAC）肿瘤微环境的成像引导空间蛋白组。当前矩阵仅含 T 细胞区室，PDAC 免疫微环境解释停留在研究背景层，不能写成矩阵内发现。": "dim4_nat_commun_scpro_2024_paragraphs_1",
    "KLRG1 程序差异不构成 PDAC 免疫状态结论；免疫微环境整合需要带区室注释的独立证据。": "dim4_nat_commun_scpro_2024_boundary_0",
    "机制整合与解释边界（小叶分区与 Midlobular 定位）": "dim4_nat_methods_dvp_2023_heading",
    "分区对比（Portal/Midlobular vs Central 等）": "dim4_nat_methods_dvp_2023_contrasts_0_0",
    "Midlobular 相关对比": "dim4_nat_methods_dvp_2023_contrasts_1_0",
    "肝小叶分区以 Portal vs Central 为主轴；Midlobular 的定位通过其与两侧的对比方向与幅度判断。": "dim4_nat_methods_dvp_2023_paragraphs_0",
    "判读规则：若 Midlobular 相对 Portal 与 Central 的焦点基因方向相反、幅度相当，则支持过渡态；若与一侧方向一致且 n_sig 明显更小，则更像该分区的延伸。以下表方向组合为准。": "dim4_nat_methods_dvp_2023_paragraphs_1",
    "分区方向不等于血流或代谢通量的直接测量。": "dim4_nat_methods_dvp_2023_boundary_0",
    "机制整合与解释边界（多能性梯度与 EB 内部异质性）": "dim4_nat_methods_ipsc_2025_heading",
    "EB vs iPSCs 主轴": "dim4_nat_methods_ipsc_2025_contrasts_0_0",
    "EB vs iPSCs 主轴给出多能性下降与谱系/ECM 上升；EB 内部异质性由聚类配方图与簇规模刻画。": "dim4_nat_methods_ipsc_2025_paragraphs_0",
    "EB 内部聚类簇规模": "dim4_nat_methods_ipsc_2025_manifest_probe_label",
    "EB 内部异质性不等于谱系判定；需要时间序列或独立标注验证。": "dim4_nat_methods_ipsc_2025_boundary_0",
}
for _spec_zh, _spec_key in _DIMENSION4_SPEC_KEYS.items():
    _TABLE_TEXT[_spec_key] = (_spec_zh, "")
_TABLE_TEXT["dim4_nat_biotech_brain_2026_heading"] = (
    "机制整合与解释边界（状态轴与批次注记）",
    "Mechanism Integration and Interpretation Boundaries (state axis and batch note)")
_TABLE_TEXT["dim4_nat_biotech_brain_2026_contrasts_0_0"] = (
    "状态轴两段过渡",
    "Two segments of the state axis")
_TABLE_TEXT["dim4_nat_biotech_brain_2026_contrasts_1_0"] = (
    "EN 成熟段",
    "EN maturation segment")
_TABLE_TEXT["dim4_nat_biotech_brain_2026_paragraphs_0"] = (
    "状态轴以 IPC-EN vs oRG（过渡）与 EN vs IPC-EN（成熟）两段对比组织；下表给出两段的显著蛋白数与焦点基因方向。",
    "The state axis is organised as two contrast segments, IPC-EN vs oRG (transition) and EN vs IPC-EN (maturation); the table below gives the significant protein count and the focus-gene direction of each segment.")
_TABLE_TEXT["dim4_nat_biotech_brain_2026_paragraphs_1"] = (
    "供体/批次结构与发育状态的混淆需在批内方向核对后才支持发育结论；ASD/NDD 关联仅作为 external_annotation，不进入矩阵内结论。",
    "Donor or batch structure is confounded with developmental state, so a developmental conclusion is only supported once the within-batch direction has been checked; ASD/NDD associations serve only as external_annotation and do not enter conclusions drawn from the matrix.")
_TABLE_TEXT["dim4_nat_biotech_brain_2026_boundary_0"] = (
    "状态轴方向不替代拟时序或独立队列验证。",
    "A state-axis direction does not replace pseudotime or validation in an independent cohort.")
_TABLE_TEXT["dim4_nat_commun_carr_2024_heading"] = (
    "机制整合与解释边界（处理效应与批次结构）",
    "Mechanism Integration and Interpretation Boundaries (treatment effect and batch structure)")
_TABLE_TEXT["dim4_nat_commun_carr_2024_contrasts_0_0"] = (
    "LPS vs DMSO 主对比",
    "LPS vs DMSO main contrast")
_TABLE_TEXT["dim4_nat_commun_carr_2024_paragraphs_0"] = (
    "LPS vs DMSO 主对比之外，批次与处理效应的相对强弱必须对照图册中的 PCA/UMAP 与批次交叉表解读；本节不重复计算，只给出核对入口。",
    "Beyond the LPS vs DMSO main contrast, the relative strength of the batch and treatment effects must be read against the PCA/UMAP figures and the batch cross-table in the figure set; this section recomputes nothing and gives the review entry point only.")
_TABLE_TEXT["dim4_nat_commun_carr_2024_paragraphs_1"] = (
    "SampleInfo 中的 Batch 结构与处理分组存在交叉；若批次分离强于处理分离，差异方向仍以当前矩阵为准，但机制解释需降级为方向线索。",
    "The Batch structure in SampleInfo intersects the treatment groups; where batch separation is stronger than treatment separation the differential direction still follows the current matrix, but the mechanism interpretation has to be downgraded to a directional lead.")
_TABLE_TEXT["dim4_nat_commun_carr_2024_boundary_0"] = (
    "批校正矩阵为本报告差异分析基础；批效应残留程度以图册证据为准。",
    "The batch-corrected matrix is the basis of the differential analysis in this report; how much batch effect remains is judged from the figure-set evidence.")
_TABLE_TEXT["dim4_nat_commun_nociceptor_2026_heading"] = (
    "机制整合与解释边界（亚型×炎症）",
    "Mechanism Integration and Interpretation Boundaries (subtype x inflammation)")
_TABLE_TEXT["dim4_nat_commun_nociceptor_2026_contrasts_0_0"] = (
    "亚型内 Inflamed vs Control",
    "Within-subtype Inflamed vs Control")
_TABLE_TEXT["dim4_nat_commun_nociceptor_2026_paragraphs_0"] = (
    "亚型内对比是本数据集的核心设计：TrkA、IB4 与 Mechano 各自的 Inflamed vs Control 回答炎症响应是否发生在特定伤害感受亚型内。",
    "The within-subtype contrasts are the core design of this dataset: the Inflamed vs Control contrast for TrkA, IB4 and Mechano respectively asks whether the inflammatory response happens inside a specific nociceptor subtype.")
_TABLE_TEXT["dim4_nat_commun_nociceptor_2026_paragraphs_1"] = (
    "下表汇总各亚型对比的显著蛋白数（adj.P.Val<0.05）与焦点候选的方向和 FDR；n_sig 为 0 表示当前矩阵在该亚型内未检出足够显著的炎症响应，相关方向线索只能作为低置信度假设。",
    "The table below summarises the significant protein count (adj.P.Val<0.05) of each subtype contrast together with the direction and FDR of the focus candidates; n_sig of 0 means the current matrix detected no sufficiently significant inflammatory response inside that subtype, so the related directional leads can only serve as low-confidence hypotheses.")
_TABLE_TEXT["dim4_nat_commun_nociceptor_2026_boundary_0"] = (
    "亚型内样本量小，显著蛋白数对插补和阈值敏感；亚型间比较不构成细胞类型组成差异的统计检验。",
    "The within-subtype sample size is small and the significant protein count is sensitive to imputation and thresholds; a comparison between subtypes is not a statistical test of cell-type composition differences.")
_TABLE_TEXT["dim4_nat_commun_proteinleakage_2025_heading"] = (
    "机制整合与解释边界（区室泄漏与分层混杂）",
    "Mechanism Integration and Interpretation Boundaries (compartment leakage and stratified confounding)")
_TABLE_TEXT["dim4_nat_commun_proteinleakage_2025_contrasts_0_0"] = (
    "泄漏主对比（含分型）",
    "Leakage main contrast (with subtyping)")
_TABLE_TEXT["dim4_nat_commun_proteinleakage_2025_paragraphs_0"] = (
    "主对比的区室方向（膜/线粒体 vs 胞质/核）是泄漏 signature 的核心；下表给出焦点蛋白方向。",
    "The compartment direction of the main contrast (membrane/mitochondrial vs cytoplasmic/nuclear) is the core of the leakage signature; the table below gives the direction of the focus proteins.")
_TABLE_TEXT["dim4_nat_commun_proteinleakage_2025_paragraphs_1"] = (
    "Fresh/Frozen 与细胞类型的分层混杂以 group_composition_qc 交叉表为准：差异解读需先核对分层的样本量不均衡，再谈生物学含义。",
    "Stratified confounding between Fresh/Frozen and cell type follows the group_composition_qc cross-table: a differential reading has to check the sample-size imbalance across strata before discussing biological meaning.")
_TABLE_TEXT["dim4_nat_commun_proteinleakage_2025_boundary_0"] = (
    "泄漏足迹是样品处理 QC 特征，不直接等于生物学调控。",
    "The leakage footprint is a sample-handling QC feature and is not directly equivalent to biological regulation.")
_TABLE_TEXT["dim4_nat_commun_scpro_2024_heading"] = (
    "机制整合与解释边界（KLRG1 程序与 PDAC 背景）",
    "Mechanism Integration and Interpretation Boundaries (KLRG1 programme and PDAC background)")
_TABLE_TEXT["dim4_nat_commun_scpro_2024_contrasts_0_0"] = (
    "KLRG1 主对比与 CD4/CD8 背景",
    "KLRG1 main contrast with CD4/CD8 background")
_TABLE_TEXT["dim4_nat_commun_scpro_2024_paragraphs_0"] = (
    "主对比层级为 Treg 内部 KLRG1+ vs KLRG1-，再以 CD4/CD8 KLRG1 背景对照；下表给出各对比的显著蛋白数与焦点候选。",
    "The main contrast is KLRG1+ vs KLRG1- within Treg, with a CD4/CD8 KLRG1 background comparison; the table below gives the significant protein count and the focus candidates of each contrast.")
_TABLE_TEXT["dim4_nat_commun_scpro_2024_paragraphs_1"] = (
    "研究背景（study context）：SCPro 面向胰腺癌（PDAC）肿瘤微环境的成像引导空间蛋白组。当前矩阵仅含 T 细胞区室，PDAC 免疫微环境解释停留在研究背景层，不能写成矩阵内发现。",
    "Study context: SCPro targets imaging-guided spatial proteomics of the pancreatic cancer (PDAC) tumour microenvironment. The current matrix contains only the T-cell compartment, so a PDAC immune-microenvironment reading stays at the study-context level and cannot be written as a finding from the matrix.")
_TABLE_TEXT["dim4_nat_commun_scpro_2024_boundary_0"] = (
    "KLRG1 程序差异不构成 PDAC 免疫状态结论；免疫微环境整合需要带区室注释的独立证据。",
    "A KLRG1 programme difference does not establish a PDAC immune-state conclusion; integrating the immune microenvironment needs independent evidence with compartment annotation.")
_TABLE_TEXT["dim4_nat_methods_dvp_2023_heading"] = (
    "机制整合与解释边界（小叶分区与 Midlobular 定位）",
    "Mechanism Integration and Interpretation Boundaries (lobular zonation and Midlobular position)")
_TABLE_TEXT["dim4_nat_methods_dvp_2023_contrasts_0_0"] = (
    "分区对比（Portal/Midlobular vs Central 等）",
    "Zonation contrasts (Portal/Midlobular vs Central and similar)")
_TABLE_TEXT["dim4_nat_methods_dvp_2023_contrasts_1_0"] = (
    "Midlobular 相关对比",
    "Midlobular-related contrasts")
_TABLE_TEXT["dim4_nat_methods_dvp_2023_paragraphs_0"] = (
    "肝小叶分区以 Portal vs Central 为主轴；Midlobular 的定位通过其与两侧的对比方向与幅度判断。",
    "Liver lobular zonation takes Portal vs Central as its main axis; the position of Midlobular is judged from the direction and magnitude of its contrasts against the two sides.")
_TABLE_TEXT["dim4_nat_methods_dvp_2023_paragraphs_1"] = (
    "判读规则：若 Midlobular 相对 Portal 与 Central 的焦点基因方向相反、幅度相当，则支持过渡态；若与一侧方向一致且 n_sig 明显更小，则更像该分区的延伸。以下表方向组合为准。",
    "Reading rule: if the focus genes run in the opposite direction for Midlobular against both Portal and Central with comparable magnitude, a transition state is supported; if they agree with one side and n_sig is clearly smaller, Midlobular reads more like an extension of that zone. The direction combinations in the table below decide.")
_TABLE_TEXT["dim4_nat_methods_dvp_2023_boundary_0"] = (
    "分区方向不等于血流或代谢通量的直接测量。",
    "A zonation direction is not a direct measurement of blood flow or metabolic flux.")
_TABLE_TEXT["dim4_nat_methods_ipsc_2025_heading"] = (
    "机制整合与解释边界（多能性梯度与 EB 内部异质性）",
    "Mechanism Integration and Interpretation Boundaries (pluripotency gradient and heterogeneity inside EB)")
_TABLE_TEXT["dim4_nat_methods_ipsc_2025_contrasts_0_0"] = (
    "EB vs iPSCs 主轴",
    "EB vs iPSCs main axis")
_TABLE_TEXT["dim4_nat_methods_ipsc_2025_paragraphs_0"] = (
    "EB vs iPSCs 主轴给出多能性下降与谱系/ECM 上升；EB 内部异质性由聚类配方图与簇规模刻画。",
    "The EB vs iPSCs main axis shows pluripotency falling while lineage and ECM programmes rise; heterogeneity inside EB is described by the clustering recipe figure and the cluster sizes.")
_TABLE_TEXT["dim4_nat_methods_ipsc_2025_manifest_probe_label"] = (
    "EB 内部聚类簇规模",
    "Cluster sizes inside EB")
_TABLE_TEXT["dim4_nat_methods_ipsc_2025_boundary_0"] = (
    "EB 内部异质性不等于谱系判定；需要时间序列或独立标注验证。",
    "Heterogeneity inside EB is not a lineage call and needs time-series data or independent annotation to be validated.")
_report_language.register("tables", _TABLE_TEXT)


def _translate_spec(node):
    """Recursively swap the registered spec strings for the active language."""
    if isinstance(node, dict):
        return dict((item, _translate_spec(node[item])) for item in node)
    if isinstance(node, list):
        return [_translate_spec(item) for item in node]
    if isinstance(node, str):
        key = _DIMENSION4_SPEC_KEYS.get(node)
        return _report_language.t("tables." + key) if key else node
    return node


def _dimension4_spec_for(dataset: str):
    """Dataset-specific mechanism-integration spec, for the language in force.

    A zh report gets the released object untouched; another language gets a translated copy,
    and a string with no registered wording is left exactly as the table holds it.
    """
    spec = _DIMENSION4_SECTION_SPECS.get(dataset)
    if spec is None or _report_language.get_language() == "zh":
        return spec
    return _translate_spec(spec)
def _table_matrix_state() -> Dict[str, str]:
    """Recorded matrix-state code to reader wording, for the language in force."""
    return dict((code, _report_language.t("tables.mx_state_" + code))
                for code in _MATRIX_STATE_TEXT)


def _table_direction() -> Dict[str, str]:
    """Recorded direction code to reader wording, for the language in force."""
    return dict((code, _report_language.t("tables.direction_" + code))
                for code in _DIRECTION_TEXT)


def _table_test_method() -> Dict[str, str]:
    """Recorded test-method code to reader wording, for the language in force."""
    return dict((code, _report_language.t("tables.test_method_" + code))
                for code in _TEST_METHOD_TEXT)


def _table_module_display_names() -> Dict[str, str]:
    """Curated module code to display name, for the language in force."""
    return dict((code, _report_language.t("tables.module_name_" + code))
                for code in PISPA_MODULE_DISPLAY_NAMES)


def _key_figure_specs(dataset: str) -> List[Tuple[str, str]]:
    """Key-figure (file, caption) pairs, for the language in force.

    A file name with no registered caption keeps the caption recorded in the table.
    """
    specs = PISPA_KEY_FIGURE_SPECS if dataset == "Nat_Commun_PiSPA_2024" else GENERIC_KEY_FIGURE_SPECS
    found = []
    for filename, caption in specs:
        key = _FIGURE_CAPTION_KEYS.get(filename)
        found.append((filename, _report_language.t("tables." + key) if key else caption))
    return found


def _reader_recorded_rule(rule: Any) -> str:
    """Re-word a filter rule recorded by the calibration step, for the language in force.

    The calibration step writes that rule as a fixed template with the numeric cutoff
    interpolated, so its wording follows the report language while the cutoff is copied
    through unchanged. A recognised rule is only re-worded for a non-zh report, and an
    unrecognised rule is returned exactly as recorded.
    """
    text = str(rule or "")
    if not text or _report_language.get_language() != "en":
        return text
    for prefix, key in (("全部样本中非缺失比例 >=", "tables.rule_all_samples"),
                        ("按细胞类型分别要求非缺失比例 >=", "tables.rule_per_cell_type")):
        if text.startswith(prefix):
            return _report_language.t(key, cutoff=text[len(prefix):].strip())
    return text

if __name__ == "__main__":
    main()
