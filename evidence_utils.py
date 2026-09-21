import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SECRET_PATTERNS = [
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^,\s}\]]+"),
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def redact_text(value: Any, max_chars: int = 4000) -> str:
    text = str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]..."
    return text


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(v) for v in list(value)[:200]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return redact_text(value) if isinstance(value, str) else value
    return redact_text(value)


def append_jsonl(path: str | os.PathLike[str], record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = _safe_jsonable(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def evidence_path(run_folder: str) -> str:
    return str(Path(run_folder) / "evidence_ledger.jsonl")


def external_knowledge_path(run_folder: str) -> str:
    return str(Path(run_folder) / "external_knowledge.jsonl")


def append_evidence(
    run_folder: str,
    *,
    source: str,
    claim: str,
    tool: Optional[str] = None,
    files: Optional[Iterable[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    confidence: str = "moderate",
    limitations: Optional[Iterable[str]] = None,
) -> None:
    append_jsonl(
        evidence_path(run_folder),
        {
            "timestamp": utc_now_iso(),
            "evidence_source": source,
            "tool": tool,
            "claim": claim,
            "files": list(files or []),
            "metrics": metrics or {},
            "confidence": confidence,
            "limitations": list(limitations or []),
        },
    )


def append_external_knowledge(
    run_folder: str,
    *,
    tool: str,
    query: Any,
    evidence_source: str,
    records: List[Dict[str, Any]],
    status: str = "completed",
) -> None:
    append_jsonl(
        external_knowledge_path(run_folder),
        {
            "timestamp": utc_now_iso(),
            "tool": tool,
            "query": query,
            "evidence_source": evidence_source,
            "status": status,
            "records": records[:100],
        },
    )
