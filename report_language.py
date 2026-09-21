"""Reader-facing report language for scProteoAgent (zh default, en opt-in).

One small shared registry, not an internationalisation platform. Every module that
emits reader-facing text registers a flat namespace of key -> (zh, en) pairs and
calls t("namespace.key", **fmt). If the active language has no template for a key,
t returns an explicit MISSING_EN_TEMPLATE marker and records the key, so an English
report can never silently fall back to Chinese.

Invariants relied on elsewhere:

* zh templates are copied verbatim from the pre-existing literals, so a zh run
  renders byte-identical report text to the released baseline.
* The language is never inferred from the task text.
"""

from __future__ import annotations

import re
import threading

SUPPORTED = ("zh", "en")
DEFAULT_LANGUAGE = "zh"
MISSING_PREFIX = "MISSING_EN_TEMPLATE:"

TICK = chr(96)
FENCE = TICK * 3

# zh templates are the frozen originals; en templates are the added language.
_TABLES = {}
_lock = threading.Lock()
_current = DEFAULT_LANGUAGE
_gaps = []
_gap_log_path = None

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CJK_RUN_RE = re.compile(
    r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff01-\uff65]+"
)


class UnknownReportLanguage(ValueError):
    """Raised for a language value outside SUPPORTED."""


def normalize(value):
    """Return a supported language code, or raise. Never guesses from content."""
    if value is None:
        return DEFAULT_LANGUAGE
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_LANGUAGE
    if text not in SUPPORTED:
        raise UnknownReportLanguage(
            "unsupported report language %r; expected one of %s"
            % (value, ", ".join(SUPPORTED))
        )
    return text


def set_language(value):
    """Set the process-wide report language and return the normalised code."""
    global _current
    resolved = normalize(value)
    with _lock:
        _current = resolved
    return resolved


def get_language():
    return _current


def register(namespace, table):
    """Register (or replace) one namespace of key -> (zh, en) templates."""
    clean = {}
    for key, value in dict(table).items():
        if isinstance(value, (tuple, list)):
            zh = value[0] if len(value) > 0 else ""
            en = value[1] if len(value) > 1 else ""
        else:
            zh, en = value, ""
        clean[str(key)] = ("" if zh is None else str(zh), "" if en is None else str(en))
    with _lock:
        _TABLES[str(namespace)] = clean


def registered_namespaces():
    return sorted(_TABLES)


def table(namespace):
    return dict(_TABLES.get(str(namespace)) or {})


def _record_gap(key):
    with _lock:
        if key not in _gaps:
            _gaps.append(key)
        path = _gap_log_path
    if path:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(key + chr(10))
        except OSError:
            pass


def set_gap_log(path):
    """Append every missing-template key to this file as it is hit."""
    global _gap_log_path
    _gap_log_path = str(path) if path else None


def reset_gaps():
    with _lock:
        del _gaps[:]


def gaps():
    with _lock:
        return list(_gaps)


def has_template(lang, key):
    namespace, _, name = str(key).partition(".")
    entry = (_TABLES.get(namespace) or {}).get(name)
    if entry is None:
        return False
    return bool(entry[0 if normalize(lang) == "zh" else 1])


def t_in(lang, key, **fmt):
    namespace, _, name = str(key).partition(".")
    entry = (_TABLES.get(namespace) or {}).get(name)
    if entry is None:
        _record_gap(str(key))
        return MISSING_PREFIX + str(key)
    value = entry[0] if normalize(lang) == "zh" else entry[1]
    if not value:
        _record_gap(str(key))
        return MISSING_PREFIX + str(key)
    return value.format(**fmt) if fmt else value


def t(key, **fmt):
    """Template for the active language; explicit marker plus gap record if absent."""
    return t_in(_current, key, **fmt)


def cjk_count(text):
    return len(CJK_RE.findall(str(text)))


def latin_count(text):
    return len(re.findall(r"[A-Za-z]", str(text)))


def cjk_runs(text, minimum=1):
    """Contiguous CJK/punctuation runs of at least minimum characters."""
    return [
        m.group(0)
        for m in CJK_RUN_RE.finditer(str(text))
        if len(m.group(0)) >= minimum
    ]


def body_language_ok(text, lang):
    """zh keeps the released CJK-share floor; en requires Latin dominance."""
    lang = normalize(lang)
    cjk = cjk_count(text)
    latin = latin_count(text)
    denom = cjk + latin
    if not denom:
        return lang == "en"
    if lang == "zh":
        return round(cjk / denom, 4) >= 0.10
    return latin / denom >= 0.60


def _is_skippable_line(stripped):
    lower = stripped.lower()
    return (
        stripped.startswith(("|", "-", "*", TICK))
        or "\\" in stripped
        or "/" in stripped
        or ".csv" in lower
        or ".json" in lower
        or ".png" in lower
        or "current_matrix" in lower
        or "offline_enrichment" in lower
        or "external_annotation" in lower
    )


def wrong_language_paragraph_present(text, lang):
    """zh: a long purely-English paragraph. en: a long purely-Chinese paragraph."""
    lang = normalize(lang)
    for line in str(text).splitlines():
        stripped = line.strip()
        if len(stripped) < 160 or _is_skippable_line(stripped):
            continue
        cjk = cjk_count(stripped)
        latin = latin_count(stripped)
        if lang == "zh" and latin > 120 and cjk < 3:
            return True
        if lang == "en" and cjk > 120 and latin < 3:
            return True
    return False


def strip_code_and_paths(text):
    """Drop fenced blocks, inline code and path-like tokens before a residue scan."""
    body = re.sub(r"(?ms)^\\s*" + FENCE + r".*?^" + FENCE + r"\\s*$", " ", str(text))
    body = re.sub(TICK + r"[^" + TICK + r"]*" + TICK, " ", body)
    body = re.sub(r"[A-Za-z]:[\\\\/][^\\s)\\]]+", " ", body)
    body = re.sub(
        r"(?<![A-Za-z0-9_./])[\\w.-]+\\.(csv|json|md|png|pdf|svg|tsv|txt|yaml|yml|xlsx)",
        " ",
        body,
    )
    return body


def residual_cjk(text, minimum=1):
    """CJK runs still present in reader-facing text after removing code and paths."""
    return cjk_runs(strip_code_and_paths(text), minimum=minimum)


def write_gap_report(path, fmt="tsv"):
    """Persist the ordered, de-duplicated missing-template keys."""
    rows = gaps()
    lines = (["key"] + rows) if str(fmt).lower() == "tsv" else rows
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(chr(10).join(lines) + chr(10))
    return len(rows)


# --------------------------------------------------------------------------- core skeleton

# The released zh headings, copied verbatim. The en set is a NEW set, deliberately
# different from LEGACY_REPORT_HEADINGS in main_agent.py, which stays the negative
# check that tells a released report apart from a pre-release one.
CORE = {
    "h_executive": ("一、执行摘要", "1. Executive Summary"),
    "h_scoring": ("二、核心证据首页", "2. Scoring Evidence First Screen"),
    "h_findings": ("三、主要发现", "3. Main Findings"),
    "h_boundary": ("四、证据边界", "4. Evidence Boundary"),
    "h_appendix": ("五、可复核证据附录", "5. Evidence Appendix"),
    "s0": ("0. 关键结论速览", "0. Key Conclusions at a Glance"),
    "s1": ("1. 数据与预处理", "1. Data and Preprocessing"),
    "s2": (
        "2. 任务一：聚类与迁移状态的对应",
        "2. Task 1: Cluster-to-migration-state correspondence",
    ),
    "s3": (
        "3. 任务二：迁移相关 Cluster 的功能模块",
        "3. Task 2: Functional modules of the migration-related clusters",
    ),
    "s4": ("4. 任务三：骨架调控候选", "4. Task 3: Cytoskeleton regulation candidates"),
    "s5": (
        "5. 任务四：对照群体的隐性差异",
        "5. Task 4: Hidden differences in the control population",
    ),
    "s6": ("6. 总结与启发式推测", "6. Summary and Heuristic Inference"),
    "s7": ("7. 可复核资产清单", "7. Reproducible Asset List"),
    "s8": (
        "8. 结论边界与方法局限",
        "8. Conclusion Boundaries and Method Limitations",
    ),
    "story_conclusion": ("结论边界与方法局限", "Conclusion Boundaries and Method Limitations"),
    "story_assets": ("可复核资产清单", "Reproducible Asset List"),
    "story_synthesis": ("总结与启发式推测", "Summary and Heuristic Inference"),
    "task_word": ("任务", "Task"),
    "analysis_conditions": ("本次分析条件", "Analysis Conditions for This Run"),
    "figure_index": ("可复核图表索引", "Reproducible Figure Index"),
    "qc_semantics": ("缺失与检出口径", "Missingness and Detection Conventions"),
    # reader-facing tokens the report self-check looks for. These are wording probes, so
    # each language lists the words that actually appear in that language's report.
    "tok_core_contrast_table": ("核心对比表", "Core contrast table"),
    "tok_contrast_header": ("对比", "Contrast"),
    "tok_candidate_header": ("候选", "Candidate"),
    "tok_representative_header": ("代表蛋白", "Representative protein"),
    # the noun that turns a subject header into a table name ("代表蛋白" + "表" /
    # "Representative protein" + " table"). Split out because the released zh literal is a
    # concatenation, and the English branch previously appended the Chinese character.
    "tok_table_word": ("表", " table"),
    "tok_table_significance": ("通过筛选|显著", "passed screening|significant"),
    "tok_current_matrix": ("当前矩阵", "current matrix"),
    "tok_offline_enrichment": ("离线富集", "offline enrichment"),
    "tok_external_annotation": ("外部注释", "external annotation"),
    "tok_heuristic": ("启发式推测", "heuristic inference"),
    "tok_followup": ("后续验证建议", "follow-up validation"),
    "tok_boundary_word": ("结论边界", "conclusion boundaries"),
    "note_historical_check_names": (
        "注：chinese_* 是历史检查名，判定的是当前报告语言对应的标题与用词。",
        "Note: the chinese_* names are historical check identifiers; each one is evaluated "
        "against the headings and wording of the active report language.",
    ),
}
register("core", CORE)

_CANONICAL_KEYS = ("executive", "scoring", "findings", "boundary", "appendix")


def canonical_headings(lang=None):
    lang = normalize(lang or _current)
    return {name: t_in(lang, "core.h_" + name) for name in _CANONICAL_KEYS}


def story_headings(lang=None):
    lang = normalize(lang or _current)
    return [t_in(lang, "core.s%d" % i) for i in range(9)]


def reader_headings(lang=None):
    """Canonical plus story headings for the language (used to strip sections)."""
    return list(canonical_headings(lang).values()) + story_headings(lang)


def t_list(key, **fmt):
    """Template split on the pipe character, for multi-word wording probes."""
    return tuple(part for part in t(key, **fmt).split("|") if part)


# --------------------------------------------------------------------------- task tokens
# A task text names the proteins the analyst wants examined. That list must not be decided by
# what happens to be in the matrix: an explicitly named protein that is absent from the data is
# still reported, as "not matched", rather than dropped. It must also not become English prose:
# a token that cannot be resolved in the run is suppressed only when it is an ordinary word.
#
# One implementation, shared by the candidate audit in main_agent.py and the candidate trace in
# analysis_extensions.py, so the two cannot drift apart.

TASK_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{1,9}\b")

# Function words and generic domain nouns. Deliberately short, and never used on its own:
# resolution wins, so a word that is also a real symbol survives whenever the run's own data
# contains it.
PROSE_TOKENS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can cannot could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it
its itself just me more most my myself no nor not now of off on once only or other others our
ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself yourselves
analysis analyses cell cells data dataset datasets demo figure figures gene genes group groups
matrix matrices method methods note notes number numbers please protein proteins report reports
request requested result results run runs sample samples section sections software step steps
study studies synthetic table tables target targets task tasks test tests time times use used
using value values
also assess check close comment compare compute describe evaluate finally finish give include
list next overall provide reuse show state status summarize tell then val write
""".split())


def has_lowercase_occurrence(text, token):
    """True when the same word, matched whole, also occurs all lowercase in the text.

    English prose repeats its words in lowercase ("The ... the ..."); a gene symbol does not.
    """
    if not text or not token:
        return False
    pattern = re.compile(r"(?<![A-Za-z])" + re.escape(token) + r"(?![A-Za-z])", re.IGNORECASE)
    for match in pattern.finditer(text):
        piece = match.group(0)
        if piece == piece.lower():
            return True
    return False


def is_prose_token(token, text="", resolved=None, labels=None):
    """Classify one task token as prose, which is what keeps it out of the candidate list.

    Resolution wins: a token present in this run's own data is never treated as prose, so a
    protein whose symbol is also an ordinary word still survives when the matrix holds it.
    labels carries the run's design group values: a group name is not a protein either.
    """
    if not token:
        return False
    if resolved and token in resolved:
        return False
    if labels and token.upper() in labels:
        return True
    if token.lower() in PROSE_TOKENS:
        return True
    return has_lowercase_occurrence(text, token)


def classify_task_tokens(text, resolved=None, stop=None, labels=None):
    """The task-named candidates of one task text, as (kept, suppressed).

    kept preserves first-seen order and is de-duplicated. suppressed lists the prose tokens
    that were not treated as candidates, so a suppression can be reported rather than hidden.
    A token already covered by stop is dropped without being reported as prose, which keeps the
    released stop-word behaviour exactly as it was.
    """
    kept, suppressed = [], []
    for token in TASK_TOKEN_RE.findall(text or ""):
        if len(token) < 2:
            continue
        if stop and token.upper() in stop:
            continue
        if is_prose_token(token, text, resolved, labels):
            if token not in suppressed:
                suppressed.append(token)
            continue
        if token not in kept:
            kept.append(token)
    return kept, suppressed
