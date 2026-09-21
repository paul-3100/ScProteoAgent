# -*- coding: utf-8 -*-
"""S5: the three general revision requirements derived from the confirmed generation-side defects.

The general scientific writing requirements carried by the report request. They are stated once, applied to
every report, and are deliberately narrow: they do not touch pharmacological explanation, candidate
selection, or name matching, which are analysis-design choices rather than errors.

The same module provides the offline guards used to verify a revision round: a guard reports
violations, it never rewrites prose by hand.
"""
from __future__ import annotations

import re

import report_language  # same directory as this file; the shared zh/en registry

# --------------------------------------------------------------------------- language
# revision_block() is injected into every report request, so it has to exist in both
# languages. The zh value of every entry is the released literal copied byte for byte, so a zh
# run sends exactly the released requirement text; the en value is the same requirement in
# English. Placeholders stay in %-form (the block formats with %).
REVISION_TEXT = {
    "block_header": (
        "写作要求（通用科学写作约束，逐条适用于全文）：",
        "Writing requirements (general scientific writing constraints; every item applies to the "
        "whole report):",
    ),
    "item_format": ("- %s：%s", "- %s: %s"),
    "a_name": ("方向与坐标系一致", "Direction and coordinate frame are consistent"),
    "a_requirement": (
        "正文数值与方向必须使用报告显示方向；当句子引用源差异表名时，必须显式按该表自身符号叙述，"
        "不得在同一句混用两套符号。",
        "Numbers and directions in the text must use the display orientation of the report. When "
        "a sentence cites a source differential table by name, it must be narrated explicitly in "
        "that table's own sign, and the two sign conventions must never be mixed in one sentence.",
    ),
    "b_name": ("显著性表述绑定统计量", "Significance wording is bound to a statistic"),
    "b_requirement": (
        "凡使用显著/显著性表述，必须绑定源表给出的 FDR/q/adj.P 值；模块分数与样本级评分类结果"
        "只作方向性证据，不等同于显著性检验。",
        "Every statement of significance must be bound to the FDR/q/adj.P value given by the "
        "source table. Module scores and sample-level score-like results are directional evidence "
        "only and are not a significance test.",
    ),
    "c_name": ("任务结论与自身表格一致", "Task conclusions agree with their own tables"),
    "c_requirement": (
        "每个任务小节的结论必须与本节表格一致：模块归属以模块表的对应行为准，剂量方向以同一对比的"
        "差值符号为准，不得用药物类别或剂量先验替代表内数值。",
        "The conclusion of each task subsection must agree with that subsection's tables: module "
        "membership follows the matching row of the module table and the dose direction follows "
        "the sign of the difference in the same contrast; drug class or dose priors must not "
        "stand in for the values in the table.",
    ),
}
report_language.register("revisions", REVISION_TEXT)

REVISIONS = [
    {
        "id": "A",
        "name": "方向与坐标系一致",
        "requirement": "正文数值与方向必须使用报告显示方向；当句子引用源差异表名时，必须显式按该表自身符号叙述，不得在同一句混用两套符号。",
        "applies_to": ["current_matrix", "offline_enrichment"],
    },
    {
        "id": "B",
        "name": "显著性表述绑定统计量",
        "requirement": "凡使用显著/显著性表述，必须绑定源表给出的 FDR/q/adj.P 值；模块分数与样本级评分类结果只作方向性证据，不等同于显著性检验。",
        "applies_to": ["current_matrix"],
    },
    {
        "id": "C",
        "name": "任务结论与自身表格一致",
        "requirement": "每个任务小节的结论必须与本节表格一致：模块归属以模块表的对应行为准，剂量方向以同一对比的差值符号为准，不得用药物类别或剂量先验替代表内数值。",
        "applies_to": ["current_matrix"],
    },
]


SIGNIFICANCE_WORDS = ("显著", "显著性")
NEGATION = ('未达到', '未检出', '未观察到', '不显著', '未见', '未显示', '没有达到', '不能写成', '不应写成', '不足以', '不替代', '并非')
NA_STAT = ('FDR: NA', 'FDR=NA', 'FDR 为 NA', 'FDR为NA', '没有 FDR', '无 FDR')
STAT_WORDS = ("FDR", "adj.P", "q 值", "q值", "p.adjust", "P.Value", "校正后")
MODULE_WORDS = ("模块", "module", "评分", "score")
DIRECTION_WORDS = ("较高", "较低", "更高", "更低", "上调", "下调", "高于", "低于")


def _item_text(item: dict, lang: str) -> tuple:
    """(name, requirement) of one item in the requested language.

    zh returns the released fields unchanged (including the join of a list-valued requirement,
    which no entry currently uses). en reads the registered translation; a missing entry renders
    the registry's explicit missing-template marker instead of the Chinese text.
    """
    if lang == "zh":
        req = item["requirement"]
        if isinstance(req, (tuple, list)):
            req = "".join(str(part) for part in req)
        return item["name"], req
    stem = "revisions.%s_" % str(item["id"]).lower()
    return (report_language.t_in(lang, stem + "name"),
            report_language.t_in(lang, stem + "requirement"))


def revision_block() -> str:
    """The general scientific writing requirements carried by the report request.

    R19: rendered without rule ids and without any reference to scoring, evaluation or another
    model report: this text is part of the request the report is generated from, so an id such as
    "R1" would be echoed into the reader-facing report as an instruction log.

    The block is rendered in the active report language. A zh run renders the released text byte
    for byte; an en run renders the registered English wording, so the request a model receives
    carries the same constraints in the language the report is written in.
    """
    lang = report_language.get_language()
    parts = [report_language.t_in(lang, "revisions.block_header")]
    item_format = report_language.t_in(lang, "revisions.item_format")
    for item in REVISIONS:
        name, req = _item_text(item, lang)
        parts.append(item_format % (name, req))
    return chr(10).join(parts)


def _sentences(text: str):
    for raw in re.split(r"[。；\n]", text):
        item = raw.strip()
        if item:
            yield item


def guard_r1(text: str, index: dict):
    """Direction guard: every quoted protein/value pair must agree with the table sign."""
    violations = []
    candidates = index.get("candidates") or {}
    for sentence in _sentences(text):
        if not any(word in sentence for word in DIRECTION_WORDS):
            continue
        for rec in candidates.values():
            name = str(rec.get("candidate") or "")
            if not name or name not in sentence:
                continue
            window = re.search(re.escape(name) + r"[^。；]{0,80}", sentence)
            if not window:
                continue
            values = re.findall(
                r"(?:logFC|log2FC|效应量|Δ)?\s*[=＝]?\s*([-+]?\d+(?:\.\d+)?)",
                window.group(0))
            display = rec.get("logFC_display")
            if display is None or not values:
                continue
            display = float(display)
            groups = [str(g) for g in (rec.get("groups") or []) if g]
            if len(groups) < 2:
                continue
            stated = None
            for group in groups:
                for word in ("较高", "更高", "上调", "高于"):
                    if (re.search(re.escape(group) + r"[^，。；]{0,12}" + word, sentence)
                            or re.search(word + r"[^，。；]{0,12}" + re.escape(group), sentence)):
                        stated = group
            if stated is None:
                continue
            for raw in values:
                value = float(raw)
                if abs(value) > 10:
                    continue
                agrees = (stated == groups[0] and value > 0) or (stated == groups[1] and value < 0)
                if not agrees:
                    violations.append({"rule": "R1", "subject": name, "stated_higher_in": stated,
                                       "quoted_value": value, "display_logFC": display,
                                       "sentence": sentence[:200]})
                    break
    return violations


def guard_r2(text: str):
    """Significance guard (T2): module significance needs a bound statistic, not FDR: NA."""
    violations = []
    for sentence in _sentences(text):
        if not any(word in sentence for word in SIGNIFICANCE_WORDS):
            continue
        if any(neg in sentence for neg in NEGATION):
            continue
        if not any(word in sentence for word in MODULE_WORDS):
            continue
        stats_bound = any(word.lower() in sentence.lower() for word in STAT_WORDS)
        na_only = any(na.lower() in sentence.lower() for na in NA_STAT)
        if stats_bound and not na_only:
            continue
        violations.append({"rule": "R2", "sentence": sentence[:200], "fdr_na": bool(na_only)})
    return violations


def guard_r3(text: str, index: dict):
    """Cross-table guard: a claim about a module must not contradict the module table's sign."""
    violations = []
    modules = index.get("modules") or {}
    for key, rec in modules.items():
        name = str(key[1] if isinstance(key, tuple) and len(key) > 1 else key)
        delta = rec.get("delta")
        if delta is None or not name:
            continue
        delta = float(delta)
        for sentence in _sentences(text):
            if name not in sentence:
                continue
            if "下降" in sentence or "降低" in sentence:
                if delta > 0:
                    violations.append({"rule": "R3", "module": name, "table_delta": delta,
                                       "sentence": sentence[:200]})
    return violations


def run_guards(text: str, index: dict) -> dict:
    """All three guards over one finished report body."""
    return {"R1": guard_r1(text, index), "R2": guard_r2(text), "R3": guard_r3(text, index)}


def main() -> int:
    print(revision_block())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
