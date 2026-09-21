# -*- coding: utf-8 -*-
"""V23 phase C, step 1: freeze the six PiSPA GO queries and build both backgrounds.

Contract: GO_CONTRACT.md (CODEX_V23_MANUSCRIPT_SUPPORT_TASKBOOK_20260920).
Read-only on every project input; writes only inside the V23 package.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from gb_env import resolve_paths  # noqa: E402

# Release path adaptation: the frozen inputs and outputs come from a data archive instead of
# the manuscript working tree. No scientific setting below this block was changed.
_PATHS = resolve_paths(require_pispa=True)
REPO = _PATHS["data_root"]
TABLES = _PATHS["tables"]
CONFIG = _PATHS["config"]
TMP = _PATHS["tmp"]
RUN = _PATHS["pispa_run"]
GO_DIR = _PATHS["gmt_dir"]

GMT_ORDER = ["GO_BP.symbols.gmt", "GO_CC.symbols.gmt", "GO_MF.symbols.gmt"]
SPLIT_RE = re.compile("[;|,/\\s\\\\]+")
EXCLUDE_GENES: list = []
CONTRACTS = ["Cluster_1_vs_Cluster_2", "Cluster_1_vs_Cluster_3", "Cluster_2_vs_Cluster_3"]
LABEL = {
    "Cluster_1_vs_Cluster_2": "C1 - C2",
    "Cluster_1_vs_Cluster_3": "C1 - C3",
    "Cluster_2_vs_Cluster_3": "C2 - C3",
}
DIRECTIONS = ["up", "down"]
EXPECTED = {
    "Cluster_1_vs_Cluster_2_up": (325, 304),
    "Cluster_1_vs_Cluster_2_down": (1995, 1954),
    "Cluster_1_vs_Cluster_3_up": (625, 585),
    "Cluster_1_vs_Cluster_3_down": (301, 300),
    "Cluster_2_vs_Cluster_3_up": (1315, 1273),
    "Cluster_2_vs_Cluster_3_down": (42, 42),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def split_tokens(value) -> list:
    s = str(value)
    if s.lower() == "nan":
        return []
    out = []
    for tok in SPLIT_RE.split(s):
        tok = tok.strip()
        if tok and tok.lower() != "nan":
            out.append(tok.upper())
    return out


def load_gene_sets():
    """Replicate load_gene_sets('GO'): GO_BP then GO_CC then GO_MF, later overwrites."""
    combined = {}
    per_file_terms = {}
    dupes = []
    for name in GMT_ORDER:
        path = GO_DIR / name
        per_file_terms[name] = 0
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                term_id = parts[0].strip()
                desc = parts[1].strip() if parts[1].strip() else term_id
                genes = [g.strip() for g in parts[2:] if g.strip()]
                if not term_id or not genes:
                    continue
                per_file_terms[name] += 1
                if term_id in combined:
                    dupes.append(term_id)
                combined[term_id] = {"description": desc, "genes": sorted(set(genes))}
    return combined, per_file_terms, dupes


def write_tsv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            cells = []
            for c in columns:
                v = r.get(c, "")
                cells.append(str(v).replace("\t", " ").replace("\n", " "))
            fh.write("\t".join(cells) + "\n")


def main() -> int:
    started = time.time()
    for d in (TABLES, CONFIG, TMP):
        d.mkdir(parents=True, exist_ok=True)

    gene_sets, per_file_terms, dup_terms = load_gene_sets()
    merged_terms = len(gene_sets)
    U_H = set()
    for entry in gene_sets.values():
        U_H.update(entry["genes"])

    query_rows = []
    query_gene_rows = []
    manifest = {}
    dropped_report = {}
    for c in CONTRACTS:
        for d in DIRECTIONS:
            path = RUN / "processed_proteins" / c / (c + "_" + d + ".csv")
            df = pd.read_csv(path)
            ordered = {}
            raw_tokens = 0
            for value in df["PG.Genes"].tolist():
                toks = split_tokens(value)
                raw_tokens += len(toks)
                for tok in toks:
                    ordered.setdefault(tok, None)
            q_raw = set(ordered)
            q_h = q_raw & U_H
            key = c + "_" + d
            exp_rows, exp_genes = EXPECTED[key]
            query_rows.append(dict(
                contrast=LABEL[c], direction=d, contrast_key=c,
                protein_rows=len(df), raw_gene_tokens=raw_tokens,
                unique_raw_genes=len(q_raw), unique_genes_after_upper=len(q_raw),
                genes_in_U_H=len(q_h), genes_dropped_not_in_U_H=len(q_raw - U_H),
                expected_protein_rows=exp_rows, expected_unique_genes=exp_genes,
                identity_check=(
                    ("protein_rows_match" if len(df) == exp_rows else "PROTEIN_ROWS_MISMATCH")
                    + ";"
                    + ("unique_genes_match" if len(q_raw) == exp_genes else "UNIQUE_GENES_MISMATCH")
                ),
                query_file=rel(path), query_file_sha256=sha256(path),
            ))
            for g in ordered:
                if g in q_h:
                    query_gene_rows.append(dict(contrast=LABEL[c], direction=d, gene=g))
            manifest[key] = dict(
                label=LABEL[c], contrast=c, direction=d,
                protein_rows=int(len(df)), unique_genes=int(len(q_raw)),
                query_genes_in_U_H=int(len(q_h)),
                dropped_not_in_U_H=sorted(q_raw - U_H),
                query_file=rel(path), query_file_sha256=sha256(path),
            )
            dropped_report[key] = sorted(q_raw - U_H)

    write_tsv(
        TABLES / "go_queries.tsv",
        ["contrast", "direction", "protein_rows", "raw_gene_tokens", "unique_raw_genes",
         "unique_genes_after_upper", "genes_in_U_H", "genes_dropped_not_in_U_H",
         "expected_protein_rows", "expected_unique_genes", "identity_check",
         "query_file", "query_file_sha256"],
        query_rows,
    )
    write_tsv(TABLES / "go_queries_genes.tsv", ["contrast", "direction", "gene"], query_gene_rows)

    mapping_rows = []
    backgrounds = {}
    diff_info = {}
    for c in CONTRACTS:
        dpath = RUN / "processed_proteins" / ("differential_" + c + ".csv")
        df = pd.read_csv(dpath)
        missing = [col for col in ("P.Value", "adj.P.Val", "logFC") if col not in df.columns]
        if missing:
            raise SystemExit("missing columns in " + str(dpath) + ": " + str(missing))
        pv = pd.to_numeric(df["P.Value"], errors="coerce")
        adj = pd.to_numeric(df["adj.P.Val"], errors="coerce")
        lfc = pd.to_numeric(df["logFC"], errors="coerce")
        eligible = pv.notna() & adj.notna() & lfc.notna() & np.isfinite(pv) & np.isfinite(adj) & np.isfinite(lfc)
        genes_here = set()
        unnamed = 0
        for pg, pg_genes, ok in zip(df["PG.ProteinGroups"], df["PG.Genes"], eligible):
            toks = split_tokens(pg_genes)
            if not toks:
                unnamed += 1
            in_union = any(tok in U_H for tok in toks)
            for tok in toks:
                if tok in U_H:
                    genes_here.add(tok)
            if not toks:
                reason = "no_gene_annotation_row"
            elif not ok:
                reason = "not_eligible_missing_statistics"
            elif not in_union:
                reason = "gene_not_in_GO_annotation_universe"
            else:
                reason = "eligible_and_annotated"
            mapping_rows.append(dict(
                protein_group=str(pg), contrast=LABEL[c], source_gene_text=str(pg_genes),
                gene=";".join(toks), eligible="TRUE" if (ok and in_union) else "FALSE",
                reason=reason,
            ))
        backgrounds[c] = genes_here
        diff_info[c] = dict(
            total_rows=int(len(df)), eligible_rows=int(eligible.sum()),
            unnamed_gene_rows=int(unnamed),
            unique_genes=int(len({tok for v in df["PG.Genes"] for tok in split_tokens(v)})),
            annotated_genes=int(len(genes_here)),
            differential_file=rel(dpath), differential_file_sha256=sha256(dpath),
        )

    write_tsv(
        TABLES / "go_gene_mapping.tsv",
        ["protein_group", "contrast", "source_gene_text", "gene", "eligible", "reason"],
        mapping_rows,
    )

    sets_equal = len({frozenset(s) for s in backgrounds.values()}) == 1
    if not sets_equal:
        raise SystemExit("eligible annotated backgrounds differ across contrasts; contract requires separate tables")
    shared = backgrounds[CONTRACTS[0]]
    bg_rows = [dict(background_id="U_E_shared", contrast="shared_all_three", gene=g) for g in sorted(shared)]
    write_tsv(TABLES / "go_backgrounds.tsv", ["background_id", "contrast", "gene"], bg_rows)

    lock = dict(
        contract="GO_CONTRACT.md v frozen 2026-09-20",
        gmt_order=GMT_ORDER,
        gmt_files={name: dict(path=rel(GO_DIR / name), sha256=sha256(GO_DIR / name), terms=per_file_terms[name]) for name in GMT_ORDER},
        merged_terms=merged_terms,
        duplicate_term_ids_across_files=len(dup_terms),
        duplicate_term_id_examples=sorted(set(dup_terms))[:20],
        split_regex="[;|,/\\\\s\\\\\\\\]+",
        normalization="human upper-case, empty/NaN dropped",
        exclude_genes=EXCLUDE_GENES,
        min_overlap=1,
        fdr_cutoff=0.05,
        bh_scope="one contrast x direction query; family = every merged term with k>=1",
        background_historical=dict(id="U_H_union_GO_members", definition="union of all member genes of GO_BP+GO_CC+GO_MF", size=len(U_H)),
        background_experimental=dict(id="U_E_shared", definition="genes of eligible differential-table rows with a GO annotation (G_c intersect U_H), identical across the three contrasts", size=len(shared)),
        queries=manifest,
        differential_tables=diff_info,
        python=sys.version,
    )
    (CONFIG / "go_dataset_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    (TMP / "c1_timing.json").write_text(json.dumps({"seconds": round(time.time() - started, 2)}), encoding="utf-8")

    print("merged terms:", merged_terms, "dupes:", len(dup_terms), "U_H:", len(U_H), "U_E:", len(shared))
    for r in query_rows:
        print(r["contrast"], r["direction"], "rows", r["protein_rows"], "unique", r["unique_raw_genes"],
              "in U_H", r["genes_in_U_H"], "dropped", r["genes_dropped_not_in_U_H"], r["identity_check"])
    for c in CONTRACTS:
        print(LABEL[c], diff_info[c])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
