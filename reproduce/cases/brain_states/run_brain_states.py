#!/usr/bin/env python
"""Brain development case: deterministic re-derivation of the state and module tables.

Two things are computed and kept separate: per-protein summaries for a fixed marker set, and
module scores per labelled state under two zero conventions. Zeros are values in the delivered
matrix. The value convention keeps them; the missing convention drops zero members per cell
before the cell score is averaged. Neither convention is presented as the correct reading of the
zeros, and the unlabelled cells form their own group rather than being merged into a state.

Usage:
    py run_brain_states.py --data-dir <data archive root> --out-dir <output directory>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from table_compare import Field, Ledger, TableSpec  # noqa: E402

THEME = "brain_states"
UNLABELLED = "Unlabelled"
# The writer of the frozen brain tables wrote str() of the groupby key, so the unlabelled cells
# appear as the literal text "nan" in the frozen row keys. The recomputed frames carry the same
# text; the key comparison then matches row by row instead of turning one key into a missing one.
UNLABELLED_KEY = "nan"

INPUTS = {
    "matrix": "case_tables/brain/run_processed/ProQuant_Normalized.csv",
    "sampleinfo": "case_tables/brain/run_processed/SampleInfo_Filtered.csv",
    "members": "case_tables/brain/e06_brain_module_members.tsv",
}
TARGETS = {
    "all_states": "source_tables/brain_module_all_states.tsv",
    "conventions": "source_tables/brain_module_conventions.tsv",
    "markers": "source_tables/brain_markers.tsv",
    "recomputed": "case_tables/brain/e06_brain_module_recomputed.tsv",
    "coverage": "case_tables/brain/e06_brain_cell_coverage.tsv",
    "missingness": "case_tables/brain/e06_brain_missingness_sensitivity.tsv",
}
MARKER_MODULE = {"brain_rg_org_progenitor": "Progenitor",
                 "brain_ipc_en_transition": "IPC-EN transition",
                 "brain_en_maturation": "Neuronal maturation"}


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def normalize_gene(value) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def split_tokens(value) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [token for token in re.split(r"[;,|/\s]+", str(value)) if token]


# ------------------------------------------------------------------- pre-registered specs
# One TableSpec per compared frozen table. Every frozen column is either a Field (compared, with
# its kind and any declared rounding) or is declared excluded with a reason. The declared
# resolutions come from the writer's own code:
#   * V9_EXPLORATION_SUPPORT/scripts/e06_brain_recheck.py                  (members, state table)
#   * V9_EXPLORATION_SUPPORT/scripts/e06_brain_coverage.py                 (cell coverage)
#   * V9_EXPLORATION_SUPPORT/scripts/e06_brain_missingness_sensitivity.py  (zero conventions)
# The source_tables/brain_* tables are verbatim copies of the V14 figure-round panel source tables
# (V15_DATA_FIGURE_REFINEMENT_20260918/scripts/prepare_data_b.py, copy_theme); the figure pipeline
# stores those values as floats without a shorter declared format, so they carry no rounding
# allowance and are compared at the pre-registered 1e-7 relative tolerance.

IDENTITY_NOTE = "row key or label; compared as identity"

BRAIN_MEMBERS_SPEC = TableSpec(
    key_fields=("module", "gene"),
    fields=(
        Field("module", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("gene", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("matched", "text",
              note="the frozen raw text is True/False, so the boolean is compared as text; the "
                   "numeric flag kind would report these tokens as unparsable"),
        Field("n_matrix_rows", "int", note="matrix rows the gene tokens match"),
        Field("mean_detection_rate", "float", allows_missing=True, format_decimals=4,
              note="written by V9_EXPLORATION_SUPPORT/scripts/e06_brain_recheck.py "
                   "(gene_rows.append: round(float(detection_rate.loc[ids].mean()), 4)); left "
                   "empty when a member gene matches no matrix row, which the recomputation "
                   "reproduces as empty"),
        Field("gene_cell_values", "text", allows_missing=True, source_class="IDENTITY_CHECK",
              note="PG.Genes text of the matched matrix row, echoed from this same frozen member "
                   "table; empty where the gene matches no row"),
    ),
    label="module membership resolved against the delivered matrix",
)

_BRAIN_STATE_FIELDS = (
    Field("module", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
    Field("state", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
    Field("zero_treatment", "text", source_class="IDENTITY_CHECK",
          note="'value' keeps the exact zeros, 'missing' drops them per cell; the label is the "
               "switch the recomputation iterates over"),
    Field("score", "float", note="state mean of the cell-level module scores; the frozen text "
                                 "stores the value written by the figure pipeline with no shorter "
                                 "declared format, so rtol 1e-7 applies"),
    Field("n_cells", "int", note="cells of the state group"),
    Field("n_scored", "int", note="cells with a finite module score"),
    Field("mean_member_depth", "float", note="mean member detection depth of the state; no "
                                             "declared format, rtol 1e-7 applies"),
)

BRAIN_CONVENTIONS_SPEC = TableSpec(
    key_fields=("module", "state", "zero_treatment"),
    fields=_BRAIN_STATE_FIELDS,
    label="state scores under both zero conventions, annotated states only",
)

BRAIN_ALL_STATES_SPEC = TableSpec(
    key_fields=("module", "state", "zero_treatment"),
    fields=_BRAIN_STATE_FIELDS,
    label="state scores including the unlabelled pool",
)

BRAIN_MARKERS_SPEC = TableSpec(
    key_fields=("gene", "module", "state"),
    fields=(
        Field("gene", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("module", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("module_group", "identity", source_class="IDENTITY_CHECK",
              note="module group label carried by the frozen marker table"),
        Field("state", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("n_cells", "int"),
        Field("n_detected", "int", note="cells of the state whose value is not zero"),
        Field("detected_fraction", "float",
              note="detected / n_cells; the frozen write is a float division with no declared "
                   "format, so rtol 1e-7 applies"),
        Field("mean_z_value", "float", note="mean of the protein-wise z score over the state"),
        Field("mean_value", "float", note="mean of the delivered value over the state"),
        Field("matrix_rows_matched", "int", note="matrix rows matched by the gene tokens"),
    ),
    label="marker proteins across the eight labelled states",
)

BRAIN_COVERAGE_SPEC = TableSpec(
    key_fields=("module", "cluster"),
    fields=(
        Field("module", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("cluster", "identity", source_class="IDENTITY_CHECK",
              note="state label as written by the writer; the unlabelled pool is the literal text"
                   " %r (e06_brain_coverage.py writes str() of the groupby key)" % UNLABELLED_KEY),
        Field("n_cells", "int", note="cells of the group"),
        Field("mean_members_detected", "float", format_decimals=3,
              note="written by V9_EXPLORATION_SUPPORT/scripts/e06_brain_coverage.py "
                   "(cellcov_rows.append: round(float(v.mean()), 3))"),
        Field("min_members_detected", "int"),
        Field("max_members_detected", "int"),
        Field("frac_cells_zero_members", "float", format_decimals=4,
              note="written by e06_brain_coverage.py "
                   "(round(float((v.fillna(0) == 0).mean()), 4)); cells without a member count "
                   "are counted as zero members there"),
        Field("genes_defined", "int", note="member genes of the module"),
        Field("genes_matched", "int", note="member genes with at least one matrix row"),
    ),
    label="member detection coverage per module and group",
)

BRAIN_MISSINGNESS_SPEC = TableSpec(
    key_fields=("module", "cluster"),
    fields=(
        Field("module", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("cluster", "identity", source_class="IDENTITY_CHECK",
              note="state label as written by the writer; the unlabelled pool is the literal text"
                   " %r" % UNLABELLED_KEY),
        Field("n_cells", "int"),
        Field("mean_score_zeros_as_values", "float", format_decimals=5,
              note="written by e06_brain_missingness_sensitivity.py "
                   "(round(float(s['_A'].mean()), 5))"),
        Field("mean_score_zeros_as_missing", "float", format_decimals=5,
              note="written by e06_brain_missingness_sensitivity.py "
                   "(round(float(s['_B'].mean()), 5))"),
        Field("mean_cluster_depth_nonzero_frac_all_proteins", "float", format_decimals=4,
              note="written by e06_brain_missingness_sensitivity.py "
                   "(round(float(s['_depth'].mean()), 4))"),
    ),
    excluded=(
        ("spearman_rho_scoreA_vs_depth",
         "per-module Spearman rho of the state score against the cluster depth; the writer "
         "computed it with scipy.stats.spearmanr (e06_brain_missingness_sensitivity.py), which "
         "this case script does not depend on, and the value is constant per module"),
        ("spearman_p_scoreA_vs_depth",
         "P value of the same per-module Spearman correlation, from scipy.stats.spearmanr; not "
         "recomputed here"),
    ),
    label="score sensitivity to the zero convention",
)

BRAIN_STATE_SCORES_SPEC = TableSpec(
    key_fields=("module", "group"),
    fields=(
        Field("module", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("group", "identity", source_class="IDENTITY_CHECK",
              note="state label as written by the writer; the unlabelled pool is the literal text"
                   " %r" % UNLABELLED_KEY),
        Field("recomputed_n_cells", "int"),
        Field("recomputed_mean", "float",
              note="mean of the cell-level module scores of the group; the frozen write is "
                   "float(col.mean()) with no declared format"),
        Field("recomputed_median", "float", note="median of the same cell-level scores"),
        Field("n_nonmissing", "int", note="cells with a finite module score"),
        Field("genes_defined", "int"),
        Field("genes_matched", "int"),
        Field("missing_genes", "text", allows_missing=True,
              note="';'.join of the member genes without a matrix row "
                   "(e06_brain_recheck.py); empty when no member gene is missing, which the "
                   "recomputation reproduces"),
        Field("matrix_rows_matched", "int"),
        Field("mean_member_detection_rate", "float", format_decimals=5,
              note="written by e06_brain_recheck.py "
                   "(round(float(detection_rate.loc[row_ids].mean()), 5))"),
    ),
    label="state-level module scores of the nine groups",
)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Brain states case, deterministic re-run.")
    parser.add_argument("--data-dir", required=True, help="data archive root (read-only)")
    parser.add_argument("--out-dir", required=True, help="output directory (only place written)")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    wanted = {**INPUTS, **TARGETS}
    missing = [rel for rel in wanted.values() if not (data_dir / rel).is_file()]
    if missing:
        for rel in missing:
            print("FATAL: missing input " + str(data_dir / rel))
        print("FATAL: %d required input(s) missing under --data-dir; nothing written." % len(missing))
        return 2

    hashes_before = {rel: sha256_file(data_dir / rel) for rel in wanted.values()}
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(THEME)

    matrix = read_table(data_dir / INPUTS["matrix"])
    sampleinfo = read_table(data_dir / INPUTS["sampleinfo"])
    members = read_table(data_dir / INPUTS["members"])

    samples = [str(s) for s in sampleinfo["FileName"]]
    missing_columns = [s for s in samples if s not in matrix.columns]
    extra_columns = [c for c in matrix.columns
                     if c not in ("PG.ProteinGroups", "PG.Genes") and c not in samples]
    values = matrix[samples].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    gene_of_row = matrix["PG.Genes"] if "PG.Genes" in matrix.columns else matrix["Gene"]
    state_of = {}
    for sample, cluster in zip(samples, sampleinfo["Cluster"]):
        state_of[sample] = UNLABELLED if pd.isna(cluster) or str(cluster).strip() in ("", "nan") \
            else str(cluster)
    states = sorted({state_of[s] for s in samples if state_of[s] != UNLABELLED})
    groups = states + [UNLABELLED]
    cell_index = {state: [i for i, s in enumerate(samples) if state_of[s] == state]
                  for state in groups}
    counts = {state: len(cell_index[state]) for state in groups}

    shape_ok = (matrix.shape[0] > 0 and len(samples) == 2310 and not missing_columns
                and len(states) == 8 and counts[UNLABELLED] == 805)
    ledger.add("brain_cell_identity", "delivered cells and state labels",
               INPUTS["sampleinfo"],
               "2,310 cells; 8 labelled states and 805 unlabelled cells carried as their own group",
               "%d cells, %d labelled states, %d labelled and %d unlabelled; %d sample-like "
               "columns without SampleInfo rows are not used" % (
                   len(samples), len(states), len(samples) - counts[UNLABELLED],
                   counts[UNLABELLED], len(extra_columns)),
               "PASS" if shape_ok else "FAIL", "exact",
               "SampleInfo_Filtered.csv Cluster column; the unlabelled pool is a separate group")

    row_tokens = [set(normalize_gene(t) for t in split_tokens(g)) for g in gene_of_row]
    member_rows = []
    row_index = {}
    module_members = {}
    for _, row in members.iterrows():
        wanted_tokens = set(normalize_gene(t) for t in split_tokens(row["gene_cell_values"]))
        hits = [i for i, tokens in enumerate(row_tokens) if tokens & wanted_tokens]
        row_index[(str(row["module"]), str(row["gene"]))] = hits
        module_members.setdefault(str(row["module"]), []).extend(hits)
        # The writer of this frozen table (V9_EXPLORATION_SUPPORT/scripts/e06_brain_recheck.py,
        # gene_rows.append) rounds the rate to four decimals and writes no rate at all when the
        # member gene matches no matrix row; the recomputation follows that convention and leaves
        # the cell empty instead of substituting a zero rate.
        member_rows.append({"module": row["module"], "gene": row["gene"],
                            "matched": len(hits) > 0, "n_matrix_rows": len(hits),
                            "mean_detection_rate": float((values[hits] != 0).mean()) if hits
                            else np.nan,
                            "gene_cell_values": row["gene_cell_values"]})
    recomputed_members = pd.DataFrame(member_rows, columns=list(members.columns))
    ledger.add_table("brain_module_members", "module membership resolved against the matrix",
                     data_dir / INPUTS["members"], recomputed_members, BRAIN_MEMBERS_SPEC,
                     notes="the three member genes without a matrix row carry an empty detection "
                           "rate in the frozen table; the recomputation leaves those cells empty "
                           "as the writer does (see the field note)",
                     evidence="gene tokens of PG.Genes matched after upper-casing and removing "
                              "non-alphanumeric characters")

    zero_presence = (values != 0)
    depth = zero_presence.mean(axis=0)

    def zscore_rows(block: np.ndarray) -> np.ndarray:
        """Per-protein z scores across the delivered cells (sample standard deviation)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.nanmean(block, axis=1, keepdims=True)
            sd = np.nanstd(block, axis=1, ddof=1, keepdims=True)
            return (block - mu) / sd

    cell_scores = {}
    for module, hits in module_members.items():
        hits = sorted(set(hits))
        carried = values[hits]
        present = zero_presence[hits]
        with np.errstate(invalid="ignore", divide="ignore"):
            if hits:
                score_value = np.nanmean(zscore_rows(carried), axis=0)
                masked = np.where(present, carried, np.nan)
                score_missing = np.nanmean(zscore_rows(masked), axis=0)
            else:
                score_value = np.full(len(samples), np.nan)
                score_missing = np.full(len(samples), np.nan)
        cell_scores[module] = {
            "value": score_value,
            "missing": score_missing}
    score_rows = []
    for module in sorted(cell_scores):
        for convention in ("value", "missing"):
            scores = cell_scores[module][convention]
            for state in groups:
                idx = cell_index[state]
                finite = scores[idx][np.isfinite(scores[idx])]
                score_rows.append({"module": module, "state": state,
                                   "zero_treatment": convention,
                                   "score": float(finite.mean()) if finite.size else np.nan,
                                   "n_cells": len(idx), "n_scored": int(finite.size),
                                   "mean_member_depth": float(depth[idx].mean())})
    score_frame = pd.DataFrame(score_rows)
    all_states = read_table(data_dir / TARGETS["all_states"])
    conventions = read_table(data_dir / TARGETS["conventions"])
    focus_modules = sorted(set(conventions["module"]))
    conventions_frame = score_frame[score_frame["module"].isin(focus_modules)
                                    & (score_frame["state"] != UNLABELLED)].reindex(
        columns=list(conventions.columns))
    ledger.add_table("brain_module_conventions", "state scores under both zero conventions",
                     data_dir / TARGETS["conventions"], conventions_frame,
                     BRAIN_CONVENTIONS_SPEC,
                     evidence="cell score = mean over matched members; state score = mean over "
                              "cells; value keeps zeros, missing drops zero members per cell",
                     notes="the frozen table holds the two curated modules and the eight annotated "
                           "states; the recomputed frame is restricted to exactly that key set, "
                           "and the unlabelled pool is the subject of the next check")
    all_states_frame = score_frame[score_frame["module"].isin(focus_modules)].reindex(
        columns=list(all_states.columns))
    ledger.add_table("brain_module_all_states", "state scores including the unlabelled pool",
                     data_dir / TARGETS["all_states"], all_states_frame, BRAIN_ALL_STATES_SPEC,
                     evidence="same computation with the unlabelled cells kept as a group",
                     notes="the unlabelled cells are the ninth group of this table and are never "
                           "merged into an annotated state")

    marker_frame = read_table(data_dir / TARGETS["markers"])
    z_values = (values - np.nanmean(values, axis=1, keepdims=True)) / np.nanstd(
        values, axis=1, ddof=1, keepdims=True)
    marker_rows = []
    # The module-level detection rate is the mean over matched matrix rows of that row's
    # non-zero fraction, so a member that matches two rows contributes both rows.
    module_row_rate = {}
    module_missing = {}
    for _, member in members.iterrows():
        module = str(member["module"])
        hits = row_index[(module, str(member["gene"]))]
        for i in hits:
            module_row_rate.setdefault(module, []).append(float(zero_presence[i].mean()))
        if not bool(member["matched"]):
            module_missing.setdefault(module, []).append(str(member["gene"]))
    for _, row in marker_frame.iterrows():
        wanted_tokens = set(normalize_gene(t) for t in split_tokens(row["gene"]))
        hits = [i for i, tokens in enumerate(row_tokens) if tokens & wanted_tokens]
        idx = cell_index[str(row["state"])]
        hit = hits[0] if hits else None
        detected = int((values[hit][idx] != 0).sum()) if hit is not None else 0
        marker_rows.append({"gene": row["gene"], "module": row["module"],
                            "module_group": row["module_group"], "state": row["state"],
                            "n_cells": len(idx), "n_detected": detected,
                            "detected_fraction": detected / len(idx),
                            "mean_z_value": float(np.nanmean(z_values[hit][idx]))
                            if hit is not None else np.nan,
                            "mean_value": float(np.nanmean(values[hit][idx]))
                            if hit is not None else np.nan,
                            "matrix_rows_matched": len(hits)})
    recomputed_markers = pd.DataFrame(marker_rows, columns=list(marker_frame.columns))
    ledger.add_table("brain_marker_states", "marker proteins across the eight labelled states",
                     data_dir / TARGETS["markers"], recomputed_markers, BRAIN_MARKERS_SPEC,
                     evidence="per-state detection count and mean of the protein-wise z score "
                              "standardised across all 2,310 cells")

    coverage_rows = []
    for module in sorted(cell_scores):
        hits = sorted(set(module_members[module]))
        per_cell = zero_presence[hits].sum(axis=0) if hits else np.zeros(len(samples))
        for state in groups:
            idx = cell_index[state]
            block = per_cell[idx]
            coverage_rows.append({"module": module,
                                  "cluster": UNLABELLED_KEY if state == UNLABELLED else state,
                                  "n_cells": len(idx),
                                  "mean_members_detected": float(block.mean()),
                                  "min_members_detected": int(block.min()),
                                  "max_members_detected": int(block.max()),
                                  "frac_cells_zero_members": float((block == 0).mean()),
                                  "genes_defined": int((members["module"] == module).sum()),
                                  "genes_matched": int(sum(1 for key in row_index
                                                           if key[0] == module and row_index[key]))})
    coverage_frame = pd.DataFrame(coverage_rows).reindex(
        columns=BRAIN_COVERAGE_SPEC.field_names())
    ledger.add_table("brain_cell_coverage", "member detection coverage per module and group",
                     data_dir / TARGETS["coverage"], coverage_frame, BRAIN_COVERAGE_SPEC,
                     evidence="counts of matched members with a non-zero value in each cell")

    missing_frame = read_table(data_dir / TARGETS["missingness"])
    missing_rows = []
    for module in sorted(missing_frame["module"].unique()):
        for state in groups:
            idx = cell_index[state]
            missing_rows.append({"module": module,
                                 "cluster": UNLABELLED_KEY if state == UNLABELLED else state,
                                 "n_cells": len(idx),
                                 "mean_score_zeros_as_values":
                                     float(np.nanmean(cell_scores[module]["value"][idx])),
                                 "mean_score_zeros_as_missing":
                                     float(np.nanmean(cell_scores[module]["missing"][idx])),
                                 "mean_cluster_depth_nonzero_frac_all_proteins":
                                     float(depth[idx].mean())})
    missingness_frame = pd.DataFrame(missing_rows).reindex(
        columns=BRAIN_MISSINGNESS_SPEC.field_names())
    ledger.add_table("brain_missingness_sensitivity", "score sensitivity to the zero convention",
                     data_dir / TARGETS["missingness"], missingness_frame,
                     BRAIN_MISSINGNESS_SPEC,
                     evidence="same cells scored under both conventions",
                     notes="the two Spearman columns of the frozen table are declared excluded in "
                           "the spec: their writer used scipy.stats.spearmanr, which this script "
                           "does not depend on")

    state_rows = []
    for module in sorted(cell_scores):
        hits = sorted(set(module_members[module]))
        for state in groups:
            idx = cell_index[state]
            score = cell_scores[module]["value"][idx]
            finite = score[np.isfinite(score)]
            state_rows.append({"module": module,
                               "group": UNLABELLED_KEY if state == UNLABELLED else state,
                               "recomputed_n_cells": len(idx),
                               "recomputed_mean": float(finite.mean()) if finite.size else np.nan,
                               "recomputed_median": float(np.median(finite)) if finite.size else np.nan,
                               "n_nonmissing": int(finite.size),
                               "genes_defined": int((members["module"] == module).sum()),
                               "genes_matched": int(sum(1 for key in row_index
                                                        if key[0] == module and row_index[key])),
                               "missing_genes": ";".join(module_missing.get(module, [])) or np.nan,
                               "matrix_rows_matched": len(hits),
                               "mean_member_detection_rate":
                                   float(np.mean(module_row_rate[module]))
                                   if module_row_rate.get(module) else np.nan})
    state_frame = pd.DataFrame(state_rows).reindex(columns=BRAIN_STATE_SCORES_SPEC.field_names())
    ledger.add_table("brain_state_scores", "state-level module scores of all nine groups",
                     data_dir / TARGETS["recomputed"], state_frame, BRAIN_STATE_SCORES_SPEC,
                     row_filter=lambda frozen: ~frozen["group"].str.contains("_minus_"),
                     evidence="module score per cell, then the state mean and median",
                     notes="the frozen file holds two kinds of row: the nine group rows compared "
                           "here and 28 pairwise 'A_minus_B' difference rows, which the declared "
                           "row filter leaves out (a filtered comparison claims no byte identity)",
                     method_extra="; declared row filter: group rows only")
    state_frame.to_csv(out_dir / "brain_module_state_scores_recomputed.tsv", sep="\t", index=False)
    score_frame.to_csv(out_dir / "brain_module_conventions_recomputed.tsv", sep="\t", index=False)
    recomputed_markers.to_csv(out_dir / "brain_marker_states_recomputed.tsv", sep="\t", index=False)

    hashes_after = {rel: sha256_file(data_dir / rel) for rel in wanted.values()}
    changed = [rel for rel in wanted.values() if hashes_before[rel] != hashes_after[rel]]
    ledger.add("brain_inputs_read_only", "archive inputs unchanged by the run", "all inputs",
               "identical SHA-256 before and after",
               "%d files hashed; changed: %s" % (len(wanted), changed or "none"),
               "PASS" if not changed else "FAIL", "exact", "sha256_file on every input")

    payload = {
        "theme": THEME,
        "data_dir": str(args.data_dir),
        "checks": ledger.checks,
        "counts": ledger.counts(),
        "exit_code": ledger.exit_code(),
        "column_classes": ledger.class_rollup(),
        "input_sha256": hashes_after,
        "limitations": [
            "The interpretation of the zeros is unresolved; the value and missing conventions "
            "are reported separately and neither is treated as the correct reading.",
            "The unlabelled cells are carried as their own group and are never merged into a "
            "labelled state.",
            "Module membership is taken from the frozen member table of the archive, not from the "
            "curated module definition file of the historical run.",
            "The 28 pairwise 'A_minus_B' rows of the frozen module table are covered by a "
            "declared row filter and the two descriptive Spearman columns of the missingness "
            "table are declared excluded; neither is recomputed here.",
            "A member gene that matches no matrix row carries an empty detection rate, following "
            "the writer of the frozen member table, instead of a substituted zero.",
        ],
    }
    (out_dir / "case_regression.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(ledger.summary_line())
    return ledger.exit_code()


if __name__ == "__main__":
    sys.exit(main())
