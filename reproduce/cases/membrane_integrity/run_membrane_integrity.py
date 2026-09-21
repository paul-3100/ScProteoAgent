#!/usr/bin/env python
"""Airway preservation / membrane-integrity case: deterministic re-derivation.

The delivered airway matrix already carries negative log ratios: it receives no further
transform, no imputation and no batch correction in this chain. The comparison of interest is
Intact minus Permeable, reported both as a pooled mean difference and as a composition-
standardised effect. Standardisation uses the intact-cell composition of the strata with common
support (>= 5 cells in each state); both numbers are descriptive and carry no interval.

Usage:
    py run_membrane_integrity.py --data-dir <data archive root> --out-dir <output directory>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from table_compare import Field, Ledger, TableSpec  # noqa: E402

THEME = "membrane_integrity"
MIN_CELLS = 5
GENES = (("Q03265", "ATP5F1A"), ("P16858", "GAPDH"), ("P06151", "LDHA"), ("Q60932", "VDAC1"))

INPUTS = {
    "matrix": "case_tables/airway_proteinleakage/run_processed/ProQuant_Normalized.csv",
    "sampleinfo": "case_tables/airway_proteinleakage/run_processed/SampleInfo_Filtered.csv",
    "gene_map": "case_tables/airway_proteinleakage/run_processed/Protein_Gene_Map.csv",
    "raw_matrix": "inputs/Nat_Commun_ProteinLeakage_2025/ProteinQuant.csv",
    "matrix_chain": "case_tables/airway_proteinleakage/airway_matrix_chain.tsv",
}
TARGETS = {
    "composition": "source_tables/airway_composition.tsv",
    "saved_composition": "source_tables/airway_saved_composition.tsv",
    "effects": "source_tables/airway_effects.tsv",
    "strata_sizes": "case_tables/airway_proteinleakage/e06_leakage_strata_sizes.tsv",
    "mask_counts": "case_tables/airway_proteinleakage/airway_candidate_mask_counts.tsv",
}
MISSING_TRANSFORM_RECORD = "case_tables/airway_proteinleakage/run_processed/matrix_transform_record.json"
MATRIX_CHAIN = INPUTS["matrix_chain"]


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


def matrix_chain_facts(path: Path) -> dict:
    """The executed-chain statements of the frozen matrix chain table, keyed by step number."""
    facts = {}
    for _, row in read_table(path).iterrows():
        facts[str(row["step_no"]).strip()] = (str(row["process"]).strip(),
                                              str(row["value"]).strip(),
                                              str(row["evidence_locator"]).strip())
    return facts


# ------------------------------------------------------------------- pre-registered specs
# One TableSpec per compared frozen table. Every frozen column is either a Field (compared, with
# its kind and any declared rounding) or is declared excluded with a reason. The declared
# resolutions come from the writer's own code:
#   * source_tables/airway_composition.tsv and airway_saved_composition.tsv are verbatim copies of
#     the V14 figure-round panel source tables (V15_DATA_FIGURE_REFINEMENT_20260918/scripts/
#     prepare_data_b.py, copy_theme); V14_FIGURE_POLISH_20260918/scripts/panels_f5.py
#     ::leakage_composition writes the counts as integers and pct_intact as a plain float.
#   * source_tables/airway_effects.tsv is the same kind of copy of a table that
#     panels_f5.py::leakage_effects builds from the four-decimal E06 leakage table written by
#     V9_EXPLORATION_SUPPORT/scripts/e06_leakage_stratified.py.
#   * case_tables/airway_proteinleakage/e06_leakage_strata_sizes.tsv and
#     airway_candidate_mask_counts.tsv are frozen E06 tables with integer counts.

IDENTITY_NOTE = "row key or label; compared as identity"

_STRATA_FIELDS = (
    Field("preservation", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
    Field("cell_type", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
    Field("n_intact", "int", note="cells of the stratum in the Intact state"),
    Field("n_permeable", "int", note="cells of the stratum in the Permeable state"),
    Field("total_cells", "int", note="n_intact + n_permeable"),
    Field("pct_intact", "float",
          note="100 * n_intact / total_cells; V14_FIGURE_POLISH_20260918/scripts/panels_f5.py"
               "::leakage_composition stores the plain float with no declared format, so the "
               "pre-registered 1e-7 relative tolerance applies"),
    Field("comparable", "text",
          note="frozen raw text is True/False, so the boolean is compared as text; the numeric "
               "flag kind would report these tokens as unparsable"),
)

AIRWAY_COMPOSITION_SPEC = TableSpec(
    key_fields=("preservation", "cell_type"),
    fields=_STRATA_FIELDS,
    label="preservation-by-cell-type strata of the analysed cells",
)

AIRWAY_SAVED_COMPOSITION_SPEC = TableSpec(
    key_fields=("preservation", "cell_type"),
    fields=_STRATA_FIELDS,
    label="saved composition view of the same cells",
)

AIRWAY_STRATA_SIZES_SPEC = TableSpec(
    key_fields=("preservation", "cell_type"),
    fields=(
        Field("preservation", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("cell_type", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("n_Intact", "int", note="cells of the stratum in the Intact state"),
        Field("n_Permeable", "int", note="cells of the stratum in the Permeable state"),
        Field("comparable", "text",
              note="at least five cells in each membrane state; frozen raw text is True/False"),
    ),
    label="stratum sizes recorded for the panel",
)

AIRWAY_EFFECTS_SPEC = TableSpec(
    key_fields=("gene", "protein"),
    fields=(
        Field("gene", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("protein", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("pooled_effect", "float", format_decimals=4,
              note="the value is produced by V9_EXPLORATION_SUPPORT/scripts/"
                   "e06_leakage_stratified.py L95 (round(pooled, 4)) and copied through the "
                   "figure table; the frozen text stores at most four decimals"),
        Field("standardized_effect", "float", format_decimals=4,
              note="same chain, e06_leakage_stratified.py L96 (round(adj, 4)); the frozen text "
                   "stores at most four decimals"),
        Field("ratio_abs_standardized_over_pooled", "float",
              note="V14_FIGURE_POLISH_20260918/scripts/panels_f5.py::leakage_effects L402 forms "
                   "abs(std)/abs(pooled) from the four-decimal stored effect values; the "
                   "recomputation forms the same ratio from the same stored values, so no "
                   "rounding allowance is declared and rtol 1e-7 applies"),
        Field("n_strata_used", "int", note="strata with common support"),
        Field("n_strata_excluded", "int", note="strata without common support"),
        Field("weight_coverage", "float", format_decimals=3,
              note="e06_leakage_stratified.py L98 writes round(weights, 3); the frozen text shows "
                   "0.82 because str() drops the trailing zero, so three decimals are declared - "
                   "the writer's precision, never a looser one"),
        Field("n_finite_intact", "int", note="finite entries of the protein in Intact cells"),
        Field("n_finite_permeable", "int",
              note="finite entries of the protein in Permeable cells"),
    ),
    label="pooled and composition-standardised effects of the four candidates",
)

AIRWAY_MASK_COUNTS_SPEC = TableSpec(
    key_fields=("candidate", "membrane_state"),
    fields=(
        Field("candidate", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("gene", "identity", source_class="IDENTITY_CHECK",
              note="gene symbol of the candidate, carried by the frozen table"),
        Field("membrane_state", "identity", source_class="IDENTITY_CHECK", note=IDENTITY_NOTE),
        Field("n_total", "int", note="cells of that membrane state among the 2,295 analysed cells"),
        Field("n_original_observed", "int",
              note="cells whose value is present in the delivered input matrix"),
        Field("n_imputed", "int",
              note="cells absent from the input and present in the analysed matrix"),
        Field("n_unresolved", "int", note="cells absent from the analysed matrix"),
    ),
    excluded=(
        ("evidence_path",
         "prose column: the archive-relative paths the writer recorded, regenerated from the "
         "archive layout rather than recomputed"),
        ("evidence_locator",
         "prose column: the join description the writer recorded"),
        ("note",
         "prose column: the writer's own definition of the counts, not a value"),
    ),
    label="observation support of the four candidates",
)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Airway membrane-integrity case, re-run.")
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
    gene_map = read_table(data_dir / INPUTS["gene_map"])
    raw = read_table(data_dir / INPUTS["raw_matrix"])

    samples = [str(s) for s in sampleinfo["FileName"]]
    pg_col = "PG.ProteinGroups"
    proteins = [str(p) for p in matrix[pg_col]]
    values = matrix[samples].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    status = sampleinfo["Type1"].astype(str).to_numpy()
    preservation = sampleinfo["Type2"].astype(str).to_numpy()
    celltype = sampleinfo["Cluster"].astype(str).to_numpy()
    intact = status == "Intact"
    permeable = status == "Permeable"
    state_counts = {s: int((status == s).sum()) for s in sorted(set(status))}

    identity_ok = bool(len(samples) == 2295 and state_counts.get("Intact") == 1838
                       and state_counts.get("Permeable") == 457 and not matrix[samples].isna().any().any())
    ledger.add("airway_cell_identity", "delivered airway cells and membrane states",
               INPUTS["sampleinfo"],
               "2,295 cells: 1,838 Intact and 457 Permeable, no missing entries in the analysed matrix",
               "%d cells: %s; matrix missing entries %d" % (
                   len(samples), state_counts, int(matrix[samples].isna().sum().sum())),
               "PASS" if identity_ok else "FAIL", "exact",
               "SampleInfo_Filtered.csv Type1 (membrane state) and Type2 (preservation) columns")

    strata = sorted(set(zip(preservation, celltype)))
    composition_rows = []
    for preservation_state, cell_type in strata:
        mask = (preservation == preservation_state) & (celltype == cell_type)
        n_intact = int((mask & intact).sum())
        n_permeable = int((mask & permeable).sum())
        total = n_intact + n_permeable
        composition_rows.append({"preservation": preservation_state, "cell_type": cell_type,
                                 "n_intact": n_intact, "n_permeable": n_permeable,
                                 "total_cells": total,
                                 "pct_intact": n_intact / total * 100.0 if total else np.nan,
                                 "comparable": bool(n_intact >= MIN_CELLS
                                                    and n_permeable >= MIN_CELLS)})
    composition = pd.DataFrame(composition_rows)
    composition_frame = composition.reindex(columns=AIRWAY_COMPOSITION_SPEC.field_names())
    ledger.add_table("airway_composition", "twelve preservation-by-cell-type strata",
                     data_dir / TARGETS["composition"], composition_frame,
                     AIRWAY_COMPOSITION_SPEC,
                     evidence="cell counts of the 2,295 analysed cells; comparable = at least five "
                              "cells in each membrane state")
    ledger.add_table("airway_saved_composition", "saved composition view of the same cells",
                     data_dir / TARGETS["saved_composition"], composition_frame,
                     AIRWAY_SAVED_COMPOSITION_SPEC,
                     evidence="identical counts to the composition table; the saved view is kept "
                              "as its own frozen identity",
                     notes="the same recomputed frame is compared with the second frozen "
                           "presentation of the twelve strata")
    strata_frame = composition_frame.rename(columns={"n_intact": "n_Intact",
                                                     "n_permeable": "n_Permeable"})
    strata_frame = strata_frame[["preservation", "cell_type", "n_Intact", "n_Permeable",
                                 "comparable"]].reindex(
        columns=AIRWAY_STRATA_SIZES_SPEC.field_names())
    ledger.add_table("airway_strata_sizes", "stratum sizes recorded for the panel",
                     data_dir / TARGETS["strata_sizes"], strata_frame,
                     AIRWAY_STRATA_SIZES_SPEC,
                     evidence="same strata, without the percent column")

    common = composition["comparable"].to_numpy()
    included = composition[common]
    weight_total = float(included["n_intact"].sum())
    pooled_rows = []
    for accession, gene in GENES:
        if accession not in proteins:
            pooled_rows.append({"gene": gene, "protein": accession, "pooled_effect": np.nan,
                                "standardized_effect": np.nan,
                                "ratio_abs_standardized_over_pooled": np.nan,
                                "n_strata_used": 0, "n_strata_excluded": int(len(composition)),
                                "weight_coverage": 0.0,
                                "n_finite_intact": int(np.isfinite(values[0][intact]).sum()),
                                "n_finite_permeable": int(np.isfinite(values[0][permeable]).sum())})
            continue
        row = values[proteins.index(accession)]
        pooled = float(np.nanmean(row[intact]) - np.nanmean(row[permeable]))
        weighted, used = 0.0, 0
        for _, stratum in included.iterrows():
            mask = ((preservation == stratum["preservation"])
                    & (celltype == stratum["cell_type"]))
            effect = float(np.nanmean(row[mask & intact]) - np.nanmean(row[mask & permeable]))
            weighted += (stratum["n_intact"] / weight_total) * effect
            used += 1
        coverage = weight_total / float(composition["n_intact"].sum())
        pooled_stored = round(pooled, 4)
        standardised_stored = round(weighted, 4)
        pooled_rows.append({"gene": gene, "protein": accession, "pooled_effect": pooled,
                            "standardized_effect": weighted,
                            "ratio_abs_standardized_over_pooled": (abs(standardised_stored)
                                                                  / abs(pooled_stored)),
                            "n_strata_used": used,
                            "n_strata_excluded": int(len(composition) - used),
                            "weight_coverage": round(coverage, 6),
                            "n_finite_intact": int(np.isfinite(row[intact]).sum()),
                            "n_finite_permeable": int(np.isfinite(row[permeable]).sum())})
    effects = pd.DataFrame(pooled_rows)
    effects_frame = effects.reindex(columns=AIRWAY_EFFECTS_SPEC.field_names())
    ledger.add_table("airway_effects", "pooled and composition-standardised effects",
                     data_dir / TARGETS["effects"], effects_frame, AIRWAY_EFFECTS_SPEC,
                     evidence="intact minus permeable on the delivered scale; standardisation "
                              "weights are the intact composition of the common-support strata",
                     notes="the frozen pooled and standardised effects are stored with four "
                           "decimals (round(..., 4) in e06_leakage_stratified.py) and the frozen "
                           "ratio is the ratio of those stored four-decimal values, which the "
                           "recomputation forms the same way; the counts are stored exactly")
    effects_frame.to_csv(out_dir / "airway_effects_recomputed.tsv", sep="\t", index=False)
    composition_frame.to_csv(out_dir / "airway_composition_recomputed.tsv", sep="\t", index=False)

    raw_samples = [c for c in raw.columns if c not in (pg_col, "PG.Genes")]
    gene_col = "PG.Genes" if "PG.Genes" in raw.columns else "PG.Genes"
    raw_indexed = raw.set_index(pg_col)
    observed_in_input = raw_indexed.reindex(proteins)[samples].notna().to_numpy()
    analysed_present = np.isfinite(values)
    mask_rows = []
    for accession, gene in GENES:
        i = proteins.index(accession) if accession in proteins else None
        for state, state_mask in (("Intact", intact), ("Permeable", permeable)):
            n_total = int(state_mask.sum())
            if i is None:
                mask_rows.append({"candidate": accession, "gene": gene, "membrane_state": state,
                                  "n_total": n_total, "n_original_observed": 0, "n_imputed": 0,
                                  "n_unresolved": n_total})
                continue
            observed = int(observed_in_input[i][state_mask].sum())
            present = int(analysed_present[i][state_mask].sum())
            imputed = int(present - observed)
            mask_rows.append({"candidate": accession, "gene": gene, "membrane_state": state,
                              "n_total": n_total, "n_original_observed": observed,
                              "n_imputed": imputed, "n_unresolved": n_total - present})
    recomputed_mask = pd.DataFrame(mask_rows)
    mask_columns = ["candidate", "gene", "membrane_state", "n_total", "n_original_observed",
                    "n_imputed", "n_unresolved"]
    mask_frame = recomputed_mask[mask_columns].reindex(
        columns=AIRWAY_MASK_COUNTS_SPEC.field_names())
    ledger.add_table("airway_candidate_mask_counts", "observation support of the four candidates",
                     data_dir / TARGETS["mask_counts"], mask_frame, AIRWAY_MASK_COUNTS_SPEC,
                     evidence="identity join on the sample names between the delivered input "
                              "matrix and the analysed matrix; no value-equality rule is used",
                     notes="the prose columns evidence_path, evidence_locator and note of the "
                           "frozen table are regenerated and declared excluded in the spec")

    aligned_input = raw_indexed.reindex(proteins)[samples].to_numpy(dtype=float)
    finite_both = np.isfinite(aligned_input) & np.isfinite(values)
    max_abs_difference = float(np.max(np.abs(aligned_input[finite_both] - values[finite_both])))
    withheld = int((~np.isfinite(aligned_input)).sum())
    value_identity_ok = bool(max_abs_difference == 0.0 and int(matrix.shape[0]) == 1223)
    ledger.add("airway_value_identity", "analysed matrix equals the delivered input values",
               INPUTS["raw_matrix"],
               "retained rows keep the input values; no imputation, no further log transform",
               "1,223 proteins x 2,295 cells: max absolute difference %.3g; input entries "
               "missing before filtering %d; analysed cells %d" % (
                   max_abs_difference, withheld, int(np.isfinite(values).sum())),
               "PASS" if value_identity_ok else "FAIL",
               "exact", "value-by-value comparison after joining on the protein-group key")

    # The run-level matrix_transform_record.json of this study is not part of the historical run
    # directory. The archive ships the executed statements of the same chain instead, in
    # case_tables/airway_proteinleakage/airway_matrix_chain.tsv, so the statements are read from
    # there and are reported separately from the absent log.
    chain = matrix_chain_facts(data_dir / INPUTS["matrix_chain"])

    def chain_step(number: str) -> tuple:
        return chain.get(number, ("", "", ""))

    chain_order_ok = "L10698 -> L10711 -> L10712-L10716 -> L10741" in chain_step("10")[2]
    chain_impute_ok = "0 entries filled" in chain_step("7")[1]
    chain_written_ok = ("all sha256" in chain_step("11")[1]
                        and "1223 x 2297" in chain_step("11")[1])
    chain_identity_ok = "2,806,785 compared cells; 0 unequal" in chain_step("12")[1]
    chain_ok = bool(chain_order_ok and chain_impute_ok and chain_written_ok and chain_identity_ok)
    ledger.add("airway_transform_record", "run transform record of this study",
               MATRIX_CHAIN,
               "filter -> impute -> log2 decision -> write; 0 entries imputed; three written "
               "matrices with one sha256; 0 of 2,806,785 compared values unequal",
               "chain rows 7/10/11/12 of %s: imputation %s; executed order %s; written matrices "
               "%s; value identity %s; the run-level %s is absent from the historical run "
               "directory" % (MATRIX_CHAIN,
                              "0 entries filled" if chain_impute_ok else "not stated",
                              "tools.py L10698 -> L10711 -> L10712-L10716 -> L10741"
                              if chain_order_ok else "not stated",
                              "three matrices, one sha256, 1223 x 2297" if chain_written_ok
                              else "not stated",
                              "2,806,785 compared cells, 0 unequal" if chain_identity_ok
                              else "not stated",
                              MISSING_TRANSFORM_RECORD),
               "PASS" if (chain_ok and value_identity_ok) else "FAIL",
               "established from the executed statements recorded in the frozen chain table plus "
               "the value-identity comparison of the analysed matrix with the delivered input",
               "data/case_tables/airway_proteinleakage/airway_matrix_chain.tsv rows with step_no "
               "7 (imputation effect), 10 (executed order), 11 (matrices written) and 12 (value "
               "identity); the statements are located at tools.py L10698 -> L10711 -> "
               "L10712-L10716 -> L10741",
               notes="the run-level matrix_transform_record.json is absent from the historical run "
                     "directory, so the absent log is registered separately from the numbers: the "
                     "statements rest on the executed code path and on the value identity "
                     "(0 unequal of 2,806,785 compared values), not on that JSON")

    hashes_after = {rel: sha256_file(data_dir / rel) for rel in wanted.values()}
    changed = [rel for rel in wanted.values() if hashes_before[rel] != hashes_after[rel]]
    ledger.add("airway_inputs_read_only", "archive inputs unchanged by the run", "all inputs",
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
            "The delivered matrix is already a normalised, negative-capable quantity matrix: it "
            "receives no further transform, no imputation and no batch correction here.",
            "Permeable is a label of the source study, not a treatment applied in this chain.",
            "The pooled and composition-standardised effects are descriptive and carry no "
            "interval; only the common-support strata enter the standardisation.",
            "The run-level transform record of the historical run is absent from the run directory "
            "and therefore from this archive; the executed statements of the same chain and the "
            "value identity of the analysed matrix are compared instead, and the absent log is "
            "reported separately from the numbers.",
            "The three prose columns of the candidate mask table (evidence_path, evidence_locator "
            "and note) are declared excluded: they describe the writer's own definitions and are "
            "not values.",
        ],
    }
    (out_dir / "case_regression.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(ledger.summary_line())
    return ledger.exit_code()


if __name__ == "__main__":
    sys.exit(main())
