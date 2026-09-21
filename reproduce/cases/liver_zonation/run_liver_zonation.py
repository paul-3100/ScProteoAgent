#!/usr/bin/env python
"""Liver zonation continuation (DVP): deterministic re-derivation from the locked matrix.

The initial sampling-site comparison of this study was a Python two-sided Welch test with
within-contrast Benjamini-Hochberg adjustment. The continuation reported here changes the
experimental unit: the 45 sampling sites are aggregated into 15 mouse-by-zone units, and each
protein's effect is the mean within-mouse difference over the five paired mice, tested with a
two-sided one-sample t-test on those differences. No R/limma step is used in this Python chain.
ComBat is available in the tool library; no new ComBat correction was applied to this batch.

Two branches are computed separately and are never pooled: the *full* branch uses every entry of
the locked matrix, the *observed* branch masks the imputed entries first. Each branch carries two
Benjamini-Hochberg families: within each contrast, and jointly across the three contrasts. The
5,577 rows of the result table are 5,577 rows of a three-contrast family, not 5,577 independent
tests.

Comparison rules (pre-registered): every table comparison is driven by a TableSpec declared at the
top of this script while the run is being prepared, never by the difference being compared. The
key set of the frozen table and of the recomputed frame must be equal (missing, extra and
duplicate keys are reported separately); the missing mask is compared cell by cell, so a value
present on one side and missing on the other fails; inf and -inf fail in every numeric field;
counts, sample sizes, states and flags compare exactly; continuous values use a pre-registered
relative tolerance of 1e-7 with an explicit atol of 0; P and q values compare purely relatively
(atol 0, so 1e-52 is not 0); a rounding allowance exists only where a field declares
format_decimals or rounds_to, and that declaration is checked against the raw text of the frozen
file - no column of this theme needs one, because every frozen numeric column carries full double
precision. The shared comparator (reproduce/cases/table_compare.py) also records, per table,
whether the recomputed frame re-serializes to the frozen bytes.

Usage:
    py run_liver_zonation.py --data-dir <data archive root> --out-dir <output directory>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from table_compare import Field, Ledger, TableSpec  # noqa: E402

THEME = "liver_zonation"
MIN_PAIRS = 3

INPUTS = {
    "matrix": "case_tables/dvp_liver/locked_matrix/ProteinQuant_ComBat.csv",
    "mask": "case_tables/dvp_liver/imputation_mask.csv",
    "mapping": "case_tables/dvp_liver/mouse_zone_mapping.csv",
    "sampleinfo": "case_tables/dvp_liver/SampleInfo.effective.csv",
    "transform": "case_tables/dvp_liver/matrix_transform_record.json",
}
TARGETS = {
    "unit_matrix": "case_tables/dvp_liver/unit_matrix.csv",
    "unit_matrix_observed": "case_tables/dvp_liver/unit_matrix_observed.csv",
    "unit_metadata": "case_tables/dvp_liver/unit_metadata.csv",
    "unit_metadata_observed": "case_tables/dvp_liver/unit_metadata_observed.csv",
    "results": "case_tables/dvp_liver/unit_contrast_results.csv",
    "results_observed": "case_tables/dvp_liver/unit_contrast_results_observed.csv",
    "pairs": "case_tables/dvp_liver/unit_contrast_pairs.csv",
    "pairs_observed": "case_tables/dvp_liver/unit_contrast_pairs_observed.csv",
    "summary": "case_tables/dvp_liver/unit_contrast_summary.json",
    "summary_observed": "case_tables/dvp_liver/unit_contrast_summary_observed.json",
    "schematic": "source_tables/liver_unit_schematic.tsv",
    "weights": "source_tables/liver_design_weights.tsv",
    "candidate_units": "source_tables/liver_candidate_units.tsv",
    "unit_contrasts": "source_tables/liver_unit_contrasts.tsv",
    "dual_bh": "source_tables/liver_dual_bh.tsv",
    "support": "source_tables/liver_support.tsv",
    "entries": "source_tables/full_matrix_entries.tsv",
}
CONTRASTS = (
    ("Midlobular_vs_mean_Portal_Central", "Mid - mean(Portal, Central)",
     {"Midlobular": 1.0, "Portal": -0.5, "Central": -0.5}),
    ("Midlobular_vs_Portal", "Mid - Portal", {"Midlobular": 1.0, "Portal": -1.0}),
    ("Midlobular_vs_Central", "Mid - Central", {"Midlobular": 1.0, "Central": -1.0}),
)
CANDIDATES = (("Q9WU19", "HAO1", "HAO1"), ("Q78JT3", "HAAO", "HAAO"),
              ("P17156", "P17156", "HSP72 (HSPA2) [source name; not displayed]"))


# --------------------------------------------------------------------------- statistics
def _betacf(a: float, b: float, x: float) -> float:
    maxit, eps, fpmin = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    if not np.isfinite(t) or not np.isfinite(df) or df <= 0:
        return float("nan")
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def student_t_quantile(prob: float, df: float) -> float:
    """Upper tail quantile of the t distribution by bisection on the incomplete beta."""
    target = 2.0 * (1.0 - prob)
    lo, hi = 0.0, 1000.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        value = _betai(0.5 * df, 0.5, df / (df + mid * mid))
        if value > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bh_adjust(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = idx.size
    if m:
        order = idx[np.argsort(p[idx], kind="stable")]
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            prev = min(prev, p[i] * m / (rank + 1))
            q[i] = prev
    return q


def one_sample_test(differences: np.ndarray):
    """Rows of per-mouse differences -> (effect, ci_low, ci_high, n_pairs, P, estimability)."""
    n = np.sum(~np.isnan(differences), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        effect = np.nanmean(differences, axis=1)
        sd = np.nanstd(differences, axis=1, ddof=1)
    ci_low = np.full(effect.shape, np.nan)
    ci_high = np.full(effect.shape, np.nan)
    p_value = np.full(effect.shape, np.nan)
    estimability = np.array(["not_estimable_no_unit_pair"] * effect.size, dtype=object)
    for i in range(effect.size):
        if n[i] == 0:
            estimability[i] = "not_estimable_no_unit_pair"
            continue
        if n[i] < MIN_PAIRS:
            estimability[i] = "not_estimable_low_pairs"
            continue
        if not np.isfinite(sd[i]) or sd[i] == 0.0:
            estimability[i] = "degenerate_zero_variance"
            continue
        dfree = float(n[i] - 1)
        se = sd[i] / math.sqrt(n[i])
        t_stat = effect[i] / se
        p_value[i] = student_t_two_sided_p(t_stat, dfree)
        half = student_t_quantile(0.975, dfree) * se
        ci_low[i] = effect[i] - half
        ci_high[i] = effect[i] + half
        estimability[i] = "tested"
    return effect, ci_low, ci_high, n.astype(int), p_value, estimability


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


# --------------------------------------------------------------- comparison schema
# Pre-registered per frozen table, declared before any table is compared (see the shared
# comparator reproduce/cases/table_compare.py). Every frozen column is either compared through a
# Field or named in excluded with the reason why it is not compared. No frozen numeric column of
# this theme needs a rounding allowance: their raw text carries full double precision (the
# fractions of the support and entry tables included), so rtol 1e-7 with atol 0 is declared
# everywhere and no format_decimals or rounds_to appears below.
SPEC_SCHEMATIC = TableSpec(
    key_fields=("mouse", "zone"),
    fields=(
        Field("mouse", "int", source_class="RECOMPUTED", note="mouse number, exact"),
        Field("zone", "identity", source_class="RECOMPUTED", note="liver zone name"),
        Field("unit_column", "identity", source_class="RECOMPUTED",
              note="unit key of the aggregated matrix, mouse|zone"),
        Field("n_samples_pooled", "int", source_class="RECOMPUTED",
              note="sampling sites pooled into the unit, exact"),
        Field("samples", "text", source_class="RECOMPUTED",
              note="semicolon-joined FileName columns of the unit, compared as text in the "
                   "frozen order of the sites"),
    ),
    label="liver_unit_schematic")

SPEC_UNIT_METADATA = TableSpec(
    key_fields=("column",),
    fields=(
        Field("column", "identity", source_class="RECOMPUTED", note="unit key, mouse|zone"),
        Field("unit", "int", source_class="RECOMPUTED", note="mouse number, exact"),
        Field("group", "identity", source_class="RECOMPUTED", note="liver zone name"),
        Field("n_samples", "int", source_class="RECOMPUTED", note="pooled sites, exact"),
        Field("samples", "text", source_class="RECOMPUTED",
              note="semicolon-joined FileName columns of the unit, compared as text"),
    ),
    label="liver_unit_metadata")

SPEC_PAIRS_FULL = TableSpec(
    key_fields=("protein_id", "contrast_id", "unit"),
    fields=(
        Field("protein_id", "identity", source_class="RECOMPUTED",
              note="protein-group key of the locked matrix"),
        Field("contrast_id", "identity", source_class="RECOMPUTED",
              note="one of the three fixed unit contrasts"),
        Field("unit", "int", source_class="RECOMPUTED", note="mouse number, exact"),
        Field("unit_value", "float", source_class="RECOMPUTED",
              note="weighted difference of the aggregated unit values of that mouse; the frozen "
                   "column stores full double precision, rtol 1e-7 with atol 0"),
        Field("groups_used", "text", source_class="RECOMPUTED",
              note="zones entering the contrast, joined in sorted order"),
    ),
    label="liver_pairs_full")

SPEC_PAIRS_OBSERVED = TableSpec(
    key_fields=("protein_id", "contrast_id", "unit"),
    fields=(
        Field("protein_id", "identity", source_class="RECOMPUTED",
              note="protein-group key of the locked matrix"),
        Field("contrast_id", "identity", source_class="RECOMPUTED",
              note="one of the three fixed unit contrasts"),
        Field("unit", "int", source_class="RECOMPUTED", note="mouse number, exact"),
        Field("unit_value", "float", source_class="RECOMPUTED",
              note="same difference after masking the imputed entries; the frozen table lists "
                   "only the finite pairs, so a missing cell cannot occur and none is allowed"),
        Field("groups_used", "text", source_class="RECOMPUTED",
              note="zones entering the contrast, joined in sorted order"),
    ),
    label="liver_pairs_observed")

SPEC_UNIT_CONTRASTS = TableSpec(
    key_fields=("branch", "protein_id", "contrast_id"),
    fields=(
        Field("branch", "identity", source_class="RECOMPUTED",
              note="full or observed matrix branch"),
        Field("protein_id", "identity", source_class="RECOMPUTED",
              note="protein-group key of the locked matrix"),
        Field("protein_display", "identity", source_class="RECOMPUTED",
              note="displayed protein name of the fixed candidate list"),
        Field("gene_source_name", "text", source_class="RECOMPUTED",
              note="gene name of the candidate record, including the source-name note of the "
                   "third candidate"),
        Field("contrast_id", "identity", source_class="RECOMPUTED",
              note="one of the three fixed unit contrasts"),
        Field("contrast_label", "identity", source_class="RECOMPUTED",
              note="printed contrast label"),
        Field("effect", "float", source_class="RECOMPUTED",
              note="mean within-mouse difference over the five paired mice; frozen column stores "
                   "full double precision, rtol 1e-7 with atol 0"),
        Field("ci_low", "float", source_class="RECOMPUTED",
              note="one-sample t interval of the same mean, uncorrected, rtol 1e-7 with atol 0"),
        Field("ci_high", "float", source_class="RECOMPUTED",
              note="one-sample t interval of the same mean, uncorrected, rtol 1e-7 with atol 0"),
        Field("n_pairs", "int", source_class="RECOMPUTED",
              note="paired mice entering the test, exact"),
        Field("P", "pvalue", source_class="RECOMPUTED",
              note="two-sided one-sample t P of the pair differences; purely relative, atol 0"),
        Field("q_within", "pvalue", source_class="RECOMPUTED",
              note="BH step-up inside this contrast; purely relative, atol 0"),
        Field("family_size_within", "int", source_class="RECOMPUTED",
              note="finite P values of that contrast, exact"),
        Field("q_joint", "pvalue", source_class="RECOMPUTED",
              note="BH step-up across the three contrasts of the branch; purely relative, atol 0"),
        Field("family_size_joint", "int", source_class="RECOMPUTED",
              note="finite P values of the whole branch, exact"),
        Field("estimability", "identity", source_class="RECOMPUTED",
              note="tested or the reason the row carries no test"),
        Field("passes_within", "identity", source_class="RECOMPUTED",
              note="q_within < 0.05, stored as the literal text True/False and compared exactly"),
        Field("passes_joint", "identity", source_class="RECOMPUTED",
              note="q_joint < 0.05, stored as the literal text True/False and compared exactly"),
    ),
    label="liver_unit_contrasts")

SPEC_DUAL_BH = TableSpec(
    key_fields=("branch", "protein_id", "contrast_id"),
    fields=(
        Field("branch", "identity", source_class="RECOMPUTED",
              note="full or observed matrix branch"),
        Field("protein_id", "identity", source_class="RECOMPUTED",
              note="protein-group key of the locked matrix"),
        Field("protein_display", "identity", source_class="RECOMPUTED",
              note="displayed protein name of the fixed candidate list"),
        Field("gene_source_name", "text", source_class="RECOMPUTED",
              note="gene name of the candidate record"),
        Field("contrast_id", "identity", source_class="RECOMPUTED",
              note="one of the three fixed unit contrasts"),
        Field("contrast_label", "identity", source_class="RECOMPUTED",
              note="printed contrast label"),
        Field("P", "pvalue", source_class="RECOMPUTED",
              note="two-sided one-sample t P of the pair differences; purely relative, atol 0"),
        Field("q_within", "pvalue", source_class="RECOMPUTED",
              note="BH step-up inside this contrast; purely relative, atol 0"),
        Field("family_size_within", "int", source_class="RECOMPUTED",
              note="finite P values of that contrast, exact"),
        Field("q_joint", "pvalue", source_class="RECOMPUTED",
              note="BH step-up across the three contrasts of the branch; purely relative, atol 0"),
        Field("family_size_joint", "int", source_class="RECOMPUTED",
              note="finite P values of the whole branch, exact"),
        Field("passes_within_q05", "identity", source_class="RECOMPUTED",
              note="q_within < 0.05, stored as the literal text True/False and compared exactly"),
        Field("passes_joint_q05", "identity", source_class="RECOMPUTED",
              note="q_joint < 0.05, stored as the literal text True/False and compared exactly"),
        Field("estimability", "identity", source_class="RECOMPUTED",
              note="tested or the reason the row carries no test"),
    ),
    label="liver_dual_bh")

SPEC_CANDIDATE_UNITS = TableSpec(
    key_fields=("branch", "protein_id", "mouse", "zone"),
    fields=(
        Field("branch", "identity", source_class="RECOMPUTED",
              note="full or observed matrix branch"),
        Field("protein_id", "identity", source_class="RECOMPUTED",
              note="protein-group key of the locked matrix"),
        Field("protein_display", "identity", source_class="RECOMPUTED",
              note="displayed protein name of the fixed candidate list"),
        Field("gene_source_name", "text", source_class="RECOMPUTED",
              note="gene name of the candidate record"),
        Field("mouse", "int", source_class="RECOMPUTED", note="mouse number, exact"),
        Field("zone", "identity", source_class="RECOMPUTED", note="liver zone name"),
        Field("unit_column", "identity", source_class="RECOMPUTED",
              note="unit key of the aggregated matrix, mouse|zone"),
        Field("unit_mean_log2_x_plus_1", "float", source_class="RECOMPUTED",
              note="aggregated unit value; the frozen table carries a finite value in all 90 "
                   "rows, so a missing cell is not allowed and rtol 1e-7 with atol 0 applies"),
        Field("value_present", "identity", source_class="RECOMPUTED",
              note="whether the unit value is finite; stored as the literal text True/False"),
    ),
    label="liver_candidate_units")

SPEC_SUPPORT = TableSpec(
    key_fields=("protein_id", "mouse", "zone"),
    fields=(
        Field("protein_id", "identity", source_class="RECOMPUTED",
              note="protein-group key of the locked matrix"),
        Field("protein_display", "identity", source_class="RECOMPUTED",
              note="displayed protein name of the fixed candidate list"),
        Field("mouse", "int", source_class="RECOMPUTED", note="mouse number, exact"),
        Field("zone", "identity", source_class="RECOMPUTED", note="liver zone name"),
        Field("observed_sites", "int", source_class="RECOMPUTED",
              note="sites with an observed entry, exact"),
        Field("total_sites", "int", source_class="RECOMPUTED",
              note="pooled sites of the unit, exact"),
        Field("observed_fraction", "float", source_class="RECOMPUTED",
              note="observed_sites / total_sites; the frozen column stores full double "
                   "precision, rtol 1e-7 with atol 0"),
        Field("n_imputed", "int", source_class="RECOMPUTED",
              note="sites masked as imputed, exact"),
    ),
    label="liver_support")

SPEC_ENTRIES = TableSpec(
    key_fields=("mouse", "zone"),
    fields=(
        Field("mouse", "int", source_class="RECOMPUTED", note="mouse number, exact"),
        Field("zone", "identity", source_class="RECOMPUTED", note="liver zone name"),
        Field("unit_column", "identity", source_class="RECOMPUTED",
              note="unit key of the aggregated matrix, mouse|zone"),
        Field("n_samples_pooled", "int", source_class="RECOMPUTED",
              note="sites pooled into the unit, exact"),
        Field("detected_proteins", "int", source_class="RECOMPUTED",
              note="the frozen column stores the per-unit observed-entry count (it equals "
                   "observed_entries in all 15 rows), exact"),
        Field("imputed_values", "int", source_class="RECOMPUTED",
              note="masked entries of the unit, exact"),
        Field("detected_fraction", "float", source_class="RECOMPUTED",
              note="observed entries / (1,859 protein rows x pooled sites); the frozen column "
                   "stores full double precision, rtol 1e-7 with atol 0"),
        Field("observed_entries", "int", source_class="RECOMPUTED",
              note="entries with an observed value, exact"),
        Field("imputed_entries", "int", source_class="RECOMPUTED",
              note="masked entries of the unit, exact (the same quantity as imputed_values)"),
        Field("total_entries", "int", source_class="RECOMPUTED",
              note="1,859 protein rows x pooled sites = 5,577, exact"),
    ),
    label="liver_full_matrix_entries")

SPEC_WEIGHTS = TableSpec(
    key_fields=("contrast_id", "zone"),
    fields=(
        Field("contrast_id", "identity", source_class="RECOMPUTED",
              note="one of the three fixed unit contrasts"),
        Field("contrast_label", "identity", source_class="RECOMPUTED",
              note="printed contrast label"),
        Field("zone", "identity", source_class="RECOMPUTED", note="liver zone name"),
        Field("zone_full_name", "identity", source_class="RECOMPUTED",
              note="printed zone name"),
        Field("weight", "float", source_class="RECOMPUTED",
              note="contrast weight of the zone; the values are exact halves and integers, so "
                   "the default rtol 1e-7 with atol 0 reproduces them without any rounding"),
        Field("column_order", "int", source_class="RECOMPUTED",
              note="position of the zone in the weights record, exact"),
        Field("row_sum", "int", source_class="RECOMPUTED",
              note="sum of the weights of the contrast, exact (zero)"),
    ),
    label="liver_design_weights")

SPEC_RESULTS_FULL = TableSpec(
    key_fields=("protein_id", "contrast_id"),
    fields=(
        Field("protein_id", "identity", source_class="RECOMPUTED",
              note="protein-group key of the locked matrix"),
        Field("contrast_id", "identity", source_class="RECOMPUTED",
              note="one of the three fixed unit contrasts"),
        Field("effect", "float", source_class="RECOMPUTED",
              note="mean within-mouse difference over the five paired mice; the frozen column "
                   "stores full double precision, rtol 1e-7 with atol 0, and this branch has no "
                   "missing effect cell"),
        Field("n_pairs", "int", source_class="RECOMPUTED",
              note="paired mice entering the test, exact"),
        Field("P", "pvalue", allows_missing=True, source_class="RECOMPUTED",
              note="two-sided one-sample t P; purely relative with atol 0; the frozen branch "
                   "carries one degenerate row without a P value and the recomputation leaves "
                   "the same cell empty"),
        Field("q_within", "pvalue", allows_missing=True, source_class="RECOMPUTED",
              note="BH step-up inside the contrast; purely relative with atol 0; empty in the "
                   "same degenerate row"),
        Field("q_joint", "pvalue", allows_missing=True, source_class="RECOMPUTED",
              note="BH step-up across the three contrasts; purely relative with atol 0; empty in "
                   "the same degenerate row"),
        Field("ci_low", "float", allows_missing=True, source_class="RECOMPUTED",
              note="t interval of the effect, uncorrected; empty in the same degenerate row"),
        Field("ci_high", "float", allows_missing=True, source_class="RECOMPUTED",
              note="t interval of the effect, uncorrected; empty in the same degenerate row"),
        Field("estimability", "identity", source_class="RECOMPUTED",
              note="tested or the reason the row carries no test"),
    ),
    excluded=(
        ("scale", "run annotation text of the historical run; the frozen file carries a text "
                  "field here and this script does not recompute it"),
        ("test", "name of the test of the historical run (constant paired_t); the test used here "
                 "is stated in the evidence of this check instead of being compared"),
        ("matrix_hash", "digest of the analysed matrix recorded by the historical run; this "
                        "script records the SHA-256 of every archive input in input_sha256 and "
                        "does not recompute the run digest"),
        ("design_hash", "digest of the contrast design recorded by the historical run; not "
                        "recomputed here"),
    ),
    label="liver_unit_contrast_results")

SPEC_RESULTS_OBSERVED = TableSpec(
    key_fields=("protein_id", "contrast_id"),
    fields=(
        Field("protein_id", "identity", source_class="RECOMPUTED",
              note="protein-group key of the locked matrix"),
        Field("contrast_id", "identity", source_class="RECOMPUTED",
              note="one of the three fixed unit contrasts"),
        Field("effect", "float", allows_missing=True, source_class="RECOMPUTED",
              note="mean within-mouse difference after masking the imputed entries; 368 frozen "
                   "rows have no estimable unit pair and are empty on both sides, rtol 1e-7 with "
                   "atol 0 on the values that are present"),
        Field("n_pairs", "int", source_class="RECOMPUTED",
              note="paired mice entering the test, exact (zero or one for the rows without a "
                   "test)"),
        Field("P", "pvalue", allows_missing=True, source_class="RECOMPUTED",
              note="two-sided one-sample t P; purely relative with atol 0; 1,349 frozen rows "
                   "carry no P value and the recomputation leaves the same cells empty"),
        Field("q_within", "pvalue", allows_missing=True, source_class="RECOMPUTED",
              note="BH step-up inside the contrast; purely relative with atol 0; empty in the "
                   "same rows"),
        Field("q_joint", "pvalue", allows_missing=True, source_class="RECOMPUTED",
              note="BH step-up across the three contrasts; purely relative with atol 0; empty in "
                   "the same rows"),
        Field("ci_low", "float", allows_missing=True, source_class="RECOMPUTED",
              note="t interval of the effect, uncorrected; empty in the same rows"),
        Field("ci_high", "float", allows_missing=True, source_class="RECOMPUTED",
              note="t interval of the effect, uncorrected; empty in the same rows"),
        Field("estimability", "identity", source_class="RECOMPUTED",
              note="tested or the reason the row carries no test"),
    ),
    excluded=(
        ("scale", "run annotation text of the historical run; not recomputed"),
        ("test", "name of the test of the historical run (constant paired_t); not compared"),
        ("matrix_hash", "digest of the analysed matrix recorded by the historical run; not "
                        "recomputed here"),
        ("design_hash", "digest of the contrast design recorded by the historical run; not "
                        "recomputed here"),
    ),
    label="liver_unit_contrast_results_observed")


def unit_matrix_spec(label, unit_columns, allows_missing, note):
    """TableSpec of one unit matrix, from the mouse-by-zone column names of the archive."""
    return TableSpec(
        key_fields=("PG.ProteinGroups",),
        fields=(Field("PG.ProteinGroups", "identity", source_class="RECOMPUTED",
                      note="protein-group key of the locked matrix"),)
        + tuple(Field(name, "float", allows_missing=allows_missing, source_class="RECOMPUTED",
                      note=note) for name in unit_columns),
        label=label)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Liver zonation continuation (DVP), re-run.")
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
    mask = read_table(data_dir / INPUTS["mask"])
    mapping = read_table(data_dir / INPUTS["mapping"])
    sampleinfo = read_table(data_dir / INPUTS["sampleinfo"])
    transform = json.loads((data_dir / INPUTS["transform"]).read_text(encoding="utf-8-sig"))
    summary = json.loads((data_dir / TARGETS["summary"]).read_text(encoding="utf-8-sig"))
    summary_observed = json.loads((data_dir / TARGETS["summary_observed"])
                                  .read_text(encoding="utf-8-sig"))

    pg_col = "PG.ProteinGroups"
    samples = [c for c in matrix.columns if c not in (pg_col, "PG.Genes")]
    proteins = [str(p) for p in matrix[pg_col]]
    values = matrix[samples].to_numpy(dtype=float)
    observed = mask[samples].to_numpy(dtype=float) == 1.0     # 1 = observed, 0 = imputed entry
    mouse_of = dict(zip(mapping["FileName"].astype(str), mapping["mouse"].astype(int)))
    zone_of = dict(zip(mapping["FileName"].astype(str), mapping["zone"].astype(str)))

    units = []
    for _, row in sampleinfo.iterrows():
        token = "%d|%s" % (int(row["mouse"]), str(row["zone"]))
        if token not in [u for u, _ in units]:
            units.append((token, []))
    for i, sample in enumerate(samples):
        token = "%d|%s" % (mouse_of[sample], zone_of[sample])
        for unit, members in units:
            if unit == token:
                members.append(i)
    units = [(unit, members) for unit, members in units
             if len(members) > 0]
    units.sort(key=lambda item: (int(item[0].split("|")[0]), item[0].split("|")[1]))
    unit_names = [unit for unit, _ in units]

    n_imputed_recorded = int(transform.get("imputation", {}).get("n_values_imputed", -1))
    n_imputed_masked = int(values.shape[0] * len(samples) - observed.sum())
    ledger.add("liver_matrix_identity", "locked matrix, samples and recorded imputation",
               INPUTS["matrix"], "1,859 protein rows x 45 sampling sites; 28,446 imputed entries",
               "%d x %d; imputation mask counts %d imputed entries" % (matrix.shape[0], len(samples),
                                                                      n_imputed_masked),
               "PASS" if (matrix.shape[0] == 1859 and len(samples) == 45
                          and n_imputed_masked == n_imputed_recorded == 28446) else "FAIL",
               "exact", "locked matrix shape, sample count and imputation_mask.csv (1 = observed)")
    ledger.add("liver_unit_identity", "mouse-by-zone units built from the 45 sites",
               INPUTS["sampleinfo"], "15 units = 5 mice x 3 zones x 3 pooled sites each",
               "%d units; site counts %s" % (len(units), sorted({len(m) for _, m in units})),
               "PASS" if (len(units) == 15 and sorted({len(m) for _, m in units}) == [3]
                          and len({u.split("|")[0] for u in unit_names}) == 5) else "FAIL",
               "exact", "SampleInfo.effective.csv mouse and zone columns")
    schematic = pd.DataFrame({
        "mouse": [int(u.split("|")[0]) for u in unit_names],
        "zone": [u.split("|")[1] for u in unit_names],
        "unit_column": unit_names,
        "n_samples_pooled": [len(m) for _, m in units],
        "samples": [";".join(samples[i] for i in members) for _, members in units]})
    ledger.add_table("liver_unit_schematic", "unit definition against the frozen schematic",
                     data_dir / TARGETS["schematic"], schematic, SPEC_SCHEMATIC,
                     evidence="SampleInfo.effective.csv mouse/zone and the 45 FileName columns")
    unit_metadata = pd.DataFrame({
        "column": unit_names, "unit": [int(u.split("|")[0]) for u in unit_names],
        "group": [u.split("|")[1] for u in unit_names],
        "n_samples": [len(m) for _, m in units],
        "samples": [";".join(samples[i] for i in members) for _, members in units]})
    ledger.add_table("liver_unit_metadata", "unit metadata against the frozen table",
                     data_dir / TARGETS["unit_metadata"], unit_metadata, SPEC_UNIT_METADATA)

    full_units = np.column_stack([values[:, members].mean(axis=1) for _, members in units])
    observed_units = np.full(full_units.shape, np.nan)
    for k, (_, members) in enumerate(units):
        for i in range(values.shape[0]):
            keep = observed[i, members]
            if keep.any():
                observed_units[i, k] = values[i, members][keep].mean()
    unit_matrix = pd.DataFrame(full_units, columns=unit_names)
    unit_matrix.insert(0, pg_col, proteins)
    unit_matrix_observed = pd.DataFrame(observed_units, columns=unit_names)
    unit_matrix_observed.insert(0, pg_col, proteins)
    spec_unit_matrix = unit_matrix_spec(
        "liver_unit_matrix", unit_names, False,
        "mean of the three pooled sites of this mouse-by-zone unit; the frozen column stores "
        "full double precision, rtol 1e-7 with atol 0")
    spec_unit_matrix_observed = unit_matrix_spec(
        "liver_unit_matrix_observed", unit_names, True,
        "mean of the observed entries of this unit; a unit without an observed entry is empty in "
        "the frozen file and in the recomputation, and rtol 1e-7 with atol 0 applies to the "
        "values that are present")
    ledger.add_table("liver_unit_matrix_full", "full-branch unit matrix",
                     data_dir / TARGETS["unit_matrix"], unit_matrix, spec_unit_matrix,
                     evidence="mean of the three pooled sites of each mouse-by-zone unit")
    ledger.add_table("liver_unit_matrix_observed", "observed-branch unit matrix",
                     data_dir / TARGETS["unit_matrix_observed"], unit_matrix_observed,
                     spec_unit_matrix_observed,
                     evidence="mean of the observed entries of each unit; units without an "
                              "observed entry stay missing")

    zone_index = {name: k for k, name in enumerate(unit_names)}
    mice = sorted({int(u.split("|")[0]) for u in unit_names})
    zone_order = ("Portal", "Midlobular", "Central")
    pair_rows = []
    pair_rows_observed = []
    results = {}
    results_observed = {}
    for cid, label, weights in CONTRASTS:
        full_d = np.full((values.shape[0], len(mice)), np.nan)
        obs_d = np.full((values.shape[0], len(mice)), np.nan)
        for mi, mouse in enumerate(mice):
            columns = [zone_index["%d|%s" % (mouse, zone)] for zone in weights]
            w = np.array([weights[zone] for zone in weights], dtype=float)
            full_d[:, mi] = full_units[:, columns] @ w
            obs_d[:, mi] = observed_units[:, columns] @ w
        results[cid] = (full_d, one_sample_test(full_d))
        results_observed[cid] = (obs_d, one_sample_test(obs_d))
        for i in range(values.shape[0]):
            for mi, mouse in enumerate(mice):
                pair_rows.append({"protein_id": proteins[i], "contrast_id": cid, "unit": mouse,
                                  "unit_value": full_d[i, mi],
                                  "groups_used": ";".join(sorted(weights, key=str))})
                if np.isfinite(obs_d[i, mi]):
                    pair_rows_observed.append({"protein_id": proteins[i], "contrast_id": cid,
                                               "unit": mouse, "unit_value": obs_d[i, mi],
                                               "groups_used": ";".join(sorted(weights, key=str))})
    pairs_full = pd.DataFrame(pair_rows)
    pairs_observed = pd.DataFrame(pair_rows_observed)
    ledger.add_table("liver_pairs_full", "full-branch per-mouse paired differences",
                     data_dir / TARGETS["pairs"], pairs_full, SPEC_PAIRS_FULL,
                     evidence="weighted difference of the aggregated unit values per mouse")
    ledger.add_table("liver_pairs_observed", "observed-branch per-mouse paired differences",
                     data_dir / TARGETS["pairs_observed"], pairs_observed, SPEC_PAIRS_OBSERVED,
                     evidence="same aggregation after masking imputed entries; non-finite pairs "
                              "are not listed")

    def result_frame(branch_results, tag):
        rows = []
        for cid, _label, _weights in CONTRASTS:
            for i in range(values.shape[0]):
                _d, (effect, ci_low, ci_high, n, p_value, estimability) = branch_results[cid]
                rows.append({"protein_id": proteins[i], "contrast_id": cid,
                             "effect": effect[i], "n_pairs": int(n[i]), "P": p_value[i],
                             "ci_low": ci_low[i], "ci_high": ci_high[i],
                             "estimability": str(estimability[i])})
        frame = pd.DataFrame(rows)
        family_within = {}
        for cid, _label, _weights in CONTRASTS:
            sub = frame[frame["contrast_id"] == cid]
            family_within[cid] = bh_adjust(sub["P"].to_numpy(dtype=float))
        within = np.full(len(frame), np.nan)
        for cid, _label, _weights in CONTRASTS:
            mask_c = (frame["contrast_id"] == cid).to_numpy()
            within[mask_c] = family_within[cid]
        joint = bh_adjust(frame["P"].to_numpy(dtype=float))
        frame["q_within"] = within
        frame["q_joint"] = joint
        frame["family_size_within"] = [int(np.isfinite(family_within[cid]).sum()) for cid in
                                        frame["contrast_id"]]
        frame["family_size_joint"] = int(np.isfinite(joint).sum())
        print("BRANCH %s rows %d finite P %d joint family %d" % (tag, len(frame),
              int(np.isfinite(frame["P"]).sum()), int(frame["family_size_joint"].iloc[0])))
        return frame

    full_frame = result_frame(results, "full")
    observed_frame = result_frame(results_observed, "observed")
    full_ordered = full_frame[SPEC_RESULTS_FULL.field_names()].copy()
    observed_ordered = observed_frame[SPEC_RESULTS_OBSERVED.field_names()].copy()
    ledger.add_table("liver_results_full", "full-branch contrast results",
                     data_dir / TARGETS["results"], full_ordered, SPEC_RESULTS_FULL,
                     evidence="paired one-sample t on five within-mouse differences; q_within "
                              "inside each contrast, q_joint across the three contrasts")
    ledger.add_table("liver_results_observed", "observed-branch contrast results",
                     data_dir / TARGETS["results_observed"], observed_ordered,
                     SPEC_RESULTS_OBSERVED,
                     evidence="same tests after masking the imputed entries")
    full_frame.to_csv(out_dir / "unit_contrast_results_recomputed.csv", index=False)
    observed_frame.to_csv(out_dir / "unit_contrast_results_observed_recomputed.csv", index=False)
    unit_matrix.to_csv(out_dir / "unit_matrix_recomputed.csv", index=False)
    unit_matrix_observed.to_csv(out_dir / "unit_matrix_observed_recomputed.csv", index=False)

    full_joint = int(np.isfinite(full_frame["q_joint"].to_numpy(dtype=float)).sum())
    obs_joint = int(np.isfinite(observed_frame["q_joint"].to_numpy(dtype=float)).sum())
    full_finite = int(np.isfinite(full_frame["P"].to_numpy(dtype=float)).sum())
    obs_finite = int(np.isfinite(observed_frame["P"].to_numpy(dtype=float)).sum())
    summary_ok = (len(full_frame) == 5577 and full_finite == 5576 and full_joint == 5576
                  and obs_joint == obs_finite == 4228
                  and int(np.isfinite(full_frame["q_within"].to_numpy(dtype=float)).sum()) == 5576
                  and full_joint == int(summary["family_size_joint"])
                  and obs_joint == int(summary_observed["family_size_joint"]))
    ledger.add("liver_correction_families", "row and family sizes of both branches",
               TARGETS["summary"],
               "5,577 full-branch rows with 5,576 finite joint P values; 4,228 observed-branch "
               "values; q within contrasts and across the three-contrast family",
               "full rows %d, finite P %d, joint %d; observed finite P %d, joint %d" % (
                   len(full_frame), full_finite, full_joint, obs_finite, obs_joint),
               "PASS" if summary_ok else "FAIL", "exact",
               "recomputed against unit_contrast_summary.json and _observed.json; the 5,577 rows "
               "are three contrasts of 1,859 proteins, not 5,577 independent tests")

    cand_rows_unit = []
    cand_rows_contrast = []
    cand_rows_dual = []
    for branch, frame in (("full", full_frame), ("observed", observed_frame)):
        units_branch = full_units if branch == "full" else observed_units
        for pid, display, gene in CANDIDATES:
            i = proteins.index(pid)
            for mouse in mice:
                for zone in zone_order:
                    k = zone_index["%d|%s" % (mouse, zone)]
                    cand_rows_unit.append({"branch": branch, "protein_id": pid,
                                           "protein_display": display, "gene_source_name": gene,
                                           "mouse": mouse, "zone": zone,
                                           "unit_column": "%d|%s" % (mouse, zone),
                                           "unit_mean_log2_x_plus_1": units_branch[i, k],
                                           "value_present": bool(np.isfinite(units_branch[i, k]))})
            for cid, label, _weights in CONTRASTS:
                sub = frame[(frame["protein_id"] == pid) & (frame["contrast_id"] == cid)].iloc[0]
                common = {"branch": branch, "protein_id": pid, "protein_display": display,
                          "gene_source_name": gene, "contrast_id": cid, "contrast_label": label}
                cand_rows_contrast.append({**common, "effect": sub["effect"],
                                           "ci_low": sub["ci_low"], "ci_high": sub["ci_high"],
                                           "n_pairs": int(sub["n_pairs"]), "P": sub["P"],
                                           "q_within": sub["q_within"],
                                           "family_size_within": int(sub["family_size_within"]),
                                           "q_joint": sub["q_joint"],
                                           "family_size_joint": int(sub["family_size_joint"]),
                                           "estimability": sub["estimability"],
                                           "passes_within": bool(np.isfinite(sub["q_within"])
                                                                  and sub["q_within"] < 0.05),
                                           "passes_joint": bool(np.isfinite(sub["q_joint"])
                                                                 and sub["q_joint"] < 0.05)})
                cand_rows_dual.append({**common, "P": sub["P"], "q_within": sub["q_within"],
                                       "family_size_within": int(sub["family_size_within"]),
                                       "q_joint": sub["q_joint"],
                                       "family_size_joint": int(sub["family_size_joint"]),
                                       "passes_within_q05": bool(np.isfinite(sub["q_within"])
                                                                  and sub["q_within"] < 0.05),
                                       "passes_joint_q05": bool(np.isfinite(sub["q_joint"])
                                                                 and sub["q_joint"] < 0.05),
                                       "estimability": sub["estimability"]})
    recomputed_contrasts = pd.DataFrame(cand_rows_contrast)[SPEC_UNIT_CONTRASTS.field_names()]
    recomputed_dual = pd.DataFrame(cand_rows_dual)[SPEC_DUAL_BH.field_names()]
    ledger.add_table("liver_unit_contrast_table", "candidate unit contrasts (both branches)",
                     data_dir / TARGETS["unit_contrasts"], recomputed_contrasts,
                     SPEC_UNIT_CONTRASTS,
                     evidence="18 estimates = 3 candidates x 3 contrasts x 2 branches; intervals "
                              "uncorrected, q values in their own families")
    ledger.add_table("liver_dual_bh_table", "dual Benjamini-Hochberg table",
                     data_dir / TARGETS["dual_bh"], recomputed_dual, SPEC_DUAL_BH,
                     evidence="q_within and q_joint recomputed from the same P values")
    recomputed_candidates = pd.DataFrame(cand_rows_unit)[SPEC_CANDIDATE_UNITS.field_names()]
    ledger.add_table("liver_candidate_units", "per-unit values of the three candidates",
                     data_dir / TARGETS["candidate_units"], recomputed_candidates,
                     SPEC_CANDIDATE_UNITS,
                     evidence="aggregated unit values of both branches")

    support_rows = []
    entry_rows = []
    distinct_detected = []
    for mouse in mice:
        for zone in zone_order:
            unit = "%d|%s" % (mouse, zone)
            members = units[zone_index[unit]][1]
            observed_entries = int(observed[:, members].sum())
            imputed_entries = int(observed[:, members].size - observed_entries)
            distinct_detected.append(int(np.any(observed[:, members], axis=1).sum()))
            entry_rows.append({"mouse": mouse, "zone": zone, "unit_column": unit,
                               "n_samples_pooled": len(members),
                               "detected_proteins": observed_entries,
                               "imputed_values": imputed_entries,
                               "detected_fraction": observed_entries / (values.shape[0] * len(members)),
                               "observed_entries": observed_entries,
                               "imputed_entries": imputed_entries,
                               "total_entries": int(observed[:, members].size)})
    for mouse in mice:
        for pid, display, _gene in CANDIDATES:
            i = proteins.index(pid)
            for zone in zone_order:
                unit = "%d|%s" % (mouse, zone)
                members = units[zone_index[unit]][1]
                n_obs = int(observed[i, members].sum())
                support_rows.append({"protein_id": pid, "protein_display": display,
                                     "mouse": mouse, "zone": zone, "observed_sites": n_obs,
                                     "total_sites": len(members),
                                     "observed_fraction": n_obs / len(members),
                                     "n_imputed": len(members) - n_obs})
    recomputed_support = pd.DataFrame(support_rows)[SPEC_SUPPORT.field_names()]
    ledger.add_table("liver_support", "per-candidate observation support per unit",
                     data_dir / TARGETS["support"], recomputed_support, SPEC_SUPPORT,
                     evidence="counts of observed entries in imputation_mask.csv")
    recomputed_entries = pd.DataFrame(entry_rows)[SPEC_ENTRIES.field_names()]
    ledger.add_table("liver_full_matrix_entries", "per-unit matrix entry counts",
                     data_dir / TARGETS["entries"], recomputed_entries, SPEC_ENTRIES,
                     evidence="counts of observed and imputed entries per unit; total = 1,859 x 3",
                     notes="the frozen column detected_proteins stores the per-unit observed-entry "
                           "count (it equals observed_entries in all 15 rows and exceeds the 1,859 "
                           "protein rows); the number of distinct proteins with at least one "
                           "observed entry in a unit ranges from %d to %d and is not that column" % (
                               min(distinct_detected), max(distinct_detected)))

    weight_rows = []
    order = {zone: k for k, zone in enumerate(zone_order)}
    for cid, label, weights in CONTRASTS:
        for zone in zone_order:
            weight_rows.append({"contrast_id": cid, "contrast_label": label, "zone": zone,
                                "zone_full_name": {"Portal": "Portal (periportal)",
                                                   "Midlobular": "Midlobular",
                                                   "Central": "Central (pericentral)"}[zone],
                                "weight": float(weights.get(zone, 0.0)),
                                "column_order": order[zone] + 1,
                                "row_sum": 0})
    recomputed_weights = pd.DataFrame(weight_rows)[SPEC_WEIGHTS.field_names()]
    ledger.add_table("liver_design_weights", "contrast weights of the three comparisons",
                     data_dir / TARGETS["weights"], recomputed_weights, SPEC_WEIGHTS,
                     evidence="Mid - mean(Portal, Central) uses 1 / -0.5 / -0.5; row sums are zero")

    hashes_after = {rel: sha256_file(data_dir / rel) for rel in wanted.values()}
    changed = [rel for rel in wanted.values() if hashes_before[rel] != hashes_after[rel]]
    ledger.add("liver_inputs_read_only", "archive inputs unchanged by the run", "all inputs",
               "identical SHA-256 before and after",
               "%d files hashed; changed: %s" % (len(wanted), changed or "none"),
               "PASS" if not changed else "FAIL", "exact", "sha256_file on every input")

    counts_summary = ledger.counts()
    payload = {
        "theme": THEME,
        "data_dir": str(args.data_dir),
        "checks": ledger.checks,
        "counts": counts_summary,
        "exit_code": ledger.exit_code(),
        "column_classes": ledger.class_rollup(),
        "input_sha256": hashes_after,
        "limitations": [
            "The 5,577 rows of the full-branch result table are three contrasts of 1,859 "
            "proteins, not 5,577 independent tests; the joint family holds 5,576 finite P values "
            "and the observed-only family 4,228.",
            "Only the continuation is re-derived here. The initial sampling-site Welch tests of "
            "the same study belong to the historical run and are not repeated in this chain; "
            "R/limma remains a tool-library backend that this Python chain does not use.",
            "ComBat is available in the tool library, but this batch applies no new ComBat "
            "correction: the locked matrix ProteinQuant_ComBat.csv is the analysed input.",
            "Every table comparison is driven by a TableSpec declared in this script before the "
            "run: key sets equal, the missing mask compared cell by cell, inf rejected, counts "
            "exact, continuous values at the pre-registered rtol 1e-7 with atol 0, P and q "
            "purely relative, and a rounding allowance only where a field declares "
            "format_decimals or rounds_to. No column of this theme declares one, because every "
            "frozen numeric column carries full double precision; byte-level re-serialization is "
            "recorded per table, not required.",
        ],
    }
    (out_dir / "case_regression.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(ledger.summary_line())
    return ledger.exit_code()


if __name__ == "__main__":
    sys.exit(main())
