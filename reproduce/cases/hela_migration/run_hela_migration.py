#!/usr/bin/env python
"""HeLa migration case (PiSPA): deterministic re-derivation from the archived matrices.

The analyses of this case are computed separately and never merged into one estimate:

  1. the processed matrix and its recorded processing state,
  2. cluster composition and the migration odds ratio, as a reported transcription and as the
     computed object of the stable odds-ratio table,
  3. the principal-component coordinates that the panel plots,
  4. the six fixed candidate proteins across the three cluster contrasts,
  5. the offline migrated-versus-control comparison of the saved matrix.

Every input is read from the data archive passed as ``--data-dir``; nothing is read from any
other directory and nothing is written outside ``--out-dir``.

Method (manuscript v24_2, Methods): the processed matrix is the half-minimum-imputed,
log2(x+1) matrix without batch correction; cluster contrasts and the migration-label comparison
use two-sided Welch t-tests with Benjamini-Hochberg adjustment inside one 4,688-protein family
per contrast; the principal components are computed on per-protein z scores across the 89 cells.
No test, threshold, candidate or direction in this script was chosen after seeing a result.

Comparison rules (pre-registered): every table comparison is driven by a TableSpec declared at the
top of this script while the run is being prepared, never by the difference being compared. The
key set of the frozen table and of the recomputed frame must be equal (missing, extra and
duplicate keys are reported separately); the missing mask is compared cell by cell, so a value
present on one side and missing on the other fails; inf and -inf fail in every numeric field;
counts, sample sizes, states and flags compare exactly; continuous values use a pre-registered
relative tolerance of 1e-7 with an explicit atol of 0; P and q values compare purely relatively
(atol 0, so 1e-52 is not 0); a rounding allowance exists only where a field declares
format_decimals, and that declaration is checked against the raw text of the frozen file. The
shared comparator (reproduce/cases/table_compare.py) also records, per table, whether the
recomputed frame re-serializes to the frozen bytes.

Usage:
    py run_hela_migration.py --data-dir <data archive root> --out-dir <output directory>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from table_compare import Field, Ledger, TableSpec, read_frozen  # noqa: E402

THEME = "hela_migration"
REL_TOL = 1e-7          # pre-registered relative tolerance of the inline (non-table) checks

INPUTS = {
    "matrix": "case_tables/pispa_hela/run_processed/ProQuant_Normalized.csv",
    "sampleinfo": "case_tables/pispa_hela/run_processed/SampleInfo_Filtered.csv",
    "transform": "case_tables/pispa_hela/run_processed/matrix_transform_record.json",
    "raw_matrix": "inputs/Nat_Commun_PiSPA_2024/ProteinQuant.csv",
}
TARGETS = {
    "composition": "source_tables/hela_composition.tsv",
    "cluster_odds_ratio": "case_tables/pispa_hela/hela_cluster_odds_ratio.tsv",
    "pca": "source_tables/hela_pca_coordinates.tsv",
    "candidates": "source_tables/hela_candidates.tsv",
    "volcano": "source_tables/hela_volcano.tsv",
    "tbc_values": "source_tables/tbc1d10b_cell_values.tsv",
    "mask_counts": "case_tables/pispa_hela/pispa_candidate_mask_counts.tsv",
}
CONTRASTS = ("C1 - C3", "C2 - C3", "C1 - C2")
CLUSTERS = ("Cluster 1", "Cluster 2", "Cluster 3")
Z_975 = 1.959963984540054   # 97.5 percent point of the standard normal (both intervals)
# The method label of the computed odds-ratio table; an identity field of that table, so the
# recomputed frame must carry exactly this text.
ODDS_RATIO_METHOD = ("Woolf log-odds Wald interval (normal approximation, "
                     "no continuity correction)")
ODDS_RATIO_ROW = "odds ratio (Cluster 1 vs Cluster 2)"


# --------------------------------------------------------------------------- t distribution
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz method)."""
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
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Two-sided Student t tail probability; no scipy dependency (agrees with scipy ~1e-12)."""
    out = np.empty(len(t), dtype=float)
    for i, (ti, di) in enumerate(zip(np.asarray(t, dtype=float), np.asarray(df, dtype=float))):
        if not np.isfinite(ti) or not np.isfinite(di) or di <= 0:
            out[i] = np.nan
        else:
            out[i] = _betai(0.5 * di, 0.5, di / (di + ti * ti))
    return out


def welch_ttest(a: np.ndarray, b: np.ndarray):
    """Row-wise two-sided Welch t-test; returns (mean difference, t, df)."""
    n_a = np.sum(~np.isnan(a), axis=1).astype(float)
    n_b = np.sum(~np.isnan(b), axis=1).astype(float)
    m_a, m_b = np.nanmean(a, axis=1), np.nanmean(b, axis=1)
    v_a = np.nanvar(a, axis=1, ddof=1)
    v_b = np.nanvar(b, axis=1, ddof=1)
    se2 = v_a / n_a + v_b / n_b
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (m_a - m_b) / np.sqrt(se2)
        df = se2 ** 2 / ((v_a / n_a) ** 2 / (n_a - 1) + (v_b / n_b) ** 2 / (n_b - 1))
    return m_a - m_b, t, df


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up over the finite members of one family."""
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


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


# --------------------------------------------------------------- comparison schema
# Pre-registered per frozen table, declared before any table is compared (see the shared
# comparator reproduce/cases/table_compare.py). Every frozen column is either compared through a
# Field or named in excluded with the reason why it is not compared.
SPEC_COMPOSITION = TableSpec(
    key_fields=("element", "cluster"),
    fields=(
        Field("element", "identity", source_class="IDENTITY_CHECK",
              note="fixed label of the frozen row; the counts and the four statistics carry "
                   "their own element name"),
        Field("cluster", "identity", source_class="IDENTITY_CHECK",
              note="cluster name, or the compared pair / 'all three clusters' of the statistics "
                   "rows"),
        Field("control_cells", "int", allows_missing=True, source_class="RECOMPUTED",
              note="cell count, exact; the four statistics rows are empty on both sides"),
        Field("migrated_cells", "int", allows_missing=True, source_class="RECOMPUTED",
              note="cell count, exact; the four statistics rows are empty on both sides"),
        Field("total_cells", "int", allows_missing=True, source_class="RECOMPUTED",
              note="cell count, exact; the four statistics rows are empty on both sides"),
        Field("migrated_share", "float", allows_missing=True, source_class="RECOMPUTED",
              note="migrated/total; the frozen column stores full double precision (at most 16 "
                   "decimals in its raw text), so rtol 1e-7 with atol 0 applies and no rounding "
                   "is declared"),
        Field("value", "float", allows_missing=True, format_decimals=3,
              source_class="RECOMPUTED",
              note="the frozen file stores at most 3 decimals; the emitting panel code "
                   "(V11_ALL_PANELS_20260917/scripts/plot_cases.py, _a1b_stats, lines 361-369) "
                   "writes these cells at the rendering of the PiSPA_A1b report it transcribes "
                   "(odds ratio 26.0, proportion difference 0.672, chi-square 33.83, Cramer's V "
                   "0.617), so the allowance is the declared 0.5e-3"),
        Field("ci_low", "float", allows_missing=True, format_decimals=2,
              source_class="RECOMPUTED",
              note="the frozen file stores at most 2 decimals (7.41, 0.5); same emitting code "
                   "and report rendering as the value column, allowance 0.5e-2"),
        Field("ci_high", "float", allows_missing=True, format_decimals=3,
              source_class="RECOMPUTED",
              note="the frozen file stores at most 3 decimals (0.844), so the allowance is "
                   "0.5e-3; the odds-ratio statistics row, whose stored token 91.2 is the "
                   "transcription's coarser rendering of the Woolf upper bound, is separated "
                   "from this comparison by the declared row filter and is covered by "
                   "hela_odds_ratio_computed and hela_odds_ratio_reported_transcription; the "
                   "declaration of this field is unchanged"),
    ),
    excluded=(("source", "provenance column naming the file each frozen row was taken from (the "
                         "cluster-by-type cross table or the PiSPA_A1b report); not a recomputed "
                         "quantity"),),
    label="hela_composition")

SPEC_CLUSTER_ODDS_RATIO = TableSpec(
    key_fields=("comparison",),
    fields=(
        Field("comparison", "identity", source_class="IDENTITY_CHECK",
              note="the compared cluster pair; the row address of the single stored row"),
        Field("group_a", "identity", source_class="IDENTITY_CHECK",
              note="name of the first group of the pair; carried by the stable table as text"),
        Field("group_b", "identity", source_class="IDENTITY_CHECK",
              note="name of the second group of the pair; carried by the stable table as text"),
        Field("group_a_migrated_cells", "int", source_class="RECOMPUTED",
              note="migrated cells of group A, exact integer (35 in the archived cross-tab)"),
        Field("group_a_control_cells", "int", source_class="RECOMPUTED",
              note="control cells of group A, exact integer (7); a changed count must fail"),
        Field("group_b_migrated_cells", "int", source_class="RECOMPUTED",
              note="migrated cells of group B, exact integer (5)"),
        Field("group_b_control_cells", "int", source_class="RECOMPUTED",
              note="control cells of group B, exact integer (26)"),
        Field("odds_ratio", "float", rtol=1e-7, atol=0.0, source_class="RECOMPUTED",
              note="(a_migrated x b_control) / (a_control x b_migrated); the stable table stores "
                   "the full double precision, so the pre-registered rtol 1e-7 with atol 0 "
                   "applies and no rounding allowance is declared (no format_decimals, no "
                   "rounds_to)"),
        Field("ci_low", "float", rtol=1e-7, atol=0.0, source_class="RECOMPUTED",
              note="exp(log(odds_ratio) - z x se_log_odds); strict floating point at rtol 1e-7 "
                   "with atol 0 and no rounding allowance, so a truncated or re-rounded "
                   "endpoint fails"),
        Field("ci_high", "float", rtol=1e-7, atol=0.0, source_class="RECOMPUTED",
              note="exp(log(odds_ratio) + z x se_log_odds); strict floating point at rtol 1e-7 "
                   "with atol 0 and no rounding allowance: the two-decimal rendering of this "
                   "endpoint is a different number and must fail"),
        Field("confidence_level", "identity", source_class="IDENTITY_CHECK",
              note="0.95 compared as an exact token: the interval is a two-sided 95 percent "
                   "Wald interval, so the stored text is the identity of that constant, not a "
                   "measured quantity; the file stores the shortest round-trip form, which is "
                   "what the recomputed frame carries as well"),
        Field("method", "identity", source_class="IDENTITY_CHECK",
              note="identity text of the interval method; not a recomputed quantity"),
        Field("z_975", "identity", source_class="IDENTITY_CHECK",
              note="1.959963984540054 compared as an exact token: the stored text is the "
                   "identity of the module constant Z_975 that this script uses for both "
                   "intervals, so an exact-token comparison is the right kind and a relative "
                   "tolerance would add nothing"),
    ),
    excluded=(
        ("manuscript_ci_low_display", "manuscript display expectation, checked by "
                                       "hela_odds_ratio_manuscript_display"),
        ("manuscript_ci_high_display", "manuscript display expectation, checked by "
                                        "hela_odds_ratio_manuscript_display"),
        ("manuscript_anchor", "manuscript display expectation, checked by "
                               "hela_odds_ratio_manuscript_display"),
        ("computed_by", "provenance sentence of the delivered table naming the computing script "
                        "and the formula; a publication record, not a recomputed quantity (the "
                        "same role as the source column of hela_composition.tsv)"),
    ),
    label="hela_cluster_odds_ratio")

SPEC_CANDIDATES = TableSpec(
    key_fields=("protein", "contrast"),
    fields=(
        Field("protein", "identity", source_class="IDENTITY_CHECK",
              note="protein-group key of the processed matrix, carried from the frozen table as "
                   "the row address of the comparison; not recomputed"),
        Field("candidate", "identity", source_class="IDENTITY_CHECK",
              note="candidate label carried from the frozen table; not recomputed"),
        Field("gene", "identity", source_class="IDENTITY_CHECK",
              note="gene label carried from the frozen table; not recomputed"),
        Field("contrast", "identity", source_class="RECOMPUTED",
              note="one of the three fixed cluster contrasts"),
        Field("effect", "float", source_class="RECOMPUTED",
              note="difference of group means on the processed matrix; the frozen column stores "
                   "full double precision, rtol 1e-7 with atol 0"),
        Field("welch_p", "pvalue", source_class="RECOMPUTED",
              note="two-sided Welch P on the same matrix; purely relative comparison, atol 0"),
        Field("q", "pvalue", source_class="RECOMPUTED",
              note="BH step-up inside the 4,688-protein family of that contrast; purely "
                   "relative comparison, atol 0"),
        Field("q_family_size", "int", source_class="RECOMPUTED",
              note="family size of the contrast, exact"),
        Field("passes_within_bh", "identity", source_class="RECOMPUTED",
              note="q < 0.05 of this contrast; the frozen file stores the boolean as the literal "
                   "text True/False, so the field is declared as an exact text comparison - a "
                   "numeric flag comparison cannot parse the frozen token"),
    ),
    label="hela_candidates")

SPEC_VOLCANO = TableSpec(
    key_fields=("protein",),
    fields=(
        Field("protein", "identity", source_class="RECOMPUTED",
              note="protein-group key of the processed matrix"),
        Field("gene", "identity", allows_missing=True, source_class="RECOMPUTED",
              note="PG.Genes of the same row; one frozen row carries no gene name and the "
                   "recomputation leaves that cell empty as well"),
        Field("effect", "float", source_class="RECOMPUTED",
              note="migrated minus control difference of group means; the frozen column stores "
                   "about twelve significant digits, rtol 1e-7 with atol 0"),
        Field("welch_p", "pvalue", source_class="RECOMPUTED",
              note="two-sided Welch P of the same contrast; purely relative comparison, atol 0"),
        Field("bh_q", "pvalue", source_class="RECOMPUTED",
              note="BH step-up inside the single 4,688-protein family; purely relative "
                   "comparison, atol 0"),
        Field("det_rate_migrated", "float", format_decimals=6, source_class="RECOMPUTED",
              note="detected cells of the migrated group divided by 46; the frozen file stores "
                   "at most 6 decimals and the recomputation rounds the same quantity to 6 "
                   "decimals, so the allowance is 0.5e-6"),
        Field("det_rate_control", "float", format_decimals=6, source_class="RECOMPUTED",
              note="detected cells of the control group divided by 43; same emitting code and "
                   "6-decimal rendering as the migrated rate, allowance 0.5e-6"),
        Field("labelled", "identity", source_class="IDENTITY_CHECK",
              note="manuscript label of the row, carried from the frozen table as the literal "
                   "text True/False and compared exactly; not recomputed"),
    ),
    label="hela_volcano")

SPEC_MASK_COUNTS = TableSpec(
    key_fields=("protein_key", "cluster"),
    fields=(
        Field("candidate", "identity", source_class="IDENTITY_CHECK",
              note="candidate label carried from the frozen table; not recomputed"),
        Field("gene", "identity", source_class="IDENTITY_CHECK",
              note="gene label carried from the frozen table; not recomputed"),
        Field("protein_key", "identity", source_class="IDENTITY_CHECK",
              note="protein-group key used to address the processed matrix"),
        Field("cluster", "identity", source_class="RECOMPUTED",
              note="cluster whose cells are counted"),
        Field("n_total", "int", source_class="RECOMPUTED",
              note="cells of that cluster, exact"),
        Field("n_observed", "int", source_class="RECOMPUTED",
              note="cells not equal to the row minimum of the processed matrix, exact"),
        Field("n_imputed", "int", source_class="RECOMPUTED",
              note="cells equal to the row minimum (half-minimum identity), exact"),
        Field("n_unresolved", "int", source_class="RECOMPUTED",
              note="cells the reconstruction cannot resolve, exact (zero in every frozen row)"),
    ),
    excluded=(
        ("mask_rule", "prose column of the historical run, regenerated by this script with "
                      "different wording; not a compared quantity"),
        ("evidence_path", "path column of the historical run directory; the archive path used by "
                          "this recomputation is recorded in the case ledger, not in the table"),
        ("note", "per-row prose note of the historical run (row minimum and masked cell count); "
                 "regenerated by this script and not a compared quantity"),
    ),
    label="hela_candidate_mask_counts")


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HeLa migration case (PiSPA), deterministic re-run.")
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

    matrix_path = data_dir / INPUTS["matrix"]
    sampleinfo_path = data_dir / INPUTS["sampleinfo"]
    transform_path = data_dir / INPUTS["transform"]
    raw_path = data_dir / INPUTS["raw_matrix"]

    matrix = pd.read_csv(matrix_path)
    matrix.columns = [str(c).strip().lstrip("\ufeff") for c in matrix.columns]
    sampleinfo = pd.read_csv(sampleinfo_path)
    sampleinfo.columns = [str(c).strip().lstrip("\ufeff") for c in sampleinfo.columns]
    transform = json.loads(transform_path.read_text(encoding="utf-8-sig"))

    gene_col = "PG.Genes" if "PG.Genes" in matrix.columns else "PG. Genes"
    pg_col = "PG.ProteinGroups"
    samples = [str(s) for s in sampleinfo["FileName"]]
    matrix_samples = [c for c in matrix.columns if c not in (pg_col, gene_col)]
    prot = [str(p) for p in matrix[pg_col]]
    genes = ["" if bool(pd.isna(g)) else str(g) for g in matrix[gene_col]]
    prot = ["" if bool(pd.isna(p)) else str(p) for p in matrix[pg_col]]
    values = matrix[matrix_samples].to_numpy(dtype=float)
    cluster_of = dict(zip(samples, sampleinfo["Cluster"].astype(str)))
    type_of = dict(zip(samples, sampleinfo["Type"].astype(str)))
    migrated = [s for s in samples if type_of[s] == "Migrated Cell"]
    control = [s for s in samples if type_of[s] == "Control Cell"]
    col_of = {s: i for i, s in enumerate(matrix_samples)}
    cluster_index = {c: [col_of[s] for s in samples if cluster_of[s] == c] for c in CLUSTERS}

    # ---- 1. matrix identity -------------------------------------------------
    n_imp_recorded = int(transform.get("imputation", {}).get("n_values_imputed", -1))
    n_rows_recorded = int(transform.get("protein_filtering", {}).get("n_proteins_retained", -1))
    shape_ok = (matrix.shape[0] == 4688 and len(matrix_samples) == 89
                and matrix_samples == samples and n_rows_recorded == matrix.shape[0])
    ledger.add("hela_matrix_identity", "processed matrix shape and recorded processing state",
               INPUTS["matrix"], "4,688 protein rows x 89 sample columns in SampleInfo order; "
               "transform record retains 4,688 rows and records 116,904 imputed values",
               "%d x %d; retained=%d; imputed=%d" % (matrix.shape[0], len(matrix_samples),
                                                     n_rows_recorded, n_imp_recorded),
               "PASS" if shape_ok else "FAIL", "exact",
               "row count, column count, SampleInfo column order, matrix_transform_record.json")

    # ---- 2. cluster composition --------------------------------------------
    counts = {c: sum(1 for s in samples if cluster_of[s] == c) for c in CLUSTERS}
    migrated_by_cluster = {c: sum(1 for s in migrated if cluster_of[s] == c) for c in CLUSTERS}
    frozen_comp = read_tsv(data_dir / TARGETS["composition"])
    comp_rows = frozen_comp[frozen_comp["element"] == "cluster composition"]
    expected_counts = {str(r["cluster"]): (int(r["control_cells"]), int(r["migrated_cells"]),
                                            int(r["total_cells"])) for _, r in comp_rows.iterrows()}
    counts_ok = all(
        counts[c] == expected_counts[c][2]
        and migrated_by_cluster[c] == expected_counts[c][1]
        and counts[c] - migrated_by_cluster[c] == expected_counts[c][0]
        for c in CLUSTERS)
    odds = (migrated_by_cluster["Cluster 1"] * expected_counts["Cluster 2"][0]) / (
        expected_counts["Cluster 1"][0] * migrated_by_cluster["Cluster 2"])
    frozen_odds = float(frozen_comp.loc[frozen_comp["value"].notna(), "value"].iloc[0])
    ledger.add("hela_composition_counts", "cluster composition and migrated-cell counts",
               TARGETS["composition"], "42/31/16 cells; 35/5/6 migrated; 46 migrated + 43 control",
               "%d/%d/%d cells; %d/%d/%d migrated; %d + %d" % tuple(
                   [counts[c] for c in CLUSTERS] + [migrated_by_cluster[c] for c in CLUSTERS]
                   + [len(migrated), len(control)]),
               "PASS" if counts_ok else "FAIL", "exact",
               "recomputed from SampleInfo_Filtered.csv; compared with hela_composition.tsv")
    ledger.add("hela_composition_odds_ratio", "migration odds ratio C1 vs C2", TARGETS["composition"],
               "26.0 = (35 x 26) / (7 x 5)", "%.6g (frozen %.6g)" % (odds, frozen_odds),
               "PASS" if abs(odds - frozen_odds) <= REL_TOL * abs(frozen_odds) else "FAIL",
               "rel1e-7", "closed-form odds ratio from the same 2x2 table")
    # The frozen rows 4-7 are transcriptions of the PiSPA_A1b report (its own source column says
    # so). They are re-derived here from the same cluster-by-phenotype cross-tab: difference of
    # two proportions with the unpooled normal interval, Pearson chi-square of the 3 x 2 table
    # and Cramer's V. The odds-ratio row is the one row of that group whose stored interval
    # cannot be the recomputed one at any declared resolution (91.2 against the Woolf value
    # 91.187482), so it is separated from this row-by-row comparison by a declared row filter and
    # is covered by hela_odds_ratio_computed, hela_odds_ratio_reported_transcription and
    # hela_odds_ratio_manuscript_display below.
    migrated_counts = [migrated_by_cluster[c] for c in CLUSTERS]
    control_counts = [counts[c] - migrated_by_cluster[c] for c in CLUSTERS]
    or_scale = math.sqrt(1.0 / migrated_counts[0] + 1.0 / control_counts[0]
                         + 1.0 / migrated_counts[1] + 1.0 / control_counts[1])
    or_low = odds * math.exp(-Z_975 * or_scale)
    or_high = odds * math.exp(Z_975 * or_scale)
    share_1, share_2 = migrated_counts[0] / counts[CLUSTERS[0]], migrated_counts[1] / counts[CLUSTERS[1]]
    diff_scale = math.sqrt(share_1 * (1.0 - share_1) / counts[CLUSTERS[0]]
                           + share_2 * (1.0 - share_2) / counts[CLUSTERS[1]])
    prop_diff = share_1 - share_2
    pd_low, pd_high = prop_diff - Z_975 * diff_scale, prop_diff + Z_975 * diff_scale
    row_totals = [m + c for m, c in zip(migrated_counts, control_counts)]
    grand_total = float(sum(row_totals))
    chi_square = 0.0
    for row_total, detected, undetected in zip(row_totals, migrated_counts, control_counts):
        for observed, column_total in ((detected, sum(migrated_counts)),
                                       (undetected, sum(control_counts))):
            expected = row_total * column_total / grand_total
            chi_square += (observed - expected) ** 2 / expected
    cramers_v = math.sqrt(chi_square / grand_total)
    shared = {"control_cells": np.nan, "migrated_cells": np.nan, "total_cells": np.nan,
              "migrated_share": np.nan}
    recomputed_comp = pd.DataFrame(
        [{"element": "cluster composition", "cluster": c,
          "control_cells": int(counts[c] - migrated_by_cluster[c]),
          "migrated_cells": int(migrated_by_cluster[c]), "total_cells": int(counts[c]),
          "migrated_share": migrated_by_cluster[c] / counts[c],
          "value": np.nan, "ci_low": np.nan, "ci_high": np.nan} for c in CLUSTERS]
        + [{"element": "odds ratio (Cluster 1 vs Cluster 2)", "cluster": "Cluster 1 vs Cluster 2",
            **shared, "value": odds, "ci_low": or_low, "ci_high": or_high},
           {"element": "proportion difference (Cluster 1 vs Cluster 2)",
            "cluster": "Cluster 1 vs Cluster 2", **shared, "value": prop_diff,
            "ci_low": pd_low, "ci_high": pd_high},
           {"element": "chi-square (3 x 2 table)", "cluster": "all three clusters", **shared,
            "value": chi_square, "ci_low": np.nan, "ci_high": np.nan},
           {"element": "Cramer's V (3 x 2 table)", "cluster": "all three clusters", **shared,
            "value": cramers_v, "ci_low": np.nan, "ci_high": np.nan}],
        columns=["element", "cluster", "control_cells", "migrated_cells", "total_cells",
                 "migrated_share", "value", "ci_low", "ci_high"])
    ledger.add_table("hela_composition_rows", "composition rows against the frozen table",
                     data_dir / TARGETS["composition"],
                     recomputed_comp[recomputed_comp["element"] != ODDS_RATIO_ROW]
                     .reset_index(drop=True), SPEC_COMPOSITION,
                     evidence="counts from SampleInfo_Filtered.csv; migrated share as count/total",
                     notes="the statistics rows of the frozen file are transcriptions of the "
                           "PiSPA_A1b report at that report's own rendering and are re-derived "
                           "here from the same cross-tab; the provenance column source is "
                           "excluded; the odds-ratio statistics row (element '" + ODDS_RATIO_ROW +
                           "') is separated from this row-by-row comparison by the declared row "
                           "filter and is covered by the two new checks: "
                           "hela_odds_ratio_computed compares the computed table "
                           "case_tables/pispa_hela/hela_cluster_odds_ratio.tsv at rtol 1e-7 with "
                           "atol 0 and no rounding allowance, and "
                           "hela_odds_ratio_reported_transcription checks the historical tokens "
                           "as a documented rendering of the recomputed value (its stored 91.2 "
                           "is the 3-significant-digit rendering, not the 2-decimal "
                           "recomputation 91.19); no tolerance of this comparison was widened "
                           "for the separation, and the remaining ci_low token (0.5) and ci_high "
                           "token (0.844) keep their declared 2- and 3-decimal allowances",
                     method_extra=" + the four report statistics re-derived from the same "
                                  "cluster-by-phenotype cross-tab, minus the separated "
                                  "odds-ratio row",
                     row_filter=lambda table: table["element"] != ODDS_RATIO_ROW)
    recomputed_comp.to_csv(out_dir / "hela_composition_recomputed.tsv", sep="\t", index=False)

    # ---- 2b. the computed object: the stable cluster odds-ratio table --------
    # case_tables/pispa_hela/hela_cluster_odds_ratio.tsv is the object that carries the odds
    # ratio and its Woolf log-odds interval as computed values. It is compared with strict
    # floating point rules (rtol 1e-7, atol 0, no format_decimals and no rounds_to), so the
    # stored endpoints have to be the recomputed ones, not a rendering of them, and the four
    # cell counts compare exactly.
    recomputed_or = pd.DataFrame(
        [{"comparison": "Cluster 1 vs Cluster 2",
          "group_a": CLUSTERS[0], "group_b": CLUSTERS[1],
          "group_a_migrated_cells": int(migrated_counts[0]),
          "group_a_control_cells": int(control_counts[0]),
          "group_b_migrated_cells": int(migrated_counts[1]),
          "group_b_control_cells": int(control_counts[1]),
          "odds_ratio": odds, "ci_low": or_low, "ci_high": or_high,
          "confidence_level": 0.95, "method": ODDS_RATIO_METHOD, "z_975": Z_975}],
        columns=["comparison", "group_a", "group_b", "group_a_migrated_cells",
                 "group_a_control_cells", "group_b_migrated_cells", "group_b_control_cells",
                 "odds_ratio", "ci_low", "ci_high", "confidence_level", "method", "z_975"])
    ledger.add_table("hela_odds_ratio_computed", "computed cluster odds ratio against the stable table",
                     data_dir / TARGETS["cluster_odds_ratio"], recomputed_or,
                     SPEC_CLUSTER_ODDS_RATIO,
                     evidence="the four cells of the Cluster 1 versus Cluster 2 cross-tab from "
                              "SampleInfo_Filtered.csv (%d/%d and %d/%d migrated/control); odds "
                              "ratio (a_migrated x b_control)/(a_control x b_migrated) and the "
                              "Woolf log-odds interval at the same z = %.15g this script uses "
                              "for the composition rows" % (
                                  migrated_counts[0], control_counts[0],
                                  migrated_counts[1], control_counts[1], Z_975),
                     notes="strict floating point: odds_ratio, ci_low and ci_high are declared "
                           "with rtol 1e-7 and atol 0 and with no rounding allowance, so a "
                           "two-decimal or three-significant-digit rendering of an endpoint "
                           "fails and only the recomputed double passes; the four cell counts "
                           "are exact integers and the manuscript display columns are excluded "
                           "here (they are checked by hela_odds_ratio_manuscript_display); this "
                           "is the computed object of the odds ratio, never an agent output",
                     method_extra=" + the Woolf interval of the archived 2 x 2 table at rtol 1e-7 "
                                  "with atol 0 and no rounding allowance")

    # ---- 2c. the historical row: a reported transcription, not the truth ----
    # The odds-ratio row of hela_composition.tsv is retained unchanged and keeps its own source
    # column (the PiSPA_A1b agent report). Its tokens are checked as a documented rendering of
    # the recomputed value and are never presented as the recomputed truth.
    # The tokens are read as raw text (read_frozen), not through pandas: the check is about the
    # stored rendering, so '26' must not become '26.0' on the way in.
    frozen_comp_raw = read_frozen(data_dir / TARGETS["composition"])
    reported = frozen_comp_raw[frozen_comp_raw["element"] == ODDS_RATIO_ROW]
    reported_source = "PiSPA_A1b output_report_1.md"
    tokens = ({"value": str(reported["value"].iloc[0]).strip(),
               "ci_low": str(reported["ci_low"].iloc[0]).strip(),
               "ci_high": str(reported["ci_high"].iloc[0]).strip()}
              if len(reported) == 1 and "source" in frozen_comp_raw.columns else {})
    provenance_ok = bool(len(reported) == 1 and "source" in frozen_comp_raw.columns
                         and str(reported["source"].iloc[0]).strip() == reported_source)
    value_rendering = "%.3g" % odds
    low_rendering = "%.2f" % or_low
    high_renderings = {"%.2f": "%.2f" % or_high, "%.3g": "%.3g" % or_high}
    high_matched = [name for name, text in sorted(high_renderings.items())
                    if tokens.get("ci_high") == text]
    if high_matched == ["%.2f"]:
        high_text = ("ci_high token is the two-decimal rendering '%.2f' of the recomputed "
                     "value, i.e. the frozen token was changed to the two-decimal form"
                     % or_high)
    elif high_matched:
        high_text = ("ci_high token '%s' matches only the three-significant-digit rendering "
                     "'%.3g' of the recomputed value; it is not the two-decimal recomputation "
                     "'%.2f'" % (tokens.get("ci_high"), or_high, or_high))
    else:
        high_text = ("ci_high token '%s' matches neither the three-significant-digit rendering "
                     "'%.3g' nor the two-decimal rendering '%.2f' of the recomputed value"
                     % (tokens.get("ci_high"), or_high, or_high))
    transcription_ok = bool(provenance_ok and tokens
                            and tokens.get("value") == value_rendering
                            and tokens.get("ci_low") == low_rendering
                            and high_matched)
    ledger.add("hela_odds_ratio_reported_transcription",
               "historical odds-ratio row of hela_composition.tsv is a REPORTED_TRANSCRIPTION "
               "of the PiSPA_A1b agent report",
               TARGETS["composition"],
               "row retained with its own source column; value '26' = '%.3g' of the recomputed "
               "odds ratio, ci_low '7.41' = '%.2f' of the recomputed lower bound, ci_high '91.2' "
               "= '%.3g' (the two-decimal recomputation is '%.2f')"
               % (odds, or_low, or_high, or_high),
               "row source '%s'; value '%s' vs '%.3g'; ci_low '%s' vs '%.2f'; %s"
               % (str(reported["source"].iloc[0]).strip() if provenance_ok else "<missing>",
                  tokens.get("value", "<missing>"), odds, tokens.get("ci_low", "<missing>"),
                  or_low, high_text),
               "PASS" if transcription_ok else "FAIL",
               "token-by-token rendering check against the recomputed values of the same 2 x 2 "
               "table",
               "frozen tokens of the historical row and of its source column")
    ledger.checks[-1]["column_classes"] = {"value": "COPIED_REFERENCE",
                                           "ci_low": "COPIED_REFERENCE",
                                           "ci_high": "COPIED_REFERENCE"}

    # ---- 2d. the values the manuscript prints -------------------------------
    stable = read_tsv(data_dir / TARGETS["cluster_odds_ratio"])
    display_columns = ("manuscript_ci_low_display", "manuscript_ci_high_display")
    if len(stable) == 1 and all(name in stable.columns for name in display_columns):
        display_low = str(stable["manuscript_ci_low_display"].iloc[0]).strip()
        display_high = str(stable["manuscript_ci_high_display"].iloc[0]).strip()
    else:
        display_low = display_high = "<missing>"
    shown_low = "%.1f" % round(or_low, 1)
    shown_high = "%.1f" % round(or_high, 1)
    display_ok = bool(shown_low == display_low and shown_high == display_high)
    ledger.add("hela_odds_ratio_manuscript_display",
               "manuscript prints the odds-ratio interval to one decimal",
               TARGETS["cluster_odds_ratio"],
               "the manuscript display columns of the stable table read back and equal to the "
               "one-decimal rounding of the recomputed interval (%.1f and %.1f)"
               % (round(or_low, 1), round(or_high, 1)),
               "computed one decimal %.1f / %.1f against the stable table %s / %s - display %s"
               % (round(or_low, 1), round(or_high, 1), display_low, display_high,
                  "consistent" if display_ok else "inconsistent"),
               "PASS" if display_ok else "FAIL", "one-decimal rendering",
               "manuscript_ci_low_display / manuscript_ci_high_display of "
               "case_tables/pispa_hela/hela_cluster_odds_ratio.tsv")

    # ---- 3. principal components -------------------------------------------
    standardised = (values.T - values.T.mean(axis=0, keepdims=True)) / values.T.std(
        axis=0, ddof=0, keepdims=True)
    u, s, _ = np.linalg.svd(standardised, full_matrices=False)
    variance_ratio = s ** 2 / (s ** 2).sum()
    frozen_pca = read_tsv(data_dir / TARGETS["pca"])
    recorded = [0.12053490000108308, 0.1048803731936116]
    ratio_rel = max(abs(variance_ratio[i] - recorded[i]) / abs(recorded[i]) for i in range(2))
    ledger.add("hela_pca_variance_ratio", "explained variance ratio of PC1 and PC2",
               TARGETS["pca"], "recorded PC1 0.12053490000108308, PC2 0.1048803731936116",
               "PC1 %.17g, PC2 %.17g (max rel %.3g)" % (variance_ratio[0], variance_ratio[1],
                                                        ratio_rel),
               "PASS" if ratio_rel <= REL_TOL else "FAIL", "rel1e-7",
               "numpy SVD of the per-protein standardised 89 x 4,688 matrix")
    scores = u * s
    signs = []
    coord_rel = []
    coord_corr = []
    for k in range(2):
        x = scores[:, k]
        y = frozen_pca["pc%d" % (k + 1)].to_numpy(dtype=float)
        sign = 1.0 if float(np.dot(x, y)) >= 0 else -1.0
        signs.append(int(sign))
        coord_rel.append(float(np.max(np.abs(sign * x - y) / np.maximum(np.abs(y), 1e-300))))
        coord_corr.append(float(np.corrcoef(sign * x, y)[0, 1]))
    solver_bound = max(coord_rel) <= 1e-4 and min(coord_corr) >= 1.0 - 1e-9
    ledger.add("hela_pca_coordinates", "sign-aligned PC1/PC2 coordinates of the 89 cells",
               TARGETS["pca"],
               "agreement up to the documented solver non-determinism (randomized solver, no "
               "seed recorded): correlation >= 1 - 1e-9 and max relative difference <= 1e-4",
               "signs %s; max rel %.3g / %.3g; correlation %.12f / %.12f" % (
                   signs, coord_rel[0], coord_rel[1], coord_corr[0], coord_corr[1]),
               "PASS" if solver_bound else "FAIL", "rel1e-4 + correlation",
               "numpy SVD; the historical run used sklearn's randomized solver without a seed, "
               "so a 1e-7-exact coordinate reproduction cannot be requested")
    recomputed_pca = pd.DataFrame({
        "sample_id": samples, "cluster": [cluster_of[s] for s in samples],
        "phenotype": [type_of[s] for s in samples],
        "pc1": signs[0] * scores[:, 0], "pc2": signs[1] * scores[:, 1]})
    recomputed_pca.to_csv(out_dir / "hela_pca_coordinates_recomputed.tsv", sep="\t", index=False)
    pca_identity_ok = bool((recomputed_pca[["sample_id", "cluster", "phenotype"]].astype(str)
                            .to_numpy() == frozen_pca[["sample_id", "cluster", "phenotype"]]
                            .astype(str).to_numpy()).all())
    ledger.add("hela_pca_coordinate_identity", "cell order and labels of the PCA table",
               TARGETS["pca"], "89 rows with the same sample, cluster and phenotype labels",
               "identity mismatches %d" % int(
                   (recomputed_pca[["sample_id", "cluster", "phenotype"]].astype(str).to_numpy()
                    != frozen_pca[["sample_id", "cluster", "phenotype"]].astype(str)
                    .to_numpy()).sum()),
               "PASS" if pca_identity_ok else "FAIL", "exact", "row-by-row label comparison")

    # ---- 4. fixed candidate cluster contrasts -------------------------------
    frozen_cand = read_tsv(data_dir / TARGETS["candidates"])
    computed = {}
    for cname in CONTRASTS:
        first, second = ["Cluster " + t[1:] for t in [x.strip() for x in cname.split("-")]]
        effect, t_stat, dfree = welch_ttest(values[:, cluster_index[first]],
                                            values[:, cluster_index[second]])
        p_value = student_t_two_sided_p(t_stat, dfree)
        computed[cname] = (effect, p_value, bh_adjust(p_value))
    cand_rows = []
    for _, row in frozen_cand.iterrows():
        effect, p_value, q_value = computed[str(row["contrast"])]
        i = prot.index(str(row["protein"]))
        cand_rows.append({"protein": row["protein"], "candidate": row["candidate"],
                          "gene": row["gene"], "contrast": row["contrast"], "effect": effect[i],
                          "welch_p": p_value[i], "q": q_value[i], "q_family_size": int(p_value.size),
                          "passes_within_bh": bool(q_value[i] < 0.05)})
    recomputed_cand = pd.DataFrame(cand_rows, columns=list(frozen_cand.columns))
    recomputed_cand.to_csv(out_dir / "hela_candidates_recomputed.tsv", sep="\t", index=False)
    ledger.add_table("hela_candidate_contrasts", "six fixed candidates x three cluster contrasts",
                     data_dir / TARGETS["candidates"], recomputed_cand, SPEC_CANDIDATES,
                     evidence="two-sided Welch on the processed matrix, BH step-up within each "
                              "4,688-protein contrast family, frozen row order")

    # ---- 5. offline migration-label comparison ------------------------------
    raw = pd.read_csv(raw_path)
    raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]
    annotation = [c for c in raw.columns if not (raw[c].dtype.kind in "if")]
    raw_samples = [c for c in raw.columns if c not in annotation]
    detection = raw.set_index(pg_col).reindex(prot)[raw_samples].notna().reset_index(drop=True)
    detection.columns = raw_samples
    detected_migrated = detection[migrated].to_numpy().sum(axis=1)
    detected_control = detection[control].to_numpy().sum(axis=1)
    effect, t_stat, dfree = welch_ttest(values[:, [col_of[s] for s in migrated]],
                                        values[:, [col_of[s] for s in control]])
    p_value = student_t_two_sided_p(t_stat, dfree)
    q_value = bh_adjust(p_value)
    frozen_vol = read_tsv(data_dir / TARGETS["volcano"])
    vol_rel = {}
    for col, arr in (("effect", effect), ("welch_p", p_value), ("bh_q", q_value)):
        ref = frozen_vol[col].to_numpy(dtype=float)
        vol_rel[col] = float(np.nanmax(np.abs(arr - ref) / np.abs(ref)))
    det_rates_ok = bool(np.all(np.round(detected_migrated / len(migrated), 6)
                               == frozen_vol["det_rate_migrated"].to_numpy(dtype=float))
                        and np.all(np.round(detected_control / len(control), 6)
                                   == frozen_vol["det_rate_control"].to_numpy(dtype=float)))
    ledger.add("hela_migration_label_contrast", "offline migrated-versus-control comparison",
               TARGETS["volcano"], "4,688 rows; single 4,688-protein BH family",
               "max rel effect %.3g, P %.3g, q %.3g; detection rates equal to 6 stored decimals %s"
               % (vol_rel["effect"], vol_rel["welch_p"], vol_rel["bh_q"], det_rates_ok),
               "PASS" if all(v <= REL_TOL for v in vol_rel.values()) and det_rates_ok else "FAIL",
               "rel1e-7 (detection rates: the frozen column stores 6 significant digits)",
               "Welch t-test on the saved matrix; detection mask from the delivered raw matrix")
    significant = q_value < 0.05
    higher = int((significant & (effect > 0)).sum())
    lower = int((significant & (effect < 0)).sum())
    frozen_significant = int((frozen_vol["bh_q"].to_numpy(dtype=float) < 0.05).sum())
    ledger.add("hela_migration_label_headline", "q < 0.05 proteins of the migration contrast",
               TARGETS["volcano"], "513 significant, 104 higher and 409 lower in migrated cells",
               "%d significant, %d higher, %d lower (frozen table %d significant)" % (
                   int(significant.sum()), higher, lower, frozen_significant),
               "PASS" if (int(significant.sum()) == 513 == frozen_significant and higher == 104
                          and lower == 409) else "FAIL", "exact",
               "counts recomputed from the recomputed q values and from the frozen q values")
    recomputed_vol = pd.DataFrame({
        "protein": prot, "gene": genes, "effect": effect, "welch_p": p_value, "bh_q": q_value,
        "det_rate_migrated": np.round(detected_migrated / len(migrated), 6),
        "det_rate_control": np.round(detected_control / len(control), 6),
        "labelled": frozen_vol["labelled"].to_numpy(dtype=bool)})
    recomputed_vol.to_csv(out_dir / "hela_volcano_recomputed.tsv", sep="\t", index=False)
    ledger.add_table("hela_volcano_table", "recomputed volcano table against the frozen table",
                     data_dir / TARGETS["volcano"], recomputed_vol, SPEC_VOLCANO,
                     evidence="4,688 rows, one BH family; the frozen file stores 12 significant "
                              "digits for effect/P/q and 6 for the detection rates")

    # ---- 6. pre-imputation observation support ------------------------------
    # The pre-imputation matrix (ProteinQuant_Filtered.csv) is not part of this archive. The
    # imputed entries are reconstructed from the half-minimum rule recorded in the transform
    # record: a row carries imputed values exactly when its second-smallest distinct value v
    # satisfies v ~ 2u in the pre-transform scale (u = 2**min - 1), because the filled value is
    # half of the smallest observed value of that row. Cells equal to the row minimum are then
    # the filled cells. The reconstruction is validated against the recorded number of filled
    # values, against the frozen TBC1D10B flags and against the frozen support table.
    n_at_min = np.zeros(values.shape[0], dtype=int)
    has_imputed = np.zeros(values.shape[0], dtype=bool)
    for i in range(values.shape[0]):
        row = values[i]
        row_min = float(row.min())
        n_at_min[i] = int((row == row_min).sum())
        distinct = np.unique(row)
        if distinct.size >= 2:
            u = 2.0 ** row_min - 1.0
            v = 2.0 ** float(distinct[1]) - 1.0
            has_imputed[i] = abs((v - 2.0 * u) / v) <= 1e-9
        else:
            has_imputed[i] = True
    imputed_cells = int(n_at_min[has_imputed].sum())
    ledger.add("hela_imputation_reconstruction", "reconstruction of the half-minimum imputation",
               INPUTS["transform"], "n_values_imputed = 116,904 in the transform record",
               "%d cells in %d rows satisfy the half-minimum identity" % (imputed_cells,
                                                                          int(has_imputed.sum())),
               "PASS" if imputed_cells == n_imp_recorded else "FAIL", "exact",
               "value-identity derivation from the processed matrix",
               "the pre-imputation matrix itself is not distributed; used only for "
               "observation-support counts")
    tbc_i = prot.index("Q4KMP7")
    tbc_c3 = int((values[tbc_i][cluster_index["Cluster 3"]] == values[tbc_i].min()).sum())
    frozen_tbc = read_tsv(data_dir / TARGETS["tbc_values"])
    tbc_flag_match = int(np.sum((values[tbc_i] == values[tbc_i].min())
                                != frozen_tbc["at_filled_minimum"].to_numpy(dtype=bool)))
    ledger.add("hela_tbc1d10b_support", "TBC1D10B observation support", TARGETS["tbc_values"],
               "31 of 89 values filled at the imputation minimum, 14 of 16 in Cluster 3",
               "%d of 89 filled, %d of 16 in Cluster 3; flag mismatches %d" % (
                   int(n_at_min[tbc_i]), tbc_c3, tbc_flag_match),
               "PASS" if (int(n_at_min[tbc_i]) == 31 and tbc_c3 == 14 and tbc_flag_match == 0)
               else "FAIL", "exact", "cell-by-cell comparison with tbc1d10b_cell_values.tsv")
    frozen_mask = read_tsv(data_dir / TARGETS["mask_counts"])
    mask_rows = []
    for _, row in frozen_mask.iterrows():
        i = prot.index(str(row["protein_key"]))
        cluster = str(row["cluster"])
        n_total = int(len(cluster_index[cluster]))
        n_imputed = (int((values[i][cluster_index[cluster]] == values[i].min()).sum())
                     if has_imputed[i] else 0)
        mask_rows.append({"candidate": row["candidate"], "gene": row["gene"],
                          "protein_key": row["protein_key"], "cluster": cluster,
                          "n_total": n_total, "n_observed": n_total - n_imputed,
                          "n_imputed": n_imputed, "n_unresolved": 0,
                          "mask_rule": "cells equal to the row minimum of the processed matrix "
                                       "(half-minimum identity); the pre-imputation matrix is "
                                       "not distributed",
                          "evidence_path": INPUTS["matrix"],
                          "note": "derived support counts, compared with the frozen table"})
    recomputed_mask = pd.DataFrame(mask_rows, columns=list(frozen_mask.columns))
    recomputed_mask.to_csv(out_dir / "hela_candidate_mask_counts_recomputed.tsv", sep="\t", index=False)
    ledger.add_table("hela_candidate_mask_counts", "pre-imputation observation support per candidate",
                     data_dir / TARGETS["mask_counts"],
                     recomputed_mask[SPEC_MASK_COUNTS.field_names()], SPEC_MASK_COUNTS,
                     evidence="derived from the half-minimum identity, validated by the recorded "
                              "116,904 filled values and the TBC1D10B flags",
                     notes="derived, not read from ProteinQuant_Filtered.csv (not distributed); "
                           "the prose columns mask_rule/evidence_path/note are regenerated and "
                           "excluded from the comparison")

    # ---- 7. input immutability ----------------------------------------------
    hashes_after = {rel: sha256_file(data_dir / rel) for rel in wanted.values()}
    changed = [rel for rel in wanted.values() if hashes_before[rel] != hashes_after[rel]]
    ledger.add("hela_inputs_read_only", "archive inputs unchanged by the run", "all inputs",
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
            "The principal components were produced historically with sklearn's randomized "
            "solver without a recorded seed, so they are reproducible only up to that solver "
            "non-determinism; the check is sign-aligned coordinate agreement plus the variance "
            "ratio at 1e-7.",
            "Observation-support counts are reconstructed by value identity from the processed "
            "matrix because the pre-imputation matrix (ProteinQuant_Filtered.csv) is not part of "
            "the archive; the reconstruction reproduces the recorded 116,904 filled values, the "
            "TBC1D10B flags and all 18 frozen support rows.",
            "Every table comparison is driven by a TableSpec declared in this script before the "
            "run: key sets equal, the missing mask compared cell by cell, inf rejected, counts "
            "exact, continuous values at the pre-registered rtol 1e-7 with atol 0, P and q "
            "purely relative, and a rounding allowance only where a field declares "
            "format_decimals (checked against the raw text of the frozen file). Byte-level "
            "re-serialization is recorded per table, not required; the residuals of the "
            "full-precision columns are last-digit differences and stay below the pre-registered "
            "tolerance.",
            "The four statistics rows of hela_composition.tsv are a REPORTED transcription of "
            "the PiSPA_A1b agent report (the source column of the frozen file names it, and the "
            "emitting panel code records the same literals); the file is retained unchanged and "
            "those rows are not the recomputed truth. The odds-ratio row stores the report's own "
            "rendering (value 26, interval 7.41 to 91.2), in which 91.2 is the "
            "three-significant-digit rendering of the Woolf upper bound 91.18748229323833 and "
            "not its two-decimal rendering (91.19); that row is therefore separated from the "
            "row-by-row comparison of the frozen table by a declared row filter and is covered "
            "by hela_odds_ratio_computed (the stable table "
            "case_tables/pispa_hela/hela_cluster_odds_ratio.tsv compared at rtol 1e-7 with atol "
            "0 and no rounding allowance), by hela_odds_ratio_reported_transcription (the "
            "historical tokens as a documented rendering of the recomputed value) and by "
            "hela_odds_ratio_manuscript_display (the one-decimal values the manuscript prints, "
            "7.4 and 91.2). The other three statistics rows and the three cluster-composition "
            "rows are still compared row by row under their declared allowances, which were not "
            "widened.",
        ],
    }
    (out_dir / "case_regression.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(ledger.summary_line())
    return ledger.exit_code()


if __name__ == "__main__":
    sys.exit(main())
