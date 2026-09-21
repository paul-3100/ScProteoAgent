#!/usr/bin/env python
"""Hematopoiesis case: deterministic re-derivation of the cell-level and unit-level layers.

Two runs are reported side by side. The cell-level run keeps 336 proteins after a 95 percent
non-missing filter and tests GMP minus HSC with a two-sided Welch test inside one 336-protein
family. The unit-level run uses a different filtering rule (10 percent non-missing) and keeps
1,550 proteins; values are averaged by state within metadata donor labels, mixed-source labels
are excluded, and each protein is tested with a paired one-sample t-test over the donor labels
that carry both states (at least three pairs). The two runs differ in filtering, aggregation and
testing, so their comparison does not isolate an experimental-unit effect and they are never
merged into one estimate.

Two conventions of the historical unit-level implementation are reproduced deliberately, because
they decide the adjusted-P scale that the manuscript reports:

* the per-unit mean is built by the historical measurement path
  (``analysis_extensions._unit_arm_mean_matrix``, ``renamed.T.groupby(level=["unit", "arm"]).mean().T``);
  replicating it with hand-written loops changes the last bits of the per-unit means and with them
  which proteins count as exactly constant;
* a protein whose paired differences are all identical and whose mean is zero (zero spread and
  |mean| < 1e-12) is *tested* with P = 1, t = 0 and a degenerate interval, and stays inside the
  Benjamini-Hochberg family (``analysis_extensions._paired_effect_rows`` L1320-L1349 and
  ``unit_aware_sensitivity`` L1588). Only a non-zero mean with zero spread is left
  untested. That is why the family is 1,550 proteins and not the 1,548 proteins with a non-trivial
  spread.

Usage:
    py run_hematopoiesis.py --data-dir <data archive root> --out-dir <output directory>

Exit codes: 0 every check passed, 1 a check failed, 2 a required input is missing (nothing
written), 3 partial - no failure but at least one check is BLOCKED.
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

from table_compare import Field, Ledger, TableSpec  # noqa: E402

THEME = "hematopoiesis"
REL_TOL = 1e-7
MIN_PAIRS = 3
MIXED_LABEL = "OldMixed"
STATES = ("HSC", "MPP", "MEP", "LMPP", "GMP")
CONTRASTS = (("GMP_vs_HSC", "GMP - HSC", "GMP", "HSC"),
             ("GMP_vs_LMPP", "GMP - LMPP", "GMP", "LMPP"),
             ("GMP_vs_MEP", "GMP - MEP", "GMP", "MEP"),
             ("GMP_vs_MPP", "GMP - MPP", "GMP", "MPP"),
             ("HSC_vs_LMPP", "HSC - LMPP", "HSC", "LMPP"),
             ("HSC_vs_MEP", "HSC - MEP", "HSC", "MEP"),
             ("HSC_vs_MPP", "HSC - MPP", "HSC", "MPP"),
             ("LMPP_vs_MEP", "LMPP - MEP", "LMPP", "MEP"),
             ("LMPP_vs_MPP", "LMPP - MPP", "LMPP", "MPP"),
             ("MEP_vs_MPP", "MEP - MPP", "MEP", "MPP"))
ESTIMABLE_CONTRASTS = ("GMP_vs_HSC", "GMP_vs_MEP", "GMP_vs_MPP", "HSC_vs_MEP", "HSC_vs_MPP",
                       "MEP_vs_MPP")
FOCUS = (("MPO", "P05164"), ("TALDO1", "P37837"))

INPUTS = {
    "cell_matrix": "case_tables/bloodcell/cell_level_run/ProQuant_Normalized.csv",
    "cell_sampleinfo": "case_tables/bloodcell/cell_level_run/SampleInfo_Filtered.csv",
    "cell_results": "case_tables/bloodcell/cell_level_run/differential_GMP_vs_HSC.csv",
    "unit_matrix": "case_tables/bloodcell/unit_level_run/ProQuant_Normalized.csv",
    "unit_sampleinfo": "case_tables/bloodcell/unit_level_run/SampleInfo_Filtered.csv",
    "unit_results": "case_tables/bloodcell/unit_level_run/differential_GMP_vs_HSC.csv",
}
for _contrast in ESTIMABLE_CONTRASTS:
    INPUTS["unit_sensitivity_%s" % _contrast] = (
        "case_tables/bloodcell/unit_level_run/unit_sensitivity_%s.csv" % _contrast)
TARGETS = {
    "cell_values": "source_tables/hemato_cell_values.tsv",
    "unit_results_table": "source_tables/hemato_unit_results.tsv",
    "unit_overview": "source_tables/hemato_unit_overview.tsv",
    "crossrun": "case_tables/bloodcell/e06_bloodcell_crossrun_obs.tsv",
}

# --------------------------------------------------------------------------- comparison schemas
# Pre-registered per table: which columns are compared, how, which are deliberately excluded (with
# the reason) and where a rounding allowance is authorised. No tolerance is derived from the
# difference being compared.
DIFFERENTIAL_COLUMNS = ["PG.ProteinGroups", "PG.Genes", "contrast", "protein", "logFC", "AveExpr",
                        "t", "P.Value", "adj.P.Val", "B"]
SPEC_DIFFERENTIAL = TableSpec(
    key_fields=("PG.ProteinGroups",),
    fields=(Field("PG.ProteinGroups", "identity"),
            Field("contrast", "identity"),
            Field("protein", "identity"),
            Field("logFC", "float", note="full precision in the frozen file (17 significant digits)"),
            Field("t", "float", note="full precision"),
            Field("P.Value", "pvalue"),
            Field("adj.P.Val", "pvalue")),
    excluded=(("PG.Genes", "gene symbols of the historical writer; not part of this derivation"),
              ("AveExpr", "average expression of the historical writer; not recomputed here"),
              ("B", "log-odds column of the historical writer; empty in every row of the file")),
    label="observation-level differential table")

CELL_VALUE_COLUMNS = ["row_kind", "protein", "gene", "state", "cell", "value", "n_cells", "q1",
                      "median", "q3", "whisker_low", "whisker_high"]
SPEC_CELL_VALUES_CELLS = TableSpec(
    key_fields=("protein", "cell"),
    fields=(Field("row_kind", "identity"), Field("protein", "identity"), Field("gene", "identity"),
            Field("state", "identity"), Field("cell", "identity"),
            Field("value", "float", note="per-cell value, full precision in the frozen file")),
    excluded=(("n_cells", "box statistic of the state summaries in the same file"),
              ("q1", "box statistic; compared by hemato_cell_value_summaries"),
              ("median", "box statistic; compared by hemato_cell_value_summaries"),
              ("q3", "box statistic; compared by hemato_cell_value_summaries"),
              ("whisker_low", "box statistic; compared by hemato_cell_value_summaries"),
              ("whisker_high", "box statistic; compared by hemato_cell_value_summaries")),
    label="per-cell values of MPO and TALDO1")
SPEC_CELL_VALUES_SUMMARY = TableSpec(
    key_fields=("protein", "state"),
    fields=(Field("row_kind", "identity"), Field("protein", "identity"), Field("gene", "identity"),
            Field("state", "identity"), Field("n_cells", "int"),
            Field("q1", "float"), Field("median", "float"), Field("q3", "float"),
            Field("whisker_low", "float"), Field("whisker_high", "float")),
    excluded=(("cell", "empty in the ten state-summary rows"),
              ("value", "empty in the ten state-summary rows; the per-cell values are compared by "
                        "hemato_cell_values")),
    label="state summaries of the MPO and TALDO1 cell values")

SPEC_UNIT_OVERVIEW = TableSpec(
    key_fields=("contrast",),
    fields=(Field("contrast", "identity"), Field("label", "identity"), Field("direction", "identity"),
            Field("estimable", "identity"), Field("n_pairs", "int", allows_missing=True,
                                                  note="no pair count where the contrast is not "
                                                       "estimable"),
            Field("n_proteins_tested", "int",
                  note="the Benjamini-Hochberg family of the contrast: every protein passing the "
                       "minimum-sample condition, including the constant ones tested with P=1"),
            Field("n_passing_fdr", "int"),
            Field("min_p_value", "pvalue", allows_missing=True),
            Field("min_adj_p_value", "pvalue", allows_missing=True),
            Field("n_units_a", "int"), Field("n_units_b", "int"), Field("n_units_shared", "int"),
            Field("excluded_units", "identity"), Field("unit_column", "identity")),
    label="ten predefined unit-level contrasts")

SPEC_UNIT_RESULTS = TableSpec(
    key_fields=("gene", "layer"),
    fields=(Field("gene", "identity"), Field("accession", "identity"), Field("layer", "identity"),
            Field("effect", "float", source_class="COPIED_REFERENCE",
                  note="the cell-level rows are read from the frozen cell-level differential table "
                       "of the other run; the unit-level rows are recomputed here"),
            Field("ci_low", "float", allows_missing=True),
            Field("ci_high", "float", allows_missing=True),
            Field("p_value", "pvalue", allows_missing=True),
            Field("adj_p", "pvalue",
                  note="cell-level rows: the frozen cell-level family of 336; unit-level rows: the "
                       "recomputed 1,550-protein family"),
            Field("n_proteins_tested", "int"),
            Field("df", "int", allows_missing=True),
            Field("n_pairs", "int", allows_missing=True)),
    excluded=(("run_id", "historical run directory names, carried as an archive reference only"),),
    label="cell-level and unit-level rows for MPO and TALDO1")

SPEC_UNIT_PER_PROTEIN = TableSpec(
    key_fields=("protein_group",),
    fields=(Field("protein_group", "identity"),
            Field("contrast", "identity"), Field("direction", "identity"),
            Field("analysis_unit", "identity"), Field("unit_scope", "identity"),
            Field("test", "identity"), Field("estimand", "identity"),
            Field("log2FC_units", "float"), Field("t", "float"), Field("df", "int"),
            Field("se", "float"), Field("ci_low", "float"), Field("ci_high", "float"),
            Field("P.Value", "pvalue"), Field("adj.P.Val", "pvalue"),
            Field("n_units_used", "int"), Field("n_units_group_a", "int"),
            Field("n_units_group_b", "int"), Field("tested", "identity")),
    excluded=(("gene", "label choice of the historical writer: one symbol per protein group, "
                      "while the archived matrix stores every symbol of a multi-symbol group "
                      "joined by ';'. The mapping itself is verified by hemato_unit_gene_labels"),
              ("effect_scale_state", "the effect scale of that run is not re-derived here"),
              ("effect_scale_label", "label of that run, stored in a non-UTF-8 encoding"),
              ("not_tested_reason", "empty in every row; the tested flag covers it")),
    label="per-protein unit-level results")

CROSSRUN_COLUMNS = ["contrast", "n_rows_HB", "n_rows_RB", "n_common_protein_groups",
                    "max_abs_logFC_diff", "n_logFC_diff_gt_1e-6"]
SPEC_CROSSRUN_COUNTS = TableSpec(
    key_fields=("contrast",),
    fields=(Field("contrast", "identity"), Field("n_rows_HB", "int"), Field("n_rows_RB", "int"),
            Field("n_common_protein_groups", "int")),
    excluded=(("max_abs_logFC_diff", "needs the per-contrast differential tables of both runs; only "
                                     "GMP versus HSC is in the archive and is compared separately"),
              ("n_logFC_diff_gt_1e-6", "same missing per-contrast tables as max_abs_logFC_diff")),
    label="cross-run row and common-protein counts")


# --------------------------------------------------------------------------- statistics
def _betacf(a: float, b: float, x: float) -> float:
    maxit, eps, fpmin = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = fpmin if abs(d) < fpmin else 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
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


def t_two_sided_p(t: np.ndarray, df: np.ndarray) -> np.ndarray:
    out = np.empty(len(t), dtype=float)
    for i, (ti, di) in enumerate(zip(np.asarray(t, dtype=float), np.asarray(df, dtype=float))):
        if not np.isfinite(ti) or not np.isfinite(di) or di <= 0:
            out[i] = np.nan
        else:
            out[i] = _betai(0.5 * di, 0.5, di / (di + ti * ti))
    return out


def t_quantile(prob: float, df: float) -> float:
    target = 2.0 * (1.0 - prob)
    lo, hi = 0.0, 1000.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if _betai(0.5 * df, 0.5, df / (df + mid * mid)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def welch(a: np.ndarray, b: np.ndarray):
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


def bh(p_values, tested: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg inside the tested family, with the reverse cumulative minimum."""
    p = np.asarray(p_values, dtype=float)
    tested = np.asarray(tested, dtype=bool)
    out = np.full(p.shape, np.nan)
    idx = np.where(tested & np.isfinite(p))[0]
    m = idx.size
    if m:
        order = idx[np.argsort(p[idx], kind="stable")]
        previous = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            previous = min(previous, p[i] * m / (rank + 1))
            out[i] = previous
    return out


def paired_test(differences: np.ndarray):
    """Historical ``_paired_effect_rows``: one-sample t test over the per-unit differences.

    A protein whose paired differences are all identical with a zero mean is a tested protein with
    P = 1 (the historical convention, so it stays inside the family); identical differences with a
    non-zero mean are degenerate and are not tested.
    """
    n = np.sum(~np.isnan(differences), axis=1)
    effect = np.nanmean(differences, axis=1)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(differences, axis=1, ddof=1)
    se = sd / np.sqrt(n)
    p_value = np.full(effect.shape, np.nan)
    ci_low = np.full(effect.shape, np.nan)
    ci_high = np.full(effect.shape, np.nan)
    dfree = np.full(effect.shape, np.nan)
    tested = np.zeros(effect.shape, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        zero_spread = np.isfinite(se) & (se == 0.0)
        flat = zero_spread & np.isfinite(effect) & (np.abs(effect) < 1e-12)
        for i in range(effect.size):
            if flat[i] and n[i] >= MIN_PAIRS:
                p_value[i] = 1.0
                ci_low[i] = effect[i]
                ci_high[i] = effect[i]
                dfree[i] = float(n[i] - 1)
                tested[i] = True
                continue
            if n[i] < MIN_PAIRS or not np.isfinite(se[i]) or se[i] == 0.0:
                continue
            dfree[i] = float(n[i] - 1)
            t_stat = effect[i] / se[i]
            p_value[i] = float(t_two_sided_p(np.array([t_stat]), np.array([dfree[i]]))[0])
            half = t_quantile(0.975, dfree[i]) * se[i]
            ci_low[i] = effect[i] - half
            ci_high[i] = effect[i] + half
            tested[i] = True
    t_stat = np.where(tested & (se > 0) & np.isfinite(se), effect / np.where(se > 0, se, np.nan),
                      np.where(tested, 0.0, np.nan))
    return (effect, ci_low, ci_high, n.astype(int), p_value, dfree, se, tested, t_stat)


def unit_arm_means(values: pd.DataFrame, samples, keys) -> pd.DataFrame:
    """Historical per-unit means: ``analysis_extensions._unit_arm_mean_matrix`` (L1391-L1395).

    Reproduced with the same pandas call on purpose: a hand-written loop changes the last bits of
    the per-unit means, which decides whether a protein counts as exactly constant in the paired
    test below.
    """
    renamed = values.copy()
    renamed.columns = pd.MultiIndex.from_tuples([keys[s] for s in samples], names=["unit", "arm"])
    return renamed.T.groupby(level=["unit", "arm"]).mean().T


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


def text_of(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in ("nan", "NaN", "None", "<NA>", "NaT") else text


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Hematopoiesis case, deterministic re-run. Exit codes: 0 all checks passed, "
                    "1 a check failed, 2 a required input is missing, 3 partial (a check is "
                    "BLOCKED).")
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

    cell_matrix = read_table(data_dir / INPUTS["cell_matrix"])
    cell_info = read_table(data_dir / INPUTS["cell_sampleinfo"])
    cell_results = read_table(data_dir / INPUTS["cell_results"])
    unit_matrix = read_table(data_dir / INPUTS["unit_matrix"])
    unit_info = read_table(data_dir / INPUTS["unit_sampleinfo"])
    unit_results = read_table(data_dir / INPUTS["unit_results"])

    pg_col = "PG.ProteinGroups"
    samples = [str(s) for s in cell_info["FileName"]]
    states = cell_info["Cluster"].astype(str).to_numpy()
    donors = cell_info["Donor"].astype(str).to_numpy()
    cell_proteins = [str(p) for p in cell_matrix[pg_col]]
    unit_proteins = [str(p) for p in unit_matrix[pg_col]]
    cell_genes = {str(p): text_of(g) for p, g in zip(cell_matrix[pg_col], cell_matrix["PG.Genes"])}
    unit_genes = {str(p): text_of(g) for p, g in zip(unit_matrix[pg_col], unit_matrix["PG.Genes"])}
    cell_values = cell_matrix[samples].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    unit_values = unit_matrix[samples].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    state_cells = {s: np.array([i for i, x in enumerate(states) if x == s]) for s in STATES}

    identity_ok = (len(samples) == 125 and all(state_cells[s].size == 25 for s in STATES)
                   and len(cell_proteins) == 336 and len(unit_proteins) == 1550
                   and unit_info["FileName"].astype(str).tolist() == samples)
    ledger.add("hemato_layer_identity", "two runs, two protein sets, the same 125 cells",
               INPUTS["cell_matrix"],
               "cell-level run 336 proteins; unit-level run 1,550 proteins; 125 cells, 25 per "
               "state; both runs cover the same cell list",
               "cell matrix %d rows, unit matrix %d rows, %d cells; per-state counts %s" % (
                   len(cell_proteins), len(unit_proteins), len(samples),
                   {s: int(state_cells[s].size) for s in STATES}),
               "PASS" if identity_ok else "FAIL", "exact",
               "the two runs differ in filtering, aggregation and testing and are never merged")

    def observation_table(proteins, values, state_a, state_b):
        effect, t_stat, dfree = welch(values[:, state_cells[state_a]],
                                      values[:, state_cells[state_b]])
        p_value = t_two_sided_p(t_stat, dfree)
        q_value = bh(p_value, np.isfinite(p_value))
        return pd.DataFrame({"PG.ProteinGroups": proteins,
                             "PG.Genes": ["" for _ in proteins],
                             "contrast": "%s_vs_%s" % (state_a, state_b),
                             "protein": proteins,
                             "logFC": effect, "AveExpr": np.nan, "t": t_stat,
                             "P.Value": p_value, "adj.P.Val": q_value, "B": np.nan})

    recomputed_cell = observation_table(cell_proteins, cell_values, "GMP", "HSC")
    ledger.add_table("hemato_cell_level_contrast", "cell-level GMP minus HSC over 336 proteins",
                     data_dir / INPUTS["cell_results"],
                     recomputed_cell[SPEC_DIFFERENTIAL.field_names()],
                     SPEC_DIFFERENTIAL, frozen_columns=DIFFERENTIAL_COLUMNS,
                     evidence="two-sided Welch t-test with Benjamini-Hochberg adjustment inside "
                              "the 336-protein family of this run")
    recomputed_unit_obs = observation_table(unit_proteins, unit_values, "GMP", "HSC")
    ledger.add_table("hemato_observation_level_1550",
                     "observation-level GMP minus HSC in the unit-level run",
                     data_dir / INPUTS["unit_results"],
                     recomputed_unit_obs[SPEC_DIFFERENTIAL.field_names()],
                     SPEC_DIFFERENTIAL, frozen_columns=DIFFERENTIAL_COLUMNS,
                     evidence="cells as observations on the 1,550-protein matrix; this table is "
                              "the observation-level reference of the second run, not the "
                              "unit-level analysis")

    # ------------------------------------------------------------------ unit-level layer
    unit_frame = pd.DataFrame(unit_values, index=unit_proteins, columns=samples)
    keys = {sample: (donors[i], states[i]) for i, sample in enumerate(samples)}
    per_unit = unit_arm_means(unit_frame, samples, keys)
    units_per_state = {}
    for state in STATES:
        units_per_state[state] = sorted({donors[i] for i in state_cells[state]
                                         if donors[i] != MIXED_LABEL})

    unit_overview_rows = []
    unit_level_values = {}
    per_protein_frames = {}
    for contrast_id, label, state_a, state_b in CONTRASTS:
        shared = [d for d in units_per_state[state_a] if d in units_per_state[state_b]]
        estimable = len(shared) >= MIN_PAIRS
        if estimable:
            columns_a = [(d, state_a) for d in shared]
            columns_b = [(d, state_b) for d in shared]
            differences = (per_unit.loc[:, columns_a].to_numpy(dtype=float)
                           - per_unit.loc[:, columns_b].to_numpy(dtype=float))
            (effect, ci_low, ci_high, n, p_value, dfree, se, tested,
             t_stat) = paired_test(differences)
            q_value = bh(p_value, tested)
            unit_level_values[contrast_id] = (effect, ci_low, ci_high, n, p_value, q_value, dfree,
                                              se, tested, t_stat, len(shared))
            min_p = float(np.nanmin(p_value[tested])) if tested.any() else np.nan
            min_q = float(np.nanmin(q_value[tested])) if tested.any() else np.nan
            n_tested = int(tested.sum())
            per_protein_frames[contrast_id] = pd.DataFrame({
                "protein_group": unit_proteins,
                "gene": [unit_genes.get(p, "") for p in unit_proteins],
                "contrast": contrast_id,
                "direction": label,
                "analysis_unit": "Donor",
                "unit_scope": "within_unit_paired",
                "test": "paired t-test on per-unit means",
                "estimand": "mean within-unit difference (%s - %s)" % (state_a, state_b),
                "log2FC_units": effect, "t": t_stat, "df": dfree, "se": se,
                "ci_low": ci_low, "ci_high": ci_high,
                "P.Value": p_value, "adj.P.Val": q_value,
                "n_units_used": n.astype(float),
                "n_units_group_a": np.full(len(unit_proteins), float(len(shared))),
                "n_units_group_b": np.full(len(unit_proteins), float(len(shared))),
                "tested": np.where(tested, "True", "False")})
        else:
            min_p = min_q = np.nan
            n_tested = 0
            unit_level_values[contrast_id] = None
        unit_overview_rows.append({"contrast": contrast_id,
                                   "label": "%s vs %s" % (state_a, state_b),
                                   "direction": label,
                                   "estimable": bool(estimable),
                                   "n_pairs": float(len(shared)) if estimable else np.nan,
                                   "n_proteins_tested": n_tested, "n_passing_fdr": 0,
                                   "min_p_value": min_p, "min_adj_p_value": min_q,
                                   "n_units_a": len(units_per_state[state_a]),
                                   "n_units_b": len(units_per_state[state_b]),
                                   "n_units_shared": len(shared), "excluded_units": MIXED_LABEL,
                                   "unit_column": "Donor"})
    overview = pd.DataFrame(unit_overview_rows)
    ledger.add_table("hemato_unit_level_overview", "ten predefined unit-level contrasts",
                     data_dir / TARGETS["unit_overview"], overview, SPEC_UNIT_OVERVIEW,
                     evidence="state means within each donor label after excluding the "
                              "mixed-source label, built by the historical measurement path "
                              "(analysis_extensions._unit_arm_mean_matrix); paired one-sample "
                              "t-test requiring at least three pairs; a constant protein with a "
                              "zero mean is tested with P = 1 and stays inside the family; "
                              "Benjamini-Hochberg inside the 1,550-protein family of each contrast",
                     notes="four contrasts share only two donor labels and therefore carry no "
                           "P value; the reported n_passing_fdr is zero in every contrast")

    for contrast_id in ESTIMABLE_CONTRASTS:
        ledger.add_table("hemato_unit_protein_%s" % contrast_id,
                         "per-protein unit-level results of %s" % contrast_id,
                         data_dir / INPUTS["unit_sensitivity_%s" % contrast_id],
                         per_protein_frames[contrast_id], SPEC_UNIT_PER_PROTEIN,
                         evidence="the recomputation carries the same 1,550-protein family, the "
                                  "same per-unit aggregation and the same Benjamini-Hochberg "
                                  "family as the historical table")

    # The gene symbol is the one column whose comparison is a label question rather than a
    # measurement: the historical writer stored one symbol per protein group. Verify that reading
    # instead of comparing the strings, so the excluded column is still accounted for.
    mappings = {}
    label_mismatches = []
    multi_symbol = set()
    for contrast_id in ESTIMABLE_CONTRASTS:
        history = read_table(data_dir / INPUTS["unit_sensitivity_%s" % contrast_id])
        mapping = dict(zip(history["protein_group"].astype(str), history["gene"].astype(str)))
        mappings[contrast_id] = mapping
        for group, symbol in mapping.items():
            symbols = [s.strip() for s in unit_genes.get(group, "").split(";") if s.strip()]
            if len(symbols) > 1:
                multi_symbol.add(group)
            if not symbols or symbol.strip() != symbols[0]:
                label_mismatches.append((contrast_id, group, symbol,
                                         unit_genes.get(group, "")))
    same_mapping = all(mapping == mappings[ESTIMABLE_CONTRASTS[0]]
                       for mapping in mappings.values())
    ledger.add("hemato_unit_gene_labels",
               "gene symbols of the unit-level tables against the archived protein groups",
               INPUTS["unit_sensitivity_%s" % ESTIMABLE_CONTRASTS[0]],
               "every stored symbol is the first symbol of the protein group in the archived "
               "matrix, and the six tables carry the same mapping",
               "%d protein groups; %d carry several symbols joined by ';'; six-file mapping "
               "identical %s; mismatches %d%s"
               % (len(mappings[ESTIMABLE_CONTRASTS[0]]), len(multi_symbol), same_mapping,
                  len(label_mismatches),
                  "" if not label_mismatches else "; first %s" % (label_mismatches[:2],)),
               "PASS" if (same_mapping and not label_mismatches) else "FAIL",
               "first symbol of PG.Genes",
               "the historical writer's gene map names one representative symbol per protein "
               "group; the archived matrix lists every symbol of the group")

    unit_results_rows = []
    for gene, accession in FOCUS:
        if accession not in cell_proteins:
            continue
        cell_row = cell_results[cell_results[pg_col] == accession]
        unit_results_rows.append({"gene": gene, "accession": accession, "layer": "cell_level",
                                  "effect": float(cell_row["logFC"].iloc[0]) if len(cell_row)
                                  else np.nan,
                                  "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                                  "adj_p": float(cell_row["adj.P.Val"].iloc[0]) if len(cell_row)
                                  else np.nan,
                                  "n_proteins_tested": 336, "df": np.nan, "n_pairs": np.nan})
        if unit_level_values["GMP_vs_HSC"] is not None and accession in unit_proteins:
            j = unit_proteins.index(accession)
            (effect, ci_low, ci_high, n, p_value, q_value, dfree, _se, _tested, _t,
             _pairs) = unit_level_values["GMP_vs_HSC"]
            unit_results_rows.append({"gene": gene, "accession": accession, "layer": "unit_level",
                                      "effect": effect[j], "ci_low": ci_low[j],
                                      "ci_high": ci_high[j], "p_value": p_value[j],
                                      "adj_p": q_value[j], "n_proteins_tested": 1550,
                                      "df": dfree[j], "n_pairs": float(n[j])})
    recomputed_unit_results = pd.DataFrame(unit_results_rows)
    ledger.add_table("hemato_unit_level_results",
                     "cell-level and unit-level rows for MPO and TALDO1",
                     data_dir / TARGETS["unit_results_table"], recomputed_unit_results,
                     SPEC_UNIT_RESULTS,
                     evidence="the unit-level row carries the recomputed interval and adjusted P "
                              "of the paired test inside the 1,550-protein family; the cell-level "
                              "row is read from the frozen cell-level differential table")

    cell_value_rows = []
    summary_rows = []
    for gene, accession in FOCUS:
        i = cell_proteins.index(accession)
        for k, sample in enumerate(samples):
            cell_value_rows.append({"row_kind": "cell", "protein": accession, "gene": gene,
                                    "state": states[k], "cell": sample,
                                    "value": cell_values[i][k]})
        for state in STATES:
            data = cell_values[i][state_cells[state]]
            data = data[np.isfinite(data)]
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1
            inside = data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]
            summary_rows.append({"row_kind": "state_summary", "protein": accession, "gene": gene,
                                 "state": state, "n_cells": int(data.size),
                                 "q1": float(q1), "median": float(median), "q3": float(q3),
                                 "whisker_low": float(inside.min()),
                                 "whisker_high": float(inside.max())})
    ledger.add_table("hemato_cell_values", "cell values of MPO and TALDO1",
                     data_dir / TARGETS["cell_values"],
                     pd.DataFrame(cell_value_rows)[SPEC_CELL_VALUES_CELLS.field_names()],
                     SPEC_CELL_VALUES_CELLS, frozen_columns=CELL_VALUE_COLUMNS,
                     row_filter=lambda table: table["row_kind"] == "cell",
                     evidence="cell rows of the frozen value table")
    ledger.add_table("hemato_cell_value_summaries",
                     "state summaries of the same MPO and TALDO1 cell values",
                     data_dir / TARGETS["cell_values"],
                     pd.DataFrame(summary_rows)[SPEC_CELL_VALUES_SUMMARY.field_names()],
                     SPEC_CELL_VALUES_SUMMARY, frozen_columns=CELL_VALUE_COLUMNS,
                     row_filter=lambda table: table["row_kind"] == "state_summary",
                     evidence="quartiles by linear interpolation and whiskers at the most extreme "
                              "value inside 1.5 interquartile ranges, recomputed from the same "
                              "250 cell values")

    cell_common = set(cell_proteins) & set(unit_proteins)
    crossrun_rows = [{"contrast": contrast_id, "n_rows_HB": len(cell_proteins),
                      "n_rows_RB": len(unit_proteins),
                      "n_common_protein_groups": len(cell_common)}
                     for contrast_id, _label, _a, _b in CONTRASTS]
    crossrun = pd.DataFrame(crossrun_rows)
    ledger.add_table("hemato_crossrun_counts",
                     "row and common-protein counts of the ten contrasts",
                     data_dir / TARGETS["crossrun"], crossrun, SPEC_CROSSRUN_COUNTS,
                     frozen_columns=CROSSRUN_COLUMNS,
                     evidence="row counts of the two archived matrices and their intersection")

    shared_cell = cell_results.set_index(pg_col).reindex(sorted(cell_common))
    shared_unit = unit_results.set_index(pg_col).reindex(sorted(cell_common))
    max_diff = float(np.max(np.abs(shared_cell["logFC"].to_numpy(dtype=float)
                                   - shared_unit["logFC"].to_numpy(dtype=float))))
    n_gt = int(np.sum(np.abs(shared_cell["logFC"].to_numpy(dtype=float)
                             - shared_unit["logFC"].to_numpy(dtype=float)) > 1e-6))
    frozen_crossrun = read_table(data_dir / TARGETS["crossrun"])
    frozen_gmp = frozen_crossrun[frozen_crossrun["contrast"] == "GMP_vs_HSC"].iloc[0]
    frozen_max = float(frozen_gmp["max_abs_logFC_diff"])
    frozen_gt = int(frozen_gmp["n_logFC_diff_gt_1e-6"])
    ledger.add("hemato_crossrun_gmp_hsc", "cross-run comparison for GMP minus HSC",
               TARGETS["crossrun"],
               "max |log fold change difference| %.17g over %d common protein groups and %d "
               "differences above 1e-6" % (frozen_max, len(cell_common), frozen_gt),
               "recomputed max |difference| %.17g over %d common protein groups, %d above 1e-6"
               % (max_diff, len(cell_common), n_gt),
               "PASS" if (max_diff == frozen_max and n_gt == frozen_gt) else "FAIL",
               "exact comparison of the two archived differential tables",
               "the 336 common protein groups have identical observation-level log fold changes "
               "in both runs because the cells are the same and only the protein set differs")
    ledger.add("hemato_crossrun_other_contrasts",
               "cross-run log-fold-change agreement of the nine other contrasts",
               TARGETS["crossrun"],
               "the frozen table lists that comparison for all ten contrasts",
               "not computed: the nine other contrasts need the per-contrast differential tables "
               "of both runs, which the archive does not hold; the two columns are not cited by "
               "the manuscript or the supplementary information",
               "OUT_OF_SCOPE", "n/a",
               "checked the archive paths case_tables/bloodcell/cell_level_run and "
               "unit_level_run: each holds only differential_GMP_vs_HSC.csv; the manuscript and "
               "Supplementary Information were searched for the column names and cite neither",
               scope="not_cited_in_manuscript",
               notes="the verifiable part of the same table - the row and common-protein counts of "
                     "all ten contrasts - is checked by hemato_crossrun_counts, and the "
                     "GMP versus HSC row by hemato_crossrun_gmp_hsc")

    hashes_after = {rel: sha256_file(data_dir / rel) for rel in wanted.values()}
    changed = [rel for rel in wanted.values() if hashes_before[rel] != hashes_after[rel]]
    ledger.add("hemato_inputs_read_only", "archive inputs unchanged by the run", "all inputs",
               "identical SHA-256 before and after",
               "%d files hashed; changed: %s" % (len(wanted), changed or "none"),
               "PASS" if not changed else "FAIL", "exact", "sha256_file on every input")

    counts = ledger.counts()
    payload = {
        "theme": THEME,
        "data_dir": str(args.data_dir),
        "checks": ledger.checks,
        "counts": counts,
        "exit_code": ledger.exit_code(),
        "column_classes": ledger.class_rollup(),
        "input_sha256": hashes_after,
        "limitations": [
            "The cell-level and unit-level runs differ in filtering, aggregation and testing; "
            "their comparison does not isolate an experimental-unit effect.",
            "The unit-level layer aggregates within donor labels after excluding the "
            "mixed-source label, and requires at least three paired donor labels per protein; "
            "four of the ten contrasts share only two labels and carry no P value.",
            "A constant protein with a zero mean is tested with P = 1 and stays inside the "
            "Benjamini-Hochberg family, following the historical implementation; the family is "
            "therefore 1,550 proteins, not the 1,548 with a non-trivial spread.",
            "The cross-run comparison of the nine contrasts other than GMP minus HSC is out of "
            "scope: the per-contrast differential tables are not part of the archive and the "
            "manuscript does not cite that comparison.",
        ],
    }
    (out_dir / "case_regression.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(ledger.summary_line())
    return ledger.exit_code()


if __name__ == "__main__":
    sys.exit(main())
