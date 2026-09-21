# Case-study reproduction

## What this directory is

Five themes, one script each, and one input interface. Each script re-derives the numbers that the
manuscript reports for its case from a read-only data archive and compares them with the frozen
tables of the same archive, field by field, under a pre-registered schema:

| theme | script | manuscript anchor | re-derives |
|---|---|---|---|
| HeLa migration | `hela_migration/run_hela_migration.py` | Fig. 3, Supplementary Fig. 4 | matrix identity, cluster composition, the computed cluster odds-ratio table of Cluster 1 versus Cluster 2, principal components, the six fixed candidate contrasts, the migrated-versus-control comparison, observation support of the imputation |
| Liver zonation | `liver_zonation/run_liver_zonation.py` | Fig. 4, Supplementary Fig. 5 | 45 sites to 15 mouse-by-zone units, both matrix branches, three contrasts, paired one-sample tests, the two Benjamini-Hochberg families, unit matrices, support and entry counts |
| Brain states | `brain_states/run_brain_states.py` | Fig. 5a,b, Supplementary Fig. 6 | module membership, per-state marker summaries, module scores under both zero conventions, member coverage, missingness sensitivity |
| Membrane integrity | `membrane_integrity/run_membrane_integrity.py` | Fig. 5c,d, Supplementary Fig. 7a | preservation-by-cell-type strata, pooled and composition-standardised effects, observation support of four candidates, value identity with the delivered input |
| Hematopoiesis | `hematopoiesis/run_hematopoiesis.py` | Fig. 5e, Supplementary Fig. 7b,c | the 336-protein cell-level contrast, the 1,550-protein observation-level contrast, ten unit-level contrasts on donor labels, cell values of MPO and TALDO1, cross-run identity of GMP minus HSC |

## Running a theme

```bash
python reproduce/cases/hela_migration/run_hela_migration.py --data-dir <archive> --out-dir <out>/hela_migration
python reproduce/cases/liver_zonation/run_liver_zonation.py   --data-dir <archive> --out-dir <out>/liver_zonation
python reproduce/cases/brain_states/run_brain_states.py       --data-dir <archive> --out-dir <out>/brain_states
python reproduce/cases/membrane_integrity/run_membrane_integrity.py --data-dir <archive> --out-dir <out>/membrane_integrity
python reproduce/cases/hematopoiesis/run_hematopoiesis.py     --data-dir <archive> --out-dir <out>/hematopoiesis
```

`--data-dir` is the data layer root, the directory that holds `case_tables/`, `source_tables/`
and `inputs/` (in the submission package that is `data/`). Both flags are required: there is no
default, no environment fallback and no search of another directory. A missing input prints one
`FATAL: missing input <path>` line per path checked and exits with status 2 without writing
anything.

Exit codes: **0** every check passed, **1** a check failed, **2** a required input is missing
(nothing written), **3** partial - no failure, but at least one check is `BLOCKED` because a number
the manuscript reports cannot be checked with what the archive ships. A check reported as
`OUT_OF_SCOPE` asks for an artefact or comparison the manuscript does not use; it carries its reason
in the ledger and does not make the run partial.

The scripts need numpy and pandas only - no scipy, no R, no network and no model call - and write
only inside `--out-dir`: a recomputed copy of each compared table plus `case_regression.json`.
Running `python tests/test_case_regression.py` executes all five and checks the ledger; without an
archive it prints `SKIP` and exits 0, and the archive can be pointed at with the environment
variable `SCPROTEOMICS_CASE_ARCHIVE`.

## How the comparison is done

Every comparison goes through `table_compare.py`, driven by a schema declared in the case script
itself. The schema fixes, per table: the key fields, one entry per compared column (kind
`identity`, `text`, `int`, `flag`, `float` or `pvalue`), which columns are deliberately excluded
and why, and where a rounding allowance is authorised.

* **Key set.** The key set of the frozen table and of the recomputed frame must be equal. Missing
  keys, extra keys and duplicate keys are listed by name and fail the comparison; the rows are then
  matched by key, so a different row order is legal and is recorded.
* **Missing values.** The missing mask is compared first. A value present on one side and missing
  on the other always fails; a value missing on both sides passes only where the column declares
  `allows_missing`. `inf` and `-inf` fail in every numeric column.
* **Tolerances.** Counts, sample sizes, family sizes, states, ranks and labels compare exactly
  (`3.0` equals the count `3`; `3.1` does not). Continuous columns use a pre-registered relative
  tolerance of **1e-7** plus an explicit absolute tolerance, and P/q columns a purely relative one,
  so a very small P is never treated as zero. A rounding allowance exists only where the frozen
  file's raw text and the writing format justify the declared number of decimals, and the
  comparator fails the declaration itself if the frozen file stores more decimals than declared.
* **Column provenance.** Every compared column is labelled `RECOMPUTED` (derived from the archived
  inputs), `COPIED_REFERENCE` (read from another frozen table) or `IDENTITY_CHECK`; the ledger and
  the printed summary carry the roll-up, so a copied value is never presented as a recomputation.

Each comparison also re-serialises the frame in the frozen column order and reports whether the
bytes are identical. Byte equality is only reached where the frozen file uses the default float
format; elsewhere the ledger records the maximum absolute and relative difference instead. No
threshold, candidate, unit, contrast direction or correction family is ever adjusted to make a
comparison pass: a mismatch is reported as `FAIL` with the field, the key and both values, and a
check that fails stays failing.

## What can be re-run and what cannot

Re-runnable offline, exactly as listed above: the derived quantities of the five cases, including
the Welch and paired t tests, the Benjamini-Hochberg families, the composition standardisation,
the two hematopoietic layers, and the principal components up to the solver non-determinism below.
The hematopoietic unit layer reproduces through the historical measurement path (the per-unit mean
is built with the same pandas group-by the historical implementation used) and the historical
convention for a protein that is constant across the paired units (tested with P = 1 and kept
inside the family), so its six per-protein result tables, which are part of the archive, are
compared row by row. The per-check result of the last verification run, and which checks are
`BLOCKED` or `OUT_OF_SCOPE`, are in the component status table of `docs/REPRODUCTION.md`.

Not re-run here, and why:

* The original agent runs. They require the user's own model credentials and are not part of this
  repository; the archived reports and the frozen scorer cover that layer.
* The hidden benchmark references. `evaluation_references/` stays out of every case script.
* The principal components, exactly. The historical PCA used a randomized solver without a
  recorded seed, so the coordinates are reproducible only up to that solver non-determinism: the
  check requires sign alignment, a correlation of at least 1 - 1e-9 and a maximum relative
  difference of 1e-4, and it separately reproduces the explained-variance ratio at 1e-7.
* The HeLa pre-imputation matrix. `ProteinQuant_Filtered.csv` is not part of the archive. The
  observation-support counts are reconstructed from the half-minimum identity of the processed
  matrix; the reconstruction reproduces the recorded 116,904 filled values, the TBC1D10B flags and
  all eighteen frozen support rows.
* The HeLa migration odds ratio, as two objects. The archive keeps the agent report's
  transcription of it (`source_tables/hela_composition.tsv`, value 26, interval 7.41 to 91.2)
  exactly as written; `case_tables/pispa_hela/hela_cluster_odds_ratio.tsv` is the computed object
  of the same statistic, whose Woolf log-odds interval is 7.413298218127537 to 91.18748229323833
  and agrees with an independent recomputation of the same formula to machine precision (2.40e-16
  and 1.56e-16 relative on the two endpoints, two and one unit in the last place). The stored
  91.2 is the three-significant-digit rendering of the upper bound, not its two-decimal
  rendering (91.19), and the manuscript prints the interval to one decimal (7.4-91.2). The case
  script compares the computed table at the pre-registered 1e-7 with no rounding allowance,
  reports the historical row as a REPORTED_TRANSCRIPTION of the agent report rather than as the
  recomputed truth, and checks the one-decimal display separately; that transcription row is
  separated from the strict row-by-row comparison of `hela_composition.tsv` and no tolerance was
  widened.
* The cross-run log-fold-change agreement of the nine hematopoietic contrasts other than GMP
  versus HSC. It needs the per-contrast differential tables of both runs; the archive holds only
  the GMP-versus-HSC tables, and the manuscript and its supplementary information do not cite that
  comparison, so the check is reported as `OUT_OF_SCOPE` with the paths that were searched. The
  verifiable part of the same frozen table - the row counts and the common-protein count of all ten
  contrasts, and the GMP-versus-HSC row itself - is checked.
* The airway transform record. The historical run directory contains no
  `matrix_transform_record.json`; the same statements (no imputation, no further transform, no new
  batch correction) are verified by comparing the analysed matrix with the delivered input, where
  all 2,806,785 compared values are identical.

Each script lists these limits in the `limitations` field of its `case_regression.json` and reports
the affected checks as `BLOCKED` or `OUT_OF_SCOPE` with the paths it looked for.

## Method boundaries that the scripts keep

* The liver chain is Python. The initial branch of that study was a two-sided Welch test with
  within-contrast BH; the continuation reported here tests within-mouse differences with a paired
  one-sample t test, uncorrected 95 percent intervals and two BH families (within each contrast,
  and across the three contrasts). R/limma is a tool-library backend and is not used in this chain;
  ComBat is available but this batch applies no new ComBat correction. The 5,577 result rows are
  three contrasts of 1,859 proteins, not 5,577 independent tests.
* The brain zeros are values. The value convention and the missing convention are computed and
  reported separately, neither is presented as the correct reading, and the unlabelled cells stay
  their own group.
* The airway matrix receives no further transform and no imputation, `Permeable` is a label of the
  source study rather than a treatment applied here, and the pooled and standardised effects are
  descriptive without intervals.
* The hematopoietic cell-level and unit-level runs differ in filtering, aggregation and testing;
  they are never merged into one estimate, and unit-level contrasts with fewer than three shared
  donor labels carry no P value.

## Data layout expected by the scripts

```
<archive>/
  case_tables/pispa_hela/run_processed/        ProQuant_Normalized.csv, SampleInfo_Filtered.csv,
                                               matrix_transform_record.json, qc_metrics.csv,
                                               limma_summary.json, Protein_Gene_Map.csv
  case_tables/pispa_hela/hela_cluster_odds_ratio.tsv
                                               the computed cluster odds-ratio table of the
                                               HeLa case
  case_tables/dvp_liver/locked_matrix/         ProteinQuant_ComBat.csv
  case_tables/brain/run_processed/             ProQuant_Normalized.csv, SampleInfo_Filtered.csv
  case_tables/brain/                           e06_brain_module_members.tsv and the frozen
                                               e06_brain_* comparison tables
  case_tables/airway_proteinleakage/run_processed/
                                               ProQuant_Normalized.csv, SampleInfo_Filtered.csv,
                                               Protein_Gene_Map.csv
  case_tables/bloodcell/cell_level_run/        ProQuant_Normalized.csv, SampleInfo_Filtered.csv,
                                               differential_GMP_vs_HSC.csv
  case_tables/bloodcell/unit_level_run/        ProQuant_Normalized.csv, SampleInfo_Filtered.csv,
                                               differential_GMP_vs_HSC.csv and the six
                                               unit_sensitivity_<contrast>.csv per-protein tables
  inputs/Nat_Commun_PiSPA_2024/ProteinQuant.csv
  inputs/Nat_Commun_ProteinLeakage_2025/ProteinQuant.csv
  source_tables/                               the frozen topic tables of the manuscript
```

If a case table disagrees with the manuscript, the manuscript and its Source Data sheet remain the
authority: report the disagreement instead of editing either side.

The archive's own `RIGHTS_STATUS.tsv` lists every file these scripts read, with its rights status,
access route and the author decision that covers it - most case inputs are
`THIRD_PARTY_UNCONFIRMED`, because redistribution is an open author decision - and
`docs/DATA_MANIFEST.md` describes the layout and how to obtain what is not in the archive.
