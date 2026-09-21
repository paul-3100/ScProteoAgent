# Reproducing the reported analyses

Everything here runs offline and calls no model. The data archive that supplies the frozen inputs
is a local candidate of the same submission, not a published download; `docs/DATA_MANIFEST.md`
describes its layout and resolution order, and every command below is written against
`<archive>`, the archive root.

Every command runs with the interpreter of the environment installed above
(`.venv\Scripts\python` on Windows, `.venv/bin/python` on POSIX); the plain `python` of the
command lines means that interpreter. The verified versions are recorded in
`requirements-lock.txt`.

## Component status

Three separate questions per component, because "has been run" and "reproduces every reported
number" are different claims. `runnable` means the entry point runs offline against the archive;
`checked fields agree` means every field the component compares matched, at the tolerance the
component declares; `all manuscript fields reproduced` is true only when no manuscript-bearing
field is left unverified.

| component | runnable | checked fields agree | all manuscript fields reproduced | open items |
|---|---|---|---|---|
| cross-system scoring (`reproduce/scoring/`) | yes | yes: 88 of 88 study-system cells, `rule_total` and all six dimensions, maximum absolute difference 0, 88 of 88 archived reports hash-matched | yes | the replay needs the assembled scoring input tree; an explicit `--datasets` subset reports `SUBSET_PASS` with its coverage and never the 88-cell result |
| paired intervals (`reproduce/intervals/`) | yes | yes: all seven endpoints reproduce and the verifier's ten checks pass | yes | the endpoints are comparable only at the frozen seed, resample count and quantile method |
| GO background (`reproduce/go_background/`) | yes | yes: 12 families, 62,584 term rows, the historical gate 90 of 90, the fixed 24 display positions 15/5/0/4 and the verifier verdict PASS | yes | the three GMT files are third-party resources and are not redistributed; the arbitrary-precision layer covers a sample of the rows, not all of them |
| case study: HeLa migration | yes | yes: 18 of 18 checks pass, no FAIL, exit 0 | yes: the manuscript prints the interval as 7.4-91.2 and the computed interval at one decimal is exactly that | the historical transcription cell is retained and is checked by a separate provenance/display check (see `DATA_DICTIONARY.md` in the data archive) instead of by the strict table comparison |
| case study: liver zonation | yes | yes: 18 of 18 | yes | the 5,577 result rows are three contrasts of 1,859 proteins, not 5,577 independent tests |
| case study: brain states | yes | yes: 9 of 9 | yes | the source study leaves the meaning of the zeros open; both zero conventions are reported and neither is presented as the correct reading |
| case study: membrane integrity | yes | yes: 9 of 9 | yes | the historical run directory holds no transform record; the processing order is established from the executed statements and from the identity of the analysed matrix with the delivered input |
| case study: haematopoiesis | yes | yes: 17 of 17, plus one `OUT_OF_SCOPE` check | yes | the cell-level and unit-level runs differ in filtering, aggregation and testing and are never merged; the cross-run comparison of nine contrasts is not cited by the manuscript and is not reproduced |

**The HeLa odds-ratio interval: two objects, kept apart.** `hela_composition_rows` compares
the four rows of `source_tables/hela_composition.tsv` that the frozen table itself marks as
transcriptions from the agent report (its `source` column). The odds ratio 26.0, the proportion
difference, the chi-square and Cramer's V reproduce, and the lower limit 7.41 reproduces as a
rendering. Two objects are deliberately not merged:

* the **computed** object - the Woolf log-odds interval from the archived 2x2 counts (cluster 1:
  35 migrated / 7 control; cluster 2: 5 migrated / 26 control). It is stored at 17 significant
  digits in `case_tables/pispa_hela/hela_cluster_odds_ratio.tsv` (`ci_low` 7.4132982181275366,
  `ci_high` 91.187482293238332) and compared strictly, with no rounding allowance; an
  independent recomputation of the same formula agrees to machine precision (about 2e-16
  relative on both endpoints), which is not a digit-for-digit identity between implementations;
* the **historical reported transcription** in `source_tables/hela_composition.tsv` (value 26,
  `ci_low` 7.41, `ci_high` 91.2, from the PiSPA_A1b report). The cell is retained unchanged and
  is verified as a documented rendering of the computed value, not as an independent
  recomputation: 7.41 is the two-decimal rendering of the computed lower limit and 91.2 the
  three-significant-digit rendering of the computed upper limit (the two-decimal rendering of
  that upper limit is 91.19).

The manuscript prints 7.4-91.2, which is the computed interval at one decimal, so no reported
number changes. The strict table comparison covers the computed interval, and the transcription
is covered by the separate provenance/display check; nothing was widened, dropped or rounded to
make a check pass, and the historical table is not overwritten.

The scoring, interval and GO statuses above were established in the previous verification round on
byte-identical files: their inputs and outputs were hash-compared and are unchanged in this
revision, so they were not re-run.

## Task table

| manuscript item | command | expected output |
|---|---|---|
| cross-system scores and dimension contributions | `python reproduce/scoring/prepare_scoring_inputs.py --archive <archive> --out-dir <work>` then `python reproduce/scoring/run_replay.py --runs-root <work>/scoring/runs --examples-root <work>/examples --out-dir <work>/replay --expected-tsv <archive>/scores/HW_RULE_SCORES.tsv --check-report-sha256` | one row per (study, system) with six dimensions and `rule_total`; 88 unique keys; maximum absolute difference 0 against the frozen table; 88/88 archived reports hash-matched |
| paired differences and their seven intervals | `python reproduce/intervals/recompute_paired_ci.py --data-dir <archive> --out-dir <work>` then `python reproduce/intervals/ci_verify.py --data-dir <archive> --out-dir <work>` | `paired_ci_recomputed.tsv` at 17 significant digits, `paired_ci_indices.npy`, `environment.json`, and the verifier's ten checks |
| GO background comparison for the migration case | `python reproduce/go_background/c1_go_datasets.py --data-dir <archive>/go_background --gmt-dir <gmt> --out-dir <work>`, then `c2` -> `c3` -> `c4` with the same flags | six query sets, 12 families, 62,584 term rows, the historical reproduction gate, the fixed 24 display positions (15/5/0/4) and the verifier verdict |
| case-study computations (HeLa migration, liver zonation, brain states, membrane integrity, haematopoiesis) | `python reproduce/cases/<theme>/run_<theme>.py --data-dir <archive> --out-dir <work>` | a per-check ledger over the frozen tables of that theme; exit 0 all checks passed, 1 a check failed, 2 a required input is missing, 3 partial (a check is BLOCKED). See `reproduce/cases/README.md` |
| statistical figure panels | `python reproduce/figures/plot_panels.py --panel <id> --data-dir <archive> --out-dir <work>` | the panel as PNG/PDF/SVG plus a copy of the values used |
| input-format checks | `python scripts/validate_dataset_inputs.py <dataset-dir>` | human-readable summary, exit code 0 (clean) or 2 (problems found) |

## What reproduction means here, per item

**Scores.** `prepare_scoring_inputs.py` first assembles the **scoring input set** into a working
directory: a `scoring/runs/` tree with the archived reports and artefacts of the eight systems and
eleven studies, and an `examples/` tree with the dataset inputs plus the independent evaluation
references the scorer reads for every study. The assembly step fails if any required file is
missing instead of substituting an empty one, and the assembled directory must not be used as
generation input for the agent. `run_replay.py` then runs the frozen scorer
(`reproduce/scoring/score_full_run.py`, sha256 `2B37C1...A74B`, 782 lines) unchanged, one pass
per system, through `guarded_launcher.py`, which blocks socket operations, with the language-model
review layer disabled (`--skip-llm`). The comparison against the frozen rule table is a completeness
check before it is a numerical check: the expected and observed key sets must both contain exactly
the same 88 unique (study, system) pairs, duplicate keys or rows are an error, and every one of
`rule_total` and the six dimensions must be present and finite. Only then are values compared, at
tolerance 0 by default. An explicit `--datasets` subset runs an exact check of the projected key set
and reports `SUBSET_PASS` with its coverage instead of a full 88-cell pass.

**Intervals.** The algorithm is fixed by the frozen implementation: eleven studies as the pairing
unit, one `PCG64(20260919)` generator, exactly one `rng.integers(0, 11, size=(20000, 11))` call shared
by all seven comparisons, percentile limits at 2.5/97.5 with linear interpolation. `ci_verify.py`
recomputes everything in a fresh process and additionally replays the sampling indices with an
explicit loop as a spot check. Changing the seed, the resample count or the quantile method
produces different endpoints that must not be compared with the published values.

**GO background.** The chain is reproduced from frozen inputs and now writes only to its working
directory (`--out-dir`, default `<cwd>/go_background_out`); the archive is read-only. The verifier
re-derives P for the whole table with `scipy` and q with an independent implementation, and
additionally re-derives a sampled set of terms (113 in the delivered run) with a 60-digit mpmath
tail: the arbitrary-precision layer covers the sample, not all 62,584 rows. The GMT files are
third-party resources that are not redistributed; obtain a matching release as described in
`reproduce/go_background/RESOURCE_ACQUISITION.md`, which also records the verified release identity.

**Cases.** Each theme script recomputes the numbers of its case from the archived inputs and
compares them with the frozen tables field by field: HeLa migration (composition, cluster
contrasts, the 4,688-protein offline comparison, observation support), liver zonation (the 45
sites, the 15 mouse-by-zone units, three contrasts, both branches and both BH families), brain
states (the two zero conventions and the module scores), membrane integrity (membrane-state
composition and the composition-standardized effects) and haematopoiesis (cell-level and
unit-level results from two separate runs). The method descriptions match the manuscript's
Methods: the initial HeLa, liver and cell-level haematopoietic comparisons are two-sided Welch
tests with within-contrast BH, the mouse-level liver analysis uses paired one-sample t-tests over
mice with separate within-contrast and joint BH families, R/limma is an available back end of the
tool library rather than the implementation of these Python case branches, and ComBat is available
but no new ComBat correction was applied in them.

How a case comparison is decided is set by `reproduce/cases/table_compare.py`. Each table is
described by a pre-registered schema: which columns are compared, how each one is compared, which
columns are deliberately excluded (with the reason), and where a rounding allowance is authorised.
The key set of the frozen table and of the recomputed frame must be equal, missing keys, extra keys
and duplicate keys are reported by name, a value missing on one side always fails, `inf` fails,
counts and labels compare exactly, continuous values use a pre-registered relative tolerance and
P/q values a purely relative one. A rounding allowance exists only where the frozen file's own raw
text and the writing format justify it. Every check reports which of its columns were recomputed,
which were read from another frozen table and which are pure identity checks, so a copied value is
never presented as a recomputation.

Two kinds of check stay open by design. `BLOCKED` means a number the manuscript reports cannot be
checked with what the archive ships, and makes the run partial (exit code 3). `OUT_OF_SCOPE`
means the check asks for an artefact or comparison the manuscript does not use; it is reported with
the reason and does not make the run partial.

**Figures.** `reproduce/figures/plot_panels.py` redraws statistical panels from the frozen source
tables (PNG at 450 dpi, PDF and SVG with editable text). The assembled manuscript figures were
placed, lettered and cropped by hand in the authors' files; no script here claims to reproduce that
layout.

## Environment

    python -m venv .venv
    .venv/Scripts/python -m pip install -r requirements-reproduction.txt      # POSIX: .venv/bin/python

This installs numpy, pandas, scipy, statsmodels, mpmath and matplotlib; the scoring replay, the GO
steps 1 and 3, the validation scripts and the examples need less. The environment used for the checks,
including which interpreter ran what, is recorded in `VERIFIED_ENVIRONMENT.md`.

## Running the tests

    python tests/test_module_imports.py [--strict]
    python tests/test_frozen_scorer_contract.py
    python tests/test_replay_completeness.py
    python tests/test_pair_ci_determinism.py --data-dir <archive>/source_tables
    python tests/test_go_verifier_smoke.py --data-dir <archive>/go_background --gmt-dir <gmt>
    python tests/test_case_regression.py --data-dir <archive>
    python tests/test_example_input_validation.py
    python tests/test_no_credentials_no_network.py

Each test prints its own checks and returns a non-zero exit code on failure. A test that cannot run
because the archive or the GMT files are absent says so and reports `SMOKE_ONLY` or `BLOCKED`
instead of claiming a pass.

