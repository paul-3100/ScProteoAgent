# Paired confidence intervals

`recompute_paired_ci.py` and `ci_verify.py` are the frozen implementation of the seven paired
percentile-bootstrap intervals reported for the cross-system comparison (study = unit, eleven
studies, 20,000 resamples, one shared `PCG64(20260919)` index matrix, linear quantiles).

## Where the inputs live

Either the archive root or the `intervals/` subdirectory of the archive can be passed as
`--data-dir`: the inputs are looked for in the directory itself, in `tables/`, `inputs/`, and in the
documented source-table, score and interval directories of the archive layout.

The scripts read a data directory that is **not** part of this repository. In the submission
package it is `data/intervals/`, holding:

| file | archive name | role | required |
|---|---|---|---|
| `benchmark_rule_scores.tsv` | `rule_scores.tsv` | frozen 88-cell rule table (`mode == with_run_dir`) | yes |
| `benchmark_paired_difference_means.tsv` | `paired_means.tsv` | endpoints to compare the recomputed ones against | yes |
| `HW_RULE_SCORES.tsv` | same | independent history cross-check of the same 88 cells | optional |
| `CI_PROVENANCE.md` | same | provenance note hashed into `environment.json` | optional |

Either name is accepted. Each input is looked up in `<data-dir>`, `<data-dir>/tables`, `<data-dir>/inputs`,
and (for the archive layout of this release) in `<data-dir>/../source_tables`, `<data-dir>/../scores` and
`<data-dir>/../intervals`; that is what lets the defaults work both for a single-purpose directory and
for the archive of this submission, where the rule table and the endpoints live in `source_tables/` and the
history cross-check in `scores/`.

Resolution order: `--data-dir` > `SCPROTEOMICS_INTERVALS_DIR` > `<repository>/data/intervals` >
`<repository>/../data/intervals` > `<repository>/../data/source_tables` > `<repository>/data/source_tables`.
Outputs go to `--out-dir` (default `<data-dir>/outputs`); the verification report goes to
`--verify-dir` (default `<data-dir>/verification`).

## Running

```bash
python reproduce/intervals/recompute_paired_ci.py --data-dir /path/to/data/intervals
python reproduce/intervals/ci_verify.py --data-dir /path/to/data/intervals --tag release
```

`recompute_paired_ci.py` writes `paired_ci_recomputed.tsv`, `paired_ci_old_new.tsv`,
`paired_ci_indices.npy` and `environment.json` (17 significant digits for the raw endpoints).
`ci_verify.py` recomputes everything in a fresh process and writes a ten-row
`PASS`/`FAIL` report; it exits non-zero if any check fails. Both need `numpy` only.

## Verified in this release

Running both scripts against a copy of the frozen inputs reproduced the delivered endpoints
**bit for bit** at 17 significant digits (all seven comparisons, both endpoints), and the
sampling index matrix hash `d0ee2ba4533b31a88aa4e793416c06c89bc897e3133a70ca42cf80f768fc33fb`
matched the value recorded when the intervals were first computed, on a different Python and
numpy build (3.13.9 / 2.3.5 versus 3.14.3 / 2.5.2). `ci_verify.py` reported 10/10 checks
passing, including the independent loop-based spot check (max |difference| within 1e-12) and
the bounded old/new shift (maximum 0.2 points).

The mean differences are reproduced from the frozen table; no new seed, resample count or
quantile method was introduced. Changing the seed or the number of resamples would produce
different endpoints that must not be compared with the published ones.
