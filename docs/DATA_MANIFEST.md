# Data manifest

The supporting data have been uploaded to an unpublished Zenodo draft (reserved data DOI:
10.5281/zenodo.22918085). Editors and reviewers receive the confidential preview link with the
manuscript; the data record is not yet a public download. This page describes what the reproduction
scripts expect to find, how they find it, and what the studies are. The authoritative file-by-file
list, with byte sizes, hashes and rights status, accompanies the archive. The reviewer access token
is intentionally absent from this public code repository.

## Layout expected by the scripts

```
data/
  README.md, DATA_DICTIONARY.md, RIGHTS_STATUS.tsv   archive entry points
  inputs/                 study inputs (published files, study directory names kept)
  evaluation_references/  task and grading references (evaluation use only)
  study_mapping/          study descriptors, source DOIs, accessions (STUDY_INPUT_MAP.tsv)
  source_tables/          tables behind the main and supplementary figures, plus the
                          rule-score table (rule_scores.tsv), the comparison endpoints
                          (paired_means.tsv) and the GO display table (hela_go_terms.tsv)
  case_tables/            per-case tables (composition, contrasts, unit mappings, states)
  scoring/                the 88-cell closure: the frozen scorer, its registry and the archived
                          reports and artefacts the rule layer reads (scoring/runs/<system>/...)
  scores/                 frozen score tables (HW_RULE_SCORES.tsv, passes/)
  intervals/              scripts/ (the two interval scripts) and tables/ (recomputed endpoints,
                          index matrix, old/new comparison, environment record)
  go_background/          scripts/ (the four GO steps), tables/ (the ten GO tables),
                          RESOURCE_ACQUISITION.md; a working run also needs gmt/ (yours) and,
                          for the first two steps, pispa_run/
```

Resolution of a data directory, when a script is not given an explicit path:

| script group | flag | environment variable | fallback order |
|---|---|---|---|
| scoring replay | `--runs-root`, `--examples-root` | — | explicit paths only (no implicit discovery) |
| intervals | `--data-dir`, `--out-dir`, `--verify-dir` | `SCPROTEOMICS_INTERVALS_DIR` | `<repo>/data/intervals`, `<repo>/../data/intervals`, `<repo>/../data/source_tables`, `<repo>/data/source_tables` |
| GO background (inputs) | `--data-dir`, `--gmt-dir` | `SCPROTEOMICS_GO_DIR`, `SCPROTEOMICS_GMT_DIR` | `<repo>/data/go_background`; GMTs in the `--gmt-dir` directory, `<data-dir>/gmt` or `<data-dir>` |
| GO background (outputs) | `--out-dir` | `SCPROTEOMICS_GO_WORKDIR` | `<cwd>/go_background_out`; never inside the data archive |

Within a data directory the interval inputs are accepted under either their historical name
(`benchmark_rule_scores.tsv`, `benchmark_paired_difference_means.tsv`) or the archive name
(`rule_scores.tsv`, `paired_means.tsv`), searched in the directory itself, in `tables/`, in
`inputs/` and in the sibling directories `source_tables/`, `scores/` and `intervals/`. The GO
display table is searched in `<data-dir>/tables/` and in the source-table directory of the
archive.

Every location above is a documented location of the release archive, and the error message of a
step lists the paths it tried. No script probes unrelated directories on the machine, and the GO
chain writes only to its working directory, so the data archive can be mounted read-only.

## The eleven benchmark studies

Study directory names are part of the interface: the engine recognises these names when naming
run folders, and the frozen scorer matches datasets by directory name. Keep them unchanged.

| study directory | biological setting | protein rows x columns (annotation columns included) |
|---|---|---:|
| `Nat_Commun_PiSPA_2024` | HeLa migration, cluster/type composition and hidden heterogeneity | 5,191 x 94 |
| `Nat_Methods_iPSC_2025` | hiPSC/EB differentiation and within-EB heterogeneity | 8,834 x 69 |
| `Nat_Biotech_Brain_2026` | brain development trajectories and state composition | 4,279 x 2,601 |
| `Nat_Commun_Carr_2024` | THP-1 inflammatory response with batch structure | 1,369 x 163 |
| `Nat_Methods_pSCoPE_2023` | LPS treatment, phagocytosis/acidification states | 1,123 x 95 |
| `Nat_Methods_DVP_2023` | liver lobule zonation and spatial gradients | 2,149 x 50 |
| `Nat_Commun_SCPro_2024` | pancreatic tumour T-cell states, Treg heterogeneity | 3,777 x 28 |
| `Nat_Commun_Nociceptor_2026` | DRG nociceptor subtypes and inflammatory response | 6,430 x 23 |
| `Science_BloodCell_2025` | haematopoietic stem/progenitor states with donor identity | 2,934 x 130 |
| `Cell_TurnoverDynamics_2025` | drug perturbation and abundance footprints | 7,753 x 45 |
| `Nat_Commun_ProteinLeakage_2025` | membrane integrity / protein leakage QC | 1,564 x 2,297 |

The shape column was measured on the inputs used for this release and counts every column of the
file, annotation columns included. Four quantities are easy to confuse and are therefore named
separately throughout the archive:

* **protein rows** - the data rows of the matrix, the first number above;
* **columns** - all columns of the file, including **annotation columns** (protein identifiers,
  gene names, descriptions), which are not measurements;
* **matched observations** - the sample columns that have a row in `SampleInfo.csv` (for example
  2,310 of the 2,601 brain columns; 5,262 observations in total across the eleven studies);
* **biological units** - the independent units a comparison can be paired on (donor, mouse,
  patient, batch); one unit may contribute several observations and several columns.

## Fields that matter when reading a table

| field family | meaning | notes |
|---|---|---|
| sample key (`FileName`) | links a matrix column to a metadata row | matched exactly, case-sensitive |
| grouping (`Cluster`, `Type`, `Condition`, …) | the comparison groups | a contrast must name the grouping column explicitly |
| experimental unit (`Donor`, `Batch`, `Mouse`) | the level at which replicate structure exists | a contrast over pooled cells is a different question from a contrast over individuals |
| `logFC`, `P.Value`, `adj.P.Val`, `q` | effect size, raw and adjusted significance | q families are per contrast/direction/branch; never mix them |
| `mode`, `branch`, `direction`, `contrast` | processing or analysis branch identity | two branches that share a name are not interchangeable |
| unit-level tables | values aggregated to the experimental unit before testing | they answer different questions than cell-level tables |

Missing values stay missing: empty cells are empty in the archive, small P values are stored at
full precision, and no column is filled with zeros to make a table rectangular.

## Inputs, references and rights

* `inputs/` holds all eleven study matrices and metadata sets used for the analyses. The archive's
  `RIGHTS_AND_SOURCES.tsv` and `SOURCE_LINKS.tsv` record the source, transformation, attribution
  and applicable terms for each third-party input. The Brain and SCPro processed matrices are
  included under a documented author decision; rights remain with the original authors. The
  record-level data licence does not override those third-party terms.
* `evaluation_references/` holds the task and grading references. They are evaluation material:
  they are kept in a directory of their own, outside the study inputs, and the released engine
  does not read them from there; the scoring replay assembles its own copy when it needs them.
  Some archived reports were produced in earlier rounds in which reference-derived material took
  part in report assembly, so this separation describes the code path of this release rather
  than every historical run.
* Raw mass-spectrometry files are **not** in the archive; use the original repositories and the
  accessions recorded in `study_mapping/`.
* Gene-set resources (MSigDB / GO GMT files) are not redistributed; see
  `reproduce/go_background/RESOURCE_ACQUISITION.md` for the acquisition steps and the recorded
  hashes, and `THIRD_PARTY_NOTICES.md` for the rights status.
* The data archive's own terms are recorded separately from the software licence. See
  `LICENSE_PENDING.md`.

## Known gaps

* One study directory in the working collection contains columns that have no `SampleInfo` row;
  analyses use the matched columns and the validator reports the remainder as a warning.
* Reports that the manuscript does not use (for example alternative scoring layers) are not part
  of the archive; they remain in the local project history.
* If a table in the archive disagrees with the manuscript, the manuscript and its Source Data
  sheet are authoritative: report the disagreement rather than editing either side.
