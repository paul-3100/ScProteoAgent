# ScProteoAgent

## What this repository is

ScProteoAgent is an analysis system for single-cell proteomics. It takes a protein
quantification matrix, sample metadata and a natural-language request, and runs a domain-specific
workflow: it checks sample and protein coverage, processes the matrix, runs the differential
analysis the request implies, adds functional interpretation, and assembles a report with tables,
figures and saved analytical state. Because the system keeps the analysis design, the
matrix-processing history and the result objects, a follow-up question can reuse what still
applies and compute what changed.

This repository accompanies the manuscript *ScProteoAgent enables natural-language-driven
single-cell proteomics analysis and interpretation* (Runwen Hu, Keyan Ding, Wenwen Wang, Zhihui
Zhu, Peilin Chen, Shiqi Wang, Yu Wang). The software version is **1.0.0-rc.1**, recorded once in
`CITATION.cff` and mirrored here. A preprint link is added here once the preprint is posted. This
repository contains the
current engine, the frozen scoring implementation used for the cross-system comparison, the
reproduction scripts and frozen tables for the reported deterministic results, the scripts that
re-derive the reported case-study numbers field by field, a statistical panel redraw entry point,
one synthetic example and the offline tests. No software DOI, release tag or version badge is
claimed, because none exists yet.

This repository hosts the **1.0.0-rc.1 release candidate** at
https://github.com/paul-3100/ScProteoAgent. The author team has confirmed the software licence
with every contributor; `LICENSE_PENDING.md` records that decision and its scope. No software
release tag or software DOI is claimed yet. The supporting data have been uploaded to an
unpublished Zenodo draft (reserved data DOI: 10.5281/zenodo.22918085); its confidential review
link is provided in the manuscript submission, not in this public repository. The reserved DOI
will become active when the data record is published. `docs/DATA_MANIFEST.md` describes the data
archive; `docs/REPRODUCTION.md` records what each component checks and what remains unresolved.

## Choose a workflow

| workflow | what it does | what it needs |
|---|---|---|
| **Run the agent** | `main_agent.py -i <dataset>` plans, computes and writes a report into a run directory | your own credentials, a reachable OpenAI-compatible endpoint, and the agent dependencies; results depend on the model service |
| **Reproduce reported analyses** | `reproduce/` recomputes the scores, the paired intervals, the GO background analysis and the statistical panels from frozen tables, and compares the case-study tables field by field | the data archive; no model, no key, no network for the computations |
| **Inspect reported results** | read frozen tables directly, for example with `examples/saved_results/read_saved_table.py` | only Python; check a manuscript number without running anything |

## Installation

Tested on Windows 10 (build 10.0.19045, 64-bit) with CPython 3.13.9; not tested on Linux or
macOS. R is **not** installed on the test host, so the rpy2-based limma and enrichment paths were
not exercised. Hardware requirements and runtimes are **not measured**.

For the reproduction scripts, tests and examples:

    python -m venv .venv
    .venv/Scripts/python -m pip install -r requirements-reproduction.txt     # POSIX: .venv/bin/python

This installs numpy, pandas, scipy, statsmodels, mpmath and matplotlib. The scoring, validation,
example and most test scripts are standard-library only.

**Every command in this file runs with that environment's interpreter** - `.venv\Scripts\python`
on Windows, `.venv/bin/python` on POSIX. The plain `python` of the command lines below means that
interpreter, never the system interpreter, so no step falls back to an environment that was never
installed. The exact versions of the environment that was verified are recorded in
`requirements-lock.txt`, which is a record of an existing environment rather than an installer.

For a live agent run, install the larger set (see `VERIFIED_ENVIRONMENT.md` for exactly what
was measured when this release was assembled):

    .venv/Scripts/python -m pip install -r requirements-agent.txt

The agent stack is large (language-model clients, orchestration, single-cell and graph libraries)
and includes optional paths; `requirements-agent.txt` marks per line which entries were
exercised on the release host and which are derived from imports only. There is no published
package, so no `pip install scproteoagent` command exists.

## Quick start (offline, no key, no network)

Validate an input directory:

    python scripts/validate_dataset_inputs.py examples/minimal_demo

Read a frozen result table without running anything:

    python examples/saved_results/read_saved_table.py examples/saved_results/SYNTHETIC_rule_scores_sample.tsv --group-by method

Redraw one statistical panel from the frozen tables of the data archive:

    python reproduce/figures/plot_panels.py --list
    python reproduce/figures/plot_panels.py --panel score_distribution --data-dir <archive> --out-dir panels_out

## Running the agent

    python main_agent.py -i examples/minimal_demo -rf runs

This one **needs your key, contacts your endpoint and may be billed by your provider**. The
`-i` directory holds `ProteinQuant.csv`, `SampleInfo.csv` and `user_input.txt`;
`--design` fixes an `analysis_design.yaml` instead of inferring the design. Configuration is read
from `<project>/.env` only; `.env.example` lists every variable with empty values. Missing
required values stop the run with a message naming the variable instead of borrowing configuration
from elsewhere on the machine. The model name is recorded per run and is a scientific parameter:
results from one model service are not transferable to another without re-running.

A live agent run is **not** reproducible bit for bit - planning, tool choice and report assembly
depend on a model service whose version cannot be pinned from here. No live agent run was executed
while assembling this release, and `examples/minimal_demo` is synthetic demonstration data, never a
result. See `COMPONENT_ROLES.md` before attributing a historical number to the current engine.

## Reproducing the paper

Every row below was executed while assembling this release unless the status column says
otherwise, and `docs/REPRODUCTION.md` states per component which fields have been checked and what
is not reproduced. `<archive>` is the root of the data archive (the directory holding `inputs/`,
`source_tables/`, `scores/`, `intervals/`, `scoring/`, `go_background/`).

| reported item | command | status |
|---|---|---|
| cross-system scores (88 study-system cells) | `python reproduce/scoring/prepare_scoring_inputs.py --archive <archive> --out-dir <work>` then `python reproduce/scoring/run_replay.py --runs-root <work>/scoring/runs --examples-root <work>/examples --out-dir <work>/replay --expected-tsv <archive>/scores/HW_RULE_SCORES.tsv --check-report-sha256 --public-report-hashes <archive>/scoring/PUBLIC_REPORT_HASHES.tsv` | executed: 88/88 cells, maximum absolute difference 0 on `rule_total` and all six dimensions, 88/88 report hashes checked against their original bytes or the exact registered public-copy mapping |
| paired differences and their seven intervals | `python reproduce/intervals/recompute_paired_ci.py --data-dir <archive> --out-dir <work>` then `python reproduce/intervals/ci_verify.py --data-dir <archive> --out-dir <work>` | executed: endpoints reproduced, verifier 10/10 |
| GO background analysis of the migration case | `python reproduce/go_background/c1_go_datasets.py --data-dir <archive>/go_background --gmt-dir <gmt> --out-dir <work>`, then `c2`, `c3`, `c4` with the same flags | executed: 12 families, 62,584 term rows, fixed 24-position display 15/5/0/4, verifier PASS |
| five case-study computations | `python reproduce/cases/<theme>/run_<theme>.py --data-dir <archive> --out-dir <work>` | executed: each theme compares the reported tables of its case field by field, and every case exits 0 with no FAIL and no BLOCKED check; the per-check status is in `docs/REPRODUCTION.md` and `reproduce/cases/README.md` |
| statistical figure panels | `python reproduce/figures/plot_panels.py --panel <id> --data-dir <archive> --out-dir <work>` | executed: six panels redraw from the frozen source tables |
| input-format checks | `python scripts/validate_dataset_inputs.py <dataset-dir>` | executed on the demo dataset and on four negative fixtures |

`docs/REPRODUCTION.md` states, per item, what is recomputed, what is re-read, and what this
repository cannot reproduce.

A case script exits **0** when every check passed, **1** when a check failed, **2** when a required
input is missing (nothing is written) and **3** when the run is partial: no failure, but at least
one check is `BLOCKED` because a number the manuscript reports cannot be checked with what the
archive ships. A check that is `OUT_OF_SCOPE` asks for something the manuscript does not use; it is
reported separately and does not make the run partial.

## Data archive and third-party resources

The eleven benchmark studies come from published single-cell proteomics work; their matrices,
metadata and accessions are catalogued in the data archive (`study_mapping/`) and raw mass
spectrometry files stay with the original repositories. Every matrix is described by four separate
numbers - protein rows, columns (annotation columns included), matched observations and biological
units - and `docs/DATA_MANIFEST.md` defines them.

The uploaded Zenodo draft contains all eleven study input sets. The public record is not yet
available; editors and reviewers receive its confidential preview link with the manuscript.
`RIGHTS_AND_SOURCES.tsv` and `SOURCE_LINKS.tsv` in the data record identify the upstream sources,
transformations, attribution and terms for each third-party input. Brain and SCPro inputs are
included under a documented author decision, with source rights retained by the original authors.
The record-level data licence does not relicense third-party matrices. Resources that are not
redistributed at all are listed in `THIRD_PARTY_NOTICES.md` with acquisition steps.

The GO analysis needs MSigDB/Gene Ontology gene-set files, third-party resources that are **not**
redistributed here. `reproduce/go_background/RESOURCE_ACQUISITION.md` gives the download steps, the
recorded byte sizes, term counts and hashes, the verified release identity, and how to re-derive
the three files byte for byte. Local annotation resources and the optional online services
(UniProt, KEGG, STRING, DGIdb, gnomAD) are listed in `THIRD_PARTY_NOTICES.md`; the online tools are
off by default.

## Citation, licence and contact

If you use this software, cite the manuscript named in `CITATION.cff`, with the authors, title and
software version given there. The file validates against the Citation File Format 1.2.0 schema. The
code accompanying this text is the `main` branch of https://github.com/paul-3100/ScProteoAgent;
no software DOI, release tag or archived software snapshot is claimed yet.

**Licence.** The software is released under the ScProteoAgent Research Licence 1.0 (`LICENSE`).
Research, teaching, reproducing published results, and non-commercial modification and
redistribution are free of charge. Commercial use - research and development inside a company, a
paid analysis service, or integration into a commercial product - requires prior written
authorisation from the grantor, the author team named in `CITATION.cff`; contacting the authors
does not itself grant that right. This is not an OSI-approved licence and it is not on the SPDX
licence list, so `CITATION.cff` gives it by URL (`license-url`, the `LICENSE` file in this
repository) rather than as an SPDX identifier, and this repository must not be described as open
source. Copyright remains with the authors and their institutions. Software terms
and data terms are separate: the data archive carries its own rights table, and third-party
components and their status are listed in `THIRD_PARTY_NOTICES.md`.

Correspondence: Shiqi Wang (shiqwang@cityu.edu.hk) and Yu Wang (wangyu310@zju.edu.cn). Please open
an issue for software problems rather than mailing the authors first; questions about the data
archive or the manuscript go to the corresponding authors.

## Limits and support

* The reported deterministic results replay from frozen inputs: the rule layer of the frozen
  scorer, the paired intervals, the two-background GO chain, the case computations and the
  statistical panels. `docs/REPRODUCTION.md` states per component which fields were checked and
  which are not reproduced, and `VERIFIED_ENVIRONMENT.md` records what was executed on the
  verification host. The status is reported per component rather than as one overall word: the
  scoring replay (88 of 88 study-system cells), the paired intervals (all seven
  endpoints, verifier 10 of 10) and the GO background chain (12 families, 62,584 term rows,
  gate 90 of 90, verifier PASS) are verified from frozen inputs, and all five case studies exit
  0 with no FAIL and no BLOCKED check. The HeLa odds-ratio interval is the one place where a
  computed value and a historical transcription of it are kept apart instead of merged.
* A live agent run is a new result, not a reproduction, and the models used for the manuscript are
  recorded there rather than recommended here.
* The assembled scoring input tree is scoring-only: it carries the evaluation references so that the
  frozen scorer reads the same bytes that were scored, it is not an agent input directory, and it can
  be deleted after the replay. An explicit `--datasets`/`--system` run reports `SUBSET_PASS` with its
  coverage rather than the 88-cell result.
* The assembled manuscript figures were laid out by hand in the authors' document/presentation
  files. `reproduce/figures/plot_panels.py` redraws statistical panels from the frozen tables; it
  does not reproduce that layout, and the released image files are the authors' exports.
* This repository mixes code from different project phases. Read `COMPONENT_ROLES.md` before
  attributing any historical number to the current engine, and treat the frozen scorer as a
  separate artefact identified by content hash.
* Two identifier-resolution limits are open, and this repository does not claim that every protein
  identifier in a task text is resolved. The candidate-exclusion trace that the analysis extensions
  write can still come from the older extractor, so a trace can name an ordinary word that is not a
  protein; that path is left frozen because changing it would change the released Chinese run
  output. The candidate audit resolves gene symbols only, so a protein-group identifier that does
  exist in the matrix can still be reported as unresolved. Both are candidates for a later minor
  release, and both are listed here rather than smoothed over.
* Anything a bounded check could not settle is listed as unresolved in `docs/REPRODUCTION.md`
  together with the paths that were searched, rather than smoothed over. The same page names the
  checks that ask for artefacts the manuscript does not require.

## Report language

The report language is an explicit option. It is never inferred from the task text, so a
Chinese request can produce an English report and an English request can produce a Chinese
one.

    py -3 main_agent.py -i <dataset> -rf runs                       # zh (default)
    py -3 main_agent.py -i <dataset> -rf runs --report-language en  # English

`--report-language` accepts `zh` (the default) or `en`; any other value fails argument
parsing before the agent starts. The resolved language and where it came from (`cli`,
`inherited_from_parent` or `default`) are recorded in `run_metadata.json` and
`parameters.json`.

`--continue-from <run_folder>` continues an earlier run. It inherits that run's report
language unless an explicit `--report-language` is given, and copies the saved analysis
design and normalized matrix into the new run. Contrast tables are deliberately not copied,
so a continuation recomputes the comparisons it needs instead of replaying the previous
report; and because a run may legitimately recompute and overwrite a copied file, the record
under `continued_artifacts` gives each carried-over artefact a source path and a seed-time
destination hash, labelled as such, rather than a claim that the file was consumed. Entries
that were skipped are listed too, so a skipped subdirectory is never silent.

Scope of the English mode: the report skeleton, the headings, the deterministic report prose,
the figure captions and the report fact-check vocabulary are bilingual. Anything that still
has no English template is reported as an explicit `MISSING_EN_TEMPLATE:<key>` marker and
listed in the run's gap list, and the report self-check reports any Chinese residue it finds;
an English report never silently falls back to Chinese.

The released Chinese behaviour is unchanged: with `zh` the report text is byte-identical to
the 1.0.0-rc.1 baseline. The frozen scoring code under `reproduce/scoring/` is not modified,
and the English mode does not re-score the paper: the frozen scorer's Chinese-language
dimension is not applied to English output.
