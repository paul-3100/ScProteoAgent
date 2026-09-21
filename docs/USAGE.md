# Using the agent

This page is the operational companion to the README: it describes the entry point, the inputs,
the outputs of a run, the design file, and the helper scripts that ship with the engine.

## Entry point

```bash
python main_agent.py -i <dataset-dir> -rf <runs-folder> [--name NAME] [--design FILE]
```

| flag | meaning |
|---|---|
| `-i`, `--input-folder` | dataset directory: `ProteinQuant.csv`, `SampleInfo.csv`, `user_input.txt` |
| `-rf`, `--runs-folder` | where run directories are created (default `./runs`) |
| `-n`, `--name` | experiment name used for folder naming |
| `--design` | explicit `analysis_design.yaml`; priority: CLI > design in the dataset > inferred design |
| `--publication-mode` | on by default: evaluator-only files stay out of generation state |
| `--evaluator-mode {none,internal-dev}` | `internal-dev` is a development-only mode and must not be used for reported results |

The run directory is named `<dataset>_<model>_<YYYYMMDD>` for the eleven benchmark studies and
`Agent-<model>-<name>-<timestamp>` for any other dataset name.

## Report language and continuation

Two optional flags control the report language and follow-up runs.

| Flag | Effect |
|---|---|
| `--report-language {zh,en}` | Report language. Default `zh`; any other value fails argument parsing before the agent starts. |
| `--continue-from <run_folder>` | Continue a previous run: inherits its report language and reuses its saved design and normalized matrix. |

Resolution order is explicit `--report-language` > the language recorded by
`--continue-from` > `zh`. The task text is never inspected. The resolved language and its
source are written to `run_metadata.json` and `parameters.json`, and reused artefacts are
listed under `continued_artifacts` with their source paths and both hashes.

English output is not a translation of the Chinese report: the prompts, the headings, the
deterministic prose and the captions have English templates of their own. Where a template is
missing, the report carries a `MISSING_EN_TEMPLATE:<key>` marker instead of Chinese text.

## Inputs

* `ProteinQuant.csv` — proteins in rows, sample columns as in `SampleInfo.csv`, annotation columns
  in the leading block (and, in some studies, a gene column at the end). Missing values are empty
  fields.
* `SampleInfo.csv` — one row per sample; the sample key column is matched against the matrix
  column names (in the released studies it is `FileName`); grouping and unit columns live here.
* `user_input.txt` — the natural-language request: background, column description, tasks and the
  required output form.
* Optional `analysis_design.yaml` — groups, contrasts, directions, experimental units and matrix
  state, when the design must be fixed rather than inferred.
* Optional `ground_truth.txt` and `grading_standard.txt` — evaluation material. They are **not**
  read in publication mode; they exist for retrospective scoring and must not be treated as
  generation input.

Validate a dataset before running anything (offline, no key):

```bash
python scripts/validate_dataset_inputs.py <dataset-dir>
```

## Outputs of a run

| file or directory | content |
|---|---|
| `report.md` | the assembled report (the main text output) |
| `record_file.md` | step-by-step execution log with node names and tool results |
| `run_events.jsonl` | machine-readable event stream, including the report-production configuration snapshot |
| `run_metadata.json`, `parameters.json` | run inputs, model name, mode and resolved paths |
| `memory.json` | the shared state passed between nodes |
| `analysis_design.used.yaml`, `analysis_design.inferred.yaml` | the design actually used and the inferred one |
| `processed_proteins/` | matrices after each processing step, differential tables, QC tables, unit-structure records |
| `enrichment_results/`, `ppi_results/`, `umap_results/`, `cluster_distribution/`, `visualize_results/` | per-tool outputs |
| `figures/`, `figure_index`, `assets.txt` | figures and the artifact index |
| `evidence_ledger.jsonl`, `report_evidence_binding.json`, `report_fact_check.json` | evidence binding between statements and artifacts |
| `llm_usage.jsonl` | per-request token accounting |

The report is assembled from structured sections when the run uses the structured assembly mode;
the mode is recorded in the run events, and a request for a mode that the pipeline does not
implement raises instead of silently falling back.

## Helper scripts

All of these run offline unless their name says otherwise:

| script | purpose |
|---|---|
| `scripts/validate_dataset_inputs.py` | input contract, orientation, ID alignment, duplicates, missing markers, grouping/unit fields |
| `scripts/validate_analysis_designs.py` | validates or infers an analysis design for every dataset under a root |
| `scripts/validate_scientific_report.py <run-dir>` | report-level validation for a finished run |
| `scripts/validate_annotation_manifest.py --manifest <yaml>` | checks the local annotation-resource manifest |
| `scripts/audit_hidden_file_access.py --run-dir <dir> --out-json ... --out-md ...` | audits generation artifacts for evaluator-only file access |
| `scripts/export_reproducibility_manifest.py --out-json ... --out-md ...` | writes a reproducibility manifest for a release root |
| `scripts/prepare_blind_analysis_inputs.py` | builds publication-safe input copies without evaluation material |
| `scripts/run_batch_evaluation.py` | batch driver over many datasets; it starts one agent process per dataset and is the only script here that can spend model budget |

Example of a batch run against a dataset root (needs the agent environment, a key and a paid
endpoint; not executed during this release):

```bash
python scripts/run_batch_evaluation.py --examples-root <dataset-root> --runs-root <runs-root> \
    --publication-mode --audit-hidden-file-access
```

## Configuration and knobs that matter for interpretation

* `OPENAI_MODEL` is recorded per run; treat a model or service change as a new run.
* `SCPROTEO_REPORT_ASSEMBLY` selects the report assembly path; an unknown value raises.
* `SCPROTEO_REQUEST_EVIDENCE_INDEX`, `SCPROTEO_PERSIST_RAW_RESPONSE` control request evidence.
* The `R27_` (or `R26_`) envelope variables bound attempts, input/output tokens, per-call output,
  reserve and timeout. They stop a run before an unauthorised request is sent.
* `SC_ENRICH_ALLOW_ONLINE` and `SC_PPI_ALLOW_ONLINE` default to offline; enable them only when you
  intend to query remote services.

## Data-protection notes

Evaluation material (`ground_truth.txt`, `grading_standard.txt`, evaluator-only notes) must not
enter generation state. `--publication-mode` is the default, and
`scripts/audit_hidden_file_access.py` records any violation it finds in a run directory.
