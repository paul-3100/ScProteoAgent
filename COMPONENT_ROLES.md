# Component roles: four kinds of code, four different claims

This repository holds code from four different phases of the project. They must not be presented
as one system built by one commit, and results from one phase must not be attributed to another.

| phase | files | what it is | what it can and cannot support |
|---|---|---|---|
| **Current engine** | `main_agent.py`, `tools.py`, `llm.py`, `prompts.py`, `tools_prompts.py`, `config.py`, `path_config.py`, `analysis_design.py`, `analysis_extensions.py`, `report_assembly.py`, `report_evidence.py`, `evidence_utils.py`, `evaluation_evidence.py`, `figure_captions.py`, `figure_recipes.py`, `offline_enrichment.py`, `revision_requirements.py`, and `scripts/` | the agent that performs the analyses described in the manuscript: planning/execution nodes, domain tools, report assembly and evidence binding | supports the description of the current system. It is **not** the code that produced every historical result in the manuscript: several comparison systems and earlier project stages produced their own outputs. |
| **Frozen historical scorer** | `reproduce/scoring/score_full_run.py` (hash-pinned), `guarded_launcher.py`, `run_replay.py` | the rule-based scoring implementation used for the cross-system comparison, frozen at the revision that produced the reported 88 cells | defines the comparison's rule layer and can reproduce it offline. It is not the current engine and must not be replaced by a later, extended scorer; its LLM review layer is not used for the manuscript numbers. |
| **Continuation branch** | `reproduce/intervals/`, `reproduce/go_background/`, `reproduce/cases/` | follow-up analyses performed after the main comparison: paired intervals, the two-background GO analysis with its verifier, and the case study tables | reproduce deterministic follow-up results from frozen inputs. They are downstream of both the engine and the frozen scorer and do not re-derive the analyses themselves. |
| **Offline supplementary analysis** | `scripts/validate_dataset_inputs.py`, `scripts/validate_analysis_designs.py`, `scripts/validate_annotation_manifest.py`, `scripts/validate_scientific_report.py`, `scripts/audit_hidden_file_access.py`, `scripts/export_reproducibility_manifest.py`, `scripts/prepare_blind_analysis_inputs.py`, `scripts/run_batch_evaluation.py`, `tests/`, `examples/` | environment, input, report and release checks, plus the packaged examples | support reproducibility and quality control. They are not part of the scientific method and produce no manuscript result. |

## Rules that follow from this

1. **No single-commit claim.** Do not write that one release commit produced all historical
   results. The engine here is the current one; the frozen scorer is a separate artefact pinned
   by content hash; the case tables came from analysis runs that are older than both.
2. **Hash identity, not similarity.** The frozen scorer is identified by
   `sha256 2B37C1835D397FBF132BAD9DEB8900220BCD3ABCD299ECEE8112CC0B3BEEA74B` and 782 lines.
   A different scorer that produces similar numbers is a different artefact and cannot be
   substituted for it; the release keeps the exact file and verifies it in
   `tests/test_frozen_scorer_contract.py`.
3. **The continuation scripts are wrappers around frozen science.** `gb_env.py` and the
   `--data-dir` flags only relocate inputs and outputs. They do not change thresholds, families,
   seeds, quantile methods or outputs, and nothing in them writes to the historical working tree.
4. **Model versions are parameters, not constants.** The engine reads the model name from the
   environment and records it per run. Results produced by one model service are not transferable
   to another without re-running.
5. **Historical results stay attributed to their own runs.** Where a manuscript number comes from
   a comparison system or from an earlier project stage, the corresponding code is either not part
   of this repository or is present as a frozen artefact with its own provenance.
