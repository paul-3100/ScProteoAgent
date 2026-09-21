# Third-party notices

This repository contains only the authors' own code and documentation. It redistributes no
third-party source code, no third-party data and no third-party gene-set content. Components
that are used but not shipped are listed below with their role and their status, so that no
licence of one artefact is read as covering another.

## Python dependencies (not bundled)

Installing the requirements downloads these from PyPI under their own licences. Where the
installed distribution reported a licence in the environment used to assemble this release, the
reported value is quoted; everything else is marked *not verified here* rather than guessed.

| distribution | role | licence reported by the installed package |
|---|---|---|
| numpy | matrices, bootstrap indexing | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` (numpy 2.5.2 metadata) |
| pandas | tabular I/O | BSD 3-Clause (pandas 2.3.3 metadata) |
| scipy | hypergeometric tail in the GO computation | SciPy Developers' BSD-style licence (scipy 1.16.3 metadata) |
| statsmodels | Benjamini-Hochberg adjustment | BSD (statsmodels 0.14.5 metadata) |
| mpmath | high-precision independent P values | BSD (mpmath 1.3.0 metadata) |
| scikit-learn | clustering/decomposition in the agent tools | BSD-3-Clause (scikit-learn 1.7.2 metadata) |
| matplotlib | plotting in the agent tools | matplotlib licence (3.10.6 metadata) |
| networkx, requests, python-dotenv, PyYAML, plotly | agent/tool support | Apache-2.0 (requests), BSD-3-Clause (python-dotenv), MIT (PyYAML, plotly), *not verified here* (networkx) |
| openai, langchain, langchain-core, langchain-openai, langgraph, google-genai | model access and orchestration | *not verified here* |
| scanpy, gseapy, joblib, tqdm, umap-learn, adjustText, biopython, mygene, pillow, seaborn, tiktoken, ddgs, igraph | analysis, enrichment and token accounting | installed and exercised only where the release checks reached them; otherwise *not verified here* |
| rpy2 (+ rpy2-rinterface) | the R bridge used by the limma and clusterProfiler paths | **optional extra**, kept out of the base install on purpose: rpy2 3.6.8 requires `rpy2-rinterface>=3.6.7`, which has no Windows wheel for CPython 3.13. `requirements-agent.txt` documents the pinned pair to install if you need it |
| R and the R packages clusterProfiler, org.Hs.eg.db, org.Mm.eg.db, ReactomePA | called through rpy2 by the enrichment and limma paths | not installed in the environment used here, so those paths were not exercised |

The full dependency lists, including which ones were exercised in the release environment, are
in `requirements-agent.txt` and `requirements-reproduction.txt`.

## External resources referenced but not redistributed

| resource | used for | status |
|---|---|---|
| MSigDB GO collections (`GO_BP/CC/MF.symbols.gmt`) | the ORA background of the HeLa migration case | **not shipped**; acquisition steps and recorded hashes in `reproduce/go_background/RESOURCE_ACQUISITION.md`; terms of use not reviewed for redistribution |
| KEGG, UniProt, STRING, DGIdb, gnomAD | optional online knowledge lookups inside the agent tools | queried only when the user enables the corresponding switch; no bulk data shipped; each service's own terms apply |
| Public mass-spectrometry datasets from the 11 source studies | the benchmark inputs | obtained from the original repositories and accession numbers; redistribution is handled with the data archive, not this repository |

## Gene-set and annotation content

The GO statistics that the manuscript reports (`go_results_all.tsv` and the other GO tables) are
derived tables. They are shipped in the data archive, not here, and their redistribution status
is recorded with that archive. Publishing derived statistics does not grant rights to the
underlying gene-set membership, which is why the full membership tables require the same review
as the GMT files themselves.

## What to do before publishing

## Derived inputs added for the case reproduction

`reproduce/cases/` recomputes the five case-study results from archived inputs. The
run-processed artifacts it reads - normalized matrices, QC tables, protein-to-gene maps and
per-protein result tables produced by the analysis runs - are derived from third-party study
data. They ship with the data archive rather than with this repository, and every one of them
carries `rights_status=UNCONFIRMED` together with an author-decision number in the rights table of
the archive. Treat them under the same review as the study matrices themselves.

1. Confirm the software licence with the rights holder (`LICENSE_PENDING.md`).
2. Review the exact MSigDB release licence and the GO Consortium terms before any gene-set
   content is redistributed.
3. Keep the data archive's terms separate from the software licence.
