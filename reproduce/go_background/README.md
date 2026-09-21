# GO background analysis (four frozen steps)

These four scripts are the frozen implementation of the enrichment analysis reported for the
HeLa migration study: two backgrounds (a historical whole-MSigDB-GO background `HIST_REPLAY`
and an annotated-matrix background `EXPERIMENTAL_BG`), six query sets, and the fixed
24-position display comparison.

| script | job | inputs | outputs |
|---|---|---|---|
| `c1_go_datasets.py` | freeze the six query sets and build both backgrounds | PiSPA differential tables, the three GMT files | `<out>/tables/go_queries.tsv`, `go_queries_genes.tsv`, `go_gene_mapping.tsv`, `go_backgrounds.tsv`, `<out>/config/go_dataset_lock.json` |
| `c2_go_compute.py` | run both ORA branches and the historical reproduction gate | c1 outputs, stored enrichment tables | `<out>/tables/go_results_all.tsv`, `go_term_status.tsv`, `go_query_summary.tsv`, `go_historical_reproduction.tsv`, `<out>/tmp/c2_gate.json` |
| `c3_display_compare.py` | structural gate between branches and the fixed 24 display positions | c2 outputs, the frozen display table | `<out>/tables/go_fixed_display_comparison.tsv`, `go_significant_set_comparison.tsv`, `<out>/tmp/c3_structural_gate.json` |
| `c4_verify.py` | independent re-derivation with a different numerical path | c1/c2 outputs, GMT files | `<out>/verification/GO_INDEPENDENT_CHECK.tsv`, `GO_INDEPENDENT_CHECK.json` |

`gb_env.py` is the release wrapper: it resolves where the read-only data archive and the writable
working directory are. It is the only addition to the frozen scripts; the statistics, thresholds,
family definitions and output formats are unchanged.

## Inputs (read-only data archive)

    <data-dir>/
      tables/      go_results_all.tsv, go_backgrounds.tsv, go_queries_genes.tsv, ...
      tables/      hela_go_terms.tsv          (frozen display table of step 3)
      pispa_run/processed_proteins/           (frozen differential tables, steps 1-2)
      pispa_run/enrichment_results/           (stored enrichment tables, step 2)

    <gmt-dir>/     GO_BP.symbols.gmt, GO_CC.symbols.gmt, GO_MF.symbols.gmt  (obtain yourself)

In the release archive the display table is stored next to the other source tables, at
`<data-dir>/../source_tables/hela_go_terms.tsv`; that path is one of the two documented locations.

## Outputs (writable working directory)

    <out-dir>/
      tables/        c1-c3 derived tables
      config/        c1 dataset lock
      tmp/           c1-c3 timings and gate records
      verification/  c4 independent check

The archive is **never written to**. Everything the chain produces goes to `<out-dir>`, which
defaults to `<cwd>/go_background_out` and can be set with `--out-dir` (or `SCPROTEOMICS_GO_WORKDIR`).
An `--out-dir` that resolves inside the data archive is rejected with an explanatory message.

Resolution order, in each case the first candidate that satisfies the check:

    data archive   --data-dir > SCPROTEOMICS_GO_DIR > <repository>/data/go_background
    GMT directory  --gmt-dir > SCPROTEOMICS_GMT_DIR > <data-dir>/gmt > <data-dir>
    work directory --out-dir > SCPROTEOMICS_GO_WORKDIR > <cwd>/go_background_out

Nothing outside those candidates is probed. A missing input stops the step with the exact paths
it tried; there is no fallback to another copy of the data. Steps `c1` and `c2` require the
differential tables of the study, step `c3` requires the display table, and steps `c3`/`c4` can
otherwise verify a delivered run on their own.

## Running

    python reproduce/go_background/c1_go_datasets.py --data-dir <data-dir> --gmt-dir <gmt-dir> --out-dir <out-dir>
    python reproduce/go_background/c2_go_compute.py   --data-dir <data-dir> --gmt-dir <gmt-dir> --out-dir <out-dir>
    python reproduce/go_background/c3_display_compare.py --data-dir <data-dir> --gmt-dir <gmt-dir> --out-dir <out-dir>
    python reproduce/go_background/c4_verify.py      --data-dir <data-dir> --gmt-dir <gmt-dir> --out-dir <out-dir>

`c2` and `c4` import `scipy`, `statsmodels` and `mpmath`; `c1` and `c3` need `pandas` and
`numpy` only. Nothing here contacts the network, and none of these scripts needs a model key.

## What a successful run looks like

    c1  merged terms: 10490 dupes: 0 U_H: 19591 U_E: 4227
        ... (six query sets, each protein_rows_match;unique_genes_match)
    c2  result rows: 62584
        12 families printed, e.g. HIST_REPLAY C1 - C2 up N 19591 n 303 m 4829 sig 826
        gate: {'stored_rows': 90, 'matched': 90, 'mismatched': 0, ... 'gate': 'PASS'}
    c3  structural gate: PASS []
        fixed 24 positions: {'SUPPORTED_BOTH': 15, 'HIST_ONLY': 5, 'EXP_ONLY': 0, 'NEITHER': 4, ...}
    c4  {'families_checked': 12, 'terms_in_families': 62584, 'sampled_terms': 113, 'verdict': 'PASS'}

## Reproduction scope reached in this release

The whole chain was re-run from the delivered archive with a separate working directory. Eight of
the ten delivered tables came back **byte-identical**. Two differ, both for recorded reasons and
neither changing a count, a significance call or a support label:

* `go_queries.tsv` records the path of each query file; the release run stores it relative to a
  different base directory than the historical working tree did. Every count, check flag and file
  hash in that table is identical.
* `go_fixed_display_comparison.tsv` differs in eight last-digit strings of `stored_p_adjust`
  (17-digit values in the archive versus the shorter values written by the historical run). The
  support labels of the 24 fixed positions are unaffected.

The verifier's numerical summary is identical to the delivered one. The GMT files themselves are
not part of the release, so a run by a third party depends on obtaining a matching release; the
release identity and a byte-identical re-derivation are documented in `RESOURCE_ACQUISITION.md`.

## What the verifier checks, precisely

`c4_verify.py` re-derives P for every row of every family with `scipy.stats.hypergeom` and q with an
independent Benjamini-Hochberg implementation over the whole table, and additionally re-derives a
sampled set of terms (113 in the delivered run) with a 60-digit mpmath tail. The verdict covers
the full table for the scipy/statsmodels layer and the sampled terms for the arbitrary-precision
layer; it is **not** an arbitrary-precision re-derivation of all 62,584 rows.

## One field name is mapped for compatibility

The distributed copy of `hela_go_terms.tsv` names the stored member-count column
`matched_member_genes`, while the frozen consumer reads `matched_proteins`. `c3` renames that single
column back before use; no value is changed.

