# Acquiring the GO gene-set files for the background analysis

The three files below are **third-party gene-set resources (MSigDB / Gene Ontology derived)**.
They are **not redistributed** in this repository or in the accompanying data archive, because
their redistribution terms are not confirmed for the exact release in use. The GO term
statistics that the manuscript reports are distributed; the gene-set content is not. This file
records which files the frozen scripts read, how to obtain them, and how their identity was
established.

## Files the frozen scripts load

The scripts in this directory (`c1`-`c4`) read one directory, by default `<data-dir>/gmt/`,
containing exactly these three files:

| file | bytes | terms | sha256 |
|---|---:|---:|---|
| `GO_BP.symbols.gmt` | 4,872,550 | 7,538 | `9BE09DD06D6652566EB52EED530D62E6DFECC4365C1E81AFD6F0B7F2E86DD4F9` |
| `GO_CC.symbols.gmt` | 778,106 | 1,080 | `1EE846C446A87C1B6CC2B097DC415BED1C2F6B539312D2611F946038C5F7E9F8` |
| `GO_MF.symbols.gmt` | 957,205 | 1,872 | `72DA03F5438F1005566ECD5C85BF32E81F343910241948DA603D2FAFDF47F555` |

Format: tab-separated GMT, one term per line, `term_id<TAB>description<TAB>gene1<TAB>gene2...`.
The first identifiers in each file are `GOBP_10_FORMYLTETRAHYDROFOLATE_METABOLIC_PROCESS`,
`GOCC_3M_COMPLEX` and `GOMF_11_CIS_RETINAL_BINDING`. The merged set (BP, then CC, then MF,
later files overwriting an identical term identifier) contains 10,490 terms and 19,591 unique
human gene symbols; `c1_go_datasets.py` prints both numbers.

## Release identity (verified, not inferred from a file name)

The release these files came from is **MSigDB v2026.1**. The evidence is a byte-identical
re-derivation, not a name match:

* source bundle `msigdb.v2026.1.Hs.symbols.gmt`, 30,255,149 bytes, 35,361 terms,
  sha256 `FA938F38C8F41BAFA31E9C479C997A4CDAF69621E89446F149A1D9442A80DEAD`;
* the released module `offline_enrichment.py` builds the per-ontology exports from that bundle
  (`build_from_msigdb`, prefixes `GOBP_` / `GOCC_` / `GOMF_`);
* rebuilding the three files from that bundle with that function reproduced all three
  **byte for byte** (sha256 above), and `GO.symbols.gmt` (6,607,861 bytes, 10,490 terms) as well;
* the 10,490 exported terms are exactly the `GOBP_`/`GOCC_`/`GOMF_` subset of the bundle, and every
  one of the 10,490 gene sets is identical to the bundle entry with the same term identifier
  (0 differences, 0 terms missing on either side).

Two neighbouring files in the same resource directory are **not** used by the manuscript GO
analysis and are recorded here only to avoid confusion: `Reactome.symbols.gmt` also rebuilds
identically from the bundle, while `KEGG.symbols.gmt` does not come from the bundle (it is
produced by the KEGG REST refresh path) and its released copy differs from a bundle-derived one.

## How to obtain them

1. Register for a free account at the Molecular Signatures Database (MSigDB,
   <https://www.gsea-msigdb.org/gsea/msigdb/>) and accept its licence terms.
2. Download the human **C5 GO** collection of any release in *symbols* format:
   `c5.go.bp`, `c5.go.cc` and `c5.go.mf` (`*.symbols.gmt`). MSigDB names GO terms with the
   `GOBP_` / `GOCC_` / `GOMF_` prefixes used by the files above.
3. Rename or copy the three downloads into one directory as `GO_BP.symbols.gmt`,
   `GO_CC.symbols.gmt` and `GO_MF.symbols.gmt` (a copy, so the originals stay untouched).
4. Verify identity before running the scripts:

   ```powershell
   Get-FileHash -Algorithm SHA256 <dir>\GO_BP.symbols.gmt,<dir>\GO_CC.symbols.gmt,<dir>\GO_MF.symbols.gmt
   (Get-Content <dir>\GO_BP.symbols.gmt).Count   # 7538 terms
   (Get-Content <dir>\GO_CC.symbols.gmt).Count   # 1080 terms
   (Get-Content <dir>\GO_MF.symbols.gmt).Count   # 1872 terms
   ```

   ```bash
   sha256sum GO_BP.symbols.gmt GO_CC.symbols.gmt GO_MF.symbols.gmt
   wc -l GO_BP.symbols.gmt GO_CC.symbols.gmt GO_MF.symbols.gmt
   ```

5. Point the scripts at that directory: `--gmt-dir <dir>`, or place it at the default location
   `<data-dir>/gmt/`.

## Re-deriving the three files yourself

If you obtained the full v2026.1 human bundle instead of the C5 GO exports, the released module
reproduces the three files exactly:

    python -c "import offline_enrichment as oe; print(oe.build_from_msigdb('human', base_dir='<out-dir>', overwrite=True))"

The function reads `<kb_source>/gene_sets/msigdb.v2026.1.Hs.symbols.gmt` next to the released
modules; point `oe.MSIGDB_SOURCE` at your own copy if it lives elsewhere. The expected counts are
`{'GO': 10490, 'GO_BP': 7538, 'GO_CC': 1080, 'GO_MF': 1872, ...}` and the expected sha256 values are
those in the table above.

## If a later release is downloaded instead

The ORA background and the multiple-testing family sizes change, so the reproduced numbers will
differ from the published ones even though the code is unchanged. In that case the run is a
*bounded re-check with a different resource release*, not a reproduction; report it that way
rather than comparing endpoints against the published tables.

## Terms of use

MSigDB collections are distributed under the MSigDB licence of the release used; Gene Ontology
annotations are distributed under the terms of the GO Consortium. Neither licence was reviewed
for redistribution here. Do not add these files to a public repository or release archive
without checking the licence of the exact release and obtaining author sign-off (see
`../../THIRD_PARTY_NOTICES.md`).

