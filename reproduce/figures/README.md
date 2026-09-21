# Figures: statistics vs. layout

Two different things produce a manuscript figure, and this repository can only reproduce the
first of them.

**Statistical panels.** Every quantitative panel of the main and supplementary figures is drawn
from a frozen source table. The tables and the mapping panel -> table are listed in
`docs/DATA_MANIFEST.md`; the panel-to-figure mapping is kept as text, not as a data file name.
The analysis-side plotting code that generated the panels lives with the analysis runs and is
not part of this release. Reproducing a panel therefore means recomputing it from the table,
not re-running an analysis.

**Final layout.** The submitted figures are assembled in the authors' word-processor/
presentation files: panels are placed, relabelled and cropped by hand after the statistical
panels are exported. The released image files are those hand-assembled exports. No code in this
repository claims to reproduce that layout pixel by pixel, and no script here should be read as
proof that the assembled figure can be regenerated automatically.

What is verifiable offline:

* the frozen GO background panels, through `reproduce/go_background/` (the two branches and the
  fixed 24 displayed terms are reproduced from the frozen inputs);
* the paired interval panels, through `reproduce/intervals/` (endpoints are reproduced exactly);
* the score-distribution and dimension-contribution panels, through
  `reproduce/scoring/run_replay.py`, which reproduces the rule layer of every replayed cell;
* table-to-panel consistency, by reading the source tables in `data/source_tables/` and
  `data/case_tables/`.

A figure file whose statistical panel disagrees with its source table is an author-level
issue: the panel is either redrawn from the frozen table, or the disagreement is documented.
