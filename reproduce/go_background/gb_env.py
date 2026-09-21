"""Path resolution for the frozen GO background scripts (c1-c4).

The four scripts were written inside the manuscript working tree, where their inputs, working
tables and outputs all lived in one package directory. In this release they read and write a
**data archive** instead. This module only resolves locations; every scientific function,
threshold, family definition and output format in c1-c4 is unchanged.

Read-only inputs (data archive, see RESOURCE_ACQUISITION.md):

    <data-dir>/tables/go_results_all.tsv, go_backgrounds.tsv, go_queries_genes.tsv, ...
    <data-dir>/tables/hela_go_terms.tsv        (c3 display table)
    <data-dir>/pispa_run/processed_proteins/    (frozen differential tables, c1/c2)
    <data-dir>/pispa_run/enrichment_results/    (stored enrichment tables, c2)
    <gmt-dir>/GO_BP.symbols.gmt, GO_CC.symbols.gmt, GO_MF.symbols.gmt

Writable outputs (working directory):

    <out-dir>/tables/        c1-c3 derived tables
    <out-dir>/config/        c1 dataset lock
    <out-dir>/tmp/           c1-c3 timings and gate records
    <out-dir>/verification/  c4 independent check

Resolution order, in each case the first candidate that satisfies the check:

  data archive   --data-dir > SCPROTEOMICS_GO_DIR > <repository>/data/go_background
  GMT directory  --gmt-dir > SCPROTEOMICS_GMT_DIR > <data-dir>/gmt > <data-dir>
  work directory --out-dir > SCPROTEOMICS_GO_WORKDIR > <cwd>/go_background_out

Nothing is probed outside those candidates: if a required input is missing, the script stops
with the exact paths it tried and never falls back to another copy of the data. A work
directory that resolves inside the data archive is rejected, because the archive is an input,
not a scratch space.

The GMT files are third-party gene-set resources. They are not redistributed here; see
RESOURCE_ACQUISITION.md for a matching release and for the check that ties the three files to
the MSigDB bundle. Steps c3 and c4 also run without them, on the delivered tables.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = "data/go_background"
DEFAULT_WORK_DIR = "go_background_out"
GMT_ORDER = ["GO_BP.symbols.gmt", "GO_CC.symbols.gmt", "GO_MF.symbols.gmt"]


def cli_overrides(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the two release flags without touching the scripts' own argument handling."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--gmt-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return args


def _data_root_candidates(cli_value: str | None) -> list[tuple[str, Path]]:
    """Documented archive locations only; unrelated directories are never probed."""
    candidates: list[tuple[str, Path]] = []
    if cli_value:
        candidates.append(("--data-dir", Path(cli_value).expanduser()))
    env_value = os.getenv("SCPROTEOMICS_GO_DIR", "").strip()
    if env_value:
        candidates.append(("SCPROTEOMICS_GO_DIR", Path(env_value).expanduser()))
    candidates.append(("default data/go_background of the repository", REPO / DEFAULT_DATA_DIR))
    return candidates


def _gmt_candidates(cli_value: str | None, data_root: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    if cli_value:
        candidates.append(("--gmt-dir", Path(cli_value).expanduser()))
    env_value = os.getenv("SCPROTEOMICS_GMT_DIR", "").strip()
    if env_value:
        candidates.append(("SCPROTEOMICS_GMT_DIR", Path(env_value).expanduser()))
    candidates.append(("data archive", data_root / "gmt"))
    candidates.append(("data archive root", data_root))
    return candidates


def _work_root_candidates(cli_value: str | None) -> list[tuple[str, Path]]:
    """Writable root. Never the archive: the archive is an input and stays read-only."""
    candidates: list[tuple[str, Path]] = []
    if cli_value:
        candidates.append(("--out-dir", Path(cli_value).expanduser()))
    env_value = os.getenv("SCPROTEOMICS_GO_WORKDIR", "").strip()
    if env_value:
        candidates.append(("SCPROTEOMICS_GO_WORKDIR", Path(env_value).expanduser()))
    candidates.append(("current working directory", Path.cwd() / DEFAULT_WORK_DIR))
    return candidates


def _caller_name() -> str:
    """Name of the script that is resolving paths, so each step fails on its own inputs."""
    return Path(sys.argv[0]).name if sys.argv and sys.argv[0] else ""


def resolve_paths(argv: list[str] | None = None, require_pispa: bool = False,
                  require_display: bool | None = None) -> dict:
    """Resolve every location the frozen GO scripts use, or fail with an actionable message.

    ``require_pispa`` is set by the two steps that read the study's differential tables; the
    verification and display steps do not need them and therefore do not fail without them.
    ``require_display`` is set by the display step (``c3``), the only consumer of
    ``hela_go_terms.tsv``; when it is not given, the name of the calling script decides.
    """
    args = cli_overrides(argv)
    if require_display is None:
        require_display = _caller_name().startswith("c3")
    data_candidates = _data_root_candidates(args.data_dir)
    data_root = None
    for _, candidate in data_candidates:
        resolved = candidate.resolve()
        if (resolved / "tables").is_dir():
            data_root = resolved
            break
    if data_root is None:
        raise SystemExit(
            "FATAL: no GO data archive found. Checked %s. Pass --data-dir or set "
            "SCPROTEOMICS_GO_DIR (see reproduce/go_background/RESOURCE_ACQUISITION.md)."
            % "; ".join("%s=%s" % (source, path) for source, path in data_candidates))

    gmt_candidates = _gmt_candidates(args.gmt_dir, data_root)
    gmt_dir = None
    for _, candidate in gmt_candidates:
        resolved = candidate.resolve()
        if all((resolved / name).is_file() for name in GMT_ORDER):
            gmt_dir = resolved
            break
    if gmt_dir is None:
        raise SystemExit(
            ("FATAL: %s not found. Checked %s. Obtain the GMT files as described in "
             "reproduce/go_background/RESOURCE_ACQUISITION.md and pass --gmt-dir.")
            % (", ".join(GMT_ORDER),
               "; ".join("%s=%s" % (source, path) for source, path in gmt_candidates)))

    pispa_run = data_root / "pispa_run"
    if require_pispa and not (pispa_run / "processed_proteins").is_dir():
        raise SystemExit(
            ("FATAL: the frozen differential tables of the study are not in the archive: "
             "expected %s/processed_proteins/ (plus enrichment_results/ for step 2). Place a "
             "copy of the PiSPA S01 run outputs there, or run only c3/c4, which read only "
             "delivered tables.")
            % pispa_run)
    # The display table of step 3 ships inside the data archive. The two documented locations
    # are the table directory of the GO archive and the source-table directory of the release
    # archive; nothing outside them is probed, and the step that needs the table fails if it is
    # absent instead of continuing with an empty input.
    display_candidates = [data_root / "tables" / "hela_go_terms.tsv",
                          data_root.parent / "source_tables" / "hela_go_terms.tsv"]
    display_table = next((path for path in display_candidates if path.is_file()),
                         display_candidates[0])
    if require_display and not display_table.is_file():
        raise SystemExit(
            "FATAL: hela_go_terms.tsv is not in the archive. Checked %s. Step 3 reads the "
            "frozen display table of the manuscript; steps 1, 2 and 4 do not."
            % "; ".join(str(path) for path in display_candidates))

    # c1 and c2 record the paths they read relative to _PATHS["data_root"], so that value has
    # to be an ancestor of every reported path. When the GMT release is obtained outside the
    # archive (the documented way to use the resource), the reported base becomes the common
    # ancestor of the two; counts, hashes and thresholds are unaffected.
    report_root = data_root
    if gmt_dir != data_root and data_root not in gmt_dir.parents:
        report_root = Path(os.path.commonpath([str(data_root), str(gmt_dir)]))

    # Everything the chain writes goes to a working directory outside the archive: the archive
    # is an input in this release, so a run can never modify the data it reads.
    work_root = _work_root_candidates(args.out_dir)[0][1].resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    if work_root == data_root or data_root in work_root.parents:
        raise SystemExit(
            ("FATAL: --out-dir %s resolves inside the read-only data archive %s. The archive "
             "is an input: choose a working directory outside it (the default is <cwd>/%s).")
            % (work_root, data_root, DEFAULT_WORK_DIR))
    tables = work_root / "tables"
    for writable in (tables, work_root / "config", work_root / "tmp", work_root / "verification"):
        writable.mkdir(parents=True, exist_ok=True)
    return {
        "data_root": report_root,
        "archive_root": data_root,
        "work_root": work_root,
        "tables": tables,
        "config": work_root / "config",
        "tmp": work_root / "tmp",
        "verification": work_root / "verification",
        "input_tables": data_root / "tables",
        "display_candidates": [str(path) for path in display_candidates],
        "pispa_run": pispa_run,
        "gmt_dir": gmt_dir,
        "display_table": display_table,
    }
