"""Recompute the seven paired percentile-bootstrap intervals (contract section 4).

Contract-locked algorithm:
  * rule totals from the frozen V15 table, mode == with_run_dir (88 unique cells)
  * study order = dataset_id ascending by Unicode; system order fixed below
  * seven paired differences ScProteoAgent - comparator, float64, study is the unit
  * one PCG64(20260919) generator, exactly one rng.integers(0, 11, size=(20000, 11))
    call shared by all seven comparisons
  * boot = differences[idx, :].mean(axis=1); 2.5/97.5 percentiles, linear interpolation

Outputs (17 significant digits for raw endpoints), written to the output directory
(default <cwd>/intervals_out, never inside the data archive):
  paired_ci_recomputed.tsv
  paired_ci_old_new.tsv
  paired_ci_indices.npy
  environment.json

Release packaging note: the algorithm above is unchanged. Only input/output locations are
resolved at run time by resolve_data_dir()/configure() because the frozen tables ship in the
data archive instead of the historical directory tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]                      # repository root of this release layout
DEFAULT_DATA_DIR = "data/intervals"            # inside the repository ...
DEFAULT_DATA_DIR_SIBLING = "../data/intervals"  # ... or next to it, as in the submission package

# Frozen input filenames and the archive names they appear under. The algorithm below is
# unchanged; only the location of these files is resolved at run time.
RULES_NAMES = ("benchmark_rule_scores.tsv", "rule_scores.tsv")
OLD_MEANS_NAMES = ("benchmark_paired_difference_means.tsv", "paired_means.tsv")
HW_NAMES = ("HW_RULE_SCORES.tsv",)
PROVENANCE_NAMES = ("CI_PROVENANCE.md", "CI_PROVENANCE.tsv")

RULES = Path(DEFAULT_DATA_DIR) / "benchmark_rule_scores.tsv"
OLD_MEANS = Path(DEFAULT_DATA_DIR) / "benchmark_paired_difference_means.tsv"
HW = Path(DEFAULT_DATA_DIR) / "HW_RULE_SCORES.tsv"
CI_PROVENANCE = Path(DEFAULT_DATA_DIR) / "CI_PROVENANCE.md"
TABLES = Path(DEFAULT_DATA_DIR) / "outputs"


def resolve_data_dir(cli_value=None) -> Path:
    """Locate the frozen interval inputs: --data-dir > SCPROTEOMICS_INTERVALS_DIR > defaults."""
    candidates = []
    if cli_value:
        candidates.append(("--data-dir", Path(cli_value).expanduser()))
    env_value = os.getenv("SCPROTEOMICS_INTERVALS_DIR", "").strip()
    if env_value:
        candidates.append(("SCPROTEOMICS_INTERVALS_DIR", Path(env_value).expanduser()))
    candidates.append(("repository", REPO / DEFAULT_DATA_DIR))
    candidates.append(("sibling of the repository", (REPO / DEFAULT_DATA_DIR_SIBLING)))
    candidates.append(("data archive source tables", REPO / "../data/source_tables"))
    candidates.append(("repository source tables", REPO / "data/source_tables"))
    for source, candidate in candidates:
        resolved = candidate.resolve()
        if find_input(resolved, RULES_NAMES, search_parent=False) is not None:
            return resolved
    raise SystemExit(
        "FATAL: no frozen interval inputs found. Checked %s. Pass --data-dir or set "
        "SCPROTEOMICS_INTERVALS_DIR."
        % "; ".join("%s=%s" % (source, path) for source, path in candidates))


def find_input(data_dir: Path, names, search_parent: bool = True) -> Path | None:
    """Find one frozen input under a data directory, tolerating the archive's own naming."""
    roots = [data_dir, data_dir / "tables", data_dir / "inputs",
             data_dir / "source_tables", data_dir / "scores", data_dir / "intervals"]
    if search_parent and data_dir.parent != data_dir:
        roots += [data_dir.parent / "source_tables", data_dir.parent / "scores",
                  data_dir.parent / "intervals"]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def configure(data_dir: Path, out_dir: Path | None = None) -> None:
    """Point the frozen functions at a data archive and an output directory."""
    global RULES, OLD_MEANS, HW, CI_PROVENANCE, TABLES, REPO
    archive = data_dir.resolve()
    RULES = find_input(archive, RULES_NAMES)
    OLD_MEANS = find_input(archive, OLD_MEANS_NAMES)
    HW = find_input(archive, HW_NAMES) or archive / "HW_RULE_SCORES.tsv"
    CI_PROVENANCE = (find_input(archive, PROVENANCE_NAMES)
                     or archive / "CI_PROVENANCE.md")
    for found, label in ((RULES, "/".join(RULES_NAMES)), (OLD_MEANS, "/".join(OLD_MEANS_NAMES))):
        if found is None:
            raise SystemExit("FATAL: required frozen input missing under %s: %s" % (archive, label))
    REPO = archive
    for found in (RULES, OLD_MEANS, HW):
        if found is not None and found.is_file():
            while not str(found).startswith(str(REPO) + os.sep) and REPO.parent != REPO:
                REPO = REPO.parent
    TABLES = Path(out_dir).resolve() if out_dir else archive / "outputs"
    if TABLES == archive or archive in TABLES.parents:
        raise SystemExit(
            ("FATAL: --out-dir %s resolves inside the read-only data archive %s. The archive "
             "is an input: choose a working directory outside it (the default is "
             "<cwd>/intervals_out).") % (TABLES, archive))
    TABLES.mkdir(parents=True, exist_ok=True)


def input_record(path: Path):
    """Provenance record for a frozen input; None when the file is absent from the archive."""
    if not path.is_file():
        return None
    try:
        relative = path.relative_to(REPO).as_posix()
    except ValueError:
        relative = path.as_posix()
    return {"path": relative, "sha256": sha256_file(path)}

SEED = 20260919
NBOOT = 20000
METHOD_ORDER = ["scProteoAgent", "Hermes", "Codex", "Claude Code", "CellVoyager",
                "Biomni", "BioAgent", "SpatialAgent"]
ALIAS_HW = {"Proposed (gpt-5.5)": "scProteoAgent", "Claudecode": "Claude Code",
            "Bioagent": "BioAgent"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rules() -> tuple[list, list, dict]:
    rows = list(csv.DictReader(RULES.open(encoding="utf-8-sig"), delimiter="\t"))
    rows = [r for r in rows if r["mode"] == "with_run_dir"]
    datasets = sorted({r["dataset_id"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    if len(rows) != 88 or len(datasets) != 11 or len(methods) != 8:
        raise SystemExit("FATAL: frozen rule table is not 11 x 8 = 88 rows")
    unknown = [m for m in methods if m not in METHOD_ORDER]
    if unknown:
        raise SystemExit("FATAL: unknown method keys: %s" % unknown)
    value = {}
    for r in rows:
        key = (r["dataset_id"], r["method"])
        if key in value:
            raise SystemExit("FATAL: duplicated cell %s" % (key,))
        v = float(r["rule_total"])
        if not np.isfinite(v):
            raise SystemExit("FATAL: non-finite rule total for %s" % (key,))
        value[key] = v
    if len(value) != 88:
        raise SystemExit("FATAL: expected 88 unique cells, got %d" % len(value))
    return datasets, methods, value


def cross_check_hw(value: dict) -> list:
    rows = list(csv.DictReader(HW.open(encoding="utf-8-sig"), delimiter="\t"))
    rows = [r for r in rows if r["mode"] == "with_run_dir"]
    if len(rows) != 88:
        raise SystemExit("FATAL: HW_RULE_SCORES with_run_dir is not 88 rows")
    dataset_ids = {k[0] for k in value}
    report = []
    for r in rows:
        name = ALIAS_HW.get(r["method"], r["method"])
        if name not in METHOD_ORDER:
            raise SystemExit("FATAL: unmapped HW method %r" % r["method"])
        if r["dataset"] not in dataset_ids:
            raise SystemExit("FATAL: HW dataset %r not in the frozen table" % r["dataset"])
        got = round(float(r["rule_total"]), 6)
        want = round(value[(r["dataset"], name)], 6)
        report.append({"dataset": r["dataset"], "hw_method": r["method"],
                       "display_method": name, "rule_total": got,
                       "match_v15": abs(got - want) < 1e-9})
    bad = [x for x in report if not x["match_v15"]]
    if bad:
        raise SystemExit("FATAL: HW/V15 mismatch on %d cells: %s" % (len(bad), bad[:3]))
    return report


def old_endpoints() -> dict:
    out = {}
    rows = list(csv.DictReader(OLD_MEANS.open(encoding="utf-8-sig"), delimiter="\t"))
    for r in rows:
        name = r["comparator"].strip()
        out[name] = (float(r["ci_low"]), float(r["ci_high"]), float(r["mean_difference"]))
    return out


def tsv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the seven paired percentile-bootstrap intervals from frozen inputs.")
    parser.add_argument("--data-dir", default=None,
                        help="directory holding the frozen interval inputs "
                             "(default: resolve_data_dir())")
    parser.add_argument("--out-dir", default=None,
                        help="directory for the output tables "
                             "(default: <cwd>/intervals_out, outside the data archive)")
    args = parser.parse_args(argv)
    configure(resolve_data_dir(args.data_dir), args.out_dir)
    datasets, methods, value = load_rules()
    if HW.is_file():
        hw_report = cross_check_hw(value)
    else:
        hw_report = []
        print("warning: %s is absent from the data archive; the HW cross-check was not run" % HW)
    comparators = METHOD_ORDER[1:]
    diff = np.array([[value[(d, "scProteoAgent")] - value[(d, m)] for m in comparators]
                     for d in datasets], dtype=np.float64)
    means = diff.mean(axis=0)

    rng = np.random.Generator(np.random.PCG64(SEED))
    idx = rng.integers(0, 11, size=(NBOOT, 11), dtype=np.int64)
    boot = diff[idx, :].mean(axis=1)
    q = np.quantile(boot, [0.025, 0.975], axis=0, method="linear")
    low, high = q[0], q[1]

    idx_bytes = idx.tobytes()
    idx_sha = hashlib.sha256(idx_bytes).hexdigest()
    np.save(TABLES / "paired_ci_indices.npy", idx)

    rows = []
    for i, m in enumerate(comparators):
        rows.append([i + 1, m, m, format(means[i], ".17g"), format(low[i], ".17g"),
                     format(high[i], ".17g"), len(datasets)])
    tsv(TABLES / "paired_ci_recomputed.tsv",
        ["system_order", "comparator", "comparator_source_key", "mean_difference",
         "ci_low", "ci_high", "n_studies"], rows)

    old = old_endpoints()
    old_rows = []
    for i, m in enumerate(comparators):
        key = {"Claude Code": "Claude Code", "BioAgent": "BioAgent"}.get(m, m)
        if key not in old:
            raise SystemExit("FATAL: no registered old endpoint for %r" % m)
        ol, oh, om = old[key]
        old_rows.append([i + 1, m, "%.2f" % om, "%.4f" % ol, "%.4f" % oh,
                         "%.2f" % low[i], "%.2f" % high[i],
                         format(low[i], ".17g"), format(high[i], ".17g"),
                         "%.4f" % (low[i] - ol), "%.4f" % (high[i] - oh),
                         "V15 benchmark_paired_difference_means.tsv (registered); "
                         "cross-checked against V17 verification/CI_PROVENANCE.md"])
    tsv(TABLES / "paired_ci_old_new.tsv",
        ["system_order", "comparator", "mean_difference_display", "ci_low_old",
         "ci_high_old", "ci_low_new_display", "ci_high_new_display", "ci_low_new_raw",
         "ci_high_new_raw", "delta_low", "delta_high", "old_endpoint_source"], old_rows)

    env = {
        "algorithm": "percentile bootstrap of the paired study differences, 2.5/97.5 percentiles, "
                     "linear interpolation",
        "seed": SEED,
        "generator": "np.random.Generator(np.random.PCG64(20260919)), single rng.integers call",
        "n_resamples": NBOOT,
        "n_studies": len(datasets),
        "n_comparisons": len(comparators),
        "quantile_method": "linear",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "study_order": datasets,
        "system_order": METHOD_ORDER,
        "method_alias_map": {"benchmark_rule_scores.tsv": "display names already applied",
                             "HW_RULE_SCORES.tsv": ALIAS_HW},
        "index_dtype": str(idx.dtype),
        "index_shape": list(idx.shape),
        "index_byteorder": "little" if idx.dtype.byteorder in ("<", "=", "|") else "big",
        "index_c_contiguous": bool(idx.flags["C_CONTIGUOUS"]),
        "index_sha256": idx_sha,
        "index_file": "tables/paired_ci_indices.npy",
        "inputs": {key: record for key, record in {
            "benchmark_rule_scores.tsv": input_record(RULES),
            "benchmark_paired_difference_means.tsv": input_record(OLD_MEANS),
            "HW_RULE_SCORES.tsv": input_record(HW),
            "CI_PROVENANCE.md": input_record(CI_PROVENANCE),
        }.items() if record is not None},
        "hw_cross_check_rows": len(hw_report),
        "hw_cross_check_all_match": (all(x["match_v15"] for x in hw_report)
                                     if hw_report else None),
        "endpoint_reference_values": {
            m: {"mean_difference": format(means[i], ".17g"),
                "ci_low": format(low[i], ".17g"),
                "ci_high": format(high[i], ".17g")}
            for i, m in enumerate(comparators)},
    }
    (TABLES / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("datasets", len(datasets), "comparators", len(comparators))
    print("means", [round(float(x), 6) for x in means])
    print("idx sha256", idx_sha)
    for i, m in enumerate(comparators):
        print("%-12s %.4f - %.4f (raw %.17g / %.17g)" % (m, low[i], high[i], low[i], high[i]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
